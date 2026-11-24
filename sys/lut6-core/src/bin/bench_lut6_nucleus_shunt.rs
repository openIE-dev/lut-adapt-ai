//! Phase 30 (`Claim LA-26` & `Claim LA-28`): Cross-Silicon & Physical DC Shunt (`measured_j`)
//! Benchmark for `lut6-core` (`CompactLut6Page`, `Nucleus`, Vertical Bit-Sliced `eval_batch64`,
//! and Yosys `.lut6` Binary Crystal Verification vs. `INT8` Micro-MLP Fallback).
//!
//! Compiles with `cargo build --release --features std --bin bench_lut6_nucleus_shunt`
//! (`zero` external crates) across:
//!   - `NVIDIA Jetson AGX Thor` (`CUDA_VISIBLE_DEVICES=""`, physical `TI INA3221` `VDD_CPU_SOC_MSS`
//!     and `INA238` `VIN` DC shunts -> `measured_j`, `board_metered = true`),
//!   - `AMD Ryzen AI Max+ 395` (`fw128b`, Linux `amd_rapl` `package-0` -> `reported_j`),
//!   - `Apple M5 Max` (`Darwin RUSAGE_INFO_V6.ri_energy_nj` -> `reported_j`).

#![cfg(feature = "std")]

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use lut6_core::bootstrap::SEED_AGENCY;
use lut6_core::ratchet_demo::{SEED_KNOWLEDGE, synthesize_wobble_patch};
use lut6_core::{AgencyAction, CompactLut6Page, Nucleus, PageLedger};

// ============================================================================
// Platform Telemetry (`measured_j` strictly separated from `reported_j`)
// ============================================================================

#[cfg(target_os = "macos")]
mod macos_energy {
    #[repr(C)]
    struct RUsageInfoV6 {
        words: [u64; 64],
    }
    unsafe extern "C" {
        fn getpid() -> i32;
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RUsageInfoV6) -> i32;
    }
    pub fn read_nj_and_cycles() -> (u64, u64) {
        let mut ru = RUsageInfoV6 { words: [0; 64] };
        let rc = unsafe { proc_pid_rusage(getpid(), 6, &mut ru as *mut _) };
        if rc == 0 {
            (ru.words[42], ru.words[31])
        } else {
            (0, 0)
        }
    }
}

#[derive(Clone, Debug)]
struct TegraShunts {
    vdd_cpu_mv: PathBuf,
    vdd_cpu_ma: PathBuf,
    vin_uw: Option<PathBuf>,
    vin_mv: Option<PathBuf>,
    vin_ma: Option<PathBuf>,
}

fn read_u64(p: &Path) -> Option<u64> {
    fs::read_to_string(p).ok()?.trim().parse::<u64>().ok()
}

fn discover_tegra_shunts() -> Option<TegraShunts> {
    let root = Path::new("/sys/class/hwmon");
    if !root.is_dir() {
        return None;
    }
    let mut d3221: Option<PathBuf> = None;
    let mut d238: Option<PathBuf> = None;
    for ent in fs::read_dir(root).ok()?.flatten() {
        let p = ent.path();
        if let Ok(nm) = fs::read_to_string(p.join("name")) {
            match nm.trim() {
                "ina3221" => d3221 = Some(p),
                "ina238" => d238 = Some(p),
                _ => {}
            }
        }
    }
    let d3221 = d3221?;
    let mut cpu_ch: Option<usize> = None;
    for ch in 1..=3 {
        if let Ok(lbl) = fs::read_to_string(d3221.join(format!("in{ch}_label"))) {
            if lbl.contains("CPU_SOC") {
                cpu_ch = Some(ch);
            }
        }
    }
    let c = cpu_ch?;
    let vdd_cpu_mv = d3221.join(format!("in{c}_input"));
    let vdd_cpu_ma = d3221.join(format!("curr{c}_input"));
    if !vdd_cpu_mv.is_file() || !vdd_cpu_ma.is_file() {
        return None;
    }
    let (vin_uw, vin_mv, vin_ma) = if let Some(d) = d238 {
        let p_uw = d.join("power1_input");
        let p_mv = d.join("in1_input");
        let p_ma = d.join("curr1_input");
        (
            p_uw.is_file().then_some(p_uw),
            p_mv.is_file().then_some(p_mv),
            p_ma.is_file().then_some(p_ma),
        )
    } else {
        (None, None, None)
    };
    Some(TegraShunts {
        vdd_cpu_mv,
        vdd_cpu_ma,
        vin_uw,
        vin_mv,
        vin_ma,
    })
}

fn discover_rapl_pkg() -> Option<(PathBuf, u64)> {
    let p = Path::new("/sys/class/powercap/intel-rapl:0/energy_uj");
    if p.is_file() {
        let max_r = read_u64(Path::new(
            "/sys/class/powercap/intel-rapl:0/max_energy_range_uj",
        ))
        .unwrap_or(u64::MAX);
        return Some((p.to_path_buf(), max_r));
    }
    None
}

#[derive(Clone, Debug, Default)]
struct WindowTelemetry {
    window_name: &'static str,
    iterations: u64,
    elapsed_ms: f64,
    ns_per_op: f64,
    cpu_cycles_per_op: Option<f64>,
    measured_j_vdd_cpu: Option<f64>,
    measured_j_vin: Option<f64>,
    measured_nj_per_op_vdd_cpu: Option<f64>,
    reported_j: Option<f64>,
    reported_nj_per_op: Option<f64>,
    board_metered: bool,
}

fn measure_window<F: FnMut()>(
    name: &'static str,
    iterations: u64,
    shunts: &Option<TegraShunts>,
    rapl: &Option<(PathBuf, u64)>,
    mut body: F,
) -> WindowTelemetry {
    // Warmup
    body();

    let rapl_before = rapl.as_ref().and_then(|(p, _)| read_u64(p));
    #[cfg(target_os = "macos")]
    let (mac_nj_0, mac_cyc_0) = macos_energy::read_nj_and_cycles();

    let stop = Arc::new(AtomicBool::new(false));
    let shunt_handle = shunts.clone().map(|sh| {
        let stop_c = Arc::clone(&stop);
        thread::spawn(move || {
            let mut prev = Instant::now();
            let mut cpu_j = 0.0f64;
            let mut vin_j = 0.0f64;
            let mut samples = 0u64;
            while !stop_c.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(2));
                let now = Instant::now();
                let dt = now.duration_since(prev).as_secs_f64();
                prev = now;
                if let (Some(mv), Some(ma)) = (read_u64(&sh.vdd_cpu_mv), read_u64(&sh.vdd_cpu_ma))
                {
                    let watts = (mv as f64 * ma as f64) * 1e-6;
                    cpu_j += watts * dt;
                    samples += 1;
                }
                if let Some(ref puw) = sh.vin_uw
                    && let Some(uw) = read_u64(puw)
                {
                    vin_j += (uw as f64 * 1e-6) * dt;
                } else if let (Some(pmv), Some(pma)) = (&sh.vin_mv, &sh.vin_ma)
                    && let (Some(mv), Some(ma)) = (read_u64(pmv), read_u64(pma))
                {
                    vin_j += (mv as f64 * ma as f64 * 1e-6) * dt;
                }
            }
            (cpu_j, vin_j, samples)
        })
    });

    let t0 = Instant::now();
    body();
    let elapsed = t0.elapsed();
    stop.store(true, Ordering::Relaxed);

    let (measured_cpu_j, measured_vin_j, board_metered) = if let Some(h) = shunt_handle {
        let (cj, vj, cnt) = h.join().unwrap_or((0.0, 0.0, 0));
        if cnt > 0 {
            (Some(cj), (vj > 0.0).then_some(vj), true)
        } else {
            (None, None, false)
        }
    } else {
        (None, None, false)
    };

    let rapl_after = rapl.as_ref().and_then(|(p, _)| read_u64(p));
    #[allow(unused_mut)]
    let mut reported_j: Option<f64> = None;
    #[allow(unused_mut)]
    let mut cyc_per_op: Option<f64> = None;

    if let (Some(b), Some(a), Some((_, max_r))) = (rapl_before, rapl_after, rapl.as_ref()) {
        let diff_uj = if a >= b { a - b } else { (*max_r - b) + a };
        if diff_uj > 0 {
            reported_j = Some(diff_uj as f64 * 1e-6);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let (mac_nj_1, mac_cyc_1) = macos_energy::read_nj_and_cycles();
        if mac_nj_1 > mac_nj_0 {
            reported_j = Some((mac_nj_1 - mac_nj_0) as f64 * 1e-9);
        }
        if mac_cyc_1 > mac_cyc_0 {
            cyc_per_op = Some((mac_cyc_1 - mac_cyc_0) as f64 / iterations as f64);
        }
    }

    let elapsed_ms = elapsed.as_secs_f64() * 1e3;
    let ns_per_op = (elapsed.as_secs_f64() * 1e9) / (iterations as f64);

    WindowTelemetry {
        window_name: name,
        iterations,
        elapsed_ms,
        ns_per_op,
        cpu_cycles_per_op: cyc_per_op,
        measured_j_vdd_cpu: measured_cpu_j,
        measured_j_vin: measured_vin_j,
        measured_nj_per_op_vdd_cpu: measured_cpu_j.map(|j| j * 1e9 / iterations as f64),
        reported_j,
        reported_nj_per_op: reported_j.map(|j| j * 1e9 / iterations as f64),
        board_metered,
    }
}

/// Baseline `INT8` `64 -> 32 -> 16` Micro-MLP fallback (representing standard MCU INT8 micro-model).
struct Int8MicroMlp {
    w1: [[i8; 64]; 32],
    b1: [i32; 32],
    w2: [[i8; 32]; 16],
    b2: [i32; 16],
}

impl Int8MicroMlp {
    fn deterministic_init() -> Self {
        let mut w1 = [[0i8; 64]; 32];
        let mut b1 = [0i32; 32];
        let mut w2 = [[0i8; 32]; 16];
        let mut b2 = [0i32; 16];
        for r in 0..32 {
            b1[r] = (r as i32 * 7) - 50;
            for c in 0..64 {
                w1[r][c] = (((r * 131 + c * 17 + 5) & 0xFF) as i8).wrapping_add(1);
            }
        }
        for r in 0..16 {
            b2[r] = (r as i32 * 5) - 20;
            for c in 0..32 {
                w2[r][c] = (((r * 97 + c * 31 + 11) & 0xFF) as i8).wrapping_add(1);
            }
        }
        Self { w1, b1, w2, b2 }
    }

    #[inline(never)]
    fn eval_int8(&self, state: u64) -> u16 {
        let mut x = [0i8; 64];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = if ((state >> i) & 1) != 0 { 64 } else { -64 };
        }
        let mut h = [0i8; 32];
        for r in 0..32 {
            let mut acc = self.b1[r];
            for c in 0..64 {
                acc += (self.w1[r][c] as i32) * (x[c] as i32);
            }
            h[r] = (acc >> 7).clamp(0, 127) as i8;
        }
        let mut out_bits = 0u16;
        for r in 0..16 {
            let mut acc = self.b2[r];
            for c in 0..32 {
                acc += (self.w2[r][c] as i32) * (h[c] as i32);
            }
            if acc > 0 {
                out_bits |= 1u16 << r;
            }
        }
        out_bits
    }
}

fn opt_num(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.6}"),
        None => "null".to_string(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut host_tag = String::from("local");
    let mut crystal_path: Option<PathBuf> = None;
    let mut out_json: Option<PathBuf> = None;
    let mut scale_iters: u64 = 10_000_000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host-tag" if i + 1 < args.len() => {
                host_tag = args[i + 1].clone();
                i += 2;
            }
            "--verify-crystal" if i + 1 < args.len() => {
                crystal_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out-json" if i + 1 < args.len() => {
                out_json = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--iters" if i + 1 < args.len() => {
                scale_iters = args[i + 1].parse::<u64>().unwrap_or(10_000_000);
                i += 2;
            }
            _ => i += 1,
        }
    }

    // Optional cross-language binary crystal check (`pack_lut6_crystal.py` -> Rust `CompactLut6Page`)
    let mut yosys_crystal_verified = false;
    if let Some(ref cp) = crystal_path {
        let bytes = fs::read(cp).expect("Failed to read .lut6 crystal file");
        assert_eq!(bytes.len(), 4096, ".lut6 file must be exact 4096 bytes");
        let mut aligned = SEED_KNOWLEDGE;
        // SAFETY: `CompactLut6Page` is `#[repr(C, align(8))]` of exact size 4,096 bytes of plain integers
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                &mut aligned as *mut CompactLut6Page as *mut u8,
                4096,
            );
        }
        assert!(aligned.is_valid(), "Loaded Yosys .lut6 crystal failed is_valid()");
        assert!(
            aligned.verify_crc32(),
            "Loaded Yosys .lut6 crystal failed IEEE CRC32 check!"
        );
        // Check Yosys Operon outputs on Reject (`bit 4 = 1 -> 0x3C`) and Tool Ready (`bits 0,1 = 1,1 -> 0x20`)
        assert_eq!(aligned.eval_raw(0b01_0000) & 0x3F, 0x3C);
        assert_eq!(aligned.eval_raw(0b00_0011) & 0x3F, 0x20);
        assert_eq!(aligned.eval_raw(0b00_0000) & 0x3F, 0x00);
        yosys_crystal_verified = true;
    }

    let shunts = discover_tegra_shunts();
    let rapl = discover_rapl_pkg();

    let mut nucleus = Nucleus {
        knowledge_root: &SEED_KNOWLEDGE,
        agency_root: &SEED_AGENCY,
        knowledge_overlay: None,
        agency_overlay: None,
        k_ledger: PageLedger::default(),
        a_ledger: PageLedger::default(),
        k_anchors: [(0x08, 0xFF, 0xA1), (0x0B, 0xFF, 0xA1)],
        a_anchors: [
            ((1u64 << 40) | (1u64 << 48), 0x3F, 0x20),
            (1u64 << 63, 0x3F, 0x3C),
        ],
    };

    // Window 1: Certified Innate Rung-6 Hit (`0x08 -> 0xA1`)
    let w1 = measure_window("Window1_Innate_Rung6_Hit", scale_iters, &shunts, &rapl, || {
        let mut acc = 0u32;
        for step in 0..scale_iters {
            let inp = 0x08u64 + (step & 0x3);
            if let AgencyAction::EmitCertified { payload } =
                nucleus.step(black_box(inp), black_box(1))
            {
                acc ^= payload;
            }
        }
        black_box(acc);
    });

    // Window 2: Frontier Miss -> 64-Codon Agency Escalation (`0x34 -> WakeRefiner`)
    let miss_iters = scale_iters / 2;
    let w2 = measure_window(
        "Window2_Frontier_Miss_Agency_Escalate",
        miss_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for step in 0..miss_iters {
                let inp = 0x34u64 + (step & 0x3);
                if let AgencyAction::WakeRefiner { frontier_bits } =
                    nucleus.step(black_box(inp), black_box(0))
                {
                    acc ^= frontier_bits;
                }
            }
            black_box(acc);
        },
    );

    // Window 3: On-Device Walsh FWHT Harmonic Filter + Anchor Non-Regression Refinement
    let refine_iters = scale_iters / 50;
    let w3 = measure_window(
        "Window3_Walsh_FWHT_Anchor_Refine",
        refine_iters,
        &shunts,
        &rapl,
        || {
            let mut ok_cnt = 0u64;
            for step in 0..refine_iters {
                let patch =
                    synthesize_wobble_patch(black_box(0x34), black_box(0x5Eu8 ^ ((step & 1) as u8)));
                if nucleus.refine_knowledge(patch, 16) {
                    ok_cnt += 1;
                }
            }
            black_box(ok_cnt);
        },
    );
    // Lock clean `0x5E` wobble patch for Window 4
    assert!(nucleus.refine_knowledge(synthesize_wobble_patch(0x34, 0x5E), 16));

    // Window 4: Post-Compilation Noisy Wobble Hit (`0x34..0x37 -> 0x5E`, 0 escalations!)
    let w4 = measure_window(
        "Window4_PostCompile_Noisy_Wobble_Hit",
        scale_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u32;
            for step in 0..scale_iters {
                let jittered = 0x34u64 | (step & 0x3);
                if let AgencyAction::EmitCertified { payload } =
                    nucleus.step(black_box(jittered), black_box(0))
                {
                    acc ^= payload;
                }
            }
            black_box(acc);
        },
    );

    // Window 5: Vertical Bit-Sliced `eval_batch64` (64 parallel canonical inputs per call)
    let batch_calls = scale_iters / 16;
    let total_batch_decisions = batch_calls * 64;
    let mut batch_inputs = [0u64; 64];
    for (idx, slot) in batch_inputs.iter_mut().enumerate() {
        *slot = 0x08u64 + ((idx as u64) & 0x3);
    }
    let w5 = measure_window(
        "Window5_Vertical_BitSliced_Batch64",
        total_batch_decisions,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for _ in 0..batch_calls {
                let outs = SEED_KNOWLEDGE.eval_batch64(black_box(&batch_inputs));
                acc ^= outs[0] ^ outs[63];
            }
            black_box(acc);
        },
    );

    // Window 6: Baseline `INT8` `64 -> 32 -> 16` Micro-MLP Fallback (same decision count as `scale_iters / 2`)
    let mlp = Int8MicroMlp::deterministic_init();
    let mlp_iters = scale_iters / 2;
    let w6 = measure_window(
        "Window6_Baseline_INT8_MicroMLP",
        mlp_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u16;
            for step in 0..mlp_iters {
                acc ^= mlp.eval_int8(black_box(0x34u64 + (step & 0xF)));
            }
            black_box(acc);
        },
    );

    let windows = [w1, w2, w3, w4, w5, w6];
    let mut json_rows = Vec::new();
    for w in &windows {
        println!(
            "[{host_tag}] {:<38} | {:>8.3} ns/op | measured_nj/op={} | reported_nj/op={} | board_metered={}",
            w.window_name,
            w.ns_per_op,
            opt_num(w.measured_nj_per_op_vdd_cpu),
            opt_num(w.reported_nj_per_op),
            w.board_metered
        );
        json_rows.push(format!(
            "    {{\"window\": \"{}\", \"iterations\": {}, \"elapsed_ms\": {:.4}, \"ns_per_op\": {:.6}, \"cpu_cycles_per_op\": {}, \"measured_j_vdd_cpu\": {}, \"measured_j_vin\": {}, \"measured_nj_per_op_vdd_cpu\": {}, \"reported_j\": {}, \"reported_nj_per_op\": {}, \"board_metered\": {}}}",
            w.window_name,
            w.iterations,
            w.elapsed_ms,
            w.ns_per_op,
            opt_num(w.cpu_cycles_per_op),
            opt_num(w.measured_j_vdd_cpu),
            opt_num(w.measured_j_vin),
            opt_num(w.measured_nj_per_op_vdd_cpu),
            opt_num(w.reported_j),
            opt_num(w.reported_nj_per_op),
            w.board_metered
        ));
    }

    let doc = format!(
        "{{\n  \"host_tag\": \"{}\",\n  \"arch\": \"{}\",\n  \"os\": \"{}\",\n  \"yosys_crystal_verified\": {},\n  \"board_metered\": {},\n  \"windows\": [\n{}\n  ]\n}}\n",
        host_tag,
        env::consts::ARCH,
        env::consts::OS,
        yosys_crystal_verified,
        shunts.is_some(),
        json_rows.join(",\n")
    );

    if let Some(p) = out_json {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&p, doc).expect("Failed to write output JSON");
    } else {
        print!("{doc}");
    }
}
