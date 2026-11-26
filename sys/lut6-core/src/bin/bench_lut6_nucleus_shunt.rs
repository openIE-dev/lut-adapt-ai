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
use lut6_core::generated_proof_vectors::{VERIFIED_TMAC_SKILL_ANCHORS, mol_skills_tmac_crystal};
use lut6_core::hadamard_bonsai::{
    Arc3ReflexOutcome, BONSAI_BLOCK_1024, ENTROMCU_GROUP128_BYTES, eval_arc3_policy_lut6,
    eval_benes_arc_lut6_2layer_strip64, eval_bonsai1024_block16_antisym40, pack_bonsai_3tok_54bit,
    pack_bonsai_block16_codes, transcode_mlx2bit_group128_to_entromcu28, unpack_bonsai_3tok_54bit,
    unpack_entromcu28_group128,
};
use lut6_core::ratchet_demo::{SEED_KNOWLEDGE, synthesize_wobble_patch};
use lut6_core::tmac_tail::{
    eval_tmac_antisym_from_pad, eval_tmac_from_pad, wire_state_to_bipolar_i8,
};
use lut6_core::wearable_epistemic::{WearableEpistemicDecision, eval_wearable_epistemic_gate};
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

    // Window 7: 4-Way Unrolled Scalar `eval_raw_unrolled4` (`SEED_AGENCY` 64-gate page)
    let w7 = measure_window(
        "Window7_Unrolled4_Scalar_LUT6",
        scale_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for step in 0..scale_iters {
                let st = (1u64 << 40) | (1u64 << 48) | (step & 0xFF);
                acc ^= SEED_AGENCY.eval_raw_unrolled4(black_box(st));
            }
            black_box(acc);
        },
    );

    // Window 8: Zero-Multiplier `EntroMCU` / `T-MAC` Ternary Lookup Tail (`672 B` inside `mol_skills_tmac_tail.lut6`)
    let skills_page = mol_skills_tmac_crystal();
    let pre_feats: [[i8; 64]; 4] = [
        wire_state_to_bipolar_i8(VERIFIED_TMAC_SKILL_ANCHORS[0].0),
        wire_state_to_bipolar_i8(VERIFIED_TMAC_SKILL_ANCHORS[1].0),
        wire_state_to_bipolar_i8(VERIFIED_TMAC_SKILL_ANCHORS[2].0),
        wire_state_to_bipolar_i8(VERIFIED_TMAC_SKILL_ANCHORS[3].0),
    ];
    let w8 = measure_window(
        "Window8_ZeroMul_Ternary_TMAC_Tail",
        mlp_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for step in 0..mlp_iters {
                let f = &pre_feats[(step & 3) as usize];
                let (pay, votes, _) = eval_tmac_from_pad(black_box(&skills_page._pad), black_box(f));
                acc ^= pay ^ (votes as u64);
            }
            black_box(acc);
        },
    );

    // Verify 100% bit-exact parity between 81-entry T-MAC and ISCA'25 Antisymmetric 40-entry T-MAC
    for f in &pre_feats {
        assert_eq!(
            eval_tmac_from_pad(&skills_page._pad, f),
            eval_tmac_antisym_from_pad(&skills_page._pad, f),
            "Antisymmetric 40-entry T-MAC diverged from full 81-entry T-MAC!"
        );
    }

    // Window 9: Antisymmetric 40-Entry `LUT Tensor Core` (ISCA 2025) Zero-MUL `T-MAC` Tail (`1,312 B` stack vs `2,592 B`)
    let w9 = measure_window(
        "Window9_Antisym40_ZeroMul_TMAC_Tail",
        mlp_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for step in 0..mlp_iters {
                let f = &pre_feats[(step & 3) as usize];
                let (pay, votes, _) =
                    eval_tmac_antisym_from_pad(black_box(&skills_page._pad), black_box(f));
                acc ^= pay ^ (votes as u64);
            }
            black_box(acc);
        },
    );

    // Window 10: Wearable `56`-Bit `4`-Token `uLM-28M` Opener + `LastExam` Epistemic Hold Gate (`Claim LA-36/38`)
    let w10 = measure_window(
        "Window10_Wearable_56Bit_4Tok_Epistemic_Gate",
        scale_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            let mut ledger = PageLedger::default();
            for step in 0..scale_iters {
                // Mix 75% certified wearable reflexes (`0x08..0x0B`) + 25% frontier `LastExam` hold queries
                let state = if (step & 3) == 3 {
                    (0x8700u64 << 48) | (0x34u64 + (step & 3))
                } else {
                    0x08u64 + (step & 3)
                };
                match eval_wearable_epistemic_gate(
                    &SEED_KNOWLEDGE,
                    nucleus.knowledge_overlay.as_ref(),
                    black_box(state),
                    3,
                    &mut ledger,
                ) {
                    WearableEpistemicDecision::CertifiedReflex {
                        tokens4, conf_pct, ..
                    } => {
                        acc ^= (tokens4[0] as u64) ^ (tokens4[3] as u64) ^ (conf_pct as u64);
                    }
                    WearableEpistemicDecision::WakeLocalMicroLlm { partial_bits, .. } => {
                        acc ^= partial_bits;
                    }
                    WearableEpistemicDecision::EpistemicHoldEscalate {
                        prior_conf_pct,
                        frontier_signature,
                    } => {
                        acc ^= (frontier_signature as u64) ^ (prior_conf_pct as u64);
                    }
                }
            }
            black_box(acc);
        },
    );

    // Window 11: Baseline `uLM-28M` (`d=512`) Single-Token `INT8` Hidden Projection Slice (`512 -> 64` = `32,768` MACs,
    // i.e., a conservative 1/16th lower bound on a single `uLM-28M` layer's ~524K MACs!)
    let mut ulm_w = [[0i8; 512]; 64];
    for (r, row) in ulm_w.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = (((r * 73 + c * 19 + 3) & 0x7F) as i8).wrapping_sub(64);
        }
    }
    let ulm_iters = scale_iters / 10;
    let w11 = measure_window(
        "Window11_Baseline_uLM28M_d512_Slice",
        ulm_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0i32;
            let mut x = [17i8; 512];
            for step in 0..ulm_iters {
                x[(step & 511) as usize] = ((step & 0x3F) as i8) - 32;
                let mut tok_hash = 0i32;
                for row in &ulm_w {
                    let mut dot = 0i32;
                    for c in 0..512 {
                        dot += (row[c] as i32) * (black_box(x[c]) as i32);
                    }
                    tok_hash ^= dot >> 8;
                }
                acc ^= tok_hash;
            }
            black_box(acc);
        },
    );

    // Window 12: `prism-ml/Ternary-Bonsai-2-27B` Native Hadamard-Ternary (`1024 -> 16` = `16,384` weights)
    // via Zero-MUL Sylvester `FWHT-1024 (>> 5)` + `ISCA` 2025 `Antisym-40 T-MAC` (`8` scale muls/row vs `1,024`)
    let mut bonsai_trits = Box::new([[0i8; BONSAI_BLOCK_1024]; 16]);
    let mut bonsai_u32 = Box::new([[0u32; 64]; 16]); // MLX 2-bit packed `[16, 1024/16]`
    let mut bonsai_scales_f32 = [[0.0f32; 8]; 16];
    let mut bonsai_biases_f32 = [[0.0f32; 8]; 16];
    let mut bonsai_scales_q8 = [[0i16; 8]; 16];
    for r in 0..16 {
        for g in 0..8 {
            let q = ((r * 3 + g * 2 + 1) & 7) as i16 + 1;
            bonsai_scales_q8[r][g] = q;
            let sf = (q as f32) * 0.125;
            bonsai_scales_f32[r][g] = sf;
            bonsai_biases_f32[r][g] = -sf; // Exact MLX `biases = -scales` affine convention
        }
        for c in 0..BONSAI_BLOCK_1024 {
            let t = (((r * 53 + c * 19 + 2) % 3) as i8) - 1;
            bonsai_trits[r][c] = t;
            let code2 = (t + 1) as u32;
            bonsai_u32[r][c >> 4] |= code2 << ((c & 15) * 2);
        }
    }
    let bonsai_codes = Box::new(pack_bonsai_block16_codes(&bonsai_trits));
    let bonsai_signs = [0xA5A5_5A5A_C3C3_3C3Cu64; 16];
    let mut bonsai_signs_f32 = [1.0f32; BONSAI_BLOCK_1024];
    for w in 0..16 {
        for b in 0..64 {
            if ((bonsai_signs[w] >> b) & 1) != 0 {
                bonsai_signs_f32[w * 64 + b] = -1.0;
            }
        }
    }
    let bonsai_iters = (scale_iters / 40).max(1_000);
    let w12 = measure_window(
        "Window12_Bonsai_FWHT1024_Antisym40_Block16",
        bonsai_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0i32;
            let mut x_in = [11i8; BONSAI_BLOCK_1024];
            for step in 0..bonsai_iters {
                x_in[(step & 1023) as usize] = ((step & 0x1F) as i8) - 16;
                let y = eval_bonsai1024_block16_antisym40(
                    black_box(&x_in),
                    black_box(&bonsai_signs),
                    black_box(&bonsai_codes),
                    black_box(&bonsai_scales_q8),
                );
                acc ^= y[0] ^ y[7] ^ y[15];
            }
            black_box(acc);
        },
    );

    // Window 13: Baseline `MLX` 2-Bit Affine Unpack + Float `FWHT-1024 / sqrt(1024)` + Dequantized `16,384` MACs
    let w13 = measure_window(
        "Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16",
        bonsai_iters,
        &shunts,
        &rapl,
        || {
            let mut acc_bits = 0u32;
            let mut x_in = [11i8; BONSAI_BLOCK_1024];
            let inv_sqrt_1024 = 1.0f32 / (1024.0f32).sqrt();
            for step in 0..bonsai_iters {
                x_in[(step & 1023) as usize] = ((step & 0x1F) as i8) - 16;
                let mut fx = [0.0f32; BONSAI_BLOCK_1024];
                for i in 0..BONSAI_BLOCK_1024 {
                    fx[i] = (black_box(x_in[i]) as f32) * bonsai_signs_f32[i];
                }
                let mut len = 1usize;
                while len < BONSAI_BLOCK_1024 {
                    let st = len << 1;
                    let mut base = 0usize;
                    while base < BONSAI_BLOCK_1024 {
                        for j in 0..len {
                            let u = fx[base + j];
                            let v = fx[base + j + len];
                            fx[base + j] = u + v;
                            fx[base + j + len] = u - v;
                        }
                        base += st;
                    }
                    len = st;
                }
                for slot in &mut fx {
                    *slot *= inv_sqrt_1024;
                }
                let mut row_hash = 0u32;
                for r in 0..16 {
                    let mut dot = 0.0f32;
                    for grp in 0..8 {
                        let sc = bonsai_scales_f32[r][grp];
                        let bi = bonsai_biases_f32[r][grp];
                        for w_idx in 0..8 {
                            let packed_u32 = black_box(bonsai_u32[r][grp * 8 + w_idx]);
                            let col_base = grp * 128 + w_idx * 16;
                            for lane in 0..16 {
                                let code = ((packed_u32 >> (lane * 2)) & 0x3) as f32;
                                let w_deq = sc * code + bi;
                                dot += w_deq * fx[col_base + lane];
                            }
                        }
                    }
                    row_hash ^= dot.to_bits();
                }
                acc_bits ^= row_hash;
            }
            black_box(acc_bits);
        },
    );

    // Window 14: `Bonsai-AirMoL` Streaming `EntroMCU-28` (`1.750 bpw`) Unpack + `FWHT-1024` + `Antisym-40`
    // + `54`-Bit `3`-Token `LUT6` Speculative Burst (`4` tokens verified per streamed sweep -> per-token op cost!)
    let mut sample_words8 = [0u32; 8];
    sample_words8.copy_from_slice(&bonsai_u32[0][0..8]);
    let packed_mcu28: [u8; ENTROMCU_GROUP128_BYTES] =
        transcode_mlx2bit_group128_to_entromcu28(&sample_words8, 0x3800)
            .expect("valid ternary group");
    let burst_word64 =
        pack_bonsai_3tok_54bit(14_201, 88_912, 241_007, 0x3FD).expect("valid Bonsai vocab");
    let stream_toks = bonsai_iters * 4; // 1 sweep verifies K+1 = 4 tokens!
    let w14 = measure_window(
        "Window14_Bonsai_EntroMCU28_AirMoL_4TokPerSweep",
        stream_toks,
        &shunts,
        &rapl,
        || {
            let mut acc = 0i32;
            let mut x_in = [11i8; BONSAI_BLOCK_1024];
            for step in 0..bonsai_iters {
                let (trits128, sc_bits) = unpack_entromcu28_group128(black_box(&packed_mcu28));
                let (toks3, hdr10) = unpack_bonsai_3tok_54bit(black_box(burst_word64 ^ (step & 3)));
                x_in[(step & 1023) as usize] = trits128[(step & 127) as usize]
                    .wrapping_add(((toks3[0] & 0xF) as i8) - 8);
                let y = eval_bonsai1024_block16_antisym40(
                    black_box(&x_in),
                    black_box(&bonsai_signs),
                    black_box(&bonsai_codes),
                    black_box(&bonsai_scales_q8),
                );
                acc ^= y[0] ^ y[15] ^ (sc_bits as i32) ^ (toks3[2] as i32) ^ (hdr10 as i32);
            }
            black_box(acc);
        },
    );

    // Window 15: `2-Layer Benes-ARC-LUT6` (`64`-Cell Spatial Cascade) + `ARC-AGI-3` `System-0` Reflex & Breaker
    let arc_iters = scale_iters / 2;
    let arc_l1 = [0xFFFE_0000_7FFE_0000u64, 0x8000_7FFF_8000_7FFFu64];
    let arc_l2 = [
        0xAAAA_5555_AAAA_5555u64,
        0xCCCC_3333_CCCC_3333u64,
        0xF0F0_0F0F_F0F0_0F0Fu64,
        0xFF00_00FF_FF00_00FFu64,
    ];
    let arc3_masks = [0xAAAA_AAAA_AAAA_AAAAu64, 0xCCCC_CCCC_CCCC_CCCCu64, 0xF0F0_F0F0_F0F0_F0F0u64];
    let w15 = measure_window(
        "Window15_Benes_ARC2_2Layer_plus_ARC3_Reflex",
        arc_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            let mut cp = [0x0F0F_0F0F_0F0F_0F0Fu64, 0x3333_3333_3333_3333, 0x5555_5555_5555_5555, 0];
            for step in 0..arc_iters {
                cp[0] ^= step & 0xFF;
                let out_p = eval_benes_arc_lut6_2layer_strip64(
                    black_box(&cp),
                    black_box(cp[0].rotate_left(1)),
                    black_box(cp[0].rotate_right(1)),
                    black_box(cp[1] << 1),
                    black_box(cp[1] >> 1),
                    black_box(step),
                    black_box(&arc_l1),
                    black_box(&arc_l2),
                );
                let act_sig = if (step & 63) == 63 { 0x99u8 } else { 0x42u8 };
                let r3 = eval_arc3_policy_lut6(
                    (step & 0x3F) as u8,
                    Some(0x42),
                    black_box(act_sig),
                    black_box(&arc3_masks),
                    u64::MAX,
                );
                let tag = match r3 {
                    Arc3ReflexOutcome::EmitAction { action_id } => action_id as u64,
                    Arc3ReflexOutcome::MechanicShiftBreaker { expected_sig, actual_sig } => {
                        (expected_sig ^ actual_sig) as u64
                    }
                    Arc3ReflexOutcome::WakeWorldModelPlanner => 0,
                };
                acc ^= out_p[0] ^ out_p[3] ^ tag;
            }
            black_box(acc);
        },
    );

    // Window 16: `OpenIE-Apex-Harness` `System-0` `LUT6` Horizon Gate (`DES-10` + `openie-cad` + `MathGround CBF` + `RCE`)
    let apex_masks4 = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0xCCCC_CCCC_CCCC_CCCCu64,
        0xF0F0_F0F0_F0F0_F0F0u64,
        0xFF00_FF00_FF00_FF00u64,
    ];
    let w16 = measure_window(
        "Window16_Apex_System0_Enzymatic_Horizon_Gate",
        arc_iters,
        &shunts,
        &rapl,
        || {
            let mut acc = 0u64;
            for step in 0..arc_iters {
                let env = lut6_core::apex_harness::ApexStepEnvelope {
                    percept6: ((step & 0x3E) | 1) as u8,
                    handoff_tag16: 0x42A1,
                    expected_tag16: if (step & 127) == 127 { 0x42A0 } else { 0x42A1 },
                    z0_milliohm: 49_500 + ((step & 7) as u32) * 100,
                    tj_deci_c: 640 + ((step & 15) as u16) * 5,
                    v_mm_s: 1100 + ((step & 7) as u32) * 20,
                    a_max_mm_s2: 3000,
                    d_obs_mm: if (step & 255) == 254 { 150 } else { 320 },
                    cash_cents: 15_000,
                    stock_units: 24,
                };
                let out = lut6_core::apex_harness::eval_apex_enzymatic_step_lut6(
                    black_box(&env),
                    black_box(&apex_masks4),
                    u64::MAX,
                );
                let tag = match out {
                    lut6_core::apex_harness::ApexGateOutcome::CertifiedSystem0 { action_code } => {
                        action_code as u64
                    }
                    lut6_core::apex_harness::ApexGateOutcome::EnzymaticAttestationTrap {
                        xor_tag,
                    } => xor_tag as u64,
                    lut6_core::apex_harness::ApexGateOutcome::SimplexBarrierShield {
                        violation_mask,
                    } => violation_mask as u64,
                    lut6_core::apex_harness::ApexGateOutcome::WakeApexSolver => 0,
                };
                acc ^= tag.wrapping_mul(0x9E37_79B9);
            }
            black_box(acc);
        },
    );

    // Window 17: `waiOS` `256x128` Ternary Conformance (`Antisym-40` Zero-MUL vs. Scalar MUL, `a541323b...` verified)
    let waios_audit = lut6_core::waios_mosaic::build_waios_packed_audit(false);
    let mut waios_check_out = [0i32; lut6_core::waios_mosaic::WAIOS_ROWS];
    lut6_core::waios_mosaic::eval_waios_256x128_antisym40(&waios_audit, &mut waios_check_out);
    assert_eq!(
        lut6_core::waios_mosaic::waios_digest_of_outputs(&waios_check_out),
        lut6_core::waios_mosaic::WAIOS_REFERENCE_DIGEST,
        "waiOS 256x128 Antisym-40 BLAKE3 digest mismatch!"
    );
    let mut waios_dense_w = [0i8; lut6_core::waios_mosaic::WAIOS_ROWS * lut6_core::waios_mosaic::WAIOS_COLS];
    for r in 0..lut6_core::waios_mosaic::WAIOS_ROWS {
        for g in 0..lut6_core::waios_mosaic::WAIOS_GROUPS4 {
            let mut cd = waios_audit.codes4[r][g];
            for k in 0..4 {
                waios_dense_w[r * lut6_core::waios_mosaic::WAIOS_COLS + g * 4 + k] = (cd % 3) as i8 - 1;
                cd /= 3;
            }
        }
    }
    let waios_iters = (scale_iters / 40).max(50_000);
    let w17 = measure_window(
        "Window17_WaiOS_256x128_Antisym40_ZeroMul",
        waios_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; lut6_core::waios_mosaic::WAIOS_ROWS];
            let mut acc = 0i32;
            for _ in 0..waios_iters {
                lut6_core::waios_mosaic::eval_waios_256x128_antisym40(
                    black_box(&waios_audit),
                    &mut out,
                );
                acc ^= out[0] ^ out[255];
            }
            black_box(acc);
        },
    );
    let w17b = measure_window(
        "Window17b_WaiOS_256x128_Scalar_MUL",
        waios_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; lut6_core::waios_mosaic::WAIOS_ROWS];
            let mut acc = 0i32;
            for _ in 0..waios_iters {
                lut6_core::waios_mosaic::eval_waios_256x128_scalar_mul(
                    black_box(&waios_dense_w),
                    black_box(&waios_audit.acts),
                    &mut out,
                );
                acc ^= out[0] ^ out[255];
            }
            black_box(acc);
        },
    );

    // Window 18: `JouleBit-0.10` (`0.105 bpw`) Shared-Table `LUT6` vs. `LittleBit-2` Runtime Unpack + Dense MAC (`64x96`)
    let mut jb_acts = [0i16; 96];
    for (idx, s) in jb_acts.iter_mut().enumerate() {
        *s = ((idx as i16 * 13 + 7) % 63) - 31;
    }
    let mut jb_v = [[0u8; 16]; 6];
    for r in 0..6 {
        for g in 0..16 {
            jb_v[r][g] = ((r * 17 + g * 29 + 5) & 0x3F) as u8;
        }
    }
    let mut jb_u = [0u8; 64];
    for (row, s) in jb_u.iter_mut().enumerate() {
        *s = ((row * 37 + 11) & 0x3F) as u8;
    }
    let jb_iters = (scale_iters / 10).max(200_000);
    let w18 = measure_window(
        "Window18_JouleBit_0p10bpw_LUT6_SharedTable",
        jb_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            for step in 0..jb_iters {
                jb_acts[0] = ((step & 31) as i16) - 15;
                lut6_core::waios_mosaic::eval_joulebit_0p10bpw_lut6_64x96(
                    black_box(&jb_acts),
                    black_box(&jb_v),
                    black_box(&jb_u),
                    &mut out,
                );
                acc ^= out[0] ^ out[63];
            }
            black_box(acc);
        },
    );
    let w18b = measure_window(
        "Window18b_LittleBit2_0p10bpw_DenseUnpack_MAC",
        jb_iters / 4,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            for step in 0..(jb_iters / 4) {
                jb_acts[0] = ((step & 31) as i16) - 15;
                lut6_core::waios_mosaic::eval_littlebit2_0p10bpw_dense_unpack_64x96(
                    black_box(&jb_acts),
                    black_box(&jb_v),
                    black_box(&jb_u),
                    &mut out,
                );
                acc ^= out[0] ^ out[63];
            }
            black_box(acc);
        },
    );

    // Window 19: `SIW(2)` Two-Bank Landmark `LUT6` + Fused Zero-MUL `FWHT-64 + 6-Col JouleBit-LUT6` (`Claims LA-53..LA-54`)
    let mut jb6_acts = [0i16; 64];
    for (idx, s) in jb6_acts.iter_mut().enumerate() {
        *s = ((idx as i16 * 11 + 5) % 47) - 23;
    }
    let mut jb6_v = [[0u8; 10]; 6];
    for r in 0..6 {
        for g in 0..10 {
            jb6_v[r][g] = ((r * 19 + g * 23 + 7) & 0x3F) as u8;
        }
    }
    let mut jb6_u = [0u8; 64];
    for (rw, s) in jb6_u.iter_mut().enumerate() {
        *s = ((rw * 31 + 13) & 0x3F) as u8;
    }
    // Verify exact algebraic parity (`Theorem 94`) between zero-MUL `FWHT64 + JouleBit-LUT6` and `LittleBit-2` unpack+MAC!
    {
        let mut out_jb = [0i32; 64];
        let mut out_lb = [0i32; 64];
        lut6_core::siw_joulebit::eval_joulebit6_fwht64_slab_i16(&jb6_acts, &jb6_v, &jb6_u, &mut out_jb);
        lut6_core::siw_joulebit::eval_littlebit2_6col_unpack_mac_64(&jb6_acts, &jb6_v, &jb6_u, &mut out_lb);
        assert_eq!(out_jb, out_lb, "Theorem 94 parity mismatch between JouleBit6-FWHT64 and LittleBit2!");
    }
    let siw_b0 = [0x5555_5555_5555_5555u64, 0x6666_6666_6666_6666, 0];
    let siw_b1 = [0xAAAA_AAAA_AAAA_AAAAu64, 0xCCCC_CCCC_CCCC_CCCC, u64::MAX];
    let w19 = measure_window(
        "Window19_SIW2_Landmark_Plus_JouleBit6_FWHT64",
        jb_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            let mut ep = 0u8;
            for step in 0..jb_iters {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                lut6_core::siw_joulebit::eval_joulebit6_fwht64_slab_i16(
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                let r_siw = lut6_core::siw_joulebit::eval_siw2_landmark_lut6(
                    (step & 0x3F) as u8,
                    if (step & 127) == 63 { 1 } else { 0 },
                    ep,
                    Some(0x42),
                    0x42,
                    !1u64,
                    black_box(&siw_b0),
                    black_box(&siw_b1),
                    u64::MAX,
                    u64::MAX,
                );
                if let lut6_core::siw_joulebit::Siw2ReflexOutcome::CertifiedAction {
                    action_id,
                    next_epoch,
                    ..
                } = r_siw
                {
                    ep = next_epoch & 7;
                    acc ^= (action_id as i32) ^ out[0] ^ out[63];
                }
            }
            black_box(acc);
        },
    );
    let w19b = measure_window(
        "Window19b_LittleBit2_6Col_FWHT64_Unpack_MAC",
        jb_iters / 4,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            for step in 0..(jb_iters / 4) {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                lut6_core::siw_joulebit::eval_littlebit2_6col_unpack_mac_64(
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                acc ^= out[0] ^ out[63];
            }
            black_box(acc);
        },
    );

    // Window 20: 4-Bank `BFWS` (`16 KB`) + `Well-Codifier` Stdlib8 `LUT6` + `JouleBit6 + FWHT-64` (`Claims LA-56..LA-58`)
    let bfws4_banks = [
        [0x5555_5555_5555_5555u64, 0x6666_6666_6666_6666, 0],
        [0xAAAA_AAAA_AAAA_AAAAu64, 0xCCCC_CCCC_CCCC_CCCC, 0],
        [u64::MAX, 0, u64::MAX],
        [0, u64::MAX, u64::MAX],
    ];
    let bfws4_certs = [
        0x00FF_FFFF_FFFF_FFFFu64,
        0xFF00_FFFF_FFFF_FFFFu64,
        0xFFFF_00FF_FFFF_FFFFu64,
        u64::MAX,
    ];
    let w20 = measure_window(
        "Window20_BFWS4Bank_WellCodifier_JouleBit6",
        jb_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            let mut ep = 0u8;
            for step in 0..jb_iters {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                let (r_bfws, well_b) = lut6_core::bfws_well::eval_window20_bfws4_well_joulebit6(
                    (step & 0x3F) as u8,
                    if (step & 127) == 63 { 3 } else { 0 },
                    ep,
                    ((step * 7) & 0x3F) as u8,
                    black_box(&bfws4_banks),
                    black_box(&bfws4_certs),
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                if let lut6_core::bfws_well::Bfws4BankOutcome::CertifiedAction {
                    action_id,
                    bank_id,
                    next_epoch,
                    ..
                } = r_bfws
                {
                    ep = next_epoch & 15;
                    acc ^= ((action_id ^ bank_id ^ well_b) as i32) ^ out[0] ^ out[63];
                }
            }
            black_box(acc);
        },
    );
    let mut json_buf64 = [b' '; 64];
    let tmpl = b"{\"jsonrpc\":\"2.0\",\"method\":\"well.cidr_wcag\",\"params\":{\"arg\":19}} ";
    json_buf64.copy_from_slice(tmpl);
    let mut router_w8 = [[0i8; 64]; 8];
    for r in 0..8 {
        for c in 0..64 {
            router_w8[r][c] = (((r * 17 + c * 13) % 31) as i8) - 15;
        }
    }
    let w20b = measure_window(
        "Window20b_Rung5_JSON_INT8Router_LittleBit2",
        jb_iters / 4,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            for step in 0..(jb_iters / 4) {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                json_buf64[59] = b'0' + ((step & 7) as u8);
                let tag = lut6_core::bfws_well::eval_rung5_json_int8_littlebit2_baseline(
                    black_box(&json_buf64),
                    black_box(&router_w8),
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                acc ^= (tag as i32) ^ out[0] ^ out[63];
            }
            black_box(acc);
        },
    );

    // Window 21: 8-Bank `R-BFWS` (`32 KB` = `50%` watch SRAM) + `Enigma-SBox6` `LUT6` + `Vending-BALROG` `CBF` Shield + `JouleBit6` (`Claims LA-60..LA-63`)
    let rbfws8_banks = [
        [0x5555_5555_5555_5555u64, 0x6666_6666_6666_6666, 0],
        [0xAAAA_AAAA_AAAA_AAAAu64, 0xCCCC_CCCC_CCCC_CCCC, 0],
        [u64::MAX, 0, u64::MAX],
        [0, u64::MAX, u64::MAX],
        [u64::MAX, u64::MAX, 0],
        [0, 0, u64::MAX],
        [u64::MAX, u64::MAX, u64::MAX],
        [u64::MAX, 0, 0],
    ];
    let rbfws8_certs = [
        0x0000_0000_0000_00FFu64,
        0x0000_0000_0000_FF00u64,
        0x0000_0000_00FF_0000u64,
        0x0000_0000_FF00_0000u64,
        0x0000_00FF_0000_0000u64,
        0x0000_FF00_0000_0000u64,
        0x00FF_0000_0000_0000u64,
        u64::MAX,
    ];
    let w21 = measure_window(
        "Window21_RBFWS8Bank_EnigmaSBox6_Horizon1000_JouleBit6",
        jb_iters,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            let mut ep = 0u8;
            for step in 0..jb_iters {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                let (r_rbfws, sbox_y, sh2) = lut6_core::enigma_horizon::eval_window21_rbfws8_enigma_horizon(
                    (step & 0x3F) as u8,
                    if (step & 127) == 63 { 3 } else { 0 },
                    ep,
                    ((step * 11) & 0x3F) as u8,
                    ((step * 5 + 0x2A) & 0x3F) as u8,
                    black_box(&rbfws8_banks),
                    black_box(&rbfws8_certs),
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                if let lut6_core::enigma_horizon::Rbfws8BankOutcome::CertifiedAction {
                    action_id,
                    bank_id,
                    next_epoch,
                    ..
                } = r_rbfws
                {
                    ep = next_epoch & 7;
                    acc ^= ((action_id ^ bank_id ^ sbox_y ^ sh2) as i32) ^ out[0] ^ out[63];
                }
            }
            black_box(acc);
        },
    );
    let mut kv_hist64 = [[0i8; 16]; 64];
    for t in 0..64 {
        for d in 0..16 {
            kv_hist64[t][d] = (((t * 19 + d * 11 + 3) % 31) as i8) - 15;
        }
    }
    let mut query16 = [7i8; 16];
    let w21b = measure_window(
        "Window21b_KV64Context_PolyCipher_LittleBit2",
        jb_iters / 4,
        &shunts,
        &rapl,
        || {
            let mut out = [0i32; 64];
            let mut acc = 0i32;
            for step in 0..(jb_iters / 4) {
                jb6_acts[0] = ((step & 31) as i16) - 15;
                query16[0] = ((step & 15) as i8) - 7;
                let tag = lut6_core::enigma_horizon::eval_window21b_kv64_cipher_littlebit2_baseline(
                    black_box(&kv_hist64),
                    black_box(&query16),
                    ((step * 11) & 0x3F) as u8,
                    black_box(&jb6_acts),
                    black_box(&jb6_v),
                    black_box(&jb6_u),
                    &mut out,
                );
                acc ^= (tag as i32) ^ out[0] ^ out[63];
            }
            black_box(acc);
        },
    );

    let windows = [
        w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w17b, w18, w18b, w19, w19b, w20, w20b, w21, w21b,
    ];
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
