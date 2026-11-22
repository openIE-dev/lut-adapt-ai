//! Native Rust 1.100 + Burn 0.22.1 / CubeCL Full 5,120-Episode Collapse Curve Runner.
//! Executes the complete 4-Rung Thermodynamic Cascade + Sleep Crystallizer + KV260 Verilog
//! export directly on Apple Metal / CUDA / ROCm / CPU without Python.

use burn::tensor::Device;
use clai_cubecl::collapse::{
    BASE_CLASSES, CompiledMetalRung0, RustMixtureOfLimits, RustModularConstraintEnv,
};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn select_hardware_device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::cuda(0);

    #[cfg(feature = "rocm")]
    return Device::rocm(0);

    #[cfg(feature = "metal")]
    return Device::metal(Default::default());

    #[cfg(feature = "vulkan")]
    return Device::vulkan(Default::default());

    #[cfg(not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "metal",
        feature = "vulkan"
    )))]
    return Device::cpu();
}

fn resolve_artifacts_dir() -> std::path::PathBuf {
    if Path::new("artifacts").exists() {
        Path::new("artifacts").to_path_buf()
    } else {
        let p = Path::new("../../artifacts");
        let _ = fs::create_dir_all(p);
        p.to_path_buf()
    }
}

fn main() {
    let t_start = Instant::now();
    let device = select_hardware_device();
    let mb_size = 64usize;
    let total_eps = 5_120usize;
    let n_steps = total_eps / mb_size;
    let mutate_every = 2_048usize;
    let novel_class_ep = 3_840usize;
    let compile_every = 512usize;

    let mut env = RustModularConstraintEnv::new(BASE_CLASSES, 42);
    let mut runtime = RustMixtureOfLimits::new(device.clone());

    println!(
        "[Rust Burn 0.22.1 / CubeCL] Full 4-Rung Collapse Curve on {:?} ({} episodes, 64-ep batches)",
        device, total_eps
    );

    let out_dir = resolve_artifacts_dir();
    let _ = fs::create_dir_all(&out_dir);
    let csv_path = out_dir.join("rust_collapse_curve_metrics.csv");
    let mut csv = File::create(&csv_path).expect("Failed to create rust_collapse_curve_metrics.csv");
    writeln!(
        csv,
        "episode,j_per_vc,r0_satiation_pct,r2_settle_pct,r25_soinn_pct,r3_settle_pct,verified_pass_rate,lut_holdout_agreement,soinn_active_nodes"
    )
    .unwrap();

    let mut win_j = 0.0f64;
    let mut win_ver = 0usize;
    let mut win_r0 = 0usize;
    let mut win_r2 = 0usize;
    let mut win_soinn = 0usize;
    let mut win_r3 = 0usize;
    let mut win_eps = 0usize;
    let mut last_holdout = 0.0f32;
    let mut best_verilog: Option<String> = None;

    for step_idx in 1..=n_steps {
        let ep_prev = (step_idx - 1) * mb_size;
        let ep_done = step_idx * mb_size;

        if ep_prev > 0 && ep_prev % mutate_every == 0 {
            env.mutate_regime();
            println!(
                "[MUTATION] Episode {}: Rule schema shifted -> Regime {}",
                ep_prev, env.regime_id
            );
        }

        if ep_prev == novel_class_ep {
            let new_cls = env.spawn_novel_class();
            println!(
                "[OPEN-WORLD] Episode {}: Spawned unseen Novel Class ID {} (total classes={})",
                ep_prev, new_cls, env.n_classes
            );
        }

        let (episodes, _) = env.sample_microbatch(mb_size);
        let rec = runtime.step_microbatch(&env, &episodes);

        win_j += rec.joules;
        win_ver += rec.verified;
        win_r0 += rec.r0;
        win_r2 += rec.r2;
        win_soinn += rec.soinn;
        win_r3 += rec.r3;
        win_eps += mb_size;

        if ep_done % compile_every == 0 {
            if runtime.rung0.is_none() && runtime.crystallizer.count >= 512 {
                let (compiled_opt, agreement): (Option<CompiledMetalRung0>, f32) =
                    runtime.crystallizer.compile_downward(BASE_CLASSES, &device);
                last_holdout = agreement;
                if let Some(c_r0) = compiled_opt {
                    best_verilog = Some(c_r0.export_kv260_verilog("clai_rung0_core"));
                    runtime.rung0 = Some(c_r0);
                }
            }

            let j_per_vc = win_j / (win_ver.max(1) as f64);
            let r0_pct = 100.0 * (win_r0 as f64) / (win_eps as f64);
            let r2_pct = 100.0 * (win_r2 as f64) / (win_eps as f64);
            let r25_pct = 100.0 * (win_soinn as f64) / (win_eps as f64);
            let r3_pct = 100.0 * (win_r3 as f64) / (win_eps as f64);
            let v_rate = (win_ver as f64) / (win_eps as f64);
            let soinn_nodes = runtime.soinn.prototypes.len();

            writeln!(
                csv,
                "{},{:.6},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{}",
                ep_done,
                j_per_vc,
                r0_pct,
                r2_pct,
                r25_pct,
                r3_pct,
                v_rate,
                last_holdout,
                soinn_nodes
            )
            .unwrap();
            csv.flush().unwrap();

            println!(
                "[Ep {:5}] J/VC: {:9.5} J | R0(LUT6): {:5.1}% | R2(Mamba3): {:5.1}% | R2.5(SOINN): {:5.1}% ({:3} nodes) | R3(EBRM): {:5.1}% | LUT Holdout: {:5.1}%",
                ep_done,
                j_per_vc,
                r0_pct,
                r2_pct,
                r25_pct,
                soinn_nodes,
                r3_pct,
                last_holdout * 100.0
            );

            win_j = 0.0;
            win_ver = 0;
            win_r0 = 0;
            win_r2 = 0;
            win_soinn = 0;
            win_r3 = 0;
            win_eps = 0;
        }
    }

    if let Some(ref r0) = runtime.rung0 {
        best_verilog = Some(r0.export_kv260_verilog("clai_rung0_core"));
    }
    if let Some(v_src) = best_verilog {
        let v_path = out_dir.join("rust_clai_rung0_kv260.v");
        fs::write(&v_path, v_src).expect("Failed to write rust_clai_rung0_kv260.v");
        println!(
            "[Rust Burn 0.22.1 / CubeCL] Synthesizable KV260 LUT6 Verilog written to {:?} (Wall time: {:.2} s)",
            v_path,
            t_start.elapsed().as_secs_f64()
        );
    }
}
