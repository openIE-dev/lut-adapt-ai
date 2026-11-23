//! Phase 27 (Claim LA-18): Cross-Platform `Burn 0.22.1 / CubeCL` Benchmark & Numerical Parity
//! Runner for `q4_moe_ffn_pairs_gpu` and `gated_delta_recurrent_k128_gpu` across:
//!   1. Host Scalar/Rayon Dequantize+Matmul Reference (`FP32`)
//!   2. CubeCL Portable `#[cube]` LLVM SIMD Backend (`Device::cpu()`)
//!   3. CubeCL Portable `#[cube]` Hardware GPU Backend (`Metal` / `Vulkan` / `CUDA` / `ROCm`)

use burn::tensor::{Device, Int, Tensor, TensorData};
use clai_cubecl::ext::{gated_delta_recurrent_k128_gpu, q4_moe_ffn_pairs_gpu};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn select_gpu_device() -> (&'static str, Device) {
    #[cfg(feature = "cuda")]
    return ("CubeCL_CUDA_PTX", Device::cuda(0));

    #[cfg(feature = "rocm")]
    return ("CubeCL_ROCm_HIP", Device::rocm(0));

    #[cfg(feature = "vulkan")]
    return ("CubeCL_Vulkan_SPIRV", Device::vulkan(Default::default()));

    #[cfg(feature = "metal")]
    return ("CubeCL_Metal_GPU", Device::metal(Default::default()));

    #[cfg(not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal"
    )))]
    return ("CubeCL_CPU_Only", Device::cpu());
}

fn cpu_ref_q4_moe(
    packed_u32: &[i32], // [K_uniq, m_tot, 8]
    mins: &[f32],       // [K_uniq, m_tot]
    scales: &[f32],     // [K_uniq, m_tot]
    flat_h: &[f32],     // [T, D]
    pair_tok: &[i32],   // [P]
    pair_slot: &[i32],  // [P]
    pair_rw: &[f32],    // [P]
    d_model: usize,
    i_dim: usize,
    m_tot: usize,
) -> Vec<f32> {
    let p_pairs = pair_tok.len();
    let n_blks_d = d_model / 64;
    let n_blks_i = i_dim / 64;
    let m_gu = 2 * i_dim * n_blks_d;

    let mut out = vec![0.0_f32; p_pairs * d_model];
    let mut interm = vec![0.0_f32; i_dim];

    for p in 0..p_pairs {
        let tok = pair_tok[p] as usize;
        let slot = pair_slot[p] as usize;
        let rw = pair_rw[p];
        let x = &flat_h[tok * d_model..(tok + 1) * d_model];
        let slot_base = slot * m_tot;

        for i_row in 0..i_dim {
            let gm_base = slot_base + i_row * n_blks_d;
            let um_base = slot_base + (i_dim + i_row) * n_blks_d;
            let mut g_acc = 0.0_f32;
            let mut u_acc = 0.0_f32;
            for blk in 0..n_blks_d {
                let gm = gm_base + blk;
                let um = um_base + blk;
                let mut sx = 0.0_f32;
                let mut g_sq = 0.0_f32;
                let mut u_sq = 0.0_f32;
                for b_idx in 0..32usize {
                    let x0 = x[blk * 64 + b_idx * 2];
                    let x1 = x[blk * 64 + b_idx * 2 + 1];
                    sx += x0 + x1;
                    let gw = packed_u32[gm * 8 + (b_idx / 4)] as u32;
                    let gb = (gw >> ((b_idx % 4) * 8)) & 0xFF;
                    g_sq += ((gb & 0x0F) as f32) * x0 + (((gb >> 4) & 0x0F) as f32) * x1;

                    let uw = packed_u32[um * 8 + (b_idx / 4)] as u32;
                    let ub = (uw >> ((b_idx % 4) * 8)) & 0xFF;
                    u_sq += ((ub & 0x0F) as f32) * x0 + (((ub >> 4) & 0x0F) as f32) * x1;
                }
                g_acc += mins[gm] * sx + scales[gm] * g_sq;
                u_acc += mins[um] * sx + scales[um] * u_sq;
            }
            let silu = g_acc / (1.0 + (-g_acc).exp());
            interm[i_row] = silu * u_acc;
        }

        for d_row in 0..d_model {
            let dm_base = slot_base + m_gu + d_row * n_blks_i;
            let mut acc = 0.0_f32;
            for blk in 0..n_blks_i {
                let dm = dm_base + blk;
                let mut sx = 0.0_f32;
                let mut sq = 0.0_f32;
                for b_idx in 0..32usize {
                    let x0 = interm[blk * 64 + b_idx * 2];
                    let x1 = interm[blk * 64 + b_idx * 2 + 1];
                    sx += x0 + x1;
                    let dw = packed_u32[dm * 8 + (b_idx / 4)] as u32;
                    let db = (dw >> ((b_idx % 4) * 8)) & 0xFF;
                    sq += ((db & 0x0F) as f32) * x0 + (((db >> 4) & 0x0F) as f32) * x1;
                }
                acc += mins[dm] * sx + scales[dm] * sq;
            }
            out[p * d_model + d_row] = acc * rw;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_ref_deltanet(
    q_all: &[f32],    // [B, H_k, T, 128]
    k_all: &[f32],    // [B, H_k, T, 128]
    v_all: &[f32],    // [B, H_v, T, V_dim]
    g_all: &[f32],    // [B, H_v, T]
    beta_all: &[f32], // [B, H_v, T]
    state_in: &[f32], // [B, H_v, 128, V_dim]
    b_sz: usize,
    h_k: usize,
    h_v: usize,
    t_len: usize,
    v_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = state_in.to_vec();
    let mut out = vec![0.0_f32; b_sz * h_v * t_len * v_dim];
    let gqa = h_v / h_k;

    for b in 0..b_sz {
        for hv in 0..h_v {
            let hk = hv / gqa;
            let bhv = b * h_v + hv;
            let bhk = b * h_k + hk;
            for v in 0..v_dim {
                let st_base = bhv * 128 * v_dim + v;
                for t in 0..t_len {
                    let dec = g_all[bhv * t_len + t].exp();
                    let bet = beta_all[bhv * t_len + t];
                    let vt = v_all[(bhv * t_len + t) * v_dim + v];
                    let qk_base = (bhk * t_len + t) * 128;

                    let mut kv_mem = 0.0_f32;
                    for kd in 0..128 {
                        let s_dec = state[st_base + kd * v_dim] * dec;
                        state[st_base + kd * v_dim] = s_dec;
                        kv_mem += s_dec * k_all[qk_base + kd];
                    }
                    let delta = (vt - kv_mem) * bet;
                    let mut out_v = 0.0_f32;
                    for kd in 0..128 {
                        let s_new = state[st_base + kd * v_dim] + delta * k_all[qk_base + kd];
                        state[st_base + kd * v_dim] = s_new;
                        out_v += s_new * q_all[qk_base + kd];
                    }
                    out[(bhv * t_len + t) * v_dim + v] = out_v;
                }
            }
        }
    }
    (out, state)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn main() {
    println!("===================================================================================================");
    println!(" PHASE 27 (CLAIM LA-18): BURN 0.22.1 / CUBECL CROSS-PLATFORM INT4 MOE & GATED-DELTANET BENCHMARK");
    println!("===================================================================================================");

    let (gpu_name, gpu_dev) = select_gpu_device();
    let cpu_dev = Device::cpu();

    // 1. Qwen3.6-35B-A3B Expert Layer Geometry: D=2048, I_dim=512, K_uniq=28, T=4, P=32
    let d_model = 2048usize;
    let i_dim = 512usize;
    let k_uniq = 28usize;
    let t_toks = 4usize;
    let p_pairs = 32usize;
    let m_tot = 2 * i_dim * (d_model / 64) + d_model * (i_dim / 64); // 32768 + 16384 = 49152 blocks

    struct XorShift64(u64);
    impl XorShift64 {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 & 0xFFFF_FFFF) as u32
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next_u32() as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
    }
    let mut rng = XorShift64(0x2026_1123_CAFE_BABEu64);

    let packed_vec: Vec<i32> = (0..k_uniq * m_tot * 8).map(|_| rng.next_u32() as i32).collect();
    let mins_vec: Vec<f32> = (0..k_uniq * m_tot).map(|_| rng.next_f32() * 0.04).collect();
    let scales_vec: Vec<f32> = (0..k_uniq * m_tot)
        .map(|_| (rng.next_f32().abs() + 0.05) * 0.008)
        .collect();
    let flat_h_vec: Vec<f32> = (0..t_toks * d_model).map(|_| rng.next_f32() * 0.5).collect();
    let pair_tok_vec: Vec<i32> = (0..p_pairs).map(|p| (p % t_toks) as i32).collect();
    let pair_slot_vec: Vec<i32> = (0..p_pairs).map(|p| (p % k_uniq) as i32).collect();
    let pair_rw_vec: Vec<f32> = (0..p_pairs).map(|p| 0.125 + 0.01 * ((p % 4) as f32)).collect();

    let t_ref0 = Instant::now();
    let ref_moe = cpu_ref_q4_moe(
        &packed_vec,
        &mins_vec,
        &scales_vec,
        &flat_h_vec,
        &pair_tok_vec,
        &pair_slot_vec,
        &pair_rw_vec,
        d_model,
        i_dim,
        m_tot,
    );
    let ref_moe_ms = t_ref0.elapsed().as_secs_f64() * 1e3;

    let run_moe_on_dev = |dev: &Device, iters: usize| -> (Vec<f32>, f64) {
        let p_t = Tensor::<3, Int>::from_data(TensorData::new(packed_vec.clone(), [k_uniq, m_tot, 8]), dev);
        let m_t = Tensor::<2>::from_data(TensorData::new(mins_vec.clone(), [k_uniq, m_tot]), dev);
        let s_t = Tensor::<2>::from_data(TensorData::new(scales_vec.clone(), [k_uniq, m_tot]), dev);
        let h_t = Tensor::<2>::from_data(TensorData::new(flat_h_vec.clone(), [t_toks, d_model]), dev);
        let tok_t = Tensor::<1, Int>::from_data(TensorData::new(pair_tok_vec.clone(), [p_pairs]), dev);
        let slot_t = Tensor::<1, Int>::from_data(TensorData::new(pair_slot_vec.clone(), [p_pairs]), dev);
        let rw_t = Tensor::<1>::from_data(TensorData::new(pair_rw_vec.clone(), [p_pairs]), dev);

        // Warmup JIT
        let warm = q4_moe_ffn_pairs_gpu(
            p_t.clone(),
            m_t.clone(),
            s_t.clone(),
            h_t.clone(),
            tok_t.clone(),
            slot_t.clone(),
            rw_t.clone(),
            i_dim,
        )
        .into_data();
        let warm_vec = warm.as_slice::<f32>().unwrap().to_vec();

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = q4_moe_ffn_pairs_gpu(
                p_t.clone(),
                m_t.clone(),
                s_t.clone(),
                h_t.clone(),
                tok_t.clone(),
                slot_t.clone(),
                rw_t.clone(),
                i_dim,
            );
            let d = out.into_data();
            std::hint::black_box(d.as_slice::<f32>().unwrap()[0]);
        }
        let mean_ms = (t0.elapsed().as_secs_f64() * 1e3) / (iters as f64);
        (warm_vec, mean_ms)
    };

    let (cpu_cube_moe, cpu_cube_moe_ms) = run_moe_on_dev(&cpu_dev, 5);
    let (gpu_cube_moe, gpu_cube_moe_ms) = run_moe_on_dev(&gpu_dev, 40);

    let err_moe_cpu = max_abs_diff(&ref_moe, &cpu_cube_moe);
    let err_moe_gpu = max_abs_diff(&ref_moe, &gpu_cube_moe);
    println!(
        "  [Q4 Sparse-MoE 28-Exp P=32] HostRef={:.3} ms | CubeCL-CPU={:.3} ms (err={:.2e}) | {}={:.3} ms ({:.2}x vs HostRef, err={:.2e})",
        ref_moe_ms,
        cpu_cube_moe_ms,
        err_moe_cpu,
        gpu_name,
        gpu_cube_moe_ms,
        ref_moe_ms / gpu_cube_moe_ms,
        err_moe_gpu
    );
    assert!(err_moe_cpu < 1e-4, "CubeCL CPU Q4 MoE error too high: {err_moe_cpu}");
    assert!(err_moe_gpu < 1e-4, "CubeCL GPU Q4 MoE error too high: {err_moe_gpu}");

    // 2. Qwen3.6-35B-A3B GatedDeltaNet Geometry: B=1, H_k=16, H_v=32, T=256, K=128, V=128
    let b_sz = 1usize;
    let h_k = 16usize;
    let h_v = 32usize;
    let t_len = 256usize;
    let v_dim = 128usize;

    let q_vec: Vec<f32> = (0..b_sz * h_k * t_len * 128).map(|_| rng.next_f32() * 0.15).collect();
    let k_vec: Vec<f32> = (0..b_sz * h_k * t_len * 128).map(|_| rng.next_f32() * 0.15).collect();
    let v_vec: Vec<f32> = (0..b_sz * h_v * t_len * v_dim).map(|_| rng.next_f32() * 0.25).collect();
    let g_vec: Vec<f32> = (0..b_sz * h_v * t_len)
        .map(|_| -0.05 - (rng.next_f32().abs() * 0.25))
        .collect();
    let beta_vec: Vec<f32> = (0..b_sz * h_v * t_len)
        .map(|_| 0.1 + rng.next_f32().abs() * 0.4)
        .collect();
    let st0_vec: Vec<f32> = (0..b_sz * h_v * 128 * v_dim).map(|_| rng.next_f32() * 0.05).collect();

    let t_dn0 = Instant::now();
    let (ref_dn_out, ref_dn_st) = cpu_ref_deltanet(
        &q_vec, &k_vec, &v_vec, &g_vec, &beta_vec, &st0_vec, b_sz, h_k, h_v, t_len, v_dim,
    );
    let ref_dn_ms = t_dn0.elapsed().as_secs_f64() * 1e3;

    let run_dn_on_dev = |dev: &Device, iters: usize| -> (Vec<f32>, Vec<f32>, f64) {
        let q_t = Tensor::<4>::from_data(TensorData::new(q_vec.clone(), [b_sz, h_k, t_len, 128]), dev);
        let k_t = Tensor::<4>::from_data(TensorData::new(k_vec.clone(), [b_sz, h_k, t_len, 128]), dev);
        let v_t = Tensor::<4>::from_data(TensorData::new(v_vec.clone(), [b_sz, h_v, t_len, v_dim]), dev);
        let g_t = Tensor::<3>::from_data(TensorData::new(g_vec.clone(), [b_sz, h_v, t_len]), dev);
        let b_t = Tensor::<3>::from_data(TensorData::new(beta_vec.clone(), [b_sz, h_v, t_len]), dev);
        let s_t = Tensor::<4>::from_data(TensorData::new(st0_vec.clone(), [b_sz, h_v, 128, v_dim]), dev);

        let (wo, ws) = gated_delta_recurrent_k128_gpu(
            q_t.clone(),
            k_t.clone(),
            v_t.clone(),
            g_t.clone(),
            b_t.clone(),
            s_t.clone(),
        );
        let wo_data = wo.into_data();
        let ws_data = ws.into_data();
        let wo_vec = wo_data.as_slice::<f32>().unwrap().to_vec();
        let ws_vec = ws_data.as_slice::<f32>().unwrap().to_vec();

        let t0 = Instant::now();
        for _ in 0..iters {
            let (o, s) = gated_delta_recurrent_k128_gpu(
                q_t.clone(),
                k_t.clone(),
                v_t.clone(),
                g_t.clone(),
                b_t.clone(),
                s_t.clone(),
            );
            let od = o.into_data();
            let sd = s.into_data();
            std::hint::black_box(od.as_slice::<f32>().unwrap()[0] + sd.as_slice::<f32>().unwrap()[0]);
        }
        let mean_ms = (t0.elapsed().as_secs_f64() * 1e3) / (iters as f64);
        (wo_vec, ws_vec, mean_ms)
    };

    let (cpu_dn_out, cpu_dn_st, cpu_dn_ms) = run_dn_on_dev(&cpu_dev, 5);
    let (gpu_dn_out, gpu_dn_st, gpu_dn_ms) = run_dn_on_dev(&gpu_dev, 40);

    let err_dn_cpu = max_abs_diff(&ref_dn_out, &cpu_dn_out).max(max_abs_diff(&ref_dn_st, &cpu_dn_st));
    let err_dn_gpu = max_abs_diff(&ref_dn_out, &gpu_dn_out).max(max_abs_diff(&ref_dn_st, &gpu_dn_st));
    println!(
        "  [GatedDeltaNet T=256 H=32]  HostRef={:.3} ms | CubeCL-CPU={:.3} ms (err={:.2e}) | {}={:.3} ms ({:.2}x vs HostRef, err={:.2e})",
        ref_dn_ms,
        cpu_dn_ms,
        err_dn_cpu,
        gpu_name,
        gpu_dn_ms,
        ref_dn_ms / gpu_dn_ms,
        err_dn_gpu
    );
    assert!(err_dn_cpu < 1e-4, "CubeCL CPU DeltaNet error too high: {err_dn_cpu}");
    assert!(err_dn_gpu < 1e-4, "CubeCL GPU DeltaNet error too high: {err_dn_gpu}");

    let out_dir = if Path::new("artifacts").is_dir() {
        Path::new("artifacts").to_path_buf()
    } else {
        std::env::current_dir().unwrap().join("../artifacts")
    };
    let _ = std::fs::create_dir_all(&out_dir);
    let csv_path = out_dir.join("cubecl_cross_platform_moe_deltanet.csv");
    let mut f = File::create(&csv_path).expect("create cubecl csv");
    writeln!(
        f,
        "kernel_family,shape_spec,backend_target,latency_ms,speedup_vs_host_ref,max_abs_fp32_err,portable_targets"
    )
    .unwrap();
    writeln!(
        f,
        "q4_moe_gate_up_silu_down,D=2048_I=512_K=28_P=32,Host_Scalar_FP32_Ref,{:.4},1.0000,0.0e0,Rust_Std",
        ref_moe_ms
    )
    .unwrap();
    writeln!(
        f,
        "q4_moe_gate_up_silu_down,D=2048_I=512_K=28_P=32,CubeCL_LLVM_CPU,{:.4},{:.4},{:.3e},Metal+Vulkan+CUDA+ROCm+CPU",
        cpu_cube_moe_ms,
        ref_moe_ms / cpu_cube_moe_ms,
        err_moe_cpu
    )
    .unwrap();
    writeln!(
        f,
        "q4_moe_gate_up_silu_down,D=2048_I=512_K=28_P=32,{},{:.4},{:.4},{:.3e},Metal+Vulkan+CUDA+ROCm+CPU",
        gpu_name,
        gpu_cube_moe_ms,
        ref_moe_ms / gpu_cube_moe_ms,
        err_moe_gpu
    )
    .unwrap();
    writeln!(
        f,
        "gated_delta_recurrent_k128,B=1_Hk=16_Hv=32_T=256_K=128_V=128,Host_Scalar_FP32_Ref,{:.4},1.0000,0.0e0,Rust_Std",
        ref_dn_ms
    )
    .unwrap();
    writeln!(
        f,
        "gated_delta_recurrent_k128,B=1_Hk=16_Hv=32_T=256_K=128_V=128,CubeCL_LLVM_CPU,{:.4},{:.4},{:.3e},Metal+Vulkan+CUDA+ROCm+CPU",
        cpu_dn_ms,
        ref_dn_ms / cpu_dn_ms,
        err_dn_cpu
    )
    .unwrap();
    writeln!(
        f,
        "gated_delta_recurrent_k128,B=1_Hk=16_Hv=32_T=256_K=128_V=128,{},{:.4},{:.4},{:.3e},Metal+Vulkan+CUDA+ROCm+CPU",
        gpu_name,
        gpu_dn_ms,
        ref_dn_ms / gpu_dn_ms,
        err_dn_gpu
    )
    .unwrap();

    println!("Saved -> {}", csv_path.display());
}
