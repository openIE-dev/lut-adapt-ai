use burn::tensor::{Device, Int, Tensor, TensorData};
use clai_cubecl::{
    cpu_cascade::run_rayon_host_demo,
    ext::{mamba3_chunk_ssd_gpu, rung0_lut6_gpu},
};
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

fn main() {
    // 1. Run Pure-Rust Rayon Host 4-Rung Cascade
    run_rayon_host_demo();

    // 2. Run Custom #[cube] GPU Kernels via Burn 0.22.1 + CubeCL JIT (Metal / CUDA / ROCm / CPU)
    let device = select_hardware_device();
    println!("[CLAI CubeCL 0.22.1] Active JIT Target Device: {:?}", device);

    // 2A. Rung 0: 64-Episode Zero-Float LUT6 GPU Chain ([64, 16] = 64 episodes x 8x64-bit words)
    let mut words_vec = Vec::with_capacity(64 * 16);
    for ep in 0..64i32 {
        for w in 0..16i32 {
            words_vec.push((ep.wrapping_mul(0x1357_9BDF)) ^ (w.wrapping_mul(0x2468_ACE1)));
        }
    }
    let words = Tensor::<2, Int>::from_data(TensorData::new(words_vec, [64, 16]), &device);

    let mut w0_vec = Vec::with_capacity(512 * 6);
    let mut t0_vec = Vec::with_capacity(512 * 2);
    for n in 0..512i32 {
        for p in 0..6i32 {
            w0_vec.push((n * 7 + p * 31) % 512);
        }
        // Non-trivial truth table masks
        t0_vec.push(0x6996_9669u32 as i32 ^ n);
        t0_vec.push(0xA55A_5AA5u32 as i32 ^ (n * 3));
    }
    let wires_0 = Tensor::<2, Int>::from_data(TensorData::new(w0_vec.clone(), [512, 6]), &device);
    let tables_0 = Tensor::<2, Int>::from_data(TensorData::new(t0_vec.clone(), [512, 2]), &device);
    let wires_1 = Tensor::<2, Int>::from_data(TensorData::new(w0_vec, [512, 6]), &device);
    let tables_1 = Tensor::<2, Int>::from_data(TensorData::new(t0_vec, [512, 2]), &device);

    // Warmup JIT shader compilation
    let _warmup = rung0_lut6_gpu(
        words.clone(),
        wires_0.clone(),
        tables_0.clone(),
        wires_1.clone(),
        tables_1.clone(),
        8,
    )
    .into_data();

    let t0 = Instant::now();
    let votes = rung0_lut6_gpu(words, wires_0, tables_0, wires_1, tables_1, 8);
    let votes_data = votes.into_data();
    let r0_gpu_us = t0.elapsed().as_secs_f64() * 1e6;
    let v_slice = votes_data.as_slice::<i32>().unwrap();
    println!(
        "[CLAI CubeCL 0.22.1] 64-Ep LUT6 #[cube] GPU Chain: {:.1} us | Ep0 Class Votes: {:?}",
        r0_gpu_us,
        &v_slice[0..8]
    );

    // 2B. Rung 2: Mamba-3 Chunkwise SSD Forward + Custom Autodiff Backward on GPU!
    let ad_device = device.clone().autodiff();
    let x_hat = (Tensor::<4>::ones([2, 2, 256, 64], &ad_device) * 0.12).require_grad();
    let b_hat = (Tensor::<4>::ones([2, 2, 256, 64], &ad_device) * 0.15).require_grad();
    let c_rot = (Tensor::<4>::ones([2, 2, 128, 64], &ad_device) * 0.10).require_grad();
    let cum_a = (Tensor::<3>::ones([2, 2, 32], &ad_device) * (-0.04)).require_grad();
    let end_dec = Tensor::<3>::ones([2, 2, 256], &ad_device) * 0.88;
    let chunk_dec = Tensor::<2>::ones([2, 2], &ad_device) * 0.28;
    let h0 = (Tensor::<3>::ones([2, 64, 64], &ad_device) * 0.05).require_grad();

    let t_ssd = Instant::now();
    let y_ssd = mamba3_chunk_ssd_gpu(
        x_hat.clone(),
        b_hat.clone(),
        c_rot.clone(),
        cum_a.clone(),
        end_dec,
        chunk_dec,
        h0.clone(),
    );
    let loss = y_ssd.clone().powf_scalar(2.0).mean();
    let grads = loss.backward();
    let dx = x_hat.grad(&grads).expect("Missing dx_hat gradient!");
    let db = b_hat.grad(&grads).expect("Missing db_hat gradient!");
    let dc = c_rot.grad(&grads).expect("Missing dc_rot gradient!");
    let dx_norm = dx.powf_scalar(2.0).sum().sqrt().into_scalar::<f32>();
    let db_norm = db.powf_scalar(2.0).sum().sqrt().into_scalar::<f32>();
    let dc_norm = dc.powf_scalar(2.0).sum().sqrt().into_scalar::<f32>();
    let ssd_ms = t_ssd.elapsed().as_secs_f64() * 1e3;

    println!(
        "[CLAI CubeCL 0.22.1] Mamba-3 SSD #[cube] Fwd+Bwd on {:?}: {:.2} ms | Grad Norms -> dX:{:.4}, dB:{:.4}, dC:{:.4}",
        device, ssd_ms, dx_norm, db_norm, dc_norm
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubecl_lut6_matches_scalar_truth_table_reference() {
        let device = select_hardware_device();
        let mut words_vec = vec![0i32; 64 * 16];
        // Set deterministic pattern on episode 0 and episode 45
        for h in 0..8 {
            words_vec[h * 2] = 0x1357_2468;
            words_vec[h * 2 + 1] = 0x55AA_0F0F;
            words_vec[45 * 16 + h * 2] = 0x0F0F_3333;
            words_vec[45 * 16 + h * 2 + 1] = 0x1234_5678;
        }
        let words =
            Tensor::<2, Int>::from_data(TensorData::new(words_vec.clone(), [64, 16]), &device);

        // Single-layer identity + known LUT6 table check via 2-layer chain (L1 = identity on I0)
        let mut w0 = vec![0i32; 512 * 6];
        let mut t0 = vec![0i32; 512 * 2];
        let mut w1 = vec![0i32; 512 * 6];
        let mut t1 = vec![0i32; 512 * 2];
        for n in 0..512 {
            for p in 0..6 {
                w0[n * 6 + p] = ((n * 5 + p * 13) % 512) as i32;
                w1[n * 6 + p] = n as i32; // All 6 ports point to LUT n of Layer 0
            }
            t0[n * 2] = 0x6996_9669u32 as i32;
            t0[n * 2 + 1] = 0x9669_6996u32 as i32;
            // Layer 1 = pure identity on I0 (all odd minterms = 1 => 0xAAAA_AAAA_AAAA_AAAA)
            t1[n * 2] = 0xAAAA_AAAAu32 as i32;
            t1[n * 2 + 1] = 0xAAAA_AAAAu32 as i32;
        }

        let votes = rung0_lut6_gpu(
            words,
            Tensor::from_data(TensorData::new(w0.clone(), [512, 6]), &device),
            Tensor::from_data(TensorData::new(t0.clone(), [512, 2]), &device),
            Tensor::from_data(TensorData::new(w1, [512, 6]), &device),
            Tensor::from_data(TensorData::new(t1, [512, 2]), &device),
            8,
        )
        .into_data();
        let gpu_votes = votes.as_slice::<i32>().unwrap();

        for &ep in &[0usize, 45usize] {
            let mut ref_bits = [0u32; 512];
            for h in 0..8 {
                let lo = words_vec[ep * 16 + h * 2] as u32;
                let hi = words_vec[ep * 16 + h * 2 + 1] as u32;
                for bit in 0..32 {
                    ref_bits[h * 64 + bit] = (lo >> bit) & 1;
                    ref_bits[h * 64 + 32 + bit] = (hi >> bit) & 1;
                }
            }
            let mask64 = ((t0[1] as u32 as u64) << 32) | (t0[0] as u32 as u64);
            for cls in 0..8 {
                let mut expected_vote = 0i32;
                for k in 0..64 {
                    let n = cls * 64 + k;
                    let mut addr = 0u32;
                    for p in 0..6 {
                        let wire = w0[n * 6 + p] as usize;
                        addr |= ref_bits[wire] << p;
                    }
                    expected_vote += ((mask64 >> addr) & 1) as i32;
                }
                assert_eq!(
                    gpu_votes[ep * 8 + cls],
                    expected_vote,
                    "Mismatch on ep={ep}, cls={cls}"
                );
            }
        }
    }

    #[test]
    fn cubecl_mamba3_ssd_autodiff_propagates_non_zero_gradients() {
        let ad_dev = select_hardware_device().autodiff();
        let x_hat = (Tensor::<4>::ones([1, 2, 256, 64], &ad_dev) * 0.2).require_grad();
        let b_hat = (Tensor::<4>::ones([1, 2, 256, 64], &ad_dev) * 0.15).require_grad();
        let c_rot = (Tensor::<4>::ones([1, 2, 128, 64], &ad_dev) * 0.1).require_grad();
        let cum_a = (Tensor::<3>::ones([1, 2, 32], &ad_dev) * (-0.05)).require_grad();
        let end_dec = Tensor::<3>::ones([1, 2, 256], &ad_dev) * 0.85;
        let chunk_dec = Tensor::<2>::ones([1, 2], &ad_dev) * 0.25;
        let h0 = (Tensor::<3>::ones([1, 64, 64], &ad_dev) * 0.1).require_grad();

        let y = mamba3_chunk_ssd_gpu(
            x_hat.clone(),
            b_hat.clone(),
            c_rot.clone(),
            cum_a.clone(),
            end_dec,
            chunk_dec,
            h0.clone(),
        );
        let grads = y.sum().backward();
        assert!(x_hat.grad(&grads).unwrap().abs().sum().into_scalar::<f32>() > 0.0);
        assert!(b_hat.grad(&grads).unwrap().abs().sum().into_scalar::<f32>() > 0.0);
        assert!(c_rot.grad(&grads).unwrap().abs().sum().into_scalar::<f32>() > 0.0);
        assert!(h0.grad(&grads).unwrap().abs().sum().into_scalar::<f32>() > 0.0);
    }
}
