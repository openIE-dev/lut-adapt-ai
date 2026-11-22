//! Rung 2 Custom #[cube] GPU Kernels: Mamba-3 Chunkwise SSD Forward & Analytical Adjoint Backward.
//! Structured around 2D Cooperative Shared SRAM Register Tiling (16x16 = 256 threads/cube, <= 10 KB
//! shared memory per block) so the exact same Rust #[cube] kernels compile natively to Apple Metal
//! (MSL), NVIDIA CUDA (PTX), AMD ROCm (HIP), Vulkan (SPIR-V), and LLVM CPU.

use burn_backend::{Shape, TensorMetadata, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    cubecl::{self, CubeCount, CubeDim, ir::ElemType, prelude::*},
    kernel::into_contiguous,
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// Forward Phase 1: Cooperative SRAM-tiled chunk state summary S_c = (end_decay * B_hat)^T @ X_hat
/// Shapes: b_hat [BH, C, Q2R, 64], x_hat [BH, C, Q2R, 64], end_decay [BH, C, Q2R] -> s_out [BH, C, 64, 64]
#[cube(launch)]
pub fn mamba3_chunk_state_cube_kernel<F: Float>(
    b_hat: &Tensor<F>,
    x_hat: &Tensor<F>,
    end_decay: &Tensor<F>,
    s_out: &mut Tensor<F>,
    #[comptime] num_chunks: u32,
    #[comptime] q2r: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let c = CUBE_POS_Y;
    let tx = UNIT_POS_X; // 0..16
    let ty = UNIT_POS_Y; // 0..16
    let tid = tx * 16u32 + ty;

    let mut smem_b = Shared::new_slice(1024usize);
    let mut smem_x = Shared::new_slice(1024usize);

    let mut acc = Array::<F>::new(16usize);
    #[unroll]
    for i in 0usize..16usize {
        acc[i] = F::new(0.0_f32);
    }

    let base_bhc = (bh * num_chunks + c) * q2r;
    let mut k0 = 0u32;
    while k0 < q2r {
        #[unroll]
        for ld in 0u32..4u32 {
            let flat = tid * 4u32 + ld;
            let kk = flat / 64u32;
            let col = flat % 64u32;
            let g_row = base_bhc + k0 + kk;
            let dec = end_decay[g_row as usize];

            smem_b[flat as usize] = b_hat[(g_row * 64u32 + col) as usize] * dec;
            smem_x[flat as usize] = x_hat[(g_row * 64u32 + col) as usize];
        }
        sync_cube();

        for kk in 0u32..16u32 {
            let k_off = kk * 64u32;
            #[unroll]
            for ni in 0u32..4u32 {
                let b_val = smem_b[(k_off + tx * 4u32 + ni) as usize];
                #[unroll]
                for pj in 0u32..4u32 {
                    let x_val = smem_x[(k_off + ty * 4u32 + pj) as usize];
                    let reg_idx = (ni * 4u32 + pj) as usize;
                    acc[reg_idx] = acc[reg_idx] + b_val * x_val;
                }
            }
        }
        sync_cube();
        k0 += 16u32;
    }

    let s_base = (bh * num_chunks + c) * 4096u32;
    #[unroll]
    for ni in 0u32..4u32 {
        let n_row = tx * 4u32 + ni;
        #[unroll]
        for pj in 0u32..4u32 {
            let p_col = ty * 4u32 + pj;
            s_out[(s_base + n_row * 64u32 + p_col) as usize] = acc[(ni * 4u32 + pj) as usize];
        }
    }
}

/// Forward Phase 2: Exact sequential inter-chunk state scan over C chunks in registers.
#[cube(launch)]
pub fn mamba3_interchunk_scan_cube_kernel<F: Float>(
    s_chunks: &Tensor<F>,      // [BH, C, 64, 64]
    chunk_decay: &Tensor<F>,   // [BH, C]
    h0: &Tensor<F>,            // [BH, 64, 64]
    prev_h_out: &mut Tensor<F>,// [BH, C, 64, 64]
    h_final: &mut Tensor<F>,   // [BH, 64, 64]
    #[comptime] num_chunks: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let tid = UNIT_POS_X * 16u32 + UNIT_POS_Y;

    let mut h_reg = Array::<F>::new(16usize);
    let h0_base = bh * 4096u32;

    #[unroll]
    for i in 0u32..16u32 {
        let elem = tid * 16u32 + i;
        h_reg[i as usize] = h0[(h0_base + elem) as usize];
    }

    for c in 0u32..num_chunks {
        let c_base = (bh * num_chunks + c) * 4096u32;
        let dec = chunk_decay[(bh * num_chunks + c) as usize];

        #[unroll]
        for i in 0u32..16u32 {
            let elem = tid * 16u32 + i;
            let idx = (c_base + elem) as usize;
            prev_h_out[idx] = h_reg[i as usize];
            h_reg[i as usize] = dec * h_reg[i as usize] + s_chunks[idx];
        }
    }

    #[unroll]
    for i in 0u32..16u32 {
        let elem = tid * 16u32 + i;
        h_final[(h0_base + elem) as usize] = h_reg[i as usize];
    }
}

/// Forward Phase 3: Fused Off-Diagonal State Readout + Intra-Chunk Surrogate Attention SSD Output.
#[cube(launch)]
pub fn mamba3_ssd_output_cube_kernel<F: Float>(
    c_rot: &Tensor<F>,     // [BH, C, QR, 64]
    b_hat: &Tensor<F>,     // [BH, C, Q2R, 64]
    x_hat: &Tensor<F>,     // [BH, C, Q2R, 64]
    cum_a: &Tensor<F>,     // [BH, C, Q]
    prev_h: &Tensor<F>,    // [BH, C, 64, 64]
    y_out: &mut Tensor<F>, // [BH, C, QR, 64]
    #[comptime] num_chunks: u32,
    #[comptime] q_len: u32,
    #[comptime] rank_r: u32,
    #[comptime] qr: u32,
    #[comptime] q2r: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let c = CUBE_POS_Y;
    let row_blk = CUBE_POS_Z;
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let tid = tx * 16u32 + ty;

    let qr_row = row_blk * 16u32 + tx;
    let t_i = qr_row / rank_r;
    let a_base = (bh * num_chunks + c) * q_len;
    let a_i = cum_a[(a_base + t_i) as usize];
    let exp_ai = a_i.exp();

    let mut smem_c = Shared::new_slice(1024usize);
    let mut smem_b = Shared::new_slice(1024usize);
    let mut smem_x = Shared::new_slice(1024usize);
    let mut smem_s = Shared::new_slice(256usize);

    let c_gbase = (bh * num_chunks + c) * qr * 64u32 + row_blk * 16u32 * 64u32;
    #[unroll]
    for ld in 0u32..4u32 {
        let flat = tid * 4u32 + ld;
        smem_c[flat as usize] = c_rot[(c_gbase + flat) as usize];
    }
    sync_cube();

    let mut y_acc = Array::<F>::new(4usize);
    #[unroll]
    for pj in 0usize..4usize {
        y_acc[pj] = F::new(0.0_f32);
    }

    let h_gbase = (bh * num_chunks + c) * 4096u32;
    for n in 0u32..64u32 {
        let c_val = smem_c[(tx * 64u32 + n) as usize] * exp_ai;
        #[unroll]
        for pj in 0u32..4u32 {
            let p_col = ty * 4u32 + pj;
            y_acc[pj as usize] =
                y_acc[pj as usize] + c_val * prev_h[(h_gbase + n * 64u32 + p_col) as usize];
        }
    }
    sync_cube();

    let bx_gbase = (bh * num_chunks + c) * q2r * 64u32;
    let mut k0 = 0u32;
    while k0 < q2r {
        #[unroll]
        for ld in 0u32..4u32 {
            let flat = tid * 4u32 + ld;
            smem_b[flat as usize] = b_hat[(bx_gbase + k0 * 64u32 + flat) as usize];
            smem_x[flat as usize] = x_hat[(bx_gbase + k0 * 64u32 + flat) as usize];
        }
        sync_cube();

        let k_col = k0 + ty;
        let t_j = k_col / (2u32 * rank_r);
        let mut dot_cb = F::new(0.0_f32);
        for n in 0u32..64u32 {
            dot_cb += smem_c[(tx * 64u32 + n) as usize] * smem_b[(ty * 64u32 + n) as usize];
        }

        let a_j = cum_a[(a_base + t_j) as usize];
        let masked_score = if t_i >= t_j {
            dot_cb * (a_i - a_j).exp()
        } else {
            F::new(0.0_f32)
        };
        smem_s[(tx * 16u32 + ty) as usize] = masked_score;
        sync_cube();

        for kk in 0u32..16u32 {
            let s_val = smem_s[(tx * 16u32 + kk) as usize];
            #[unroll]
            for pj in 0u32..4u32 {
                let p_col = ty * 4u32 + pj;
                y_acc[pj as usize] =
                    y_acc[pj as usize] + s_val * smem_x[(kk * 64u32 + p_col) as usize];
            }
        }
        sync_cube();
        k0 += 16u32;
    }

    let y_row_base = (bh * num_chunks + c) * qr * 64u32 + qr_row * 64u32;
    #[unroll]
    for pj in 0u32..4u32 {
        let p_col = ty * 4u32 + pj;
        y_out[(y_row_base + p_col) as usize] = y_acc[pj as usize];
    }
}

/// Backward Phase 1: Local incoming-state gradient per chunk: dH_local[c] = (exp(a) * C_rot)^T @ dY
#[cube(launch)]
pub fn mamba3_bwd_offdiag_cube_kernel<F: Float>(
    c_rot: &Tensor<F>,        // [BH, C, QR, 64]
    dy: &Tensor<F>,           // [BH, C, QR, 64]
    cum_a: &Tensor<F>,        // [BH, C, Q]
    dh_local: &mut Tensor<F>, // [BH, C, 64, 64]
    #[comptime] num_chunks: u32,
    #[comptime] q_len: u32,
    #[comptime] rank_r: u32,
    #[comptime] qr: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let c = CUBE_POS_Y;
    let tx = UNIT_POS_X; // 0..16 -> 4 rows of N=64
    let ty = UNIT_POS_Y; // 0..16 -> 4 cols of P=64

    let a_base = (bh * num_chunks + c) * q_len;
    let qr_base = (bh * num_chunks + c) * qr * 64u32;

    let mut acc = Array::<F>::new(16usize);
    #[unroll]
    for idx in 0usize..16usize {
        acc[idx] = F::new(0.0_f32);
    }

    for i in 0u32..qr {
        let t_i = i / rank_r;
        let exp_ai = cum_a[(a_base + t_i) as usize].exp();
        let row_off = qr_base + i * 64u32;
        #[unroll]
        for ni in 0u32..4u32 {
            let c_val = c_rot[(row_off + tx * 4u32 + ni) as usize] * exp_ai;
            #[unroll]
            for pj in 0u32..4u32 {
                let dy_val = dy[(row_off + ty * 4u32 + pj) as usize];
                let r_idx = (ni * 4u32 + pj) as usize;
                acc[r_idx] = acc[r_idx] + c_val * dy_val;
            }
        }
    }

    let dh_base = (bh * num_chunks + c) * 4096u32;
    #[unroll]
    for ni in 0u32..4u32 {
        let n_row = tx * 4u32 + ni;
        #[unroll]
        for pj in 0u32..4u32 {
            let p_col = ty * 4u32 + pj;
            dh_local[(dh_base + n_row * 64u32 + p_col) as usize] = acc[(ni * 4u32 + pj) as usize];
        }
    }
}

/// Backward Phase 2: Exact reverse inter-chunk adjoint scan + boundary <dH_c, H_c> dot product.
#[cube(launch)]
pub fn mamba3_bwd_interchunk_cube_kernel<F: Float>(
    dh_local: &Tensor<F>,         // [BH, C, 64, 64]
    chunk_decay: &Tensor<F>,      // [BH, C]
    prev_h: &Tensor<F>,           // [BH, C, 64, 64]
    h_final: &Tensor<F>,          // [BH, 64, 64]
    dh_final: &Tensor<F>,         // [BH, 64, 64]
    dh_c_out: &mut Tensor<F>,     // [BH, C, 64, 64]
    dh0_out: &mut Tensor<F>,      // [BH, 64, 64]
    end_boundary: &mut Tensor<F>, // [BH, C]
    #[comptime] num_chunks: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let tid = UNIT_POS_X; // 0..256

    let mut smem_red = Shared::new_slice(256usize);
    let mut dh_reg = Array::<F>::new(16usize);
    let h0_base = bh * 4096u32;

    #[unroll]
    for i in 0u32..16u32 {
        dh_reg[i as usize] = dh_final[(h0_base + tid * 16u32 + i) as usize];
    }

    for rev in 0u32..num_chunks {
        let c = num_chunks - 1u32 - rev;
        let c_base = (bh * num_chunks + c) * 4096u32;
        let mut partial_dot = F::new(0.0_f32);

        #[unroll]
        for i in 0u32..16u32 {
            let elem = tid * 16u32 + i;
            dh_c_out[(c_base + elem) as usize] = dh_reg[i as usize];
            let hc_val = if c == num_chunks - 1u32 {
                h_final[(h0_base + elem) as usize]
            } else {
                prev_h[((bh * num_chunks + c + 1u32) * 4096u32 + elem) as usize]
            };
            partial_dot += dh_reg[i as usize] * hc_val;
        }

        smem_red[tid as usize] = partial_dot;
        sync_cube();

        #[unroll]
        for shift in 0u32..8u32 {
            let s = 128u32 >> shift;
            if tid < s {
                smem_red[tid as usize] = smem_red[tid as usize] + smem_red[(tid + s) as usize];
            }
            sync_cube();
        }
        if tid == 0u32 {
            end_boundary[(bh * num_chunks + c) as usize] = smem_red[0usize];
        }
        sync_cube();

        let dec = chunk_decay[(bh * num_chunks + c) as usize];
        #[unroll]
        for i in 0u32..16u32 {
            let elem = tid * 16u32 + i;
            dh_reg[i as usize] = dec * dh_reg[i as usize] + dh_local[(c_base + elem) as usize];
        }
    }

    #[unroll]
    for i in 0u32..16u32 {
        dh0_out[(h0_base + tid * 16u32 + i) as usize] = dh_reg[i as usize];
    }
}

/// Backward Phase 3: Fused Adjoint Kernel computing dC_rot, dB_hat, dX_hat, and telescoping d(cum_a).
#[cube(launch)]
pub fn mamba3_ssd_backward_cube_kernel<F: Float>(
    c_rot: &Tensor<F>,          // [BH, C, QR, 64]
    b_hat: &Tensor<F>,          // [BH, C, Q2R, 64]
    x_hat: &Tensor<F>,          // [BH, C, Q2R, 64]
    dy: &Tensor<F>,             // [BH, C, QR, 64]
    cum_a: &Tensor<F>,          // [BH, C, Q]
    prev_h: &Tensor<F>,         // [BH, C, 64, 64]
    dh_c: &Tensor<F>,           // [BH, C, 64, 64]
    end_boundary: &Tensor<F>,   // [BH, C]
    dc_out: &mut Tensor<F>,     // [BH, C, QR, 64]
    db_out: &mut Tensor<F>,     // [BH, C, Q2R, 64]
    dx_out: &mut Tensor<F>,     // [BH, C, Q2R, 64]
    dcum_a_out: &mut Tensor<F>, // [BH, C, Q]
    #[comptime] num_chunks: u32,
    #[comptime] q_len: u32,
    #[comptime] rank_r: u32,
    #[comptime] qr: u32,
    #[comptime] q2r: u32,
    #[define(F)] _dtype: ElemType,
) {
    let bh = CUBE_POS_X;
    let c = CUBE_POS_Y;
    let lane = UNIT_POS_X; // 0..64

    let mut smem_dot_a = Shared::new_slice(64usize);
    let mut smem_dot_b = Shared::new_slice(64usize);
    let mut smem_da_q = Shared::new_slice(32usize);

    if lane < q_len {
        smem_da_q[lane as usize] = F::new(0.0_f32);
    }
    sync_cube();

    let a_base = (bh * num_chunks + c) * q_len;
    let a_end = cum_a[(a_base + q_len - 1u32) as usize];
    let h_base = (bh * num_chunks + c) * 4096u32;
    let qr_base = (bh * num_chunks + c) * qr * 64u32;
    let q2r_base = (bh * num_chunks + c) * q2r * 64u32;

    for i in 0u32..qr {
        let t_i = i / rank_r;
        let exp_ai = cum_a[(a_base + t_i) as usize].exp();
        let mut off_grad = F::new(0.0_f32);
        for p in 0u32..64u32 {
            let dy_ip = dy[(qr_base + i * 64u32 + p) as usize];
            let h_np = prev_h[(h_base + lane * 64u32 + p) as usize];
            off_grad += dy_ip * h_np;
        }
        dc_out[(qr_base + i * 64u32 + lane) as usize] = off_grad * exp_ai;
    }

    for j in 0u32..q2r {
        let t_j = j / (2u32 * rank_r);
        let a_j = cum_a[(a_base + t_j) as usize];
        let end_scale = (a_end - a_j).exp();

        let mut db_st = F::new(0.0_f32);
        let mut dx_st = F::new(0.0_f32);
        for k in 0u32..64u32 {
            db_st += x_hat[(q2r_base + j * 64u32 + k) as usize]
                * dh_c[(h_base + lane * 64u32 + k) as usize];
            dx_st += b_hat[(q2r_base + j * 64u32 + k) as usize]
                * dh_c[(h_base + k * 64u32 + lane) as usize];
        }
        db_out[(q2r_base + j * 64u32 + lane) as usize] = db_st * end_scale;
        dx_out[(q2r_base + j * 64u32 + lane) as usize] = dx_st * end_scale;
    }
    sync_cube();

    for i in 0u32..qr {
        let t_i = i / rank_r;
        let a_i = cum_a[(a_base + t_i) as usize];
        let c_in = c_rot[(qr_base + i * 64u32 + lane) as usize];
        let dy_ip = dy[(qr_base + i * 64u32 + lane) as usize];
        let mut dc_acc = dc_out[(qr_base + i * 64u32 + lane) as usize];

        let max_j = (t_i + 1u32) * 2u32 * rank_r;
        for j in 0u32..max_j {
            let t_j = j / (2u32 * rank_r);
            let l_ij = (a_i - cum_a[(a_base + t_j) as usize]).exp();

            let b_jn = b_hat[(q2r_base + j * 64u32 + lane) as usize];
            let x_jp = x_hat[(q2r_base + j * 64u32 + lane) as usize];

            smem_dot_a[lane as usize] = c_in * b_jn;
            smem_dot_b[lane as usize] = dy_ip * x_jp;
            sync_cube();

            #[unroll]
            for stride_shift in 0u32..6u32 {
                let s = 32u32 >> stride_shift;
                if lane < s {
                    smem_dot_a[lane as usize] =
                        smem_dot_a[lane as usize] + smem_dot_a[(lane + s) as usize];
                    smem_dot_b[lane as usize] =
                        smem_dot_b[lane as usize] + smem_dot_b[(lane + s) as usize];
                }
                sync_cube();
            }

            let m_ij = smem_dot_a[0usize] * l_ij;
            let dm_ij = smem_dot_b[0usize] * l_ij;

            dc_acc += dm_ij * b_jn;
            db_out[(q2r_base + j * 64u32 + lane) as usize] =
                db_out[(q2r_base + j * 64u32 + lane) as usize] + dm_ij * c_in;
            dx_out[(q2r_base + j * 64u32 + lane) as usize] =
                dx_out[(q2r_base + j * 64u32 + lane) as usize] + m_ij * dy_ip;
            sync_cube();
        }
        dc_out[(qr_base + i * 64u32 + lane) as usize] = dc_acc;
    }
    sync_cube();

    for q in 0u32..q_len {
        let mut local_da = F::new(0.0_f32);
        for r in 0u32..4u32 {
            let i = q * rank_r + r;
            local_da += dc_out[(qr_base + i * 64u32 + lane) as usize]
                * c_rot[(qr_base + i * 64u32 + lane) as usize];
        }
        for r2 in 0u32..8u32 {
            let j = q * (2u32 * rank_r) + r2;
            local_da -= db_out[(q2r_base + j * 64u32 + lane) as usize]
                * b_hat[(q2r_base + j * 64u32 + lane) as usize];
        }
        smem_dot_a[lane as usize] = local_da;
        sync_cube();

        #[unroll]
        for stride_shift in 0u32..6u32 {
            let s = 32u32 >> stride_shift;
            if lane < s {
                smem_dot_a[lane as usize] =
                    smem_dot_a[lane as usize] + smem_dot_a[(lane + s) as usize];
            }
            sync_cube();
        }
        if lane == 0u32 {
            let bound = if q == q_len - 1u32 {
                end_boundary[(bh * num_chunks + c) as usize]
            } else {
                F::new(0.0_f32)
            };
            smem_da_q[q as usize] = smem_dot_a[0usize] + bound;
        }
        sync_cube();
    }

    if lane < q_len {
        dcum_a_out[(a_base + lane) as usize] = smem_da_q[lane as usize];
    }
}

/// Host Launcher for Forward Mamba-3 Chunkwise SSD on CubeBackend.
pub fn launch_mamba3_ssd_forward(
    x_hat: CubeTensor,     // [BH, C, 256, 64]
    b_hat: CubeTensor,     // [BH, C, 256, 64]
    c_rot: CubeTensor,     // [BH, C, 128, 64]
    cum_a: CubeTensor,     // [BH, C, 32]
    end_decay: CubeTensor, // [BH, C, 256]
    chunk_dec: CubeTensor, // [BH, C]
    h0: CubeTensor,        // [BH, 64, 64]
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [bh, c, q2r, _p] = x_hat.shape().dims::<4>();
    let qr = c_rot.shape().dims::<4>()[2];
    let client = x_hat.client.clone();
    let device = x_hat.device.clone();
    let f_dtype = x_hat.dtype;
    let storage_ty = dtype_to_storage_type(f_dtype);

    let x_c = into_contiguous(x_hat);
    let b_c = into_contiguous(b_hat);
    let c_c = into_contiguous(c_rot);
    let ca_c = into_contiguous(cum_a);
    let ed_c = into_contiguous(end_decay);
    let cd_c = into_contiguous(chunk_dec);
    let h0_c = into_contiguous(h0);

    let s_chunks = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, 64, 64]),
        f_dtype,
    );
    let prev_h = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, 64, 64]),
        f_dtype,
    );
    let h_final = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, 64, 64]),
        f_dtype,
    );
    let y_out = empty_device_dtype(client.clone(), device, Shape::new([bh, c, qr, 64]), f_dtype);

    mamba3_chunk_state_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, c as u32, 1),
        CubeDim::new_2d(16, 16),
        b_c.clone().into_tensor_arg(),
        x_c.clone().into_tensor_arg(),
        ed_c.into_tensor_arg(),
        s_chunks.clone().into_tensor_arg(),
        c as u32,
        q2r as u32,
        storage_ty,
    );

    mamba3_interchunk_scan_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, 1, 1),
        CubeDim::new_2d(16, 16),
        s_chunks.into_tensor_arg(),
        cd_c.into_tensor_arg(),
        h0_c.into_tensor_arg(),
        prev_h.clone().into_tensor_arg(),
        h_final.clone().into_tensor_arg(),
        c as u32,
        storage_ty,
    );

    mamba3_ssd_output_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, c as u32, (qr / 16) as u32),
        CubeDim::new_2d(16, 16),
        c_c.into_tensor_arg(),
        b_c.into_tensor_arg(),
        x_c.into_tensor_arg(),
        ca_c.into_tensor_arg(),
        prev_h.clone().into_tensor_arg(),
        y_out.clone().into_tensor_arg(),
        c as u32,
        32u32,
        4u32,
        qr as u32,
        q2r as u32,
        storage_ty,
    );

    (y_out, h_final, prev_h)
}

/// Host Launcher for Full Reverse-Mode Mamba-3 SSD Adjoint Backward on CubeBackend.
#[allow(clippy::too_many_arguments)]
pub fn launch_mamba3_ssd_backward(
    x_hat: CubeTensor,
    b_hat: CubeTensor,
    c_rot: CubeTensor,
    cum_a: CubeTensor,
    chunk_dec: CubeTensor,
    prev_h: CubeTensor,
    h_final: CubeTensor,
    dy: CubeTensor,
    dh_final: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [bh, c, q2r, _p] = x_hat.shape().dims::<4>();
    let qr = c_rot.shape().dims::<4>()[2];
    let q_len = cum_a.shape().dims::<3>()[2];
    let client = x_hat.client.clone();
    let device = x_hat.device.clone();
    let f_dtype = x_hat.dtype;
    let storage_ty = dtype_to_storage_type(f_dtype);

    let x_c = into_contiguous(x_hat);
    let b_c = into_contiguous(b_hat);
    let c_c = into_contiguous(c_rot);
    let ca_c = into_contiguous(cum_a);
    let cd_c = into_contiguous(chunk_dec);
    let ph_c = into_contiguous(prev_h);
    let hf_c = into_contiguous(h_final);
    let dy_c = into_contiguous(dy);
    let dhf_c = into_contiguous(dh_final);

    let dh_local = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, 64, 64]),
        f_dtype,
    );
    let dh_c = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, 64, 64]),
        f_dtype,
    );
    let dh0 = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, 64, 64]),
        f_dtype,
    );
    let end_boundary = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c]),
        f_dtype,
    );

    let dc_out = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, qr, 64]),
        f_dtype,
    );
    let db_out = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, q2r, 64]),
        f_dtype,
    );
    let dx_out = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([bh, c, q2r, 64]),
        f_dtype,
    );
    let dcum_a_out =
        empty_device_dtype(client.clone(), device, Shape::new([bh, c, q_len]), f_dtype);

    mamba3_bwd_offdiag_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, c as u32, 1),
        CubeDim::new_2d(16, 16),
        c_c.clone().into_tensor_arg(),
        dy_c.clone().into_tensor_arg(),
        ca_c.clone().into_tensor_arg(),
        dh_local.clone().into_tensor_arg(),
        c as u32,
        q_len as u32,
        4u32,
        qr as u32,
        storage_ty,
    );

    mamba3_bwd_interchunk_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, 1, 1),
        CubeDim::new_1d(256),
        dh_local.into_tensor_arg(),
        cd_c.into_tensor_arg(),
        ph_c.clone().into_tensor_arg(),
        hf_c.into_tensor_arg(),
        dhf_c.into_tensor_arg(),
        dh_c.clone().into_tensor_arg(),
        dh0.clone().into_tensor_arg(),
        end_boundary.clone().into_tensor_arg(),
        c as u32,
        storage_ty,
    );

    mamba3_ssd_backward_cube_kernel::launch(
        &client,
        CubeCount::Static(bh as u32, c as u32, 1),
        CubeDim::new_1d(64),
        c_c.into_tensor_arg(),
        b_c.into_tensor_arg(),
        x_c.into_tensor_arg(),
        dy_c.into_tensor_arg(),
        ca_c.into_tensor_arg(),
        ph_c.into_tensor_arg(),
        dh_c.into_tensor_arg(),
        end_boundary.into_tensor_arg(),
        dc_out.clone().into_tensor_arg(),
        db_out.clone().into_tensor_arg(),
        dx_out.clone().into_tensor_arg(),
        dcum_a_out.clone().into_tensor_arg(),
        c as u32,
        q_len as u32,
        4u32,
        qr as u32,
        q2r as u32,
        storage_ty,
    );

    (dx_out, db_out, dc_out, dcum_a_out, dh0)
}
