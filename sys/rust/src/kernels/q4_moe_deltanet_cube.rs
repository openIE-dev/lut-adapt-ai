//! Phase 27 (Claim LA-18): Cross-Platform `#[cube]` Fused `INT4` Block-64 Sparse-MoE &
//! Multi-Step `GatedDeltaNet` Linear-Attention Kernels for Burn 0.22.1 / CubeCL.
//!
//! A single zero-Python Rust kernel source compiles natively via CubeCL JIT to:
//!   - Apple Metal (`MSL`)
//!   - Vulkan / WebGPU (`SPIR-V`)
//!   - NVIDIA CUDA (`PTX`)
//!   - AMD ROCm (`HIP`)
//!   - Multi-core SIMD (`LLVM CPU`)
//!
//! Zero `sync_cube()` barriers and zero shared-memory emulation traps: each GPU/SIMD lane owns
//! one independent `(p, row)` output and unpacks `8 x INT4` nibbles per `u32` word in registers.

use burn_backend::{Shape, TensorMetadata, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    cubecl::{self, CubeCount, CubeDim, ir::ElemType, prelude::*},
    kernel::into_contiguous,
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// Fused INT4 Block-64 Indexed-Expert Gate + Up + SiLU Kernel (Zero-Barrier).
/// Grid: `(ceil(I_dim / 64), P, 1)` | CubeDim: `(64, 1, 1)`.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
pub fn q4_moe_gate_up_silu_cube_kernel<F: Float, I: Int>(
    packed_u32: &Tensor<I>,     // [K_uniq, m_tot, 8] (8 x u32 = 32B = 64 nibbles per block)
    mins_all: &Tensor<F>,       // [K_uniq, m_tot]
    scales_all: &Tensor<F>,     // [K_uniq, m_tot]
    flat_h: &Tensor<F>,         // [T, D]
    pair_tok: &Tensor<I>,       // [P]
    pair_slot: &Tensor<I>,      // [P]
    interm_out: &mut Tensor<F>, // [P, I_dim]
    #[comptime] d_model: u32,
    #[comptime] i_dim: u32,
    #[comptime] m_tot: u32,
    #[comptime] p_pairs: u32,
    #[define(F)] _f_dtype: ElemType,
    #[define(I)] _i_dtype: ElemType,
) {
    let i_row = CUBE_POS_X * 64u32 + UNIT_POS_X;
    let p = CUBE_POS_Y;

    if i_row < i_dim && p < p_pairs {
        let n_blks_d = d_model / 64u32;
        let tok = u32::cast_from(pair_tok[p as usize]);
        let slot = u32::cast_from(pair_slot[p as usize]);

        let x_base = tok * d_model;
        let slot_m_base = slot * m_tot;
        let gate_m_base = slot_m_base + i_row * n_blks_d;
        let up_m_base = slot_m_base + (i_dim + i_row) * n_blks_d;

        let mut g_acc = F::new(0.0_f32);
        let mut u_acc = F::new(0.0_f32);

        let mut blk = 0u32;
        while blk < n_blks_d {
            let gm = gate_m_base + blk;
            let um = up_m_base + blk;
            let mut sx = F::new(0.0_f32);
            let mut g_sq = F::new(0.0_f32);
            let mut u_sq = F::new(0.0_f32);

            let x_blk = x_base + blk * 64u32;
            let mut w_idx = 0u32;
            while w_idx < 8u32 {
                let gw = u32::cast_from(packed_u32[(gm * 8u32 + w_idx) as usize]);
                let uw = u32::cast_from(packed_u32[(um * 8u32 + w_idx) as usize]);
                let x_w = x_blk + w_idx * 8u32;

                #[unroll]
                for nib in 0u32..8u32 {
                    let xv = flat_h[(x_w + nib) as usize];
                    sx += xv;
                    let shift = nib * 4u32;
                    let gq = F::cast_from((gw >> shift) & 0x0Fu32);
                    let uq = F::cast_from((uw >> shift) & 0x0Fu32);
                    g_sq += gq * xv;
                    u_sq += uq * xv;
                }
                w_idx += 1u32;
            }

            g_acc += mins_all[gm as usize] * sx + scales_all[gm as usize] * g_sq;
            u_acc += mins_all[um as usize] * sx + scales_all[um as usize] * u_sq;
            blk += 1u32;
        }

        let silu_g = g_acc / (F::new(1.0_f32) + F::exp(F::new(0.0_f32) - g_acc));
        interm_out[(p * i_dim + i_row) as usize] = silu_g * u_acc;
    }
}

/// Fused INT4 Block-64 Indexed-Expert Down-Projection + Routing Weight Kernel (Zero-Barrier).
/// Grid: `(ceil(D / 64), P, 1)` | CubeDim: `(64, 1, 1)`.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
pub fn q4_moe_down_weighted_cube_kernel<F: Float, I: Int>(
    packed_u32: &Tensor<I>,   // [K_uniq, m_tot, 8]
    mins_all: &Tensor<F>,     // [K_uniq, m_tot]
    scales_all: &Tensor<F>,   // [K_uniq, m_tot]
    interm_in: &Tensor<F>,    // [P, I_dim]
    pair_slot: &Tensor<I>,    // [P]
    pair_rw: &Tensor<F>,      // [P]
    pair_out: &mut Tensor<F>, // [P, D]
    #[comptime] d_model: u32,
    #[comptime] i_dim: u32,
    #[comptime] m_gu: u32,
    #[comptime] m_tot: u32,
    #[comptime] p_pairs: u32,
    #[define(F)] _f_dtype: ElemType,
    #[define(I)] _i_dtype: ElemType,
) {
    let d_row = CUBE_POS_X * 64u32 + UNIT_POS_X;
    let p = CUBE_POS_Y;

    if d_row < d_model && p < p_pairs {
        let n_blks_i = i_dim / 64u32;
        let slot = u32::cast_from(pair_slot[p as usize]);
        let x_base = p * i_dim;
        let dn_m_base = slot * m_tot + m_gu + d_row * n_blks_i;

        let mut acc = F::new(0.0_f32);
        let mut blk = 0u32;
        while blk < n_blks_i {
            let dm = dn_m_base + blk;
            let mut sx = F::new(0.0_f32);
            let mut sq = F::new(0.0_f32);
            let x_blk = x_base + blk * 64u32;

            let mut w_idx = 0u32;
            while w_idx < 8u32 {
                let dw = u32::cast_from(packed_u32[(dm * 8u32 + w_idx) as usize]);
                let x_w = x_blk + w_idx * 8u32;
                #[unroll]
                for nib in 0u32..8u32 {
                    let xv = interm_in[(x_w + nib) as usize];
                    sx += xv;
                    let dq = F::cast_from((dw >> (nib * 4u32)) & 0x0Fu32);
                    sq += dq * xv;
                }
                w_idx += 1u32;
            }

            acc += mins_all[dm as usize] * sx + scales_all[dm as usize] * sq;
            blk += 1u32;
        }

        pair_out[(p * d_model + d_row) as usize] = acc * pair_rw[p as usize];
    }
}

/// Fused Multi-Step Gated-DeltaNet Recurrence (`K_dim == 128`) in GPU Registers.
/// Grid: `(ceil(V_dim / 64), H_v, B)` | CubeDim: `(64, 1, 1)`.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
pub fn gated_delta_recurrent_k128_cube_kernel<F: Float>(
    q_all: &Tensor<F>,         // [B, H_k, T, 128]
    k_all: &Tensor<F>,         // [B, H_k, T, 128]
    v_all: &Tensor<F>,         // [B, H_v, T, V_dim]
    g_all: &Tensor<F>,         // [B, H_v, T]
    beta_all: &Tensor<F>,      // [B, H_v, T]
    state_in: &Tensor<F>,      // [B, H_v, 128, V_dim]
    out_all: &mut Tensor<F>,   // [B, H_v, T, V_dim]
    state_out: &mut Tensor<F>, // [B, H_v, 128, V_dim]
    #[comptime] t_len: u32,
    #[comptime] v_dim: u32,
    #[comptime] h_k: u32,
    #[comptime] h_v: u32,
    #[define(F)] _f_dtype: ElemType,
) {
    let v = CUBE_POS_X * 64u32 + UNIT_POS_X;
    let hv = CUBE_POS_Y;
    let b = CUBE_POS_Z;

    if v < v_dim && hv < h_v {
        let gqa_ratio = h_v / h_k;
        let hk = hv / gqa_ratio;

        let bhv = b * h_v + hv;
        let bhk = b * h_k + hk;
        let st_base = bhv * 128u32 * v_dim + v;

        let mut s_reg = Array::<F>::new(128usize);
        let mut kd = 0u32;
        while kd < 128u32 {
            s_reg[kd as usize] = state_in[(st_base + kd * v_dim) as usize];
            kd += 1u32;
        }

        let mut t = 0u32;
        while t < t_len {
            let dec = F::exp(g_all[(bhv * t_len + t) as usize]);
            let bet = beta_all[(bhv * t_len + t) as usize];
            let vt = v_all[((bhv * t_len + t) * v_dim + v) as usize];
            let qk_base = (bhk * t_len + t) * 128u32;

            let mut kv_mem = F::new(0.0_f32);
            let mut k1 = 0u32;
            while k1 < 128u32 {
                let s_dec = s_reg[k1 as usize] * dec;
                s_reg[k1 as usize] = s_dec;
                kv_mem += s_dec * k_all[(qk_base + k1) as usize];
                k1 += 1u32;
            }

            let delta = (vt - kv_mem) * bet;
            let mut out_v = F::new(0.0_f32);
            let mut k2 = 0u32;
            while k2 < 128u32 {
                let s_new = s_reg[k2 as usize] + delta * k_all[(qk_base + k2) as usize];
                s_reg[k2 as usize] = s_new;
                out_v += s_new * q_all[(qk_base + k2) as usize];
                k2 += 1u32;
            }

            out_all[((bhv * t_len + t) * v_dim + v) as usize] = out_v;
            t += 1u32;
        }

        let mut kw = 0u32;
        while kw < 128u32 {
            state_out[(st_base + kw * v_dim) as usize] = s_reg[kw as usize];
            kw += 1u32;
        }
    }
}

/// Host Launcher: Fused INT4 Block-64 Sparse-MoE Gate+Up+SiLU + Weighted Down-Projection (`P` pairs).
#[allow(clippy::too_many_arguments)]
pub fn launch_q4_moe_ffn_pairs(
    packed_u32: CubeTensor, // [K_uniq, m_tot, 8] Int
    mins_all: CubeTensor,   // [K_uniq, m_tot] Float
    scales_all: CubeTensor, // [K_uniq, m_tot] Float
    flat_h: CubeTensor,     // [T, D] Float
    pair_tok: CubeTensor,   // [P] Int
    pair_slot: CubeTensor,  // [P] Int
    pair_rw: CubeTensor,    // [P] Float
    i_dim: usize,
) -> CubeTensor {
    let [_t, d_model] = flat_h.shape().dims::<2>();
    let [_k_uniq, m_tot, _eight] = packed_u32.shape().dims::<3>();
    let [p_pairs] = pair_tok.shape().dims::<1>();
    let m_gu = 2 * i_dim * (d_model / 64);

    let client = flat_h.client.clone();
    let device = flat_h.device.clone();
    let f_dtype = flat_h.dtype;
    let i_dtype = packed_u32.dtype;
    let f_storage = dtype_to_storage_type(f_dtype);
    let i_storage = dtype_to_storage_type(i_dtype);

    let packed_c = into_contiguous(packed_u32);
    let mins_c = into_contiguous(mins_all);
    let scales_c = into_contiguous(scales_all);
    let flat_h_c = into_contiguous(flat_h);
    let tok_c = into_contiguous(pair_tok);
    let slot_c = into_contiguous(pair_slot);
    let rw_c = into_contiguous(pair_rw);

    let interm = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([p_pairs, i_dim]),
        f_dtype,
    );

    q4_moe_gate_up_silu_cube_kernel::launch(
        &client,
        CubeCount::Static(i_dim.div_ceil(64) as u32, p_pairs as u32, 1),
        CubeDim::new_1d(64),
        packed_c.clone().into_tensor_arg(),
        mins_c.clone().into_tensor_arg(),
        scales_c.clone().into_tensor_arg(),
        flat_h_c.into_tensor_arg(),
        tok_c.into_tensor_arg(),
        slot_c.clone().into_tensor_arg(),
        interm.clone().into_tensor_arg(),
        d_model as u32,
        i_dim as u32,
        m_tot as u32,
        p_pairs as u32,
        f_storage,
        i_storage,
    );

    let pair_out = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([p_pairs, d_model]),
        f_dtype,
    );

    q4_moe_down_weighted_cube_kernel::launch(
        &client,
        CubeCount::Static(d_model.div_ceil(64) as u32, p_pairs as u32, 1),
        CubeDim::new_1d(64),
        packed_c.into_tensor_arg(),
        mins_c.into_tensor_arg(),
        scales_c.into_tensor_arg(),
        interm.into_tensor_arg(),
        slot_c.into_tensor_arg(),
        rw_c.into_tensor_arg(),
        pair_out.clone().into_tensor_arg(),
        d_model as u32,
        i_dim as u32,
        m_gu as u32,
        m_tot as u32,
        p_pairs as u32,
        f_storage,
        i_storage,
    );

    pair_out
}

/// Host Launcher: Fused Multi-Step Gated-DeltaNet Recurrence (`K_dim == 128`).
pub fn launch_gated_delta_recurrent_k128(
    q_all: CubeTensor,    // [B, H_k, T, 128]
    k_all: CubeTensor,    // [B, H_k, T, 128]
    v_all: CubeTensor,    // [B, H_v, T, V_dim]
    g_all: CubeTensor,    // [B, H_v, T]
    beta_all: CubeTensor, // [B, H_v, T]
    state_in: CubeTensor, // [B, H_v, 128, V_dim]
) -> (CubeTensor, CubeTensor) {
    let [b, h_k, t_len, _k_dim] = q_all.shape().dims::<4>();
    let [_b2, h_v, _t2, v_dim] = v_all.shape().dims::<4>();

    let client = q_all.client.clone();
    let device = q_all.device.clone();
    let f_dtype = q_all.dtype;
    let f_storage = dtype_to_storage_type(f_dtype);

    let q_c = into_contiguous(q_all);
    let k_c = into_contiguous(k_all);
    let v_c = into_contiguous(v_all);
    let g_c = into_contiguous(g_all);
    let beta_c = into_contiguous(beta_all);
    let st_c = into_contiguous(state_in);

    let out_all = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([b, h_v, t_len, v_dim]),
        f_dtype,
    );
    let state_out = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([b, h_v, 128, v_dim]),
        f_dtype,
    );

    gated_delta_recurrent_k128_cube_kernel::launch(
        &client,
        CubeCount::Static(v_dim.div_ceil(64) as u32, h_v as u32, b as u32),
        CubeDim::new_1d(64),
        q_c.into_tensor_arg(),
        k_c.into_tensor_arg(),
        v_c.into_tensor_arg(),
        g_c.into_tensor_arg(),
        beta_c.into_tensor_arg(),
        st_c.into_tensor_arg(),
        out_all.clone().into_tensor_arg(),
        state_out.clone().into_tensor_arg(),
        t_len as u32,
        v_dim as u32,
        h_k as u32,
        h_v as u32,
        f_storage,
    );

    (out_all, state_out)
}
