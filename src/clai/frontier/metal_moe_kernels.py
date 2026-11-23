"""Phase 26: Native Apple Metal (`torch.mps.compile_shader`) Fused Compute Kernels for:
  1. `INT4` Block-64 Indexed Sparse-MoE Expert Execution (`q4_moe_gate_up_silu` + `q4_moe_down_weighted`):
     - Reads `4.50 bpw` (`32B` packed nibbles + `2B` `fp16` min + `2B` `fp16` scale per 64 weights) directly
       into Metal SIMDgroup registers (`simd_sum`), eliminating all intermediate `FP16/FP32` weight allocations
       and collapsing `2 * |U_l|` Python loop launches into **2 fused Metal GPU dispatches per layer**.
  2. Fused Multi-Step `GatedDeltaNet` Linear-Attention Recurrence (`gated_delta_recurrent_f32`):
     - Exploits exact column independence across `v in 0..V_dim-1` of recurrent state `S[b, hv, :, v]` (`K_dim <= 256`),
       keeping `S[:, v]` inside GPU registers across all `T` prefill or burst-verification tokens in **1 Metal launch**
       instead of `11 * T` Python-loop tensor allocations.
  3. Parallel Queue-Depth-8 NVMe `pread` + Online `BenesLUT6` Router-Lookahead Prefetcher.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_METAL_MOE_SHADER_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Kernel 1: Fused INT4 Block-64 Indexed-Expert Gate+Up + SiLU
//   Computes `interm[p, i_row] = silu(dot(W_gate[slot, i_row], x[tok])) * dot(W_up[slot, i_row], x[tok])`
//   Each SIMDgroup (32 threads) computes BOTH `gate` and `up` for one `(p, i_row)` pair,
//   reusing `x[tok]` in registers across both projections!
// ============================================================================
kernel void q4_moe_gate_up_silu(
    device const uchar* packed_all   [[buffer(0)]],  // [K_uniq, m_tot, 32]
    device const half*  mins_all     [[buffer(1)]],  // [K_uniq, m_tot]
    device const half*  scales_all   [[buffer(2)]],  // [K_uniq, m_tot]
    device const float* flat_h       [[buffer(3)]],  // [T, D]
    device const int*   pair_tok     [[buffer(4)]],  // [P]
    device const int*   pair_slot    [[buffer(5)]],  // [P]
    device float*       interm_out   [[buffer(6)]],  // [P, I_dim]
    constant uint&      D            [[buffer(7)]],
    constant uint&      I_dim        [[buffer(8)]],
    constant uint&      m_tot        [[buffer(9)]],
    constant uint&      P            [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  sglid [[thread_index_in_simdgroup]]
) {
    // 8 SIMDgroups per threadgroup (group_size = 256)
    uint i_row = tgpig.x * 8u + sgid;
    uint p     = tgpig.y;
    if (i_row >= I_dim || p >= P) return;

    uint n_blks_d = D >> 6;  // D / 64
    uint tok  = uint(pair_tok[p]);
    uint slot = uint(pair_slot[p]);

    device const float* x_ptr = flat_h + tok * D;
    uint slot_m_base = slot * m_tot;
    uint gate_m_base = slot_m_base + i_row * n_blks_d;
    uint up_m_base   = slot_m_base + (I_dim + i_row) * n_blks_d;

    float g_acc = 0.0f;
    float u_acc = 0.0f;

    for (uint blk = 0; blk < n_blks_d; ++blk) {
        uint x_idx = (blk << 6) + (sglid << 1);
        float x0 = x_ptr[x_idx];
        float x1 = x_ptr[x_idx + 1u];
        float sx = simd_sum(x0 + x1);

        uint gm = gate_m_base + blk;
        uchar gb = packed_all[(gm << 5) + sglid];
        float g_sqx = simd_sum(float(gb & 0x0Fu) * x0 + float(gb >> 4) * x1);
        g_acc += float(mins_all[gm]) * sx + float(scales_all[gm]) * g_sqx;

        uint um = up_m_base + blk;
        uchar ub = packed_all[(um << 5) + sglid];
        float u_sqx = simd_sum(float(ub & 0x0Fu) * x0 + float(ub >> 4) * x1);
        u_acc += float(mins_all[um]) * sx + float(scales_all[um]) * u_sqx;
    }

    if (sglid == 0u) {
        float silu_g = g_acc / (1.0f + exp(-g_acc));
        interm_out[p * I_dim + i_row] = silu_g * u_acc;
    }
}

// ============================================================================
// Kernel 2: Fused INT4 Block-64 Indexed-Expert Down-Projection + Routing Weight
//   Computes `pair_out[p, d_row] = rw[p] * dot(W_down[slot, d_row], interm[p])`
// ============================================================================
kernel void q4_moe_down_weighted(
    device const uchar* packed_all   [[buffer(0)]],  // [K_uniq, m_tot, 32]
    device const half*  mins_all     [[buffer(1)]],  // [K_uniq, m_tot]
    device const half*  scales_all   [[buffer(2)]],  // [K_uniq, m_tot]
    device const float* interm_in    [[buffer(3)]],  // [P, I_dim]
    device const int*   pair_slot    [[buffer(4)]],  // [P]
    device const float* pair_rw      [[buffer(5)]],  // [P]
    device float*       pair_out     [[buffer(6)]],  // [P, D]
    constant uint&      D            [[buffer(7)]],
    constant uint&      I_dim        [[buffer(8)]],
    constant uint&      m_gu         [[buffer(9)]],
    constant uint&      m_tot        [[buffer(10)]],
    constant uint&      P            [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  sglid [[thread_index_in_simdgroup]]
) {
    uint d_row = tgpig.x * 8u + sgid;
    uint p     = tgpig.y;
    if (d_row >= D || p >= P) return;

    uint n_blks_i = I_dim >> 6;  // I_dim / 64
    uint slot = uint(pair_slot[p]);
    device const float* x_ptr = interm_in + p * I_dim;
    uint dn_m_base = slot * m_tot + m_gu + d_row * n_blks_i;

    float acc = 0.0f;
    for (uint blk = 0; blk < n_blks_i; ++blk) {
        uint x_idx = (blk << 6) + (sglid << 1);
        float x0 = x_ptr[x_idx];
        float x1 = x_ptr[x_idx + 1u];
        float sx = simd_sum(x0 + x1);

        uint dm = dn_m_base + blk;
        uchar db = packed_all[(dm << 5) + sglid];
        float sqx = simd_sum(float(db & 0x0Fu) * x0 + float(db >> 4) * x1);
        acc += float(mins_all[dm]) * sx + float(scales_all[dm]) * sqx;
    }

    if (sglid == 0u) {
        pair_out[p * D + d_row] = acc * pair_rw[p];
    }
}

// ============================================================================
// Kernel 3: Fused Multi-Step Gated-DeltaNet Linear-Attention Recurrence (K_dim == 128)
//   Exploits exact independence of state columns `v in 0..V_dim-1`:
//   One thread per `(b, hv, v)` holds `s[0..127]` (`32 x float4`) in registers across all `T` steps!
// ============================================================================
kernel void gated_delta_recurrent_k128(
    device const float4* q_all    [[buffer(0)]],  // [B, H_k, T, 32] (float4 x 32 = 128)
    device const float4* k_all    [[buffer(1)]],  // [B, H_k, T, 32]
    device const float*  v_all    [[buffer(2)]],  // [B, H_v, T, V_dim]
    device const float*  g_all    [[buffer(3)]],  // [B, H_v, T]
    device const float*  beta_all [[buffer(4)]],  // [B, H_v, T]
    device float*        state_io [[buffer(5)]],  // [B, H_v, 128, V_dim] (read initial, write final)
    device float*        out_all  [[buffer(6)]],  // [B, H_v, T, V_dim]
    constant uint&       T        [[buffer(7)]],
    constant uint&       V_dim    [[buffer(8)]],
    constant uint&       H_k      [[buffer(9)]],
    constant uint&       H_v      [[buffer(10)]],
    constant uint&       write_st [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint v  = gid.x;
    uint hv = gid.y;
    uint b  = gid.z;
    if (v >= V_dim || hv >= H_v) return;

    uint gqa_ratio = H_v / H_k;
    uint hk = hv / gqa_ratio;

    uint bhv = b * H_v + hv;
    uint bhk = b * H_k + hk;
    device float* st_col = state_io + (bhv * 128u * V_dim) + v;

    float4 s[32];
    for (uint j = 0; j < 32u; ++j) {
        uint k0 = j << 2;
        s[j] = float4(
            st_col[(k0 + 0u) * V_dim],
            st_col[(k0 + 1u) * V_dim],
            st_col[(k0 + 2u) * V_dim],
            st_col[(k0 + 3u) * V_dim]
        );
    }

    for (uint t = 0; t < T; ++t) {
        float dec = exp(g_all[bhv * T + t]);
        float bet = beta_all[bhv * T + t];
        float vt  = v_all[(bhv * T + t) * V_dim + v];
        device const float4* kt = k_all + (bhk * T + t) * 32u;
        device const float4* qt = q_all + (bhk * T + t) * 32u;

        float kv_mem = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            s[j] *= dec;
            kv_mem += dot(s[j], kt[j]);
        }
        float delta = (vt - kv_mem) * bet;
        float out_v = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            s[j] = fma(float4(delta), kt[j], s[j]);
            out_v += dot(s[j], qt[j]);
        }
        out_all[(bhv * T + t) * V_dim + v] = out_v;
    }

    if (write_st != 0u) {
        for (uint j = 0; j < 32u; ++j) {
            uint k0 = j << 2;
            st_col[(k0 + 0u) * V_dim] = s[j].x;
            st_col[(k0 + 1u) * V_dim] = s[j].y;
            st_col[(k0 + 2u) * V_dim] = s[j].z;
            st_col[(k0 + 3u) * V_dim] = s[j].w;
        }
    }
}
"""

_COMPILED_LIB: Any = None
_DELTANET_PATCH_INSTALLED = False
_ORIG_QWEN_MOE_RULES: tuple[Any, Any] | None = None
_ORIG_QWEN_DENSE_RULES: tuple[Any, Any] | None = None


def get_metal_moe_library() -> Any:
    """Compiles (once per process) and returns the Metal compute shader library on MPS."""
    global _COMPILED_LIB
    if _COMPILED_LIB is None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _COMPILED_LIB = torch.mps.compile_shader(_METAL_MOE_SHADER_SRC)
    return _COMPILED_LIB


def fused_q4_moe_ffn_metal(
    flat_h: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    uniq_experts: list[int],
    packed_t: torch.Tensor,
    mins_t: torch.Tensor,
    scales_t: torch.Tensor,
    d_dim: int,
    i_dim: int,
    m_gu: int,
    m_tot: int,
) -> torch.Tensor:
    """
    Executes all active Sparse-MoE experts on Apple Metal directly from `4.50 bpw` `INT4` buffers
    (`packed_t: uint8 [K_uniq, m_tot, 32]`, `mins_t: fp16 [K_uniq, m_tot]`, `scales_t: fp16 [K_uniq, m_tot]`)
    without allocating dequantized `FP16/FP32` expert weight matrices!
    """
    lib = get_metal_moe_library()
    orig_dtype = flat_h.dtype
    dev = flat_h.device
    T = flat_h.size(0)
    k_per_tok = selected_experts.size(1)
    P = T * k_per_tok

    # Map global expert IDs `0..E-1` -> compact slot `0..K_uniq-1` on GPU
    max_e = max(uniq_experts) + 1 if uniq_experts else 1
    e2slot = torch.zeros(max_e, dtype=torch.int32, device=dev)
    e2slot[torch.tensor(uniq_experts, dtype=torch.long, device=dev)] = torch.arange(
        len(uniq_experts), dtype=torch.int32, device=dev
    )

    pair_tok = torch.arange(T, dtype=torch.int32, device=dev).unsqueeze(1).expand(T, k_per_tok).reshape(-1).contiguous()
    pair_slot = e2slot[selected_experts.reshape(-1).long()].contiguous()
    pair_rw = routing_weights.reshape(-1).to(torch.float32).contiguous()

    x_f32 = flat_h.to(torch.float32).contiguous()
    interm = torch.empty((P, i_dim), dtype=torch.float32, device=dev)
    pair_out = torch.empty((P, d_dim), dtype=torch.float32, device=dev)

    tg_gu_x = (i_dim + 7) // 8
    lib.q4_moe_gate_up_silu(
        packed_t,
        mins_t,
        scales_t,
        x_f32,
        pair_tok,
        pair_slot,
        interm,
        d_dim,
        i_dim,
        m_tot,
        P,
        threads=(tg_gu_x * 256, P, 1),
        group_size=(256, 1, 1),
    )

    tg_dn_x = (d_dim + 7) // 8
    lib.q4_moe_down_weighted(
        packed_t,
        mins_t,
        scales_t,
        interm,
        pair_slot,
        pair_rw,
        pair_out,
        d_dim,
        i_dim,
        m_gu,
        m_tot,
        P,
        threads=(tg_dn_x * 256, P, 1),
        group_size=(256, 1, 1),
    )

    if T == 1:
        return pair_out.sum(dim=0, keepdim=True).to(orig_dtype)
    ffn_out = torch.zeros((T, d_dim), dtype=torch.float32, device=dev)
    ffn_out.index_add_(0, pair_tok.long(), pair_out)
    return ffn_out.to(orig_dtype)


def fused_q4_moe_ffn_metal_direct(
    flat_h: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    packed_all: torch.Tensor,
    mins_all: torch.Tensor,
    scales_all: torch.Tensor,
    d_dim: int,
    i_dim: int,
    m_gu: int,
    m_tot: int,
) -> torch.Tensor:
    """
    Zero-copy Unified-Memory variant of `fused_q4_moe_ffn_metal` where `packed_all [E, m_tot, 32]`
    (`4.50 bpw` `INT4`) is already resident on `mps`, indexing `selected_experts` (`0..E-1`) directly
    in GPU registers with zero per-token host-to-device memory copies!
    """
    lib = get_metal_moe_library()
    orig_dtype = flat_h.dtype
    dev = flat_h.device
    T = flat_h.size(0)
    k_per_tok = selected_experts.size(1)
    P = T * k_per_tok

    pair_tok = torch.arange(T, dtype=torch.int32, device=dev).unsqueeze(1).expand(T, k_per_tok).reshape(-1).contiguous()
    pair_slot = selected_experts.reshape(-1).to(torch.int32).contiguous()
    pair_rw = routing_weights.reshape(-1).to(torch.float32).contiguous()

    x_f32 = flat_h.to(torch.float32).contiguous()
    interm = torch.empty((P, i_dim), dtype=torch.float32, device=dev)
    pair_out = torch.empty((P, d_dim), dtype=torch.float32, device=dev)

    tg_gu_x = (i_dim + 7) // 8
    lib.q4_moe_gate_up_silu(
        packed_all,
        mins_all,
        scales_all,
        x_f32,
        pair_tok,
        pair_slot,
        interm,
        d_dim,
        i_dim,
        m_tot,
        P,
        threads=(tg_gu_x * 256, P, 1),
        group_size=(256, 1, 1),
    )

    tg_dn_x = (d_dim + 7) // 8
    lib.q4_moe_down_weighted(
        packed_all,
        mins_all,
        scales_all,
        interm,
        pair_slot,
        pair_rw,
        pair_out,
        d_dim,
        i_dim,
        m_gu,
        m_tot,
        P,
        threads=(tg_dn_x * 256, P, 1),
        group_size=(256, 1, 1),
    )

    if T == 1:
        return pair_out.sum(dim=0, keepdim=True).to(orig_dtype)
    ffn_out = torch.zeros((T, d_dim), dtype=torch.float32, device=dev)
    ffn_out.index_add_(0, pair_tok.long(), pair_out)
    return ffn_out.to(orig_dtype)


def fused_gated_delta_rule_metal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Drop-in Apple Metal (`MPS`) replacement for `torch_recurrent_gated_delta_rule` and
    `torch_chunk_gated_delta_rule` when `k_head_dim == 128`. Falls back to exact PyTorch on CPU/non-128.
    """
    B, T, H_k, K_dim = key.shape
    H_v, V_dim = value.shape[-2], value.shape[-1]
    if query.device.type != "mps" or K_dim != 128 or get_metal_moe_library() is None:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

        orig_dt = query.dtype
        if H_k != H_v:
            query = query.repeat_interleave(H_v // H_k, dim=2)
            key = key.repeat_interleave(H_v // H_k, dim=2)
        q, k, v, b_t, d_t = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)]
        if use_qk_l2norm_in_kernel:
            q = l2norm(q, dim=-1, eps=1e-6)
            k = l2norm(k, dim=-1, eps=1e-6)
        q = q * (K_dim**-0.5)
        st = (
            torch.zeros((B, H_v, K_dim, V_dim), dtype=torch.float32, device=query.device)
            if initial_state is None
            else initial_state.to(torch.float32).clone()
        )
        out = torch.zeros_like(v)
        for i in range(T):
            st = st * d_t[:, :, i].exp()[..., None, None]
            kv_m = (st * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
            dl = (v[:, :, i] - kv_m) * b_t[:, :, i].unsqueeze(-1)
            st = st + k[:, :, i].unsqueeze(-1) * dl.unsqueeze(-2)
            out[:, :, i] = (st * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
        return out.transpose(1, 2).contiguous().to(orig_dt), (st if output_final_state else None)

    orig_dtype = query.dtype
    dev = query.device
    q_f = query.transpose(1, 2).to(torch.float32).contiguous()  # [B, H_k, T, 128]
    k_f = key.transpose(1, 2).to(torch.float32).contiguous()    # [B, H_k, T, 128]
    if use_qk_l2norm_in_kernel:
        q_f = q_f * torch.rsqrt(q_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k_f = k_f * torch.rsqrt(k_f.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    q_f = (q_f * (128.0**-0.5)).contiguous()
    k_f = k_f.contiguous()

    v_f = value.transpose(1, 2).to(torch.float32).contiguous()  # [B, H_v, T, V_dim]
    g_f = g.transpose(1, 2).to(torch.float32).contiguous()      # [B, H_v, T]
    b_f = beta.transpose(1, 2).to(torch.float32).contiguous()   # [B, H_v, T]

    if initial_state is None:
        st_f = torch.zeros((B, H_v, 128, V_dim), dtype=torch.float32, device=dev)
    else:
        st_f = initial_state.to(torch.float32).contiguous().clone()

    out_f = torch.empty((B, H_v, T, V_dim), dtype=torch.float32, device=dev)
    lib = get_metal_moe_library()
    grp_x = min(128, V_dim)
    nth_x = ((V_dim + grp_x - 1) // grp_x) * grp_x
    lib.gated_delta_recurrent_k128(
        q_f,
        k_f,
        v_f,
        g_f,
        b_f,
        st_f,
        out_f,
        T,
        V_dim,
        H_k,
        H_v,
        1 if output_final_state else 0,
        threads=(nth_x, H_v, B),
        group_size=(grp_x, 1, 1),
    )
    core_out = out_f.transpose(1, 2).contiguous().to(orig_dtype)
    return core_out, (st_f if output_final_state else None)


def set_metal_gated_deltanet_enabled(enabled: bool) -> bool:
    """Enables or restores `fused_gated_delta_rule_metal` on HuggingFace `qwen3_5_moe` and `qwen3_5` modules."""
    global _DELTANET_PATCH_INSTALLED, _ORIG_QWEN_MOE_RULES, _ORIG_QWEN_DENSE_RULES
    if enabled and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    if enabled:
        get_metal_moe_library()
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qm

        if _ORIG_QWEN_MOE_RULES is None:
            _ORIG_QWEN_MOE_RULES = (qm.torch_recurrent_gated_delta_rule, qm.torch_chunk_gated_delta_rule)
        if enabled:
            qm.torch_recurrent_gated_delta_rule = fused_gated_delta_rule_metal
            qm.torch_chunk_gated_delta_rule = fused_gated_delta_rule_metal
        else:
            qm.torch_recurrent_gated_delta_rule, qm.torch_chunk_gated_delta_rule = _ORIG_QWEN_MOE_RULES
    except Exception:
        pass
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qd

        if _ORIG_QWEN_DENSE_RULES is None:
            _ORIG_QWEN_DENSE_RULES = (qd.torch_recurrent_gated_delta_rule, qd.torch_chunk_gated_delta_rule)
        if enabled:
            qd.torch_recurrent_gated_delta_rule = fused_gated_delta_rule_metal
            qd.torch_chunk_gated_delta_rule = fused_gated_delta_rule_metal
        else:
            qd.torch_recurrent_gated_delta_rule, qd.torch_chunk_gated_delta_rule = _ORIG_QWEN_DENSE_RULES
    except Exception:
        pass
    _DELTANET_PATCH_INSTALLED = bool(enabled)
    return _DELTANET_PATCH_INSTALLED


def install_metal_gated_deltanet_patch() -> bool:
    """Installs `fused_gated_delta_rule_metal` into HuggingFace `qwen3_5_moe` and `qwen3_5` modules."""
    return set_metal_gated_deltanet_enabled(True)


# =============================================================================
# Parallel QD=8 NVMe `pread` + Online `LUT6` Router-Lookahead Prefetcher
# =============================================================================

_NVME_POOL: ThreadPoolExecutor | None = None


def get_nvme_pread_pool() -> ThreadPoolExecutor:
    global _NVME_POOL
    if _NVME_POOL is None:
        _NVME_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="airmol-nvme-qd8")
    return _NVME_POOL


def pread_packed_experts_qd8(
    pack_path: Path,
    uniq_experts: list[int],
    meta: dict,
    meter: Any,
    device: torch.device,
    open_uncached_fn: Any,
    prefetched_raw: dict[int, bytes] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Reads active experts `uniq_experts` into GPU-ready raw `INT4` buffers (`packed_t`, `mins_t`, `scales_t`)
    using Queue-Depth-8 parallel `os.pread` threads + optional `BenesLUT6` pre-fetched raw slices!
    """
    stride = int(meta["expert_stride_bytes"])
    m_tot = int(meta["m_tot"])
    p_len = m_tot * 32
    m_len = m_tot * 2
    k_cnt = len(uniq_experts)

    packed_np = np.empty((k_cnt, m_tot, 32), dtype=np.uint8)
    mins_np = np.empty((k_cnt, m_tot), dtype=np.float16)
    scales_np = np.empty((k_cnt, m_tot), dtype=np.float16)

    missing_indices = []
    for idx, e_id in enumerate(uniq_experts):
        if prefetched_raw is not None and e_id in prefetched_raw:
            raw = prefetched_raw[e_id]
            mv = memoryview(raw)
            packed_np[idx] = np.frombuffer(mv[:p_len], dtype=np.uint8).reshape(m_tot, 32)
            mins_np[idx] = np.frombuffer(mv[p_len : p_len + m_len], dtype=np.float16)
            scales_np[idx] = np.frombuffer(mv[p_len + m_len : p_len + 2 * m_len], dtype=np.float16)
        else:
            missing_indices.append((idx, e_id))

    if missing_indices:
        fd = open_uncached_fn(pack_path, os.O_RDONLY)
        try:
            if len(missing_indices) >= 4:
                pool = get_nvme_pread_pool()

                def _read_one(pair: tuple[int, int]) -> tuple[int, bytes]:
                    i_slot, e = pair
                    return i_slot, os.pread(fd, stride, e * stride)

                for i_slot, raw in pool.map(_read_one, missing_indices):
                    meter.bytes_read += len(raw)
                    mv = memoryview(raw)
                    packed_np[i_slot] = np.frombuffer(mv[:p_len], dtype=np.uint8).reshape(m_tot, 32)
                    mins_np[i_slot] = np.frombuffer(mv[p_len : p_len + m_len], dtype=np.float16)
                    scales_np[i_slot] = np.frombuffer(mv[p_len + m_len : p_len + 2 * m_len], dtype=np.float16)
            else:
                for i_slot, e in missing_indices:
                    raw = os.pread(fd, stride, e * stride)
                    meter.bytes_read += len(raw)
                    mv = memoryview(raw)
                    packed_np[i_slot] = np.frombuffer(mv[:p_len], dtype=np.uint8).reshape(m_tot, 32)
                    mins_np[i_slot] = np.frombuffer(mv[p_len : p_len + m_len], dtype=np.float16)
                    scales_np[i_slot] = np.frombuffer(mv[p_len + m_len : p_len + 2 * m_len], dtype=np.float16)
            meter.shard_loads += 1
        finally:
            os.close(fd)

    packed_t = torch.from_numpy(packed_np).to(device=device)
    mins_t = torch.from_numpy(mins_np).to(device=device)
    scales_t = torch.from_numpy(scales_np).to(device=device)
    ram_bytes = k_cnt * stride
    return packed_t, mins_t, scales_t, ram_bytes


class BenesLUT6RouterPrefetcher:
    """
    Online `6`-input `LUT6` Sign-Sketch Router-Lookahead Prefetcher:
      - Maps a `6`-bit sign sketch of layer `l`'s hidden state `h_l` (`0..63`) to empirical expert
        co-activation counts at layer `l + 1`.
      - Only prefetches experts whose empirical conditional probability `P(e in U_{l+1} | sketch_6b) >= 0.80`
        so NVMe I/O is never wasted on low-confidence guesses while overlapping high-certainty expert reads
        asynchronously with GPU compute on layer `l`!
    """

    def __init__(self, num_layers: int, num_experts: int, min_obs: int = 6, min_prob: float = 0.92) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.min_obs = min_obs
        self.min_prob = min_prob
        self.bucket_obs = np.zeros((num_layers, 64), dtype=np.int32)
        self.expert_hits = np.zeros((num_layers, 64, num_experts), dtype=np.int32)
        self._pending: tuple[int, Future[dict[int, bytes]]] | None = None

    @staticmethod
    def sketch_6bit(h: torch.Tensor) -> int:
        """Extracts a deterministic 6-bit sign sketch (`0..63`) from the last token's hidden vector."""
        d_use = (int(h.size(-1)) // 6) * 6
        if d_use < 6:
            return 0
        v = h[0, -1, :d_use].detach().reshape(6, -1).mean(dim=-1)
        bits = (v > 0).to(torch.int32).cpu().tolist()
        idx = 0
        for b_i, bit in enumerate(bits):
            if bit:
                idx |= 1 << b_i
        return idx

    def predict_high_conf_experts(self, next_layer: int, sketch: int, max_prefetch: int = 6) -> list[int]:
        if next_layer >= self.num_layers:
            return []
        obs = int(self.bucket_obs[next_layer, sketch])
        if obs < self.min_obs:
            return []
        probs = self.expert_hits[next_layer, sketch] / float(obs)
        cand = np.where(probs >= self.min_prob)[0]
        if cand.size == 0:
            return []
        ranked = cand[np.argsort(-probs[cand])[:max_prefetch]]
        return [int(x) for x in ranked.tolist()]

    def record_actual_experts(self, layer_idx: int, sketch: int, uniq_experts: list[int]) -> None:
        self.bucket_obs[layer_idx, sketch] += 1
        if uniq_experts:
            self.expert_hits[layer_idx, sketch, uniq_experts] += 1

    def launch_async_prefetch(
        self,
        next_layer: int,
        sketch: int,
        pack_path: Path,
        stride: int,
        open_uncached_fn: Any,
    ) -> None:
        preds = self.predict_high_conf_experts(next_layer, sketch)
        if not preds or not pack_path.exists():
            self._pending = None
            return

        pool = get_nvme_pread_pool()

        def _bg_fetch() -> dict[int, bytes]:
            fd = open_uncached_fn(pack_path, os.O_RDONLY)
            try:
                return {e: os.pread(fd, stride, e * stride) for e in preds}
            finally:
                os.close(fd)

        self._pending = (next_layer, pool.submit(_bg_fetch))

    def collect_prefetch(self, layer_idx: int, meter: Any) -> dict[int, bytes] | None:
        if self._pending is None:
            return None
        pend_layer, fut = self._pending
        self._pending = None
        if pend_layer != layer_idx:
            return None
        res = fut.result()
        for raw in res.values():
            meter.bytes_read += len(raw)
        if res:
            meter.shard_loads += 1
        return res
