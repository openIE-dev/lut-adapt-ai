"""Fused Chunkwise Parallel SSD Engine + Exact Analytical Telescoping Adjoints for Mamba-3 MIMO."""

from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

CHUNK_Q = 32  # 32 steps x rank R=4 => QR=128, Q2R=256 Tensor Core tiles


if HAS_TRITON:

    @triton.jit
    def _mamba3_chunk_state_kernel(
        b_hat_ptr,
        x_hat_ptr,
        end_decay_ptr,
        states_out_ptr,
        stride_bh,
        stride_bc,
        stride_bq,
        stride_bn,
        stride_xh,
        stride_xc,
        stride_xq,
        stride_xp,
        stride_dh,
        stride_dc,
        stride_dq,
        stride_sh,
        stride_sc,
        stride_sn,
        stride_sp,
        Q2R: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        c_id = tl.program_id(1)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)
        acc = tl.zeros([N, P], dtype=tl.float32)

        for k0 in range(0, Q2R, 64):
            offs_k = k0 + tl.arange(0, 64)
            d_ptrs = end_decay_ptr + bh_id * stride_dh + c_id * stride_dc + offs_k * stride_dq
            decay_k = tl.load(d_ptrs).to(tl.float32)

            b_ptrs = (
                b_hat_ptr
                + bh_id * stride_bh
                + c_id * stride_bc
                + offs_k[:, None] * stride_bq
                + offs_n[None, :] * stride_bn
            )
            x_ptrs = (
                x_hat_ptr
                + bh_id * stride_xh
                + c_id * stride_xc
                + offs_k[:, None] * stride_xq
                + offs_p[None, :] * stride_xp
            )
            b_tile = (tl.load(b_ptrs).to(tl.float32) * decay_k[:, None]).to(tl.bfloat16)
            x_tile = tl.load(x_ptrs).to(tl.bfloat16)
            acc += tl.dot(tl.trans(b_tile), x_tile, out_dtype=tl.float32)

        s_ptrs = (
            states_out_ptr
            + bh_id * stride_sh
            + c_id * stride_sc
            + offs_n[:, None] * stride_sn
            + offs_p[None, :] * stride_sp
        )
        tl.store(s_ptrs, acc)

    @triton.jit
    def _mamba3_interchunk_scan_kernel(
        chunk_states_ptr,
        chunk_decay_ptr,
        h0_ptr,
        h_final_ptr,
        prev_h_out_ptr,
        C,
        stride_sh,
        stride_sc,
        stride_sn,
        stride_sp,
        stride_ch,
        stride_cc,
        stride_h0h,
        stride_h0n,
        stride_h0p,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        h0_ptrs = (
            h0_ptr
            + bh_id * stride_h0h
            + offs_n[:, None] * stride_h0n
            + offs_p[None, :] * stride_h0p
        )
        h_running = tl.load(h0_ptrs).to(tl.float32)

        for c in range(C):
            prev_ptrs = (
                prev_h_out_ptr
                + bh_id * stride_sh
                + c * stride_sc
                + offs_n[:, None] * stride_sn
                + offs_p[None, :] * stride_sp
            )
            tl.store(prev_ptrs, h_running)

            d_c = tl.load(chunk_decay_ptr + bh_id * stride_ch + c * stride_cc).to(tl.float32)
            s_ptrs = (
                chunk_states_ptr
                + bh_id * stride_sh
                + c * stride_sc
                + offs_n[:, None] * stride_sn
                + offs_p[None, :] * stride_sp
            )
            s_c = tl.load(s_ptrs).to(tl.float32)
            h_running = d_c * h_running + s_c

        hf_ptrs = (
            h_final_ptr
            + bh_id * stride_h0h
            + offs_n[:, None] * stride_h0n
            + offs_p[None, :] * stride_h0p
        )
        tl.store(hf_ptrs, h_running)

    @triton.jit
    def _mamba3_ssd_output_kernel(
        c_rot_ptr,
        b_hat_ptr,
        x_hat_ptr,
        cum_a_ptr,
        prev_h_ptr,
        y_out_ptr,
        stride_ch,
        stride_cc,
        stride_cq,
        stride_cn,
        stride_bh,
        stride_bc,
        stride_bq,
        stride_bn,
        stride_xh,
        stride_xc,
        stride_xq,
        stride_xp,
        stride_ah,
        stride_ac,
        stride_at,
        stride_hh,
        stride_hc,
        stride_hn,
        stride_hp,
        stride_yh,
        stride_yc,
        stride_yq,
        stride_yp,
        Q: tl.constexpr,
        R: tl.constexpr,
        QR: tl.constexpr,
        Q2R: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        c_id = tl.program_id(1)
        offs_qr = tl.arange(0, QR)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        t_i = offs_qr // R
        a_base = cum_a_ptr + bh_id * stride_ah + c_id * stride_ac
        a_i = tl.load(a_base + t_i * stride_at).to(tl.float32)

        c_ptrs = (
            c_rot_ptr
            + bh_id * stride_ch
            + c_id * stride_cc
            + offs_qr[:, None] * stride_cq
            + offs_n[None, :] * stride_cn
        )
        c_tile = tl.load(c_ptrs).to(tl.bfloat16)

        h_ptrs = (
            prev_h_ptr
            + bh_id * stride_hh
            + c_id * stride_hc
            + offs_n[:, None] * stride_hn
            + offs_p[None, :] * stride_hp
        )
        h_prev = tl.load(h_ptrs).to(tl.bfloat16)
        c_scaled = (c_tile.to(tl.float32) * tl.exp(a_i)[:, None]).to(tl.bfloat16)
        acc_y = tl.dot(c_scaled, h_prev, out_dtype=tl.float32)

        for k0 in range(0, Q2R, 64):
            offs_k = k0 + tl.arange(0, 64)
            t_j = offs_k // (2 * R)
            a_j = tl.load(a_base + t_j * stride_at).to(tl.float32)

            b_ptrs = (
                b_hat_ptr
                + bh_id * stride_bh
                + c_id * stride_bc
                + offs_k[:, None] * stride_bq
                + offs_n[None, :] * stride_bn
            )
            x_ptrs = (
                x_hat_ptr
                + bh_id * stride_xh
                + c_id * stride_xc
                + offs_k[:, None] * stride_xq
                + offs_p[None, :] * stride_xp
            )
            b_sub = tl.load(b_ptrs).to(tl.bfloat16)
            x_sub = tl.load(x_ptrs).to(tl.bfloat16)

            scores = tl.dot(c_tile, tl.trans(b_sub), out_dtype=tl.float32)
            causal = t_i[:, None] >= t_j[None, :]
            diff = tl.where(causal, a_i[:, None] - a_j[None, :], -float("inf"))
            masked_scores = (scores * tl.exp(diff)).to(tl.bfloat16)
            acc_y += tl.dot(masked_scores, x_sub, out_dtype=tl.float32)

        y_ptrs = (
            y_out_ptr
            + bh_id * stride_yh
            + c_id * stride_yc
            + offs_qr[:, None] * stride_yq
            + offs_p[None, :] * stride_yp
        )
        tl.store(y_ptrs, acc_y)

    @triton.jit
    def _mamba3_bwd_offdiag_grad_kernel(
        c_rot_ptr,
        dy_ptr,
        cum_a_ptr,
        dh_local_ptr,
        stride_ch,
        stride_cc,
        stride_cq,
        stride_cn,
        stride_yh,
        stride_yc,
        stride_yq,
        stride_yp,
        stride_ah,
        stride_ac,
        stride_at,
        stride_dh,
        stride_dc,
        stride_dn,
        stride_dp,
        R: tl.constexpr,
        QR: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        c_id = tl.program_id(1)
        offs_qr = tl.arange(0, QR)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        t_i = offs_qr // R
        a_i = tl.load(cum_a_ptr + bh_id * stride_ah + c_id * stride_ac + t_i * stride_at).to(
            tl.float32
        )

        c_ptrs = (
            c_rot_ptr
            + bh_id * stride_ch
            + c_id * stride_cc
            + offs_qr[:, None] * stride_cq
            + offs_n[None, :] * stride_cn
        )
        dy_ptrs = (
            dy_ptr
            + bh_id * stride_yh
            + c_id * stride_yc
            + offs_qr[:, None] * stride_yq
            + offs_p[None, :] * stride_yp
        )

        c_scaled = (tl.load(c_ptrs).to(tl.float32) * tl.exp(a_i)[:, None]).to(tl.bfloat16)
        dy_tile = tl.load(dy_ptrs).to(tl.bfloat16)
        dh_loc = tl.dot(tl.trans(c_scaled), dy_tile, out_dtype=tl.float32)

        out_ptrs = (
            dh_local_ptr
            + bh_id * stride_dh
            + c_id * stride_dc
            + offs_n[:, None] * stride_dn
            + offs_p[None, :] * stride_dp
        )
        tl.store(out_ptrs, dh_loc)

    @triton.jit
    def _mamba3_bwd_interchunk_scan_kernel(
        dh_local_ptr,
        chunk_decay_ptr,
        prev_h_ptr,
        h_final_ptr,
        dh_final_ptr,
        dh_out_ptr,
        dh0_out_ptr,
        end_boundary_ptr,
        C,
        stride_sh,
        stride_sc,
        stride_sn,
        stride_sp,
        stride_ch,
        stride_cc,
        stride_h0h,
        stride_h0n,
        stride_h0p,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        dhf_ptrs = (
            dh_final_ptr
            + bh_id * stride_h0h
            + offs_n[:, None] * stride_h0n
            + offs_p[None, :] * stride_h0p
        )
        dh_running = tl.load(dhf_ptrs).to(tl.float32)

        for rev_step in range(C):
            c = C - 1 - rev_step
            dhc_ptrs = (
                dh_out_ptr
                + bh_id * stride_sh
                + c * stride_sc
                + offs_n[:, None] * stride_sn
                + offs_p[None, :] * stride_sp
            )
            tl.store(dhc_ptrs, dh_running)

            if c == C - 1:
                hc_ptrs = (
                    h_final_ptr
                    + bh_id * stride_h0h
                    + offs_n[:, None] * stride_h0n
                    + offs_p[None, :] * stride_h0p
                )
            else:
                hc_ptrs = (
                    prev_h_ptr
                    + bh_id * stride_sh
                    + (c + 1) * stride_sc
                    + offs_n[:, None] * stride_sn
                    + offs_p[None, :] * stride_sp
                )
            h_c = tl.load(hc_ptrs).to(tl.float32)

            boundary_dot = tl.sum(dh_running * h_c)
            tl.store(end_boundary_ptr + bh_id * stride_ch + c * stride_cc, boundary_dot)

            d_c = tl.load(chunk_decay_ptr + bh_id * stride_ch + c * stride_cc).to(tl.float32)
            loc_ptrs = (
                dh_local_ptr
                + bh_id * stride_sh
                + c * stride_sc
                + offs_n[:, None] * stride_sn
                + offs_p[None, :] * stride_sp
            )
            dh_loc = tl.load(loc_ptrs).to(tl.float32)
            dh_running = d_c * dh_running + dh_loc

        dh0_ptrs = (
            dh0_out_ptr
            + bh_id * stride_h0h
            + offs_n[:, None] * stride_h0n
            + offs_p[None, :] * stride_h0p
        )
        tl.store(dh0_ptrs, dh_running)

    @triton.jit
    def _mamba3_bwd_ssd_main_kernel(
        c_rot_ptr,
        b_hat_ptr,
        x_hat_ptr,
        dy_ptr,
        cum_a_ptr,
        prev_h_ptr,
        dh_c_ptr,
        end_boundary_ptr,
        dc_out_ptr,
        db_out_ptr,
        dx_out_ptr,
        dla_out_ptr,
        stride_ch,
        stride_cc,
        stride_cq,
        stride_cn,
        stride_bh,
        stride_bc,
        stride_bq,
        stride_bn,
        stride_xh,
        stride_xc,
        stride_xq,
        stride_xp,
        stride_yh,
        stride_yc,
        stride_yq,
        stride_yp,
        stride_ah,
        stride_ac,
        stride_at,
        stride_hh,
        stride_hc,
        stride_hn,
        stride_hp,
        stride_ebh,
        stride_ebc,
        Q: tl.constexpr,
        R: tl.constexpr,
        QR: tl.constexpr,
        Q2R: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        c_id = tl.program_id(1)
        offs_qr = tl.arange(0, QR)
        offs_n = tl.arange(0, N)
        offs_p = tl.arange(0, P)

        t_i = offs_qr // R
        a_base = cum_a_ptr + bh_id * stride_ah + c_id * stride_ac
        a_i = tl.load(a_base + t_i * stride_at).to(tl.float32)
        a_end = tl.load(a_base + (Q - 1) * stride_at).to(tl.float32)

        c_ptrs = (
            c_rot_ptr
            + bh_id * stride_ch
            + c_id * stride_cc
            + offs_qr[:, None] * stride_cq
            + offs_n[None, :] * stride_cn
        )
        dy_ptrs = (
            dy_ptr
            + bh_id * stride_yh
            + c_id * stride_yc
            + offs_qr[:, None] * stride_yq
            + offs_p[None, :] * stride_yp
        )
        c_f32 = tl.load(c_ptrs).to(tl.float32)
        c_tile = c_f32.to(tl.bfloat16)
        dy_tile = tl.load(dy_ptrs).to(tl.bfloat16)

        h_prev_ptrs = (
            prev_h_ptr
            + bh_id * stride_hh
            + c_id * stride_hc
            + offs_n[:, None] * stride_hn
            + offs_p[None, :] * stride_hp
        )
        dh_c_ptrs = (
            dh_c_ptr
            + bh_id * stride_hh
            + c_id * stride_hc
            + offs_n[:, None] * stride_hn
            + offs_p[None, :] * stride_hp
        )
        h_prev = tl.load(h_prev_ptrs).to(tl.bfloat16)
        dh_c = tl.load(dh_c_ptrs).to(tl.bfloat16)

        dc_acc = tl.dot(dy_tile, tl.trans(h_prev), out_dtype=tl.float32) * tl.exp(a_i)[:, None]
        offs_q = tl.arange(0, Q)
        neg_db_b_per_q = tl.zeros([Q], dtype=tl.float32)

        for k0 in range(0, Q2R, 64):
            offs_k = k0 + tl.arange(0, 64)
            t_j = offs_k // (2 * R)
            a_j = tl.load(a_base + t_j * stride_at).to(tl.float32)
            end_scale = tl.exp(tl.minimum(a_end - a_j, 0.0))

            b_ptrs = (
                b_hat_ptr
                + bh_id * stride_bh
                + c_id * stride_bc
                + offs_k[:, None] * stride_bq
                + offs_n[None, :] * stride_bn
            )
            x_ptrs = (
                x_hat_ptr
                + bh_id * stride_xh
                + c_id * stride_xc
                + offs_k[:, None] * stride_xq
                + offs_p[None, :] * stride_xp
            )
            b_f32 = tl.load(b_ptrs).to(tl.float32)
            b_sub = b_f32.to(tl.bfloat16)
            x_sub = tl.load(x_ptrs).to(tl.bfloat16)

            causal = t_i[:, None] >= t_j[None, :]
            diff = tl.where(causal, a_i[:, None] - a_j[None, :], -float("inf"))
            L_sub = tl.exp(diff)

            m_raw = tl.dot(c_tile, tl.trans(b_sub), out_dtype=tl.float32)
            dm_raw = tl.dot(dy_tile, tl.trans(x_sub), out_dtype=tl.float32)
            m_sub = (m_raw * L_sub).to(tl.bfloat16)
            dm_sub = (dm_raw * L_sub).to(tl.bfloat16)

            dc_acc += tl.dot(dm_sub, b_sub, out_dtype=tl.float32)

            db_state = tl.dot(x_sub, tl.trans(dh_c), out_dtype=tl.float32) * end_scale[:, None]
            db_diag = tl.dot(tl.trans(dm_sub), c_tile, out_dtype=tl.float32)
            db_sub = db_state + db_diag

            dx_state = tl.dot(b_sub, dh_c, out_dtype=tl.float32) * end_scale[:, None]
            dx_diag = tl.dot(tl.trans(m_sub), dy_tile, out_dtype=tl.float32)
            dx_sub = dx_state + dx_diag

            db_out_ptrs = (
                db_out_ptr
                + bh_id * stride_bh
                + c_id * stride_bc
                + offs_k[:, None] * stride_bq
                + offs_n[None, :] * stride_bn
            )
            dx_out_ptrs = (
                dx_out_ptr
                + bh_id * stride_xh
                + c_id * stride_xc
                + offs_k[:, None] * stride_xq
                + offs_p[None, :] * stride_xp
            )
            tl.store(db_out_ptrs, db_sub)
            tl.store(dx_out_ptrs, dx_sub)

            row_dot_b = tl.sum(db_sub * b_f32, axis=1)
            match_q_b = (offs_q[:, None] == t_j[None, :]).to(tl.float32)
            neg_db_b_per_q += tl.sum(match_q_b * row_dot_b[None, :], axis=1)

        dc_out_ptrs = (
            dc_out_ptr
            + bh_id * stride_ch
            + c_id * stride_cc
            + offs_qr[:, None] * stride_cq
            + offs_n[None, :] * stride_cn
        )
        tl.store(dc_out_ptrs, dc_acc)

        row_dot_c = tl.sum(dc_acc * c_f32, axis=1)
        match_q_c = (offs_q[:, None] == t_i[None, :]).to(tl.float32)
        pos_dc_c_per_q = tl.sum(match_q_c * row_dot_c[None, :], axis=1)

        end_b = tl.load(end_boundary_ptr + bh_id * stride_ebh + c_id * stride_ebc).to(tl.float32)
        is_last_q = (offs_q == (Q - 1)).to(tl.float32)
        da_q = pos_dc_c_per_q - neg_db_b_per_q + is_last_q * end_b

        suffix_mask = (offs_q[None, :] >= offs_q[:, None]).to(tl.float32)
        dlog_alpha = tl.sum(suffix_mask * da_q[None, :], axis=1)

        dla_ptrs = dla_out_ptr + bh_id * stride_ah + c_id * stride_ac + offs_q * stride_at
        tl.store(dla_ptrs, dlog_alpha)


def _chunked_ssd_reference(
    X_hat: torch.Tensor,
    B_hat: torch.Tensor,
    C_rot: torch.Tensor,
    log_alpha: torch.Tensor,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable chunkwise SSD matmul reference."""
    B, C, Q, R2, H, P = X_hat.shape
    R = R2 // 2
    N = B_hat.size(-1)

    Xh = X_hat.permute(0, 4, 1, 2, 3, 5).reshape(B * H, C, Q * R2, P)
    Bh = B_hat.permute(0, 4, 1, 2, 3, 5).reshape(B * H, C, Q * R2, N)
    Cr = C_rot.permute(0, 4, 1, 2, 3, 5).reshape(B * H, C, Q * R, N)
    la = log_alpha.permute(0, 3, 1, 2).reshape(B * H, C, Q)

    cum_a = torch.cumsum(la, dim=-1)
    end_a = cum_a[..., -1:]
    decay_q2r = torch.exp((end_a - cum_a).clamp(max=0.0)).repeat_interleave(R2, dim=-1)

    S_c = torch.matmul((Bh * decay_q2r.unsqueeze(-1)).transpose(-1, -2), Xh)
    chunk_decay = torch.exp(cum_a[..., -1])
    h_cur = h0.reshape(B * H, N, P)
    prev_states = []
    for c in range(C):
        prev_states.append(h_cur)
        h_cur = chunk_decay[:, c].view(-1, 1, 1) * h_cur + S_c[:, c]
    H_prev = torch.stack(prev_states, dim=1)

    a_qr = cum_a.repeat_interleave(R, dim=-1)
    a_q2r = cum_a.repeat_interleave(R2, dim=-1)
    Y_off = torch.matmul(Cr * torch.exp(a_qr).unsqueeze(-1), H_prev)

    t_i = torch.arange(Q, device=X_hat.device).repeat_interleave(R).view(1, 1, Q * R, 1)
    t_j = torch.arange(Q, device=X_hat.device).repeat_interleave(R2).view(1, 1, 1, Q * R2)
    L_mask = torch.where(
        t_i >= t_j, torch.exp((a_qr.unsqueeze(-1) - a_q2r.unsqueeze(-2)).clamp(max=0.0)), 0.0
    )
    Y_diag = torch.matmul(torch.matmul(Cr, Bh.transpose(-1, -2)) * L_mask, Xh)

    Y_total = (Y_off + Y_diag).view(B, H, C, Q, R, P).permute(0, 2, 3, 1, 5, 4)
    return Y_total, h_cur.view(B, H, N, P)


def _chunked_ssd_analytical_forward_and_cache(
    X_hat: torch.Tensor,
    B_hat: torch.Tensor,
    C_rot: torch.Tensor,
    log_alpha: torch.Tensor,
    h0: torch.Tensor,
):
    """Zero-autograd-graph chunkwise forward saving compact state boundaries for analytical backward."""
    B, C, Q, R2, H, P = X_hat.shape
    R = R2 // 2
    N = B_hat.size(-1)
    BH = B * H

    Xh = X_hat.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R2, P).float()
    Bh = B_hat.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R2, N).float()
    Cr = C_rot.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R, N).float()
    la = log_alpha.permute(0, 3, 1, 2).reshape(BH, C, Q).float()

    cum_a = torch.cumsum(la, dim=-1)
    end_a = cum_a[..., -1:]
    decay_q2r = torch.exp((end_a - cum_a).clamp(max=0.0)).repeat_interleave(R2, dim=-1)

    S_c = torch.matmul((Bh * decay_q2r.unsqueeze(-1)).transpose(-1, -2), Xh)
    chunk_decay = torch.exp(cum_a[..., -1])
    h_cur = h0.reshape(BH, N, P).float()
    prev_states = []
    for c in range(C):
        prev_states.append(h_cur)
        h_cur = chunk_decay[:, c].view(-1, 1, 1) * h_cur + S_c[:, c]
    H_prev = torch.stack(prev_states, dim=1)
    h_final = h_cur

    a_qr = cum_a.repeat_interleave(R, dim=-1)
    a_q2r = cum_a.repeat_interleave(R2, dim=-1)
    Y_off = torch.matmul(Cr * torch.exp(a_qr).unsqueeze(-1), H_prev)

    t_i = torch.arange(Q, device=X_hat.device).repeat_interleave(R).view(1, 1, Q * R, 1)
    t_j = torch.arange(Q, device=X_hat.device).repeat_interleave(R2).view(1, 1, 1, Q * R2)
    L_mask = torch.where(
        t_i >= t_j, torch.exp((a_qr.unsqueeze(-1) - a_q2r.unsqueeze(-2)).clamp(max=0.0)), 0.0
    )
    Y_diag = torch.matmul(torch.matmul(Cr, Bh.transpose(-1, -2)) * L_mask, Xh)

    Y_total = (
        (Y_off + Y_diag).view(B, H, C, Q, R, P).permute(0, 2, 3, 1, 5, 4).to(X_hat.dtype)
    )
    return Y_total, h_final.view(B, H, N, P).to(h0.dtype), (Xh, Bh, Cr, cum_a, chunk_decay, H_prev, h_final)


def _chunked_ssd_analytical_backward(
    grad_Y: torch.Tensor,
    grad_h_final: torch.Tensor | None,
    saved: tuple[torch.Tensor, ...],
    shapes: tuple[int, ...],
    dtypes: tuple[torch.dtype, ...],
):
    """Exact reverse-mode SSD adjoints + telescoping d(log_alpha) on MPS/CPU without autograd tape."""
    B, C, Q, R, R2, H, P, N = shapes
    dt_x, dt_b, dt_c, dt_la, dt_h0 = dtypes
    Xh, Bh, Cr, cum_a, chunk_decay, H_prev, h_final = saved
    BH = B * H

    dY = grad_Y.permute(0, 3, 1, 2, 5, 4).reshape(BH, C, Q * R, P).float()
    dh_running = (
        grad_h_final.reshape(BH, N, P).float()
        if grad_h_final is not None
        else torch.zeros((BH, N, P), device=Xh.device, dtype=torch.float32)
    )

    a_qr = cum_a.repeat_interleave(R, dim=-1)  # [BH, C, QR]
    a_q2r = cum_a.repeat_interleave(R2, dim=-1)  # [BH, C, Q2R]
    c_scaled = Cr * torch.exp(a_qr).unsqueeze(-1)
    dh_local = torch.matmul(c_scaled.transpose(-1, -2), dY)  # [BH, C, N, P]

    # Reverse inter-chunk adjoint scan + boundary <dH_c, H_c>
    dH_c_list = [None] * C
    end_boundary = torch.empty((BH, C), device=Xh.device, dtype=torch.float32)
    for c in range(C - 1, -1, -1):
        dH_c_list[c] = dh_running
        h_c = h_final if c == C - 1 else H_prev[:, c + 1]
        end_boundary[:, c] = (dh_running * h_c).sum(dim=(-2, -1))
        dh_running = chunk_decay[:, c].view(-1, 1, 1) * dh_running + dh_local[:, c]

    dH_c = torch.stack(dH_c_list, dim=1)  # [BH, C, N, P]
    dh0 = dh_running

    # Recompute causal surrogate M and dM per chunk
    t_i = torch.arange(Q, device=Xh.device).repeat_interleave(R).view(1, 1, Q * R, 1)
    t_j = torch.arange(Q, device=Xh.device).repeat_interleave(R2).view(1, 1, 1, Q * R2)
    L_mask = torch.where(
        t_i >= t_j, torch.exp((a_qr.unsqueeze(-1) - a_q2r.unsqueeze(-2)).clamp(max=0.0)), 0.0
    )

    M = torch.matmul(Cr, Bh.transpose(-1, -2)) * L_mask
    dM = torch.matmul(dY, Xh.transpose(-1, -2)) * L_mask

    end_scale = torch.exp((cum_a[..., -1:] - cum_a).clamp(max=0.0)).repeat_interleave(R2, dim=-1)

    dCr = torch.matmul(dY, H_prev.transpose(-1, -2)) * torch.exp(a_qr).unsqueeze(-1) + torch.matmul(
        dM, Bh
    )
    dBh = torch.matmul(Xh, dH_c.transpose(-1, -2)) * end_scale.unsqueeze(-1) + torch.matmul(
        dM.transpose(-1, -2), Cr
    )
    dXh = torch.matmul(Bh, dH_c) * end_scale.unsqueeze(-1) + torch.matmul(
        M.transpose(-1, -2), dY
    )

    # Telescoping scalar identity for d(log_alpha)
    pos_dc_c = (dCr * Cr).sum(dim=-1).view(BH, C, Q, R).sum(dim=-1)  # [BH, C, Q]
    neg_db_b = (dBh * Bh).sum(dim=-1).view(BH, C, Q, R2).sum(dim=-1)  # [BH, C, Q]
    da_q = pos_dc_c - neg_db_b
    da_q[:, :, -1] = da_q[:, :, -1] + end_boundary

    # Reverse cumulative sum along Q
    dla = torch.flip(torch.cumsum(torch.flip(da_q, dims=[-1]), dim=-1), dims=[-1])

    grad_X = dXh.view(B, H, C, Q, R2, P).permute(0, 2, 3, 4, 1, 5).to(dt_x)
    grad_B = dBh.view(B, H, C, Q, R2, N).permute(0, 2, 3, 4, 1, 5).to(dt_b)
    grad_C = dCr.view(B, H, C, Q, R, N).permute(0, 2, 3, 4, 1, 5).to(dt_c)
    grad_la = dla.view(B, H, C, Q).permute(0, 2, 3, 1).to(dt_la)
    grad_h0 = dh0.view(B, H, N, P).to(dt_h0)
    return grad_X, grad_B, grad_C, grad_la, grad_h0


class _Mamba3TritonSSDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_hat, B_hat, C_rot, log_alpha, h0):
        B, C, Q, R2, H, P = X_hat.shape
        R = R2 // 2
        N = B_hat.size(-1)
        use_triton = (
            HAS_TRITON and X_hat.is_cuda and Q == CHUNK_Q and N == 64 and P == 64 and R == 4
        )
        ctx.use_triton = use_triton
        ctx.shapes = (B, C, Q, R, R2, H, P, N)
        ctx.orig_dtypes = (X_hat.dtype, B_hat.dtype, C_rot.dtype, log_alpha.dtype, h0.dtype)

        if use_triton:
            BH = B * H
            Xh = X_hat.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R2, P).to(torch.bfloat16).contiguous()
            Bh = B_hat.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R2, N).to(torch.bfloat16).contiguous()
            Cr = C_rot.permute(0, 4, 1, 2, 3, 5).reshape(BH, C, Q * R, N).to(torch.bfloat16).contiguous()
            la = log_alpha.permute(0, 3, 1, 2).reshape(BH, C, Q).to(torch.float32).contiguous()

            cum_a = torch.cumsum(la, dim=-1).contiguous()
            end_decay = (
                torch.exp((cum_a[..., -1:] - cum_a).clamp(max=0.0))
                .repeat_interleave(R2, dim=-1)
                .contiguous()
            )
            chunk_decay = torch.exp(cum_a[..., -1]).contiguous()

            S_c = torch.empty((BH, C, N, P), device=X_hat.device, dtype=torch.float32)
            H_prev = torch.empty((BH, C, N, P), device=X_hat.device, dtype=torch.float32)
            h_final = torch.empty((BH, N, P), device=X_hat.device, dtype=torch.float32)
            Y_out = torch.empty((BH, C, Q * R, P), device=X_hat.device, dtype=torch.float32)
            h0_flat = h0.reshape(BH, N, P).to(torch.float32).contiguous()

            _mamba3_chunk_state_kernel[(BH, C)](
                Bh,
                Xh,
                end_decay,
                S_c,
                Bh.stride(0),
                Bh.stride(1),
                Bh.stride(2),
                Bh.stride(3),
                Xh.stride(0),
                Xh.stride(1),
                Xh.stride(2),
                Xh.stride(3),
                end_decay.stride(0),
                end_decay.stride(1),
                end_decay.stride(2),
                S_c.stride(0),
                S_c.stride(1),
                S_c.stride(2),
                S_c.stride(3),
                Q2R=Q * R2,
                N=N,
                P=P,
            )

            _mamba3_interchunk_scan_kernel[(BH,)](
                S_c,
                chunk_decay,
                h0_flat,
                h_final,
                H_prev,
                C,
                S_c.stride(0),
                S_c.stride(1),
                S_c.stride(2),
                S_c.stride(3),
                chunk_decay.stride(0),
                chunk_decay.stride(1),
                h0_flat.stride(0),
                h0_flat.stride(1),
                h0_flat.stride(2),
                N=N,
                P=P,
            )

            _mamba3_ssd_output_kernel[(BH, C)](
                Cr,
                Bh,
                Xh,
                cum_a,
                H_prev,
                Y_out,
                Cr.stride(0),
                Cr.stride(1),
                Cr.stride(2),
                Cr.stride(3),
                Bh.stride(0),
                Bh.stride(1),
                Bh.stride(2),
                Bh.stride(3),
                Xh.stride(0),
                Xh.stride(1),
                Xh.stride(2),
                Xh.stride(3),
                cum_a.stride(0),
                cum_a.stride(1),
                cum_a.stride(2),
                H_prev.stride(0),
                H_prev.stride(1),
                H_prev.stride(2),
                H_prev.stride(3),
                Y_out.stride(0),
                Y_out.stride(1),
                Y_out.stride(2),
                Y_out.stride(3),
                Q=Q,
                R=R,
                QR=Q * R,
                Q2R=Q * R2,
                N=N,
                P=P,
            )

            ctx.save_for_backward(Xh, Bh, Cr, cum_a, chunk_decay, H_prev, h_final)
            Y_reshaped = Y_out.view(B, H, C, Q, R, P).permute(0, 2, 3, 1, 5, 4).to(X_hat.dtype)
            return Y_reshaped, h_final.view(B, H, N, P).to(h0.dtype)

        Y_total, hf, saved = _chunked_ssd_analytical_forward_and_cache(
            X_hat, B_hat, C_rot, log_alpha, h0
        )
        ctx.save_for_backward(*saved)
        return Y_total, hf

    @staticmethod
    def backward(ctx, grad_Y, grad_h_final):
        if ctx.use_triton:
            B, C, Q, R, R2, H, P, N = ctx.shapes
            dt_x, dt_b, dt_c, dt_la, dt_h0 = ctx.orig_dtypes
            Xh, Bh, Cr, cum_a, chunk_decay, H_prev, h_final = ctx.saved_tensors
            BH = B * H

            dY = (
                grad_Y.permute(0, 3, 1, 2, 5, 4)
                .reshape(BH, C, Q * R, P)
                .to(torch.bfloat16)
                .contiguous()
            )
            dh_f = (
                grad_h_final.reshape(BH, N, P).to(torch.float32).contiguous()
                if grad_h_final is not None
                else torch.zeros((BH, N, P), device=Xh.device, dtype=torch.float32)
            )

            dh_local = torch.empty((BH, C, N, P), device=Xh.device, dtype=torch.float32)
            dH_c = torch.empty((BH, C, N, P), device=Xh.device, dtype=torch.float32)
            dh0_flat = torch.empty((BH, N, P), device=Xh.device, dtype=torch.float32)
            end_boundary = torch.empty((BH, C), device=Xh.device, dtype=torch.float32)

            dCr = torch.empty((BH, C, Q * R, N), device=Xh.device, dtype=torch.float32)
            dBh = torch.empty((BH, C, Q * R2, N), device=Xh.device, dtype=torch.float32)
            dXh = torch.empty((BH, C, Q * R2, P), device=Xh.device, dtype=torch.float32)
            dla = torch.empty((BH, C, Q), device=Xh.device, dtype=torch.float32)

            _mamba3_bwd_offdiag_grad_kernel[(BH, C)](
                Cr,
                dY,
                cum_a,
                dh_local,
                Cr.stride(0),
                Cr.stride(1),
                Cr.stride(2),
                Cr.stride(3),
                dY.stride(0),
                dY.stride(1),
                dY.stride(2),
                dY.stride(3),
                cum_a.stride(0),
                cum_a.stride(1),
                cum_a.stride(2),
                dh_local.stride(0),
                dh_local.stride(1),
                dh_local.stride(2),
                dh_local.stride(3),
                R=R,
                QR=Q * R,
                N=N,
                P=P,
            )

            _mamba3_bwd_interchunk_scan_kernel[(BH,)](
                dh_local,
                chunk_decay,
                H_prev,
                h_final,
                dh_f,
                dH_c,
                dh0_flat,
                end_boundary,
                C,
                dh_local.stride(0),
                dh_local.stride(1),
                dh_local.stride(2),
                dh_local.stride(3),
                chunk_decay.stride(0),
                chunk_decay.stride(1),
                dh_f.stride(0),
                dh_f.stride(1),
                dh_f.stride(2),
                N=N,
                P=P,
            )

            _mamba3_bwd_ssd_main_kernel[(BH, C)](
                Cr,
                Bh,
                Xh,
                dY,
                cum_a,
                H_prev,
                dH_c,
                end_boundary,
                dCr,
                dBh,
                dXh,
                dla,
                Cr.stride(0),
                Cr.stride(1),
                Cr.stride(2),
                Cr.stride(3),
                Bh.stride(0),
                Bh.stride(1),
                Bh.stride(2),
                Bh.stride(3),
                Xh.stride(0),
                Xh.stride(1),
                Xh.stride(2),
                Xh.stride(3),
                dY.stride(0),
                dY.stride(1),
                dY.stride(2),
                dY.stride(3),
                cum_a.stride(0),
                cum_a.stride(1),
                cum_a.stride(2),
                H_prev.stride(0),
                H_prev.stride(1),
                H_prev.stride(2),
                H_prev.stride(3),
                end_boundary.stride(0),
                end_boundary.stride(1),
                Q=Q,
                R=R,
                QR=Q * R,
                Q2R=Q * R2,
                N=N,
                P=P,
            )

            grad_X = dXh.view(B, H, C, Q, R2, P).permute(0, 2, 3, 4, 1, 5).to(dt_x)
            grad_B = dBh.view(B, H, C, Q, R2, N).permute(0, 2, 3, 4, 1, 5).to(dt_b)
            grad_C = dCr.view(B, H, C, Q, R, N).permute(0, 2, 3, 4, 1, 5).to(dt_c)
            grad_la = dla.view(B, H, C, Q).permute(0, 2, 3, 1).to(dt_la)
            grad_h0 = dh0_flat.view(B, H, N, P).to(dt_h0)
            return grad_X, grad_B, grad_C, grad_la, grad_h0

        return _chunked_ssd_analytical_backward(
            grad_Y, grad_h_final, ctx.saved_tensors, ctx.shapes, ctx.orig_dtypes
        )


def mamba3_mimo_ssd_scan(
    X: torch.Tensor,
    B_rot: torch.Tensor,
    C_rot: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    h0: torch.Tensor,
    prev_bx0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes arbitrary-length T Mamba-3 MIMO trapezoidal scan via Q=32 SSD chunks.
    Returns: (Y_mimo [B, T, H, P, R], h_T [B, H, N, P], last_bx [B, H, N, P])
    """
    B, T, H, P, R = X.shape
    N = B_rot.size(3)

    last_bx = torch.matmul(B_rot[:, -1], X[:, -1].transpose(-1, -2))

    B_prev = torch.cat([torch.zeros_like(B_rot[:, :1]), B_rot[:, :-1]], dim=1)
    X_prev = torch.cat([torch.zeros_like(X[:, :1]), X[:, :-1]], dim=1)

    alpha_0 = torch.exp(log_alpha[:, 0]).clamp(min=1e-6).view(B, H, 1, 1)
    beta_0 = beta[:, 0].view(B, H, 1, 1)
    h0_eff = h0 + (beta_0 / alpha_0) * prev_bx0

    g_scale = gamma.view(B, T, H, 1, 1)
    b_scale = beta.clone()
    b_scale[:, 0] = 0.0
    b_scale = b_scale.view(B, T, H, 1, 1)

    B_hat_t = torch.cat([g_scale * B_rot, b_scale * B_prev], dim=-1)
    X_hat_t = torch.cat([X, X_prev], dim=-1)

    C_chunks = math.ceil(T / CHUNK_Q)
    pad_t = C_chunks * CHUNK_Q - T
    if pad_t > 0:
        X_hat_t = torch.nn.functional.pad(X_hat_t, (0, 0, 0, 0, 0, 0, 0, pad_t), value=0.0)
        B_hat_t = torch.nn.functional.pad(B_hat_t, (0, 0, 0, 0, 0, 0, 0, pad_t), value=0.0)
        C_rot_p = torch.nn.functional.pad(C_rot, (0, 0, 0, 0, 0, 0, 0, pad_t), value=0.0)
        log_alpha_p = torch.nn.functional.pad(log_alpha, (0, 0, 0, pad_t), value=0.0)
    else:
        C_rot_p = C_rot
        log_alpha_p = log_alpha

    X_chunk = X_hat_t.permute(0, 1, 4, 2, 3).view(B, C_chunks, CHUNK_Q, 2 * R, H, P)
    B_chunk = B_hat_t.permute(0, 1, 4, 2, 3).view(B, C_chunks, CHUNK_Q, 2 * R, H, N)
    C_chunk = C_rot_p.permute(0, 1, 4, 2, 3).view(B, C_chunks, CHUNK_Q, R, H, N)
    la_chunk = log_alpha_p.view(B, C_chunks, CHUNK_Q, H)

    Y_chunk, h_final = _Mamba3TritonSSDFn.apply(X_chunk, B_chunk, C_chunk, la_chunk, h0_eff)
    Y_mimo = Y_chunk.reshape(B, C_chunks * CHUNK_Q, H, P, R)[:, :T]
    return Y_mimo, h_final, last_bx
