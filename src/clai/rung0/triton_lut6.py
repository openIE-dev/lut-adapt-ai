"""Fused 64-episode-per-lane uint64 bit-plane transpose, LUT6 Shannon/minterm engine, & voter."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def pack_episodes_to_uint64(x_bin: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Bit-slices [B, W] binary tensor into [C, W] int64 bit-planes (C = ceil(B / 64)).
    Bit `b` of out[c, w] stores episode `c * 64 + b` for wire `w`.
    """
    B, W = x_bin.shape
    C = (B + 63) // 64
    pad = C * 64 - B
    if pad > 0:
        x_bin = torch.nn.functional.pad(x_bin, (0, 0, 0, pad), value=0)

    bits = (x_bin > 0.5).to(torch.int64).view(C, 64, W)
    shifts = torch.arange(64, device=x_bin.device, dtype=torch.int64).view(1, 64, 1)
    packed = torch.sum(bits << shifts, dim=1)
    return packed.contiguous(), B


def unpack_uint64_to_episodes(packed: torch.Tensor, orig_b: int) -> torch.Tensor:
    """Unpacks [C, N] int64 bit-planes back to [B, N] float32 {0.0, 1.0}."""
    C, N = packed.shape
    shifts = torch.arange(64, device=packed.device, dtype=torch.int64).view(1, 64, 1)
    unpacked = ((packed.unsqueeze(1) >> shifts) & 1).reshape(C * 64, N)
    return unpacked[:orig_b].to(torch.float32)


def pack_truth_tables_to_uint64(tables_u8: torch.Tensor) -> torch.Tensor:
    """Packs [N, 64] uint8 {0, 1} truth tables into [N] int64 scalar INIT masks."""
    shifts = torch.arange(64, device=tables_u8.device, dtype=torch.int64).unsqueeze(0)
    return torch.sum((tables_u8.to(torch.int64) & 1) << shifts, dim=-1).contiguous()


if HAS_TRITON:

    @triton.jit
    def _transpose_wire_words_to_bitplanes_kernel(
        words_ptr,
        planes_ptr,
        B,
        stride_wb,
        stride_wh,
        stride_pc,
        stride_pw,
    ):
        chunk_id = tl.program_id(0)
        head_id = tl.program_id(1)

        ep_idx = chunk_id * 64 + tl.arange(0, 64)
        ep_mask = ep_idx < B

        in_ptrs = words_ptr + ep_idx * stride_wb + head_id * stride_wh
        row_words = tl.load(in_ptrs, mask=ep_mask, other=0).to(tl.uint64)

        ep_shifts = tl.arange(0, 64).to(tl.uint64)[:, None]
        wire_shifts = tl.arange(0, 64).to(tl.uint64)[None, :]

        bits_2d = (row_words[:, None] >> wire_shifts) & tl.full([1, 1], 1, dtype=tl.uint64)
        col_planes = tl.sum(bits_2d << ep_shifts, axis=0)

        wire_offs = head_id * 64 + tl.arange(0, 64)
        out_ptrs = planes_ptr + chunk_id * stride_pc + wire_offs * stride_pw
        tl.store(out_ptrs, col_planes.to(tl.int64))

    @triton.jit
    def _lut6_shannon_bitslice_kernel(
        x_ptr,
        wires_ptr,
        tables_ptr,
        out_ptr,
        C,
        in_wires,
        num_luts,
        stride_xc,
        stride_xw,
        stride_wn,
        stride_wp,
        stride_oc,
        stride_on,
        BLOCK_N: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        lut_block_id = tl.program_id(1)

        lut_offs = lut_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = lut_offs < num_luts

        w_base = wires_ptr + lut_offs * stride_wn
        idx0 = tl.load(w_base + 0 * stride_wp, mask=mask_n, other=0)
        idx1 = tl.load(w_base + 1 * stride_wp, mask=mask_n, other=0)
        idx2 = tl.load(w_base + 2 * stride_wp, mask=mask_n, other=0)
        idx3 = tl.load(w_base + 3 * stride_wp, mask=mask_n, other=0)
        idx4 = tl.load(w_base + 4 * stride_wp, mask=mask_n, other=0)
        idx5 = tl.load(w_base + 5 * stride_wp, mask=mask_n, other=0)

        x_row = x_ptr + chunk_id * stride_xc
        w0 = tl.load(x_row + idx0 * stride_xw, mask=mask_n, other=0).to(tl.uint64)
        w1 = tl.load(x_row + idx1 * stride_xw, mask=mask_n, other=0).to(tl.uint64)
        w2 = tl.load(x_row + idx2 * stride_xw, mask=mask_n, other=0).to(tl.uint64)
        w3 = tl.load(x_row + idx3 * stride_xw, mask=mask_n, other=0).to(tl.uint64)
        w4 = tl.load(x_row + idx4 * stride_xw, mask=mask_n, other=0).to(tl.uint64)
        w5 = tl.load(x_row + idx5 * stride_xw, mask=mask_n, other=0).to(tl.uint64)

        t = tl.load(tables_ptr + lut_offs, mask=mask_n, other=0).to(tl.uint64)

        out = tl.zeros([BLOCK_N], dtype=tl.uint64)
        nw0, nw1, nw2 = ~w0, ~w1, ~w2
        nw3, nw4, nw5 = ~w3, ~w4, ~w5

        for m in tl.static_range(64):
            t_bit = (t >> m) & tl.full([BLOCK_N], 1, dtype=tl.uint64)
            t_mask = tl.zeros([BLOCK_N], dtype=tl.uint64) - t_bit

            b0 = w0 if (m & 1) else nw0
            b1 = w1 if (m & 2) else nw1
            b2 = w2 if (m & 4) else nw2
            b3 = w3 if (m & 8) else nw3
            b4 = w4 if (m & 16) else nw4
            b5 = w5 if (m & 32) else nw5

            out |= t_mask & b0 & b1 & b2 & b3 & b4 & b5

        out_base = out_ptr + chunk_id * stride_oc + lut_offs * stride_on
        tl.store(out_base, out.to(tl.int64), mask=mask_n)

    @triton.jit
    def _bitplane_grouped_voter_kernel(
        planes_ptr,
        votes_ptr,
        B,
        stride_pc,
        stride_pn,
        stride_vb,
        stride_vk,
        LUTS_PER_CLASS: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        cls_id = tl.program_id(1)

        lut_offs = cls_id * LUTS_PER_CLASS + tl.arange(0, LUTS_PER_CLASS)
        plane_words = tl.load(planes_ptr + chunk_id * stride_pc + lut_offs * stride_pn).to(
            tl.uint64
        )

        ep_shifts = tl.arange(0, 64).to(tl.uint64)[:, None]
        bits = (
            (plane_words[None, :] >> ep_shifts) & tl.full([1, 1], 1, dtype=tl.uint64)
        ).to(tl.int32)
        cls_votes = tl.sum(bits, axis=1)

        ep_idx = chunk_id * 64 + tl.arange(0, 64)
        ep_mask = ep_idx < B
        tl.store(votes_ptr + ep_idx * stride_vb + cls_id * stride_vk, cls_votes, mask=ep_mask)


def transpose_engram_words_to_bitplanes(words_i64: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Converts [B, 8] int64 Engram wire-words directly into [C, 512] int64 episode bit-planes.
    Zero float32 intermediate allocation.
    """
    B, H = words_i64.shape
    C = (B + 63) // 64

    if HAS_TRITON and words_i64.is_cuda:
        planes = torch.empty((C, H * 64), device=words_i64.device, dtype=torch.int64)
        _transpose_wire_words_to_bitplanes_kernel[(C, H)](
            words_i64,
            planes,
            B,
            words_i64.stride(0),
            words_i64.stride(1),
            planes.stride(0),
            planes.stride(1),
        )
        return planes, B

    pad = C * 64 - B
    if pad > 0:
        words_i64 = torch.nn.functional.pad(words_i64, (0, 0, 0, pad), value=0)
    w_chunk = words_i64.view(C, 64, H)
    wire_shifts = torch.arange(64, device=words_i64.device, dtype=torch.int64).view(1, 1, 1, 64)
    ep_shifts = torch.arange(64, device=words_i64.device, dtype=torch.int64).view(1, 64, 1, 1)
    bits = (w_chunk.unsqueeze(-1) >> wire_shifts) & 1
    planes = torch.sum(bits << ep_shifts, dim=1).reshape(C, H * 64)
    return planes.contiguous(), B


def eval_lut6_bitpacked(
    x_packed: torch.Tensor,
    compiled_wires_i32: torch.Tensor,
    packed_tables_i64: torch.Tensor,
) -> torch.Tensor:
    """
    Executes one LUT6 layer on bit-packed [C, in_wires] int64 bit-planes -> [C, num_luts] int64.
    Uses fused Triton CUDA kernel when on CUDA, or factored 8x8 bitwise decoder on MPS/CPU
    (evaluating all 64 minterms in 8 vectorized passes instead of 64 sequential loops).
    """
    C, in_wires = x_packed.shape
    num_luts = compiled_wires_i32.size(0)

    if HAS_TRITON and x_packed.is_cuda:
        out = torch.empty((C, num_luts), device=x_packed.device, dtype=torch.int64)
        BLOCK_N = 128
        grid = (C, triton.cdiv(num_luts, BLOCK_N))
        _lut6_shannon_bitslice_kernel[grid](
            x_packed,
            compiled_wires_i32,
            packed_tables_i64,
            out,
            C,
            in_wires,
            num_luts,
            x_packed.stride(0),
            x_packed.stride(1),
            compiled_wires_i32.stride(0),
            compiled_wires_i32.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=BLOCK_N,
        )
        return out

    # Factored 8x8 Two-Half Decoder in pure int64 bit-planes (bit-identical across all 64 lanes)
    selected = x_packed[:, compiled_wires_i32.long()]  # [C, N, 6] int64
    w0, w1, w2, w3, w4, w5 = selected.unbind(dim=-1)
    nw0, nw1, nw2, nw3, nw4, nw5 = ~w0, ~w1, ~w2, ~w3, ~w4, ~w5

    # Low 3-wire 1-of-8 disjoint minterm planes: [C, N, 8] int64
    d_lo = torch.stack(
        [
            nw0 & nw1 & nw2,
            w0 & nw1 & nw2,
            nw0 & w1 & nw2,
            w0 & w1 & nw2,
            nw0 & nw1 & w2,
            w0 & nw1 & w2,
            nw0 & w1 & w2,
            w0 & w1 & w2,
        ],
        dim=-1,
    )
    # High 3-wire 1-of-8 disjoint minterm planes: [C, N, 8] int64
    d_hi = [
        nw3 & nw4 & nw5,
        w3 & nw4 & nw5,
        nw3 & w4 & nw5,
        w3 & w4 & nw5,
        nw3 & nw4 & w5,
        w3 & nw4 & w5,
        nw3 & w4 & w5,
        w3 & w4 & w5,
    ]

    # Expand truth table [N] -> [1, N, 8_hi, 8_lo] all-ones (-1) or all-zeros (0) int64 masks
    shifts_64 = torch.arange(64, device=x_packed.device, dtype=torch.int64).view(1, 64)
    t_masks = (-((packed_tables_i64.unsqueeze(-1) >> shifts_64) & 1)).view(1, num_luts, 8, 8)

    # Because d_lo's 8 minterms are strictly mutually exclusive per bit-lane, summing masked d_lo
    # across dim=-1 has zero inter-bit carries and equals exact bitwise OR!
    lo_active = (d_lo.unsqueeze(2) & t_masks).sum(dim=-1)  # [C, N, 8_hi] int64
    out = (
        (lo_active[:, :, 0] & d_hi[0])
        | (lo_active[:, :, 1] & d_hi[1])
        | (lo_active[:, :, 2] & d_hi[2])
        | (lo_active[:, :, 3] & d_hi[3])
        | (lo_active[:, :, 4] & d_hi[4])
        | (lo_active[:, :, 5] & d_hi[5])
        | (lo_active[:, :, 6] & d_hi[6])
        | (lo_active[:, :, 7] & d_hi[7])
    )
    return out


def count_votes_from_bitplanes(
    final_planes_i64: torch.Tensor, orig_b: int, num_classes: int
) -> torch.Tensor:
    """Computes [B, num_classes] int32 votes directly from [C, num_luts] int64 bit-planes."""
    C, num_luts = final_planes_i64.shape
    luts_per_class = num_luts // num_classes

    if HAS_TRITON and final_planes_i64.is_cuda:
        votes = torch.empty(
            (orig_b, num_classes), device=final_planes_i64.device, dtype=torch.int32
        )
        _bitplane_grouped_voter_kernel[(C, num_classes)](
            final_planes_i64,
            votes,
            orig_b,
            final_planes_i64.stride(0),
            final_planes_i64.stride(1),
            votes.stride(0),
            votes.stride(1),
            LUTS_PER_CLASS=luts_per_class,
        )
        return votes

    shifts = torch.arange(64, device=final_planes_i64.device, dtype=torch.int64).view(1, 64, 1)
    bits = ((final_planes_i64.unsqueeze(1) >> shifts) & 1).reshape(C * 64, num_luts)[:orig_b]
    active_luts = num_classes * luts_per_class
    return (
        bits[:, :active_luts]
        .view(orig_b, num_classes, luts_per_class)
        .sum(dim=-1)
        .to(torch.int32)
    )
