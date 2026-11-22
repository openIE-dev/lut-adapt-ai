"""DiffLUT-Net: Annealed Straight-Through Multilinear LUT6 Relaxation + Bit-Packed Hardware Fabric."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from clai.rung0.triton_lut6 import (
    count_votes_from_bitplanes,
    eval_lut6_bitpacked,
    pack_episodes_to_uint64,
    pack_truth_tables_to_uint64,
    transpose_engram_words_to_bitplanes,
    unpack_uint64_to_episodes,
)


class DiffLUT6Layer(nn.Module):
    """
    Hardware-native 6-input Lookup Table layer.
    - Sleep Compiler: Straight-through Gumbel-Softmax 6-wire routing + exact 64-corner multilinear
      polynomial relaxation with sharpened table logits so snapped truth tables match training.
    - Rung 0 Runtime: Zero-float 64-episode-per-lane uint64 bit-plane lookup.
    """

    def __init__(self, in_wires: int, num_luts: int):
        super().__init__()
        self.in_wires = in_wires
        self.num_luts = num_luts

        self.wire_logits = nn.Parameter(torch.randn(num_luts, 6, in_wires) * 0.15)

        idx = torch.arange(64, dtype=torch.long)
        basis = torch.stack([(idx >> p) & 1 for p in range(6)], dim=-1).float()
        self.register_buffer("basis", basis)  # [64, 6]
        self.register_buffer("bit_weights", 2 ** torch.arange(6, dtype=torch.long))

        # Residual passthrough initialization (DiffLogicNet): bias truth table toward Port 0 (I0)
        # so multi-layer gradients propagate with O(1) gain instead of vanishing as (1/64)^L.
        passthrough = (2.0 * basis[:, 0] - 1.0).unsqueeze(0) * 1.1
        self.table_logits = nn.Parameter(passthrough + torch.randn(num_luts, 64) * 0.25)

        self.register_buffer("compiled_wires", torch.zeros(num_luts, 6, dtype=torch.int32))
        self.register_buffer("compiled_tables", torch.zeros(num_luts, 64, dtype=torch.uint8))
        self.register_buffer("packed_tables_i64", torch.zeros(num_luts, dtype=torch.int64))
        self.is_compiled = False

    def forward_relaxation(
        self, x: torch.Tensor, tau: float = 0.5, hard_routing: bool = False
    ) -> torch.Tensor:
        """Differentiable multilinear relaxation over x in [0, 1]^{B x in_wires}."""
        if hard_routing:
            # Straight-through binarization of incoming wires + discrete routing so every layer
            # trains on the EXACT 6-bit addresses executed after snap_to_hardware()!
            x_bin_st = (x > 0.5).to(x.dtype) - x.detach() + x
            soft_w = F.softmax(self.wire_logits / max(tau, 0.1), dim=-1)
            idx = soft_w.argmax(dim=-1)
            hard_w = F.one_hot(idx, num_classes=self.in_wires).to(soft_w.dtype)
            weights = (hard_w - soft_w).detach() + soft_w
            u = torch.einsum("b w, n p w -> b n p", x_bin_st, weights)
        else:
            weights = F.gumbel_softmax(self.wire_logits, tau=tau, hard=True, dim=-1)
            u = torch.einsum("b w, n p w -> b n p", x, weights)

        m = self.basis.view(1, 1, 64, 6)
        u_exp = u.unsqueeze(2)
        corner_probs = (m * u_exp + (1.0 - m) * (1.0 - u_exp)).prod(dim=-1)  # [B, N, 64]

        t_prob = torch.sigmoid(self.table_logits / max(tau, 0.2)).unsqueeze(0)
        if hard_routing:
            t_hard = (self.table_logits > 0.0).to(t_prob.dtype).unsqueeze(0)
            t_prob = (t_hard - t_prob).detach() + t_prob

        return (corner_probs * t_prob).sum(dim=-1)  # [B, N]

    @torch.no_grad()
    def snap_to_hardware(self) -> None:
        """Snaps continuous logits into int32 wire muxes and packed 64-bit LUT6 INIT masks."""
        self.compiled_wires.copy_(self.wire_logits.argmax(dim=-1).to(torch.int32))
        self.compiled_tables.copy_((self.table_logits > 0.0).to(torch.uint8))
        self.packed_tables_i64.copy_(pack_truth_tables_to_uint64(self.compiled_tables))
        self.is_compiled = True

    @torch.no_grad()
    def forward_hardware_packed(self, x_packed: torch.Tensor) -> torch.Tensor:
        """Bit-packed fast path: [C, in_wires] int64 -> [C, num_luts] int64 (64 episodes/lane)."""
        return eval_lut6_bitpacked(x_packed, self.compiled_wires, self.packed_tables_i64)

    @torch.no_grad()
    def forward_hardware(self, x_bin: torch.Tensor) -> torch.Tensor:
        """Unpacked compatibility wrapper executing via the 64-lane uint64 bit-plane kernel."""
        x_packed, orig_b = pack_episodes_to_uint64(x_bin)
        out_packed = self.forward_hardware_packed(x_packed)
        return unpack_uint64_to_episodes(out_packed, orig_b)

    @torch.no_grad()
    def forward_discrete_reference(self, x_bin: torch.Tensor) -> torch.Tensor:
        """Direct unpacked 6-bit table-gather reference for bit-exact parity verification."""
        b_int = (x_bin > 0.5).long()
        selected = b_int[:, self.compiled_wires.long()]  # [B, N, 6]
        addr = (selected * self.bit_weights.view(1, 1, 6)).sum(dim=-1)  # [B, N] in 0..63
        lut_ids = torch.arange(self.num_luts, device=x_bin.device).unsqueeze(0)
        return self.compiled_tables[lut_ids, addr].float()


class DiffLUTNetwork(nn.Module):
    """Multi-layer LUT6 fabric chained purely in bit-packed uint64 bit-planes across layers."""

    def __init__(
        self, bus_bits: int = 512, width: int = 512, num_layers: int = 3, num_classes: int = 8
    ):
        super().__init__()
        self.bus_bits = bus_bits
        self.width = width
        self.num_layers = num_layers
        self.num_classes = num_classes
        dims = [bus_bits] + [width] * num_layers
        self.layers = nn.ModuleList(
            [DiffLUT6Layer(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def resize_num_classes(self, new_num_classes: int) -> None:
        """Re-partitions final LUT6 layer voter groups when open-world classes crystallize."""
        if new_num_classes > self.num_classes and new_num_classes <= self.width // 8:
            self.num_classes = new_num_classes

    def forward_train(
        self, x_float: torch.Tensor, tau: float = 0.5, hard_routing: bool = False
    ) -> torch.Tensor:
        h = x_float
        for layer in self.layers:
            h = layer.forward_relaxation(h, tau=tau, hard_routing=hard_routing)
        luts_per_class = self.width // self.num_classes
        active = self.num_classes * luts_per_class
        return h[:, :active].view(x_float.size(0), self.num_classes, luts_per_class).sum(dim=-1)

    @torch.no_grad()
    def compile_all(self) -> None:
        for layer in self.layers:
            layer.snap_to_hardware()

    @torch.no_grad()
    def invalidate(self) -> None:
        for layer in self.layers:
            layer.is_compiled = False

    @torch.no_grad()
    def forward_packed_chain(self, x_packed: torch.Tensor) -> torch.Tensor:
        h_packed = x_packed
        for layer in self.layers:
            h_packed = layer.forward_hardware_packed(h_packed)
        return h_packed

    @torch.no_grad()
    def infer_from_engram_words(
        self, words_i64: torch.Tensor, margin_threshold: int = 5
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Zero-float32, zero-repack pipeline:
        [B, 8] int64 Engram words -> [C, 512] bit-planes -> LUT6 layers -> [B, K] int32 votes.
        """
        if not self.layers[0].is_compiled:
            return None, None, None

        h_planes, orig_b = transpose_engram_words_to_bitplanes(words_i64)
        h_planes = self.forward_packed_chain(h_planes)

        votes = count_votes_from_bitplanes(h_planes, orig_b, self.num_classes)
        top2_vals, top2_idx = torch.topk(votes, k=2, dim=-1)
        margins = top2_vals[:, 0] - top2_vals[:, 1]
        preds = top2_idx[:, 0]
        return preds, margins, (margins >= margin_threshold)

    @torch.no_grad()
    def infer_batch_with_margin(
        self, x_bin: torch.Tensor, margin_threshold: int = 5
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.layers[0].is_compiled:
            return None, None, None

        x_packed, orig_b = pack_episodes_to_uint64(x_bin)
        h_packed = self.forward_packed_chain(x_packed)
        votes = count_votes_from_bitplanes(h_packed, orig_b, self.num_classes)
        top2_vals, top2_idx = torch.topk(votes, k=2, dim=-1)
        margins = top2_vals[:, 0] - top2_vals[:, 1]
        preds = top2_idx[:, 0]
        return preds, margins, (margins >= margin_threshold)

    @torch.no_grad()
    def infer_with_margin(
        self, x_bin: torch.Tensor, margin_threshold: int = 5
    ) -> tuple[int | None, int, bool]:
        preds, margins, ok = self.infer_batch_with_margin(x_bin, margin_threshold=margin_threshold)
        if preds is None or margins is None or ok is None:
            return None, 0, False
        return int(preds[0].item()), int(margins[0].item()), bool(ok[0].item())
