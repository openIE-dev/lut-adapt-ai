"""Rung 2: Mamba-3 Complex-Valued MIMO Trapezoidal Recurrence & Variable-Length Streamer."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from clai.rung2.triton_mamba3_ssd import mamba3_mimo_ssd_scan


@dataclass
class Mamba3State:
    """O(1) streaming state for variable-length sequences (zero conv1d cache required)."""

    h: torch.Tensor  # [B, H, N, P] real state matrix
    prev_bx: torch.Tensor  # [B, H, N, P] prior trapezoidal B_{t-1} X_{t-1}^T term
    phase: torch.Tensor  # [B, H, N // 2] cumulative data-dependent RoPE angle


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def apply_data_dependent_rope(tensor: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Applies 2x2 block-diagonal rotation R(-angles) across state pairs (2k, 2k+1).
    tensor: [..., N, R] where N is even
    angles: [..., N // 2]
    """
    cos_a = torch.cos(angles).unsqueeze(-1)
    sin_a = torch.sin(angles).unsqueeze(-1)

    t_pairs = tensor.view(*tensor.shape[:-2], tensor.shape[-2] // 2, 2, tensor.shape[-1])
    t0, t1 = t_pairs[..., 0, :], t_pairs[..., 1, :]

    r0 = t0 * cos_a + t1 * sin_a
    r1 = -t0 * sin_a + t1 * cos_a
    return torch.stack([r0, r1], dim=-2).reshape_as(tensor)


class Mamba3ComplexMIMOBlock(nn.Module):
    """
    Single Mamba-3 layer with:
      1. Data-dependent RoPE (complex SSM isomorphism for modulo/parity state tracking)
      2. Exponential-Trapezoidal discretization (replaces short causal conv1d)
      3. Rank-R MIMO state update + BCNorm with explicit B, C biases
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_state: int = 64,
        mimo_rank: int = 4,
    ):
        super().__init__()
        assert d_model % n_heads == 0 and d_state % 2 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # P
        self.d_state = d_state  # N
        self.mimo_rank = mimo_rank  # R

        self.in_norm = RMSNorm(d_model)

        self.proj_x = nn.Linear(d_model, n_heads * self.head_dim * mimo_rank, bias=False)
        self.proj_z = nn.Linear(d_model, d_model, bias=False)
        self.proj_b = nn.Linear(d_model, n_heads * d_state * mimo_rank, bias=False)
        self.proj_c = nn.Linear(d_model, n_heads * d_state * mimo_rank, bias=False)

        self.proj_dt = nn.Linear(d_model, n_heads, bias=True)
        self.proj_lam = nn.Linear(d_model, n_heads, bias=True)
        self.proj_theta = nn.Linear(d_model, n_heads * (d_state // 2), bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32)))

        self.b_norm = RMSNorm(d_state)
        self.c_norm = RMSNorm(d_state)
        self.b_bias = nn.Parameter(torch.zeros(1, 1, n_heads, d_state, mimo_rank))
        self.c_bias = nn.Parameter(torch.zeros(1, 1, n_heads, d_state, mimo_rank))

        self.rank_down = nn.Parameter(
            torch.ones(n_heads, self.head_dim, mimo_rank) / (mimo_rank ** 0.5)
        )
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> Mamba3State:
        return Mamba3State(
            h=torch.zeros(batch_size, self.n_heads, self.d_state, self.head_dim, device=device),
            prev_bx=torch.zeros(
                batch_size, self.n_heads, self.d_state, self.head_dim, device=device
            ),
            phase=torch.zeros(batch_size, self.n_heads, self.d_state // 2, device=device),
        )

    def forward_sequence(
        self, u: torch.Tensor, state: Mamba3State | None = None
    ) -> tuple[torch.Tensor, Mamba3State]:
        """
        Dual-mode execution:
          - T == 1: O(1) single-step recurrent update for live byte-by-byte streaming.
          - T > 1 : Fused chunkwise SSD scan (`mamba3_mimo_ssd_scan`).
        """
        B, T, D = u.shape
        H, P, N, R = self.n_heads, self.head_dim, self.d_state, self.mimo_rank
        if state is None:
            state = self.init_state(B, u.device)

        x_norm = self.in_norm(u)
        X = self.proj_x(x_norm).view(B, T, H, P, R)
        Z = self.proj_z(x_norm)
        B_raw = self.proj_b(x_norm).view(B, T, H, R, N)
        C_raw = self.proj_c(x_norm).view(B, T, H, R, N)

        B_mat = self.b_norm(B_raw).permute(0, 1, 2, 4, 3) + self.b_bias
        C_mat = self.c_norm(C_raw).permute(0, 1, 2, 4, 3) + self.c_bias

        dt = F.softplus(self.proj_dt(x_norm))
        lam = torch.sigmoid(self.proj_lam(x_norm))
        d_theta = self.proj_theta(x_norm).view(B, T, H, N // 2)

        step_phase = dt.unsqueeze(-1) * d_theta
        cum_phase = state.phase.unsqueeze(1) + torch.cumsum(step_phase, dim=1)
        B_rot = apply_data_dependent_rope(B_mat, cum_phase)
        C_rot = apply_data_dependent_rope(C_mat, cum_phase)

        A = -torch.exp(self.A_log).view(1, 1, H)
        log_alpha = dt * A
        alpha = torch.exp(log_alpha)
        gamma = lam * dt
        beta = (1.0 - lam) * dt * alpha

        if T == 1:
            bx_0 = torch.matmul(B_rot[:, 0], X[:, 0].transpose(-1, -2))
            h_next = (
                alpha[:, 0].view(B, H, 1, 1) * state.h
                + beta[:, 0].view(B, H, 1, 1) * state.prev_bx
                + gamma[:, 0].view(B, H, 1, 1) * bx_0
            )
            y_mimo = (
                torch.matmul(C_rot[:, 0].transpose(-1, -2), h_next)
                .transpose(-1, -2)
                .unsqueeze(1)
            )
            last_bx = bx_0
        else:
            y_mimo, h_next, last_bx = mamba3_mimo_ssd_scan(
                X, B_rot, C_rot, log_alpha, beta, gamma, state.h, state.prev_bx
            )

        y_heads = (y_mimo * self.rank_down.view(1, 1, H, P, R)).sum(dim=-1)
        y = y_heads.reshape(B, T, D)
        gated = self.out_norm(y * F.silu(Z))
        next_state = Mamba3State(h=h_next, prev_bx=last_bx, phase=cum_phase[:, -1])
        return u + self.out_proj(gated), next_state


class Rung2Mamba3Streamer(nn.Module):
    """
    Variable-length Rung 2 streaming backbone (~3.5-6.2M parameters).
    Includes explicit modulo-16 frame slot phase embeddings so running 16-slot parity
    and byte accumulation resolve cleanly in a single streaming pass.
    """

    def __init__(
        self,
        d_mamba: int = 256,
        d_rung3: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        d_state: int = 64,
        mimo_rank: int = 4,
        n_concepts: int = 16,
        n_classes: int = 8,
    ):
        super().__init__()
        self.d_mamba = d_mamba
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.byte_emb = nn.Embedding(256, d_mamba)
        self.mod16_emb = nn.Embedding(16, d_mamba)
        self.layers = nn.ModuleList(
            [
                Mamba3ComplexMIMOBlock(
                    d_model=d_mamba, n_heads=n_heads, d_state=d_state, mimo_rank=mimo_rank
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_mamba)
        self.classifier = nn.Linear(d_mamba, n_classes)

        self.slot_queries = nn.Parameter(torch.randn(1, n_concepts, d_mamba) * 0.02)
        self.to_rung3_context = nn.Linear(d_mamba, d_rung3)
        self.rung3_goal_emb = nn.Parameter(torch.randn(1, 2, d_rung3) * 0.02)

    def init_stream_states(
        self, batch_size: int, device: torch.device
    ) -> list[Mamba3State]:
        return [layer.init_state(batch_size, device) for layer in self.layers]

    def forward_stream(
        self,
        x_bytes: torch.Tensor,
        lengths: torch.Tensor | None = None,
        states: list[Mamba3State] | None = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Mamba3State]]:
        B, T = x_bytes.shape
        if states is None:
            states = self.init_stream_states(B, x_bytes.device)
        if lengths is None:
            lengths = torch.full((B,), T, device=x_bytes.device, dtype=torch.long)

        pos_mod16 = (torch.arange(T, device=x_bytes.device) + pos_offset) % 16
        h = self.byte_emb(x_bytes.long()) + self.mod16_emb(pos_mod16).unsqueeze(0)
        next_states = []
        for layer, st in zip(self.layers, states):
            h, next_st = layer.forward_sequence(h, st)
            next_states.append(next_st)

        h = self.final_norm(h)

        last_idx = (lengths - 1).clamp(min=0)
        batch_arange = torch.arange(B, device=x_bytes.device)
        last_state = h[batch_arange, last_idx, :]
        logits = self.classifier(last_state)

        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=min(2, logits.size(-1)), dim=-1).values
        prob_margin = top2[:, 0] - top2[:, 1]

        attn_scores = torch.matmul(
            self.slot_queries.expand(B, -1, -1), h.transpose(-1, -2)
        ) / (h.size(-1) ** 0.5)

        t_idx = torch.arange(T, device=x_bytes.device).view(1, 1, T)
        valid_mask = t_idx < lengths.view(B, 1, 1)
        attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))

        pooled_slots = torch.matmul(F.softmax(attn_scores, dim=-1), h)
        s_x_rung3 = self.to_rung3_context(pooled_slots)

        return logits, prob_margin, s_x_rung3, next_states
