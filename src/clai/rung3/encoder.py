"""Standalone ByteStateEncoder trunk for Rung 3 direct byte-frame testing."""

from __future__ import annotations

import torch
import torch.nn as nn


class ByteStateEncoder(nn.Module):
    """Maps 16-byte input frames + goal contract token into d=512 latent context."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.byte_emb = nn.Embedding(256, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        self.goal_emb = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_bytes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_x = self.norm(self.byte_emb(x_bytes.long()) + self.pos_emb)
        g_spec = self.goal_emb.expand(x_bytes.size(0), -1, -1)
        return s_x, g_spec
