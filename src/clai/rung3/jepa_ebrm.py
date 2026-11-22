"""Rung 3: ~25M-parameter NCM + JEPA + EBRM continuous latent reasoning & verification core."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentJEPA_EBRM(nn.Module):
    """
    ~25M-parameter NCM + JEPA + EBRM continuous latent reasoning & verification core.

    Fixes the zero-vector collapse bug from the raw prototype by anchoring valid class valleys
    to non-zero orthogonal concept prototypes `class_basins[y]` + context shift, training the
    JEPA predictor to propose initial basin coordinates and EBRM to carve deep energy valleys
    around NCM-quantized concept codes while pushing origin/rival states above margin 1.0.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_concepts: int = 16,
        n_levels: int = 4,
        n_classes: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_concepts = n_concepts
        self.n_levels = n_levels
        self.n_classes = n_classes

        # 1. NCM Multiscale Residual Vector Quantizer (4 levels x 1024 codes)
        self.rvq_codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(1024, d_model) * 0.08) for _ in range(n_levels)]
        )

        # Non-zero separated concept valley anchors per class (prevents z=0 trivial collapse)
        init_basins = F.normalize(torch.randn(n_classes, n_concepts, d_model), p=2, dim=-1) * 0.8
        self.class_basins = nn.Parameter(init_basins)

        # 2. JEPA Predictor (6 Transformer layers, dropout=0.0 for deterministic energy surface & MPS SDPA)
        jepa_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.jepa_predictor = nn.TransformerEncoder(
            jepa_layer, num_layers=6, enable_nested_tensor=False
        )

        # 3. EBRM Global Scalar Energy Verifier (6 Transformer layers, dropout=0.0)
        ebrm_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1536,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.ebrm_trunk = nn.TransformerEncoder(
            ebrm_layer, num_layers=6, enable_nested_tensor=False
        )
        self.energy_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

        # Direct structured action readout from settled latent valley
        self.action_head = nn.Linear(d_model, n_classes)

    def target_basin(self, s_x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Constructs non-degenerate target concept coordinates z*(s_x, y) != 0."""
        anchors = self.class_basins[labels]  # [B, n_concepts, D]
        ctx_shift = 0.25 * F.normalize(s_x.detach(), p=2, dim=-1)
        return anchors + ctx_shift

    def quantize_ncm(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Straight-through Residual Vector Quantization returning (z_quant, codes, commit_loss)."""
        residual = z
        z_quant = torch.zeros_like(z)
        codes = []
        commit_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        for cb in self.rvq_codebooks:
            idx = torch.cdist(residual, cb).argmin(dim=-1)
            q = F.embedding(idx, cb)
            commit_loss = commit_loss + F.mse_loss(q, residual.detach())
            z_quant = z_quant + (q - residual).detach() + residual
            residual = residual - q.detach()
            codes.append(idx)
        return z_quant, torch.stack(codes, dim=-1), commit_loss / len(self.rvq_codebooks)

    def propose_initial_z(self, s_x: torch.Tensor, g_spec: torch.Tensor) -> torch.Tensor:
        """Uses JEPA predictor to propose warm initial latent concept state z_0 before settling."""
        z_seed = torch.zeros(
            s_x.size(0), self.n_concepts, s_x.size(-1), device=s_x.device, dtype=s_x.dtype
        )
        seq = torch.cat([s_x, g_spec, z_seed], dim=1)
        return 0.5 * self.jepa_predictor(seq)[:, -self.n_concepts :, :]

    def free_energy_and_logits(
        self, s_x: torch.Tensor, z: torch.Tensor, g_spec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, ncm_codes, _ = self.quantize_ncm(z)
        seq = torch.cat([s_x, g_spec, z_q], dim=1)

        z_pred = self.jepa_predictor(seq)[:, -self.n_concepts :, :]
        jepa_mse = F.mse_loss(z_pred, z_q.detach(), reduction="none").mean(dim=(1, 2))

        ebrm_h = self.ebrm_trunk(seq)
        ebrm_e = self.energy_proj(ebrm_h).mean(dim=1).squeeze(-1)
        action_logits = self.action_head(ebrm_h[:, -self.n_concepts :, :].mean(dim=1))

        return ebrm_e + 0.5 * jepa_mse, action_logits, ncm_codes

    @torch.enable_grad()
    def settle_batch(
        self,
        s_x: torch.Tensor,
        g_spec: torch.Tensor,
        max_steps: int = 16,
        step_lr: float = 0.12,
        tau_commit: float = 0.22,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Batched continuous latent relaxation over [B_r3, 16, d_model] with per-lane satiation freeze.
        Starts from a warm JEPA proposal and descends -grad_z F(s_x, z, g) into the NCM energy basin.
        Returns: (preds [B_r3], energies [B_r3], settled_mask [B_r3], steps_taken)
        """
        B_sub = s_x.size(0)
        s_x_det = s_x.detach()
        g_det = g_spec.detach()
        with torch.no_grad():
            z_init = self.propose_initial_z(s_x_det, g_det)
        z = z_init.detach().requires_grad_(True)
        steps_run = 0

        for step in range(max_steps):
            steps_run = step + 1
            energy, logits, _ = self.free_energy_and_logits(s_x_det, z, g_det)
            active = energy > tau_commit

            if not torch.any(active):
                return logits.argmax(dim=-1).detach(), energy.detach(), ~active, steps_run

            grad = torch.autograd.grad(energy.sum(), z)[0]
            # Normalize gradient per concept slot for stable step sizes across energy cliffs
            grad_norm = grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)
            step_mask = active.to(z.dtype).view(B_sub, 1, 1)
            z = (z - step_lr * step_mask * grad_norm).detach().requires_grad_(True)

        energy, logits, _ = self.free_energy_and_logits(s_x_det, z, g_det)
        settled_mask = energy.detach() <= tau_commit
        return logits.argmax(dim=-1).detach(), energy.detach(), settled_mask, steps_run

    @torch.enable_grad()
    def settle_and_propose(
        self,
        s_x: torch.Tensor,
        g_spec: torch.Tensor,
        max_steps: int = 16,
        step_lr: float = 0.12,
        tau_commit: float = 0.22,
    ) -> tuple[int, float, torch.Tensor, int, bool]:
        preds, energies, settled_mask, steps = self.settle_batch(
            s_x, g_spec, max_steps=max_steps, step_lr=step_lr, tau_commit=tau_commit
        )
        _, _, codes = self.free_energy_and_logits(s_x.detach(), self.propose_initial_z(s_x.detach(), g_spec.detach()), g_spec.detach())
        return (
            int(preds[0].item()),
            float(energies[0].item()),
            codes.detach(),
            steps,
            bool(settled_mask[0].item()),
        )
