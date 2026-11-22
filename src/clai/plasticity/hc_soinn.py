"""Rung 2.5: Gradient-Free Hyperspherical HC-SOINN / Multi-Prototype Graph on S^{D-1}."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypersphericalSOINN(nn.Module):
    """
    Gradient-Free HC-SOINN / Multi-Prototype Nearest-Class-Mean Graph on S^{D-1}.
    Sits at Rung 2.5 (between Mamba-3 streaming slots and Rung 3 JEPA-EBRM):
      - Attaches novel classes & post-mutation basins instantaneously in O(1) without backprop.
      - Gates commits via topological cosine margin & adaptive neighbor radius tau_i.
    """

    def __init__(
        self,
        d_model: int = 512,
        max_nodes: int = 1024,
        max_edge_age: int = 50,
        prune_every: int = 256,
        default_tau: float = 0.22,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.max_edge_age = max_edge_age
        self.prune_every = prune_every
        self.default_tau = default_tau
        self.insert_steps = 0

        self.register_buffer("prototypes", torch.zeros(max_nodes, d_model))
        self.register_buffer("node_labels", torch.full((max_nodes,), -1, dtype=torch.long))
        self.register_buffer("win_counts", torch.zeros(max_nodes, dtype=torch.float32))
        self.register_buffer("node_radii", torch.full((max_nodes,), default_tau, dtype=torch.float32))
        self.register_buffer("active_mask", torch.zeros(max_nodes, dtype=torch.bool))

        self.register_buffer("adj", torch.zeros(max_nodes, max_nodes, dtype=torch.bool))
        self.register_buffer("edge_age", torch.zeros(max_nodes, max_nodes, dtype=torch.int32))

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    @staticmethod
    def pool_bridge_embedding(s_x_slots: torch.Tensor) -> torch.Tensor:
        """Maps Rung 2's [B, 16, d_model] slot context onto the unit hypersphere S^{D-1}."""
        summary = (
            0.5 * s_x_slots.mean(dim=1)
            + 0.25 * s_x_slots[:, 0, :]
            + 0.25 * s_x_slots[:, -1, :]
        )
        return F.normalize(summary, p=2, dim=-1)

    @torch.no_grad()
    def _compute_adaptive_thresholds(self, active_idx: torch.Tensor) -> torch.Tensor:
        M = active_idx.numel()
        if M == 0:
            return torch.empty(0, device=self.prototypes.device)
        if M == 1:
            return torch.full((1,), self.default_tau, device=self.prototypes.device)

        W = self.prototypes[active_idx]
        dists = (1.0 - torch.matmul(W, W.T)).clamp(min=0.0)
        sub_adj = self.adj[active_idx][:, active_idx]
        has_nbrs = sub_adj.any(dim=-1)

        nbr_dists = torch.where(sub_adj, dists, torch.zeros_like(dists))
        radii_floor = self.node_radii[active_idx]
        tau_conn = torch.maximum(nbr_dists.max(dim=-1).values, radii_floor).clamp(
            min=0.04, max=0.45
        )

        eye = torch.eye(M, dtype=torch.bool, device=W.device)
        non_self_dists = torch.where(eye, torch.full_like(dists, 2.0), dists)
        tau_iso = torch.maximum(
            0.65 * non_self_dists.min(dim=-1).values, radii_floor
        ).clamp(min=0.05, max=0.40)

        return torch.where(has_nbrs, tau_conn, tau_iso)

    @torch.no_grad()
    def query_batch(
        self, s_x_slots: torch.Tensor, margin_threshold: float = 0.10
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        active_idx = torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)
        M = active_idx.numel()
        if M < 2:
            return None, None, None

        q = self.pool_bridge_embedding(s_x_slots)
        W = self.prototypes[active_idx]
        labels = self.node_labels[active_idx]
        taus = self._compute_adaptive_thresholds(active_idx)

        dists = (1.0 - torch.matmul(q, W.T)).clamp(min=0.0)
        d_win, win_local = dists.min(dim=-1)
        pred_labels = labels[win_local]
        win_tau = taus[win_local]

        same_class_mask = labels.unsqueeze(0) == pred_labels.unsqueeze(1)
        rival_dists = torch.where(same_class_mask, torch.full_like(dists, 2.0), dists)
        d_rival = rival_dists.min(dim=-1).values

        has_rival = (~same_class_mask).any(dim=-1)
        margins = torch.where(has_rival, d_rival - d_win, torch.zeros_like(d_win))

        in_basin = d_win <= win_tau
        settled = in_basin & has_rival & (margins >= margin_threshold)
        return pred_labels, margins, settled

    @torch.no_grad()
    def clear(self) -> None:
        self.active_mask.zero_()
        self.node_labels.fill_(-1)
        self.win_counts.zero_()
        self.adj.zero_()
        self.edge_age.zero_()
        self.insert_steps = 0

    @torch.no_grad()
    def assimilate_verified_batch(
        self, s_x_slots: torch.Tensor, verified_labels: torch.Tensor
    ) -> None:
        z_batch = self.pool_bridge_embedding(s_x_slots.detach())
        for i in range(z_batch.size(0)):
            z = z_batch[i]
            y = int(verified_labels[i].item())
            self._assimilate_single(z, y)
            self.insert_steps += 1
            if self.insert_steps % self.prune_every == 0:
                self._prune_noisy_islands()

    def _alloc_node(self, z: torch.Tensor, y: int) -> int:
        free_slots = torch.nonzero(~self.active_mask, as_tuple=False).squeeze(-1)
        if free_slots.numel() > 0:
            slot = int(free_slots[0].item())
        else:
            scores = torch.where(
                self.active_mask,
                self.win_counts,
                torch.full_like(self.win_counts, float("inf")),
            )
            slot = int(scores.argmin().item())
            self._remove_node(slot)

        self.prototypes[slot] = z
        self.node_labels[slot] = y
        self.win_counts[slot] = 1.0
        self.node_radii[slot] = self.default_tau
        self.active_mask[slot] = True
        return slot

    def _remove_node(self, slot: int) -> None:
        self.active_mask[slot] = False
        self.node_labels[slot] = -1
        self.win_counts[slot] = 0.0
        self.node_radii[slot] = self.default_tau
        self.adj[slot, :] = False
        self.adj[:, slot] = False
        self.edge_age[slot, :] = 0
        self.edge_age[:, slot] = 0

    def _assimilate_single(self, z: torch.Tensor, y: int) -> None:
        active_idx = torch.nonzero(self.active_mask, as_tuple=False).squeeze(-1)
        M = active_idx.numel()

        if M == 0 or not torch.any(self.node_labels[active_idx] == y):
            self._alloc_node(z, y)
            return

        W = self.prototypes[active_idx]
        taus = self._compute_adaptive_thresholds(active_idx)
        dists = (1.0 - torch.mv(W, z)).clamp(min=0.0)

        if M >= 2:
            top2_d, top2_loc = torch.topk(dists, k=2, largest=False)
            d1, s1 = float(top2_d[0].item()), int(active_idx[top2_loc[0]].item())
            d2, s2 = float(top2_d[1].item()), int(active_idx[top2_loc[1]].item())
            tau1 = float(taus[top2_loc[0]].item())
            tau2 = float(taus[top2_loc[1]].item())
        else:
            d1, s1 = float(dists[0].item()), int(active_idx[0].item())
            d2, s2, tau2 = 2.0, -1, 0.0
            tau1 = float(taus[0].item())

        y1 = int(self.node_labels[s1].item())

        if d1 <= tau1 and y1 != y:
            self.adj[s1, :] = False
            self.adj[:, s1] = False
            self.edge_age[s1, :] = 0
            self.edge_age[:, s1] = 0
            self.node_radii[s1] = max(0.04, d1 * 0.75)
            self._alloc_node(z, y)
            return

        if d1 > tau1 or y1 != y:
            self._alloc_node(z, y)
            return

        self.win_counts[s1] += 1.0
        eta1 = 1.0 / float(self.win_counts[s1].item())
        self.node_radii[s1] = max(
            0.05, 0.85 * float(self.node_radii[s1].item()) + 0.15 * (2.2 * d1)
        )
        self.prototypes[s1] = F.normalize(
            self.prototypes[s1] + eta1 * (z - self.prototypes[s1]), p=2, dim=-1
        )

        nbr_mask = self.adj[s1]
        if torch.any(nbr_mask):
            self.edge_age[s1, nbr_mask] += 1
            self.edge_age[nbr_mask, s1] += 1
            eta_nbr = 0.01 * eta1
            nbr_w = self.prototypes[nbr_mask]
            self.prototypes[nbr_mask] = F.normalize(
                nbr_w + eta_nbr * (z.unsqueeze(0) - nbr_w), p=2, dim=-1
            )

            expired = self.edge_age[s1] > self.max_edge_age
            if torch.any(expired):
                self.adj[s1, expired] = False
                self.adj[expired, s1] = False
                self.edge_age[s1, expired] = 0
                self.edge_age[expired, s1] = 0

        if (
            s2 >= 0
            and int(self.node_labels[s2].item()) == y
            and d2 <= max(tau2, self.default_tau)
        ):
            self.adj[s1, s2] = True
            self.adj[s2, s1] = True
            self.edge_age[s1, s2] = 0
            self.edge_age[s2, s1] = 0

    def _prune_noisy_islands(self) -> None:
        if self.num_active <= 16:
            return
        has_edges = self.adj.any(dim=-1)
        noise = self.active_mask & (~has_edges) & (self.win_counts <= 1.0)
        noise_idx = torch.nonzero(noise, as_tuple=False).squeeze(-1)
        for slot in noise_idx[: noise_idx.numel() // 2]:
            self._remove_node(int(slot.item()))
