"""MixtureOfLimitsRuntime: 64-Episode Dynamic Escalation Compaction across Rung 0 -> 2 -> 2.5 -> 3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from clai.compiler.sleep_crystallizer import SleepCrystallizer
from clai.contracts.joule_meter import meter_joules
from clai.contracts.rce_gate import CommitResult, MicroBatchReceipt, ModularConstraintEnv
from clai.plasticity.hc_soinn import HypersphericalSOINN
from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus, pad_ragged_episodes
from clai.rung2.mamba3_mimo import Rung2Mamba3Streamer
from clai.rung3.jepa_ebrm import LatentJEPA_EBRM


class MixtureOfLimitsRuntime:
    """
    4-Rung Thermodynamic Compute Cascade with Metered Commit Gate:
      - Rung 0   (Nanoseconds / Microjoules): Native uint64 Engram -> 64-lane LUT6 bit-planes
      - Rung 2   (Microseconds / Millijoules): Mamba-3 Complex MIMO Chunkwise SSD Streamer
      - Rung 2.5 (Microseconds / Zero-Grad):   Hyperspherical HC-SOINN Continual Prototype Graph
      - Rung 3   (Milliseconds / Joules):      25M NCM + JEPA + EBRM Iterative Basin Settling
    """

    def __init__(self, cfg: dict, env: ModularConstraintEnv, device: torch.device):
        self.cfg = cfg
        self.env = env
        self.device = device

        self.engram = EngramBitBus(
            num_buckets=cfg["rung0"]["engram_buckets"],
            seed=cfg["experiment"]["seed"],
        )
        self.lut_net = DiffLUTNetwork(
            bus_bits=cfg["rung0"]["bus_bits"],
            width=cfg["rung0"]["lut_width"],
            num_layers=cfg["rung0"]["lut_layers"],
            num_classes=env.n_classes,
        ).to(device)

        r2_cfg = cfg["rung2"]
        self.rung2 = Rung2Mamba3Streamer(
            d_mamba=r2_cfg["d_mamba"],
            d_rung3=cfg["rung3"]["d_model"],
            n_layers=r2_cfg["n_layers"],
            n_heads=r2_cfg["n_heads"],
            d_state=r2_cfg["d_state"],
            mimo_rank=r2_cfg["mimo_rank"],
            n_concepts=cfg["rung3"]["n_concepts"],
            n_classes=env.n_classes,
        ).to(device)

        p_cfg = cfg.get("plasticity", {})
        self.soinn = HypersphericalSOINN(
            d_model=cfg["rung3"]["d_model"],
            max_nodes=p_cfg.get("max_nodes", 1024),
            max_edge_age=p_cfg.get("max_edge_age", 50),
        ).to(device)
        self.soinn_margin = p_cfg.get("cosine_margin_threshold", 0.10)

        self.ebrm = LatentJEPA_EBRM(
            d_model=cfg["rung3"]["d_model"],
            n_concepts=cfg["rung3"]["n_concepts"],
            n_levels=cfg["rung3"]["n_rvq_levels"],
            n_classes=env.n_classes,
        ).to(device)

        self.upper_opt = torch.optim.AdamW(
            list(self.rung2.parameters()) + list(self.ebrm.parameters()),
            lr=8e-4,
            weight_decay=1e-4,
        )

        self.crystallizer = SleepCrystallizer(
            self.engram,
            self.lut_net,
            device,
            lr=cfg["compiler"]["lr"],
            epochs=cfg["compiler"]["epochs"],
            min_agreement=cfg["compiler"].get("min_holdout_agreement", 0.80),
        )
        self.rung0_rejection_accum = 0

    def step_microbatch(self, episodes: list[np.ndarray]) -> MicroBatchReceipt:
        B = len(episodes)
        j_r0 = j_r2 = j_r3 = 0.0
        n_r0 = n_r2 = n_soinn = n_r3 = n_refused = 0

        # =====================================================================
        # RUNG 0: 64-Lane Native uint64 Engram -> Bit-Sliced LUT6 (C = 1 chunk)
        # =====================================================================
        with meter_joules(mode="rung0", device=self.device) as m0:
            words_u64 = self.engram.fetch_batch_words_u64(episodes, self.device)
            r0_preds, _, r0_margin_ok = self.lut_net.infer_from_engram_words(
                words_u64, margin_threshold=self.cfg["rung0"]["margin_threshold"]
            )
        j_r0 += m0.total_joules

        escalate_to_r2 = np.ones(B, dtype=bool)

        if r0_preds is not None and r0_margin_ok is not None:
            cand_mask = r0_margin_ok.cpu().numpy()
            if np.any(cand_mask):
                cand_idx = np.nonzero(cand_mask)[0]
                cand_acts = r0_preds[cand_mask].cpu().numpy()
                verified_r0 = self.env.verify_subset(episodes, cand_idx, cand_acts)

                satiated_idx = cand_idx[verified_r0]
                rejected_idx = cand_idx[~verified_r0]

                n_r0 = len(satiated_idx)
                escalate_to_r2[satiated_idx] = False

                if len(rejected_idx) > 2 * len(satiated_idx) and len(rejected_idx) >= 3:
                    self.rung0_rejection_accum += len(rejected_idx)
                    if self.rung0_rejection_accum >= 5:
                        # Schema-mutation circuit breaker: flush stale LUT6, ring buffer & SOINN
                        self.crystallizer.clear_stale_traces()
                        self.soinn.clear()
                        self.rung0_rejection_accum = 0
                elif len(satiated_idx) >= len(rejected_idx):
                    self.rung0_rejection_accum = 0

        r2_idx = np.nonzero(escalate_to_r2)[0]
        if len(r2_idx) == 0:
            return MicroBatchReceipt(B, n_r0, n_r0, 0, 0, 0, 0, j_r0, 0.0, 0.0)

        # =====================================================================
        # RUNG 2 + RUNG 2.5 BRIDGE: Mamba-3 SSD + Gradient-Free HC-SOINN Graph
        # =====================================================================
        with meter_joules(mode="rung2", device=self.device) as m2:
            r2_episodes = [episodes[i] for i in r2_idx]
            x_pad_np, len_np = pad_ragged_episodes(r2_episodes, multiple_of=32)
            x_r2 = torch.from_numpy(x_pad_np).to(self.device)
            len_r2 = torch.from_numpy(len_np).to(self.device)

            with torch.no_grad():
                r2_logits, r2_margins, s_x_r2, _ = self.rung2.forward_stream(
                    x_r2, lengths=len_r2
                )
                r2_preds = r2_logits.argmax(dim=-1)
                r2_conf_mask = (
                    r2_margins >= self.cfg["rung2"]["prob_margin_threshold"]
                ).cpu().numpy()

            escalate_local_r3 = np.ones(len(r2_idx), dtype=bool)

            # 2A. Parametric Mamba-3 Settlements
            if np.any(r2_conf_mask):
                local_cand = np.nonzero(r2_conf_mask)[0]
                global_cand = r2_idx[local_cand]
                verified_r2 = self.env.verify_subset(
                    episodes, global_cand, r2_preds[r2_conf_mask].cpu().numpy()
                )

                settled_local = local_cand[verified_r2]
                settled_global = global_cand[verified_r2]
                if len(settled_global) > 0:
                    n_r2 = len(settled_global)
                    escalate_local_r3[settled_local] = False

                    st_loc_t = torch.from_numpy(settled_local).to(self.device)
                    st_glb_t = torch.from_numpy(settled_global).to(self.device)
                    acts_t = r2_preds[st_loc_t]

                    self.soinn.assimilate_verified_batch(s_x_r2[st_loc_t], acts_t)
                    in_lut_bounds = acts_t < self.lut_net.num_classes
                    if torch.any(in_lut_bounds):
                        raw_sub = [
                            episodes[int(g)]
                            for g, ok in zip(settled_global, in_lut_bounds.cpu().numpy())
                            if ok
                        ]
                        self.crystallizer.record_batch_words(
                            words_u64[st_glb_t[in_lut_bounds]],
                            acts_t[in_lut_bounds],
                            raw_episodes=raw_sub,
                        )

            # 2B. RUNG 2.5 BRIDGE: Gradient-Free HC-SOINN Prototype Graph on Unsettled Lanes
            rem_after_r2 = np.nonzero(escalate_local_r3)[0]
            if len(rem_after_r2) > 0:
                rem_t = torch.from_numpy(rem_after_r2).to(self.device)
                soinn_preds, _, soinn_ok = self.soinn.query_batch(
                    s_x_r2[rem_t], margin_threshold=self.soinn_margin
                )
                if soinn_preds is not None and soinn_ok is not None:
                    s_mask = soinn_ok.cpu().numpy()
                    if np.any(s_mask):
                        sub_cand = rem_after_r2[s_mask]
                        glb_cand = r2_idx[sub_cand]
                        cand_acts = soinn_preds[s_mask].cpu().numpy()
                        ver_soinn = self.env.verify_subset(episodes, glb_cand, cand_acts)

                        ok_sub = sub_cand[ver_soinn]
                        ok_glb = glb_cand[ver_soinn]
                        if len(ok_glb) > 0:
                            n_soinn = len(ok_glb)
                            escalate_local_r3[ok_sub] = False

                            ok_sub_t = torch.from_numpy(ok_sub).to(self.device)
                            ok_glb_t = torch.from_numpy(ok_glb).to(self.device)
                            ok_acts_t = torch.from_numpy(cand_acts[ver_soinn]).to(
                                self.device, dtype=torch.long
                            )

                            self.soinn.assimilate_verified_batch(
                                s_x_r2[ok_sub_t], ok_acts_t
                            )
                            in_lut_bounds = ok_acts_t < self.lut_net.num_classes
                            if torch.any(in_lut_bounds):
                                raw_sub = [
                                    episodes[int(g)]
                                    for g, ok in zip(ok_glb, in_lut_bounds.cpu().numpy())
                                    if ok
                                ]
                                self.crystallizer.record_batch_words(
                                    words_u64[ok_glb_t[in_lut_bounds]],
                                    ok_acts_t[in_lut_bounds],
                                    raw_episodes=raw_sub,
                                )
        j_r2 += m2.total_joules

        local_r3 = np.nonzero(escalate_local_r3)[0]
        if len(local_r3) == 0:
            return MicroBatchReceipt(
                B, n_r0 + n_r2 + n_soinn, n_r0, n_r2, n_soinn, 0, 0, j_r0, j_r2, 0.0
            )

        # =====================================================================
        # RUNG 3: Compacted 25M JEPA-EBRM Iterative Settling (B_r3 <= B_r2)
        # =====================================================================
        with meter_joules(mode="rung3", device=self.device) as m3:
            local_r3_t = torch.from_numpy(local_r3).to(self.device)
            global_r3_idx = r2_idx[local_r3]

            s_x_r3 = s_x_r2[local_r3_t]
            g_spec = self.rung2.rung3_goal_emb.expand(s_x_r3.size(0), -1, -1)

            r3_preds, _, r3_settled_mask, _ = self.ebrm.settle_batch(
                s_x_r3,
                g_spec,
                max_steps=self.cfg["rung3"]["max_settle_steps"],
                step_lr=self.cfg["rung3"]["step_lr"],
                tau_commit=self.cfg["rung3"]["tau_commit"],
            )

            r3_acts_np = r3_preds.cpu().numpy()
            r3_settled_np = r3_settled_mask.cpu().numpy()
            verified_r3 = self.env.verify_subset(episodes, global_r3_idx, r3_acts_np)

            accepted_r3 = r3_settled_np & verified_r3
            n_r3 = int(accepted_r3.sum())
            n_refused = len(global_r3_idx) - n_r3

            gt_r3 = torch.tensor(
                [self.env.ground_truth(episodes[int(idx)]) for idx in global_r3_idx],
                device=self.device,
                dtype=torch.long,
            )

            # 1. Instantaneous gradient-free attachment into HC-SOINN (handles ANY novel class y >= K!)
            self.soinn.assimilate_verified_batch(s_x_r3, gt_r3)

            # 2. Parametric gradient shaping on known class subset
            known_mask = gt_r3 < self.ebrm.n_classes
            if torch.any(known_mask):
                self._adapt_upper_rungs_batch(
                    x_r2[local_r3_t[known_mask]],
                    len_r2[local_r3_t[known_mask]],
                    gt_r3[known_mask],
                )

            in_lut_bounds = gt_r3 < self.lut_net.num_classes
            if torch.any(in_lut_bounds):
                global_r3_t = torch.from_numpy(global_r3_idx).to(self.device)
                raw_sub = [
                    episodes[int(g)]
                    for g, ok in zip(global_r3_idx, in_lut_bounds.cpu().numpy())
                    if ok
                ]
                self.crystallizer.record_batch_words(
                    words_u64[global_r3_t[in_lut_bounds]],
                    gt_r3[in_lut_bounds],
                    raw_episodes=raw_sub,
                )
        j_r3 += m3.total_joules

        total_verified = n_r0 + n_r2 + n_soinn + n_r3
        return MicroBatchReceipt(
            B, total_verified, n_r0, n_r2, n_soinn, n_r3, n_refused, j_r0, j_r2, j_r3
        )

    def _adapt_upper_rungs_batch(
        self, x_sub: torch.Tensor, len_sub: torch.Tensor, gt_targets: torch.Tensor
    ) -> None:
        """
        Batched contrastive energy shaping across escalated streams:
          - Carves energy valleys at non-zero class concept anchors z*(s_x, y_gt) -> 0
          - Pushes origin z=0 and rival class anchors above contrastive energy thresholds
          - Trains JEPA initial proposal + RVQ codebook commitment + Mamba-3 & EBRM classifiers
        """
        self.upper_opt.zero_grad()
        r2_logits, _, s_x, _ = self.rung2.forward_stream(x_sub, lengths=len_sub)
        B_sub = s_x.size(0)
        g_spec = self.rung2.rung3_goal_emb.expand(B_sub, -1, -1)

        z_target = self.ebrm.target_basin(s_x, gt_targets)
        z_pos = z_target + 0.03 * torch.randn_like(z_target)

        wrong_labels = (gt_targets + torch.randint(1, self.ebrm.n_classes, (B_sub,), device=self.device)) % self.ebrm.n_classes
        z_rival = self.ebrm.target_basin(s_x, wrong_labels) + 0.10 * torch.randn_like(z_target)
        z_origin = 0.15 * torch.randn_like(z_target)

        z_init = self.ebrm.propose_initial_z(s_x, g_spec)
        _, _, rvq_commit = self.ebrm.quantize_ncm(z_pos)

        e_pos, r3_logits, _ = self.ebrm.free_energy_and_logits(s_x, z_pos, g_spec)
        e_rival, _, _ = self.ebrm.free_energy_and_logits(s_x, z_rival, g_spec)
        e_orig, _, _ = self.ebrm.free_energy_and_logits(s_x, z_origin, g_spec)

        loss = (
            F.cross_entropy(r2_logits, gt_targets)
            + F.cross_entropy(r3_logits, gt_targets)
            + 0.5 * F.mse_loss(z_init, z_target.detach())
            + e_pos.pow(2).mean()
            + 0.5 * F.relu(0.85 - e_rival).pow(2).mean()
            + 0.5 * F.relu(0.55 - e_orig).pow(2).mean()
            + 0.25 * rvq_commit
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.rung2.parameters()) + list(self.ebrm.parameters()), 1.0
        )
        self.upper_opt.step()

    def step(self, x_bytes: np.ndarray) -> CommitResult:
        """Backwards-compatible single-episode wrapper."""
        rec = self.step_microbatch([x_bytes])
        if rec.rung0_satiated:
            rung = 0
        elif rec.rung2_settled or rec.soinn_settled:
            rung = 2
        else:
            rung = 3
        ok = rec.verified_count == 1
        return CommitResult(ok, ok, -1, rung, rec.total_joules, "MICROBATCH_1")
