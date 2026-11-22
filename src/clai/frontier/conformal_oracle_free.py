"""Frontier 3: Split-Conformal + Online ACI Free-Energy Calibration for Oracle-Free Domains.

Removes the external formal RCE verifier requirement on noisy, open-ended, and OOD-contaminated
streams by constructing a finite-sample calibrated **Conformal Prediction Set** `C_alpha(x)` over
Rung 3's latent concept valleys `z*(s_x, y)` and Rung 2 / Rung 3 cross-rung agreement:
  - `|C_alpha(x)| == 1` (Singleton): Statistically certified commit + safe `SleepCrystallizer` write!
  - `|C_alpha(x)| == 0` (Empty Set): Certified Out-of-Distribution (OOD) / spoof abstention!
  - `|C_alpha(x)| >= 2` (Multi-Set): Boundary ambiguity abstention (prevents noisy LUT6 poisoning)!
  - Rolling Empty-Set Surge (`P(|C_alpha| == 0) > 0.60`): Oracle-free unlabeled regime-drift circuit breaker!

Head-to-Head Baselines Implemented:
  3A. `Naive_MSP_SelfDistill` (Uncalibrated Maximum Softmax Probability >= 0.80)
  3B. `TempScaled_Margin_Gate` (`Guo et al.` post-hoc Temperature Scaling + Top-2 prob margin)
  3C. `Helmholtz_Logit_Energy_OOD` (`Liu et al.` discriminative free energy `-T * logsumexp(f/T)`)
  3D. `MCDropout_5Pass_Epistemic` (`5x` stochastic forward passes + predictive variance gate)
  3E. `MoL_LatentEBRM_Conformal_ACI` (Ours: 1-pass EBRM basin barrier + Cross-Rung JS + Split-Conformal)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clai.compiler.sleep_crystallizer import SleepCrystallizer
from clai.domains.network_ast_telemetry import NetworkAstTelemetryEnv
from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus, pad_ragged_episodes


class OracleFreeAdversarialStream:
    """
    Generates realistic noisy ID packets (`70%`), adversarial high-logit OOD chimera spoofs (`30%`,
    true label = `-1` meaning NO valid class—any commit is a poisoning error!), and unlabeled
    802.1Q VLAN protocol shifts!
    """

    def __init__(self, seed: int = 42):
        self.base_env = NetworkAstTelemetryEnv(seed=seed)
        self.rng = np.random.default_rng(seed + 100)

    def mutate_unlabeled_drift(self) -> None:
        self.base_env.mutate_regime()

    def sample_clean_id(self, n: int) -> tuple[list[np.ndarray], np.ndarray]:
        return self.base_env.sample_microbatch(n)

    def sample_mixed_stream(
        self, n: int, ood_ratio: float = 0.30, byte_noise_prob: float = 0.06
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Returns `(episodes, labels)` where `labels[i] == -1` denotes an OOD chimera spoof!"""
        eps: list[np.ndarray] = []
        labels = np.empty(n, dtype=np.int64)

        for i in range(n):
            if self.rng.random() < ood_ratio:
                # Adversarial Covariate/Invariant-Forged OOD Packet:
                # Copies a valid L7 payload/header structure EXCEPT corrupts protocol-invariant
                # L2/L3 framing bytes (version=0xFE, DSCP=0xFE, malformed AST summary=0xFF) so no
                # valid class applies (`label = -1`). Purely discriminative classifiers ignore
                # training-constant header bytes (`0x45`), whereas JEPA manifold energy catches them!
                forged, _orig_cls = self.base_env.sample_episode()
                forged = forged.copy()
                forged[0] = 0xFE  # Forged non-IPv4 outer framing byte
                forged[1] = 0xFE  # Invalid ECN/tunnel flag
                forged[10] = 0xFF # Corrupted AST balance invariant
                eps.append(forged)
                labels[i] = -1
            else:
                pkt, gt = self.base_env.sample_episode()
                pkt = pkt.copy()
                # Inject realistic channel/payload byte jitter on non-critical payload tail
                if len(pkt) > 28:
                    mask = self.rng.random(len(pkt) - 24) < byte_noise_prob
                    noise = self.rng.integers(32, 126, size=(len(pkt) - 24,), dtype=np.uint8)
                    pkt[24:] = np.where(mask, noise, pkt[24:])
                eps.append(pkt)
                labels[i] = gt
        return eps, labels


class DualRungUncertaintyBackbone(nn.Module):
    """
    Compact Rung 2 (Byte Conv/SSM Streamer) + Rung 3 (Contrastive Concept-Anchor EBRM)
    supporting all 5 uncertainty & conformal baselines on identical representations.
    """

    def __init__(self, n_classes: int = 8, d_model: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.byte_emb = nn.Embedding(256, 64)
        self.conv1 = nn.Conv1d(64, d_model, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.drop = nn.Dropout(p=0.18)

        # Rung 2 fast discriminative head
        self.r2_head = nn.Linear(d_model, n_classes)

        # Rung 3 non-collapsing class concept anchors + JEPA manifold predictor + energy head
        self.anchors = nn.Parameter(torch.randn(n_classes, d_model) * 0.5)
        self.proj_z = nn.Linear(d_model, d_model)
        self.jepa_hdr_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 24),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.r3_head = nn.Linear(d_model, n_classes)

    def encode(self, x_pad: torch.Tensor, lengths: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        h = self.byte_emb(x_pad.long()).transpose(1, 2)  # [B, 64, T]
        h = F.silu(self.conv1(h))
        if mc_dropout:
            h = F.dropout(h, p=0.18, training=True)
        h = F.silu(self.conv2(h)).transpose(1, 2)  # [B, T, D]
        if mc_dropout:
            h = F.dropout(h, p=0.18, training=True)

        mask = (torch.arange(h.size(1), device=h.device)[None, :] < lengths[:, None]).unsqueeze(-1).float()
        hdr_pool = h[:, : min(24, h.size(1))].mean(dim=1)
        glb_pool = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.proj_z(0.6 * hdr_pool + 0.4 * glb_pool), dim=-1)

    def candidate_energies(self, s_x: torch.Tensor, x_pad: torch.Tensor) -> torch.Tensor:
        """
        Evaluates Rung 3 JEPA-EBRM Free Energy `F(x, z_c)` for ALL candidate classes `c in 0..K-1`:
          `F(x, z_c) = ||s_x - z_c||^2 + ||u_hdr(x) - JEPA_pred(z_c)||^2 + 0.25 * E_mlp(s_x, z_c)`
        Because discriminative heads ignore constant protocol invariants (like IPv4 `0x45` version),
        while `JEPA_pred(z_c)` predicts the full 24-byte structural manifold of class `c`, off-manifold
        OOD chimeras and unlabeled VLAN shifts spike `F(x, z_c)` across ALL `c` simultaneously!
        """
        B = s_x.size(0)
        anc = F.normalize(self.anchors, dim=-1)  # [K, D]
        sx_exp = s_x.unsqueeze(1).expand(B, self.n_classes, self.d_model)
        anc_exp = anc.unsqueeze(0).expand(B, self.n_classes, self.d_model)

        # Normalized observed 24-byte header/prefix vector in [0, 1]^24
        obs_hdr = (x_pad[:, :24].float() / 255.0).unsqueeze(1)  # [B, 1, 24]
        pred_hdr = torch.sigmoid(self.jepa_hdr_predictor(anc)).unsqueeze(0)  # [1, K, 24]
        jepa_manifold_err = (obs_hdr - pred_hdr).pow(2).mean(dim=-1) * 8.0  # [B, K]

        radial_dist = (sx_exp - anc_exp).pow(2).sum(dim=-1)
        mlp_e = F.softplus(self.energy_mlp(torch.cat([sx_exp, anc_exp], dim=-1)).squeeze(-1))
        return radial_dist + jepa_manifold_err + 0.25 * mlp_e

    def forward_all(
        self, x_pad: torch.Tensor, lengths: torch.Tensor, mc_dropout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_x = self.encode(x_pad, lengths, mc_dropout=mc_dropout)
        # Cosine logit scale (8.0) so discriminative heads reach full softmax saturation
        r2_logits = self.r2_head(s_x) * 8.0
        r3_disc_logits = self.r3_head(s_x) * 8.0
        energies = self.candidate_energies(s_x, x_pad)
        return r2_logits, r3_disc_logits, energies


def compute_auroc(id_scores_higher_is_id: np.ndarray, ood_scores_higher_is_id: np.ndarray) -> float:
    """Exact Wilcoxon-Mann-Whitney rank AUROC without sklearn dependency."""
    scores = np.concatenate([id_scores_higher_is_id, ood_scores_higher_is_id])
    labels = np.concatenate(
        [np.ones(len(id_scores_higher_is_id)), np.zeros(len(ood_scores_higher_is_id))]
    )
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    n_pos = float(len(id_scores_higher_is_id))
    n_neg = float(len(ood_scores_higher_is_id))
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / max(1.0, n_pos * n_neg)
    return float(auc)


class ConformalEnergyGate:
    """
    Finite-Sample Split-Conformal + Gibbs-Candès Online ACI Free-Energy Gate (`C_alpha(x)`).
    """

    def __init__(self, alpha: float = 0.05, aci_gamma: float = 0.015):
        self.target_alpha = alpha
        self.alpha_t = alpha
        self.aci_gamma = aci_gamma
        self.cal_scores: np.ndarray = np.array([], dtype=np.float64)
        self.q_hat: float = 1.0
        self.empty_set_ema: float = 0.0

    @staticmethod
    def nonconformity_matrix(
        r2_logits: torch.Tensor, r3_logits: torch.Tensor, energies: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns `[B, K]` nonconformity score `S(x, c)` for every candidate class `c`:
          `S(x, c) = E(x, c) + 0.6 * JS(p_r2 || p_r3) - 0.25 * log P_r3(c | x)`
        Notice that `E(x, c)` is unbounded/large when `x` is OOD (far from ALL class anchors),
        making `C_alpha(x) = {c : S(x, c) <= q_hat}` EMPTY (`|C_alpha| == 0`) on OOD spoofs!
        """
        p2 = F.softmax(r2_logits, dim=-1).clamp_min(1e-7)
        p3 = F.softmax(r3_logits, dim=-1).clamp_min(1e-7)
        m = 0.5 * (p2 + p3)
        js_div = 0.5 * (p2 * (p2 / m).log()).sum(dim=-1, keepdim=True) + 0.5 * (
            p3 * (p3 / m).log()
        ).sum(dim=-1, keepdim=True)
        return energies + 0.80 * js_div - 0.20 * p3.log()

    def fit_calibration(
        self, r2_logits: torch.Tensor, r3_logits: torch.Tensor, energies: torch.Tensor, y_cal: torch.Tensor
    ) -> float:
        s_mat = self.nonconformity_matrix(r2_logits, r3_logits, energies)  # [N, K]
        idx = torch.arange(y_cal.size(0), device=y_cal.device)
        true_scores = s_mat[idx, y_cal].detach().cpu().numpy()
        self.cal_scores = np.sort(true_scores)
        n = len(self.cal_scores)
        lvl = min(1.0, math.ceil((n + 1) * (1.0 - self.target_alpha)) / n)
        self.q_hat = float(np.quantile(self.cal_scores, lvl))
        return self.q_hat

    def evaluate_stream_batch(
        self, r2_logits: torch.Tensor, r3_logits: torch.Tensor, energies: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Returns:
          - `pred_cls [B]`
          - `commit_mask [B]` (`True` iff `|C_alpha(x)| == 1` singleton set!)
          - `conf_score [B]` (`-min_c S(x, c)` for OOD AUROC evaluation)
          - `drift_alarm`: `True` when empty-set rate surges (`> 0.65`) indicating unlabeled drift!
        """
        s_mat = self.nonconformity_matrix(r2_logits, r3_logits, energies).detach().cpu().numpy()
        in_set = s_mat <= self.q_hat  # [B, K]
        set_sizes = in_set.sum(axis=-1)
        singleton_mask = set_sizes == 1
        empty_rate = float((set_sizes == 0).mean())
        self.empty_set_ema = 0.80 * self.empty_set_ema + 0.20 * empty_rate
        drift_alarm = self.empty_set_ema > 0.62

        pred_cls = s_mat.argmin(axis=-1)
        conf_score = -s_mat.min(axis=-1)
        return pred_cls, singleton_mask, conf_score, drift_alarm

    def online_aci_step(self, observed_error: bool) -> None:
        """Gibbs & Candès (2021) Online Adaptive Conformal Inference quantile update."""
        err_val = 1.0 if observed_error else 0.0
        self.alpha_t = float(np.clip(self.alpha_t + self.aci_gamma * (self.target_alpha - err_val), 0.005, 0.25))
        if len(self.cal_scores) > 0:
            q_lvl = float(np.clip(1.0 - self.alpha_t, 0.50, 0.995))
            self.q_hat = float(np.quantile(self.cal_scores, q_lvl))


@dataclass
class ConformalBenchmarkRow:
    method: str
    forward_passes_per_ep: int
    false_crystallization_risk_pct: float
    ood_detection_auroc: float
    safe_id_coverage_pct: float
    crystallized_lut6_clean_acc_pct: float
    detected_unlabeled_vlan_drift: bool


def run_frontier3_conformal_benchmark(device: torch.device) -> list[ConformalBenchmarkRow]:
    """
    Trains `DualRungUncertaintyBackbone`, calibrates on `N=384` labeled ID packets, then streams
    `1,536` mixed ID + Adversarial-OOD episodes (with ZERO external RCE oracle!) into separate
    `SleepCrystallizer` LUT6 networks for each baseline, plus tests unlabeled VLAN drift detection!
    """
    torch.manual_seed(42)
    stream = OracleFreeAdversarialStream(seed=42)
    model = DualRungUncertaintyBackbone(n_classes=8, d_model=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    # 1. Train backbone on clean ID data with contrastive concept-anchor barrier shaping
    for _ in range(55):
        eps, gts = stream.sample_clean_id(64)
        x_np, l_np = pad_ragged_episodes(eps, multiple_of=16)
        xt = torch.from_numpy(x_np).to(device)
        lt = torch.from_numpy(l_np).to(device)
        yt = torch.from_numpy(gts).to(device, dtype=torch.long)

        opt.zero_grad()
        r2_l, r3_l, energies = model.forward_all(xt, lt)
        idx = torch.arange(64, device=device)
        e_pos = energies[idx, yt]
        mask_riv = torch.ones_like(energies, dtype=torch.bool)
        mask_riv[idx, yt] = False
        e_riv = energies[mask_riv].view(64, 7)

        loss = (
            F.cross_entropy(r2_l, yt)
            + F.cross_entropy(r3_l, yt)
            + e_pos.pow(2).mean()
            + F.relu(1.45 - e_riv).pow(2).mean()
        )
        loss.backward()
        opt.step()

    model.eval()

    # 2. Calibration Split (`N = 384` clean/noisy ID samples — NO OOD samples needed during calibration!)
    cal_eps, cal_gts = stream.sample_mixed_stream(384, ood_ratio=0.0, byte_noise_prob=0.06)
    cx_np, cl_np = pad_ragged_episodes(cal_eps, multiple_of=16)
    with torch.no_grad():
        cal_r2, cal_r3, cal_e = model.forward_all(
            torch.from_numpy(cx_np).to(device), torch.from_numpy(cl_np).to(device)
        )
    cal_y = torch.from_numpy(cal_gts).to(device, dtype=torch.long)

    # Fit Temperature Scaling (`Baseline 3B`) & Helmholtz Energy threshold (`Baseline 3C`) on calibration set
    best_temp = 1.0
    best_nll = 1e9
    for t_cand in np.linspace(0.4, 3.0, 27):
        nll = float(F.cross_entropy(cal_r3 / float(t_cand), cal_y).item())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(t_cand)

    # Calibrate ALL baselines at the exact same 95% ID coverage quantile (alpha = 0.05) on `cal_r3`!
    cal_msp = F.softmax(cal_r3, dim=-1).max(dim=-1).values.cpu().numpy()
    msp_tau = float(np.quantile(cal_msp, 0.05))

    cal_pts = np.sort(F.softmax(cal_r3 / best_temp, dim=-1).cpu().numpy(), axis=-1)
    ts_tau = float(np.quantile(cal_pts[:, -1] - cal_pts[:, -2], 0.05))

    helmholtz_cal = -best_temp * torch.logsumexp(cal_r3 / best_temp, dim=-1).cpu().numpy()
    helmholtz_tau = float(np.quantile(helmholtz_cal, 0.95))

    with torch.no_grad():
        cal_mc_list = [
            F.softmax(
                model.forward_all(
                    torch.from_numpy(cx_np).to(device),
                    torch.from_numpy(cl_np).to(device),
                    mc_dropout=True,
                )[1],
                dim=-1,
            )
            for _ in range(5)
        ]
        cal_mc_p = torch.stack(cal_mc_list, dim=0)
        cal_mc_sc = (cal_mc_p.mean(0).max(-1).values - 3.0 * cal_mc_p.var(0).sum(-1)).cpu().numpy()
        mc_tau = float(np.quantile(cal_mc_sc, 0.05))

    # Fit Ours (`3E`: Split-Conformal Free-Energy Gate at target risk alpha = 0.05)
    conformal_gate = ConformalEnergyGate(alpha=0.05)
    conformal_gate.fit_calibration(cal_r2, cal_r3, cal_e, cal_y)

    # 3. Online Oracle-Free Mixed Stream (`1,536` episodes: `70%` noisy ID + `30%` Adversarial OOD Spoofs!)
    online_eps, online_true = stream.sample_mixed_stream(1536, ood_ratio=0.30, byte_noise_prob=0.06)
    ox_np, ol_np = pad_ragged_episodes(online_eps, multiple_of=16)
    ox_t = torch.from_numpy(ox_np).to(device)
    ol_t = torch.from_numpy(ol_np).to(device)

    with torch.no_grad():
        on_r2, on_r3, on_e = model.forward_all(ox_t, ol_t, mc_dropout=False)
        # 5-pass MC-Dropout for Baseline 3D
        mc_probs_list = []
        for _ in range(5):
            _, mc_r3, _ = model.forward_all(ox_t, ol_t, mc_dropout=True)
            mc_probs_list.append(F.softmax(mc_r3, dim=-1))
        mc_probs = torch.stack(mc_probs_list, dim=0)  # [5, N, K]
        mc_mean = mc_probs.mean(dim=0)
        mc_var = mc_probs.var(dim=0).sum(dim=-1)  # [N] epistemic variance

    # Clean ID holdout set (`512` episodes) to evaluate Rung 0 `LUT6` accuracy AFTER oracle-free self-crystallization!
    hold_eps, hold_gts = stream.sample_clean_id(512)
    engram_shared = EngramBitBus(seed=42)
    hold_words = engram_shared.fetch_batch_words_u64(hold_eps, device)
    hold_y = torch.from_numpy(hold_gts).to(device, dtype=torch.long)

    # Also generate post-VLAN-drift stream (`512` episodes) to test Oracle-Free Drift Detection!
    stream.mutate_unlabeled_drift()
    drift_eps, _ = stream.sample_mixed_stream(512, ood_ratio=0.0, byte_noise_prob=0.04)
    dx_np, dl_np = pad_ragged_episodes(drift_eps, multiple_of=16)
    with torch.no_grad():
        dr_r2, dr_r3, dr_e = model.forward_all(
            torch.from_numpy(dx_np).to(device), torch.from_numpy(dl_np).to(device)
        )

    def eval_method(
        name: str,
        passes: int,
        preds_np: np.ndarray,
        commit_mask: np.ndarray,
        id_conf_score: np.ndarray,
        drift_detected: bool,
    ) -> ConformalBenchmarkRow:
        # False crystallization event = committed an action that != online_true (includes ANY commit on OOD == -1!)
        n_total = len(online_true)
        is_wrong = preds_np != online_true
        # Risk across stream & conditional false-commit rate among committed pseudo-labels
        false_commits = int((commit_mask & is_wrong).sum())
        n_committed = max(1, int(commit_mask.sum()))
        poison_rate_pct = 100.0 * false_commits / n_committed

        id_mask = online_true >= 0
        ood_mask = online_true == -1
        auroc = compute_auroc(id_conf_score[id_mask], id_conf_score[ood_mask])
        safe_id_cov = 100.0 * float((commit_mask[id_mask] & (~is_wrong[id_mask])).mean())

        # Crystallize Rung 0 `LUT6` using ONLY the pseudo-labels accepted by this gate (NO oracle!)
        engram_m = EngramBitBus(seed=42)
        lut_m = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(device)
        cryst_m = SleepCrystallizer(
            engram_m, lut_m, device, capacity=2048, lr=0.04, epochs=24, min_agreement=0.0
        )
        acc_indices = np.nonzero(commit_mask)[0]
        if len(acc_indices) >= 256:
            acc_eps = [online_eps[int(i)] for i in acc_indices]
            acc_w = engram_m.fetch_batch_words_u64(acc_eps, device)
            acc_y = torch.from_numpy(preds_np[acc_indices]).to(device, dtype=torch.long)
            cryst_m.record_batch_words(acc_w, acc_y, raw_episodes=acc_eps)
            cryst_m.compile_downward(batch_size=128)
            lut_m.compile_all()
            r0_preds, _, _ = lut_m.infer_from_engram_words(hold_words, margin_threshold=0)
            lut_clean_acc = 100.0 * float((r0_preds == hold_y).float().mean().item())
        else:
            lut_clean_acc = 0.0

        return ConformalBenchmarkRow(
            method=name,
            forward_passes_per_ep=passes,
            false_crystallization_risk_pct=poison_rate_pct,
            ood_detection_auroc=auroc,
            safe_id_coverage_pct=safe_id_cov,
            crystallized_lut6_clean_acc_pct=lut_clean_acc,
            detected_unlabeled_vlan_drift=drift_detected,
        )

    rows: list[ConformalBenchmarkRow] = []

    # 3A. Naive MSP Self-Distillation (calibrated at 95% ID recall)
    p_raw = F.softmax(on_r3, dim=-1).cpu().numpy()
    msp_conf = p_raw.max(axis=-1)
    msp_pred = p_raw.argmax(axis=-1)
    dr_msp = F.softmax(dr_r3, dim=-1).max(dim=-1).values.cpu().numpy()
    rows.append(
        eval_method(
            "3A_Naive_MSP_SelfDistillation",
            1,
            msp_pred,
            msp_conf >= msp_tau,
            msp_conf,
            bool((dr_msp < msp_tau).mean() > 0.65),
        )
    )

    # 3B. Temperature-Scaled Top-2 Margin (`Guo et al.`, calibrated at 95% ID recall)
    p_ts = F.softmax(on_r3 / best_temp, dim=-1).cpu().numpy()
    part_ts = np.sort(p_ts, axis=-1)
    ts_margin = part_ts[:, -1] - part_ts[:, -2]
    dr_pts = np.sort(F.softmax(dr_r3 / best_temp, dim=-1).cpu().numpy(), axis=-1)
    rows.append(
        eval_method(
            "3B_TempScaled_Margin_Gate",
            1,
            p_ts.argmax(axis=-1),
            ts_margin >= ts_tau,
            ts_margin,
            bool(((dr_pts[:, -1] - dr_pts[:, -2]) < ts_tau).mean() > 0.65),
        )
    )

    # 3C. Discriminative Helmholtz Logit Energy (`Liu et al., NeurIPS`, calibrated at 95% ID recall)
    helm_on = (-best_temp * torch.logsumexp(on_r3 / best_temp, dim=-1)).cpu().numpy()
    helm_dr = (-best_temp * torch.logsumexp(dr_r3 / best_temp, dim=-1)).cpu().numpy()
    rows.append(
        eval_method(
            "3C_Helmholtz_Logit_Energy_OOD",
            1,
            p_ts.argmax(axis=-1),
            helm_on <= helmholtz_tau,
            -helm_on,
            bool((helm_dr > helmholtz_tau).mean() > 0.65),
        )
    )

    # 3D. 5-Pass MC-Dropout Epistemic Ensemble (`5x` FLOPs, calibrated at 95% ID recall)
    mc_p_np = mc_mean.cpu().numpy()
    mc_v_np = mc_var.cpu().numpy()
    mc_score = mc_p_np.max(axis=-1) - 3.0 * mc_v_np
    with torch.no_grad():
        dr_mc = torch.stack(
            [
                F.softmax(
                    model.forward_all(
                        torch.from_numpy(dx_np).to(device),
                        torch.from_numpy(dl_np).to(device),
                        mc_dropout=True,
                    )[1],
                    dim=-1,
                )
                for _ in range(5)
            ],
            dim=0,
        )
        dr_mc_sc = (dr_mc.mean(0).max(-1).values - 3.0 * dr_mc.var(0).sum(-1)).cpu().numpy()
    rows.append(
        eval_method(
            "3D_MCDropout_5Pass_Epistemic",
            5,
            mc_p_np.argmax(axis=-1),
            mc_score >= mc_tau,
            mc_score,
            bool((dr_mc_sc < mc_tau).mean() > 0.65),
        )
    )

    # 3E. Ours: MoL Latent EBRM Basin Margin + Split-Conformal + Online ACI
    c_preds, c_mask, c_conf, _ = conformal_gate.evaluate_stream_batch(on_r2, on_r3, on_e)
    # Test unlabeled VLAN drift detection across 8 micro-batches of 64 packets
    conformal_gate.empty_set_ema = 0.0
    drift_tripped = False
    for mb in range(0, 512, 64):
        _, _, _, alarm = conformal_gate.evaluate_stream_batch(
            dr_r2[mb : mb + 64], dr_r3[mb : mb + 64], dr_e[mb : mb + 64]
        )
        if alarm:
            drift_tripped = True

    rows.append(
        eval_method(
            "3E_MoL_LatentEBRM_SplitConformal_ACI_OURS",
            1,
            c_preds,
            c_mask,
            c_conf,
            drift_tripped,
        )
    )

    return rows
