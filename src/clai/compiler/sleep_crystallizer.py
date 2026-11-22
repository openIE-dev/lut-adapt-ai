"""SleepCrystallizer: Ring-buffer uint64 trace harvester + Spectral-Mux Straight-Through LUT6 compiler."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus


class SleepCrystallizer:
    """
    GPU-resident uint64 ring-buffer crystallizer for variable-length 64-episode micro-batches.
    Harvests verified commits, writes exact N-gram signatures back to Engram DRAM, synthesizes
    initial wire/table topology via 3rd-order Walsh-Hadamard cumulants, refines via Straight-Through
    DiffLUT optimization, and hot-swaps Rung 0 only when bit-exact holdout agreement passes gate.
    """

    def __init__(
        self,
        engram: EngramBitBus,
        lut_net: DiffLUTNetwork,
        device: torch.device,
        capacity: int = 4096,
        lr: float = 0.04,
        epochs: int = 28,
        min_agreement: float = 0.72,
    ):
        self.engram = engram
        self.lut_net = lut_net
        self.device = device
        self.capacity = capacity
        self.lr = lr
        self.epochs = epochs
        self.min_agreement = min_agreement

        self.word_ring = torch.zeros((capacity, 8), device=device, dtype=torch.int64)
        self.label_ring = torch.zeros((capacity,), device=device, dtype=torch.long)
        self.write_ptr = 0
        self.count = 0

    @torch.no_grad()
    def record_batch_words(
        self,
        words_u64: torch.Tensor,
        verified_actions: torch.Tensor,
        raw_episodes: list[np.ndarray] | None = None,
    ) -> None:
        """Ingests [M, 8] int64 Engram wire-words + [M] verified actions directly on device."""
        M = words_u64.size(0)
        if M == 0:
            return

        if raw_episodes is not None:
            acts_cpu = verified_actions.detach().cpu().numpy()
            for ep, act in zip(raw_episodes, acts_cpu):
                self.engram.write_association_u64(ep, int(act))

        if M >= self.capacity:
            words_u64 = words_u64[-self.capacity :]
            verified_actions = verified_actions[-self.capacity :]
            M = self.capacity

        end = self.write_ptr + M
        if end <= self.capacity:
            self.word_ring[self.write_ptr : end] = words_u64
            self.label_ring[self.write_ptr : end] = verified_actions
        else:
            first = self.capacity - self.write_ptr
            self.word_ring[self.write_ptr :] = words_u64[:first]
            self.label_ring[self.write_ptr :] = verified_actions[:first]
            rem = M - first
            self.word_ring[:rem] = words_u64[first:]
            self.label_ring[:rem] = verified_actions[first:]

        self.write_ptr = (self.write_ptr + M) % self.capacity
        self.count = min(self.capacity, self.count + M)

    def record_verified_trace(self, x_bytes: np.ndarray, verified_action: int) -> None:
        w = self.engram.fetch_wire_words_u64(x_bytes, self.device)
        y = torch.tensor([int(verified_action)], device=self.device, dtype=torch.long)
        self.record_batch_words(w, y, raw_episodes=[x_bytes])

    @torch.no_grad()
    def clear_stale_traces(self) -> None:
        """Flushes pre-mutation ring buffer and invalidates compiled LUT6 reflex."""
        self.write_ptr = 0
        self.count = 0
        self.lut_net.invalidate()

    @torch.no_grad()
    def _seed_direct_contrastive_priors(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """1st-order full-512-wire contrastive literal-detector topology (L7 packets, ASTs, Engrams)."""
        num_k = self.lut_net.num_classes
        luts_per_cls = self.lut_net.width // num_k
        s_all = 2.0 * x_train - 1.0
        s_all_c = s_all - s_all.mean(dim=0, keepdim=True)
        cls_sal_all = torch.zeros((num_k, self.lut_net.bus_bits), device=x_train.device)
        for c in range(num_k):
            mc = y_train == c
            if int(mc.sum().item()) > 4:
                cls_sal_all[c] = s_all_c[mc].mean(dim=0)

        l0 = self.lut_net.layers[0]
        l0.wire_logits.fill_(-8.0)
        b0 = l0.basis
        for c in range(num_k):
            rival_mean = (cls_sal_all.sum(dim=0) - cls_sal_all[c]) / max(1, num_k - 1)
            disc = cls_sal_all[c] - rival_mean
            top_w = torch.topk(disc.abs(), k=32).indices.tolist()
            top_sgn = [1.0 if float(disc[w].item()) >= 0.0 else 0.0 for w in top_w]
            for k in range(luts_per_cls):
                n = c * luts_per_cls + k
                acc_match = torch.zeros(64, device=x_train.device)
                for p_off in range(6):
                    idx_w = (k * 3 + p_off * 5) % len(top_w)
                    l0.wire_logits[n, p_off, top_w[idx_w]] = 8.0
                    bit_p = b0[:, p_off]
                    acc_match = acc_match + (bit_p if top_sgn[idx_w] > 0.5 else (1.0 - bit_p))
                det = (acc_match >= 4.0).float()
                l0.table_logits[n].copy_(1.8 * (2.0 * det - 1.0))

        for l_idx in range(1, len(self.lut_net.layers)):
            lyr = self.lut_net.layers[l_idx]
            lyr.wire_logits.fill_(-8.0)
            maj = (lyr.basis.sum(dim=-1) >= 3.0).float()
            for c in range(num_k):
                for k in range(luts_per_cls):
                    n = c * luts_per_cls + k
                    for p_off in range(6):
                        src = c * luts_per_cls + ((k + p_off * 11) % luts_per_cls)
                        lyr.wire_logits[n, p_off, src] = 8.0
                    lyr.table_logits[n].copy_(1.8 * (2.0 * maj - 1.0))

    @torch.no_grad()
    def _seed_parity_cumulant_priors(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """3rd-order Walsh-Hadamard cumulant parity-mux topology (non-linear XOR + linear SAT)."""
        s_xor = 2.0 * x_train[:, :128] - 1.0
        s_th_raw = 2.0 * x_train[:, 128:384] - 1.0
        s_th = s_th_raw - s_th_raw.mean(dim=0, keepdim=True)

        num_k = self.lut_net.num_classes
        # Project onto active class-salient subspace (suppresses inactive zero-weight slots!)
        cls_profiles = []
        for c in range(num_k):
            mc = y_train == c
            if int(mc.sum().item()) > 4:
                cls_profiles.append(F.normalize(s_th[mc].mean(dim=0), dim=-1))
        if cls_profiles:
            proj_u = torch.stack(cls_profiles, dim=-1)  # [256, <=num_k]
        else:
            proj_u = torch.randn((256, 8), device=x_train.device) / 16.0
        z = s_th @ proj_u
        n_proj = z.size(1)

        score_mat = torch.zeros((128, 128), device=x_train.device)
        for c in range(num_k):
            m = y_train == c
            n_c = int(m.sum().item())
            if n_c < 8:
                continue
            sx_c = s_xor[m]
            z_c = z[m] - z[m].mean(dim=0, keepdim=True)
            for r in range(n_proj):
                cov3 = ((sx_c * z_c[:, r : r + 1]).T @ sx_c).abs() / n_c
                score_mat = torch.maximum(score_mat, cov3)

        score_mat.fill_diagonal_(0.0)
        best_pair = int(score_mat.view(-1).argmax().item())
        pw1, pw2 = best_pair // 128, best_pair % 128
        p_hat = s_xor[:, pw1] * s_xor[:, pw2]  # +1 when parity==0, -1 when parity==1

        # Per-slot diverse wire selection: pick top 4 thresholds from each of the top 6 slots
        s_th_slot = s_th.view(-1, 16, 16)
        sal_pos = torch.zeros((num_k, 16, 16), device=x_train.device)
        sal_neg = torch.zeros((num_k, 16, 16), device=x_train.device)
        for c in range(num_k):
            mp = (y_train == c) & (p_hat > 0)
            mn = (y_train == c) & (p_hat < 0)
            if int(mp.sum().item()) > 4:
                sal_pos[c] = s_th_slot[mp].mean(dim=0)
            if int(mn.sum().item()) > 4:
                sal_neg[c] = s_th_slot[mn].mean(dim=0)

        luts_per_cls = self.lut_net.width // num_k
        l0 = self.lut_net.layers[0]
        l0.wire_logits.fill_(-8.0)
        b0 = l0.basis
        p_xor = ((b0[:, 4].long() ^ b0[:, 5].long()) == 1).float()

        for c in range(num_k):
            top_slots_p = torch.topk(sal_pos[c].abs().max(dim=-1).values, k=6).indices
            top_slots_n = torch.topk(sal_neg[c].abs().max(dim=-1).values, k=6).indices
            wires_p, sgns_p = [], []
            for s_id in top_slots_p.tolist():
                for th in torch.topk(sal_pos[c, s_id].abs(), k=4).indices.tolist():
                    wires_p.append(128 + s_id * 16 + th)
                    sgns_p.append(1.0 if float(sal_pos[c, s_id, th].item()) >= 0.0 else 0.0)
            wires_n, sgns_n = [], []
            for s_id in top_slots_n.tolist():
                for th in torch.topk(sal_neg[c, s_id].abs(), k=4).indices.tolist():
                    wires_n.append(128 + s_id * 16 + th)
                    sgns_n.append(1.0 if float(sal_neg[c, s_id, th].item()) >= 0.0 else 0.0)

            n_wp = len(wires_p)
            n_wn = len(wires_n)
            for k in range(luts_per_cls):
                n = c * luts_per_cls + k
                ip0, ip1 = (k * 2) % n_wp, (k * 2 + 5) % n_wp
                in0, in1 = (k * 2) % n_wn, (k * 2 + 5) % n_wn
                l0.wire_logits[n, 0, wires_p[ip0]] = 8.0
                l0.wire_logits[n, 1, wires_p[ip1]] = 8.0
                l0.wire_logits[n, 2, wires_n[in0]] = 8.0
                l0.wire_logits[n, 3, wires_n[in1]] = 8.0
                l0.wire_logits[n, 4, pw1] = 8.0
                l0.wire_logits[n, 5, pw2] = 8.0

                vp0 = b0[:, 0] if sgns_p[ip0] > 0.5 else (1.0 - b0[:, 0])
                vp1 = b0[:, 1] if sgns_p[ip1] > 0.5 else (1.0 - b0[:, 1])
                vn0 = b0[:, 2] if sgns_n[in0] > 0.5 else (1.0 - b0[:, 2])
                vn1 = b0[:, 3] if sgns_n[in1] > 0.5 else (1.0 - b0[:, 3])
                mux_t = (1.0 - p_xor) * (0.5 * (vp0 + vp1)) + p_xor * (0.5 * (vn0 + vn1))
                l0.table_logits[n].copy_(1.8 * (2.0 * mux_t - 1.0))

        for l_idx in range(1, len(self.lut_net.layers)):
            lyr = self.lut_net.layers[l_idx]
            lyr.wire_logits.fill_(-8.0)
            bl = lyr.basis
            maj = (bl.sum(dim=-1) >= 3.0).float()
            for c in range(num_k):
                for k in range(luts_per_cls):
                    n = c * luts_per_cls + k
                    for p_off in range(6):
                        src = c * luts_per_cls + ((k + p_off * 11) % luts_per_cls)
                        lyr.wire_logits[n, p_off, src] = 8.0
                    lyr.table_logits[n].copy_(1.8 * (2.0 * maj - 1.0))

    @torch.no_grad()
    def _seed_spectral_wire_priors(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Zero-heuristic dual-topology tournament (<1 ms): evaluates both 1st-order full-bus
        contrastive routing and 3rd-order Walsh-Hadamard parity-mux routing on bit-exact
        snapped LUT6 bit-planes, keeping whichever topology achieves higher empirical accuracy.
        """
        eval_n = min(512, x_train.size(0))
        x_eval = x_train[:eval_n]
        y_eval = y_train[:eval_n]

        self._seed_direct_contrastive_priors(x_train, y_train)
        self.lut_net.compile_all()
        preds_direct, _, _ = self.lut_net.infer_batch_with_margin(x_eval, margin_threshold=0)
        acc_direct = float((preds_direct == y_eval).float().mean().item())
        direct_state = {k: v.clone() for k, v in self.lut_net.state_dict().items()}

        self._seed_parity_cumulant_priors(x_train, y_train)
        self.lut_net.compile_all()
        preds_cum, _, _ = self.lut_net.infer_batch_with_margin(x_eval, margin_threshold=0)
        acc_cum = float((preds_cum == y_eval).float().mean().item())

        if acc_direct >= max(0.55, acc_cum):
            self.lut_net.load_state_dict(direct_state)
        self.lut_net.invalidate()

    def compile_downward(self, batch_size: int = 128) -> dict:
        if self.count < 256:
            return {"compiled": False, "agreement": 0.0, "traces": self.count}

        w_all = self.word_ring[: self.count]
        y_all = self.label_ring[: self.count]

        perm_split = torch.randperm(self.count, device=self.device)
        w_shuf = w_all[perm_split]
        y_shuf = y_all[perm_split]

        val_size = max(64, ((int(0.15 * self.count)) // 64) * 64)
        split = self.count - val_size

        x_train_float = self.engram.unpack_words_to_float_bus(w_shuf[:split])
        y_train = y_shuf[:split]
        w_val_u64 = w_shuf[split:]
        y_val = y_shuf[split:]

        if not self.lut_net.layers[0].is_compiled:
            self._seed_spectral_wire_priors(x_train_float, y_train)

        table_params = [p for name, p in self.lut_net.named_parameters() if "table_logits" in name]
        opt = torch.optim.AdamW(table_params, lr=self.lr * 1.5, weight_decay=1e-4)
        self.lut_net.train()

        for ep in range(self.epochs):
            progress = ep / max(1, self.epochs - 1)
            tau = max(0.18, 0.65 - 0.47 * progress)

            perm = torch.randperm(x_train_float.size(0), device=self.device)
            for i in range(0, x_train_float.size(0), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                votes = self.lut_net.forward_train(
                    x_train_float[idx], tau=tau, hard_routing=True
                )
                loss = F.cross_entropy(votes * 0.5, y_train[idx])
                loss.backward()
                opt.step()

        self.lut_net.compile_all()

        preds, _, _ = self.lut_net.infer_from_engram_words(w_val_u64, margin_threshold=0)
        assert preds is not None
        agreement = float((preds == y_val).float().mean().item())

        if agreement < self.min_agreement:
            self.lut_net.invalidate()
            return {"compiled": False, "agreement": agreement, "traces": self.count}

        return {"compiled": True, "agreement": agreement, "traces": self.count}
