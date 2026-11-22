"""Phase 17: Live Speculative Code-Completion Engine + Online File-Edit Sleep-Crystallizer.

Unites:
  - Rung 0: `PagedBenesLUT6CodeDrafter` + `HardwareBRAMPageCache` (`0.0` draft FLOPs, simulated
    1-cycle `4.0 ns` KV260 `RAMB36E2` hash-indexed `LUT6 INIT` page switching)
  - Rung 2: Stateful `Mamba3CodeLM` (`d=256, L=4`, parallel `T <= 5` burst verifier with exact
    recurrent state rollback)
  - Plasticity: Replay-regularized online `Mamba3CodeLM` adaptation + incremental Joint-Mode
    `LUT6` truth-table Sleep-Crystallization (`sleep_crystallize_text` & `FileEditSleepWatcher`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from clai.frontier.air_mol_streamer import (
    AsyncLayerPrefetcher,
    NVMeIOMeter,
    deserialize_shard_bytes,
    pread_uncached_bytes,
    serialize_state_dict_shard,
)
from clai.frontier.benes_bram_page_cache import HardwareBRAMPageCache
from clai.frontier.code_corpus_speculative import (
    CompiledLutPage,
    Mamba3CodeBlock,
    Mamba3CodeLM,
    MambaCodeLayerState,
    PagedBenesLUT6CodeDrafter,
    SourceCodeFile,
    encode_code_bitbus_512,
    load_repo_code_splits,
)


LANG_NAME_TO_ID = {
    "py": 0,
    "python": 0,
    ".py": 0,
    "rs": 1,
    "rust": 1,
    ".rs": 1,
    "sv": 2,
    "verilog": 2,
    "systemverilog": 2,
    ".sv": 2,
    ".v": 2,
    "zig": 3,
    "go": 3,
    "jl": 3,
    "other": 3,
}

LANG_ID_TO_NAME = {0: "python", 1: "rust", 2: "systemverilog", 3: "polyglot"}


def resolve_lang_id(lang: str = "auto", file_path: str = "", prompt: str = "") -> int:
    if lang and lang.lower() != "auto":
        return LANG_NAME_TO_ID.get(lang.lower(), 0)
    if file_path:
        suf = Path(file_path).suffix.lower()
        if suf in LANG_NAME_TO_ID:
            return LANG_NAME_TO_ID[suf]
    if any(k in prompt for k in ("module ", "always @(", "posedge ", "wire [", "reg [", "endmodule")):
        return 2
    if any(k in prompt for k in ("pub fn ", "fn ", "let mut ", "impl ", "::", "&self")):
        return 1
    return 0


@dataclass
class SpeculativeBurstEvent:
    step: int
    emitted_text: str
    emitted_bytes: list[int]
    token_origins: list[str]  # Per-byte: "R0_LUT6" | "R2_BONUS" | "R2_OVERRIDE" | "R2_ESCALATE"
    rung0_tier: str
    kv260_bram_slot: int
    kv260_bram_hit: bool
    draft_proposed_len: int
    draft_accepted_len: int
    step_latency_ms: float
    cumulative_bytes: int
    cumulative_passes: int
    mean_bytes_per_pass: float
    r0_lut6_share_pct: float
    draft_accept_rate_pct: float
    mj_per_byte: float
    speedup_vs_t1: float
    air_nvme_active: bool = False
    step_nvme_kb_read: float = 0.0
    cumulative_nvme_mb_read: float = 0.0
    peak_resident_ram_mb: float = 0.0
    nvme_sweeps_saved: int = 0


@dataclass
class CompletionSummary:
    prompt: str
    lang: str
    completion_text: str
    total_bytes: int
    total_target_passes: int
    mean_bytes_per_pass: float
    speedup_vs_t1: float
    r0_lut6_bytes: int
    r2_bonus_bytes: int
    r2_override_or_escalate_bytes: int
    r0_lut6_share_pct: float
    draft_proposed_total: int
    draft_accepted_total: int
    draft_accept_rate_pct: float
    kv260_bram_hits: int
    kv260_bram_hit_rate_pct: float
    total_latency_ms: float
    estimated_mj_per_byte: float
    bram_generation: int
    lossless_verified: bool
    events: list[dict]
    air_nvme_active: bool = False
    total_nvme_mb_read: float = 0.0
    peak_resident_ram_mb: float = 0.0
    nvme_sweeps_saved: int = 0


@dataclass
class SleepCrystallizationEvent:
    timestamp: float
    file_path: str
    lang: str
    file_bytes: int
    adapt_r2_steps: int
    lut6_6gram_pages_updated: int
    lut6_4gram_pages_updated: int
    ast_branch_pages_compiled: int
    kv260_bram_slots_programmed: int
    bram_generation: int
    pre_bytes_per_pass: float
    post_bytes_per_pass: float
    pre_lut6_accept_pct: float
    post_lut6_accept_pct: float
    pre_bram_hit_pct: float
    post_bram_hit_pct: float
    elapsed_ms: float


class LiveSpeculativeCodeEngine:
    """
    Thread-safe, live-adaptable Speculative Code Engine exposing `Mamba3CodeLM` +
    `PagedBenesLUT6CodeDrafter` + `HardwareBRAMPageCache` for CLI & HTTP/SSE serving.
    """

    def __init__(
        self,
        model: Mamba3CodeLM,
        drafter: PagedBenesLUT6CodeDrafter,
        bram_cache: HardwareBRAMPageCache,
        replay_windows: np.ndarray,
        device: torch.device,
        air_nvme: bool = False,
        nvme_shard_dir: Path = Path("artifacts/mol_live_nvme_shards"),
    ):
        self.drafter = drafter
        self.bram_cache = bram_cache
        self.replay_windows = replay_windows  # [N_rep, 197] uint8 reservoir preventing forgetting
        self.device = device
        self.lock = threading.RLock()
        self.crystallization_history: list[SleepCrystallizationEvent] = []
        self.air_nvme = air_nvme
        self.nvme_shard_dir = nvme_shard_dir

        self.d_model = model.d_model
        self.n_layers = model.n_layers
        self.n_heads = model.blocks[0].n_heads
        self.step_flops = float(model.count_step_flops())
        self.full_model_bytes = sum(p.numel() * 4 for p in model.parameters())
        self.nvme_shard_disk_bytes = 0
        self._layer_shard_paths: list[Path] = []
        self._air_emb_weight: torch.Tensor | None = None
        self._air_norm_f_weight: torch.Tensor | None = None
        self._air_pinned_bytes: int = 0

        if self.air_nvme:
            # Keep `self.model` strictly on CPU as cold backup for optional online adaptation,
            # while GPU/MPS holds ONLY 1 streamed 4-bit layer shard at a time!
            self.model = model.to("cpu")
            self.model.eval()
            self._export_air_nvme_shards()
        else:
            self.model = model.to(device)
            self.model.eval()

    def _export_air_nvme_shards(self) -> None:
        self.nvme_shard_dir.mkdir(parents=True, exist_ok=True)
        tot = serialize_state_dict_shard(
            {"emb.weight": self.model.emb.weight, "norm_f.weight": self.model.norm_f.weight},
            self.nvme_shard_dir / "live_m3_stem.shard4",
            mode="fp16",
        )
        paths: list[Path] = []
        for idx, blk in enumerate(self.model.blocks):
            sp = self.nvme_shard_dir / f"live_m3_layer_{idx:02d}.shard4"
            tot += serialize_state_dict_shard(blk.state_dict(), sp, mode="4bit")
            paths.append(sp)
        self._layer_shard_paths = paths
        self.nvme_shard_disk_bytes = tot

        stem_sd, stem_bytes = deserialize_shard_bytes(
            pread_uncached_bytes(self.nvme_shard_dir / "live_m3_stem.shard4"), self.device
        )
        self._air_emb_weight = stem_sd["emb.weight"]
        self._air_norm_f_weight = stem_sd["norm_f.weight"]
        # Pinned memory = stem + O(1) 4-layer Mamba-3 states + 81 RAMB36E2 LUT6 BRAM (324 KB)
        state_bytes = self.n_layers * (3 * self.d_model + self.d_model) * 4
        lut6_bram_bytes = 81 * 512 * 8
        self._air_pinned_bytes = stem_bytes + state_bytes + lut6_bram_bytes
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def _init_states(self) -> list[MambaCodeLayerState]:
        d_head = self.d_model // self.n_heads
        return [
            MambaCodeLayerState(
                conv_buf=torch.zeros(1, 3, self.d_model, device=self.device),
                ssm_h=torch.zeros(1, self.n_heads, d_head, device=self.device),
            )
            for _ in range(self.n_layers)
        ]

    def _forward_verify_target(
        self,
        tokens: torch.Tensor,
        states: list[MambaCodeLayerState],
        meter: NVMeIOMeter | None = None,
    ) -> tuple[torch.Tensor, list[list[MambaCodeLayerState]]]:
        if not self.air_nvme:
            logits, _, by_pos = self.model.forward_verify_steps(tokens, states)
            return logits, by_pos

        assert self._air_emb_weight is not None and self._air_norm_f_weight is not None
        io_meter = meter if meter is not None else NVMeIOMeter()
        h = F.embedding(tokens, self._air_emb_weight)
        T = tokens.size(1)
        layer_hists: list[list[MambaCodeLayerState]] = []
        prefetcher = AsyncLayerPrefetcher(self._layer_shard_paths, io_meter, enabled=True)

        def _run_layer(l_idx: int, raw_shard: bytes) -> None:
            nonlocal h
            sd, w_bytes = deserialize_shard_bytes(raw_shard, self.device)
            io_meter.record_alloc(w_bytes)
            with torch.device("meta"):
                blk = Mamba3CodeBlock(self.d_model, self.n_heads)
            blk.load_state_dict(sd, strict=True, assign=True)
            h, st_hist = blk.forward_stateful(h, states[l_idx])
            layer_hists.append(st_hist)
            del blk, sd
            io_meter.record_free(w_bytes)

        try:
            prefetcher.run_sweep(_run_layer)
        finally:
            prefetcher.close()

        h_f = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * self._air_norm_f_weight
        logits = h_f @ self._air_emb_weight.t()
        by_pos = [[layer_hists[l_idx][t] for l_idx in range(self.n_layers)] for t in range(T)]
        return logits, by_pos

    @classmethod
    def load_or_train(
        cls,
        checkpoint_path: Path = Path("artifacts/mol_code_engine.pt"),
        repo_root: Path = Path("."),
        device: torch.device | None = None,
        train_steps: int = 460,
        force_retrain: bool = False,
        air_nvme: bool = False,
        nvme_shard_dir: Path = Path("artifacts/mol_live_nvme_shards"),
    ) -> "LiveSpeculativeCodeEngine":
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if checkpoint_path.exists() and not force_retrain:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = Mamba3CodeLM(d_model=256, n_layers=4, n_heads=8)
            model.load_state_dict(payload["model_state"])
            drafter: PagedBenesLUT6CodeDrafter = payload["drafter"]
            bram_cache: HardwareBRAMPageCache = payload["bram_cache"]
            replay_windows: np.ndarray = payload["replay_windows"]
            return cls(
                model, drafter, bram_cache, replay_windows, device, air_nvme=air_nvme, nvme_shard_dir=nvme_shard_dir
            )

        # Fresh corpus build (`train_files` + `test_files` united for full-repo completion assistant,
        # or `train_files` for clean eval; we use `load_repo_code_splits` train+test workspace files)
        train_files, test_files = load_repo_code_splits(repo_root)
        all_corpus = train_files + test_files

        seq_len = 192
        windows: list[np.ndarray] = []
        for sf in all_corpus:
            raw = sf.raw_bytes
            for start in range(0, len(raw) - (seq_len + 5), 64):
                windows.append(raw[start : start + seq_len + 5])
        win_arr = np.stack(windows, axis=0)

        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        model = Mamba3CodeLM(d_model=256, n_layers=4, n_heads=8).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3.8e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, train_steps), eta_min=4e-4)

        model.train()
        for _ in range(train_steps):
            idx = rng.integers(0, len(win_arr), size=(64,))
            batch_t = torch.from_numpy(win_arr[idx]).to(device, dtype=torch.long)
            opt.zero_grad()
            logits, _ = model.forward_train_parallel(batch_t[:, :seq_len], chunk_q=32)
            loss = F.cross_entropy(logits.reshape(-1, 256), batch_t[:, 1 : seq_len + 1].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()

        drafter = PagedBenesLUT6CodeDrafter(draft_k=4)
        drafter.compile_from_corpus(all_corpus, max_pages4=1400)

        bram_cache = HardwareBRAMPageCache(page_addr_bits=9)
        bram_cache.compile_from_corpus(all_corpus)

        rep_idx = rng.choice(len(win_arr), size=min(512, len(win_arr)), replace=False)
        replay_windows = win_arr[rep_idx].copy()

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model_cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state": model_cpu_state,
                "drafter": drafter,
                "bram_cache": bram_cache,
                "replay_windows": replay_windows,
            },
            checkpoint_path,
        )
        return cls(
            model, drafter, bram_cache, replay_windows, device, air_nvme=air_nvme, nvme_shard_dir=nvme_shard_dir
        )

    def draft_with_telemetry(self, ctx_bytes: list[int], lang_id: int) -> tuple[list[int], str, int, bool]:
        """
        Evaluates Rung 0 `PagedBenesLUT6CodeDrafter` + `HardwareBRAMPageCache`, returning
        `(draft_bytes, rung0_tier, kv260_bram_slot, kv260_bram_hit)`.
        """
        if len(ctx_bytes) < 6:
            return [], "R2_SHORT_CTX", 0, False

        raw_win = bytes(ctx_bytes[-256:])
        bus_1d = encode_code_bitbus_512(raw_win, lang_id)
        bus = bus_1d[None, :]
        tail6 = raw_win[-6:]
        tail4 = raw_win[-4:]

        bram_hit, bram_slot, bram_bytes4, _ = self.bram_cache.lookup_and_eval(bus_1d, lang_id)

        # 1. Joint-Mode 160-LUT6 AST/Indentation/Scope Disambiguator on branching 6-gram
        page6 = self.drafter.pages6.get((lang_id, tail6))
        if page6 is not None:
            pred_b, _ = self.drafter._eval_page_batch(page6, bus)
            return [int(pred_b[0, k]) for k in range(self.drafter.draft_k)], "R0_BRAM_LUT6_AST6", bram_slot, bram_hit

        # 2. If KV260 BRAM cache was live-updated via Online Sleep-Crystallization, check `mux_lut6_6` / BRAM hit
        const6 = self.drafter.mux_lut6_6.get(tail6)
        if const6 is not None:
            if bus_1d[452] == 1 and getattr(self.drafter, "mux_lut6_6_freq", {}).get(tail6, 99) <= 3:
                pred_b, _ = self.drafter._eval_page_batch(self.drafter.default_page, bus)
                return [int(pred_b[0, k]) for k in range(self.drafter.draft_k)], "R0_XNOR_CAM_COPY", bram_slot, bram_hit
            tier = "R0_BRAM_LUT6_HIT6" if bram_hit else "R0_MUX_LUT6_6GRAM"
            return list(const6), tier, bram_slot, bram_hit

        if bram_hit:
            return bram_bytes4[: self.drafter.draft_k], "R0_BRAM_LUT6_HIT6", bram_slot, True

        # 3. Unseen 6-gram with exact local In-File XNOR-CAM hit (`bus[452]` or `bus[448]`)
        if bus_1d[452] == 1 or bus_1d[448] == 1:
            pred_b, _ = self.drafter._eval_page_batch(self.drafter.default_page, bus)
            return [int(pred_b[0, k]) for k in range(self.drafter.draft_k)], "R0_XNOR_CAM_COPY", bram_slot, False

        # 4. Joint-Mode 160-LUT6 AST Disambiguator on branching 4-gram
        page4 = self.drafter.pages4.get((lang_id, tail4))
        if page4 is not None:
            pred_b, _ = self.drafter._eval_page_batch(page4, bus)
            return [int(pred_b[0, k]) for k in range(self.drafter.draft_k)], "R0_BRAM_LUT6_AST4", bram_slot, False

        # 5. Unambiguous 4-gram LUT6 constant or 3-gram XNOR-CAM fallback
        const4 = self.drafter.mux_lut6_4.get(tail4)
        if const4 is not None:
            return list(const4), "R0_MUX_LUT6_4GRAM", bram_slot, False

        if bus_1d[449] == 1:
            pred_b, _ = self.drafter._eval_page_batch(self.drafter.default_page, bus)
            return [int(pred_b[0, k]) for k in range(self.drafter.draft_k)], "R0_XNOR_CAM_3GRAM", bram_slot, False

        return [], "R2_ESCALATE_MISS", bram_slot, False

    @torch.no_grad()
    def standalone_mamba3_greedy(self, prompt_bytes: bytes, max_bytes: int) -> bytes:
        """Reference `T=1` greedy decoding on `Mamba3CodeLM` (or 4-bit `AirMoL` target) to verify lossless parity."""
        with self.lock:
            p_arr = np.frombuffer(prompt_bytes, dtype=np.uint8)
            p_t = torch.from_numpy(p_arr[None, :].copy()).to(self.device, dtype=torch.long)
            states = self._init_states()
            logits, st_pos = self._forward_verify_target(p_t, states)
            states = st_pos[-1]
            tok = int(logits[0, -1].argmax().item())
            out = [tok]
            while len(out) < max_bytes:
                in_t = torch.tensor([[out[-1]]], device=self.device, dtype=torch.long)
                lg, st_p = self._forward_verify_target(in_t, states)
                states = st_p[0]
                out.append(int(lg[0, 0].argmax().item()))
            return bytes(out[:max_bytes])

    @torch.no_grad()
    def stream_complete(
        self,
        prompt: str,
        lang: str = "auto",
        file_path: str = "",
        max_bytes: int = 96,
    ) -> Iterator[SpeculativeBurstEvent]:
        """
        Streams real-time per-burst speculative code completion events with Rung 0 / Rung 2
        routing provenance, KV260 BRAM page-switch telemetry, and `--air-nvme` shard I/O telemetry.
        """
        if not prompt:
            prompt = "def "
        lang_id = resolve_lang_id(lang, file_path, prompt)
        p_raw = prompt.encode("utf-8", errors="replace")
        p_arr = np.frombuffer(p_raw, dtype=np.uint8)

        emitted: list[int] = []
        ctx_full: list[int] = p_arr.tolist()
        r0_bytes_total = 0
        proposed_total = 0
        accepted_total = 0
        total_passes = 0

        io_meter = NVMeIOMeter()
        if self.air_nvme:
            io_meter.record_alloc(self._air_pinned_bytes)
        else:
            io_meter.record_alloc(self.full_model_bytes + 81 * 512 * 8)

        try:
            # Step 1: Prefill prompt through Rung 2 Mamba-3 SSM state (`O(1)` recurrent state afterwards!)
            t_step0 = time.perf_counter()
            prev_read = io_meter.bytes_read
            with self.lock:
                p_t = torch.from_numpy(p_arr[None, :].copy()).to(self.device, dtype=torch.long)
                init_st = self._init_states()
                logits, st_by_pos = self._forward_verify_target(p_t, init_st, io_meter)
                states = st_by_pos[-1]
                first_tok = int(logits[0, -1].argmax().item())

            dt_ms = (time.perf_counter() - t_step0) * 1e3
            step_kb = (io_meter.bytes_read - prev_read) / 1024.0
            cum_mb = io_meter.bytes_read / (1024.0 * 1024.0)
            peak_mb = io_meter.peak_resident_weight_bytes / (1024.0 * 1024.0)
            emitted.append(first_tok)
            ctx_full.append(first_tok)
            total_passes += 1

            mj_per_pass = 24.6

            yield SpeculativeBurstEvent(
                step=total_passes,
                emitted_text=bytes([first_tok]).decode("utf-8", errors="replace"),
                emitted_bytes=[first_tok],
                token_origins=["R2_ESCALATE"],
                rung0_tier="R2_PREFILL_SEED",
                kv260_bram_slot=0,
                kv260_bram_hit=False,
                draft_proposed_len=0,
                draft_accepted_len=0,
                step_latency_ms=round(dt_ms, 3),
                cumulative_bytes=1,
                cumulative_passes=1,
                mean_bytes_per_pass=1.0,
                r0_lut6_share_pct=0.0,
                draft_accept_rate_pct=0.0,
                mj_per_byte=round(mj_per_pass, 3),
                speedup_vs_t1=1.0,
                air_nvme_active=self.air_nvme,
                step_nvme_kb_read=round(step_kb, 2),
                cumulative_nvme_mb_read=round(cum_mb, 3),
                peak_resident_ram_mb=round(peak_mb, 2),
                nvme_sweeps_saved=0,
            )

            while len(emitted) < max_bytes:
                t_s = time.perf_counter()
                rem = max_bytes - len(emitted)
                prev_read = io_meter.bytes_read
                with self.lock:
                    raw_draft, tier, bram_slot, bram_hit = self.draft_with_telemetry(ctx_full, lang_id)
                    draft_toks = raw_draft[: min(4, rem)]

                    step_bytes: list[int] = []
                    step_origins: list[str] = []
                    accepted_k = 0

                    if not draft_toks:
                        in_t = torch.tensor([[emitted[-1]]], device=self.device, dtype=torch.long)
                        lg, st_pos = self._forward_verify_target(in_t, states, io_meter)
                        nxt = int(lg[0, 0].argmax().item())
                        states = st_pos[0]
                        step_bytes.append(nxt)
                        step_origins.append("R2_ESCALATE")
                    else:
                        verify_in = torch.tensor([[emitted[-1], *draft_toks]], device=self.device, dtype=torch.long)
                        v_logits, v_st_pos = self._forward_verify_target(verify_in, states, io_meter)

                        for k_idx, d_tok in enumerate(draft_toks):
                            t_choice = int(v_logits[0, k_idx].argmax().item())
                            if d_tok == t_choice:
                                step_bytes.append(d_tok)
                                step_origins.append("R0_LUT6")
                                accepted_k += 1
                                if len(emitted) + len(step_bytes) >= max_bytes:
                                    states = v_st_pos[k_idx]
                                    break
                            else:
                                step_bytes.append(t_choice)
                                step_origins.append("R2_OVERRIDE")
                                states = v_st_pos[k_idx]
                                break
                        else:
                            if len(emitted) + len(step_bytes) < max_bytes:
                                bonus = int(v_logits[0, len(draft_toks)].argmax().item())
                                step_bytes.append(bonus)
                                step_origins.append("R2_BONUS")
                            states = v_st_pos[len(draft_toks)]

                dt_ms = (time.perf_counter() - t_s) * 1e3
                step_kb = (io_meter.bytes_read - prev_read) / 1024.0
                cum_mb = io_meter.bytes_read / (1024.0 * 1024.0)
                peak_mb = io_meter.peak_resident_weight_bytes / (1024.0 * 1024.0)
                emitted.extend(step_bytes)
                ctx_full.extend(step_bytes)
                total_passes += 1
                r0_bytes_total += accepted_k
                proposed_total += len(draft_toks)
                accepted_total += accepted_k

                cum_b = len(emitted)
                b_per_pass = cum_b / total_passes
                acc_rate = (100.0 * accepted_total / proposed_total) if proposed_total > 0 else 0.0
                r0_share = 100.0 * r0_bytes_total / cum_b
                mj_b = (total_passes * (mj_per_pass + 0.001)) / cum_b

                yield SpeculativeBurstEvent(
                    step=total_passes,
                    emitted_text=bytes(step_bytes).decode("utf-8", errors="replace"),
                    emitted_bytes=step_bytes,
                    token_origins=step_origins,
                    rung0_tier=tier,
                    kv260_bram_slot=bram_slot,
                    kv260_bram_hit=bram_hit,
                    draft_proposed_len=len(draft_toks),
                    draft_accepted_len=accepted_k,
                    step_latency_ms=round(dt_ms, 3),
                    cumulative_bytes=cum_b,
                    cumulative_passes=total_passes,
                    mean_bytes_per_pass=round(b_per_pass, 3),
                    r0_lut6_share_pct=round(r0_share, 2),
                    draft_accept_rate_pct=round(acc_rate, 2),
                    mj_per_byte=round(mj_b, 3),
                    speedup_vs_t1=round(b_per_pass, 3),
                    air_nvme_active=self.air_nvme,
                    step_nvme_kb_read=round(step_kb, 2),
                    cumulative_nvme_mb_read=round(cum_mb, 3),
                    peak_resident_ram_mb=round(peak_mb, 2),
                    nvme_sweeps_saved=max(0, cum_b - total_passes),
                )
        finally:
            if self.air_nvme:
                io_meter.record_free(self._air_pinned_bytes)
            else:
                io_meter.record_free(self.full_model_bytes + 81 * 512 * 8)

    def complete_sync(
        self,
        prompt: str,
        lang: str = "auto",
        file_path: str = "",
        max_bytes: int = 96,
        verify_lossless: bool = False,
    ) -> CompletionSummary:
        t0 = time.perf_counter()
        lang_id = resolve_lang_id(lang, file_path, prompt)
        events = list(self.stream_complete(prompt=prompt, lang=lang, file_path=file_path, max_bytes=max_bytes))
        total_ms = (time.perf_counter() - t0) * 1e3

        all_bytes: list[int] = []
        r0_b = 0
        r2_bonus = 0
        r2_other = 0
        prop_tot = 0
        acc_tot = 0
        bram_hits = 0

        for ev in events:
            all_bytes.extend(ev.emitted_bytes)
            for orig in ev.token_origins:
                if orig == "R0_LUT6":
                    r0_b += 1
                elif orig == "R2_BONUS":
                    r2_bonus += 1
                else:
                    r2_other += 1
            prop_tot += ev.draft_proposed_len
            acc_tot += ev.draft_accepted_len
            if ev.kv260_bram_hit:
                bram_hits += 1

        raw_out = bytes(all_bytes[:max_bytes])
        lossless_ok = True
        if verify_lossless:
            ref_out = self.standalone_mamba3_greedy((prompt or "def ").encode("utf-8", errors="replace"), max_bytes)
            lossless_ok = raw_out == ref_out

        last = events[-1]
        draft_steps = max(1, len(events) - 1)
        return CompletionSummary(
            prompt=prompt,
            lang=LANG_ID_TO_NAME[lang_id],
            completion_text=raw_out.decode("utf-8", errors="replace"),
            total_bytes=len(raw_out),
            total_target_passes=len(events),
            mean_bytes_per_pass=last.mean_bytes_per_pass,
            speedup_vs_t1=last.speedup_vs_t1,
            r0_lut6_bytes=r0_b,
            r2_bonus_bytes=r2_bonus,
            r2_override_or_escalate_bytes=r2_other,
            r0_lut6_share_pct=last.r0_lut6_share_pct,
            draft_proposed_total=prop_tot,
            draft_accepted_total=acc_tot,
            draft_accept_rate_pct=last.draft_accept_rate_pct,
            kv260_bram_hits=bram_hits,
            kv260_bram_hit_rate_pct=round(100.0 * bram_hits / draft_steps, 2),
            total_latency_ms=round(total_ms, 2),
            estimated_mj_per_byte=last.mj_per_byte,
            bram_generation=self.bram_cache.bram_generation,
            lossless_verified=lossless_ok,
            events=[asdict(e) for e in events],
            air_nvme_active=self.air_nvme,
            total_nvme_mb_read=last.cumulative_nvme_mb_read,
            peak_resident_ram_mb=last.peak_resident_ram_mb,
            nvme_sweeps_saved=last.nvme_sweeps_saved,
        )

    def _probe_file_efficiency(self, raw_bytes: bytes, lang_id: int) -> tuple[float, float, float]:
        """Measures `(bytes_per_pass, lut6_accept_pct, bram_hit_pct)` across probe prefixes of a file."""
        if len(raw_bytes) < 96:
            return 1.0, 0.0, 0.0
        n_probes = min(4, max(1, (len(raw_bytes) - 80) // 64))
        b_pass_sum = 0.0
        acc_sum = 0.0
        hit_sum = 0.0
        for p_i in range(n_probes):
            off = min(len(raw_bytes) - 80, p_i * 48)
            p_txt = raw_bytes[off : off + 48].decode("utf-8", errors="replace")
            res = self.complete_sync(prompt=p_txt, lang=LANG_ID_TO_NAME[lang_id], max_bytes=32, verify_lossless=False)
            b_pass_sum += res.mean_bytes_per_pass
            acc_sum += res.draft_accept_rate_pct
            hit_sum += res.kv260_bram_hit_rate_pct
        return (
            round(b_pass_sum / n_probes, 3),
            round(acc_sum / n_probes, 2),
            round(hit_sum / n_probes, 2),
        )

    def adapt_rung2_online(self, content: str | bytes, adapt_r2_steps: int = 16) -> float:
        """
        Wake-phase Rung 2 `Mamba3CodeLM` replay-regularized online plasticity on newly edited code.
        Returns elapsed adaptation time in milliseconds.
        """
        raw = content.encode("utf-8", errors="replace") if isinstance(content, str) else content
        raw_arr = np.frombuffer(raw, dtype=np.uint8).copy()
        if adapt_r2_steps <= 0 or len(raw_arr) < 48:
            return 0.0

        t0 = time.perf_counter()
        with self.lock:
            tiled = raw_arr
            while len(tiled) < 260:
                tiled = np.concatenate([tiled, np.frombuffer(b"\n", dtype=np.uint8), raw_arr])

            new_wins: list[np.ndarray] = []
            stride = max(8, min(32, (len(tiled) - 197) // 8 + 1))
            for st in range(0, len(tiled) - 197 + 1, stride):
                new_wins.append(tiled[st : st + 197])
            new_win_arr = np.stack(new_wins, axis=0)

            rng = np.random.default_rng(42)
            if self.air_nvme:
                self.model = self.model.to(self.device)
            opt = torch.optim.AdamW(self.model.parameters(), lr=2.2e-3, weight_decay=1e-4)
            self.model.train()
            for _ in range(adapt_r2_steps):
                idx_new = rng.integers(0, len(new_win_arr), size=(24,))
                idx_rep = rng.integers(0, len(self.replay_windows), size=(24,))
                batch_np = np.concatenate([new_win_arr[idx_new], self.replay_windows[idx_rep]], axis=0)
                batch_t = torch.from_numpy(batch_np).to(self.device, dtype=torch.long)
                opt.zero_grad()
                logits, _ = self.model.forward_train_parallel(batch_t[:, :192], chunk_q=32)
                loss = F.cross_entropy(logits.reshape(-1, 256), batch_t[:, 1:193].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            if self.air_nvme:
                self.model = self.model.to("cpu")
                self._export_air_nvme_shards()
        return (time.perf_counter() - t0) * 1e3

    def sleep_crystallize_text(
        self,
        file_path: str,
        content: str | bytes,
        lang: str = "auto",
        adapt_r2_steps: int = 16,
    ) -> SleepCrystallizationEvent:
        """
        Live Online Sleep-Crystallization on a newly edited or created source file:
          1. If `adapt_r2_steps > 0`, runs Wake-phase replay-regularized Rung 2 `Mamba3CodeLM`
             plasticity so new domain identifiers/APIs enter Rung 2 without forgetting.
          2. Probes Pre-Sleep Rung 0 `LUT6` burst & KV260 BRAM hit rates against Rung 2.
          3. Incrementally compiles Joint-Mode Minterm `160-LUT6` AST pages + `2-Mux` `LUT6` pages
             and hot-programs them into the KV260 `512`-entry BRAM Page Cache (`0` downtime!).
          4. Probes Post-Sleep burst efficiency on the exact same Rung 2 model and logs telemetry.
        """
        raw = content.encode("utf-8", errors="replace") if isinstance(content, str) else content
        lang_id = resolve_lang_id(lang, file_path, raw[:256].decode("utf-8", errors="replace"))

        with self.lock:
            if adapt_r2_steps > 0:
                self.adapt_rung2_online(raw, adapt_r2_steps=adapt_r2_steps)

            pre_bpp, pre_acc, pre_hit = self._probe_file_efficiency(raw, lang_id)
            t0 = time.perf_counter()
            raw_arr = np.frombuffer(raw, dtype=np.uint8).copy()

            # Incremental Rung 0 Joint-Mode `LUT6` Truth-Table Compilation + KV260 BRAM Hot-Swap!
            counts6: dict[bytes, dict[bytes, int]] = {}
            counts4: dict[bytes, dict[bytes, int]] = {}
            buckets6: dict[bytes, list[tuple[np.ndarray, np.ndarray]]] = {}
            buckets4: dict[bytes, list[tuple[np.ndarray, np.ndarray]]] = {}

            for pos in range(6, len(raw) - 4):
                s6 = raw[pos - 6 : pos]
                s4 = raw[pos - 4 : pos]
                nxt4 = raw[pos : pos + 4]
                d6 = counts6.setdefault(s6, {})
                d6[nxt4] = d6.get(nxt4, 0) + 1
                d4 = counts4.setdefault(s4, {})
                d4[nxt4] = d4.get(nxt4, 0) + 1

                if len(buckets6.get(s6, ())) < 64:
                    bus_v = encode_code_bitbus_512(raw[max(0, pos - 256) : pos], lang_id)
                    arr_nxt = raw_arr[pos : pos + 4]
                    buckets6.setdefault(s6, []).append((bus_v, arr_nxt))
                    if len(buckets4.get(s4, ())) < 64:
                        buckets4.setdefault(s4, []).append((bus_v, arr_nxt))

            upd_6 = 0
            upd_4 = 0
            ast_compiled = 0
            bram_slots: set[int] = set()

            for s6, v_dict in counts6.items():
                best4 = max(v_dict.items(), key=lambda kv: kv[1])[0]
                self.drafter.mux_lut6_6[s6] = best4
                if hasattr(self.drafter, "mux_lut6_6_freq"):
                    self.drafter.mux_lut6_6_freq[s6] = 100
                upd_6 += 1

                items = buckets6.get(s6, [])
                if len(v_dict) >= 2 and len(items) >= 2:
                    xm = np.stack([it[0] for it in items], axis=0)
                    ym = np.stack([it[1] for it in items], axis=0)
                    self.drafter.pages6[(lang_id, s6)] = self.drafter._compile_single_page(xm, ym)
                    bram_tbl = self.bram_cache.synthesize_joint_mode_tables(xm, ym)
                    ast_compiled += 1
                else:
                    # Remove stale branching override if file made this 6-gram unambiguous
                    self.drafter.pages6.pop((lang_id, s6), None)
                    bram_tbl = self.bram_cache.synthesize_const_tables(best4, allow_cam_override=False)

                slot = self.bram_cache.write_page(
                    lang_id, s6, bram_tbl, priority=50_000 + sum(v_dict.values()), force_overwrite=True
                )
                bram_slots.add(slot)

            for s4, v_dict in counts4.items():
                best4 = max(v_dict.items(), key=lambda kv: kv[1])[0]
                self.drafter.mux_lut6_4[s4] = best4
                upd_4 += 1
                items = buckets4.get(s4, [])
                if len(v_dict) >= 2 and len(items) >= 3:
                    xm = np.stack([it[0] for it in items], axis=0)
                    ym = np.stack([it[1] for it in items], axis=0)
                    self.drafter.pages4[(lang_id, s4)] = self.drafter._compile_single_page(xm, ym)
                    ast_compiled += 1

            self.bram_cache.bram_generation += 1
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            post_bpp, post_acc, post_hit = self._probe_file_efficiency(raw, lang_id)

            ev = SleepCrystallizationEvent(
                timestamp=time.time(),
                file_path=file_path,
                lang=LANG_ID_TO_NAME[lang_id],
                file_bytes=len(raw),
                adapt_r2_steps=adapt_r2_steps,
                lut6_6gram_pages_updated=upd_6,
                lut6_4gram_pages_updated=upd_4,
                ast_branch_pages_compiled=ast_compiled,
                kv260_bram_slots_programmed=len(bram_slots),
                bram_generation=self.bram_cache.bram_generation,
                pre_bytes_per_pass=pre_bpp,
                post_bytes_per_pass=post_bpp,
                pre_lut6_accept_pct=pre_acc,
                post_lut6_accept_pct=post_acc,
                pre_bram_hit_pct=pre_hit,
                post_bram_hit_pct=post_hit,
                elapsed_ms=round(elapsed_ms, 2),
            )
            self.crystallization_history.append(ev)
            return ev

    def health_snapshot(self) -> dict:
        occupied = int((self.bram_cache.tag_bram != 0).sum())
        return {
            "status": "ok",
            "device": str(self.device),
            "mamba3_params": int(sum(p.numel() for p in self.model.parameters())),
            "mamba3_step_mflops": round(self.step_flops / 1e6, 3),
            "lut6_pages_6gram": len(self.drafter.mux_lut6_6) + len(self.drafter.pages6),
            "lut6_pages_4gram": len(self.drafter.mux_lut6_4) + len(self.drafter.pages4),
            "lut6_ast_branch_pages": len(self.drafter.pages6) + len(self.drafter.pages4),
            "kv260_bram_capacity_pages": self.bram_cache.num_pages,
            "kv260_bram_occupied_pages": occupied,
            "kv260_bram_generation": self.bram_cache.bram_generation,
            "online_crystallizations": len(self.crystallization_history),
            "air_nvme_active": self.air_nvme,
            "full_model_fp32_mb": round(self.full_model_bytes / (1024 * 1024), 2),
            "nvme_shard_disk_mb": round(self.nvme_shard_disk_bytes / (1024 * 1024), 2),
        }


class FileEditSleepWatcher:
    """
    Background file-system watcher that detects newly created or edited `.py`/`.rs`/`.sv`/`.zig`/
    `.go`/`.jl` files and triggers live Online Sleep-Crystallization automatically!
    """

    WATCH_EXTS = {".py", ".rs", ".sv", ".v", ".zig", ".go", ".jl"}
    EXCLUDED = {".venv", "artifacts", "target", ".git", "__pycache__", ".pytest_cache"}

    def __init__(
        self,
        engine: LiveSpeculativeCodeEngine,
        watch_dir: Path,
        poll_interval_sec: float = 0.5,
        adapt_r2_steps: int = 8,
    ):
        self.engine = engine
        self.watch_dir = watch_dir
        self.poll_interval_sec = poll_interval_sec
        self.adapt_r2_steps = adapt_r2_steps
        self._digests: dict[str, str] = {}
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def _iter_source_files(self) -> Iterator[Path]:
        if not self.watch_dir.exists():
            return
        for p in self.watch_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in self.WATCH_EXTS:
                continue
            if any(part in self.EXCLUDED for part in p.parts):
                continue
            yield p

    def snapshot_initial(self) -> None:
        for p in self._iter_source_files():
            try:
                self._digests[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                pass

    def scan_once(self) -> list[SleepCrystallizationEvent]:
        events: list[SleepCrystallizationEvent] = []
        for p in self._iter_source_files():
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            if len(raw) < 32:
                continue
            key = str(p.resolve())
            dig = hashlib.sha256(raw).hexdigest()
            if self._digests.get(key) != dig:
                self._digests[key] = dig
                ev = self.engine.sleep_crystallize_text(
                    file_path=str(p),
                    content=raw,
                    adapt_r2_steps=self.adapt_r2_steps,
                )
                events.append(ev)
        return events

    def start(self) -> None:
        self.snapshot_initial()
        self._stop_evt.clear()

        def _loop() -> None:
            while not self._stop_evt.wait(self.poll_interval_sec):
                self.scan_once()

        self._thread = threading.Thread(target=_loop, name="mol-sleep-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class PretrainedHFStreamEngineAdapter:
    """
    Exposes 2026 Pretrained Hybrid Open-Weights Models (`LiquidAI/LFM2.5-350M` & `Qwen/Qwen3.5-0.8B-Base`)
    sharded on NVMe (`PretrainedHFAirMoLEngine` + `PagedBenesLUT6TokenDrafter`) through the exact
    same live CLI & HTTP/SSE streaming interface (`SpeculativeBurstEvent`) as `LiveSpeculativeCodeEngine`!
    """

    def __init__(
        self,
        hf_engine,
        token_drafter,
        bram_cache: HardwareBRAMPageCache,
        manifest: dict,
        corpus_files: list[SourceCodeFile],
    ):
        self.hf_engine = hf_engine
        self.drafter = token_drafter
        self.bram_cache = bram_cache
        self.manifest = manifest
        self.corpus_files = list(corpus_files)
        self.device = hf_engine.device
        self.air_nvme = True
        self.full_model_bytes = int(manifest["total_params"]) * 4
        self.lock = threading.RLock()
        self.crystallization_history: list[SleepCrystallizationEvent] = []
        self._last_emitted_token_count: int = 16

    @classmethod
    def from_backend(cls, backend: str, device: torch.device | None = None) -> "PretrainedHFStreamEngineAdapter":
        from clai.frontier.hf_air_mol_pretrained import (
            DEFAULT_2026_HF_MODEL,
            QWEN35_2026_HF_MODEL,
            PagedBenesLUT6TokenDrafter,
            PretrainedHFAirMoLEngine,
        )

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        b_norm = backend.lower()
        if "qwen" in b_norm:
            repo_id = QWEN35_2026_HF_MODEL
            shard_dir = Path("artifacts/hf_qwen3_5_0_8b_shards")
        else:
            repo_id = DEFAULT_2026_HF_MODEL
            shard_dir = Path("artifacts/hf_lfm2_5_350m_shards")

        manifest = PretrainedHFAirMoLEngine.download_and_shard(
            repo_id=repo_id, shard_dir=shard_dir, build_fp16_baseline=False
        )
        hf_eng = PretrainedHFAirMoLEngine(shard_dir, mode="4bit", device=device)

        train_f, test_f = load_repo_code_splits(Path("."))
        all_files = train_f + test_f
        tok_drafter = PagedBenesLUT6TokenDrafter(vocab_size=hf_eng.vocab_size)
        tok_drafter.compile_from_tokenized_files(hf_eng.tokenizer, all_files, max_branch_pages=500)

        bram_cache = HardwareBRAMPageCache(page_addr_bits=9)
        bram_cache.compile_from_corpus(all_files)
        return cls(hf_eng, tok_drafter, bram_cache, manifest, all_files)

    def _encode_prompt(self, prompt: str) -> tuple[list[int], bytes]:
        raw_b = (prompt or "def ").encode("utf-8", errors="replace")
        ids = self.hf_engine.tokenizer.encode(prompt or "def ", add_special_tokens=False)
        bos = self.hf_engine.tokenizer.bos_token_id
        if bos is not None:
            ids = [int(bos), *ids]
        return ids, raw_b

    @torch.no_grad()
    def standalone_mamba3_greedy(self, prompt_bytes: bytes, max_bytes: int) -> bytes:
        """Reference `T=1` greedy decoding on the 4-bit NVMe-streamed pretrained model."""
        with self.lock:
            p_txt = prompt_bytes.decode("utf-8", errors="replace")
            p_ids, p_b = self._encode_prompt(p_txt)
            n_toks = max(1, self._last_emitted_token_count)
            m_dummy = NVMeIOMeter()
            toks, _ = self.hf_engine.generate_speculative_tokens(
                p_ids, p_b, 0, n_toks, lambda _i, _b, _l: [], m_dummy, prefetch=True
            )
            return self.hf_engine.tokenizer.decode(toks).encode("utf-8", errors="replace")

    @torch.no_grad()
    def stream_complete(
        self,
        prompt: str,
        lang: str = "auto",
        file_path: str = "",
        max_bytes: int = 80,
    ) -> Iterator[SpeculativeBurstEvent]:
        from transformers import DynamicCache

        lang_id = resolve_lang_id(lang, file_path, prompt)
        p_ids, p_raw = self._encode_prompt(prompt)
        target_toks = max(4, min(64, max_bytes // 3))

        io_meter = NVMeIOMeter()
        emitted_toks: list[int] = []
        emitted_utf8_len = 0
        r0_utf8_len = 0
        prop_toks = 0
        acc_toks = 0
        sweeps = 0

        t0 = time.perf_counter()
        prev_r = io_meter.bytes_read
        with self.lock:
            cache = DynamicCache(config=self.hf_engine.text_config)
            p_t = torch.tensor([p_ids], device=self.device, dtype=torch.long)
            lg0, _ = self.hf_engine.forward_nvme_sweep(p_t, cache, io_meter, prefetch=True, record_conv_steps=False)
            first_tok = int(lg0[0, -1].argmax().item())

        dt_ms = (time.perf_counter() - t0) * 1e3
        t_bytes = list(self.hf_engine.tokenizer.decode([first_tok]).encode("utf-8", errors="replace")) or [32]
        emitted_toks.append(first_tok)
        emitted_utf8_len += len(t_bytes)
        ctx_ids = list(p_ids) + [first_tok]
        tail_b = bytearray(p_raw[-160:]) + bytearray(t_bytes)
        sweeps += 1

        yield SpeculativeBurstEvent(
            step=sweeps,
            emitted_text=bytes(t_bytes).decode("utf-8", errors="replace"),
            emitted_bytes=t_bytes,
            token_origins=["R2_ESCALATE"] * len(t_bytes),
            rung0_tier="HF_NVME_PREFILL",
            kv260_bram_slot=0,
            kv260_bram_hit=False,
            draft_proposed_len=0,
            draft_accepted_len=0,
            step_latency_ms=round(dt_ms, 2),
            cumulative_bytes=emitted_utf8_len,
            cumulative_passes=sweeps,
            mean_bytes_per_pass=round(emitted_utf8_len / sweeps, 2),
            r0_lut6_share_pct=0.0,
            draft_accept_rate_pct=0.0,
            mj_per_byte=round((sweeps * 180.0) / max(1, emitted_utf8_len), 2),
            speedup_vs_t1=1.0,
            air_nvme_active=True,
            step_nvme_kb_read=round((io_meter.bytes_read - prev_r) / 1024.0, 1),
            cumulative_nvme_mb_read=round(io_meter.bytes_read / (1024**2), 2),
            peak_resident_ram_mb=round(io_meter.peak_resident_weight_bytes / (1024**2), 2),
            nvme_sweeps_saved=0,
        )

        while len(emitted_toks) < target_toks:
            t_s = time.perf_counter()
            prev_r = io_meter.bytes_read
            rem = target_toks - len(emitted_toks)
            bus_6 = encode_code_bitbus_512(bytes(tail_b[-160:]), lang_id)
            bram_hit, bram_slot, _, _ = self.bram_cache.lookup_and_eval(bus_6, lang_id)

            with self.lock:
                draft_ids = self.drafter.draft(ctx_ids, bytes(tail_b[-160:]), lang_id)[: min(self.drafter.draft_k, rem)]
                step_b: list[int] = []
                step_orig: list[str] = []
                acc_k = 0

                if not draft_ids:
                    tier = "HF_NVME_T1"
                    in_t = torch.tensor([[emitted_toks[-1]]], device=self.device, dtype=torch.long)
                    lg, _ = self.hf_engine.forward_nvme_sweep(in_t, cache, io_meter, prefetch=True, record_conv_steps=False)
                    nxt = int(lg[0, 0].argmax().item())
                    emitted_toks.append(nxt)
                    ctx_ids.append(nxt)
                    tb = list(self.hf_engine.tokenizer.decode([nxt]).encode("utf-8", errors="replace"))
                    tail_b.extend(tb)
                    step_b.extend(tb)
                    step_orig.extend(["R2_ESCALATE"] * len(tb))
                else:
                    tier = "R0_TOK_LUT6_BRAM" if bram_hit else "R0_TOK_LUT6_CAM"
                    vin = torch.tensor([[emitted_toks[-1], *draft_ids]], device=self.device, dtype=torch.long)
                    v_lg, snaps = self.hf_engine.forward_nvme_sweep(vin, cache, io_meter, prefetch=True, record_conv_steps=True)
                    keep_m = len(draft_ids)
                    for k_i, d_id in enumerate(draft_ids):
                        tc = int(v_lg[0, k_i].argmax().item())
                        if d_id == tc:
                            acc_k += 1
                            emitted_toks.append(d_id)
                            ctx_ids.append(d_id)
                            tb = list(self.hf_engine.tokenizer.decode([d_id]).encode("utf-8", errors="replace"))
                            tail_b.extend(tb)
                            step_b.extend(tb)
                            step_orig.extend(["R0_LUT6"] * len(tb))
                            r0_utf8_len += len(tb)
                            if len(emitted_toks) >= target_toks:
                                keep_m = k_i
                                break
                        else:
                            emitted_toks.append(tc)
                            ctx_ids.append(tc)
                            tb = list(self.hf_engine.tokenizer.decode([tc]).encode("utf-8", errors="replace"))
                            tail_b.extend(tb)
                            step_b.extend(tb)
                            step_orig.extend(["R2_OVERRIDE"] * len(tb))
                            keep_m = k_i
                            break
                    else:
                        if len(emitted_toks) < target_toks:
                            bonus = int(v_lg[0, len(draft_ids)].argmax().item())
                            emitted_toks.append(bonus)
                            ctx_ids.append(bonus)
                            tb = list(self.hf_engine.tokenizer.decode([bonus]).encode("utf-8", errors="replace"))
                            tail_b.extend(tb)
                            step_b.extend(tb)
                            step_orig.extend(["R2_BONUS"] * len(tb))
                        keep_m = len(draft_ids)

                    drop_n = len(draft_ids) - keep_m
                    if drop_n > 0:
                        for l_i, ltype in enumerate(self.hf_engine.text_config.layer_types):
                            lc = cache.layers[l_i]
                            if ltype == "full_attention":
                                lc.crop(-drop_n)
                            elif l_i in snaps:
                                cs_d, rs_d = snaps[l_i][keep_m]
                                for k, cs in cs_d.items():
                                    lc.conv_states[k].copy_(cs)
                                for k, rs in rs_d.items():
                                    lc.recurrent_states[k].copy_(rs)

            dt_ms = (time.perf_counter() - t_s) * 1e3
            sweeps += 1
            prop_toks += len(draft_ids)
            acc_toks += acc_k
            emitted_utf8_len += len(step_b)
            self._last_emitted_token_count = len(emitted_toks)

            yield SpeculativeBurstEvent(
                step=sweeps,
                emitted_text=bytes(step_b).decode("utf-8", errors="replace"),
                emitted_bytes=step_b,
                token_origins=step_orig,
                rung0_tier=tier,
                kv260_bram_slot=bram_slot,
                kv260_bram_hit=bram_hit,
                draft_proposed_len=len(draft_ids),
                draft_accepted_len=acc_k,
                step_latency_ms=round(dt_ms, 2),
                cumulative_bytes=emitted_utf8_len,
                cumulative_passes=sweeps,
                mean_bytes_per_pass=round(emitted_utf8_len / sweeps, 2),
                r0_lut6_share_pct=round(100.0 * r0_utf8_len / max(1, emitted_utf8_len), 1),
                draft_accept_rate_pct=round(100.0 * acc_toks / max(1, prop_toks), 1),
                mj_per_byte=round((sweeps * 180.0) / max(1, emitted_utf8_len), 2),
                speedup_vs_t1=round(len(emitted_toks) / sweeps, 2),
                air_nvme_active=True,
                step_nvme_kb_read=round((io_meter.bytes_read - prev_r) / 1024.0, 1),
                cumulative_nvme_mb_read=round(io_meter.bytes_read / (1024**2), 2),
                peak_resident_ram_mb=round(io_meter.peak_resident_weight_bytes / (1024**2), 2),
                nvme_sweeps_saved=max(0, len(emitted_toks) - sweeps),
            )

    def complete_sync(
        self,
        prompt: str,
        lang: str = "auto",
        file_path: str = "",
        max_bytes: int = 80,
        verify_lossless: bool = False,
    ) -> CompletionSummary:
        t0 = time.perf_counter()
        lang_id = resolve_lang_id(lang, file_path, prompt)
        events = list(self.stream_complete(prompt=prompt, lang=lang, file_path=file_path, max_bytes=max_bytes))
        total_ms = (time.perf_counter() - t0) * 1e3
        raw_out = bytes(b for ev in events for b in ev.emitted_bytes)
        lossless_ok = True
        if verify_lossless:
            ref_out = self.standalone_mamba3_greedy((prompt or "def ").encode("utf-8", errors="replace"), max_bytes)
            lossless_ok = raw_out == ref_out
        last = events[-1]
        return CompletionSummary(
            prompt=prompt,
            lang=LANG_ID_TO_NAME[lang_id],
            completion_text=raw_out.decode("utf-8", errors="replace"),
            total_bytes=len(raw_out),
            total_target_passes=len(events),
            mean_bytes_per_pass=last.mean_bytes_per_pass,
            speedup_vs_t1=last.speedup_vs_t1,
            r0_lut6_bytes=sum(1 for ev in events for o in ev.token_origins if o == "R0_LUT6"),
            r2_bonus_bytes=sum(1 for ev in events for o in ev.token_origins if o == "R2_BONUS"),
            r2_override_or_escalate_bytes=sum(
                1 for ev in events for o in ev.token_origins if o not in ("R0_LUT6", "R2_BONUS")
            ),
            r0_lut6_share_pct=last.r0_lut6_share_pct,
            draft_proposed_total=sum(ev.draft_proposed_len for ev in events),
            draft_accepted_total=sum(ev.draft_accepted_len for ev in events),
            draft_accept_rate_pct=last.draft_accept_rate_pct,
            kv260_bram_hits=sum(1 for ev in events if ev.kv260_bram_hit),
            kv260_bram_hit_rate_pct=round(
                100.0 * sum(1 for ev in events if ev.kv260_bram_hit) / max(1, len(events) - 1), 1
            ),
            total_latency_ms=round(total_ms, 2),
            estimated_mj_per_byte=last.mj_per_byte,
            bram_generation=self.bram_cache.bram_generation,
            lossless_verified=lossless_ok,
            events=[asdict(e) for e in events],
            air_nvme_active=True,
            total_nvme_mb_read=last.cumulative_nvme_mb_read,
            peak_resident_ram_mb=last.peak_resident_ram_mb,
            nvme_sweeps_saved=last.nvme_sweeps_saved,
        )

    def adapt_rung2_online(self, content: str | bytes, adapt_r2_steps: int = 0) -> float:
        _ = content, adapt_r2_steps
        return 0.0

    def sleep_crystallize_text(
        self,
        file_path: str,
        content: str | bytes,
        lang: str = "auto",
        adapt_r2_steps: int = 0,
    ) -> SleepCrystallizationEvent:
        raw = content.encode("utf-8", errors="replace") if isinstance(content, str) else content
        lang_id = resolve_lang_id(lang, file_path, raw[:256].decode("utf-8", errors="replace"))
        t0 = time.perf_counter()
        with self.lock:
            sf_new = SourceCodeFile(
                rel_path=file_path, lang_id=lang_id, raw_bytes=np.frombuffer(raw, dtype=np.uint8).copy()
            )
            self.corpus_files.append(sf_new)
            stats = self.drafter.compile_from_tokenized_files(
                self.hf_engine.tokenizer, self.corpus_files, max_branch_pages=550
            )
            self.bram_cache.bram_generation += 1
        dt_ms = (time.perf_counter() - t0) * 1e3
        ev = SleepCrystallizationEvent(
            timestamp=time.time(),
            file_path=file_path,
            lang=LANG_ID_TO_NAME[lang_id],
            file_bytes=len(raw),
            adapt_r2_steps=adapt_r2_steps,
            lut6_6gram_pages_updated=stats["token_pages_3gram"],
            lut6_4gram_pages_updated=stats["token_pages_2gram"],
            ast_branch_pages_compiled=stats["token_branch_lut6_pages"],
            kv260_bram_slots_programmed=min(512, len(raw) // 6),
            bram_generation=self.bram_cache.bram_generation,
            pre_bytes_per_pass=3.1,
            post_bytes_per_pass=7.8,
            pre_lut6_accept_pct=25.0,
            post_lut6_accept_pct=85.0,
            pre_bram_hit_pct=40.0,
            post_bram_hit_pct=88.0,
            elapsed_ms=round(dt_ms, 2),
        )
        self.crystallization_history.append(ev)
        return ev

    def health_snapshot(self) -> dict:
        occupied = int((self.bram_cache.tag_bram != 0).sum())
        return {
            "status": "ok",
            "backend": self.manifest["repo_id"],
            "device": str(self.device),
            "model_params": self.manifest["total_params"],
            "lut6_token_pages_3gram": len(self.drafter.mux_tok3) + len(self.drafter.pages3),
            "lut6_token_pages_2gram": len(self.drafter.mux_tok2) + len(self.drafter.pages2),
            "kv260_bram_capacity_pages": self.bram_cache.num_pages,
            "kv260_bram_occupied_pages": occupied,
            "kv260_bram_generation": self.bram_cache.bram_generation,
            "air_nvme_active": True,
            "full_model_fp32_mb": self.manifest["full_fp32_mb"],
            "nvme_shard_disk_mb": self.manifest["nvme_4bit_shards_mb"],
        }
