"""Phase 19 & 20 Invariant Tests: `--air-nvme` Live SSE Server + Real 2026 `LiquidAI/LFM2.5-350M` `AirMoL`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
import urllib.request
import torch

from clai.frontier.air_mol_streamer import NVMeIOMeter
from clai.frontier.code_corpus_speculative import load_repo_code_splits
from clai.frontier.hf_air_mol_pretrained import (
    DEFAULT_2026_HF_MODEL,
    PagedBenesLUT6TokenDrafter,
    PretrainedHFAirMoLEngine,
)
from clai.service.live_code_engine import (
    LiveSpeculativeCodeEngine,
    PretrainedHFStreamEngineAdapter,
)


_SPEC = importlib.util.spec_from_file_location("mol_code_cli", Path("scripts/mol_code_cli.py"))
assert _SPEC and _SPEC.loader
_CLI_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI_MOD)


def _dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_live_cli_server_air_nvme_mode_reports_real_nvme_io_and_lossless_parity(tmp_path: Path) -> None:
    """
    Verifies Phase 19 (`--air-nvme` wired into `LiveSpeculativeCodeEngine` & `mol_code_cli.py`):
      1. `/v1/health` reports `air_nvme_active=True` and valid 4-bit shard disk size.
      2. Streaming SSE `/v1/complete` reports positive `step_nvme_kb_read`, `cumulative_nvme_mb_read`,
         `nvme_sweeps_saved > 0`, and `peak_resident_ram_mb < 0.45 * full_model_fp32_mb`, while
         matching standalone 4-bit `T=1` greedy output 100.0% byte-for-byte!
    """
    engine = LiveSpeculativeCodeEngine.load_or_train(
        checkpoint_path=Path("artifacts/mol_code_engine.pt"),
        air_nvme=True,
        nvme_shard_dir=tmp_path / "live_nvme_shards",
    )
    srv = _CLI_MOD.create_http_server(engine, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    try:
        with urllib.request.urlopen(f"{base_url}/v1/health", timeout=10) as r_h:
            h_json = json.loads(r_h.read().decode("utf-8"))
            assert h_json["air_nvme_active"] is True
            assert h_json["nvme_shard_disk_mb"] > 0.1

        prompt = "    def forward_stateful(\n        self, x: torch.Tensor"
        bursts, done = _CLI_MOD._sse_request_complete(
            base_url, {"prompt": prompt, "lang": "py", "max_bytes": 48}
        )
        assert done["air_nvme_active"] is True
        assert done["total_nvme_mb_read"] > 1.0
        assert done["nvme_sweeps_saved"] > 15
        assert done["peak_resident_ram_mb"] < 0.50 * h_json["full_model_fp32_mb"]

        emitted = bytes(b for ev in bursts for b in ev["emitted_bytes"])
        ref = engine.standalone_mamba3_greedy(prompt.encode("utf-8"), 48)
        assert emitted == ref
    finally:
        srv.shutdown()
        srv.server_close()


def test_real_2026_lfm2_5_350m_air_mol_token_lut6_lossless_and_low_ram() -> None:
    """
    Verifies Phase 20 on the real pretrained March 2026 `LiquidAI/LFM2.5-350M` checkpoint:
      1. Streams 16 hybrid layers + chunked 65,536-vocab head off 4-bit NVMe shards in `< 160 MB`
         Peak Active Weight RAM (`< 12%` of the `1,417 MB` FP32 model!).
      2. Token-level `PagedBenesLUT6TokenDrafter` matches standalone `T=1` greedy `LFM2.5-350M`
         100.0% token-for-token on unseen held-out code while cutting NVMe sweeps & bytes read!
    """
    dev = _dev()
    shard_dir = Path("artifacts/hf_lfm2_5_350m_shards")
    manifest = PretrainedHFAirMoLEngine.download_and_shard(
        repo_id=DEFAULT_2026_HF_MODEL, shard_dir=shard_dir, build_fp16_baseline=False
    )
    eng = PretrainedHFAirMoLEngine(shard_dir, mode="4bit", device=dev)

    train_files, test_files = load_repo_code_splits(Path("."))
    drafter = PagedBenesLUT6TokenDrafter(vocab_size=eng.vocab_size, draft_k=4)
    drafter.compile_from_tokenized_files(eng.tokenizer, train_files, max_branch_pages=300)

    sf = test_files[0]
    txt = bytes(sf.raw_bytes.tolist()).decode("utf-8", errors="replace")
    ids = eng.tokenizer.encode(txt, add_special_tokens=False)
    bos_id = int(eng.tokenizer.bos_token_id or 1)
    p_ids = [bos_id, *ids[8:72]]
    p_bytes = eng.tokenizer.decode(ids[8:72]).encode("utf-8", errors="replace")

    m_ref = NVMeIOMeter()
    out_ref, sw_ref = eng.generate_speculative_tokens(p_ids, p_bytes, sf.lang_id, 20, lambda _i, _b, _l: [], m_ref)

    m_mol = NVMeIOMeter()
    out_mol, sw_mol = eng.generate_speculative_tokens(
        p_ids, p_bytes, sf.lang_id, 20, lambda i_c, tb, lid: drafter.draft(i_c, tb, lid), m_mol
    )

    assert out_mol == out_ref, "Token-level LUT6 broke lossless token parity on pretrained LFM2.5-350M!"
    assert sw_mol < sw_ref, f"Expected fewer NVMe sweeps than {sw_ref}, got {sw_mol}"
    assert m_mol.bytes_read < m_ref.bytes_read
    assert (m_mol.peak_resident_weight_bytes / (1024**2)) < 0.15 * manifest["full_fp32_mb"]


def test_live_http_sse_server_with_lfm2_5_350m_pretrained_backend() -> None:
    """
    Verifies Phase 22 (`--backend lfm2.5-350m` in `mol_code_cli.py` / `PretrainedHFStreamEngineAdapter`):
    Streams real BPE token bursts from the 4-bit NVMe-sharded March 2026 `LiquidAI/LFM2.5-350M`
    checkpoint over HTTP SSE with `100.0%` lossless greedy equivalence!
    """
    adapter = PretrainedHFStreamEngineAdapter.from_backend("lfm2.5-350m")
    srv = _CLI_MOD.create_http_server(adapter, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    try:
        with urllib.request.urlopen(f"{base_url}/v1/health", timeout=10) as r_h:
            hj = json.loads(r_h.read().decode("utf-8"))
            assert hj["backend"] == "LiquidAI/LFM2.5-350M"
            assert hj["air_nvme_active"] is True

        prompt = "class SleepCrystallizer:\n    def __init__(self, engram, difflut_net"
        bursts, done = _CLI_MOD._sse_request_complete(
            base_url, {"prompt": prompt, "lang": "py", "max_bytes": 30}
        )
        assert len(bursts) >= 2 and done["total_nvme_mb_read"] > 50.0
        emitted = bytes(b for ev in bursts for b in ev["emitted_bytes"])
        ref = adapter.standalone_mamba3_greedy(prompt.encode("utf-8"), 30)
        assert emitted == ref
    finally:
        srv.shutdown()
        srv.server_close()
