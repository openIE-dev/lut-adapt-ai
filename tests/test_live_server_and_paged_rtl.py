"""Phase 16 & 17 Invariant Tests: KV260 1-Cycle Multi-Page BRAM RTL + Live Streaming CLI/Server."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

from clai.frontier.benes_bram_page_cache import (
    verify_verilator_paged_bram_rtl,
    yosys_synth_stats,
)
from clai.service.live_code_engine import LiveSpeculativeCodeEngine

# Import HTTP server helper from `scripts/mol_code_cli.py`
import importlib.util

_SPEC = importlib.util.spec_from_file_location("mol_code_cli", Path("scripts/mol_code_cli.py"))
assert _SPEC and _SPEC.loader
_CLI_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI_MOD)


def test_kv260_paged_bram_rtl_verilator_and_yosys_zero_dsp(tmp_path: Path) -> None:
    """
    Verifies `clai_benes_paged_hotswap_core`:
      1. Back-to-back `CPI = 1.0` single-cycle (`4.0 ns`) BRAM `LUT6 INIT` page switching in Verilator (`0` errors)
      2. Exact `81` `RAMB36E2` Block RAM inference and `0` `DSP48`s in Yosys `synth_xilinx -family xcup`!
    """
    cosim = verify_verilator_paged_bram_rtl(tmp_path / "paged_cosim")
    assert cosim["errors"] == 0
    assert cosim["cpi"] == 1.0
    assert cosim["page_hits"] == 5
    assert cosim["cam_fallbacks"] == 2

    synth = yosys_synth_stats(["hw/kv260/rtl/clai_benes_hotswap_core.sv"], "clai_benes_paged_hotswap_core")
    assert synth["dsps"] == 0, f"Expected 0 DSP48s, got {synth['dsps']}"
    assert synth["ramb36"] == 81, f"Expected 81 RAMB36E2 Block RAMs (1 Tag + 80 LUT6 banks), got {synth['ramb36']}"
    assert synth["luts"] < 46_000, f"Unexpected CLB LUT inflation: {synth['luts']}"


def test_live_sse_server_and_online_file_sleep_crystallization() -> None:
    """
    Verifies `LiveSpeculativeCodeEngine` + HTTP/SSE Server:
      1. 100.0% lossless byte-exact equivalence over streaming `text/event-stream` SSE
      2. Live online `/v1/crystallize` on a newly edited domain file increases `bytes/pass` & BRAM generation!
    """
    engine = LiveSpeculativeCodeEngine.load_or_train(
        checkpoint_path=Path("artifacts/mol_code_engine.pt"),
        train_steps=180,
    )
    srv = _CLI_MOD.create_http_server(engine, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    try:
        # 1. Check `/v1/health` and HTML dashboard `/`
        with urllib.request.urlopen(f"{base_url}/", timeout=10) as r_ui:
            assert b"Mixture of Limits" in r_ui.read()
        with urllib.request.urlopen(f"{base_url}/v1/health", timeout=10) as r_h:
            h_json = json.loads(r_h.read().decode("utf-8"))
            assert h_json["status"] == "ok"
            gen0 = h_json["kv260_bram_generation"]

        # 2. Stream SSE completion and verify 100.0% lossless byte-for-byte parity
        prompt = "    def forward_parallel(self, x: torch.Tensor, chunk_q: int = 32)"
        bursts, done = _CLI_MOD._sse_request_complete(
            base_url, {"prompt": prompt, "lang": "py", "max_bytes": 48}
        )
        assert len(bursts) >= 1 and done["total_bytes"] == 48
        emitted_sse = bytes(b for ev in bursts for b in ev["emitted_bytes"])
        ref_greedy = engine.standalone_mamba3_greedy(prompt.encode("utf-8"), 48)
        assert emitted_sse == ref_greedy

        # 3. Wake Rung 2 Plasticity (`POST /v1/adapt_r2`) -> Stale Rung 0 SSE ->
        #    Online Sleep-Crystallization (`POST /v1/crystallize`) -> Crystallized Rung 0 SSE!
        edited_code = (
            "def compute_benes_parity_tag(slot_id: int, lane_mask: int) -> int:\n"
            "    folded_tag = ((slot_id * 131) ^ lane_mask) & 0xFFFF\n"
            "    return folded_tag ^ 0xA55A\n\n"
            "def verify_benes_parity_tag(slot_id: int, lane_mask: int) -> int:\n"
            "    folded_tag = ((slot_id * 131) ^ lane_mask) & 0xFFFF\n"
            "    return folded_tag ^ 0xA55A\n"
        )
        adapt_req = urllib.request.Request(
            f"{base_url}/v1/adapt_r2",
            data=json.dumps({"content": edited_code, "adapt_r2_steps": 28}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(adapt_req, timeout=30) as a_resp:
            assert json.loads(a_resp.read().decode("utf-8"))["status"] == "adapted_r2"

        probe_prefix = "def compute_benes_parity_tag(slot_id: int, lane_mask: int) -> "
        bursts_pre, pre_done = _CLI_MOD._sse_request_complete(
            base_url, {"prompt": probe_prefix, "lang": "py", "max_bytes": 56}
        )
        emitted_pre = bytes(b for ev in bursts_pre for b in ev["emitted_bytes"])

        cryst_req = urllib.request.Request(
            f"{base_url}/v1/crystallize",
            data=json.dumps(
                {
                    "file_path": "src/new_benes_tag.py",
                    "content": edited_code,
                    "lang": "py",
                    "adapt_r2_steps": 0,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(cryst_req, timeout=30) as c_resp:
            c_json = json.loads(c_resp.read().decode("utf-8"))

        assert c_json["bram_generation"] == gen0 + 1
        assert c_json["kv260_bram_slots_programmed"] >= 10
        assert c_json["post_bytes_per_pass"] > c_json["pre_bytes_per_pass"]

        bursts_post, post_done = _CLI_MOD._sse_request_complete(
            base_url, {"prompt": probe_prefix, "lang": "py", "max_bytes": 56}
        )
        emitted_post = bytes(b for ev in bursts_post for b in ev["emitted_bytes"])
        ref_post = engine.standalone_mamba3_greedy(probe_prefix.encode("utf-8"), 56)
        assert emitted_post == ref_post and emitted_post == emitted_pre
        assert post_done["mean_bytes_per_pass"] > pre_done["mean_bytes_per_pass"]
    finally:
        srv.shutdown()
        srv.server_close()
