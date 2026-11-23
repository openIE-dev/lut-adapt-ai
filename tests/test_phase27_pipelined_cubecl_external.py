"""Phase 27 Pytest Suite: 250 MHz Pipelined `xck26` RTL (`Claim LA-16`), Cross-Platform `CubeCL`
`INT4` MoE & `GatedDeltaNet` (`Claim LA-18`), `Rung 3d` `<think>` Ceiling (`Claim LA-17`), and
Official External `BigCodeBench v0.1.3` + `SWE-bench Verified` Benchmarks (`Claim LA-15`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from clai.frontier.external_public_benchmarks import (
    BCB_V013_EXPECTED_SHA256,
    SWEBENCH_VERIFIED_EXPECTED_SHA256,
    _sha256_file,
    apply_swe_candidate_edit,
    verify_bcb_candidate,
)


def test_pipelined_250mhz_kv260_rtl_artifacts_and_4x_ltp_collapse() -> None:
    csv_p = Path("artifacts/kv260_pipelined_250mhz_rtl.csv")
    meta_p = Path("artifacts/kv260_pipelined_250mhz_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    rows = {r["design_id"]: r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    assert meta["verilator_zero_bubble_stream"]["checked_beats"] == 256
    assert meta["verilator_zero_bubble_stream"]["errors"] == 0
    assert meta["verilator_zero_bubble_stream"]["max_zero_bubble_streak"] == 256
    assert meta["measured_j"] is None and meta["board_metered"] is False

    u1_ltp = int(rows["U1_Unpipelined_Paged_OOC"]["ltp_combinational_levels"])
    p4_ltp = int(rows["P4_Pipelined_250MHz_OOC_Ours"]["ltp_combinational_levels"])
    ax_row = rows["P4_AXIS_IOBSafe_250MHz_Top_Ours"]
    assert p4_ltp <= 20 and u1_ltp >= 3 * p4_ltp
    assert int(ax_row["top_iob_ports"]) <= 189
    assert int(ax_row["dsp48e2_slices"]) == 0
    assert int(ax_row["ramb36e2_blocks"]) == 81


def test_cubecl_cross_platform_q4_moe_and_deltanet_parity() -> None:
    csv_p = Path("artifacts/cubecl_cross_platform_moe_deltanet.csv")
    meta_p = Path("artifacts/cubecl_cross_platform_moe_deltanet_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    rows = list(csv.DictReader(csv_p.open(encoding="utf-8")))
    assert len(rows) == 6
    for r in rows:
        assert float(r["max_abs_fp32_err"]) < 1e-4
        if "Metal_GPU" in r["backend_target"]:
            assert float(r["speedup_vs_host_ref"]) > 25.0

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["measured_j"] is None and meta["board_metered"] is False


def test_official_bigcodebench_and_swebench_sha256_and_zero_leak_verifiers() -> None:
    bcb_p = hf_hub_download("bigcode/bigcodebench", "data/v0.1.3-00000-of-00001.parquet", repo_type="dataset")
    swe_p = hf_hub_download("princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet", repo_type="dataset")
    assert _sha256_file(bcb_p) == BCB_V013_EXPECTED_SHA256
    assert _sha256_file(swe_p) == SWEBENCH_VERIFIED_EXPECTED_SHA256

    r0 = pq.read_table(bcb_p).slice(0, 1).to_pylist()[0]
    ok_gold, _, _ = verify_bcb_candidate(
        r0["code_prompt"] + r0["canonical_solution"], r0["code_prompt"], r0["test"], r0["entry_point"]
    )
    ok_bad, _, _ = verify_bcb_candidate(
        "def task_func(*args, **kwargs):\n    return -999999\n", r0["code_prompt"], r0["test"], r0["entry_point"]
    )
    assert ok_gold and not ok_bad

    orig = "def f(x):\n    return x - 1\n"
    edited = apply_swe_candidate_edit(
        orig,
        "```python\n<<<<<<< SEARCH\n    return x - 1\n=======\n    return x + 1\n>>>>>>> REPLACE\n```",
    )
    assert edited == "def f(x):\n    return x + 1\n"


def test_external_public_and_rung3d_artifacts_soundness_invariants() -> None:
    r3d_p = Path("artifacts/frontier_ai_rung3d_ceiling.csv")
    ext_p = Path("artifacts/external_public_bcb_swebench.csv")
    assert r3d_p.is_file() and ext_p.is_file()

    r3d_rows = {r["arm"]: r for r in csv.DictReader(r3d_p.open(encoding="utf-8"))}
    arm_h = r3d_rows["H_OpenIE_MoL_Plus_Rung3d_Qwen3.6_35B_NativeThink_OURS"]
    assert float(arm_h["broken_commits_admitted_pct"]) == 0.0
    assert float(arm_h["overall_verified_pass_pct"]) >= 87.5

    ext_rows = list(csv.DictReader(ext_p.open(encoding="utf-8")))
    mol_rows = [r for r in ext_rows if "OpenIE_MoL" in r["arm"]]
    assert len(mol_rows) == 2
    for mr in mol_rows:
        assert float(mr["broken_outputs_admitted_pct"]) == 0.0
        assert float(mr["wave2_mol_fast_path_pct"]) > 0.0
