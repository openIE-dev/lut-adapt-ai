"""Phase 28 Pytest Suite:
  - Claim LA-19: Lean 4.36.0 Machine-Checked Formal Proofs (`proofs/`, 27 theorems, 0 `sorry`, 0 custom axioms)
  - Claim LA-20: Multi-Repository Upstream `SWE-bench_Verified` (`django`, `pytest`, `requests`, `sympy`) + Call-Graph Localizer
  - Claim LA-21: Measured `120B`-Scale Hybrid Sparse-MoE (`Nemotron-3-Super-120B-A12B`) & `Devstral-2-24B` Apex Cascade
  - Claim LA-22: `CE`-Gated (`ENARDEN` + Valid-Bubble `FDRE`) `250 MHz` `P4-AXIS` Kria KV260 Verilator `VCD` Power Audit
  - Claim LA-23: Zero-Python Single-Binary Native Rust `mol-agent-rs` (`MCP` + `/v1/chat/completions` + `CubeCL` GPU)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile

from clai.frontier.multirepo_swebench import zero_leak_localize_multirepo_callgraph


def test_claim_la19_lean4_formal_proofs_zero_sorry_zero_custom_axioms() -> None:
    csv_p = Path("artifacts/lean4_formal_proofs_summary.csv")
    meta_p = Path("artifacts/lean4_formal_proofs_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_p.open(encoding="utf-8")))
    assert meta["total_kernel_checked_theorems"] >= 25
    assert len(rows) == meta["total_kernel_checked_theorems"]
    assert meta["sorry_count"] == 0
    assert meta["custom_axiom_count"] == 0
    assert meta["measured_j"] is None and meta["board_metered"] is False

    lean_files = list(Path("proofs").rglob("*.lean"))
    assert len(lean_files) >= 5
    for lf in lean_files:
        src = lf.read_text(encoding="utf-8")
        no_comments = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
        no_comments = re.sub(r"--[^\n]*", "", no_comments)
        assert not re.search(r"\bsorry\b", no_comments), f"Found `sorry` in {lf}"
        assert not re.search(r"^\s*axiom\b", no_comments, flags=re.MULTILINE), f"Found custom `axiom` in {lf}"


def test_claim_la20_multirepo_swebench_verified_and_callgraph_localizer() -> None:
    # 1. Unit-test zero-leakage multi-window AST call-graph localizer on synthetic 2-window module
    with tempfile.TemporaryDirectory(prefix="cg_loc_") as td:
        wt = Path(td)
        (wt / "requests").mkdir()
        (wt / "tests").mkdir()
        lines = ["def _private_helper(url: str) -> str:", "    return url.strip()", ""]
        lines += [f"def pad_fn_{i}(x):\n    return x + {i}\n" for i in range(45)]
        lines += [
            "def prepare_proxy_url(raw_url: str) -> str:",
            "    cleaned = _private_helper(raw_url)",
            "    return cleaned.lower()",
        ]
        (wt / "requests" / "utils.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (wt / "tests" / "test_utils.py").write_text(
            "from requests.utils import prepare_proxy_url\n", encoding="utf-8"
        )
        rel_f, snippet, n_win = zero_leak_localize_multirepo_callgraph(
            "psf/requests",
            wt,
            "Bug in `prepare_proxy_url` when whitespace URL is passed.",
            ["tests/test_utils.py"],
            'File "requests/utils.py", line 140, in prepare_proxy_url\nAssertionError',
        )
        assert rel_f == "requests/utils.py"
        assert n_win == 2
        assert "def prepare_proxy_url" in snippet and "def _private_helper" in snippet

    # 2. Verify saved multi-repo SWE-bench Verified benchmark artifacts
    csv_p = Path("artifacts/multirepo_swebench_verified.csv")
    json_p = Path("artifacts/multirepo_swebench_verified_traces.json")
    assert csv_p.is_file() and json_p.is_file()

    rows = {r["arm"]: r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    data = json.loads(json_p.read_text(encoding="utf-8"))
    meta = data["meta"]

    assert meta["zero_gold_leakage_enforced"] is True
    assert meta["multi_window_ast_callgraph_localizer"] is True
    assert meta["measured_j"] is None and meta["board_metered"] is False
    assert set(meta["repositories"]) == {"django/django", "pytest-dev/pytest", "psf/requests", "sympy/sympy"}

    ours = rows["E_OpenIE_MoL_MultiRepo_MCP_Agent_OURS"]
    assert int(ours["distinct_repos"]) == 4
    assert int(ours["total_worktree_evals"]) == 24
    assert float(ours["broken_patches_admitted_pct"]) == 0.0
    assert float(ours["wave2_mol_fast_path_pct"]) > 0.0
    assert float(ours["overall_verified_pass_pct"]) >= float(rows["C_Qwen3.6_35B_A3B_MetalAirMoL_1Shot"]["overall_verified_pass_pct"])


def test_claim_la21_measured_120b_hybrid_moe_and_24b_apex_cascade() -> None:
    csv_p = Path("artifacts/frontier_120b_hybrid_moe_apex.csv")
    json_p = Path("artifacts/frontier_120b_hybrid_moe_apex_traces.json")
    assert csv_p.is_file() and json_p.is_file()

    rows = {r["arm_id"]: r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    meta = json.loads(json_p.read_text(encoding="utf-8"))["meta"]

    assert meta["replaces_section6_70b_projection_caveat"] is True
    assert meta["measured_j"] is None and meta["board_metered"] is False

    nem = rows["B_NVIDIA_Nemotron3_Super_120B_A12B_1Shot"]
    ours = rows["C_OpenIE_MoL_Apex_24B_120B_Cascade_OURS"]
    assert float(nem["total_params_b"]) == 120.0
    assert float(nem["active_params_b"]) == 12.0
    assert float(nem["mean_decode_tok_per_s"]) > 10.0
    assert float(nem["mean_reported_j_per_tok"]) > 0.0
    assert float(ours["broken_commits_admitted_pct"]) == 0.0
    assert float(ours["wave2_mol_fast_path_pct"]) > 0.0
    assert float(ours["overall_verified_pct"]) >= float(nem["overall_verified_pct"])


def test_claim_la22_ce_gated_kv260_rtl_vcd_power_audit() -> None:
    csv_p = Path("artifacts/kv260_ce_gated_vcd_power.csv")
    meta_p = Path("artifacts/kv260_ce_gated_vcd_power_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    rows = {(r["design_variant"], r["workload"]): r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["measured_j"] is None and meta["board_metered"] is False
    assert meta["verilator_checked_valid_outputs"] == 320
    assert meta["bit_exact_ce0_vs_ce1_parity"] is True

    idle_ce = rows[("P4_AXIS_CE_Gated_Ours", "idle_quiescent_256")]
    axis_ce = rows[("P4_AXIS_CE_Gated_Ours", "axis_beat_shift_512")]
    assert float(idle_ce["verilator_parity_pct"]) == 100.0
    assert float(idle_ce["power_reduction_vs_ungated_x"]) >= 3.5
    assert float(axis_ce["power_reduction_vs_ungated_x"]) >= 2.0
    assert int(idle_ce["ltp_sync_levels"]) <= 15
    assert int(idle_ce["top_iob_ports"]) <= 189
    assert int(idle_ce["ramb36e2_blocks"]) == 81


def test_claim_la23_zero_python_rust_mol_agent_server_parity() -> None:
    csv_p = Path("artifacts/rust_mol_agent_server_bench.csv")
    meta_p = Path("artifacts/rust_mol_agent_server_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    rows = {r["endpoint_arm"]: r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["measured_j"] is None and meta["board_metered"] is False
    assert int(meta["loaded_crystallized_skills"]) >= 17

    rust_mcp = rows["MCP_JSONRPC_SkillLookup_Rust_Ours"]
    assert rust_mcp["zero_python_deps"] == "true"
    assert float(rust_mcp["speedup_vs_python_mcp"]) >= 10.0
    assert float(rust_mcp["mean_loopback_latency_ms"]) < 1.0

    # Live `--mcp-stdio` round-trip check against compiled native Rust binary
    bin_p = Path("sys/rust/target/release/mol_agent")
    assert bin_p.is_file()
    reqs = (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        + "\n"
        + json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "mol_lookup_crystallized_skill",
                    "arguments": {
                        "fn_name": "route_benes_layer4",
                        "specification": "Route a 4x4 Waksman Benes 6-switch permutation network",
                    },
                },
            }
        )
        + "\n"
    )
    proc = subprocess.run(
        [str(bin_p), "--mcp-stdio", "--mol-dir", "artifacts/.mol_verified_skills_r3d"],
        input=reqs,
        text=True,
        capture_output=True,
        timeout=5.0,
        check=True,
    )
    out_lines = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(out_lines) == 2
    assert len(out_lines[0]["result"]["tools"]) >= 3
    code_txt = out_lines[1]["result"]["content"][0]["text"]
    assert "def route_benes_layer4" in code_txt
