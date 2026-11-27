"""Phase 43 Invariant & Regression Suite (`Claims LA-68..LA-71` + `130`-Theorem `Claim LA-19` Milestone):
  1. `Claim LA-68`: Live Official `Magnitude 0.3.5` (`Seismic`) vs. `llama.cpp` (`llama-server`) vs.
     `Magnitude 0.3.5 + LUT-Adapt` (`Rung-0 EngramSpanLUT6 + C_span` & `mol-agent-rs` `C(z)` gate)
     across `Apple M5 Max`, `NVIDIA Jetson AGX Thor` (physical `TI INA238`/`INA3221` DC shunts
     `measured_j`, `board_metered = true`), and `AMD Ryzen AI Max+ 395` (`Zen 5` `RAPL`) on SHA-256
     pinned `Moby-Dick` (`repeat` & `twist`, `0` raw book text persisted) and `18`-task agentic coding
     (`100.0%` verified, `0.00%` broken commits on `Wave 1` & `Wave 2`).
  2. `Claim LA-69`: Three-Tier `VRAM -> System RAM LinkMover -> F_NOCACHE NVMe` Predictive
     `Tiered-Expert-AirMoL` (`Reuse::expected_with` admission + `K = 3` multi-token `LUT6` lookahead).
  3. `Claim LA-70`: `DeepSeek-V4.1-Flash` Zero-`FLOP` `Engram-LUT6` (`144/144` `Yosys 0.71+` `abc -lut 6`
     `$lut` miter) & Cross-Silicon `Window 23` Physical `Thor` DC Shunt Telemetry.
  4. `Claim LA-71` & `Claim LA-19`: `130` Six-Kernel `Lean 4.36.0` Proofs (`1,494` Linux `bwrap` Fleet),
     `18`-Export `<= 16 KB` `lut6_core.wasm`, & Zero-`DSP48E1` `> 100 MHz` `openXC7` `XC7A200T` `.bit`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from clai.frontier.enigma_zerobench_codifier import (
    compile_truth_table_6bit_to_lut6_masks,
    eval_lut6_masks6,
)
from clai.frontier.magnitude_tiered_engram import (
    EngramSpanLut6Index,
    MOBY_DICK_EXPECTED_SHA256,
    build_agentic_coding_tasks,
    deterministic_repair_or_solve,
    tiered_residency_action2_py,
    verify_agentic_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la68_magnitude_head_to_head_and_agentic_cz_gate() -> None:
    paras = [
        "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse.",
        "There now is your insular city of the Manhattoes belted round by wharves as Indian isles by coral reefs.",
    ]
    idx = EngramSpanLut6Index(paras)
    span = idx.resolve_verbatim_span("There now is your insular city", 8, skip_anchor_words=6)
    assert span == "of the Manhattoes belted round by wharves as"

    tasks = build_agentic_coding_tasks()
    assert len(tasks) == 18
    for tk in tasks:
        gold = deterministic_repair_or_solve(tk)
        assert verify_agentic_candidate(tk, gold) is True
        assert verify_agentic_candidate(tk, "94") is False  # Hallucinated inverse mod 101 rejected!

    moby = list(csv.DictReader((ART_DIR / "phase43_magnitude_head_to_head.csv").open(encoding="utf-8")))
    meta = json.loads((ART_DIR / "phase43_magnitude_tiered_engram_meta.json").read_text(encoding="utf-8"))
    assert len(moby) == 18  # 3 hosts x 3 engines x 2 modes (repeat, twist)
    assert meta["moby_dick_gutenberg_sha256"] == MOBY_DICK_EXPECTED_SHA256
    assert meta["raw_book_or_prompts_persisted"] == 0

    for r in moby:
        if r["board_metered"] == "True":
            assert "Thor" in r["host_label"]
            assert r["measured_j"] != "" and r["reported_j"] == ""
        else:
            assert "Thor" not in r["host_label"]
            assert r["measured_j"] == "" and r["reported_j"] != ""
        if "LUT_Adapt" in r["engine"]:
            assert float(r["verbatim_word_accuracy_pct"]) == 100.0

    ag = list(csv.DictReader((ART_DIR / "phase43_magnitude_agentic_coding.csv").open(encoding="utf-8")))
    assert len(ag) == 5
    w1, w2 = ag[-2], ag[-1]
    assert float(w1["verified_pass_pct"]) == 100.0 and float(w1["broken_commit_admitted_pct"]) == 0.0
    assert float(w2["verified_pass_pct"]) == 100.0 and float(w2["broken_commit_admitted_pct"]) == 0.0
    # Unverified 2-bit/4-bit runs admit > 0% broken commits
    assert any(float(r["broken_commit_admitted_pct"]) > 0.0 for r in ag[:3])


def test_claim_la69_tiered_expert_airmol_reuse_and_multi_token_lookahead() -> None:
    masks = compile_truth_table_6bit_to_lut6_masks(
        lambda a: tiered_residency_action2_py(a & 3, (a >> 2) & 1, (a >> 3) & 3, (a >> 5) & 1)
    )
    for x in range(64):
        assert eval_lut6_masks6(masks, x) == tiered_residency_action2_py(
            x & 3, (x >> 2) & 1, (x >> 3) & 3, (x >> 5) & 1
        )

    rows = list(csv.DictReader((ART_DIR / "phase43_tiered_expert_airmol.csv").open(encoding="utf-8")))
    assert len(rows) == 12  # 3 sparse-MoE families x 4 strategies
    for fam in ("Qwen3.8-Flash-Next-180B-A6B", "DeepSeek-V4.1-Flash-552B-A16B", "Nemotron-3-Nano-30B-A3B"):
        sub = [r for r in rows if r["model_family"] == fam]
        s1, s2, s3, s4 = sub
        assert float(s4["effective_tok_per_s"]) > float(s3["effective_tok_per_s"]) > float(s2["effective_tok_per_s"]) > float(s1["effective_tok_per_s"])
        assert int(s3["thrash_evictions_prevented"]) > 0
        assert int(s4["thrash_evictions_prevented"]) > 0
        assert float(s4["nvme_mb_per_token"]) < float(s3["nvme_mb_per_token"])


def test_claim_la70_engram_lut6_yosys_miter_and_window23_thor_shunt() -> None:
    miter_rows = list(csv.DictReader((ART_DIR / "phase43_engram_lut6_miter.csv").open(encoding="utf-8")))
    assert sum(int(r["lut6_cells"]) for r in miter_rows) == 144
    assert sum(int(r["minterms_verified"]) for r in miter_rows) == 9216
    assert all(int(r["yosys_miter_mismatches"]) == 0 for r in miter_rows)

    w23_rows = list(csv.DictReader((ART_DIR / "phase43_window23_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w23_rows) == 3
    metered = [r["host_label"] for r in w23_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w23_rows:
        assert float(r["latency_speedup_x"]) > 1.0
        assert float(r["energy_reduction_x"]) > 1.0
        if r["board_metered"] == "True":
            assert r["w23_measured_nj_per_op_vdd_cpu"] != "" and r["w23_reported_nj_per_op"] == ""
        else:
            assert r["w23_measured_nj_per_op_vdd_cpu"] == "" and r["w23_reported_nj_per_op"] != ""


def test_claim_la71_openxc7_tiered_engram_18export_wasm_and_130_six_kernel_proofs() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_tiered_engram_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["tiered_engram_64minterm_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 130
    assert lean_meta["module_theorem_counts"].get("MagnitudeTieredEngram") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1494
    assert fleet_meta["all_six_kernels_passed_sandboxed"] is True

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const fnCount = Object.keys(ex).filter(k => k.startsWith('wasm_')).length;
  const argOk = (57) | (3 << 6) | (10 << 12) | (0 << 16) | (63 << 22);
  const rOk = ex.wasm_eval_tiered_engram_lut6(argOk >>> 0, 7) >>> 0;
  const argThrash = (57) | (3 << 6) | (10 << 12) | (43 << 16) | (57 << 22);
  const rThrash = ex.wasm_eval_tiered_engram_lut6(argThrash >>> 0, 7) >>> 0;
  const argBrk = argOk | (1 << 28);
  const rBrk = ex.wasm_eval_tiered_engram_lut6(argBrk >>> 0, 7) >>> 0;
  console.log(JSON.stringify({{ fnCount, rOk, rThrash, rBrk }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 18
    assert ((out["rOk"] >> 30) & 3) == 1
    assert ((out["rOk"] >> 26) & 0xF) == 11
    assert ((out["rOk"] >> 20) & 3) == 3
    assert ((out["rOk"] >> 18) & 3) == 3
    assert ((out["rThrash"] >> 20) & 3) == 0
    assert ((out["rThrash"] >> 18) & 3) == 0
    assert ((out["rBrk"] >> 30) & 3) == 2
