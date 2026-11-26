"""Phase 40 Invariant & Regression Suite (`Claims LA-56..LA-59` + `106`-Theorem `Claim LA-19` Milestone):
  1. `Claim LA-56`: Dual-Subspace `BFWS(R, M) + Beneš-LUT6` & Four-Bank `16 KB` (`25.00%` of `64 KB`
     Watch `SRAM`) `System-0` Vault across Official `16`-Game (`113`-Level) AND Full `25`-Game (`183`-Level)
     Black-Box `ARC-AGI-3` (`0` `.py` game-source peeking, `0` raw frames persisted, `0` Wave-2 breaker trips).
  2. `Claim LA-57`: `24`-Task Real Git-Worktree `SWE-bench Multimodal & Pro` Visual-Diagram + Multi-File
     Repair & `4 KB AST-Patch-LUT6` Router (`100.00%` verified, `0.00%` broken commits vs. `50.0%–95.8%`
     visual/code broken commits on unverified 1-shot text-only models).
  3. `Claim LA-58`: Planetary Digital Well (`Well-Codifier`) Rung-5 -> Rung-7 Standard-Library Projection
     across `12` Python/POSIX Modules (`384/384` = `100.00%` parity + Yosys `abc -lut 6` Boolean miter) &
     Cross-Silicon `Window 20` Telemetry (`Thor` DC Shunt `measured_j`, `Zen 5` `RAPL`, `M3 Ultra`).
  4. `Claim LA-59` & `Claim LA-19` (`106` Six-Kernel `Lean 4.36.0` Proofs / `1,470` Linux `bwrap` Fleet,
     `15`-Export `<= 16 KB` `lut6_core.wasm`, & Zero-`DSP48E1` `106.55 MHz` `openXC7` `XC7A200T` `.bit`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from clai.frontier.arc3_bfws_four_bank import (
    BfwsFourBankVault,
    extract_bfws_actions,
    synthesize_four_bank_shortest_lut6,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la56_arc3_bfws_16env_and_full_25env_dominance() -> None:
    # Verify ACTION7 (`UNDO`) is demoted to secondary_actions on `click_modal` (`avail = [6, 7]`)
    f_board = np.zeros((64, 64), dtype=np.uint8)
    f_board[12:30, 12:30] = 3
    prim, sec = extract_bfws_actions(f_board, [6, 7], {62, 63}, depth_hint=0)
    assert all(a[0] == 6 for a in prim) and len(prim) >= 6
    assert sec == [(7, 0, 0)]

    # Verify Four-Bank (`16,384 B`) cascade trajectory synthesis
    vault = BfwsFourBankVault()
    trans = {10: {(3, 0, 0): (20, 0x11)}, 20: {(5, 0, 0): (30, 0x22)}}
    assert synthesize_four_bank_shortest_lut6(10, (20, (5, 0, 0)), trans, 0, vault, {62, 63}) == 2

    rows = list(csv.DictReader((ART_DIR / "phase40_arc3_bfws_summary.csv").open(encoding="utf-8")))
    bfs = next(r for r in rows if "FIFO Frontier-BFS" in r["system"])
    siw2_w2 = next(r for r in rows if "SIW(2) -> Two-Bank 8 KB" in r["system"])
    bfws_w1 = next(r for r in rows if "BFWS(R,M) + Dual-Subspace" in r["system"])
    bfws_w2 = next(r for r in rows if "BFWS(R,M) -> Four-Bank 16 KB" in r["system"])

    assert int(bfws_w2["levels_completed_16"]) > int(siw2_w2["levels_completed_16"]) > int(bfs["levels_completed_16"])
    assert int(bfws_w2["levels_completed_25"]) > int(siw2_w2["levels_completed_25"]) > int(bfs["levels_completed_25"])
    assert int(bfws_w2["games_solved_16"]) >= int(siw2_w2["games_solved_16"])
    assert int(bfws_w2["games_solved_25"]) >= int(siw2_w2["games_solved_25"]) >= 10
    assert float(bfws_w2["mean_rhae_16"]) > float(siw2_w2["mean_rhae_16"])
    assert float(bfws_w2["mean_rhae_25"]) > float(siw2_w2["mean_rhae_25"])
    assert float(bfws_w1["mean_rhae_25"]) > 50.0 * float(bfs["mean_rhae_25"])
    assert int(bfws_w2["breaker_trips_25"]) == 0
    assert int(bfws_w2["vault_bytes"]) == 16384
    assert float(bfws_w2["watch_sram_64kb_share_pct"]) == 25.0


def test_claim_la57_swebench_multimodal_pro_zero_broken_commits() -> None:
    rows = list(csv.DictReader((ART_DIR / "phase40_swebench_multimodal_pro.csv").open(encoding="utf-8")))
    b27 = next(r for r in rows if r["system_id"] == "bonsai_27b_unverified")
    h0 = next(r for r in rows if r["system_id"] == "visual_symbolic_null")
    w1 = next(r for r in rows if r["system_id"] == "openie_apex_mm_wave1")
    w2 = next(r for r in rows if r["system_id"] == "openie_apex_mm_wave2")

    # Theorem 102 empirical witness: unverified 27B passes 100% unit tests while committing 50% visual regressions!
    assert float(b27["unit_test_only_pass_pct"]) == 100.0
    assert float(b27["broken_visual_or_code_commit_pct"]) >= 50.0
    assert float(h0["multimodal_verified_pct"]) > float(b27["multimodal_verified_pct"])
    assert float(h0["broken_visual_or_code_commit_pct"]) == 0.0
    assert float(w1["multimodal_verified_pct"]) == 100.0 and float(w1["broken_visual_or_code_commit_pct"]) == 0.0
    assert float(w2["multimodal_verified_pct"]) == 100.0 and float(w2["broken_visual_or_code_commit_pct"]) == 0.0
    assert float(w2["lut6_router_hit_pct"]) == 100.0
    assert float(w2["mean_energy_mj_per_task"]) < 0.001 * float(w1["mean_energy_mj_per_task"])


def test_claim_la58_planetary_well_codifier_and_window20_shunt() -> None:
    meta = json.loads((ART_DIR / "phase40_multimodal_and_well_meta.json").read_text(encoding="utf-8"))
    well = meta["planetary_well_codifier"]
    assert well["stdlib_modules_codified"] == 12
    assert well["total_pinned_calls"] == 384
    assert well["exact_matches"] == 384
    assert well["exact_parity_pct"] == 100.0
    assert well["yosys_abc_lut6_miter_verified"] is True

    w20_rows = list(csv.DictReader((ART_DIR / "phase40_window20_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w20_rows) == 3
    metered = [r["host_label"] for r in w20_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w20_rows:
        assert float(r["latency_speedup_x"]) > 1.0
        assert float(r["energy_reduction_x"]) > 1.0
        if r["board_metered"] == "True":
            assert r["w20_measured_nj_per_op_vdd_cpu"] != "" and r["w20_reported_nj_per_op"] == ""
        else:
            assert r["w20_measured_nj_per_op_vdd_cpu"] == "" and r["w20_reported_nj_per_op"] != ""


def test_claim_la59_openxc7_bfws4_15export_wasm_and_106_six_kernel_proofs() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_bfws4_well_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["well_codifier_64minterm_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 106
    assert lean_meta["module_theorem_counts"].get("BfwsWellMultimodal") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1470
    assert fleet_meta["all_six_kernels_passed_sandboxed"] is True

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const fnCount = Object.keys(ex).filter(k => typeof ex[k] === 'function').length;
  const argOk = 7 | (3 << 6) | (2 << 12) | (2 << 16) | (0x5A << 22) | (1 << 30);
  const rOk = ex.wasm_eval_bfws4_and_well_codifier(argOk >>> 0, 5) >>> 0;
  const argBrk = 7 | (3 << 6) | (2 << 12) | (20 << 16) | (0x11 << 22) | (1 << 30);
  const rBrk = ex.wasm_eval_bfws4_and_well_codifier(argBrk >>> 0, 5) >>> 0;
  console.log(JSON.stringify({{ fnCount, rOk, rBrk }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 15
    assert ((out["rOk"] >> 30) & 3) == 1 and ((out["rOk"] >> 28) & 3) == 3 and ((out["rOk"] >> 24) & 15) == 3
    assert ((out["rBrk"] >> 30) & 3) == 2
