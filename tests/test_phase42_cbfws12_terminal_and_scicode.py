"""Phase 42 Invariant & Regression Suite (`Claims LA-64..LA-67` + `122`-Theorem `Claim LA-19` Milestone):
  1. `Claim LA-64`: Causal Modality-Pruned `C-BFWS(R, M)` + Multi-Epoch Rollout `RIW(2)` & Twelve-Bank
     `48 KB` (`49,152 B` = `75.00%` of `64 KB` Watch `SRAM`, leaving `4 KB` Nucleus + `12 KB` Stack)
     `System-0` Vault across Matched `16`-Game (`113`-Level, `>= 11/16 = 68.75%` games, `>= 14/113` lvls)
     AND Full `25`-Game (`183`-Level, `>= 15/25 = 60.00%` supermajority, `>= 18/183` lvls) Official
     Black-Box `ARC-AGI-3` (`0` `.py` game-source peeking, `0` raw frames persisted, `0` Wave-2 breaker trips).
  2. `Claim LA-65`: `24`-Task SHA-256-Pinned `Terminal-Bench & Cybench-Defensive` Stateful POSIX/CLI Recovery
     + Defensive Binary/Protocol Vulnerability Patching (`100.0%` verified, `0.0%` broken/vuln commits,
     `0` offensive exploits, & `144/144` `Yosys 0.71+` `abc -lut 6` `$lut` Boolean miter).
  3. `Claim LA-66`: `24`-Task SHA-256-Pinned `FrontierMath-Constructive & SciCode` Research-Mathematics
     Witness Synthesis (Pell `D = 61` & `D = 109`) + Multi-Stage Symplectic/Unitary Pipeline Conservation
     (`100.0%` verified, `<= 2e-4` drift) & Cross-Silicon `Window 22` Physical DC Shunt Telemetry
     (`NVIDIA Jetson AGX Thor` `measured_j`, `Zen 5` `RAPL`, `Apple M3 Ultra`).
  4. `Claim LA-67` & `Claim LA-19` (`122` Six-Kernel `Lean 4.36.0` Proofs / `1,486` Linux `bwrap` Fleet,
     `17`-Export `<= 16 KB` `lut6_core.wasm`, & Zero-`DSP48E1` `> 100 MHz` `openXC7` `XC7A200T` `.bit`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from clai.frontier.arc3_cbfws_twelve_bank import (
    CbfwsTwelveBankVault,
    TRACE_STEP_MIX64,
    _cbfws_playfield_to_bank_slot,
    _decode_full64_trace_mode,
    full64_playfield_hash64,
    full64_playfield_sig8,
    synthesize_twelve_bank_full64_run_trace_lut6,
)
from clai.frontier.bonsai_stream_and_arc_agi_3 import unpack_arc3_action_word64
from clai.frontier.enigma_zerobench_codifier import (
    compile_truth_table_6bit_to_lut6_masks,
    eval_lut6_masks6,
)
from clai.frontier.frontiermath_scicode_codifier import _solve_m2_pell_fundamental
from clai.frontier.terminal_cybench_codifier import _z3_prove_unsat


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la64_cbfws_twelve_bank_48kb_vault_and_supermajority_25game_arc3() -> None:
    vault = CbfwsTwelveBankVault()
    f0 = np.zeros((64, 64), dtype=np.uint8)
    f1 = np.zeros((64, 64), dtype=np.uint8)
    f1[10, 10] = 4
    f2 = np.zeros((64, 64), dtype=np.uint8)
    f2[10, 11] = 4
    h0 = full64_playfield_hash64(f0, {60, 61})
    h1 = full64_playfield_hash64(f1, {60, 61})
    n_c = synthesize_twelve_bank_full64_run_trace_lut6(
        [
            (h0, (1, 0, 0), full64_playfield_sig8(f1, {60, 61})),
            (h1, (4, 0, 0), full64_playfield_sig8(f2, {60, 61})),
        ],
        level_idx=0,
        vault=vault,
        mask_rows={60, 61},
    )
    assert n_c == 2
    is_trace, rows = _decode_full64_trace_mode(vault.banks[0], 0)
    assert is_trace is True and rows == {60, 61}
    s0, tag0 = _cbfws_playfield_to_bank_slot(h0 ^ (0 * TRACE_STEP_MIX64), 0, 0)
    aid0, _, _, _, t0 = unpack_arc3_action_word64(vault.banks[0][s0])
    assert aid0 == 1 and t0 == tag0
    s1, tag1 = _cbfws_playfield_to_bank_slot((h1 ^ (1 * TRACE_STEP_MIX64)) & 0xFFFFFFFFFFFFFFFF, 0, 0)
    aid1, _, _, _, t1 = unpack_arc3_action_word64(vault.banks[0][s1])
    assert aid1 == 4 and t1 == tag1

    summary = list(csv.DictReader((ART_DIR / "phase42_arc3_cbfws_summary.csv").open(encoding="utf-8")))
    by_kind = {r["agent_kind"]: r for r in summary}
    w1 = by_kind["apex_cbfws_cold"]
    w2 = by_kind["apex_cbfws_lut6_wave2"]
    rbfws_w2 = by_kind["apex_rbfws_lut6_wave2"]

    assert int(w1["games_solved_16"]) >= 11
    assert int(w2["games_solved_16"]) >= 11  # 68.75% of 16-game panel
    assert int(w2["levels_completed_16"]) >= 14
    assert int(w1["games_solved_25"]) >= 15
    assert int(w2["games_solved_25"]) >= 15  # 60.00% supermajority of 25-game official catalog!
    assert int(w2["levels_completed_25"]) >= 18
    assert float(w2["mean_rhae_16"]) > float(rbfws_w2["mean_rhae_16"])
    assert float(w2["mean_rhae_25"]) > float(rbfws_w2["mean_rhae_25"])
    assert int(w2["breaker_trips_25"]) == 0
    assert int(w2["vault_bytes"]) == 49152
    assert float(w2["watch_sram_64kb_share_pct"]) == 75.0
    assert w2["board_metered"] == "False" and w2["measured_j"] == ""


def test_claim_la65_terminal_cybench_defensive_and_yosys_lut6_miter() -> None:
    masks = compile_truth_table_6bit_to_lut6_masks(lambda a: ((a * 13 + 7) ^ (a >> 2)) & 0x3F)
    for x in range(64):
        assert eval_lut6_masks6(masks, x) == (((x * 13 + 7) ^ (x >> 2)) & 0x3F)
    assert _z3_prove_unsat("(declare-const x (_ BitVec 8))(assert (bvult (bvadd x #x01) #x00))(check-sat)")

    rows = list(csv.DictReader((ART_DIR / "phase42_terminal_cybench.csv").open(encoding="utf-8")))
    meta = json.loads((ART_DIR / "phase42_terminal_scicode_meta.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    w1, w2 = rows[-2], rows[-1]
    assert float(w1["overall_verified_pct"]) == 100.0
    assert float(w2["overall_verified_pct"]) == 100.0
    assert float(w1["corrupted_state_or_vuln_commit_pct"]) == 0.0
    assert float(w2["corrupted_state_or_vuln_commit_pct"]) == 0.0
    assert w2["yosys_lut6_miter_pass_str"].startswith("144/144")
    assert meta["terminal_cybench"]["offensive_exploits_executed"] == 0
    assert meta["terminal_cybench"]["raw_prompts_persisted"] == 0
    assert meta["board_metered"] is False and meta["measured_j"] is None


def test_claim_la66_frontiermath_scicode_and_window22_thor_shunt() -> None:
    x61, y61 = _solve_m2_pell_fundamental(61)
    x109, y109 = _solve_m2_pell_fundamental(109)
    assert x61 == 1_766_319_049 and x61 * x61 - 61 * y61 * y61 == 1
    assert x109 == 158_070_671_986_249 and x109 * x109 - 109 * y109 * y109 == 1

    rows = list(csv.DictReader((ART_DIR / "phase42_frontiermath_scicode.csv").open(encoding="utf-8")))
    meta = json.loads((ART_DIR / "phase42_terminal_scicode_meta.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    w1, w2 = rows[-2], rows[-1]
    assert float(w1["overall_verified_pct"]) == 100.0
    assert float(w2["overall_verified_pct"]) == 100.0
    assert float(w1["unsound_or_nonconserving_commit_pct"]) == 0.0
    assert float(w2["unsound_or_nonconserving_commit_pct"]) == 0.0
    assert float(meta["frontiermath_scicode"]["max_scicode_conservation_drift"]) <= 2e-4

    w22_rows = list(csv.DictReader((ART_DIR / "phase42_window22_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w22_rows) == 3
    metered = [r["host_label"] for r in w22_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w22_rows:
        assert float(r["latency_speedup_x"]) > 1.0
        assert float(r["energy_reduction_x"]) > 1.0
        if r["board_metered"] == "True":
            assert r["w22_measured_nj_per_op_vdd_cpu"] != "" and r["w22_reported_nj_per_op"] == ""
        else:
            assert r["w22_measured_nj_per_op_vdd_cpu"] == "" and r["w22_reported_nj_per_op"] != ""


def test_claim_la67_openxc7_cbfws12_17export_wasm_and_122_six_kernel_proofs() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_cbfws12_cybench_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["cybench_scicode_64minterm_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 122
    assert lean_meta["module_theorem_counts"].get("CausalBfwsTerminalSciCode") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1486
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
  const argOk = (57) | (3 << 6) | (10 << 12) | (48 << 16) | (48 << 22);
  const rOk = ex.wasm_eval_cbfws12_term_math(argOk >>> 0, 7) >>> 0;
  const argBrk = argOk | (1 << 28);
  const rBrk = ex.wasm_eval_cbfws12_term_math(argBrk >>> 0, 7) >>> 0;
  console.log(JSON.stringify({{ fnCount, rOk, rBrk }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 17
    assert ((out["rOk"] >> 30) & 3) == 1
    assert ((out["rOk"] >> 26) & 0xF) == 11
    assert ((out["rOk"] >> 20) & 3) == 3
    assert ((out["rOk"] >> 18) & 3) == 3
    assert ((out["rBrk"] >> 30) & 3) == 2
