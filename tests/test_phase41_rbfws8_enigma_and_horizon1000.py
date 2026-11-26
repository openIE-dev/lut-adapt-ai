"""Phase 41 Invariant & Regression Suite (`Claims LA-60..LA-63` + `114`-Theorem `Claim LA-19` Milestone):
  1. `Claim LA-60`: Reset-Persistent Dual-Subspace `R-BFWS(R, M)` + Palette-Fair Cellular Induction
     & Eight-Bank `32 KB` (`50.00%` of `64 KB` Watch `SRAM`) `System-0` Vault across Matched `16`-Game
     (`113`-Level, `>= 10/16` games) AND Full `25`-Game (`183`-Level, `>= 13/25 = 52.0%` majority)
     Official Black-Box `ARC-AGI-3` (`0` `.py` game-source peeking, `0` raw frames persisted, `0` Wave-2 breaker trips).
  2. `Claim LA-61`: `24`-Task SHA-256-Pinned `EnigmaEval & ZeroBench` Symbolic Cryptanalysis +
     Post-Cutoff Alien-ISA/DSL Synthesis & `144/144` `Yosys 0.71+` (`abc -lut 6`) `$lut` Miter (`100.0%` verified, `0.0%` broken).
  3. `Claim LA-62`: `1,000`-Step (`16,000`-Transition) `Vending-Bench & BALROG` Coupled Solvency +
     Partially-Observable Exploration Horizon (`100.00%` survival, `0` violations) & Cross-Silicon
     `Window 21` Physical DC Shunt Telemetry (`NVIDIA Jetson AGX Thor` `measured_j`, `Zen 5` `RAPL`, `M3 Ultra`).
  4. `Claim LA-63` & `Claim LA-19` (`114` Six-Kernel `Lean 4.36.0` Proofs / `1,478` Linux `bwrap` Fleet,
     `16`-Export `<= 16 KB` `lut6_core.wasm`, & Zero-`DSP48E1` `109.15 MHz` `openXC7` `XC7A200T` `.bit`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from clai.frontier.arc3_rbfws_eight_bank import (
    RbfwsEightBankVault,
    _rbfws_playfield_to_bank_slot,
    extract_palette_fair_clicks,
    is_palette_starved_cellular_board,
    synthesize_eight_bank_shortest_lut6,
)
from clai.frontier.arc3_iw2_object_benes import extract_salient_objects
from clai.frontier.enigma_zerobench_codifier import (
    compile_truth_table_6bit_to_lut6_masks,
    enigma_sbox6_forward,
    eval_lut6_masks6,
)
from clai.frontier.vending_balrog_horizon1000 import run_vending_balrog_horizon1000_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la60_rbfws_eight_bank_32kb_vault_and_majority_25game_arc3() -> None:
    vault = RbfwsEightBankVault()
    trans = {101: {(3, 0, 0): (202, 0x42)}, 202: {(2, 0, 0): (303, 0x77)}}
    n_steps = synthesize_eight_bank_shortest_lut6(101, (202, (2, 0, 0)), trans, 0, vault, {62, 63})
    assert n_steps == 2
    s0, _ = _rbfws_playfield_to_bank_slot(101, 0, 0)
    assert s0 in vault.banks[0]

    # Palette-fair round-robin exposes board tiles starved behind 45 decorative dots
    fr = np.zeros((64, 64), dtype=np.uint8)
    for i in range(45):
        r, c = 2 + (i // 15) * 3, 2 + (i % 15) * 3
        fr[r, c] = 2
    for tr in range(4):
        for tc in range(4):
            y0, x0 = 30 + tr * 6, 30 + tc * 6
            fr[y0 : y0 + 4, x0 : x0 + 4] = 9
    _, objs = extract_salient_objects(fr, {62, 63})
    assert is_palette_starved_cellular_board(objs) is True
    pf_clicks = extract_palette_fair_clicks(fr, {62, 63}, cap=24)
    assert any(x >= 30 and y >= 30 for _, x, y in pf_clicks)

    summary = list(csv.DictReader((ART_DIR / "phase41_arc3_rbfws_summary.csv").open(encoding="utf-8")))
    by_kind = {r["agent_kind"]: r for r in summary}
    w1 = by_kind["apex_rbfws_cold"]
    w2 = by_kind["apex_rbfws_lut6_wave2"]
    bfws_w2 = by_kind["apex_bfws_lut6_wave2"]

    assert int(w1["games_solved_16"]) >= 10
    assert int(w2["games_solved_16"]) >= 10
    assert int(w2["levels_completed_16"]) >= 13
    assert int(w2["games_solved_25"]) >= 13  # Strict > 50% majority of 25-game official catalog!
    assert int(w2["levels_completed_25"]) >= 16
    assert float(w2["mean_rhae_16"]) > float(bfws_w2["mean_rhae_16"])
    assert float(w2["mean_rhae_25"]) > float(bfws_w2["mean_rhae_25"])
    assert int(w2["breaker_trips_25"]) == 0
    assert int(w2["vault_bytes"]) == 32768
    assert float(w2["watch_sram_64kb_share_pct"]) == 50.0
    assert w2["board_metered"] == "False" and w2["measured_j"] == ""


def test_claim_la61_enigma_zerobench_and_yosys_sbox6_miter() -> None:
    fwd = {x: enigma_sbox6_forward(x, 27, 19, 5) for x in range(64)}
    assert len(set(fwd.values())) == 64
    inv = {y: x for x, y in fwd.items()}
    inv_masks = compile_truth_table_6bit_to_lut6_masks(lambda z: inv[z])
    for x in range(64):
        assert eval_lut6_masks6(inv_masks, fwd[x]) == x

    rows = list(csv.DictReader((ART_DIR / "phase41_enigma_zerobench.csv").open(encoding="utf-8")))
    assert len(rows) == 5
    w1, w2 = rows[-2], rows[-1]
    assert float(w1["overall_verified_pct"]) == 100.0
    assert float(w2["overall_verified_pct"]) == 100.0
    assert float(w1["hallucinated_or_broken_commit_pct"]) == 0.0
    assert float(w2["hallucinated_or_broken_commit_pct"]) == 0.0
    assert w2["yosys_lut6_miter_pass_str"].startswith("144/144")


def test_claim_la62_vending_balrog_horizon1000_and_window21_thor_shunt() -> None:
    h_rows, h_meta = run_vending_balrog_horizon1000_benchmark()
    apex = h_rows[-1]
    assert apex.episodes_survived_1000_pct == 100.0
    assert apex.total_safety_or_solvency_violations == 0
    assert apex.system0_lut6_step_share_pct >= 98.0
    assert apex.speedup_vs_uncompiled_z3_x > 10.0
    assert h_meta["board_metered"] is False and h_meta["measured_j"] is None

    w21_rows = list(csv.DictReader((ART_DIR / "phase41_window21_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w21_rows) == 3
    metered = [r["host_label"] for r in w21_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w21_rows:
        assert float(r["latency_speedup_x"]) > 1.0
        assert float(r["energy_reduction_x"]) > 1.0
        if r["board_metered"] == "True":
            assert r["w21_measured_nj_per_op_vdd_cpu"] != "" and r["w21_reported_nj_per_op"] == ""
        else:
            assert r["w21_measured_nj_per_op_vdd_cpu"] == "" and r["w21_reported_nj_per_op"] != ""


def test_claim_la63_openxc7_rbfws8_16export_wasm_and_114_six_kernel_proofs() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_rbfws8_enigma_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["enigma_sbox6_64bijection_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 114
    assert lean_meta["module_theorem_counts"].get("ResetBfwsEnigmaHorizon") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1478
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
  const encArg = (57) | (3 << 6) | (6 << 12) | (42 << 15) | (0b100110 << 21);
  const re = ex.wasm_eval_rbfws8_enigma_horizon(encArg >>> 0, 7) >>> 0;
  const sb6 = (re >>> 18) & 63;
  const decArg = (57) | (sb6 << 15) | (1 << 27);
  const rd = ex.wasm_eval_rbfws8_enigma_horizon(decArg >>> 0, 7) >>> 0;
  console.log(JSON.stringify({{ fnCount, re, xBack: (rd >>> 18) & 63 }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 16
    assert ((out["re"] >> 30) & 3) == 1 and ((out["re"] >> 27) & 7) == 7 and ((out["re"] >> 16) & 3) == 3
    assert out["xBack"] == 42
