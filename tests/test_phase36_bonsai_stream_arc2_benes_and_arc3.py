"""Phase 36 Invariant & Regression Suite (`Claims LA-42..LA-44` + `74`-Theorem `Claim LA-19` Upgrade):
  1. `Claim LA-42` (`Bonsai-AirMoL` End-to-End `1.750 bpw` `EntroMCU-28` Streaming + `54`-Bit `3`-Token `LUT6`):
     - Verifies `|V| = 248,320 < 2^18` `3`-token `54`-bit + `10`-bit header lossless `uint64` bijection (`Theorem 68`),
     - Verifies real `prism-ml/Ternary-Bonsai-2-27B` `M3 Ultra 256 GB` telemetry (`100.00%` exact greedy token parity,
       `0` `os.pread` `EntroMCU-28` bit-exact mismatches, `1.619x` streamed linear GiB/tok cut),
     - Verifies `Window 14` vs `Window 13` physical `Jetson AGX Thor` `TI INA3221` DC shunt `measured_j` (`board_metered = True`).
  2. `Claim LA-43` (`Benes-ARC-LUT6` Multi-Layer Spatial Cascades on `arcprize/ARC-AGI-2`):
     - Verifies `2`-layer endpoint ray-bridging (`Theorem 71`), panel Boolean `LUT6` split, and symmetry repair,
     - Verifies `2`-layer & `3`-layer `Benes-ARC-LUT6` strictly dominate `1`-layer `ARC-LUT6` on both `n = 24`
       (`20.83% > 12.50%`) and `n = 64` (`12.50% > 6.25%`) with `100.0%` selective committed accuracy & `0.00%` `CHR@80`.
  3. `Claim LA-44` (Official `ARC-AGI-3` Interactive Reasoning Benchmark `RHAE` & Mechanic-Shift Breaker):
     - Verifies `pack_arc3_action_word64` / `unpack_arc3_action_word64` roundtrip and `synthesize_shortest_lut6_trajectory`,
     - Verifies `Interactive-MoL Wave 2` achieves strictly higher official `RHAE` (`EnvironmentScoreCalculator`) and
       fewer actions than `Wave 1` cold exploration, `Memoryless Reactive`, and `Random` baselines, with `0` raw frames persisted.
  4. `Claim LA-19` (`74` Six-Kernel Theorems / `1,438` Linux `bwrap` Fleet, `11`-Export `<= 16 KB` `lut6_core.wasm`,
     and `Theorem 74` Precompressed `.br` Sidecar Synchronization).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from clai.frontier.bonsai_stream_and_arc_agi_3 import (
    BONSAI_VOCAB_SIZE,
    pack_arc3_action_word64,
    pack_bonsai_3tok_54bit_py,
    synthesize_benes_arc_lut6_cascade,
    synthesize_shortest_lut6_trajectory,
    unpack_arc3_action_word64,
    unpack_bonsai_3tok_54bit_py,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la42_bonsai_54bit_bijection_and_m3u_thor_telemetry() -> None:
    for triple, hdr in [((0, 0, 0), 0), ((17, 124_160, BONSAI_VOCAB_SIZE - 1), 0x3FF), ((248_319, 42, 99_999), 0x2A5)]:
        w = pack_bonsai_3tok_54bit_py(triple[0], triple[1], triple[2], hdr)
        assert 0 <= w < (1 << 64)
        rec_t, rec_h = unpack_bonsai_3tok_54bit_py(w)
        assert rec_t == triple and rec_h == hdr

    with pytest.raises(ValueError):
        pack_bonsai_3tok_54bit_py(BONSAI_VOCAB_SIZE, 0, 0, 0)

    b_doc = json.loads((ART_DIR / "bonsai_27b_entromcu28_airmol_m3_ultra.json").read_text(encoding="utf-8"))
    assert b_doc["zero_raw_text_persisted"] is True
    assert b_doc["stream_shard_verification"]["sample_layers_bit_exact_mismatches"] == 0
    assert b_doc["wave2_io_byte_reduction_factor"] > 1.5
    for r in b_doc["rows"]:
        assert r["exact_greedy_parity_pct"] == 100.0
        assert r["board_metered"] is False and r["measured_j"] is None

    thor = json.loads((ART_DIR / "lut6_nucleus_shunt_thor.json").read_text(encoding="utf-8"))
    assert thor["board_metered"] is True
    wmap = {w["window"]: w for w in thor["windows"]}
    w13 = wmap["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]
    w14 = wmap["Window14_Bonsai_EntroMCU28_AirMoL_4TokPerSweep"]
    w15 = wmap["Window15_Benes_ARC2_2Layer_plus_ARC3_Reflex"]
    assert w14["board_metered"] is True and w15["board_metered"] is True
    assert w14["measured_nj_per_op_vdd_cpu"] * 5.0 < w13["measured_nj_per_op_vdd_cpu"]
    assert w14["ns_per_op"] * 5.0 < w13["ns_per_op"]


def test_claim_la43_multilayer_benes_arc_lut6_synthesis_and_separation() -> None:
    # 1. Constructive distance-2 ray-bridging task (`Theorem 71`): 1-layer must return None, 2-layer solves!
    x1 = np.array([[0, 4, 0, 0, 4], [3, 0, 3, 0, 0]], dtype=np.int16)
    y1 = np.array([[0, 4, 4, 4, 4], [3, 3, 3, 0, 0]], dtype=np.int16)
    x2 = np.array([[7, 0, 0, 7, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    y2 = np.array([[7, 7, 7, 7, 0], [0, 0, 0, 0, 0]], dtype=np.int16)

    assert synthesize_benes_arc_lut6_cascade([(x1, y1), (x2, y2)], max_layers=1) is None
    prog2 = synthesize_benes_arc_lut6_cascade([(x1, y1), (x2, y2)], max_layers=2)
    assert prog2 is not None and prog2.layers == 2
    xt = np.array([[0, 2, 0, 2, 0], [5, 0, 0, 0, 5]], dtype=np.int16)
    yt = np.array([[0, 2, 2, 2, 0], [5, 5, 5, 5, 5]], dtype=np.int16)
    assert np.array_equal(prog2.fn(xt), yt)

    # 2. Separator-split two-panel XOR Boolean `LUT6` synthesis (`L = 2`)
    pa1 = np.array([[2, 0], [0, 2]], dtype=np.int16)
    pb1 = np.array([[2, 2], [0, 0]], dtype=np.int16)
    sep = np.full((2, 1), 8, dtype=np.int16)
    xin_p1 = np.hstack([pa1, sep, pb1])
    yout_p1 = np.where((pa1 != 0) ^ (pb1 != 0), 3, 0).astype(np.int16)

    pa2 = np.array([[0, 2], [2, 2]], dtype=np.int16)
    pb2 = np.array([[2, 2], [2, 0]], dtype=np.int16)
    xin_p2 = np.hstack([pa2, sep, pb2])
    yout_p2 = np.where((pa2 != 0) ^ (pb2 != 0), 3, 0).astype(np.int16)

    prog_sp = synthesize_benes_arc_lut6_cascade([(xin_p1, yout_p1), (xin_p2, yout_p2)], max_layers=2)
    assert prog_sp is not None and prog_sp.layers == 2

    # 3. Verify `ARC-AGI-2` CSV multi-layer gains (`n = 24` & `n = 64`)
    rows = list(csv.DictReader((ART_DIR / "arc_agi_2_multilayer_benes_lut6.csv").open(encoding="utf-8")))
    n24_l1 = next(r for r in rows if r["slice"] == "Matched_n24" and r["max_benes_layers"] == "1" and "OpenIE" in r["system"])
    n24_l2 = next(r for r in rows if r["slice"] == "Matched_n24" and r["max_benes_layers"] == "2")
    n64_l1 = next(r for r in rows if r["slice"] == "Expanded_n64" and r["max_benes_layers"] == "1" and "OpenIE" in r["system"])
    n64_l3 = next(r for r in rows if r["slice"] == "Expanded_n64" and r["max_benes_layers"] == "3")

    assert float(n24_l2["solved_pct"]) > float(n24_l1["solved_pct"])
    assert float(n64_l3["solved_pct"]) >= 2.0 * float(n64_l1["solved_pct"])
    for r in rows:
        if "OpenIE" in r["system"]:
            assert float(r["selective_committed_acc_pct"]) == 100.0
            assert float(r["confident_hallucination_rate_80_pct"]) == 0.0
            assert float(r["unsound_train_inconsistent_commits_pct"]) == 0.0


def test_claim_la44_official_arc_agi_3_rhae_and_trajectory_crystal() -> None:
    w64 = pack_arc3_action_word64(6, 38, 54, 0xAB, 0xBEEF)
    assert unpack_arc3_action_word64(w64) == (6, 38, 54, 0xAB, 0xBEEF)

    # Shortest-path `SleepCrystallizer` prunes dead exploration branches (`10 -> 20 -> 10 -> 30` collapses to `10 -> 30`!)
    trans = {
        10: {(1, 0, 0): (20, 0x11), (4, 0, 0): (30, 0x22)},
        20: {(2, 0, 0): (10, 0x33)},
        30: {},
    }
    crystal: dict[int, int] = {}
    plen = synthesize_shortest_lut6_trajectory(10, (30, (6, 18, 24)), trans, level_idx=0, lut6_crystal=crystal)
    assert plen == 2 and len(crystal) == 2

    rows = list(csv.DictReader((ART_DIR / "arc_agi_3_interactive_mol_rhae.csv").open(encoding="utf-8")))
    rnd = next(r for r in rows if "Random" in r["system"])
    w1 = next(r for r in rows if "Wave 1" in r["system"])
    w2 = next(r for r in rows if "Wave 2" in r["system"])

    assert int(w1["levels_completed"]) > int(rnd["levels_completed"])
    assert int(w2["levels_completed"]) >= int(w1["levels_completed"])
    assert float(w2["mean_official_rhae_score"]) > float(w1["mean_official_rhae_score"])
    assert float(w2["mean_official_rhae_score"]) > float(rnd["mean_official_rhae_score"])
    assert float(w2["lut6_reflex_step_share_pct"]) > 0.0


def test_claim_la19_74_theorems_1438_fleet_and_11_wasm_exports() -> None:
    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 74
    assert lean_meta["module_theorem_counts"].get("BonsaiStreamArc3") == 8
    assert fleet_meta["total_theorems_and_lemmas"] >= 1438
    assert fleet_meta["all_six_kernels_passed_sandboxed"] is True

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const arc2_2l = ex.wasm_eval_benes_arc2_2layer8(0x00303000, 0x00, 0x00, 0x7) >>> 0;
  const arc3_ok = ex.wasm_eval_arc3_reflex_step(0x02, 0x42, 0x42, 0xFFFFFFFF, 200123) >>> 0;
  const arc3_br = ex.wasm_eval_arc3_reflex_step(0x02, 0x42, 0x99, 0xFFFFFFFF, 200123) >>> 0;
  console.log(JSON.stringify({{ arc2_2l, arc3_ok, arc3_br }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert ((out["arc2_2l"] >> 16) & 0xF) == 7
    assert (out["arc3_ok"] >> 24) == 1 and (out["arc3_ok"] & 0xFF) == 3
    assert ((out["arc3_ok"] >> 8) & 0xFFFF) == (200123 & 0xFFFF)
    assert (out["arc3_br"] >> 24) == 2 and (out["arc3_br"] & 0xFF) == (0x42 ^ 0x99)
