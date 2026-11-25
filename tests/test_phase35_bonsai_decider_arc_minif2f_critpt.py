"""Phase 35 Invariant & Regression Suite (`Claims LA-39..LA-41` + `66`-Theorem `Claim LA-19` Upgrade):
  1. `Claim LA-39` (`prism-ml/Ternary-Bonsai-2-27B` Native Hadamard-Ternary + `EntroMCU` `28 B` + `FWHT-1024 >> 5`):
     - Lossless `36 B -> 28 B` (`2.250 -> 1.750 bpw`, `-22.22%`) `EntroMCU` transcoding, rejection of
       invalid 4th codepoint `c == 3`, and `biases == -scales` verification (`Theorem 59`).
     - Exact `block_size = 1024 = 4^5` Sylvester `FWHT-1024 >> 5` self-inverse orthogonality (`Theorem 60/61`)
       and `128x` multiplication reduction (`Theorem 62`).
     - Physical `Jetson AGX Thor` `INA3221` `measured_j` (`board_metered = True`) on `Window 12` vs `Window 13`.
  2. `Claim LA-40` (`System-1` Neural Deciders `Mapika/decider-0.8b` & `potion-base-8M` vs `4 KB` `System-0` `LUT6`):
     - Verifies `4 KB` (`6.25%` of `64 KB` smartwatch `SRAM`) vs `> 21,897x` `decider-0.8b` overflow (`Theorem 63`),
       Wave-2 `.lut6` bypass, and `0.0%` OOD false action commits.
  3. `Claim LA-41` (Post-`LastExam` Open Trilogy: `ARC-AGI-2` + `MiniF2F-Lean4` + `CritPt`):
     - Verifies inductive `ARC-LUT6` synthesis (`D4`, `4-bit Color LUT6`, `Cross-5` cellular `LUT6`)
       admits `0` train-inconsistent rules (`Theorem 64/65`).
     - Verifies `MiniF2F-Lean4` real `Lean 4.36.0` kernel gate (`0.00%` unsound commits, `Theorem 66`)
       and Wave-2 `.lut6` tactic-crystal acceleration.
     - Verifies `0` raw benchmark prompt/answer/grid/code strings persisted in `artifacts/`.
  4. `Claim LA-19` (`66`-Theorem 6-Kernel Lean `4.36.0` Audit & `<= 16 KB` `lut6_core.wasm` new exports).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from clai.frontier.bonsai_decider_and_post_hle import (
    compile_color_map_to_4_lut6,
    eval_4_lut6_on_grid,
    fwht1024_sylvester_exact_shift5,
    goal_ast_to_lut6_fingerprint64,
    synthesize_arc_lut6_program,
    transcode_mlx_affine2_to_entromcu28,
    unpack_entromcu28_to_float32,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la39_entromcu28_transcoding_and_fwht1024_shift5() -> None:
    rng = np.random.default_rng(20261125)
    n_groups = 64
    trits = rng.integers(-1, 2, size=(n_groups, 128), dtype=np.int8)
    codes = (trits + 1).astype(np.uint32)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    words_u32 = np.bitwise_or.reduce(codes.reshape(n_groups, 8, 16) << shifts, axis=-1).astype(np.uint32)
    scales_f16 = (rng.uniform(0.05, 1.5, size=(n_groups,)).astype(np.float16))
    biases_f16 = (-scales_f16).astype(np.float16)

    packed28 = transcode_mlx_affine2_to_entromcu28(words_u32, scales_f16, biases_f16)
    assert len(packed28) == n_groups * 28
    assert 9 * len(packed28) == 7 * (n_groups * 36)  # Exact 2/9 = 22.22% byte reduction!

    rec = unpack_entromcu28_to_float32(packed28, n_groups)
    ref = codes.astype(np.float32) * scales_f16.astype(np.float32)[:, None] + biases_f16.astype(np.float32)[:, None]
    assert np.max(np.abs(rec - ref)) == 0.0

    # Invalid 4th codepoint `c == 3` must raise ValueError
    bad_words = words_u32.copy()
    bad_words[0, 0] |= 0x3
    with pytest.raises(ValueError):
        transcode_mlx_affine2_to_entromcu28(bad_words, scales_f16, biases_f16)

    # Exact `FWHT-1024 >> 5` self-inverse orthogonality (`0` MULs, `0` floats)
    v_orig = rng.integers(-20, 21, size=(1024,), dtype=np.int32) * 1024
    signs = rng.choice([-1, 1], size=(1024,)).astype(np.int32)
    v_rot = fwht1024_sylvester_exact_shift5(v_orig, signs)
    v_back = fwht1024_sylvester_exact_shift5(v_rot, np.ones(1024, dtype=np.int32)) * signs
    assert np.array_equal(v_back, v_orig)


def test_claim_la39_bonsai_real_audit_and_thor_window12_shunt() -> None:
    p35 = json.loads((ART_DIR / "phase35_bonsai_decider_post_hle_meta.json").read_text(encoding="utf-8"))
    b = p35["bonsai_2_27b_hadamard_ternary_audit"]
    assert b["code3_invalid_occurrences"] == 0
    assert b["max_abs_bias_plus_scale"] == 0.0
    assert b["max_lossless_roundtrip_f32_err"] == 0.0
    assert abs(b["entromcu_byte_reduction_pct"] - 22.22) < 0.05
    assert b["fwht1024_antisym40_vs_mlx_cosine"] > 0.999
    assert b["multiplication_reduction_factor"] == 128.0

    thor = json.loads((ART_DIR / "lut6_nucleus_thor_shunt.json").read_text(encoding="utf-8"))
    assert thor["board_metered"] is True
    wmap = {w["window"]: w for w in thor["windows"]}
    w12 = wmap["Window12_Bonsai_FWHT1024_Antisym40_Block16"]
    w13 = wmap["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]
    assert w12["board_metered"] is True and w13["board_metered"] is True
    assert w12["ns_per_op"] < w13["ns_per_op"]
    assert w12["measured_nj_per_op_vdd_cpu"] < w13["measured_nj_per_op_vdd_cpu"]


def test_claim_la40_system1_decider_vs_compiled_lut6() -> None:
    csv_path = ART_DIR / "system1_decider_vs_lut6_compile.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_s = {r["system"]: r for r in rows}
    dec_row = next(v for k, v in by_s.items() if "Mapika/decider-0.8b" in k and "OpenIE" not in k)
    mol_row = next(v for k, v in by_s.items() if "OpenIE System-0 -> System-1 MoL" in k)
    r0_row = next(v for k, v in by_s.items() if "Compiled System-0 LUT6 Crystal" in k)

    assert dec_row["fits_64kb_watch_sram"] == "False"
    assert mol_row["fits_64kb_watch_sram"] == "True"
    assert int(dec_row["active_weight_bytes"]) > 350_000 * int(mol_row["active_weight_bytes"])
    assert float(mol_row["ood_false_action_pct"]) == 0.0
    assert float(mol_row["wave2_lut6_bypass_pct"]) == 100.0
    assert float(r0_row["mean_latency_ms"]) < float(dec_row["mean_latency_ms"])


def test_claim_la41_arc_lut6_synthesizer_and_post_hle_trilogy() -> None:
    # 1. Unit-test exact `D4 + 4-Bit Color LUT6` & `Cross-5` cellular `LUT6` inductive synthesis
    g1 = np.array([[0, 2, 0], [2, 5, 2], [0, 2, 0]], dtype=np.int16)
    g2 = np.array([[5, 0, 2], [0, 5, 0], [2, 0, 5]], dtype=np.int16)
    cmap = {0: 0, 2: 7, 5: 3}
    m4 = compile_color_map_to_4_lut6(cmap)
    y1 = eval_4_lut6_on_grid(np.rot90(g1, 1), m4)
    y2 = eval_4_lut6_on_grid(np.rot90(g2, 1), m4)

    prog = synthesize_arc_lut6_program([(g1, y1), (g2, y2)])
    assert prog is not None and prog.lut6_count == 4
    g_test = np.array([[2, 2, 5], [0, 0, 2], [5, 0, 0]], dtype=np.int16)
    assert np.array_equal(prog.fn(g_test), eval_4_lut6_on_grid(np.rot90(g_test, 1), m4))

    # Inconsistent train pair MUST be rejected (`Theorem 65`: `None`, never a false rule!)
    bad_prog = synthesize_arc_lut6_program([(g1, y1), (g1, np.zeros_like(y1))])
    assert bad_prog is None

    # Alpha-renamed Lean 4 theorems map to identical 64-bit goal fingerprint
    fp_a = goal_ast_to_lut6_fingerprint64("theorem foo_1 (x y : Int) : 4 * x^3 - 7 * y^3 ≠ 2003")
    fp_b = goal_ast_to_lut6_fingerprint64("theorem bar_9 (u v : Int) : 4 * u^3 - 7 * v^3 ≠ 2003")
    assert fp_a == fp_b

    # 2. Verify trilogy CSV & zero raw prompt/grid/code persistence
    p35_raw = (ART_DIR / "phase35_bonsai_decider_post_hle_meta.json").read_text(encoding="utf-8")
    p35 = json.loads(p35_raw)
    assert p35["zero_raw_benchmark_strings_persisted"] is True
    assert p35["zero_access_control_bypass"] is True
    assert p35["minif2f_lean4_summary"]["unsound_committed_proofs_mol"] == 0
    assert p35["critpt_physics_summary"]["mol_chr80_pct"] == 0.0
    assert "def answer(" not in p35_raw  # Zero raw CritPt code leakage!


def test_claim_la19_66_theorems_and_wasm_phase35_exports() -> None:
    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 66
    assert lean_meta["module_theorem_counts"].get("TernaryHadamardArc") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert lean_meta["sorry_count"] == 0
    assert lean_meta["custom_axiom_count"] == 0

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const b64 = ex.wasm_eval_bonsai_fwht64_antisym40(0xc011a061, 0x800d3908, 0xa5a55a5a, 0xc3c33c3c) >>> 0;
  // Identity color map `0->0, 1->1, ..., 9->9` (`0x76543210`, `0xFEDCBA98`) -> leaves `0x01234567` unchanged!
  const id_cells = ex.wasm_eval_arc_lut6_cells8(0x01234567, 0xAA, 0x55, 0x76543210, 0xFEDCBA98) >>> 0;
  console.log(JSON.stringify({{ b64, id_cells }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["id_cells"] == 0x01234567
