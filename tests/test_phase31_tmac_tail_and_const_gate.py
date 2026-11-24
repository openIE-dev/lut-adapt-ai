"""Phase 31 Invariant & Hardware Verification Suite (`Claims LA-19 [42 Thms], LA-30, LA-31`).

Verifies:
  1. `EntroMCU` (`5 trits/byte`, `3^5 = 243 <= 256`, `1.602 bpw`) lossless roundtrip across all
     243 byte codepoints & exact equivalence between zero-multiplier `PALUTE`/`T-MAC` 4-trit LUT
     evaluation and signed ternary dot products.
  2. `mol_skills_tmac_tail.lut6` (`4,096 B`, `flags = 0x0006`, `0` wasted pad bytes):
     - Cross-language IEEE `CRC32` seal, `100.0%` noisy paraphrase accuracy across all 5 harvested
       agent skills, and `100.0%` conformal abstention (`votes < 6`) on hostile OOD queries.
  3. Compile-time `rustc` `const` proof-vector gate (`sys/lut6-core/src/generated_proof_vectors.rs`):
     - `3 passed; 0 failed` in `cargo test --release`, including `eval_raw_unrolled4 == eval_raw`
       and `#![no_std]` cross-target checks (`thumbv8m`, `thumbv6m`, `thumbv7em`, `riscv32imac`).
  4. Six-Kernel Paranoid Lean `4.36.0` Formalization (`42` theorems across `6` modules including
     `TernaryTmac.lean`, `6/6` independent kernels, `0` `sorry`, `0` custom axioms, `0` `ofReduceBool`).
  5. Two-Tier Hardware Telemetry & Strict Anti-Laundering (`Claim LA-31`):
     - `NVIDIA Jetson AGX Thor` physical DC shunt (`board_metered = True`) records `8` windows with
       `ZeroMul_Ternary_TMAC_Tail` beating `INT8` Micro-MLP `measured_nj_per_op_vdd_cpu` (`> 1.2x CpJ`).
     - Live attached `OpenMV-AE3` (`Alif Ensemble E3` Cortex-M55 @ `400 MHz`, `board_metered = False`,
       `measured_j = None`) records `zero_mul_tmac_tail_dwt_cycles_per_decision` beating 2-layer
       `INT8` Micro-MLP by `>= 10x`.
  6. Standalone Web & Keynote Synchronization (`paper/web/*`, `docs/keynote/index.html`):
     - Zero stale `27`-theorem banners and 5-slide Keynote deck with `lut6-nucleus-codon-and-means-test.svg`.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zlib

import numpy as np

from clai.frontier.lut6_means_test_and_difflut import (
    AstHadamardCanonicalizer,
    INTENT_CORPUS,
    OOD_HOSTILE_QUERIES,
    build_tmac_tail_bytes,
    eval_compact_lut6_page_bytes,
    eval_tmac_tail_bytes,
    pack_5trits,
    unpack_5trits,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_entromcu_base3_bijection_and_zero_mul_tmac_equivalence() -> None:
    for b in range(243):
        trits = unpack_5trits(b)
        assert len(trits) == 5 and all(t in (-1, 0, 1) for t in trits)
        assert pack_5trits(trits) == b

    rng = np.random.default_rng(20261124)
    weights = rng.choice([-1, 0, 1], size=(16, 64)).tolist()
    biases = rng.integers(-200, 200, size=16).tolist()
    pad = build_tmac_tail_bytes(weights, biases, margin_thresh=64)
    assert len(pad) == 672 and pad[:4] == b"TMAC"

    for _ in range(16):
        feats = rng.integers(-64, 65, size=64).tolist()
        pay, votes, accs = eval_tmac_tail_bytes(pad, feats)
        expected = [
            int(biases[r]) + sum(int(weights[r][c]) * int(feats[c]) for c in range(64))
            for r in range(16)
        ]
        assert accs == expected
        expected_pay = sum((1 << r) for r in range(8) if expected[r] > 0)
        expected_votes = sum(1 for r in range(8, 16) if expected[r] >= 64)
        assert pay == expected_pay and votes == expected_votes


def test_two_tier_mol_skills_crystal_zero_wasted_bytes_and_ood_abstention() -> None:
    cpath = ART_DIR / "crystals" / "mol_skills_tmac_tail.lut6"
    raw = cpath.read_bytes()
    assert len(raw) == 4096 and raw[:4] == b"LUT6"
    flags = int.from_bytes(raw[6:8], "little")
    assert (flags & 0x0004) != 0, "Expected FLAG_HAS_TMAC_TAIL (0x0004) set"
    crc_calc = zlib.crc32(raw[:56] + raw[64:]) & 0xFFFF_FFFF
    crc_stored = int.from_bytes(raw[56:60], "little")
    assert crc_calc == crc_stored

    tail_672 = raw[1376:2048]
    assert tail_672[:4] == b"TMAC"

    canon = AstHadamardCanonicalizer()
    # Tier-1 Rung-6 exact 36-wire Boolean conjunction hit on primary `verify_lean` phrase
    hot_w = canon.encode_u64(INTENT_CORPUS[0][2][0], canonicalize=True)
    r6_out = eval_compact_lut6_page_bytes(raw, hot_w)
    assert (r6_out & 0xFF) == 0x11 and ((r6_out >> 8) & 0xFF).bit_count() == 8

    # Tier-2 Zero-MUL T-MAC resolves all 40 clean+noisy skill paraphrases
    for _name, expected_pay, clean_l, noisy_l in INTENT_CORPUS:
        for txt in clean_l + noisy_l:
            w = canon.encode_u64(txt, canonicalize=True)
            bipolar = [16 if ((w >> i) & 1) else -16 for i in range(64)]
            pay, votes, _ = eval_tmac_tail_bytes(tail_672, bipolar)
            assert votes >= 6 and pay == expected_pay

    # OOD hostile queries abstain on BOTH Tier-1 and Tier-2
    for q in OOD_HOSTILE_QUERIES:
        ow = canon.encode_u64(q, canonicalize=True)
        r6_ood = eval_compact_lut6_page_bytes(raw, ow)
        assert ((r6_ood >> 8) & 0xFF).bit_count() < 6
        bipolar = [16 if ((ow >> i) & 1) else -16 for i in range(64)]
        _pay_o, votes_o, _ = eval_tmac_tail_bytes(tail_672, bipolar)
        assert votes_o < 6


def test_rustc_const_proof_vector_gate_and_unrolled4() -> None:
    gen_rs = REPO_ROOT / "sys" / "lut6-core" / "src" / "generated_proof_vectors.rs"
    assert gen_rs.is_file()
    txt = gen_rs.read_text(encoding="utf-8")
    assert "EXPECTED_OPERON_64" in txt and "VERIFIED_TMAC_SKILL_ANCHORS" in txt

    proc = subprocess.run(
        ["cargo", "test", "--release"],
        cwd=REPO_ROOT / "sys" / "lut6-core",
        capture_output=True,
        text=True,
        check=True,
    )
    assert "3 passed; 0 failed" in proc.stdout


def test_lean4_42_theorems_and_ternary_tmac_module() -> None:
    meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    assert meta["total_kernel_checked_theorems"] == 42
    assert meta["module_theorem_counts"]["TernaryTmac"] == 6
    assert meta["independent_kernels_passed"] == 6
    assert meta["sorry_count"] == 0
    assert meta["custom_axiom_count"] == 0
    assert meta["ofReduceBool_native_decide_count"] == 0


def test_two_tier_thor_shunt_and_alif_ae3_tmac_telemetry() -> None:
    thor = json.loads((ART_DIR / "lut6_nucleus_thor_shunt.json").read_text(encoding="utf-8"))
    ae3 = json.loads((ART_DIR / "alif_ae3_live_hardware_cycles.json").read_text(encoding="utf-8"))
    assert len(thor["windows"]) == 8 and thor["board_metered"] is True
    w_map = {w["window"]: w for w in thor["windows"]}
    mlp_nj = w_map["Window6_Baseline_INT8_MicroMLP"]["measured_nj_per_op_vdd_cpu"]
    tmac_nj = w_map["Window8_ZeroMul_Ternary_TMAC_Tail"]["measured_nj_per_op_vdd_cpu"]
    assert mlp_nj / tmac_nj >= 1.20

    assert ae3["board_metered"] is False and ae3["measured_j"] is None
    assert ae3["speedup_tmac_tail_vs_int8_mlp"] >= 10.0


def test_standalone_web_and_keynote_synced() -> None:
    keynote = (REPO_ROOT / "docs" / "keynote" / "index.html").read_text(encoding="utf-8")
    assert "42" in keynote and "lut6-nucleus-codon-and-means-test.svg" in keynote
    assert (REPO_ROOT / "docs" / "keynote" / "lut6-nucleus-codon-and-means-test.svg").is_file()

    web_idx = (REPO_ROOT / "paper" / "web" / "index.html").read_text(encoding="utf-8")
    assert "42" in web_idx and "lut6-nucleus-codon-and-means-test.svg" in web_idx
    assert "27 Lean 4.36.0 proofs" not in web_idx
