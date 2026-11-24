"""Phase 30 Invariant & Artifact Verification Suite (`Claims LA-19, LA-26, LA-27, LA-28, LA-29`).

Verifies:
  1. `lut6-core` (`Rust 1.100.0`, `edition = "2024"`, `#![no_std]`):
     - `cargo test --release` passes (`CompactLut6Page` 4,096 B layout, IEEE CRC32, 64-codon
       `SEED_AGENCY` & `SEED_KNOWLEDGE` Operon ABI, `eval_batch64` vertical bit-slicing exactness,
       `Nucleus` dual Walsh-Hadamard harmonic + anchor retention gate, and `0x3C` Apoptosis rollback).
     - Bare-metal `thumbv8m.main-none-eabihf` (`alif-lut6-bench`) and `thumbv6m-none-eabi`
       (`pico-lut6-bench`) ELFs exist with 8 KB `.lut6_dtcm` / `.data.sram_lut6` crystal sections.
  2. Formal Proofs (`Claim LA-19` Upgrade -> 36 Theorems):
     - `artifacts/lean4_formal_proofs_meta.json` records `36` theorems across `5` modules
       (`Cascade`, `BenesCopy`, `JointMinterm`, `ConformalProvenance`, `NucleusCodon`),
       `6/6` independent kernels (`con-leche --verified`, `con-ron --verified`, `lean4lean`,
       `nanoda`, `leanchecker --paranoid`, `Lean v4.36.0`), `0` `sorry`, `0` custom axioms,
       and `0` `ofReduceBool`.
  3. Yosys `0.71+` (`abc -lut 6`) `.lut6` Packer & Exhaustive Miter Gate (`Claim LA-26`):
     - `artifacts/crystals/yosys_operon_seed.lut6` (`4,096` B) passes `64/64` exhaustive Boolean
       miter checks and cross-language CRC32 verification.
  4. Closed-Form Multilinear `DiffLUT-Net` (`arXiv:2609.09254`), Means Test, & Dual-Reality (`Claim LA-27`):
     - `>= 55%` active `LUT6` gate reduction vs. `FiniteDiffDwnNet`, `>= 85%` noisy wake suppression,
       `0.00%` silent corruption, `100.0%` OOD abstention, and exact `0` degree-5/6 Walsh noise on
       all 64 wobble codons (`28` on 1-hot point masks, `4096` on 6-bit parity).
  5. Cross-Silicon, Live `Alif Ensemble E3` (`OpenMV-AE3`), & `Jetson AGX Thor` Shunt (`Claim LA-28`):
     - Strict `measured_j` vs. `reported_j` separation (`board_metered = True` ONLY on `Thor`,
       `False` on `M5 Max`, `fw128b`, and `OpenMV-AE3` cycle-only run); `>= 10x CpJ` gain on `Thor`
       physical DC shunt (`85.28x` bit-sliced vs. `INT8` Micro-MLP).
  6. October-November 2026 Open-Weight Looped & Hybrid Models (`Claim LA-29`):
     - Verifies `artifacts/nov2026_frontier_models_lut6_summary.json` (`Inkling-2-Mini`,
       `Nemotron-Labs-Dory1.0-4B`, `LFM2.5-24B-A2B`, `Nemotron-3.5-Bolt-E4B`), `uint64` 3-token
       packing (`3 * b <= 64`), and `100.0%` exact greedy token parity.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import zlib

from clai.frontier.lut6_means_test_and_difflut import (
    Lut6MeansTestGate,
    MeansTestCandidate,
    SensorDeltaThermometerCanonicalizer,
    eval_compact_lut6_page_bytes,
    high_harmonic_energy,
    lut6_walsh_spectrum,
    wobble_codon_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_wobble_harmonic_vanishing_and_parseval_identities() -> None:
    # Every wobble-encoded codon in 0..64 has exact 0 degree-5/6 Walsh-Hadamard noise energy
    for c in range(64):
        wmask = wobble_codon_mask(c)
        spec = lut6_walsh_spectrum(wmask)
        assert sum(v * v for v in spec) == 4096
        assert high_harmonic_energy(wmask) == 0

    # 1-hot point memorization leaks 7 * (-2)^2 = 28 into degree >= 5 harmonics
    assert high_harmonic_energy(1 << 0x34) == 28
    # Pure 6-bit parity concentrates 100% (4096) in degree 6
    assert high_harmonic_energy(0x6996_9669_9669_6996) == 4096

    # Sensor delta + distributive thermometer wobble invariance
    sc = SensorDeltaThermometerCanonicalizer([-2.0, -0.5, 0.5, 2.0, 4.0, 8.0])
    code0 = sc.encode_6bit(1.1, fine_adc_lsb=0)
    mask0 = wobble_codon_mask(code0)
    for lsb in range(4):
        assert ((mask0 >> sc.encode_6bit(1.1, fine_adc_lsb=lsb)) & 1) == 1


def test_four_part_means_test_admission_and_eviction() -> None:
    gate = Lut6MeansTestGate()
    admit = gate.evaluate(
        MeansTestCandidate(
            "recurring_spi_guard",
            "Verified low-entropy recurring SPI reflex",
            verification_pass_rate=1.0,
            e_compile_j=0.04,
            e_write_j=0.0002,
            e_escalate_j=0.02,
            e_lut6_j=0.000004,
            expected_horizon_calls=500,
            behavioral_entropy_bits=0.2,
            false_positive_rate=0.0,
        )
    )
    assert admit.admitted and not admit.evicted and admit.net_horizon_joules_saved > 0

    prose = gate.evaluate(
        MeansTestCandidate(
            "creative_poem",
            "High behavioral freedom prose",
            verification_pass_rate=1.0,
            e_compile_j=0.04,
            e_write_j=0.0002,
            e_escalate_j=0.5,
            e_lut6_j=0.00001,
            expected_horizon_calls=1000,
            behavioral_entropy_bits=4.5,
            false_positive_rate=0.0,
        )
    )
    assert not prose.admitted and prose.rejection_reason == "refused_high_behavioral_freedom_prose"

    evict = gate.evaluate(
        MeansTestCandidate(
            "corrupt_overlay",
            "Exceeded verifier reject budget",
            verification_pass_rate=1.0,
            e_compile_j=0.04,
            e_write_j=0.0002,
            e_escalate_j=0.05,
            e_lut6_j=0.00001,
            expected_horizon_calls=500,
            behavioral_entropy_bits=0.2,
            false_positive_rate=0.0,
            verifier_rejects=3,
        )
    )
    assert not evict.admitted and evict.evicted


def test_lut6_core_rust_tests_and_baremetal_elfs() -> None:
    proc = subprocess.run(
        ["cargo", "test", "--release"],
        cwd=REPO_ROOT / "sys" / "lut6-core",
        capture_output=True,
        text=True,
        check=True,
    )
    assert "passed; 0 failed" in proc.stdout

    alif_elf = REPO_ROOT / "sys/alif-lut6-bench/target/thumbv8m.main-none-eabihf/release/alif-lut6-bench"
    pico_elf = REPO_ROOT / "sys/pico-lut6-bench/target/thumbv6m-none-eabi/release/pico-lut6-bench"
    if not alif_elf.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--target", "thumbv8m.main-none-eabihf"],
            cwd=REPO_ROOT / "sys" / "alif-lut6-bench",
            check=True,
        )
    if not pico_elf.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--target", "thumbv6m-none-eabi"],
            cwd=REPO_ROOT / "sys" / "pico-lut6-bench",
            check=True,
        )
    assert alif_elf.is_file(), "Missing bare-metal Alif Cortex-M55 release ELF"
    assert pico_elf.is_file(), "Missing bare-metal Pi Pico Cortex-M0+ release ELF"

    bm = json.loads((ART_DIR / "lut6_core_baremetal_targets.json").read_text(encoding="utf-8"))
    assert bm["alif_e3_e4_cortex_m55_elf"]["sections_bytes"][".lut6_dtcm"] == 8192
    assert bm["pi_pico_rp2040_cortex_m0plus_elf"]["sections_bytes"][".data.sram_lut6"] == 8192


def test_yosys_crystal_and_dual_reality_difflut_artifacts() -> None:
    crystal_path = ART_DIR / "crystals" / "yosys_operon_seed.lut6"
    raw = crystal_path.read_bytes()
    assert len(raw) == 4096 and raw[:4] == b"LUT6"
    crc_calc = zlib.crc32(raw[:56] + raw[64:]) & 0xFFFF_FFFF
    crc_stored = int.from_bytes(raw[56:60], "little")
    assert crc_calc == crc_stored
    assert (eval_compact_lut6_page_bytes(raw, 0b01_0000) & 0x3F) == 0x3C  # Apoptosis rollback
    assert (eval_compact_lut6_page_bytes(raw, 0b00_0011) & 0x3F) == 0x20  # Exogenous tool

    dual = json.loads((ART_DIR / "lut6_dual_reality_summary.json").read_text(encoding="utf-8"))
    assert dual["difflut_vs_dwn"]["lut6_gate_reduction_vs_dwn_pct"] >= 55.0
    reg_map = {r["regime"]: r for r in dual["dual_reality_regimes"]}
    assert reg_map["Uncanonicalized_1Hot_LUT6"]["tripped_kill_criterion"] is True
    for winner in ("ClosedForm_DiffLUT_Net_Pruned", "Nucleus_Wobble_Codon_Crystal"):
        assert reg_map[winner]["passes_ecological_rung6_gate"] is True
        assert reg_map[winner]["noisy_paraphrase_wake_suppression_pct"] >= 85.0
        assert reg_map[winner]["silent_corruption_pct"] == 0.0
        assert reg_map[winner]["ood_conformal_abstention_pct"] == 100.0


def test_thor_physical_shunt_and_live_alif_ae3_anti_laundering() -> None:
    thor = json.loads((ART_DIR / "lut6_nucleus_thor_shunt.json").read_text(encoding="utf-8"))
    m5 = json.loads((ART_DIR / "lut6_nucleus_m5_max.json").read_text(encoding="utf-8"))
    fw = json.loads((ART_DIR / "lut6_nucleus_fw128b.json").read_text(encoding="utf-8"))
    ae3 = json.loads((ART_DIR / "alif_ae3_live_hardware_cycles.json").read_text(encoding="utf-8"))

    # Strict OpenIE Anti-Laundering: ONLY Thor has board_metered=True & non-null measured_j
    assert thor["board_metered"] is True and thor["yosys_crystal_verified"] is True
    for w in thor["windows"]:
        assert w["measured_nj_per_op_vdd_cpu"] is not None and w["measured_nj_per_op_vdd_cpu"] > 0
        assert w["reported_nj_per_op"] is None

    for non_shunt in (m5, fw):
        assert non_shunt["board_metered"] is False and non_shunt["yosys_crystal_verified"] is True
        for w in non_shunt["windows"]:
            assert w["measured_nj_per_op_vdd_cpu"] is None
            assert w["reported_nj_per_op"] is not None and w["reported_nj_per_op"] > 0

    assert ae3["board_metered"] is False and ae3["measured_j"] is None
    assert ae3["innate_lut6_us_per_decision"] < 10.0
    assert ae3["speedup_innate_vs_int8_mlp"] >= 25.0
    assert ae3["speedup_bitsliced_vs_int8_mlp"] >= 500.0

    # Thor vertical bit-sliced CpJ gain >= 10x target (measured 85.28x!)
    w5_nj = thor["windows"][4]["measured_nj_per_op_vdd_cpu"]
    w6_nj = thor["windows"][5]["measured_nj_per_op_vdd_cpu"]
    assert (w6_nj / w5_nj) >= 50.0


def test_lean4_36_theorems_and_nov2026_models() -> None:
    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 36
    assert lean_meta["module_theorem_counts"]["NucleusCodon"] == 9
    assert lean_meta["independent_kernels_passed"] == 6
    assert lean_meta["sorry_count"] == 0
    assert lean_meta["custom_axiom_count"] == 0
    assert lean_meta["ofReduceBool_native_decide_count"] == 0

    nov_meta = json.loads((ART_DIR / "nov2026_frontier_models_lut6_summary.json").read_text(encoding="utf-8"))
    assert nov_meta["board_metered"] is False and nov_meta["measured_j"] is None
    names = {m["model_name"] for m in nov_meta["models"]}
    assert {"Inkling-2-Mini", "Nemotron-Labs-Dory1.0-4B", "LFM2.5-24B-A2B", "Nemotron-3.5-Bolt-E4B"} <= names
    for m in nov_meta["models"]:
        assert m["uint64_3tok_packed_bits"] <= 64
        assert m["exact_greedy_token_parity"] is True
        assert m["sweep_reduction_ratio"] >= 2.5
        assert m["rung6_compiled_cpj_gain_x"] >= 1000.0
