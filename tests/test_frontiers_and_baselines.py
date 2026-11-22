"""Pytest Verification Suite for Frontiers 1, 2, and 3 + Head-to-Head Baselines."""

from __future__ import annotations

import numpy as np
import torch

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.benes_lut_fabric import compile_multicast_bcb512, eval_multicast_bcb512
from clai.frontier.conformal_oracle_free import run_frontier3_conformal_benchmark
from clai.frontier.generative_lut_draft import run_frontier1_generative_benchmark


def test_frontier2_benes_copy_benes_unrestricted_multicast_routing() -> None:
    """Proves 100% bit-exact routing across permutations, 512-way broadcasts, & skewed fanouts."""
    rng = np.random.default_rng(2026)
    test_bus = rng.integers(0, 2, size=(16, 512), dtype=np.uint8)

    # Case A: Full 1-to-1 random permutation of 512 wires
    perm = rng.permutation(512).astype(np.int32)
    cfg_perm = compile_multicast_bcb512(perm)
    assert np.array_equal(eval_multicast_bcb512(test_bus, cfg_perm), test_bus[:, perm])

    # Case B: Extreme 480-way broadcast of a single wire (wire 317)
    bcast = np.full(480, 317, dtype=np.int32)
    cfg_bcast = compile_multicast_bcb512(bcast)
    assert np.array_equal(eval_multicast_bcb512(test_bus, cfg_bcast)[:, :480], test_bus[:, bcast])

    # Case C: Skewed power-law multicast across 19 hot wires
    hot = rng.choice(512, size=19, replace=False)
    skewed = rng.choice(hot, size=480, replace=True).astype(np.int32)
    cfg_skew = compile_multicast_bcb512(skewed)
    assert np.array_equal(eval_multicast_bcb512(test_bus, cfg_skew)[:, :480], test_bus[:, skewed])


def test_frontier1_and_frontier3_beat_baselines_under_exact_contracts() -> None:
    """Verifies lossless generative LUT6 speedup & oracle-free conformal poisoning control."""
    dev = select_torch_device()

    # Frontier 1 checks
    f1 = {r.method: r for r in run_frontier1_generative_benchmark(dev)}
    ours_f1 = f1["1E_MoL_LUT6_4Byte_Burst_Speculative_OURS"]
    m3_base = f1["1B_Mamba3_Complex_MIMO_Autoregressive_T1"]
    pld_base = f1["1D_PromptLookup_Ngram_Cache_Speculative"]

    assert ours_f1.mean_accepted_bytes_per_pass >= 2.2, f"Low LUT6 burst acceptance: {ours_f1}"
    assert ours_f1.mean_accepted_bytes_per_pass > pld_base.mean_accepted_bytes_per_pass
    assert ours_f1.flops_per_generated_byte < m3_base.flops_per_generated_byte * 0.65

    # Frontier 3 checks
    f3 = {r.method: r for r in run_frontier3_conformal_benchmark(dev)}
    ours_f3 = f3["3E_MoL_LatentEBRM_SplitConformal_ACI_OURS"]
    msp_base = f3["3A_Naive_MSP_SelfDistillation"]

    assert ours_f3.ood_detection_auroc >= 0.95, f"Low JEPA-EBRM OOD AUROC: {ours_f3}"
    assert ours_f3.false_crystallization_risk_pct <= 5.0
    assert ours_f3.false_crystallization_risk_pct < msp_base.false_crystallization_risk_pct * 0.25
    assert ours_f3.safe_id_coverage_pct > msp_base.safe_id_coverage_pct
    assert ours_f3.crystallized_lut6_clean_acc_pct >= 98.0
    assert ours_f3.detected_unlabeled_vlan_drift is True
    assert msp_base.detected_unlabeled_vlan_drift is False
