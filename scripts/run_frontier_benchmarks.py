"""Master Runner for All 3 Next-Gen Architectural Frontiers vs. Latest SOTA Baselines.

Executes:
  1. Frontier 1 (`Generative LUT6 4-Byte Speculative Draft Head` vs. Transformer KV-Cache,
     Mamba-3 T=1, EAGLE-2/Medusa Neural Speculative, and Prompt-Lookup N-Gram Cache)
  2. Frontier 2 (`512x512 Logarithmic Beneš-Copy-Beneš Hot-Swap RTL Fabric` vs. Flat 512:1
     Crossbars & Fixed-Wire DiffLogicNet WNNs in Yosys UltraScale+ & Verilator)
  3. Frontier 3 (`Latent JEPA-EBRM Split-Conformal + Online ACI Gate` vs. MSP, Temperature
     Scaling, Discriminative Helmholtz Energy OOD, and 5-Pass MC-Dropout Ensemble)
"""

from __future__ import annotations

from dataclasses import asdict
import csv
from pathlib import Path

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.benes_lut_fabric import run_frontier2_benes_benchmark
from clai.frontier.conformal_oracle_free import run_frontier3_conformal_benchmark
from clai.frontier.generative_lut_draft import run_frontier1_generative_benchmark


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    device = select_torch_device()
    print(f"[CLAI Frontier Suite] Running systematic SOTA baseline comparisons on {device}...\n")

    # =========================================================================
    # FRONTIER 1: Generative Multi-Byte LUT6 Speculative Draft Heads
    # =========================================================================
    print("=== FRONTIER 1: Generative Multi-Byte LUT6 Burst Drafter vs. LLM/SSM Baselines ===")
    f1_rows = run_frontier1_generative_benchmark(device)
    _write_csv(out_dir / "frontier1_generative_baselines.csv", f1_rows)
    for r in f1_rows:
        print(
            f"  {r.method:42s} | Acc={r.exact_match_pct:5.1f}% | "
            f"Bytes/Pass={r.mean_accepted_bytes_per_pass:4.2f} | "
            f"Passes={r.target_passes_per_seq:5.2f} | "
            f"FLOPs/B={r.flops_per_generated_byte / 1e6:5.2f}M | "
            f"mJ/B={r.joules_per_generated_byte * 1e3:6.3f}"
        )
    print()

    # =========================================================================
    # FRONTIER 2: 512x512 Logarithmic Beneš-Copy-Beneš Silicon Switching Fabric
    # =========================================================================
    print("=== FRONTIER 2: 512x512 Beneš-Copy-Beneš Hot-Swap RTL vs. Crossbars & WNNs ===")
    f2_rows = run_frontier2_benes_benchmark(device, out_dir)
    _write_csv(out_dir / "frontier2_benes_silicon_baselines.csv", f2_rows)
    for r in f2_rows:
        print(
            f"  {r.architecture:43s} | L0_LUTs={r.dynamic_luts_l0:2d} ({r.routable_ports_l0:3d} ports) | "
            f"Xilinx_LUTs={r.synthesized_xilinx_luts:7d} ({r.kv260_lut_utilization_pct:5.1f}% KV260) | "
            f"CLB/LUT6={r.clb_luts_per_routable_lut6:6.1f} | "
            f"DSP={r.dsp48_count} | VLAN_Acc={r.post_vlan_drift_accuracy_pct:5.1f}%"
        )
    print()

    # =========================================================================
    # FRONTIER 3: Split-Conformal + Online ACI Free-Energy Calibration
    # =========================================================================
    print("=== FRONTIER 3: JEPA-EBRM Split-Conformal + ACI vs. Uncertainty/OOD Baselines ===")
    f3_rows = run_frontier3_conformal_benchmark(device)
    _write_csv(out_dir / "frontier3_conformal_baselines.csv", f3_rows)
    for r in f3_rows:
        print(
            f"  {r.method:42s} | Passes={r.forward_passes_per_ep} | "
            f"PoisonRisk={r.false_crystallization_risk_pct:5.2f}% | "
            f"OOD_AUROC={r.ood_detection_auroc:5.3f} | "
            f"ID_Cov={r.safe_id_coverage_pct:5.1f}% | "
            f"R0_LUT6_Acc={r.crystallized_lut6_clean_acc_pct:5.1f}% | "
            f"DriftDet={r.detected_unlabeled_vlan_drift}"
        )
    print("\n[CLAI Frontier Suite] All 3 CSV benchmark reports written to artifacts/.")


if __name__ == "__main__":
    main()
