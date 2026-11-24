"""Phase 30 (`Claim LA-27` & `Claim LA-28`): Runner for the Four-Part Thermodynamic `LUT6`
Means Test, Closed-Form Multilinear `DiffLUT-Net` (`arXiv:2609.09254`) vs. `DWN` Synthesis,
and Dual-Reality Evaluation Protocol (`scripts/run_dual_reality_lut6_eval.py`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from clai.contracts.joule_meter import meter_joules
from clai.frontier.lut6_means_test_and_difflut import run_dual_reality_evaluation


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "artifacts"
    crystals_dir = art_dir / "crystals"
    crystals_dir.mkdir(parents=True, exist_ok=True)

    with meter_joules("compile") as jr:
        res = run_dual_reality_evaluation()

    crystal_bytes: bytes = res.pop("difflut_crystal_bytes")
    difflut_path = crystals_dir / "difflut_intent_router.lut6"
    difflut_path.write_bytes(crystal_bytes)

    dual_csv = art_dir / "lut6_dual_reality_eval.csv"
    regimes = res["dual_reality_regimes"]
    with dual_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(regimes[0].keys()))
        w.writeheader()
        w.writerows(regimes)

    mt_csv = art_dir / "lut6_means_test_decisions.csv"
    decisions = res["means_test_decisions"]
    with mt_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
        w.writeheader()
        w.writerows(decisions)

    res["telemetry"] = {
        "eval_elapsed_ms": round(jr.elapsed_ms, 2),
        "reported_j": round(float(jr.reported_j or 0.0), 4),
        "measured_j": None,
        "board_metered": False,
        "difflut_crystal_path": str(difflut_path.relative_to(repo_root)),
    }
    summary_json = art_dir / "lut6_dual_reality_summary.json"
    summary_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    dv = res["difflut_vs_dwn"]
    print("=" * 108)
    print(" PHASE 30: CLOSED-FORM DiffLUT-Net (arXiv:2609.09254) + 4-PART MEANS TEST + DUAL-REALITY EVAL")
    print("=" * 108)
    print(
        f"  DiffLUT-Net vs DWN: {dv['difflut_pruned_active_lut6_gates']} active LUT6s vs {dv['dwn_active_lut6_gates']} DWN LUT6s "
        f"({dv['lut6_gate_reduction_vs_dwn_pct']:.2f}% reduction) | CRC32={dv['difflut_page_crc32_hex']}"
    )
    for r in regimes:
        print(
            f"  [{r['regime']:<30}] LUT6s={r['active_lut6_gates']:>3} | "
            f"NoisyWakeSupp={r['noisy_paraphrase_wake_suppression_pct']:>6.2f}% | "
            f"SilentCorrupt={r['silent_corruption_pct']:>4.2f}% | "
            f"OOD_Abstain={r['ood_conformal_abstention_pct']:>6.2f}% | "
            f"PassRung6={r['passes_ecological_rung6_gate']}"
        )
    print(f"  Saved -> {dual_csv.relative_to(repo_root)}, {mt_csv.relative_to(repo_root)}, {summary_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
