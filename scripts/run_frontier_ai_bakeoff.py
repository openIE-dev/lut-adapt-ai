"""Phase 25 Runner: Head-to-Head Multi-Wave Frontier AI Coding & Verification Bakeoff.

Compares 7 Real Late-2026 Systems across 48 Systems/Algorithmic/Protocol Tasks (`16` specs x `3` waves):
  A. `LittleBitLLM/littlebit-qwen3-8b-0.55bpw` (Oct 2026 `LittleBit-2` `0.55 bpw` QAT `8B`, Ungated 1-Shot)
  B. `LiquidAI/LFM3-1.7B` (Oct 2026 Flagship Tri-Hybrid `1.7B`, Ungated 1-Shot)
  C. `google/gemma-4-E2B-it` (2026 Flagship On-Device Hybrid, Ungated 1-Shot)
  D. `LiquidAI/LFM2.5-8B-A1B` (`8.47B` Sparse-MoE, Ungated 1-Shot)
  E. `Qwen/Qwen3.6-35B-A3B` (`34.66B` Frontier Sparse-MoE via `Expert-AirMoL`, Ungated 1-Shot)
  F. `RouteLLM` Confidence-Threshold Router (`LFM3-1.7B -> Gemma4-E2B -> LFM2.5-8B-A1B`, No `C(z)` Gate)
  G. `OpenIE Mixture of Limits (MoL)` (`Rung 0 BenesLUT6` + `Rung 1 AST Skill` + `C(z)` Verifier
     + Counterexample-Guided Escalation up to `35B-A3B` + Online Downward Crystallization)
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.verified_code_cascade import run_frontier_head_to_head_bakeoff


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    dev = select_device()
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    print("=" * 144)
    print(f" PHASE 25: HEAD-TO-HEAD FRONTIER AI CODING & VERIFICATION BAKEOFF (48 TASKS = 16 SPECS x 3 WAVES) [Device: {dev}]")
    print("=" * 144)

    rows, payload = run_frontier_head_to_head_bakeoff(device=dev)

    out_csv = art_dir / "frontier_ai_head_to_head_bakeoff.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "frontier_ai_head_to_head_traces.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 156)
    print(
        f"{'System Arm (48 Tasks = 3 Waves)':<40} | {'W1 Cold':>7} | {'W2 Adpt':>7} | {'W3 Warm':>7} | "
        f"{'AllPass':>7} | {'Broken%':>7} | {'W3 R0/1':>7} | {'W1 ms':>8} | {'W3 ms':>8} | {'estJ/ok':>7} | {'repJ/ok':>8}"
    )
    print("-" * 156)
    for r in rows:
        rep_s = f"{r.reported_joules_per_verified_solve:>8.2f}" if r.reported_joules_per_verified_solve is not None else "    None"
        print(
            f"{r.arm:<40} | {r.wave1_cold_pass_pct:>6.1f}% | {r.wave2_adapt_pass_pct:>6.1f}% | "
            f"{r.wave3_warm_pass_pct:>6.1f}% | {r.overall_verified_pass_pct:>6.1f}% | "
            f"{r.broken_commits_admitted_pct:>6.1f}% | {r.rung0_rung1_hit_pct_wave3:>6.1f}% | "
            f"{r.mean_latency_ms_wave1:>8.2f} | {r.mean_latency_ms_wave3:>8.2f} | "
            f"{r.est_joules_per_verified_solve:>7.2f} | {rep_s}"
        )
    print("-" * 156)
    print("  Meta:", json.dumps(payload["meta"], indent=2))
    print(f"Saved {out_csv} and {out_json}.")
    print("=" * 144)


if __name__ == "__main__":
    main()
