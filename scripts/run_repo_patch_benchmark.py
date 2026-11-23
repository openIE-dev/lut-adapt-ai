"""Phase 26 Runner: Multi-File Git-Repository `pytest` Patching Benchmark (`SWE-Mini` / `RepoPatchCascade`).

Evaluates 5 Late-2026 Systems across `16` isolated multi-file git-repository bug-fixing tasks
(`8` cross-module Python packages x `2` waves, including `1` Wave-2 adversarial specification mutation):
  A. `LiquidAI/LFM3-1.7B` (Ungated 1-Shot Repo Patch)
  B. `google/gemma-4-E2B-it` (Ungated 1-Shot Repo Patch)
  C. `LiquidAI/LFM2.5-8B-A1B` (`8.47B` Sparse-MoE, Ungated 1-Shot Repo Patch)
  D. `Qwen/Qwen3.6-35B-A3B` (`34.66B` Metal `INT4` `Expert-AirMoL`, Ungated 1-Shot Repo Patch)
  E. `OpenIE Mixture of Limits (MoL)` (`Rung 0 BenesLUT6` + `.mol/` Patch Skill Store + Subprocess `pytest` `C(z)`
     + Counterexample Traceback Escalation + Online Downward Crystallization)
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.repo_patch_cascade import run_repo_patch_benchmark


def main() -> None:
    dev = select_torch_device()
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    print("=" * 150)
    print(f" PHASE 26: MULTI-FILE GIT-REPOSITORY PYTEST PATCHING BENCHMARK (`SWE-Mini`, 16 REPO TASKS) [Device: {dev}]")
    print("=" * 150)

    rows, payload = run_repo_patch_benchmark(device=dev)

    out_csv = art_dir / "repo_patch_swebench_mini.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "repo_patch_swebench_mini_traces.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 150)
    print(
        f"{'Repository Patching Arm (16 Git-Repo Tasks)':<46} | {'W1 Cold':>7} | {'W2 Warm':>7} | "
        f"{'AllPass':>7} | {'Broken%':>7} | {'W2 .mol':>7} | {'W1 ms':>8} | {'W2 ms':>8} | {'repJ/fix':>8}"
    )
    print("-" * 150)
    for r in rows:
        rep_s = f"{r.reported_joules_per_verified_fix:>8.2f}" if r.reported_joules_per_verified_fix is not None else "    None"
        print(
            f"{r.arm:<46} | {r.wave1_cold_pass_pct:>6.1f}% | {r.wave2_warm_pass_pct:>6.1f}% | "
            f"{r.overall_verified_pass_pct:>6.1f}% | {r.broken_patches_committed_pct:>6.1f}% | "
            f"{r.rung0_rung1_hit_pct_wave2:>6.1f}% | {r.mean_latency_ms_wave1:>8.1f} | "
            f"{r.mean_latency_ms_wave2:>8.1f} | {rep_s}"
        )
    print("-" * 150)
    print("  Meta:", json.dumps(payload["meta"], indent=2))
    print(f"Saved {out_csv} and {out_json}.")
    print("=" * 150)


if __name__ == "__main__":
    main()
