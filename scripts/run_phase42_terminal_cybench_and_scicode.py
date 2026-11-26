"""Phase 42 (`Claims LA-65 & LA-66`): Executes:
  1. `24`-Task SHA-256-Pinned `Terminal-Bench & Cybench-Defensive` Stateful POSIX/CLI Recovery +
     Defensive Binary/Protocol Vulnerability Patching & `144/144` `Yosys 0.71+` (`abc -lut 6`)
     `$lut` Guard Miter (`artifacts/phase42_terminal_cybench.csv`), and
  2. `24`-Task SHA-256-Pinned `FrontierMath-Constructive & SciCode` Research-Mathematics Witness
     Synthesis + Multi-Stage Computational Science Pipeline Verification
     (`artifacts/phase42_frontiermath_scicode.csv`),
with `0` raw benchmark prompts persisted, `0` offensive exploits, and strict 5-tier evidence separation.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

from clai.contracts.joule_meter import meter_joules
from clai.frontier.frontiermath_scicode_codifier import run_frontiermath_scicode_benchmark
from clai.frontier.terminal_cybench_codifier import run_terminal_cybench_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)

    with meter_joules("terminal_cybench_24") as jm_tc:
        tc_rows, tc_meta = run_terminal_cybench_benchmark()

    with meter_joules("frontiermath_scicode_24") as jm_fs:
        fs_rows, fs_meta = run_frontiermath_scicode_benchmark()

    tc_csv = ART_DIR / "phase42_terminal_cybench.csv"
    with tc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(tc_rows[0]).keys()))
        w.writeheader()
        for r in tc_rows:
            w.writerow(asdict(r))

    fs_csv = ART_DIR / "phase42_frontiermath_scicode.csv"
    with fs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(fs_rows[0]).keys()))
        w.writeheader()
        for r in fs_rows:
            w.writerow(asdict(r))

    meta = {
        "phase": 42,
        "claims": ["LA-65", "LA-66"],
        "terminal_cybench": {
            **tc_meta,
            "wall_ms": round(jm_tc.elapsed_ms, 2),
            "reported_j": round(float(jm_tc.reported_j or 0.0), 4),
        },
        "frontiermath_scicode": {
            **fs_meta,
            "wall_ms": round(jm_fs.elapsed_ms, 2),
            "reported_j": round(float(jm_fs.reported_j or 0.0), 4),
        },
        "board_metered": False,
        "measured_j": None,
    }
    (ART_DIR / "phase42_terminal_scicode_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("Terminal-Cybench W1/W2:", tc_rows[-2], tc_rows[-1])
    print("FrontierMath-SciCode W1/W2:", fs_rows[-2], fs_rows[-1])


if __name__ == "__main__":
    main()
