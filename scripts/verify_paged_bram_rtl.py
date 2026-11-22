"""Phase 16 Verification & KV260 Multi-Page BRAM Silicon Benchmark Runner.

Synthesizes `clai_benes_paged_hotswap_core` in Yosys 0.71+ (`synth_xilinx -family xcup`),
runs cycle-accurate Verilator 5.x `CPI=1.0` back-to-back page-switching + XNOR-CAM fallback +
live Online Sleep-Crystallization co-simulation, and exports `artifacts/kv260_paged_bram_baselines.csv`.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from clai.frontier.benes_bram_page_cache import (
    HardwareBRAMPageCache,
    run_paged_bram_silicon_comparison,
)
from clai.frontier.code_corpus_speculative import load_repo_code_splits


def main() -> None:
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print(" PHASE 16: KV260 1-CYCLE HASH-INDEXED MULTI-PAGE BRAM LUT6 CACHE + BENES RTL VERIFICATION")
    print("=" * 100)

    train_files, _ = load_repo_code_splits(Path("."))
    bram_cache = HardwareBRAMPageCache(page_addr_bits=9)
    stats = bram_cache.compile_from_corpus(train_files)
    print(
        f"  [Corpus -> KV260 BRAM Image] Capacity={stats['bram_capacity_pages']} pages | "
        f"Occupied={stats['bram_occupied_pages']} ({stats['bram_occupancy_pct']}%) | "
        f"Compiled in {stats['bram_compile_sec']}s"
    )

    rows, meta = run_paged_bram_silicon_comparison(art_dir / "paged_bram_rtl_sim")

    out_csv = art_dir / "kv260_paged_bram_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print("-" * 100)
    print(
        f"{'Hardware Architecture':<44} | {'Pages':>5} | {'Sw(cyc)':>7} | {'Sw(ns)':>6} | "
        f"{'CLB LUTs':>9} | {'FFs':>9} | {'RAMB36':>6} | {'DSP':>3}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r.architecture:<44} | {r.resident_lut6_pages:>5d} | {r.context_switch_cycles:>7d} | "
            f"{r.context_switch_ns_at_250mhz:>6.1f} | {r.synthesized_clb_luts:>9,d} | "
            f"{r.synthesized_ffs:>9,d} | {r.synthesized_ramb36_equiv:>6.1f} | {r.dsp48_count:>3d}"
        )
    print("-" * 100)
    print(f"Saved KV260 Multi-Page BRAM synthesis & co-sim metrics -> {out_csv}")
    assert meta["errors"] == 0


if __name__ == "__main__":
    main()
