"""Phase 25b Runner: Clean-Room Sub-1-Bit Spectral-Rotated `LUT6` Factorization Benchmark
comparing LittleBit-1 / LittleBit-2 PTQ principles vs. OpenIE `LUT6` Group-Lookup & Lossless `MoL`.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.sub1bit_lut_factor import run_sub1bit_vs_mol_real_weights_benchmark


def main() -> None:
    dev = select_torch_device()
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    rows, meta = run_sub1bit_vs_mol_real_weights_benchmark(device=dev)

    out_csv = art_dir / "sub1bit_littlebit_vs_mol_lut6.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    (art_dir / "sub1bit_littlebit_vs_mol_meta.json").write_text(
        json.dumps({"meta": meta, "rows": [asdict(r) for r in rows]}, indent=2), encoding="utf-8"
    )

    print("=" * 136)
    print(f" PHASE 25b: SUB-1-BIT LATENT FACTORIZATION (LittleBit-1/2 PTQ) vs. OpenIE LUT6-GROUPED + MoL [Device: {dev}]")
    print("=" * 136)
    print(f"  Target Matrix: {meta['evaluated_tensor']} | LUT6 Rank r={meta['lut6_rank_r']} ({meta['lut6_groups_per_latent']} LUT6 groups) | Max LUT6 vs Dense Err={meta['lut6_vs_dense_max_abs_err']}")
    print(f"  35B Full-Model RAM-Pinned Sub-1-Bit Draft Footprint: {meta['projected_35b_ram_pinned_draft_gb']} GB (vs. 70.0 GB FP16)")
    print("-" * 136)
    print(
        f"{'Method':<68} | {'bpw':>5} | {'Comp':>6} | {'SNR dB':>6} | {'CosSim':>6} | {'Top1%':>6} | {'BinMACs':>9} | {'QAT'}"
    )
    print("-" * 136)
    for r in rows:
        print(
            f"{r.method:<68} | {r.bits_per_weight:>5.3f} | {r.compression_vs_fp16:>5.1f}x | "
            f"{r.weight_snr_db:>6.2f} | {r.activation_cosine_sim:>6.4f} | {r.top1_token_agreement_pct:>5.1f}% | "
            f"{r.binary_matmul_macs_per_tok:>9,d} | {r.qat_epochs_required:>2d} ep"
        )
    print("=" * 136)


if __name__ == "__main__":
    main()
