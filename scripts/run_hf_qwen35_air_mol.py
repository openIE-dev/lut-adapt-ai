"""Phase 21 Runner: Pulls & Shards Real Feb 2026 `Qwen/Qwen3.5-0.8B-Base` (Gated-DeltaNet 3:1 Hybrid)
through `AirMoL` 4-Bit NVMe Layer Streaming + 18-Bit Token-Level `PagedBenesLUT6` Burst Speculation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.hf_air_mol_pretrained import (
    QWEN35_2026_HF_MODEL,
    run_pretrained_2026_hf_air_mol_benchmark,
)


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

    print("=" * 126)
    print(f" PHASE 21: REAL FEB 2026 Qwen/Qwen3.5-0.8B-Base (GATED-DELTANET 3:1 HYBRID) THROUGH AirMoL [Device: {dev}]")
    print("=" * 126)

    rows, samples, meta = run_pretrained_2026_hf_air_mol_benchmark(
        device=dev,
        shard_dir=art_dir / "hf_qwen3_5_0_8b_shards",
        n_eval_files=4,
        gen_tokens=24,
        repo_id=QWEN35_2026_HF_MODEL,
        build_fp16_baseline=False,  # Keep disk footprint compact (~430 MB 4-Bit shards!)
    )

    out_csv = art_dir / "hf_2026_qwen3_5_air_mol_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "hf_2026_qwen3_5_air_mol_samples.json"
    out_json.write_text(json.dumps({"meta": meta, "samples": samples}, indent=2), encoding="utf-8")

    print(
        f"  [2026 Pretrained Model] {meta['repo_id']} ({meta['num_layers']} layers: "
        f"{meta['conv_ssm_layers']} Gated-DeltaNet LinearAttn O(1) + {meta['full_attn_layers']} FullAttn) | "
        f"LM Params={meta['total_params']:,} ({meta['full_fp32_mb']} MB FP32 -> {meta['nvme_4bit_shards_mb']} MB 4-Bit = {meta['compression_vs_fp32']}x)"
    )
    print(
        f"  [18-Bit Token PagedBenesLUT6] Vocab={meta['vocab_size']:,} ({meta['bits_per_token']}b x K={meta['draft_k_tokens']} = 54b <= 64b uint64) | "
        f"3-tok Pages={meta['token_pages_3gram']:,} | 2-tok Pages={meta['token_pages_2gram']:,} ({meta['compile_sec']}s)"
    )
    print("-" * 126)
    print(
        f"{'2026 Pretrained Qwen3.5 NVMe Method (24 BPE Toks)':<56} | {'Tok/Sw':>6} | {'Chr/Sw':>6} | "
        f"{'Sweeps':>6} | {'NVMe GB':>8} | {'PeakRAM':>8} | {'tok/s':>6} | {'chr/s':>6}"
    )
    print("-" * 126)
    for r in rows:
        print(
            f"{r.method:<56} | {r.mean_tokens_per_nvme_sweep:>6.2f} | "
            f"{r.mean_code_chars_per_nvme_sweep:>6.2f} | {r.mean_nvme_sweeps_for_32tok:>6.1f} | "
            f"{r.nvme_gb_read_per_completion:>6.3f}GB | {r.peak_active_weight_ram_mb:>6.1f}MB | "
            f"{r.measured_tok_per_sec:>6.2f} | {r.measured_code_chars_per_sec:>6.1f}"
        )
    print("-" * 126)
    for idx_s, s in enumerate(samples):
        print(
            f"  Sample #{idx_s+1} [{s['file']} | {s['nvme_sweeps']}/{s['naive_sweeps']} NVMe sweeps]: "
            f"{repr(s['completion'][:84])}..."
        )
    print("=" * 126)


if __name__ == "__main__":
    main()
