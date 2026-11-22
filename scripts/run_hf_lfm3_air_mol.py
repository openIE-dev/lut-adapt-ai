"""Phase 25c Runner: Shards & Benchmarks Real Oct 2026 `LiquidAI/LFM3-1.7B` (Tri-Hybrid:
`13` Short-Conv + `5` Sliding-Window Attention + `6` Global Attention layers, `128,000` vocab)
through `AirMoL` 4-Bit NVMe Streaming + `17-Bit x 3` `PagedBenesLUT6` Burst Speculation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.hf_air_mol_pretrained import (
    LFM3_2026_HF_MODEL,
    run_pretrained_2026_hf_air_mol_benchmark,
)
from clai.frontier.moe_air_mol import default_nvme_shard_root


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
    shard_dir = default_nvme_shard_root() / "hf_lfm3_1_7b_shards"

    print("=" * 126)
    print(f" PHASE 25c: REAL OCT 2026 LiquidAI/LFM3-1.7B (TRI-HYBRID: CONV + SLIDING + GLOBAL ATTN) THROUGH AirMoL [Device: {dev}]")
    print("=" * 126)

    rows, samples, meta = run_pretrained_2026_hf_air_mol_benchmark(
        device=dev,
        shard_dir=shard_dir,
        n_eval_files=4,
        gen_tokens=24,
        repo_id=LFM3_2026_HF_MODEL,
        build_fp16_baseline=False,
    )

    out_csv = art_dir / "hf_2026_lfm3_1_7b_air_mol_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "hf_2026_lfm3_1_7b_air_mol_samples.json"
    out_json.write_text(json.dumps({"meta": meta, "samples": samples}, indent=2), encoding="utf-8")

    print(
        f"  [Oct 2026 Model] {meta['repo_id']} ({meta['num_layers']} layers: "
        f"{meta['conv_ssm_layers']} ShortConv O(1) + {meta.get('sliding_attn_layers', 0)} SlidingAttn O(W) + "
        f"{meta['full_attn_layers']} FullAttn) | Params={meta['total_params']:,} "
        f"({meta['full_fp32_mb']} MB FP32 -> {meta['nvme_4bit_shards_mb']} MB 4-Bit = {meta['compression_vs_fp32']}x)"
    )
    print("-" * 126)
    print(
        f"{'Oct 2026 LiquidAI/LFM3-1.7B NVMe Method (24 Toks)':<56} | {'Tok/Sw':>6} | {'Chr/Sw':>6} | "
        f"{'Sweeps':>6} | {'NVMe GB':>8} | {'PeakRAM':>8} | {'tok/s':>6} | {'chr/s':>6}"
    )
    print("-" * 126)
    for r in rows:
        print(
            f"{r.method:<56} | {r.mean_tokens_per_nvme_sweep:>6.2f} | "
            f"{r.mean_code_chars_per_nvme_sweep:>6.2f} | {r.mean_nvme_sweeps_for_32tok:>6.2f} | "
            f"{r.nvme_gb_read_per_completion:>6.3f}GB | {r.peak_active_weight_ram_mb:>6.1f}MB | "
            f"{r.measured_tok_per_sec:>6.2f} | {r.measured_code_chars_per_sec:>6.1f}"
        )
    print("=" * 126)


if __name__ == "__main__":
    main()
