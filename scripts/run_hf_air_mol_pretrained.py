"""Phase 20 Runner: Pulls & Shards Real 2026 `LiquidAI/LFM2.5-350M` through `AirMoL` NVMe Streamer."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.hf_air_mol_pretrained import run_pretrained_2026_hf_air_mol_benchmark


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

    print("=" * 124)
    print(f" PHASE 20: REAL 2026 PRETRAINED LiquidAI/LFM2.5-350M SHARDED THROUGH AirMoL + TOKEN LUT6 [Device: {dev}]")
    print("=" * 124)

    rows, samples, meta = run_pretrained_2026_hf_air_mol_benchmark(
        device=dev,
        shard_dir=art_dir / "hf_lfm2_5_350m_shards",
        n_eval_files=5,
        gen_tokens=32,
    )

    out_csv = art_dir / "hf_2026_lfm2_5_air_mol_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "hf_2026_lfm2_5_air_mol_samples.json"
    out_json.write_text(json.dumps({"meta": meta, "samples": samples}, indent=2), encoding="utf-8")

    print(
        f"  [2026 Pretrained Model] {meta['repo_id']} ({meta['num_layers']} layers: "
        f"{meta['conv_ssm_layers']} Liquid Conv O(1) + {meta['full_attn_layers']} GQA Attn) | "
        f"Params={meta['total_params']:,} ({meta['full_fp32_mb']} MB FP32 -> {meta['nvme_4bit_shards_mb']} MB 4-Bit = {meta['compression_vs_fp32']}x)"
    )
    print(
        f"  [Token-Level PagedBenesLUT6] Vocab={meta['vocab_size']:,} (exact 16-bit uint16) | "
        f"3-tok Pages={meta['token_pages_3gram']:,} | 2-tok Pages={meta['token_pages_2gram']:,} | "
        f"320-LUT6 AST Pages={meta['token_branch_lut6_pages']:,} ({meta['compile_sec']}s)"
    )
    print("-" * 124)
    print(
        f"{'2026 Pretrained NVMe Method (32 BPE Toks)':<52} | {'Tok/Sw':>6} | {'Chr/Sw':>6} | "
        f"{'Sweeps':>6} | {'NVMe GB':>8} | {'PeakRAM':>8} | {'tok/s':>6} | {'chr/s':>6}"
    )
    print("-" * 124)
    for r in rows:
        print(
            f"{r.method:<52} | {r.mean_tokens_per_nvme_sweep:>6.2f} | "
            f"{r.mean_code_chars_per_nvme_sweep:>6.2f} | {r.mean_nvme_sweeps_for_32tok:>6.1f} | "
            f"{r.nvme_gb_read_per_completion:>6.3f}GB | {r.peak_active_weight_ram_mb:>6.1f}MB | "
            f"{r.measured_tok_per_sec:>6.2f} | {r.measured_code_chars_per_sec:>6.1f}"
        )
    print("-" * 124)
    for idx_s, s in enumerate(samples):
        comp_str = s.get("completion", s.get("lfm2_5_completion", ""))
        print(
            f"  Sample #{idx_s+1} [{s['file']} | {s['nvme_sweeps']}/{s['naive_sweeps']} NVMe sweeps]: "
            f"{repr(comp_str[:78])}..."
        )
    print("=" * 124)


if __name__ == "__main__":
    main()
