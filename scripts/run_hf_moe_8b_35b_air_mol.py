"""Phase 25 Runner: Shards & Benchmarks Real 2026 Sparse-MoE Frontier Checkpoints
(`LiquidAI/LFM2.5-8B-A1B` [8.3B, 32 experts, top-4] & `Qwen/Qwen3.6-35B-A3B` [35B, 256 experts, top-8])
through Router-Guided `Expert-AirMoL` NVMe Streaming + Zero-VRAM `FrontierBenesLUT6` Speculation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.moe_air_mol import (
    LFM25_MOE_8B_MODEL,
    QWEN36_MOE_35B_MODEL,
    MoEAirMoLBenchmarkRow,
    run_pretrained_2026_moe_air_mol_benchmark,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2026 Sparse-MoE 8B & 35B Expert-AirMoL Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[LFM25_MOE_8B_MODEL, QWEN36_MOE_35B_MODEL],
        help="2026 Sparse-MoE repo IDs to shard and benchmark",
    )
    parser.add_argument("--n-files", type=int, default=3)
    parser.add_argument("--gen-tokens-8b", type=int, default=16)
    parser.add_argument("--gen-tokens-35b", type=int, default=12)
    args = parser.parse_args()

    dev = select_device()
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    all_rows: list[MoEAirMoLBenchmarkRow] = []
    all_meta: dict[str, dict] = {}
    all_samples: dict[str, list[dict]] = {}

    print("=" * 136)
    print(f" PHASE 25: 2026 FRONTIER SPARSE-MoE (8.3B & 35B) EXPERT-ROUTED AirMoL + PagedBenesLUT6 [Device: {dev}]")
    print("=" * 136)

    for repo_id in args.models:
        gen_tok = args.gen_tokens_35b if "35B" in repo_id else args.gen_tokens_8b
        print(f"\n>>> Sharding & Benchmarking {repo_id} ({args.n_files} held-out code files x {gen_tok} BPE tokens)...")
        rows, samples, meta = run_pretrained_2026_moe_air_mol_benchmark(
            repo_id=repo_id,
            device=dev,
            n_eval_files=args.n_files,
            gen_tokens=gen_tok,
        )
        all_rows.extend(rows)
        all_meta[repo_id] = meta
        all_samples[repo_id] = samples

        print(
            f"  [Model] {meta['repo_id']} ({meta['num_layers']} layers: "
            f"{meta['conv_ssm_layers']} O(1) Conv/DeltaNet + {meta['full_attn_layers']} FullAttn | "
            f"{meta['num_experts_per_tok']}/{meta['num_experts']} active experts) | "
            f"Params={meta['total_params']:,} ({meta['full_fp16_mb']} MB FP16 -> {meta['nvme_4bit_total_mb']} MB 4-Bit)"
        )
        if "apple_pmu_system_energy_delta_j" in meta:
            print(
                f"  [Apple M5 Max PMU Telemetry] System Energy Delta = {meta['apple_pmu_system_energy_delta_j']} J "
                f"({meta['apple_pmu_accumulator_ticks']} PMU ticks)"
            )
        print("-" * 136)
        print(
            f"{'Method':<52} | {'ActExp%':>7} | {'Tok/Sw':>6} | {'GB/Tok':>7} | "
            f"{'I/O Red':>7} | {'PeakRAM':>8} | {'%FP16':>6} | {'tok/s':>6} | {'estJ/t':>6}"
        )
        print("-" * 136)
        for r in rows:
            print(
                f"{r.method:<52} | {r.active_expert_pct:>6.2f}% | "
                f"{r.mean_tokens_per_nvme_sweep:>6.2f} | {r.nvme_gb_read_per_token:>6.3f}G | "
                f"{r.nvme_io_reduction_vs_airllm_4bit:>6.2f}x | {r.peak_active_weight_ram_mb:>6.1f}MB | "
                f"{r.pct_of_fp16_model_in_ram:>5.2f}% | {r.measured_tok_per_sec:>6.2f} | "
                f"{r.est_joules_per_token:>6.2f}"
            )
        print("-" * 136)
        for idx_s, s in enumerate(samples):
            print(
                f"  Sample #{idx_s+1} [{s['file']} | {s['nvme_sweeps']}/{s['naive_sweeps']} sweeps]: "
                f"{repr(s['completion'][:80])}..."
            )

    out_csv = art_dir / "hf_2026_moe_8b_35b_air_mol_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    out_json = art_dir / "hf_2026_moe_8b_35b_air_mol_samples.json"
    out_json.write_text(json.dumps({"meta": all_meta, "samples": all_samples}, indent=2), encoding="utf-8")
    print(f"\nSaved {out_csv} and {out_json}.")
    print("=" * 136)


if __name__ == "__main__":
    main()
