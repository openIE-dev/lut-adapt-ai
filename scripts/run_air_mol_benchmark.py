"""Phase 18 Runner: `AirMoL` Layer-Wise NVMe Shard Streaming vs. AirLLM Baselines."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import torch

from clai.frontier.air_mol_streamer import run_air_mol_nvme_benchmark


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
    print(f" PHASE 18: AirMoL (4-BIT NVMe LAYER STREAMING + ZERO-VRAM BENES-LUT6) vs. AirLLM BASELINES [Device: {dev}]")
    print("=" * 124)

    rows, meta = run_air_mol_nvme_benchmark(
        work_dir=art_dir / "air_mol_nvme_shards",
        device=dev,
        d_model=384,
        n_layers=8,
        n_heads=8,
        train_steps=240,
        n_eval_prompts=8,
        ctx_len=512,
        gen_len=64,
    )

    out_csv = art_dir / "air_mol_nvme_baselines.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    out_meta = art_dir / "air_mol_nvme_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"  [Matched 8-Layer Models] TF Params={meta['tf_params']:,} | Mamba3 Params={meta['m3_params']:,} "
        f"(Ratio={meta['param_match_ratio']}) | Full FP32={meta['full_model_fp32_mb']} MB"
    )
    print(
        f"  [NVMe Shards on Disk] TF FP16={meta['tf_fp16_disk_mb']} MB -> 4Bit={meta['tf_4bit_disk_mb']} MB | "
        f"Mamba3 FP16={meta['m3_fp16_disk_mb']} MB -> 4Bit={meta['m3_4bit_disk_mb']} MB "
        f"({meta['compression_ratio_fp32_to_4bit']}x vs FP32)"
    )
    print(
        f"  [4-Bit Block64 Fidelity] FP32 Val={meta['fp32_val_bpb']} BPB vs INT4 Streamed={meta['int4_val_bpb']} BPB "
        f"(Delta={meta['bpb_degradation']:+.4f} BPB) | Mamba3 State={meta['m3_pinned_state_kb']} KB | "
        f"LUT6 BRAM={meta['lut6_bram_kb']} KB"
    )
    print("-" * 124)
    print(
        f"{'Method (512B Context -> 64B Code)':<45} | {'B/Swp':>5} | {'Swps':>5} | "
        f"{'NVMe Rd':>9} | {'NVMe Wr':>9} | {'PeakRAM':>8} | {'70B Rd':>8} | {'70B tok/s':>9}"
    )
    print("-" * 124)
    for r in rows:
        print(
            f"{r.method:<45} | {r.mean_bytes_per_nvme_sweep:>5.2f} | {r.mean_nvme_sweeps_for_64b:>5.1f} | "
            f"{r.nvme_mb_read_per_64b:>7.1f}MB | {r.nvme_mb_written_per_64b:>7.1f}MB | "
            f"{r.peak_resident_ram_mb:>6.2f}MB | {r.projected_70b_int4_nvme_gb_per_64b:>6.0f}GB | "
            f"{r.projected_70b_int4_tok_per_sec:>7.2f}/s"
        )
    print("-" * 124)
    print(f"Saved AirMoL vs. AirLLM NVMe streaming metrics -> {out_csv}")


if __name__ == "__main__":
    main()
