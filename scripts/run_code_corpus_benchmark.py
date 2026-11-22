"""Runner for Phase 15: Real Polyglot Source-Code Corpus (`520 KB`) Speculative Benchmark.

Trains `Mamba3CodeLM` (`d=256, L=4`) & `CausalTransformerLM` (`d=256, L=4`) on `44` repository
source files across Python, Rust, SystemVerilog, Zig, Go, and Julia, compiles `PagedBenesLUT6CodeDrafter`
(`0` draft FLOPs), and benchmarks all 7 methods on `48` unseen `128`-byte code prefixes from `11`
strictly held-out `.py`, `.rs`, and `.sv` files!
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

from clai.contracts.joule_meter import select_torch_device
from clai.frontier.code_corpus_speculative import run_real_code_corpus_benchmark


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    device = select_torch_device()

    print(f"[CLAI Real-Code Scaling] Training & benchmarking on polyglot repo corpus ({device})...")
    rows, showcases, meta = run_real_code_corpus_benchmark(
        device=device,
        train_steps=950,
        n_eval_Windows=48,
        gen_len=64,
    )

    print(
        f"\n[Corpus Split] Train: {meta['train_files']} files ({meta['train_bytes']:,} B) | "
        f"Held-Out Test: {meta['test_files']} files ({meta['test_bytes']:,} B) | "
        f"Train BPB: {meta['train_bpb']} | LUT6 Pages: 6g={meta['pages_6gram']}, "
        f"4g={meta['pages_4gram']}, Branch160LUT={meta['pages_branch_lut6']} (Compiled in {meta['compile_sec']}s)\n"
    )

    csv_path = out_dir / "real_code_corpus_baselines.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print("=== REAL UNSEEN SOURCE-CODE CORPUS (.py / .rs / .sv) — 7-WAY SOTA COMPARISON ===")
    for r in rows:
        print(
            f"  {r.method:41s} | Lossless={r.lossless_exact_match_pct:5.1f}% | "
            f"Bytes/Pass={r.mean_bytes_per_target_pass:4.2f} | "
            f"Passes(64B)={r.mean_target_passes_for_64b:5.2f} | "
            f"DraftFLOPs={r.draft_flops_per_pass / 1e3:6.1f}K | "
            f"TotMFLOPs/B={r.total_mflops_per_byte:5.2f}M | "
            f"mJ/B={r.millijoules_per_byte:6.3f}"
        )

    print("\n=== QUALITATIVE LOSSLESS GENERATED CODE SAMPLES ON UNSEEN FILES ===")
    for sc in showcases:
        speedup = sc.naive_passes / max(1, sc.lut6_passes)
        print(f"  File: {sc.file_path} | Passes: {sc.lut6_passes}/{sc.naive_passes} ({speedup:.2f}x speedup)")
        print(f"    Prompt[-56:]: {sc.prompt_tail}")
        print(f"    Generated   : {sc.generated_code}")

    (out_dir / "real_code_corpus_samples.json").write_text(
        json.dumps({"meta": meta, "samples": [asdict(s) for s in showcases]}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[CLAI Real-Code Scaling] Saved {csv_path} and real_code_corpus_samples.json")


if __name__ == "__main__":
    main()
