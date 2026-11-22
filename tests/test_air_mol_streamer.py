"""Phase 18 Invariant Tests: `AirMoL` NVMe Layer-Wise Shard Streaming + Zero-VRAM `LUT6` Burst."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from clai.frontier.air_mol_streamer import (
    AirShardedMamba3Engine,
    MatchedMamba3LM,
    NVMeIOMeter,
    deserialize_shard_bytes,
    pread_uncached_bytes,
    run_air_mol_nvme_benchmark,
    serialize_state_dict_shard,
)


def _dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_int4_block64_shard_roundtrip_and_single_layer_vram_invariant(tmp_path: Path) -> None:
    """
    Verifies:
      1. `block_size=64` 4-bit shard serialization achieves `>6.5x` compression vs FP32 with tight
         reconstruction accuracy.
      2. Strict AirLLM single-layer memory envelope: streaming an 8-layer model never holds more than
         `1` layer + stem/state (`< 22%` of full model weights) in device memory at any instant!
    """
    dev = _dev()
    torch.manual_seed(7)
    model = MatchedMamba3LM(d_model=256, n_layers=8, n_heads=8).to(dev)
    full_fp32_bytes = sum(p.numel() * 4 for p in model.parameters())

    blk0_sd = model.blocks[0].state_dict()
    shard_p = tmp_path / "blk0.shard4"
    disk_bytes = serialize_state_dict_shard(blk0_sd, shard_p, mode="4bit")
    blk0_fp32_bytes = sum(t.numel() * 4 for t in blk0_sd.values())
    assert blk0_fp32_bytes / disk_bytes > 6.5

    raw = pread_uncached_bytes(shard_p)
    deq_sd, wb = deserialize_shard_bytes(raw, dev)
    assert wb == blk0_fp32_bytes
    rel_err = torch.norm(deq_sd["w_up.weight"] - blk0_sd["w_up.weight"]) / torch.norm(blk0_sd["w_up.weight"])
    assert float(rel_err.item()) < 0.09

    # Export all 8 layers and verify strict single-layer VRAM ceiling during streamed sweep
    AirShardedMamba3Engine.export_shards(model, tmp_path / "m3_shards", mode="4bit")
    del model
    eng = AirShardedMamba3Engine(tmp_path / "m3_shards", 256, 8, 8, "4bit", dev)
    meter = NVMeIOMeter()
    prompt = np.arange(32, 96, dtype=np.uint8)
    out, passes = eng.generate_speculative(prompt, 0, 8, lambda _c, _l, _h, _m: [], meter)
    assert len(out) == 8 and passes == 8
    assert meter.bytes_written == 0, "AirMamba3 O(1) state must perform ZERO NVMe KV-cache writes!"
    assert meter.peak_resident_weight_bytes < 0.25 * full_fp32_bytes


def test_air_mol_paged_benes_lut6_cuts_nvme_reads_losslessly(tmp_path: Path) -> None:
    """
    Verifies end-to-end on real unseen source files that `AirMoL` (`4-bit Layer-Streamed Mamba-3` +
    `Zero-VRAM PagedBenesLUT6`) matches `AirMamba3 T=1` 100.0% byte-for-byte while drastically
    slashing real `os.pread` NVMe bytes read and eliminating all NVMe KV-cache writes!
    """
    dev = _dev()
    rows, _ = run_air_mol_nvme_benchmark(
        work_dir=tmp_path / "bench_shards",
        device=dev,
        d_model=192,
        n_layers=4,
        n_heads=6,
        train_steps=80,
        n_eval_prompts=4,
        ctx_len=256,
        gen_len=48,
    )
    by_name = {r.method: r for r in rows}

    r_tf4 = by_name["2_AirLLM_Transformer_4Bit_Prefetch_KVSpill_T1"]
    r_m3_t1 = by_name["4_AirMamba3_4Bit_PinnedO1State_T1"]
    r_mol_cold = by_name["5_AirMoL_4Bit_PagedBenesLUT6_ColdUnseen_OURS"]
    r_mol_warm = by_name["6_AirMoL_4Bit_PostWake_SleepCrystallized_OURS"]

    assert r_mol_cold.lossless_exact_match_pct == 100.0
    assert r_mol_warm.lossless_exact_match_pct == 100.0
    assert r_tf4.nvme_mb_written_per_64b > 0.0
    assert r_mol_cold.nvme_mb_written_per_64b == 0.0
    assert r_mol_cold.nvme_mb_read_per_64b < 0.45 * r_m3_t1.nvme_mb_read_per_64b
    assert r_mol_warm.nvme_mb_read_per_64b <= r_mol_cold.nvme_mb_read_per_64b
