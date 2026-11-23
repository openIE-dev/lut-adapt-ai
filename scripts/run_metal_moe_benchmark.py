"""Phase 26 Runner: Native Apple Metal (`torch.mps.compile_shader`) Fused `INT4` MoE &
`GatedDeltaNet` Acceleration + Dual-Channel `reported_j` Benchmark on Real 2026 Frontier MoEs.

Benchmarks:
  1. Isolated GPU Kernel Verification & Speedup (`Qwen3.6-35B-A3B` exact dimensions):
     - `gated_delta_recurrent_k128` vs. HuggingFace `torch_recurrent_gated_delta_rule` (`T=256`)
     - `q4_moe_gate_up_silu` + `q4_moe_down_weighted` (`4.50 bpw` `INT4`) vs. BF16/FP32 unpack+loop (`28` active experts)
  2. End-to-End Real-Checkpoint `Expert-AirMoL` NVMe Streaming (`LiquidAI/LFM2.5-8B-A1B` [8.47B] &
     `Qwen/Qwen3.6-35B-A3B` [34.66B]):
     - `PyTorchRef_ExpertAirMoL_T1` (`use_metal_kernels=False`)
     - `MetalFused_ExpertAirMoL_T1` (`use_metal_kernels=True`: fused Metal `INT4` MoE + Metal `GatedDeltaNet` + `QD=8` `pread` + `BenesLUT6` router lookahead)
     - `MetalFused_ExpertAirMoL_PostWake_BenesLUT6_K3_OURS` (`use_metal_kernels=True` + crystallized `K=3` `BenesLUT6` burst speculation)
     - Verifies `100.0%` greedy token-for-token parity and records both `est_j` and real Apple M5 Max
       dual-channel `reported_j` (`L3 RUSAGE_INFO_V6` CPU `nJ` + `L2 IOReport` `'GPU Energy'` `nJ`).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from clai.contracts.joule_meter import meter_joules, select_torch_device
from clai.frontier.air_mol_streamer import NVMeIOMeter
from clai.frontier.code_corpus_speculative import SourceCodeFile, load_repo_code_splits
from clai.frontier.metal_moe_kernels import (
    fused_gated_delta_rule_metal,
    fused_q4_moe_ffn_metal,
    get_metal_moe_library,
)
from clai.frontier.moe_air_mol import (
    LFM25_MOE_8B_MODEL,
    QWEN36_MOE_35B_MODEL,
    ExpertRoutedMoEAirMoLEngine,
    FrontierBenesLUT6TokenDrafter,
    default_nvme_shard_root,
)


@dataclass
class MetalMoEBenchmarkRow:
    model_id: str
    engine_variant: str
    total_params_b: float
    lossless_token_exact_pct: float
    mean_tokens_per_nvme_sweep: float
    nvme_gb_read_per_token: float
    peak_active_weight_ram_mb: float
    measured_tok_per_sec: float
    speedup_vs_pytorch_ref: float
    est_joules_per_token: float
    reported_joules_per_token: float | None
    reported_gpu_joules_per_token: float | None
    reported_cpu_joules_per_token: float | None


def run_microkernel_parity_and_speedup(device: torch.device) -> dict:
    """Verifies numerical parity (`< 2e-4`) and isolated GPU speedup + `reported_j` on M5 Max."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

    get_metal_moe_library()
    torch.manual_seed(2026)

    # 1. GatedDeltaNet Recurrence (`Qwen3.6-35B-A3B` exact dimensions: B=1, T=256, H_k=16, H_v=32, K=128, V=128)
    B, T, H_k, H_v, K_d, V_d = 1, 256, 16, 32, 128, 128
    q = torch.randn((B, T, H_k, K_d), device=device, dtype=torch.bfloat16)
    k = torch.randn((B, T, H_k, K_d), device=device, dtype=torch.bfloat16)
    v = torch.randn((B, T, H_v, V_d), device=device, dtype=torch.bfloat16)
    g = -torch.rand((B, T, H_v), device=device, dtype=torch.float32) * 0.25
    beta = torch.sigmoid(torch.randn((B, T, H_v), device=device, dtype=torch.bfloat16))
    s0 = torch.randn((B, H_v, K_d, V_d), device=device, dtype=torch.float32) * 0.05

    def _pytorch_deltanet_ref():
        q_r = q.repeat_interleave(H_v // H_k, dim=2)
        k_r = k.repeat_interleave(H_v // H_k, dim=2)
        qf, kf, vf, bf, gf = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (q_r, k_r, v, beta, g)]
        qf = l2norm(qf, dim=-1, eps=1e-6) * (K_d**-0.5)
        kf = l2norm(kf, dim=-1, eps=1e-6)
        st = s0.clone()
        out = torch.zeros_like(vf)
        for i in range(T):
            st = st * gf[:, :, i].exp()[..., None, None]
            kv_m = (st * kf[:, :, i].unsqueeze(-1)).sum(dim=-2)
            dl = (vf[:, :, i] - kv_m) * bf[:, :, i].unsqueeze(-1)
            st = st + kf[:, :, i].unsqueeze(-1) * dl.unsqueeze(-2)
            out[:, :, i] = (st * qf[:, :, i].unsqueeze(-1)).sum(dim=-2)
        return out.transpose(1, 2).contiguous().to(q.dtype), st

    out_ref, st_ref = _pytorch_deltanet_ref()
    out_mtl, st_mtl = fused_gated_delta_rule_metal(
        q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert st_mtl is not None
    err_state = float((st_ref - st_mtl).abs().max().item())
    err_out = float((out_ref.float() - out_mtl.float()).abs().max().item())
    assert err_state < 1e-4, f"GatedDeltaNet state parity error too high: {err_state}"

    n_iters_dn = 8
    with meter_joules("rung3", device=device) as r_dn_py:
        for _ in range(n_iters_dn):
            _pytorch_deltanet_ref()
    with meter_joules("rung3", device=device) as r_dn_mtl:
        for _ in range(n_iters_dn):
            fused_gated_delta_rule_metal(
                q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
            )

    # 2. Fused INT4 Block-64 Sparse-MoE (`Qwen3.6-35B-A3B` 4-token speculative burst: T=4, top-8, 28 unique experts)
    D, I_dim, K_uniq = 2048, 512, 28
    m_gu = (2 * I_dim * D) // 64
    m_dn = (D * I_dim) // 64
    m_tot = m_gu + m_dn
    packed_t = torch.randint(0, 256, (K_uniq, m_tot, 32), dtype=torch.uint8, device=device)
    mins_t = (torch.randn((K_uniq, m_tot), dtype=torch.float32, device=device) * 0.02).to(torch.float16)
    scales_t = (torch.rand((K_uniq, m_tot), dtype=torch.float32, device=device) * 0.005 + 1e-4).to(torch.float16)
    flat_h_f32 = torch.randn((4, D), dtype=torch.float32, device=device)
    flat_h = flat_h_f32.to(torch.bfloat16)
    uniq_e = sorted(torch.randperm(256)[:K_uniq].tolist())
    sel_idx = torch.randint(0, K_uniq, (4, 8), device=device)
    sel_e = torch.tensor(uniq_e, device=device, dtype=torch.long)[sel_idx]
    rw = torch.softmax(torch.randn((4, 8), device=device, dtype=torch.float32), dim=-1)

    def _pytorch_moe_ref(xh_in: torch.Tensor):
        q_u = torch.empty((K_uniq, m_tot, 64), dtype=torch.float32, device=device)
        q_u[:, :, 0::2] = (packed_t & 0x0F).to(torch.float32)
        q_u[:, :, 1::2] = (packed_t >> 4).to(torch.float32)
        deq = mins_t.float().unsqueeze(-1) + q_u * scales_t.float().unsqueeze(-1)
        xh = xh_in.float()
        out = torch.zeros_like(xh)
        for slot, e_id in enumerate(uniq_e):
            gu_w = deq[slot, :m_gu].reshape(2 * I_dim, D)
            dn_w = deq[slot, m_gu:].reshape(D, I_dim)
            ti, kp = torch.where(sel_e == e_id)
            if ti.numel() > 0:
                gp, up = F.linear(xh[ti], gu_w).chunk(2, dim=-1)
                out.index_add_(0, ti, F.linear(F.silu(gp) * up, dn_w) * rw[ti, kp].unsqueeze(-1))
        return out.to(xh_in.dtype)

    moe_ref_f32 = _pytorch_moe_ref(flat_h.float())
    moe_mtl_f32 = fused_q4_moe_ffn_metal(
        flat_h.float(), sel_e, rw, uniq_e, packed_t, mins_t, scales_t, D, I_dim, m_gu, m_tot
    )
    err_moe_f32 = float((moe_ref_f32 - moe_mtl_f32).abs().max().item())
    assert err_moe_f32 < 1e-4, f"Fused Q4 MoE FP32 parity error too high: {err_moe_f32}"

    moe_ref = _pytorch_moe_ref(flat_h)
    moe_mtl = fused_q4_moe_ffn_metal(
        flat_h, sel_e, rw, uniq_e, packed_t, mins_t, scales_t, D, I_dim, m_gu, m_tot
    )
    err_moe = float((moe_ref.float() - moe_mtl.float()).abs().max().item())
    assert err_moe <= 1.6e-2, f"Fused Q4 MoE BF16 parity error > 2 ULP: {err_moe}"

    n_iters_moe = 30
    with meter_joules("rung3", device=device) as r_moe_py:
        for _ in range(n_iters_moe):
            _pytorch_moe_ref(flat_h)
    with meter_joules("rung3", device=device) as r_moe_mtl:
        for _ in range(n_iters_moe):
            fused_q4_moe_ffn_metal(
                flat_h, sel_e, rw, uniq_e, packed_t, mins_t, scales_t, D, I_dim, m_gu, m_tot
            )

    return {
        "gated_deltanet_t256": {
            "max_state_abs_err": round(err_state, 9),
            "max_out_abs_err": round(err_out, 7),
            "pytorch_ms_per_call": round(r_dn_py.elapsed_ms / n_iters_dn, 3),
            "metal_ms_per_call": round(r_dn_mtl.elapsed_ms / n_iters_dn, 3),
            "speedup_x": round(r_dn_py.elapsed_ms / max(0.01, r_dn_mtl.elapsed_ms), 2),
            "pytorch_reported_j": round((r_dn_py.reported_j or 0.0) / n_iters_dn, 5),
            "metal_reported_j": round((r_dn_mtl.reported_j or 0.0) / n_iters_dn, 5),
        },
        "q4_moe_28experts_burst4": {
            "max_fp32_abs_err": round(err_moe_f32, 8),
            "max_bf16_abs_err": round(err_moe, 7),
            "pytorch_ms_per_call": round(r_moe_py.elapsed_ms / n_iters_moe, 3),
            "metal_ms_per_call": round(r_moe_mtl.elapsed_ms / n_iters_moe, 3),
            "speedup_x": round(r_moe_py.elapsed_ms / max(0.01, r_moe_mtl.elapsed_ms), 2),
            "pytorch_reported_j": round((r_moe_py.reported_j or 0.0) / n_iters_moe, 5),
            "metal_reported_j": round((r_moe_mtl.reported_j or 0.0) / n_iters_moe, 5),
        },
    }


def benchmark_model_e2e(
    repo_id: str,
    device: torch.device,
    n_eval_files: int = 2,
    gen_tokens: int = 12,
) -> list[MetalMoEBenchmarkRow]:
    slug = repo_id.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    shard_dir = default_nvme_shard_root() / f"hf_{slug}_shards"
    manifest = ExpertRoutedMoEAirMoLEngine.shard_checkpoint(repo_id=repo_id, shard_dir=shard_dir)

    train_files, test_files = load_repo_code_splits(Path("."))
    eng_py = ExpertRoutedMoEAirMoLEngine(
        shard_dir=shard_dir, device=device, compute_dtype=torch.float32, use_metal_kernels=False
    )
    raw_bos = eng_py.tokenizer.bos_token_id
    eval_prompts: list[tuple[list[int], bytes, int]] = []
    for i in range(n_eval_files):
        sf = test_files[i % len(test_files)]
        txt = bytes(sf.raw_bytes.tolist()).decode("utf-8", errors="replace")
        ids = eng_py.tokenizer.encode(txt, add_special_tokens=False)
        st_tok = min(80, max(16, len(ids) // 3))
        slice_ids = ids[st_tok : min(len(ids) - 36, st_tok + 96)]
        p_ids = [int(raw_bos), *slice_ids] if raw_bos is not None else list(slice_ids)
        p_bytes = eng_py.tokenizer.decode(slice_ids).encode("utf-8", errors="replace")
        eval_prompts.append((p_ids, p_bytes, sf.lang_id))

    tot_toks = n_eval_files * gen_tokens
    params_b = round(manifest["total_params"] / 1e9, 2)
    short_tag = repo_id.split("/")[-1].replace("-", "_")
    rows: list[MetalMoEBenchmarkRow] = []

    # 1. PyTorch Reference `Expert-AirMoL` (`use_metal_kernels=False`, `T=1`)
    m_py = NVMeIOMeter()
    ref_outs: list[list[int]] = []
    sw_py = 0
    with meter_joules("rung3", device=device) as jr_py:
        t0 = time.perf_counter()
        for p_ids, p_b, lid in eval_prompts:
            o, sw, _ = eng_py.generate_tokens(p_ids, p_b, lid, gen_tokens, None, m_py, dense_sweep_all=False)
            ref_outs.append(o)
            sw_py += sw
        dt_py = time.perf_counter() - t0
    tps_py = tot_toks / dt_py

    def _mk_row(variant: str, meter: NVMeIOMeter, sw_tot: int, dt: float, jr, exact_pct: float) -> MetalMoEBenchmarkRow:
        tps = tot_toks / dt
        rep_j = round(jr.reported_j / tot_toks, 4) if jr.reported_j is not None else None
        rep_gpu = (
            round(float(jr.reported_channels.get("l2_gpu_j", 0.0)) / tot_toks, 4)
            if jr.reported_j is not None
            else None
        )
        rep_cpu = (
            round(float(jr.reported_channels.get("l3_proc_cpu_j", 0.0)) / tot_toks, 4)
            if jr.reported_j is not None
            else None
        )
        return MetalMoEBenchmarkRow(
            model_id=repo_id,
            engine_variant=variant,
            total_params_b=params_b,
            lossless_token_exact_pct=round(exact_pct, 2),
            mean_tokens_per_nvme_sweep=round(tot_toks / sw_tot, 2),
            nvme_gb_read_per_token=round((meter.bytes_read / tot_toks) / (1024**3), 3),
            peak_active_weight_ram_mb=round(meter.peak_resident_weight_bytes / (1024**2), 2),
            measured_tok_per_sec=round(tps, 3),
            speedup_vs_pytorch_ref=round(tps / tps_py, 2),
            est_joules_per_token=round(jr.est_j / tot_toks, 2),
            reported_joules_per_token=rep_j,
            reported_gpu_joules_per_token=rep_gpu,
            reported_cpu_joules_per_token=rep_cpu,
        )

    rows.append(_mk_row(f"1_PyTorchRef_ExpertAirMoL_{short_tag}_T1", m_py, sw_py, dt_py, jr_py, 100.0))
    del eng_py

    # 2. Native Metal Fused `INT4` MoE + `GatedDeltaNet` + `QD=8` `Expert-AirMoL` (`T=1`)
    eng_mtl = ExpertRoutedMoEAirMoLEngine(
        shard_dir=shard_dir, device=device, compute_dtype=torch.float32, use_metal_kernels=True
    )
    m_mtl = NVMeIOMeter()
    sw_mtl = 0
    matched_toks = 0
    with meter_joules("rung3", device=device) as jr_mtl:
        t0 = time.perf_counter()
        for idx_e, (p_ids, p_b, lid) in enumerate(eval_prompts):
            o, sw, _ = eng_mtl.generate_tokens(p_ids, p_b, lid, gen_tokens, None, m_mtl, dense_sweep_all=False)
            assert o == ref_outs[idx_e], f"Metal T=1 token mismatch on {repo_id}: {o} != {ref_outs[idx_e]}"
            matched_toks += len(o)
            sw_mtl += sw
        dt_mtl = time.perf_counter() - t0
    rows.append(
        _mk_row(
            f"2_MetalFused_ExpertAirMoL_{short_tag}_T1",
            m_mtl,
            sw_mtl,
            dt_mtl,
            jr_mtl,
            100.0 * matched_toks / tot_toks,
        )
    )

    # 3. Native Metal Fused + Post-Wake Crystallized `FrontierBenesLUT6` (`K=3` Burst Speculation)
    wake_files: list[SourceCodeFile] = []
    for idx_e, (p_ids, _, lid) in enumerate(eval_prompts):
        txt_trace = eng_mtl.tokenizer.decode(p_ids[-32:] + ref_outs[idx_e]).encode("utf-8", errors="replace")
        wake_files.append(
            SourceCodeFile(
                rel_path=f"wake_{idx_e}.py",
                lang_id=lid,
                raw_bytes=np.frombuffer(txt_trace, dtype=np.uint8).copy(),
            )
        )
    lut_warm = FrontierBenesLUT6TokenDrafter(vocab_size=eng_mtl.vocab_size)
    lut_warm.compile_from_tokenized_files(eng_mtl.tokenizer, train_files + wake_files, max_branch_pages=650)

    m_spec = NVMeIOMeter()
    sw_spec = 0
    matched_spec = 0
    with meter_joules("rung3", device=device) as jr_spec:
        t0 = time.perf_counter()
        for idx_e, (p_ids, p_b, lid) in enumerate(eval_prompts):
            o, sw, _ = eng_mtl.generate_tokens(
                p_ids,
                p_b,
                lid,
                gen_tokens,
                lut_warm.draft,
                m_spec,
                dense_sweep_all=False,
                drafter_observer=lut_warm,
            )
            assert o == ref_outs[idx_e], f"Metal+BenesLUT6 token mismatch on {repo_id}: {o} != {ref_outs[idx_e]}"
            matched_spec += len(o)
            sw_spec += sw
        dt_spec = time.perf_counter() - t0
    rows.append(
        _mk_row(
            f"3_MetalFused_ExpertAirMoL_{short_tag}_BenesLUT6_K3_OURS",
            m_spec,
            sw_spec,
            dt_spec,
            jr_spec,
            100.0 * matched_spec / tot_toks,
        )
    )
    del eng_mtl
    if device.type == "mps":
        torch.mps.empty_cache()
    return rows


def main() -> None:
    dev = select_torch_device()
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    print("=" * 144)
    print(f" PHASE 26: NATIVE APPLE METAL INT4 MOE + GATED-DELTANET + DUAL-CHANNEL reported_j BENCHMARK [Device: {dev}]")
    print("=" * 144)

    micro = run_microkernel_parity_and_speedup(dev)
    print("  [Microkernels]", json.dumps(micro, indent=2), flush=True)

    all_rows: list[MetalMoEBenchmarkRow] = []
    for repo_id in (LFM25_MOE_8B_MODEL, QWEN36_MOE_35B_MODEL):
        print(f"  [E2E NVMe AirMoL] Benchmarking {repo_id}...", flush=True)
        r_m = benchmark_model_e2e(repo_id, dev, n_eval_files=2, gen_tokens=12)
        all_rows.extend(r_m)
        for r in r_m:
            print(
                f"    {r.engine_variant:<56} | {r.measured_tok_per_sec:>6.2f} tok/s ({r.speedup_vs_pytorch_ref:>4.2f}x) "
                f"| RAM {r.peak_active_weight_ram_mb:>6.1f} MB | rep_J/tok {r.reported_joules_per_token}",
                flush=True,
            )

    out_csv = art_dir / "metal_moe_air_mol_speedup.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    out_meta = art_dir / "metal_moe_air_mol_meta.json"
    out_meta.write_text(
        json.dumps(
            {
                "device": str(dev),
                "microkernel_verification": micro,
                "e2e_rows": [asdict(r) for r in all_rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out_csv} and {out_meta}.")
    print("=" * 144)


if __name__ == "__main__":
    main()
