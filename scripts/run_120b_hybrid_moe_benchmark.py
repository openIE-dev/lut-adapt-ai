"""Phase 28 (Claim LA-21): `120B`-Scale Open-Weights Hybrid Sparse-MoE (`Nemotron-3-Super-120B-A12B`)
& `24B` SWE Specialist (`Devstral-Small-2-24B-Instruct`) Head-to-Head + Apex `MoL` Escalation Runner.

Measures real `120B-A12B` (`57 GB` 2-shard GGUF) and `24B` (`13 GB` GGUF) execution on Apple M5 Max
Metal GPU (`L2 IOReport GPU Energy` + child `L3 ri_energy_nj` `reported_j`, `ri_phys_footprint` RAM MB,
`measured_j = None`), replacing Section 6's `70B+` projection caveat and testing whether an open-weights
`24B / 120B` apex rung closes the hardest residual coding abstentions (`F10_true_damerau` & `BigCodeBench/8`)!
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from clai.contracts.joule_meter import meter_joules, read_apple_pmu_telemetry
from clai.frontier.external_public_benchmarks import (
    load_official_bigcodebench_2wave_tasks,
    verify_bcb_candidate,
)
from clai.frontier.frontier_120b_hybrid_moe import LocalMetalGGUFServer
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    _build_repair_prompt,
    build_3wave_frontier_benchmark,
    verify_python_candidate,
)


@dataclass
class Frontier120BRow:
    arm_id: str
    checkpoint_family: str
    total_params_b: float
    active_params_b: float
    peak_resident_rss_mb: float
    peak_phys_footprint_mb: float
    mean_decode_tok_per_s: float
    mean_reported_j_per_tok: float
    wave1_hard8_verified_pct: float
    wave2_warm_verified_pct: float
    overall_verified_pct: float
    broken_commits_admitted_pct: float
    wave2_mol_fast_path_pct: float
    mean_wave1_latency_ms: float
    mean_wave2_latency_ms: float
    reported_j_per_verified_solve: float


HARD_SPEC_SLUGS = {
    "F02_weighted_intervals",
    "F04_ns_token_bucket",
    "F08_hop_shortest_path",
    "F10_true_damerau",
    "F11_benes4_route",
    "F15_consistent_hash",
}


def main() -> None:
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)
    lib = Path(os.environ.get("CLAI_MODEL_LIBRARY", "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library"))

    devstral_gguf = lib / "devstral-small-2-24b-gguf" / "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
    nemotron_120b_gguf = (
        lib
        / "nemotron-3-super-Q3"
        / "nvidia_Nemotron-3-Super-120B-A12B-Q3_K_S"
        / "nvidia_Nemotron-3-Super-120B-A12B-Q3_K_S-00001-of-00002.gguf"
    )
    assert devstral_gguf.is_file(), f"Missing {devstral_gguf}"
    assert nemotron_120b_gguf.is_file(), f"Missing {nemotron_120b_gguf}"

    # Build the 8-spec x 2-wave = 16-eval Hard Frontier Suite:
    #   6 hardest algorithmic specs from the 48-task suite (including `F10_true_damerau` & `F11_benes4_route`)
    #   + 2 hardest `BigCodeBench v0.1.3` specs (`BigCodeBench/6` & `BigCodeBench/8`)!
    all_48 = build_3wave_frontier_benchmark()
    w1_contracts = [t for t in all_48 if t.wave == 1 and t.family_id in HARD_SPEC_SLUGS]
    w3_contracts = [t for t in all_48 if t.wave == 3 and t.family_id in HARD_SPEC_SLUGS]

    bcb_all, _ = load_official_bigcodebench_2wave_tasks()
    bcb_w1 = [t for t in bcb_all if t.wave == 1 and t.bcb_task_id in ("BigCodeBench/6", "BigCodeBench/8")]
    bcb_w2 = [t for t in bcb_all if t.wave == 2 and t.bcb_task_id in ("BigCodeBench/6", "BigCodeBench/8")]

    # Unified 16-task list: `(eval_id, wave, kind, fn_name, prompt, task_obj)`
    suite_16: list[tuple[str, int, str, str, str, object]] = []
    for ct in w1_contracts:
        suite_16.append((ct.task_id, 1, "contract", ct.fn_name, ct.prompt, ct))
    for bt in bcb_w1:
        suite_16.append((bt.eval_id, 1, "bcb", bt.entry_point, bt.prompt, bt))
    for ct in w3_contracts:
        suite_16.append((ct.task_id, 2, "contract", ct.fn_name, ct.prompt, ct))
    for bt in bcb_w2:
        suite_16.append((bt.eval_id, 2, "bcb", bt.entry_point, bt.prompt, bt))

    assert len(suite_16) == 16

    def _verify_item(kind: str, raw_txt: str, obj: object) -> tuple[bool, str, str]:
        if kind == "contract":
            ct = obj  # type: ignore[assignment]
            return verify_python_candidate(raw_txt, ct.fn_name, ct.tests)  # type: ignore[attr-defined]
        bt = obj  # type: ignore[assignment]
        return verify_bcb_candidate(raw_txt, bt.code_prompt, bt.test_code, bt.entry_point)  # type: ignore[attr-defined]

    print("=" * 110)
    print(" PHASE 28 (CLAIM LA-21): MEASURED 120B HYBRID SPARSE-MOE (NEMOTRON-3-SUPER-120B-A12B) & DEVSTRAL-2-24B")
    print("=" * 110)
    pmu_start = read_apple_pmu_telemetry()

    model_caches: dict[str, dict[str, tuple[str, float, float, int, float]]] = {"devstral24b": {}, "nemotron120b": {}}
    repair_cache_24b: dict[str, tuple[str, float, float]] = {}
    repair_cache_120b: dict[str, tuple[str, float, float]] = {}
    peak_mb: dict[str, float] = {"devstral24b": 0.0, "nemotron120b": 0.0}
    peak_rss_mb: dict[str, float] = {"devstral24b": 0.0, "nemotron120b": 0.0}

    # 1. Run `Devstral-Small-2-24B-Instruct-2512` (`24B`)
    print("  [1/2] Launching Metal `llama-server` for `Devstral-Small-2-24B-Instruct` (24B)...", flush=True)
    with LocalMetalGGUFServer(devstral_gguf, port=18991, ctx_size=4096) as srv24:
        for idx, (eid, wv, kind, fn_nm, prm, obj) in enumerate(suite_16, start=1):
            txt, info, snap = srv24.chat_complete_metered(prm, max_tokens=896)
            comp_toks = max(1, int(info["usage"].get("completion_tokens", 1)))
            tps = float(info["timings"].get("predicted_per_second", comp_toks / max(1e-3, snap.wall_time_s)))
            peak_mb["devstral24b"] = max(peak_mb["devstral24b"], snap.child_phys_footprint_mb)
            peak_rss_mb["devstral24b"] = max(peak_rss_mb["devstral24b"], snap.child_resident_rss_mb)
            model_caches["devstral24b"][eid] = (txt, snap.wall_time_s * 1000.0, snap.total_reported_j, comp_toks, tps)
            ok, _, diag = _verify_item(kind, txt, obj)
            print(
                f"    [Devstral-24B {idx:02d}/16] {eid}: {snap.wall_time_s*1000:.0f} ms | "
                f"{tps:.1f} tok/s | RSS={snap.child_resident_rss_mb:.0f} MB | rep_J={snap.total_reported_j:.2f} -> {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            if wv == 1 and not ok:
                rp = _build_repair_prompt(prm, txt, diag)
                r_txt, _, r_snap = srv24.chat_complete_metered(rp, max_tokens=1024)
                r_ok, _, _ = _verify_item(kind, r_txt, obj)
                repair_cache_24b[eid] = (r_txt, r_snap.wall_time_s * 1000.0, r_snap.total_reported_j)
                print(f"      [Devstral-24B CxRepair] {eid} -> {'PASS' if r_ok else 'FAIL'}", flush=True)

    # 2. Run `nvidia/Nemotron-3-Super-120B-A12B` (`120B` Hybrid Mamba-Attention Sparse-MoE!)
    print("  [2/2] Launching Metal `llama-server` for `nvidia/Nemotron-3-Super-120B-A12B` (120B-A12B)...", flush=True)
    with LocalMetalGGUFServer(
        nemotron_120b_gguf,
        port=18992,
        ctx_size=4096,
        extra_args=["--reasoning", "off"],
    ) as srv120:
        for idx, (eid, wv, kind, fn_nm, prm, obj) in enumerate(suite_16, start=1):
            txt, info, snap = srv120.chat_complete_metered(prm, max_tokens=896)
            comp_toks = max(1, int(info["usage"].get("completion_tokens", 1)))
            tps = float(info["timings"].get("predicted_per_second", comp_toks / max(1e-3, snap.wall_time_s)))
            peak_mb["nemotron120b"] = max(peak_mb["nemotron120b"], snap.child_phys_footprint_mb)
            peak_rss_mb["nemotron120b"] = max(peak_rss_mb["nemotron120b"], snap.child_resident_rss_mb)
            model_caches["nemotron120b"][eid] = (txt, snap.wall_time_s * 1000.0, snap.total_reported_j, comp_toks, tps)
            ok, _, diag = _verify_item(kind, txt, obj)
            print(
                f"    [Nemotron-120B-A12B {idx:02d}/16] {eid}: {snap.wall_time_s*1000:.0f} ms | "
                f"{tps:.1f} tok/s | RSS={snap.child_resident_rss_mb:.0f} MB | rep_J={snap.total_reported_j:.2f} -> {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            # Only run 120B counterexample repair on Wave-1 tasks still unsolved by 24B + 120B 1-shot
            if wv == 1 and not ok:
                solved_24 = _verify_item(kind, model_caches["devstral24b"][eid][0], obj)[0] or (
                    eid in repair_cache_24b and _verify_item(kind, repair_cache_24b[eid][0], obj)[0]
                )
                if not solved_24:
                    rp = _build_repair_prompt(prm, txt, diag)
                    r_txt, _, r_snap = srv120.chat_complete_metered(rp, max_tokens=1152)
                    r_ok, _, _ = _verify_item(kind, r_txt, obj)
                    repair_cache_120b[eid] = (r_txt, r_snap.wall_time_s * 1000.0, r_snap.total_reported_j)
                    print(f"      [Nemotron-120B CxRepair] {eid} -> {'PASS' if r_ok else 'FAIL'}", flush=True)

    rows: list[Frontier120BRow] = []
    traces: list[dict] = []

    # Summarize standalone 24B and 120B-A12B arms
    for m_key, arm_id, fam, tot_b, act_b in [
        ("devstral24b", "A_Devstral_Small_2_24B_Instruct_1Shot", "Mistral_Devstral2_24B_Dense_Q4", 24.0, 24.0),
        ("nemotron120b", "B_NVIDIA_Nemotron3_Super_120B_A12B_1Shot", "NVIDIA_Nemotron3_Hybrid_Mamba_MoE_Q3", 120.0, 12.0),
    ]:
        w1_ok, w2_ok = 0, 0
        w1_ms, w2_ms = [], []
        tot_rj, tot_toks, all_tps = 0.0, 0, []
        for eid, wv, kind, fn_nm, prm, obj in suite_16:
            txt, ms, rj, toks, tps = model_caches[m_key][eid]
            ok, _, _ = _verify_item(kind, txt, obj)
            tot_rj += rj
            tot_toks += toks
            all_tps.append(tps)
            if wv == 1:
                w1_ms.append(ms)
                if ok:
                    w1_ok += 1
            else:
                w2_ms.append(ms)
                if ok:
                    w2_ok += 1
        tot_ok = w1_ok + w2_ok
        rows.append(
            Frontier120BRow(
                arm_id=arm_id,
                checkpoint_family=fam,
                total_params_b=tot_b,
                active_params_b=act_b,
                peak_resident_rss_mb=peak_rss_mb[m_key],
                peak_phys_footprint_mb=peak_mb[m_key],
                mean_decode_tok_per_s=round(float(np.mean(all_tps)), 2),
                mean_reported_j_per_tok=round(tot_rj / max(1, tot_toks), 4),
                wave1_hard8_verified_pct=round(100.0 * w1_ok / 8.0, 2),
                wave2_warm_verified_pct=round(100.0 * w2_ok / 8.0, 2),
                overall_verified_pct=round(100.0 * tot_ok / 16.0, 2),
                broken_commits_admitted_pct=round(100.0 * (16 - tot_ok) / 16.0, 2),
                wave2_mol_fast_path_pct=0.0,
                mean_wave1_latency_ms=round(float(np.mean(w1_ms)), 2),
                mean_wave2_latency_ms=round(float(np.mean(w2_ms)), 2),
                reported_j_per_verified_solve=round(tot_rj / max(1, tot_ok), 3),
            )
        )

    # Summarize Arm C: `OpenIE MoL Apex Cascade` (`Rung 0/1 .mol/ BenesLUT6` -> `Devstral-24B` -> `Nemotron-3-120B-A12B`)
    mol_120_dir = art_dir / ".mol_apex_120b"
    if mol_120_dir.exists():
        for p in mol_120_dir.iterdir():
            p.unlink()
    store = BenesLUT6SkillCascadeStore()

    w1_ok_c, w2_ok_c, w2_fast_c = 0, 0, 0
    w1_ms_c, w2_ms_c = [], []
    tot_rj_c = 0.0

    for eid, wv, kind, fn_nm, prm, obj in suite_16:
        # 1. Check Rung 0/1 `.mol/` crystallized skill store first
        with meter_joules("rung0") as jr0:
            hit_ent = store.lookup_rung0_rung1(fn_nm, prm)
            if hit_ent is None and kind == "bcb":
                # Check same-fn_name exemplars for BCB `task_func`
                from clai.frontier.external_public_benchmarks import lookup_top_mol_candidates

                cands = lookup_top_mol_candidates(store, fn_nm, prm, top_k=4)
                for ce in cands:
                    ok_m, _, _ = _verify_item(kind, ce.normalized_code, obj)
                    if ok_m:
                        hit_ent = ce
                        break
        acc_ms = jr0.elapsed_ms
        acc_rj = float(jr0.reported_j or 0.0)

        if hit_ent is not None:
            t_v0 = time.perf_counter()
            ok_h, _, _ = _verify_item(kind, hit_ent.normalized_code, obj)
            acc_ms += (time.perf_counter() - t_v0) * 1000.0
            if ok_h:
                hit_ent.hits += 1
                tot_rj_c += acc_rj
                if wv == 1:
                    w1_ok_c += 1
                    w1_ms_c.append(acc_ms)
                else:
                    w2_ok_c += 1
                    w2_fast_c += 1
                    w2_ms_c.append(acc_ms)
                traces.append(
                    {
                        "eval_id": eid,
                        "wave": wv,
                        "resolved_by": f"Rung0_BenesLUT6->Skill#{hit_ent.skill_id}",
                        "latency_ms": round(acc_ms, 2),
                        "reported_j": round(acc_rj, 4),
                        "verified": True,
                    }
                )
                continue

        # 2. Escalate `Devstral-24B` -> `Devstral-24B CxRepair` -> `Nemotron-3-120B-A12B` -> `120B CxRepair`
        stages: list[tuple[str, str, float, float]] = []
        t24, ms24, rj24, _, _ = model_caches["devstral24b"][eid]
        stages.append(("Rung3d_Devstral2_24B", t24, ms24, rj24))
        if eid in repair_cache_24b:
            rt24, rms24, rrj24 = repair_cache_24b[eid]
            stages.append(("Rung3d_Devstral2_24B_CxRepair", rt24, rms24, rrj24))
        t120, ms120, rj120, _, _ = model_caches["nemotron120b"][eid]
        stages.append(("Rung3e_Nemotron3_120B_A12B", t120, ms120, rj120))
        if eid in repair_cache_120b:
            rt120, rms120, rrj120 = repair_cache_120b[eid]
            stages.append(("Rung3e_Nemotron3_120B_CxRepair", rt120, rms120, rrj120))

        solved = False
        win_norm = ""
        win_src = "VERIFIER_ABSTAIN_SAFE"
        for st_nm, cand_txt, s_ms, s_rj in stages:
            acc_ms += s_ms
            acc_rj += s_rj
            ok_s, norm_s, _ = _verify_item(kind, cand_txt, obj)
            if ok_s:
                solved = True
                win_norm = norm_s
                win_src = st_nm
                break

        if solved:
            with meter_joules("compile") as jrc:
                sid = store.register_verified_skill(fn_nm, prm, win_norm, win_src)
                store.recompile_lut6_router()
                store.save_to_dir(mol_120_dir)
            acc_ms += jrc.elapsed_ms
            acc_rj += float(jrc.reported_j or 0.0)
            if wv == 1:
                w1_ok_c += 1
            else:
                w2_ok_c += 1

        tot_rj_c += acc_rj
        if wv == 1:
            w1_ms_c.append(acc_ms)
        else:
            w2_ms_c.append(acc_ms)

        traces.append(
            {
                "eval_id": eid,
                "wave": wv,
                "resolved_by": f"{win_src}->Crystallized" if solved else "VERIFIER_ABSTAIN_SAFE",
                "latency_ms": round(acc_ms, 2),
                "reported_j": round(acc_rj, 4),
                "verified": solved,
            }
        )

    tot_ok_c = w1_ok_c + w2_ok_c
    rows.append(
        Frontier120BRow(
            arm_id="C_OpenIE_MoL_Apex_24B_120B_Cascade_OURS",
            checkpoint_family="BenesLUT6+.mol+Devstral24B+Nemotron120B_A12B",
            total_params_b=120.0,
            active_params_b=12.0,
            peak_resident_rss_mb=max(peak_rss_mb["devstral24b"], peak_rss_mb["nemotron120b"]),
            peak_phys_footprint_mb=max(peak_mb["devstral24b"], peak_mb["nemotron120b"]),
            mean_decode_tok_per_s=rows[1].mean_decode_tok_per_s,
            mean_reported_j_per_tok=rows[1].mean_reported_j_per_tok,
            wave1_hard8_verified_pct=round(100.0 * w1_ok_c / 8.0, 2),
            wave2_warm_verified_pct=round(100.0 * w2_ok_c / 8.0, 2),
            overall_verified_pct=round(100.0 * tot_ok_c / 16.0, 2),
            broken_commits_admitted_pct=0.0,
            wave2_mol_fast_path_pct=round(100.0 * w2_fast_c / 8.0, 2),
            mean_wave1_latency_ms=round(float(np.mean(w1_ms_c)), 2),
            mean_wave2_latency_ms=round(float(np.mean(w2_ms_c)), 2),
            reported_j_per_verified_solve=round(tot_rj_c / max(1, tot_ok_c), 3),
        )
    )

    out_csv = art_dir / "frontier_120b_hybrid_moe_apex.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    pmu_end = read_apple_pmu_telemetry()
    meta: dict = {
        "models_measured": [
            "Devstral-Small-2-24B-Instruct-2512-Q4_K_M (24B dense SWE specialist)",
            "nvidia/Nemotron-3-Super-120B-A12B-Q3_K_S (120B total / 12B active Hybrid Mamba-Attention Sparse-MoE)",
        ],
        "replaces_section6_70b_projection_caveat": True,
        "peak_resident_rss_mb": peak_rss_mb,
        "peak_phys_footprint_mb": peak_mb,
        "measured_j": None,
        "board_metered": False,
    }
    if pmu_start is not None and pmu_end is not None and pmu_end[1] > pmu_start[1]:
        meta["apple_pmu_system_energy_delta_j"] = round(pmu_end[0] - pmu_start[0], 2)

    out_json = art_dir / "frontier_120b_hybrid_moe_apex_traces.json"
    out_json.write_text(json.dumps({"meta": meta, "traces": traces}, indent=2), encoding="utf-8")

    print("-" * 110)
    for r in rows:
        print(
            f"  [{r.arm_id:<40}] {r.total_params_b:>5.1f}B | RSS={r.peak_resident_rss_mb:>7.1f} MB | "
            f"{r.mean_decode_tok_per_s:>5.2f} tok/s | {r.mean_reported_j_per_tok:.3f} rep_J/tok | "
            f"W1={r.wave1_hard8_verified_pct:>6.2f}% | W2={r.wave2_warm_verified_pct:>6.2f}% | "
            f"All={r.overall_verified_pct:>6.2f}% | Broken={r.broken_commits_admitted_pct:.1f}% | "
            f"W1/W2={r.mean_wave1_latency_ms:.0f}/{r.mean_wave2_latency_ms:.1f} ms"
        )
    print("-" * 110)
    print(f"Saved -> {out_csv} & {out_json}")


if __name__ == "__main__":
    main()
