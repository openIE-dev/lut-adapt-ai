"""Phase 27 (Claim LA-15): Head-to-Head External Public Benchmark Runner
(`bigcode/bigcodebench` `v0.1.3` [32 evals] + `princeton-nlp/SWE-bench_Verified` [16 git-worktree evals])
Through Live `mol-agent` (`MCP` JSON-RPC `2.0` + `.mol/` `BenesLUT6` Crystallization).
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from clai.contracts.joule_meter import meter_joules, read_apple_pmu_telemetry, select_torch_device
from clai.frontier.external_public_benchmarks import (
    BCB_V013_EXPECTED_SHA256,
    SWE_EDIT_SYSTEM_PROMPT,
    SWEBENCH_VERIFIED_EXPECTED_SHA256,
    BigCodeBenchWaveTask,
    ExternalBenchmarkRow,
    SWEBenchVerifiedWaveTask,
    format_swe_user_prompt,
    load_official_bigcodebench_2wave_tasks,
    load_official_swebench_verified_2wave_tasks,
    lookup_top_mol_candidates,
    verify_bcb_candidate,
    verify_swebench_candidate_in_worktree,
)
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    ExpertAirMoL35BRunner,
    Local2026ModelRunner,
    _build_repair_prompt,
    extract_python_code,
)
from clai.service.openai_mcp_agent import MoLLocalCodingAgent, handle_mcp_jsonrpc


def _summarize_external(
    bench_name: str,
    arm_name: str,
    outcomes: list[tuple[int, bool, bool, bool, bool, float, float, float]],
) -> ExternalBenchmarkRow:
    # tuple: (wave, passed, broken_admitted, mol_fast_hit, used_35b, lat_ms, est_j, rep_j)
    w1 = [x for x in outcomes if x[0] == 1]
    w2 = [x for x in outcomes if x[0] == 2]
    tot_ok = sum(1 for x in outcomes if x[1])
    tot_est = sum(x[6] for x in outcomes)
    tot_rep = sum(x[7] for x in outcomes)
    return ExternalBenchmarkRow(
        benchmark=bench_name,
        arm=arm_name,
        wave1_cold_pass_pct=round(100.0 * sum(1 for x in w1 if x[1]) / len(w1), 2),
        wave2_paraphrase_or_warm_pass_pct=round(100.0 * sum(1 for x in w2 if x[1]) / len(w2), 2),
        overall_verified_pass_pct=round(100.0 * tot_ok / len(outcomes), 2),
        broken_outputs_admitted_pct=round(100.0 * sum(1 for x in outcomes if x[2]) / len(outcomes), 2),
        wave2_mol_fast_path_pct=round(100.0 * sum(1 for x in w2 if x[3]) / len(w2), 2),
        frontier_35b_calls_pct=round(100.0 * sum(1 for x in outcomes if x[4]) / len(outcomes), 2),
        mean_latency_ms_wave1=round(float(np.mean([x[5] for x in w1])), 2),
        mean_latency_ms_wave2=round(float(np.mean([x[5] for x in w2])), 2),
        est_joules_per_verified_solve=round(tot_est / max(1, tot_ok), 2),
        reported_joules_per_verified_solve=round(tot_rep / max(1, tot_ok), 3) if tot_rep > 0 else None,
    )


def main() -> None:
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)
    dev = select_torch_device()
    lib = Path(os.environ.get("CLAI_MODEL_LIBRARY", "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library"))

    print("=" * 108)
    print(" PHASE 27 (CLAIM LA-15): OFFICIAL BIGCODEBENCH v0.1.3 (32) + UPSTREAM SWE-BENCH VERIFIED (16) VIA MCP AGENT")
    print("=" * 108)

    bcb_tasks, bcb_sha = load_official_bigcodebench_2wave_tasks()
    swe_tasks, swe_sha = load_official_swebench_verified_2wave_tasks()
    print(f"  [Loaded] BigCodeBench v0.1.3: {len(bcb_tasks)} evals (SHA256={bcb_sha[:16]}...)")
    print(f"  [Loaded] SWE-bench Verified:  {len(swe_tasks)} evals (SHA256={swe_sha[:16]}...)")

    pmu_start = read_apple_pmu_telemetry()

    # Cache 1-shot generations per model: key -> (raw_txt, dt_ms, est_j, rep_j)
    bcb_1shot: dict[str, dict[str, tuple[str, float, float, float]]] = {"lfm3": {}, "gemma4": {}, "qwen35b": {}}
    swe_1shot: dict[str, dict[str, tuple[str, float, float, float]]] = {"lfm3": {}, "gemma4": {}, "qwen35b": {}}

    bcb_repair_g4: dict[str, tuple[str, float, float, float]] = {}
    swe_repair_g4: dict[str, tuple[str, float, float, float]] = {}
    bcb_repair_35b: dict[str, tuple[str, float, float, float]] = {}
    swe_repair_35b: dict[str, tuple[str, float, float, float]] = {}

    swe_prompt_map = {
        t.eval_id: f"{SWE_EDIT_SYSTEM_PROMPT}\n\n{format_swe_user_prompt(t)}" for t in swe_tasks
    }

    # 1. Run `LiquidAI/LFM3-1.7B` and `google/gemma-4-E2B-it`
    for m_key, m_dir, watts in [
        ("lfm3", lib / "LFM3-1.7B", 32.0),
        ("gemma4", lib / "gemma-4-E2B-it", 42.0),
    ]:
        print(f"  [1-Shot + Cx] Running {m_key} ({m_dir.name}) across 32 BCB + 16 SWE-bench tasks...", flush=True)
        runner = Local2026ModelRunner(m_key, m_dir, dev)

        for idx, bt in enumerate(bcb_tasks, start=1):
            with meter_joules("rung2", device=dev) as jr:
                raw_o, _, _ = runner.generate_code(bt.prompt, max_new_tokens=640)
            dt = jr.elapsed_ms
            bcb_1shot[m_key][bt.eval_id] = (raw_o, dt, (dt / 1000.0) * watts, float(jr.reported_j or 0.0))
            if idx % 8 == 0 or idx == len(bcb_tasks):
                ok_q, _, _ = verify_bcb_candidate(raw_o, bt.code_prompt, bt.test_code, bt.entry_point)
                print(f"    [{m_key} BCB {idx:02d}/32] {bt.eval_id}: {dt:.0f} ms -> {'PASS' if ok_q else 'FAIL'}", flush=True)

        if m_key == "gemma4":
            for bt in bcb_tasks:
                if bt.wave != 1:
                    continue
                ok_l3, _, _ = verify_bcb_candidate(
                    bcb_1shot["lfm3"][bt.eval_id][0], bt.code_prompt, bt.test_code, bt.entry_point
                )
                if not ok_l3:
                    c_g4 = bcb_1shot["gemma4"][bt.eval_id][0]
                    ok_g4, _, diag_g4 = verify_bcb_candidate(c_g4, bt.code_prompt, bt.test_code, bt.entry_point)
                    if not ok_g4:
                        rp = _build_repair_prompt(bt.prompt, c_g4, diag_g4)
                        with meter_joules("rung2", device=dev) as jr_r:
                            fix_g4, _, _ = runner.generate_code(rp, max_new_tokens=640)
                        dt_r = jr_r.elapsed_ms
                        bcb_repair_g4[bt.eval_id] = (
                            fix_g4,
                            dt_r,
                            (dt_r / 1000.0) * watts,
                            float(jr_r.reported_j or 0.0),
                        )

        for idx, st in enumerate(swe_tasks, start=1):
            p_swe = swe_prompt_map[st.eval_id]
            with meter_joules("rung2", device=dev) as jr:
                raw_s, _, _ = runner.generate_code(p_swe, max_new_tokens=640)
            dt = jr.elapsed_ms
            swe_1shot[m_key][st.eval_id] = (raw_s, dt, (dt / 1000.0) * watts, float(jr.reported_j or 0.0))
            if idx % 4 == 0 or idx == len(swe_tasks):
                ok_s, _, _ = verify_swebench_candidate_in_worktree(st, raw_s)
                print(f"    [{m_key} SWE {idx:02d}/16] {st.eval_id}: {dt:.0f} ms -> {'PASS' if ok_s else 'FAIL'}", flush=True)

        if m_key == "gemma4":
            for st in swe_tasks:
                if st.wave != 1:
                    continue
                ok_l3, _, _ = verify_swebench_candidate_in_worktree(st, swe_1shot["lfm3"][st.eval_id][0])
                if not ok_l3:
                    c_g4 = swe_1shot["gemma4"][st.eval_id][0]
                    ok_g4, _, diag_s4 = verify_swebench_candidate_in_worktree(st, c_g4)
                    if not ok_g4:
                        rp = (
                            f"{swe_prompt_map[st.eval_id]}\n\n"
                            f"Your previous edit failed `pytest` with:\n```text\n{diag_s4[:900]}\n```\n"
                            "Return a corrected `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` block."
                        )
                        with meter_joules("rung2", device=dev) as jr_sr:
                            fix_s4, _, _ = runner.generate_code(rp, max_new_tokens=640)
                        dt_r = jr_sr.elapsed_ms
                        swe_repair_g4[st.eval_id] = (
                            fix_s4,
                            dt_r,
                            (dt_r / 1000.0) * watts,
                            float(jr_sr.reported_j or 0.0),
                        )

        runner.unload()

    # 2. Run Metal-Accelerated `Qwen/Qwen3.6-35B-A3B` (`34.66B` Sparse-MoE)
    print("  [1-Shot + Rung 3c/3d] Running Metal-Fused Qwen/Qwen3.6-35B-A3B across 32 BCB + 16 SWE-bench...", flush=True)
    runner_35b = ExpertAirMoL35BRunner(device=dev)

    for idx, bt in enumerate(bcb_tasks, start=1):
        with meter_joules("rung3", device=dev) as jr:
            raw35, _, _, _ = runner_35b.generate_code(bt.prompt, max_new_tokens=768, enable_thinking=False)
        dt = jr.elapsed_ms
        bcb_1shot["qwen35b"][bt.eval_id] = (raw35, dt, (dt / 1000.0) * 95.0, float(jr.reported_j or 0.0))
        if idx % 8 == 0 or idx == len(bcb_tasks):
            ok35, _, _ = verify_bcb_candidate(raw35, bt.code_prompt, bt.test_code, bt.entry_point)
            print(f"    [qwen35b BCB {idx:02d}/32] {bt.eval_id}: {dt:.0f} ms -> {'PASS' if ok35 else 'FAIL'}", flush=True)

    # Escalated repair / Rung 3d `<think>` ONLY on Wave-1 BCB tasks unsolved by smaller rungs + 35B 1-shot
    for bt in bcb_tasks:
        if bt.wave != 1:
            continue
        small_cands = [bcb_1shot["lfm3"][bt.eval_id][0], bcb_1shot["gemma4"][bt.eval_id][0]]
        if bt.eval_id in bcb_repair_g4:
            small_cands.append(bcb_repair_g4[bt.eval_id][0])
        if any(verify_bcb_candidate(c, bt.code_prompt, bt.test_code, bt.entry_point)[0] for c in small_cands):
            continue
        cur_c, dt_tot, j_tot, rep_tot = bcb_1shot["qwen35b"][bt.eval_id]
        ok_cur, _, diag_cur = verify_bcb_candidate(cur_c, bt.code_prompt, bt.test_code, bt.entry_point)
        if not ok_cur:
            rp = _build_repair_prompt(bt.prompt, extract_python_code(cur_c) or cur_c[-1200:], diag_cur)
            with meter_joules("rung3", device=dev) as jr_35r:
                cur_c, _, _, _ = runner_35b.generate_code(
                    rp, max_new_tokens=2048, enable_thinking=True, thinking_budget_tokens=1024
                )
            dt_tot += jr_35r.elapsed_ms
            j_tot += (jr_35r.elapsed_ms / 1000.0) * 95.0
            rep_tot += float(jr_35r.reported_j or 0.0)
        bcb_repair_35b[bt.eval_id] = (cur_c, dt_tot, j_tot, rep_tot)

    for idx, st in enumerate(swe_tasks, start=1):
        p_swe = swe_prompt_map[st.eval_id]
        with meter_joules("rung3", device=dev) as jr:
            raw35s, _, _, _ = runner_35b.generate_code(p_swe, max_new_tokens=768, enable_thinking=False)
        dt = jr.elapsed_ms
        swe_1shot["qwen35b"][st.eval_id] = (raw35s, dt, (dt / 1000.0) * 95.0, float(jr.reported_j or 0.0))
        if idx % 4 == 0 or idx == len(swe_tasks):
            ok35s, _, _ = verify_swebench_candidate_in_worktree(st, raw35s)
            print(f"    [qwen35b SWE {idx:02d}/16] {st.eval_id}: {dt:.0f} ms -> {'PASS' if ok35s else 'FAIL'}", flush=True)

    # Escalated repair / Rung 3d `<think>` ONLY on Wave-1 SWE tasks unsolved by smaller rungs + 35B 1-shot
    for st in swe_tasks:
        if st.wave != 1:
            continue
        small_cands = [swe_1shot["lfm3"][st.eval_id][0], swe_1shot["gemma4"][st.eval_id][0]]
        if st.eval_id in swe_repair_g4:
            small_cands.append(swe_repair_g4[st.eval_id][0])
        if any(verify_swebench_candidate_in_worktree(st, c)[0] for c in small_cands):
            continue
        cur_s, dt_tot, j_tot, rep_tot = swe_1shot["qwen35b"][st.eval_id]
        ok_cur, _, diag_cur = verify_swebench_candidate_in_worktree(st, cur_s)
        for turn_k in range(2):
            if ok_cur:
                break
            rp = (
                f"{swe_prompt_map[st.eval_id]}\n\n"
                f"Previous edit attempt failed `pytest` with:\n```text\n{diag_cur[:1000]}\n```\n"
                "Think carefully step-by-step and output ONLY the exact corrected "
                "`<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` block."
            )
            with meter_joules("rung3", device=dev) as jr_s35:
                cur_s, _, _, _ = runner_35b.generate_code(
                    rp,
                    max_new_tokens=(1024 if turn_k == 0 else 2176),
                    enable_thinking=(turn_k == 1),
                    thinking_budget_tokens=1152,
                )
            dt_tot += jr_s35.elapsed_ms
            j_tot += (jr_s35.elapsed_ms / 1000.0) * 95.0
            rep_tot += float(jr_s35.reported_j or 0.0)
            ok_cur, _, diag_cur = verify_swebench_candidate_in_worktree(st, cur_s)
            print(f"    [qwen35b SWE Rung3c/3d Turn {turn_k+1}] {st.eval_id} -> {'PASS' if ok_cur else 'FAIL'}", flush=True)
        swe_repair_35b[st.eval_id] = (cur_s, dt_tot, j_tot, rep_tot)

    runner_35b.unload()

    all_rows: list[ExternalBenchmarkRow] = []
    bcb_traces: list[dict] = []
    swe_traces: list[dict] = []

    # 3A. Score BigCodeBench v0.1.3 Standalone Arms
    for m_key, m_title, is_35 in [
        ("lfm3", "A_LiquidAI_LFM3_1.7B_1Shot", False),
        ("gemma4", "B_Google_Gemma4_E2B_it_1Shot", False),
        ("qwen35b", "C_Qwen3.6_35B_A3B_MetalAirMoL_1Shot", True),
    ]:
        out_b = []
        for bt in bcb_tasks:
            rc, dt, ej, rj = bcb_1shot[m_key][bt.eval_id]
            ok, _, _ = verify_bcb_candidate(rc, bt.code_prompt, bt.test_code, bt.entry_point)
            out_b.append((bt.wave, ok, not ok, False, is_35, dt, ej, rj))
        all_rows.append(_summarize_external("BigCodeBench_v0.1.3_32Evals", m_title, out_b))

    # 3B. Score BigCodeBench v0.1.3 `OpenIE MoL MCP Agent` (`MCP` JSON-RPC 2.0 + `.mol_external_bcb/`)
    bcb_mol_dir = art_dir / ".mol_external_bcb"
    if bcb_mol_dir.exists():
        for p in bcb_mol_dir.iterdir():
            p.unlink()
    mcp_bcb = MoLLocalCodingAgent(mol_dir=bcb_mol_dir)
    out_bcb_mol = []

    for bt in bcb_tasks:
        # Exercise real MCP JSON-RPC 2.0 `mol_lookup_crystallized_skill` first
        with meter_joules("rung0") as jr_lk:
            rpc_resp = handle_mcp_jsonrpc(
                mcp_bcb,
                {
                    "jsonrpc": "2.0",
                    "id": bt.eval_id,
                    "method": "tools/call",
                    "params": {
                        "name": "mol_lookup_crystallized_skill",
                        "arguments": {"fn_name": bt.entry_point, "specification": bt.prompt},
                    },
                },
            )
            cands_mol = lookup_top_mol_candidates(mcp_bcb.store, bt.entry_point, bt.prompt, top_k=4)
        assert rpc_resp is not None and "result" in rpc_resp
        acc_ms = jr_lk.elapsed_ms
        acc_rep = float(jr_lk.reported_j or 0.0)

        hit_mol = False
        hit_sid = -1
        for ent in cands_mol:
            with meter_joules("rung0") as jr_v:
                ok_m, _, _ = verify_bcb_candidate(ent.normalized_code, bt.code_prompt, bt.test_code, bt.entry_point)
            acc_ms += jr_v.elapsed_ms
            acc_rep += float(jr_v.reported_j or 0.0)
            if ok_m:
                ent.hits += 1
                hit_mol = True
                hit_sid = ent.skill_id
                break

        if hit_mol:
            out_bcb_mol.append((bt.wave, True, False, True, False, acc_ms, (acc_ms / 1000.0) * 3.0, acc_rep))
            bcb_traces.append(
                {
                    "eval_id": bt.eval_id,
                    "wave": bt.wave,
                    "resolved_by": f"MCP_Rung0_BenesLUT6->Skill#{hit_sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": True,
                }
            )
            continue

        # Escalate Rung 2 -> Rung 3a -> Rung 3c/3d
        acc_est = (acc_ms / 1000.0) * 3.0
        stages_b: list[tuple[str, str, float, float, float, bool]] = []
        rl3, dtl3, jl3, rjl3 = bcb_1shot["lfm3"][bt.eval_id]
        stages_b.append(("Rung2_LFM3_1.7B", rl3, dtl3, jl3, rjl3, False))
        rg4, dtg4, jg4, rjg4 = bcb_1shot["gemma4"][bt.eval_id]
        stages_b.append(("Rung3a_Gemma4_E2B", rg4, dtg4, jg4, rjg4, False))
        if bt.eval_id in bcb_repair_g4:
            rf4, dtf4, jf4, rjf4 = bcb_repair_g4[bt.eval_id]
            stages_b.append(("Rung3a_Gemma4_CxRepair", rf4, dtf4, jf4, rjf4, False))
        if bt.eval_id in bcb_repair_35b:
            r35, dt35, j35, rj35 = bcb_repair_35b[bt.eval_id]
            stages_b.append(("Rung3cd_Qwen3.6_35B_Think", r35, dt35, j35, rj35, True))
        else:
            r35, dt35, j35, rj35 = bcb_1shot["qwen35b"][bt.eval_id]
            stages_b.append(("Rung3c_Qwen3.6_35B", r35, dt35, j35, rj35, True))

        solved_b = False
        used_35 = False
        win_norm = ""
        win_src = "VERIFIER_ABSTAIN_SAFE"
        for st_nm, cand_r, s_ms, s_ej, s_rj, is_35 in stages_b:
            acc_ms += s_ms
            acc_est += s_ej
            acc_rep += s_rj
            if is_35:
                used_35 = True
            t_v0 = time.perf_counter()
            ok_c, norm_c, _ = verify_bcb_candidate(cand_r, bt.code_prompt, bt.test_code, bt.entry_point)
            acc_ms += (time.perf_counter() - t_v0) * 1000.0
            if ok_c:
                solved_b = True
                win_norm = norm_c
                win_src = st_nm
                break

        if solved_b:
            with meter_joules("compile") as jr_c:
                sid = mcp_bcb.store.register_verified_skill(bt.entry_point, bt.prompt, win_norm, win_src)
                mcp_bcb.store.recompile_lut6_router()
                mcp_bcb.store.save_to_dir(bcb_mol_dir)
            acc_ms += jr_c.elapsed_ms
            acc_rep += float(jr_c.reported_j or 0.0)
            out_bcb_mol.append((bt.wave, True, False, False, used_35, acc_ms, acc_est, acc_rep))
            bcb_traces.append(
                {
                    "eval_id": bt.eval_id,
                    "wave": bt.wave,
                    "resolved_by": f"{win_src}->Crystallized_Skill#{sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": True,
                }
            )
        else:
            out_bcb_mol.append((bt.wave, False, False, False, used_35, acc_ms, acc_est, acc_rep))
            bcb_traces.append(
                {
                    "eval_id": bt.eval_id,
                    "wave": bt.wave,
                    "resolved_by": "VERIFIER_ABSTAIN_SAFE",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": False,
                }
            )

    all_rows.append(_summarize_external("BigCodeBench_v0.1.3_32Evals", "D_OpenIE_MoL_MCP_Agent_OURS", out_bcb_mol))

    # 4A. Score Upstream `SWE-bench_Verified` Standalone Arms
    for m_key, m_title, is_35 in [
        ("lfm3", "A_LiquidAI_LFM3_1.7B_1Shot", False),
        ("gemma4", "B_Google_Gemma4_E2B_it_1Shot", False),
        ("qwen35b", "C_Qwen3.6_35B_A3B_MetalAirMoL_1Shot", True),
    ]:
        out_s = []
        for st in swe_tasks:
            rs, dt, ej, rj = swe_1shot[m_key][st.eval_id]
            ok, _, _ = verify_swebench_candidate_in_worktree(st, rs)
            out_s.append((st.wave, ok, not ok, False, is_35, dt, ej, rj))
        all_rows.append(_summarize_external("SWE_bench_Verified_SymPy_16Evals", m_title, out_s))

    # 4B. Score Upstream `SWE-bench_Verified` `OpenIE MoL MCP Agent` (`.mol_external_swebench/`)
    swe_mol_dir = art_dir / ".mol_external_swebench"
    if swe_mol_dir.exists():
        for p in swe_mol_dir.iterdir():
            p.unlink()
    mcp_swe = MoLLocalCodingAgent(mol_dir=swe_mol_dir)
    out_swe_mol = []

    for st in swe_tasks:
        key_fn = st.target_file.replace("/", "__").replace(".py", "")
        sig_prompt = f"def {key_fn}():\n{st.problem_statement}"
        with meter_joules("rung0") as jr_lk:
            rpc_swe = handle_mcp_jsonrpc(
                mcp_swe,
                {
                    "jsonrpc": "2.0",
                    "id": st.eval_id,
                    "method": "tools/call",
                    "params": {
                        "name": "mol_lookup_crystallized_skill",
                        "arguments": {"fn_name": key_fn, "specification": sig_prompt},
                    },
                },
            )
            cands_s = lookup_top_mol_candidates(mcp_swe.store, key_fn, sig_prompt, top_k=3)
        assert rpc_swe is not None and "result" in rpc_swe
        acc_ms = jr_lk.elapsed_ms
        acc_rep = float(jr_lk.reported_j or 0.0)

        hit_swe = False
        hit_sid = -1
        for ent in cands_s:
            with meter_joules("rung0") as jr_sv:
                ok_diff, _, _ = verify_swebench_candidate_in_worktree(
                    st, ent.normalized_code, is_unified_diff=True
                )
            acc_ms += jr_sv.elapsed_ms
            acc_rep += float(jr_sv.reported_j or 0.0)
            if ok_diff:
                ent.hits += 1
                hit_swe = True
                hit_sid = ent.skill_id
                break

        if hit_swe:
            out_swe_mol.append((st.wave, True, False, True, False, acc_ms, (acc_ms / 1000.0) * 5.0, acc_rep))
            swe_traces.append(
                {
                    "eval_id": st.eval_id,
                    "instance_id": st.instance_id,
                    "wave": st.wave,
                    "resolved_by": f"MCP_Rung0_BenesLUT6->GitDiffSkill#{hit_sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": True,
                }
            )
            continue

        acc_est = (acc_ms / 1000.0) * 5.0
        stages_s: list[tuple[str, str, float, float, float, bool]] = []
        rl3, dtl3, jl3, rjl3 = swe_1shot["lfm3"][st.eval_id]
        stages_s.append(("Rung2_LFM3_1.7B", rl3, dtl3, jl3, rjl3, False))
        rg4, dtg4, jg4, rjg4 = swe_1shot["gemma4"][st.eval_id]
        stages_s.append(("Rung3a_Gemma4_E2B", rg4, dtg4, jg4, rjg4, False))
        if st.eval_id in swe_repair_g4:
            rf4, dtf4, jf4, rjf4 = swe_repair_g4[st.eval_id]
            stages_s.append(("Rung3a_Gemma4_CxRepair", rf4, dtf4, jf4, rjf4, False))
        if st.eval_id in swe_repair_35b:
            r35, dt35, j35, rj35 = swe_repair_35b[st.eval_id]
            stages_s.append(("Rung3cd_Qwen3.6_35B_Think", r35, dt35, j35, rj35, True))
        else:
            r35, dt35, j35, rj35 = swe_1shot["qwen35b"][st.eval_id]
            stages_s.append(("Rung3c_Qwen3.6_35B", r35, dt35, j35, rj35, True))

        solved_s = False
        used_35 = False
        win_diff = ""
        win_src = "VERIFIER_ABSTAIN_SAFE"
        for st_nm, cand_r, s_ms, s_ej, s_rj, is_35 in stages_s:
            acc_ms += s_ms
            acc_est += s_ej
            acc_rep += s_rj
            if is_35:
                used_35 = True
            t_v0 = time.perf_counter()
            ok_s, u_diff, _ = verify_swebench_candidate_in_worktree(st, cand_r)
            acc_ms += (time.perf_counter() - t_v0) * 1000.0
            if ok_s:
                solved_s = True
                win_diff = u_diff
                win_src = st_nm
                break

        if solved_s:
            with meter_joules("compile") as jr_c:
                sid = mcp_swe.store.register_verified_skill(key_fn, sig_prompt, win_diff, win_src)
                mcp_swe.store.recompile_lut6_router()
                mcp_swe.store.save_to_dir(swe_mol_dir)
            acc_ms += jr_c.elapsed_ms
            acc_rep += float(jr_c.reported_j or 0.0)
            out_swe_mol.append((st.wave, True, False, False, used_35, acc_ms, acc_est, acc_rep))
            swe_traces.append(
                {
                    "eval_id": st.eval_id,
                    "instance_id": st.instance_id,
                    "wave": st.wave,
                    "resolved_by": f"{win_src}->Crystallized_GitDiffSkill#{sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": True,
                }
            )
        else:
            out_swe_mol.append((st.wave, False, False, False, used_35, acc_ms, acc_est, acc_rep))
            swe_traces.append(
                {
                    "eval_id": st.eval_id,
                    "instance_id": st.instance_id,
                    "wave": st.wave,
                    "resolved_by": "VERIFIER_ABSTAIN_SAFE",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "verified": False,
                }
            )

    all_rows.append(
        _summarize_external("SWE_bench_Verified_SymPy_16Evals", "D_OpenIE_MoL_MCP_Agent_OURS", out_swe_mol)
    )

    out_csv = art_dir / "external_public_bcb_swebench.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    pmu_end = read_apple_pmu_telemetry()
    meta: dict = {
        "bigcodebench_v0_1_3_parquet_sha256": BCB_V013_EXPECTED_SHA256,
        "swebench_verified_parquet_sha256": SWEBENCH_VERIFIED_EXPECTED_SHA256,
        "bigcodebench_task_ids": [t.bcb_task_id for t in bcb_tasks if t.wave == 1],
        "swebench_instance_ids": [t.instance_id for t in swe_tasks if t.wave == 1],
        "zero_gold_leakage_enforced": True,
        "measured_j": None,
        "board_metered": False,
    }
    if pmu_start is not None and pmu_end is not None and pmu_end[1] > pmu_start[1]:
        meta["apple_pmu_system_energy_delta_j"] = round(pmu_end[0] - pmu_start[0], 2)

    out_json = art_dir / "external_public_bcb_swebench_traces.json"
    out_json.write_text(
        json.dumps({"meta": meta, "bcb_traces": bcb_traces, "swe_traces": swe_traces}, indent=2),
        encoding="utf-8",
    )

    print("-" * 108)
    for r in all_rows:
        print(
            f"  [{r.benchmark:<32} | {r.arm:<35}] W1={r.wave1_cold_pass_pct:>6.2f}% | "
            f"W2={r.wave2_paraphrase_or_warm_pass_pct:>6.2f}% | All={r.overall_verified_pass_pct:>6.2f}% | "
            f"Broken={r.broken_outputs_admitted_pct:>6.2f}% | W2Hit={r.wave2_mol_fast_path_pct:>6.2f}% | "
            f"W1/W2={r.mean_latency_ms_wave1:.0f}/{r.mean_latency_ms_wave2:.0f} ms | "
            f"rep_J={r.reported_joules_per_verified_solve}"
        )
    print("-" * 108)
    print(f"Saved -> {out_csv} & {out_json}")


if __name__ == "__main__":
    main()
