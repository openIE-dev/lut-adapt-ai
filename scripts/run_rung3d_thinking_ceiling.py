"""Phase 27 (Claim LA-17): Open-Weights `Rung 3d` Bounded-`<think>` Counterexample Escalation
Ceiling on the 48-Task (`16` Specs x `3` Waves) Late-2026 Frontier Coding Benchmark.

Evaluates adding `Rung 3d` (`Qwen/Qwen3.6-35B-A3B` Metal `AirMoL` with native `<think>...</think>`
test-time reasoning + counterexample feedback, 100% Apache-2.0 open-weights, zero cloud APIs)
strictly on the residual `12.50%` (`2 / 16`) tasks where greedy non-thinking `Rung 3c` abstained
(`F10_true_damerau` and `F11_benes4_route`), then verifies downward crystallization into `.mol/`
across all 48 tasks (`Waves 1, 2, 3`), confirming whether a single `<think>` wake per hard
combinatorial spec eliminates repeated thinking cost on Waves 2 and 3.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from clai.contracts.joule_meter import meter_joules, select_torch_device
from clai.frontier.verified_code_cascade import (
    BakeoffSummaryRow,
    BenesLUT6SkillCascadeStore,
    ExpertAirMoL35BRunner,
    _build_repair_prompt,
    build_3wave_frontier_benchmark,
    verify_python_candidate,
)


def main() -> None:
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)
    dev = select_torch_device()

    print("=" * 105)
    print(" PHASE 27 (CLAIM LA-17): OPEN-WEIGHTS RUNG 3d (`Qwen3.6-35B-A3B <think>`) ESCALATION CEILING (48 TASKS)")
    print("=" * 105)

    tasks = build_3wave_frontier_benchmark()
    base_traces_obj = json.loads((art_dir / "frontier_ai_head_to_head_traces.json").read_text(encoding="utf-8"))
    base_traces = {tr["task_id"]: tr for tr in base_traces_obj["mol_traces"]}
    base_skills_obj = json.loads((art_dir / ".mol_verified_skills" / "skills_v1.json").read_text(encoding="utf-8"))
    # Map (fn_name, source_rung_skill_id) from persisted verified skills
    persisted_by_id = {int(s["skill_id"]): s for s in base_skills_obj["skills"]}

    # Instantiate live `ExpertAirMoL35BRunner` ONLY when `Rung 3d` is actually needed!
    runner_35b: ExpertAirMoL35BRunner | None = None
    rung3d_wakes = 0
    spec_drift_caught = 0

    skill_store = BenesLUT6SkillCascadeStore()
    outcomes_r3d: list[tuple[int, bool, bool, bool, bool, float, float, float]] = []
    per_task_traces: list[dict] = []

    for t in tasks:
        with meter_joules("rung0") as jr_r0:
            cand_skill = skill_store.lookup_rung0_rung1(t.fn_name, t.prompt)
        dt_r0_ms = jr_r0.elapsed_ms
        rep_r0 = float(jr_r0.reported_j or 0.0)

        if cand_skill is not None:
            with meter_joules("rung0") as jr_v:
                ok_s, _, _ = verify_python_candidate(cand_skill.normalized_code, t.fn_name, t.tests)
            if ok_s:
                cand_skill.hits += 1
                tot_ms = dt_r0_ms + jr_v.elapsed_ms
                est_j = (tot_ms / 1000.0) * 3.0
                rep_tot = rep_r0 + float(jr_v.reported_j or 0.0)
                outcomes_r3d.append((t.wave, True, False, True, False, tot_ms, est_j, rep_tot))
                per_task_traces.append(
                    {
                        "task_id": t.task_id,
                        "wave": t.wave,
                        "resolved_by": f"Rung0_BenesLUT6->Rung1_Skill#{cand_skill.skill_id}",
                        "latency_ms": round(tot_ms, 3),
                        "reported_j": round(rep_tot, 5),
                        "verified": True,
                        "used_rung3d_thinking": False,
                    }
                )
                continue
            if t.spec_mutated:
                spec_drift_caught += 1

        # Miss or Wave-2 adversarial spec drift -> run Rungs 2..3c (reusing verified Wave-1/2 Rung 2..3c trace
        # cost & code where Rung 2..3c already succeeded, and escalating abstentions live to Rung 3d `<think>`!)
        b_tr = base_traces[t.task_id]
        b_res = str(b_tr["resolved_by"])
        if b_tr["verified"] and "Crystallized_Skill#" in b_res:
            sid_orig = int(b_res.split("Crystallized_Skill#")[1])
            orig_skill = persisted_by_id[sid_orig]
            ok_chk, norm_c, _ = verify_python_candidate(orig_skill["normalized_code"], t.fn_name, t.tests)
            assert ok_chk, f"Persisted skill #{sid_orig} failed verification on {t.task_id}!"
            st_src = b_res.split("->Crystallized_Skill#")[0]
            used_35 = "35B" in st_src
            acc_ms = float(b_tr["latency_ms"])
            acc_rep_j = float(b_tr["reported_j"])
            acc_j = (acc_ms / 1000.0) * (95.0 if used_35 else 38.0)

            with meter_joules("compile") as jr_c:
                new_sid = skill_store.register_verified_skill(t.fn_name, t.prompt, norm_c, st_src)
                skill_store.recompile_lut6_router()
            acc_ms += jr_c.elapsed_ms
            acc_rep_j += float(jr_c.reported_j or 0.0)
            outcomes_r3d.append((t.wave, True, False, False, used_35, acc_ms, acc_j, acc_rep_j))
            per_task_traces.append(
                {
                    "task_id": t.task_id,
                    "wave": t.wave,
                    "resolved_by": f"{st_src}->Crystallized_Skill#{new_sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep_j, 4),
                    "verified": True,
                    "used_rung3d_thinking": False,
                }
            )
            continue

        # Residual `VERIFIER_ABSTAIN_SAFE` from Rungs 2..3c -> Escalate LIVE to Rung 3d (`Qwen3.6-35B-A3B <think>`)!
        if runner_35b is None:
            print("  [Rung 3d] Loading Metal-Fused Qwen/Qwen3.6-35B-A3B for native `<think>` escalation...", flush=True)
            runner_35b = ExpertAirMoL35BRunner(device=dev)

        acc_ms = float(b_tr["latency_ms"])
        acc_rep_j = float(b_tr["reported_j"])
        acc_j = (acc_ms / 1000.0) * 95.0
        rung3d_wakes += 1

        solved_3d = False
        winning_norm = ""
        cur_prompt = t.prompt
        last_cand = ""
        last_diag = ""
        think_toks_total = 0

        for turn_idx in range(1, 4):
            p_use = cur_prompt if turn_idx == 1 else _build_repair_prompt(t.prompt, last_cand, last_diag)
            with meter_joules("rung3", device=dev) as jr_3d:
                cand_3d, n_toks, nvme_gb, _ = runner_35b.generate_code(
                    p_use,
                    max_new_tokens=2200,
                    enable_thinking=True,
                    thinking_budget_tokens=1536,
                )
            dt_3d = jr_3d.elapsed_ms
            rep_3d = float(jr_3d.reported_j or 0.0)
            acc_ms += dt_3d
            acc_rep_j += rep_3d
            acc_j += (dt_3d / 1000.0) * 95.0
            think_toks_total += n_toks

            t_v = time.perf_counter()
            ok_3d, norm_3d, diag_3d = verify_python_candidate(cand_3d, t.fn_name, t.tests)
            acc_ms += (time.perf_counter() - t_v) * 1000.0
            status_str = "PASS" if ok_3d else f"FAIL({diag_3d[:60]})"
            print(
                f"    [Rung 3d <think> Turn {turn_idx}] {t.task_id}: {n_toks} toks in {dt_3d:.0f} ms "
                f"(rep_J={rep_3d:.2f}) -> {status_str}",
                flush=True,
            )
            if ok_3d:
                solved_3d = True
                winning_norm = norm_3d
                break
            last_cand = cand_3d
            last_diag = diag_3d

        if solved_3d:
            with meter_joules("compile") as jr_c:
                new_sid = skill_store.register_verified_skill(
                    t.fn_name, t.prompt, winning_norm, "Rung3d_Qwen3.6_35B_A3B_NativeThink"
                )
                skill_store.recompile_lut6_router()
            acc_ms += jr_c.elapsed_ms
            acc_rep_j += float(jr_c.reported_j or 0.0)
            outcomes_r3d.append((t.wave, True, False, False, True, acc_ms, acc_j, acc_rep_j))
            per_task_traces.append(
                {
                    "task_id": t.task_id,
                    "wave": t.wave,
                    "resolved_by": f"Rung3d_Qwen3.6_35B_A3B_NativeThink->Crystallized_Skill#{new_sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep_j, 4),
                    "thinking_tokens": think_toks_total,
                    "verified": True,
                    "used_rung3d_thinking": True,
                }
            )
        else:
            outcomes_r3d.append((t.wave, False, False, False, True, acc_ms, acc_j, acc_rep_j))
            per_task_traces.append(
                {
                    "task_id": t.task_id,
                    "wave": t.wave,
                    "resolved_by": "VERIFIER_ABSTAIN_SAFE",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep_j, 4),
                    "thinking_tokens": think_toks_total,
                    "verified": False,
                    "used_rung3d_thinking": True,
                }
            )

    if runner_35b is not None:
        runner_35b.unload()

    persist_dir = art_dir / ".mol_verified_skills_r3d"
    skill_store.save_to_dir(persist_dir)

    w1 = [x for x in outcomes_r3d if x[0] == 1]
    w2 = [x for x in outcomes_r3d if x[0] == 2]
    w3 = [x for x in outcomes_r3d if x[0] == 3]
    tot_solved = sum(1 for x in outcomes_r3d if x[1])
    tot_j = sum(x[6] for x in outcomes_r3d)
    tot_rep_j = sum(x[7] for x in outcomes_r3d)

    row_r3d = BakeoffSummaryRow(
        arm="H_OpenIE_MoL_Plus_Rung3d_Qwen3.6_35B_NativeThink_OURS",
        wave1_cold_pass_pct=round(100.0 * sum(1 for x in w1 if x[1]) / len(w1), 2),
        wave2_adapt_pass_pct=round(100.0 * sum(1 for x in w2 if x[1]) / len(w2), 2),
        wave3_warm_pass_pct=round(100.0 * sum(1 for x in w3 if x[1]) / len(w3), 2),
        overall_verified_pass_pct=round(100.0 * tot_solved / len(outcomes_r3d), 2),
        broken_commits_admitted_pct=round(100.0 * sum(1 for x in outcomes_r3d if x[2]) / len(outcomes_r3d), 2),
        rung0_rung1_hit_pct_wave3=round(100.0 * sum(1 for x in w3 if x[3]) / len(w3), 2),
        frontier_35b_calls_pct=round(100.0 * sum(1 for x in outcomes_r3d if x[4]) / len(outcomes_r3d), 2),
        mean_latency_ms_wave1=round(float(np.mean([x[5] for x in w1])), 2),
        mean_latency_ms_wave3=round(float(np.mean([x[5] for x in w3])), 2),
        overall_mean_latency_ms=round(float(np.mean([x[5] for x in outcomes_r3d])), 2),
        est_joules_per_verified_solve=round(tot_j / max(1, tot_solved), 2),
        reported_joules_per_verified_solve=round(tot_rep_j / max(1, tot_solved), 3),
    )

    # Load existing 7 rows from `frontier_ai_head_to_head_bakeoff.csv` for direct comparison table
    existing_rows: list[dict] = []
    with (art_dir / "frontier_ai_head_to_head_bakeoff.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            existing_rows.append(r)
    existing_rows.append(asdict(row_r3d))

    out_csv = art_dir / "frontier_ai_rung3d_ceiling.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(row_r3d).keys()))
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)

    w3_solved_r0 = [tr for tr in per_task_traces if tr["wave"] == 3 and tr["resolved_by"].startswith("Rung0_")]
    w3_mean_rep_j = float(np.mean([tr["reported_j"] for tr in w3_solved_r0])) if w3_solved_r0 else 0.0
    w3_mean_ms = float(np.mean([tr["latency_ms"] for tr in w3_solved_r0])) if w3_solved_r0 else 0.0

    out_json = art_dir / "frontier_ai_rung3d_ceiling_traces.json"
    out_json.write_text(
        json.dumps(
            {
                "meta": {
                    "rung3d_model": "Qwen/Qwen3.6-35B-A3B (enable_thinking=True, bounded 1536/2200 tokens)",
                    "rung3d_invocations_across_48_tasks": rung3d_wakes,
                    "wave2_spec_mutations_caught": spec_drift_caught,
                    "total_crystallized_skills": len(skill_store.skills),
                    "wave3_rung0_1_mean_latency_ms": round(w3_mean_ms, 3),
                    "wave3_rung0_1_mean_reported_j": round(w3_mean_rep_j, 5),
                    "measured_j": None,
                    "board_metered": False,
                },
                "summary_row": asdict(row_r3d),
                "traces": per_task_traces,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 105)
    print(
        f"  [Arm H Result] W1={row_r3d.wave1_cold_pass_pct:.2f}% | W2={row_r3d.wave2_adapt_pass_pct:.2f}% | "
        f"W3={row_r3d.wave3_warm_pass_pct:.2f}% | Overall={row_r3d.overall_verified_pass_pct:.2f}% | "
        f"Broken={row_r3d.broken_commits_admitted_pct:.2f}% | R3dWakes={rung3d_wakes}/48 | "
        f"W3Mean={row_r3d.mean_latency_ms_wave3:.2f} ms ({w3_mean_rep_j:.4f} J reported_j)"
    )
    print(f"Saved -> {out_csv} & {out_json}")


if __name__ == "__main__":
    main()
