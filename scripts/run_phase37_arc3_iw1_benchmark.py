"""Phase 37 Wall 1 (`Claim LA-45`): Expanded 16-Environment Official `ARC-AGI-3` Benchmark
comparing:
  1. `Random` (`arcprize` uniform available actions),
  2. `Memoryless Reactive` (1-step visual heuristic, no world model),
  3. `Phase 36 FIFO Frontier-BFS` (`58x64` HUD strip, blind `BFS` without `IW(1)` or click pruning),
  4. `OpenIE-Apex IW(1) Wave 1` (`62x64` palette-preserving HUD canonicalization + `Iterated Width IW(1)`
     novelty-guided search + causal inert-click pruning + `SleepCrystallizer`),
  5. `OpenIE-Apex IW(1) -> 4 KB System-0 LUT6 Wave 2` (`O(1)` zero-FLOP `.lut6` trajectory execution +
     `Theorem 73` mechanic-shift circuit breaker).

Strictly `100%` black-box (`0` `.py` game-source peeking, `0` raw frames persisted!).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from clai.contracts.joule_meter import meter_joules
from clai.frontier.bonsai_stream_and_arc_agi_3 import run_arc3_agent_on_environment
from clai.frontier.ecosystem_apex_harness import run_apex_iw1_arc3_agent


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"

# First 8 = exact Phase 36 matched slice; next 8 = expanded multi-modal & ACTION6 click games!
PINNED_ARC3_16_ENVIRONMENTS: list[tuple[str, str, list[int]]] = [
    ("tu93-0768757b", "keyboard", [19, 16, 34, 42, 123, 80, 14, 23, 111]),
    ("cd82-fb555c5d", "keyboard_click", [55, 8, 41, 21, 23, 23]),
    ("ls20-9607627b", "keyboard", [29, 41, 172, 49, 53, 62, 82]),
    ("lp85-305b61c3", "click_only", [17, 38, 31, 16, 41, 60, 26, 159]),
    ("vc33-5430563c", "click_only", [7, 18, 44, 61, 131, 34, 152]),
    ("ft09-0d8bbf25", "click_only", [43, 12, 23, 28, 65, 37]),
    ("sb26-7fbdac44", "keyboard_click", [26, 20, 48, 56, 48, 72, 66, 110]),
    ("tr87-cd924810", "keyboard", [54, 58, 40, 45, 71, 146]),
    # Expanded 8 official environments (`cn04`, `ka59`, `su15`, `sk48`, `m0r0`, `r11l`, `sc25`, `s5i5`)
    ("cn04-2fe56bfb", "keyboard_click", [24, 30, 36, 42, 50, 60]),
    ("ka59-38d34dbb", "keyboard_click", [20, 28, 35, 44, 52, 64, 72]),
    ("su15-1944f8ab", "click_action7", [18, 24, 32, 40, 48, 56, 64, 72, 80]),
    ("sk48-d8078629", "keyboard_click", [22, 30, 38, 46, 54, 62, 70, 80]),
    ("m0r0-492f87ba", "keyboard_click", [16, 24, 32, 40, 48, 56]),
    ("r11l-495a7899", "click_only", [14, 22, 30, 38, 46, 54]),
    ("sc25-635fd71a", "keyboard_click", [20, 28, 36, 44, 52, 60]),
    ("s5i5-18d95033", "click_only", [15, 24, 32, 40, 48, 56, 64, 72]),
]


def main() -> None:
    import arc_agi
    from arc_agi import OperationMode

    ART_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path.home() / ".cache" / "arc_agi_environments"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_cached = all((cache_dir / gid.split("-")[0]).is_dir() for gid, _, _ in PINNED_ARC3_16_ENVIRONMENTS)
    arc = arc_agi.Arcade(
        environments_dir=str(cache_dir),
        operation_mode=OperationMode.OFFLINE if all_cached else OperationMode.NORMAL,
    )

    env_info_map: dict[str, list[int]] = {}
    for e in arc.get_environments():
        if getattr(e, "baseline_actions", None):
            env_info_map[e.game_id.split("-")[0]] = list(e.baseline_actions)

    bfs_crystals: dict[str, dict[int, int]] = {gid: {} for gid, _, _ in PINNED_ARC3_16_ENVIRONMENTS}
    iw1_crystals: dict[str, dict[int, int]] = {gid: {} for gid, _, _ in PINNED_ARC3_16_ENVIRONMENTS}

    agent_specs = [
        ("random", "Random Action Baseline (Uniform Available Actions)", "legacy"),
        ("reactive_memoryless", "Memoryless Reactive Micro-Policy (No World Model)", "legacy"),
        ("world_model_bfs_cold", "Phase 36 FIFO Frontier-BFS Wave 1 (58x64 HUD, No IW(1))", "bfs"),
        ("apex_iw1_cold", "OpenIE-Apex IW(1) + Causal Click Pruning Wave 1 (62x64 HUD, C_env==1)", "iw1"),
        ("apex_iw1_lut6_wave2", "OpenIE-Apex Compiled 4 KB System-0 LUT6 Wave 2 (IW(1) Crystal + Breaker)", "iw1"),
    ]

    summary_rows: list[dict] = []
    per_game_rows: list[dict] = []

    for akind, alabel, family in agent_specs:
        with meter_joules() as jm:
            t0 = time.perf_counter()
            runs = []
            for idx_g, (full_gid, modality, fb_base) in enumerate(PINNED_ARC3_16_ENVIRONMENTS):
                short_gid = full_gid.split("-")[0]
                b_acts = env_info_map.get(short_gid, fb_base)
                if family == "iw1":
                    res = run_apex_iw1_arc3_agent(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        lut6_crystal=iw1_crystals[full_gid],
                    )
                else:
                    crys = bfs_crystals[full_gid] if family == "bfs" else None
                    res = run_arc3_agent_on_environment(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        lut6_crystal=crys,
                    )
                runs.append((idx_g, full_gid, modality, res))
                per_game_rows.append(
                    {
                        "agent_kind": akind,
                        "system": alabel,
                        "game_id": full_gid,
                        "modality": modality,
                        "in_matched8": idx_g < 8,
                        "total_levels": res.total_levels,
                        "levels_completed": res.levels_completed,
                        "total_actions": res.total_actions,
                        "official_rhae_score": res.official_rhae_score,
                        "lut6_reflex_steps": res.lut6_reflex_steps,
                        "crystallized_lut6_slots": res.crystallized_lut6_slots,
                        "breaker_trips": res.breaker_trips,
                        "elapsed_ms": res.elapsed_ms,
                    }
                )
            wall_ms = (time.perf_counter() - t0) * 1000.0

        tot_lvls = sum(r.total_levels for _, _, _, r in runs)
        comp_lvls = sum(r.levels_completed for _, _, _, r in runs)
        m8_lvls = sum(r.total_levels for i, _, _, r in runs if i < 8)
        m8_comp = sum(r.levels_completed for i, _, _, r in runs if i < 8)
        click_comp = sum(r.levels_completed for _, _, mod, r in runs if "click" in mod)
        tot_acts = sum(r.total_actions for _, _, _, r in runs)
        mean_rhae = float(np.mean([r.official_rhae_score for _, _, _, r in runs]))
        peak_rhae = float(np.max([r.official_rhae_score for _, _, _, r in runs]))
        lut_steps = sum(r.lut6_reflex_steps for _, _, _, r in runs)
        crys_slots = sum(r.crystallized_lut6_slots for _, _, _, r in runs)
        brk_trips = sum(r.breaker_trips for _, _, _, r in runs)
        rep_j = float(jm.reported_j or 0.0)

        summary_rows.append(
            {
                "system": alabel,
                "environments_16": len(runs),
                "total_levels_16": tot_lvls,
                "levels_completed_16": comp_lvls,
                "level_completion_16_pct": round(100.0 * comp_lvls / max(tot_lvls, 1), 2),
                "matched8_levels_completed": f"{m8_comp}/{m8_lvls}",
                "click_modal_levels_completed": click_comp,
                "mean_official_rhae_16": round(mean_rhae, 4),
                "peak_game_rhae_16": round(peak_rhae, 4),
                "total_actions_taken": tot_acts,
                "lut6_reflex_steps": lut_steps,
                "lut6_reflex_share_pct": round(100.0 * lut_steps / max(tot_acts, 1), 2),
                "crystallized_lut6_slots": crys_slots,
                "mechanic_shift_breaker_trips": brk_trips,
                "active_policy_crystal_bytes": 4096 if "Wave 2" in alabel else 0,
                "total_wall_ms": round(wall_ms, 2),
                "reported_j_total": round(rep_j, 4),
                "board_metered": False,
                "measured_j": None,
            }
        )
        print(f"[{akind}] 16-env solved={comp_lvls}/{tot_lvls} (matched8={m8_comp}/{m8_lvls}, click={click_comp}) | mean_RHAE={mean_rhae:.4f} | peak_RHAE={peak_rhae:.4f}", flush=True)

    with (ART_DIR / "phase37_arc3_16env_iw1_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with (ART_DIR / "phase37_arc3_16env_per_game.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_game_rows[0].keys()))
        w.writeheader()
        w.writerows(per_game_rows)
    (ART_DIR / "phase37_arc3_16env_meta.json").write_text(
        json.dumps(
            {
                "benchmark": "ARC-AGI-3 Official 16-Environment Interactive Suite (arcprize, MIT)",
                "zero_game_source_peeking": True,
                "zero_raw_frames_persisted": True,
                "environments_count": 16,
                "summary_rows": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
