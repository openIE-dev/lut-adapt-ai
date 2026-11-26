"""Phase 42 (`Claim LA-64`): Official Black-Box `ARC-AGI-3` Benchmark across BOTH:
  - **Matched `16`-Environment (`113`-Level) Longitudinal Panel** (apples-to-apples vs.
    `Random`, `FIFO-BFS`, `SIW(2)`, `BFWS(R,M)`, and `R-BFWS(R,M)`), AND
  - **Full `25`-Environment (`183`-Level) Official `ARC-AGI-3` Catalog** (`arc.get_environments()`),
evaluating Causal Modality-Pruned `C-BFWS(R, M)` + Multi-Epoch Rollout `RIW(2)` (`Bandres, Bonet
& Geffner, AAAI 2018`) and the Twelve-Bank `48 KB` (`12 x 4,096 B = 49,152 B = 75.00%` of `64 KB`
Watch `SRAM`, leaving `4 KB` `lut6-core` Nucleus + `12 KB` Stack) `System-0` Vault.

Strictly `100%` black-box (`0` `.py` game-source peeking, `0` raw frames persisted!).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from clai.contracts.joule_meter import meter_joules
from clai.frontier.arc3_bfws_four_bank import (
    BfwsFourBankVault,
    run_apex_bfws_arc3_agent,
)
from clai.frontier.arc3_cbfws_twelve_bank import (
    CbfwsTwelveBankVault,
    run_apex_cbfws_arc3_agent,
)
from clai.frontier.arc3_rbfws_eight_bank import (
    RbfwsEightBankVault,
    run_apex_rbfws_arc3_agent,
)
from clai.frontier.arc3_siw2_landmark_benes import (
    Siw2VaultBanks,
    run_apex_siw2_arc3_agent,
)
from clai.frontier.bonsai_stream_and_arc_agi_3 import run_arc3_agent_on_environment


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"

PINNED_16_SHORT_IDS: list[str] = [
    "tu93", "cd82", "ls20", "lp85", "vc33", "ft09", "sb26", "tr87",
    "cn04", "ka59", "su15", "sk48", "m0r0", "r11l", "sc25", "s5i5",
]


def _classify_modality(avail: list[int]) -> str:
    s = {int(a) for a in avail if int(a) != 0}
    if s == {6}:
        return "click_only"
    if 6 in s and not (s & {1, 2, 3, 4}):
        return "click_modal"
    if 6 in s:
        return "keyboard_click"
    return "keyboard"


def main() -> None:
    import arc_agi
    from arc_agi import OperationMode

    ART_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path.home() / ".cache" / "arc_agi_environments"
    cache_dir.mkdir(parents=True, exist_ok=True)

    arc_net = arc_agi.Arcade(environments_dir=str(cache_dir), operation_mode=OperationMode.NORMAL)
    raw_envs = arc_net.get_environments()
    catalog: list[tuple[str, str, list[int]]] = []
    seen_short: set[str] = set()
    for e in raw_envs:
        short = e.game_id.split("-")[0]
        if short in seen_short:
            continue
        seen_short.add(short)
        if not (cache_dir / short).is_dir():
            arc_net.make(short)
        b_acts = list(getattr(e, "baseline_actions", None) or [30] * 10)
        catalog.append((e.game_id, short, b_acts))

    by_short = {short: (full_gid, short, b_acts) for full_gid, short, b_acts in catalog}
    ordered_25: list[tuple[str, str, list[int]]] = []
    for s16 in PINNED_16_SHORT_IDS:
        if s16 in by_short:
            ordered_25.append(by_short[s16])
    for short in sorted(by_short):
        if short not in PINNED_16_SHORT_IDS:
            ordered_25.append(by_short[short])

    arc = arc_agi.Arcade(environments_dir=str(cache_dir), operation_mode=OperationMode.OFFLINE)

    modalities: dict[str, str] = {}
    total_levels_25 = 0
    total_levels_16 = 0
    for idx, (full_gid, short, _) in enumerate(ordered_25):
        env0 = arc.make(short)
        obs0 = env0.reset()
        av0 = [int(a) for a in obs0.available_actions if int(a) != 0]
        modalities[full_gid] = _classify_modality(av0)
        total_levels_25 += int(obs0.win_levels)
        if idx < 16:
            total_levels_16 += int(obs0.win_levels)

    bfs_crystals: dict[str, dict[int, int]] = {fg: {} for fg, _, _ in ordered_25}
    siw2_vaults: dict[str, Siw2VaultBanks] = {fg: Siw2VaultBanks(bank0={}, bank1={}) for fg, _, _ in ordered_25}
    bfws_vaults: dict[str, BfwsFourBankVault] = {fg: BfwsFourBankVault() for fg, _, _ in ordered_25}
    rbfws_vaults: dict[str, RbfwsEightBankVault] = {fg: RbfwsEightBankVault() for fg, _, _ in ordered_25}
    cbfws_vaults: dict[str, CbfwsTwelveBankVault] = {fg: CbfwsTwelveBankVault() for fg, _, _ in ordered_25}

    agent_specs = [
        ("random", "Random Action Baseline (Uniform Available Actions)", "legacy", 0),
        ("world_model_bfs_cold", "Phase 36 FIFO Frontier-BFS Wave 1 (58x64 HUD, No IW)", "bfs", 0),
        ("apex_siw2_lut6_wave2", "Phase 39 OpenIE-Apex SIW(2) -> Two-Bank 8 KB System-0 LUT6 Wave 2", "siw2_pair", 8192),
        ("apex_bfws_lut6_wave2", "Phase 40 OpenIE-Apex BFWS(R,M) -> Four-Bank 16 KB System-0 LUT6 Wave 2", "bfws_pair", 16384),
        ("apex_rbfws_cold", "Phase 41 OpenIE-Apex R-BFWS(R,M) + Palette-Fair Cellular Wave 1", "rbfws", 0),
        ("apex_rbfws_lut6_wave2", "Phase 41 OpenIE-Apex R-BFWS(R,M) -> Eight-Bank 32 KB System-0 LUT6 Wave 2", "rbfws", 32768),
        ("apex_cbfws_cold", "Phase 42 OpenIE-Apex C-BFWS(R,M) + Causal RIW(2) Wave 1 (Ours)", "cbfws", 0),
        ("apex_cbfws_lut6_wave2", "Phase 42 OpenIE-Apex C-BFWS(R,M) -> Twelve-Bank 48 KB System-0 LUT6 Wave 2 (Ours)", "cbfws", 49152),
    ]

    # Pre-populate SIW(2) and BFWS vaults cleanly so their Wave-2 rows are 100% live-measured
    for full_gid, short_gid, b_acts in ordered_25:
        run_apex_siw2_arc3_agent(arc, short_gid, b_acts, "apex_siw2_cold", 1100, siw2_vaults[full_gid])
        run_apex_bfws_arc3_agent(arc, short_gid, b_acts, "apex_bfws_cold", 1100, bfws_vaults[full_gid])

    summary_rows: list[dict] = []
    per_game_rows: list[dict] = []

    for akind, alabel, family, vault_bytes in agent_specs:
        with meter_joules() as jm:
            t0 = time.perf_counter()
            runs_all = []
            for idx_g, (full_gid, short_gid, b_acts) in enumerate(ordered_25):
                mod = modalities[full_gid]
                if family == "cbfws":
                    res = run_apex_cbfws_arc3_agent(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        vault_banks=cbfws_vaults[full_gid],
                    )
                elif family == "rbfws":
                    res = run_apex_rbfws_arc3_agent(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        vault_banks=rbfws_vaults[full_gid],
                    )
                elif family == "bfws_pair":
                    res = run_apex_bfws_arc3_agent(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        vault_banks=bfws_vaults[full_gid],
                    )
                elif family == "siw2_pair":
                    res = run_apex_siw2_arc3_agent(
                        arc=arc,
                        game_id=short_gid,
                        baseline_actions_per_level=b_acts,
                        agent_kind=akind,
                        max_actions_per_game=1100,
                        vault_banks=siw2_vaults[full_gid],
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
                runs_all.append((idx_g, full_gid, mod, res))
                per_game_rows.append(
                    {
                        "agent_kind": akind,
                        "system": alabel,
                        "game_id": full_gid,
                        "modality": mod,
                        "in_pinned16": idx_g < 16,
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

        runs_16 = [x for x in runs_all if x[0] < 16]
        lvls_16 = sum(r.levels_completed for _, _, _, r in runs_16)
        games_16 = sum(1 for _, _, _, r in runs_16 if r.levels_completed > 0)
        click_16 = sum(r.levels_completed for _, _, m, r in runs_16 if "click" in m)
        rhae_16 = sum(r.official_rhae_score for _, _, _, r in runs_16) / max(1, len(runs_16))
        peak_16 = max((r.official_rhae_score for _, _, _, r in runs_16), default=0.0)

        lvls_25 = sum(r.levels_completed for _, _, _, r in runs_all)
        games_25 = sum(1 for _, _, _, r in runs_all if r.levels_completed > 0)
        click_25 = sum(r.levels_completed for _, _, m, r in runs_all if "click" in m)
        rhae_25 = sum(r.official_rhae_score for _, _, _, r in runs_all) / max(1, len(runs_all))
        peak_25 = max((r.official_rhae_score for _, _, _, r in runs_all), default=0.0)
        tot_slots = sum(r.crystallized_lut6_slots for _, _, _, r in runs_all)
        tot_reflex = sum(r.lut6_reflex_steps for _, _, _, r in runs_all)
        tot_breakers = sum(r.breaker_trips for _, _, _, r in runs_all)

        row = {
            "agent_kind": akind,
            "system": alabel,
            "games_solved_16": games_16,
            "levels_completed_16": lvls_16,
            "total_levels_16": total_levels_16,
            "level_solve_pct_16": round(100.0 * lvls_16 / max(1, total_levels_16), 2),
            "click_levels_16": click_16,
            "mean_rhae_16": round(rhae_16, 4),
            "peak_rhae_16": round(peak_16, 4),
            "games_solved_25": games_25,
            "levels_completed_25": lvls_25,
            "total_levels_25": total_levels_25,
            "level_solve_pct_25": round(100.0 * lvls_25 / max(1, total_levels_25), 2),
            "click_levels_25": click_25,
            "mean_rhae_25": round(rhae_25, 4),
            "peak_rhae_25": round(peak_25, 4),
            "lut6_reflex_steps_25": tot_reflex,
            "crystallized_lut6_slots_25": tot_slots,
            "breaker_trips_25": tot_breakers,
            "vault_bytes": vault_bytes,
            "watch_sram_64kb_share_pct": round(100.0 * vault_bytes / 65536.0, 2),
            "wall_ms": round(wall_ms, 2),
            "reported_j": round(jm.reported_j, 4) if jm.reported_j is not None else None,
            "board_metered": False,
            "measured_j": None,
        }
        summary_rows.append(row)
        print(
            f"[{akind:22s}] 16-Env: {games_16:2d}/16 games, {lvls_16:2d}/{total_levels_16} lvls, RHAE={rhae_16:.4f} | "
            f"25-Env: {games_25:2d}/25 games, {lvls_25:2d}/{total_levels_25} lvls, RHAE={rhae_25:.4f} (peak={peak_25:.4f}, brk={tot_breakers})"
        )

    sum_csv = ART_DIR / "phase42_arc3_cbfws_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    pg_csv = ART_DIR / "phase42_arc3_cbfws_25env_per_game.csv"
    with pg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_game_rows[0].keys()))
        w.writeheader()
        w.writerows(per_game_rows)

    solved_cbfws = [
        f"{r['game_id']}({r['levels_completed']}/{r['total_levels']},RHAE={r['official_rhae_score']})"
        for r in per_game_rows
        if r["agent_kind"] == "apex_cbfws_lut6_wave2" and int(r["levels_completed"]) > 0
    ]
    meta = {
        "claim_id": "LA-64",
        "zero_game_source_peeking": True,
        "zero_raw_frames_persisted": True,
        "environments_evaluated_matched16": 16,
        "total_levels_matched16": total_levels_16,
        "environments_evaluated_full25": len(ordered_25),
        "total_levels_full25": total_levels_25,
        "cbfws_wave2_solved_games_detail": solved_cbfws,
        "summary": summary_rows,
        "board_metered": False,
        "measured_j": None,
    }
    (ART_DIR / "phase42_arc3_cbfws_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("Solved games (C-BFWS Wave 2):", ", ".join(solved_cbfws))


if __name__ == "__main__":
    main()
