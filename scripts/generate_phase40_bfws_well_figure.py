"""Generate publication-grade vector SVG + 2480px Retina PNG for Phase 40 (`Figure 16`):
`paper/figures/bfws-arc3-well-and-multimodal.{svg,png}`
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


FONT_SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts"
    arc3_rows = list(csv.DictReader((art / "phase40_arc3_bfws_summary.csv").open(encoding="utf-8")))
    arc3_games = list(csv.DictReader((art / "phase40_arc3_bfws_25env_per_game.csv").open(encoding="utf-8")))
    mm_rows = list(csv.DictReader((art / "phase40_swebench_multimodal_pro.csv").open(encoding="utf-8")))
    mm_meta = json.loads((art / "phase40_multimodal_and_well_meta.json").read_text(encoding="utf-8"))
    w20_rows = list(csv.DictReader((art / "phase40_window20_cross_silicon.csv").open(encoding="utf-8")))
    xc7 = json.loads((art / "openxc7_bfws4_well_timing.json").read_text(encoding="utf-8"))
    lean = json.loads((art / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet = json.loads((art / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    wasm_bytes = (art / "wasm" / "lut6_core.wasm").stat().st_size
    wasm_kb = round(wasm_bytes / 1024.0, 2)

    n_thms = int(lean["total_kernel_checked_theorems"])
    n_fleet = int(fleet["total_theorems_and_lemmas"])
    thor_w20 = next(r for r in w20_rows if r["board_metered"] == "True")
    thor_nj = float(thor_w20["w20_measured_nj_per_op_vdd_cpu"])
    thor_gain = float(thor_w20["energy_reduction_x"])
    bfws_w2 = next(r for r in arc3_rows if r["agent_kind"] == "apex_bfws_lut6_wave2")

    W, H = 1240, 900
    sub_hdr = (
        f"Claims LA-19 ({n_thms} Thms / {n_fleet:,} bwrap Fleet) & LA-56..LA-59  |  "
        f"25-Env ARC-AGI-3 BFWS = {bfws_w2['levels_completed_25']}/{bfws_w2['total_levels_25']} Lvls ({bfws_w2['games_solved_25']}/25 Games)  |  "
        f"Thor Window 20 = {thor_nj:.0f} nJ ({thor_gain:.2f}x)  |  {wasm_kb:.2f} KB 15-Export WASM"
    )

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p40-t p40-d">',
        '  <title id="p40-t">Dual-Subspace BFWS on Full 25-Game ARC-AGI-3, SWE-bench Multimodal+Pro, Planetary Digital Well, and 106 Six-Kernel Proofs</title>',
        f'  <desc id="p40-d">Four-panel summary of Claims LA-56 through LA-59 and {n_thms} six-kernel Lean 4.36.0 theorems ({n_fleet} Linux bwrap fleet): (A) Official 16-game and full 25-game (183-level) black-box ARC-AGI-3 Dual-Subspace BFWS + Four-Bank 16 KB System-0 LUT6 vault; (B) 24-task real Git-worktree SWE-bench Multimodal and Pro visual-diagram + multi-file repair with 0.00% broken commits; (C) Planetary Digital Well 12-stdlib-module Rung-5 to Rung-7 projection (384/384 exact parity, Yosys abc -lut 6 miter) and Cross-Silicon Window 20 Thor DC shunt telemetry ({thor_gain:.2f}x); and (D) Zero-DSP48E1 {xc7["post_par_max_freq_mhz"]:.2f} MHz XC7A200T openXC7 gateware, {wasm_kb:.2f} KB 15-export lut6_core.wasm, and {n_thms} six-kernel theorems.</desc>',
        "  <defs>",
        '    <linearGradient id="bg40" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#040814"/><stop offset="52%" stop-color="#091628"/><stop offset="100%" stop-color="#041b1e"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{W}" height="{H}" fill="url(#bg40)"/>',
        f'  <text x="28" y="34" fill="#f8fafc" font-family="{FONT_SANS}" font-size="16.4" font-weight="800">25-Game ARC-AGI-3 BFWS (16 KB Vault), SWE-bench Multimodal+Pro, Digital Well &amp; 106 Proofs</text>',
        f'  <text x="28" y="54" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.7">{esc(sub_hdr)}</text>',
    ]

    # PANEL A: Claim LA-56 — 16-Game & Full 25-Game Official ARC-AGI-3 BFWS + 4-Bank 16 KB Vault
    p += [
        '  <rect x="24" y="68" width="584" height="398" rx="12" fill="#0d1729" stroke="#22d3ee" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="92" fill="#22d3ee" font-family="{FONT_SANS}" font-size="12.5" font-weight="800">A. Claim LA-56 · 16-Env &amp; Full 25-Env (183-Lvl) ARC-AGI-3 BFWS + 16 KB Vault</text>',
        f'  <text x="42" y="109" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.0">Theorems 99–101 · Dual-Subspace Key Prelude + UNDO Demotion + 4-Bank 16 KB (25.0% Watch SRAM)</text>',
        '  <rect x="42" y="118" width="548" height="20" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">Black-Box Agent (0 .py Peeking)</text>',
        f'  <text x="288" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">16-Env Lvls</text>',
        f'  <text x="364" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">16 RHAE</text>',
        f'  <text x="432" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">25-Env Lvls</text>',
        f'  <text x="514" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">25 RHAE</text>',
    ]
    for idx, r in enumerate(arc3_rows):
        ry = 144 + idx * 41
        s = r["system"]
        is_bfws_w2 = r["agent_kind"] == "apex_bfws_lut6_wave2"
        is_bfws_w1 = r["agent_kind"] == "apex_bfws_cold"
        bg = "#062c24" if is_bfws_w2 else ("#0b243b" if is_bfws_w1 else "#131c2e")
        tc = "#6ee7b7" if is_bfws_w2 else ("#7dd3fc" if is_bfws_w1 else "#cbd5e1")
        lbl = (
            s.replace("Phase 40 OpenIE-Apex ", "Apex ")
            .replace("Phase 39 OpenIE-Apex ", "Phase 39 ")
            .replace("Phase 36 ", "")
            .replace(" (Ours)", "")
        )[:36]
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="35" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="48" y="{ry+21}" fill="{tc}" font-family="{FONT_SANS}" font-size="8.6" font-weight="{"700" if (is_bfws_w1 or is_bfws_w2) else "500"}">{esc(lbl)}</text>',
            f'  <text x="290" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{r["levels_completed_16"]}/113 ({r["games_solved_16"]}g)</text>',
            f'  <text x="366" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{float(r["mean_rhae_16"]):.4f}</text>',
            f'  <text x="434" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{r["levels_completed_25"]}/183 ({r["games_solved_25"]}g)</text>',
            f'  <text x="516" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{float(r["mean_rhae_25"]):.4f}</text>',
        ]
    solved_g = [g for g in arc3_games if g["agent_kind"] == "apex_bfws_lut6_wave2" and int(g["levels_completed"]) > 0]
    g_line1 = " · ".join(f'{g["game_id"].split("-")[0]}:{g["levels_completed"]}' for g in solved_g[:7])
    g_line2 = " · ".join(f'{g["game_id"].split("-")[0]}:{g["levels_completed"]}' for g in solved_g[7:])
    p += [
        '  <rect x="42" y="394" width="548" height="58" rx="6" fill="#072922" stroke="#10b981" stroke-width="0.9"/>',
        f'  <text x="52" y="413" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">{len(solved_g)}/25 Solved Games (Peak RHAE = {bfws_w2["peak_rhae_25"]}, 0 Breaker Trips, 16,384 B Vault):</text>',
        f'  <text x="52" y="430" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="7.9">{esc(g_line1)}</text>',
        f'  <text x="52" y="445" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="7.9">{esc(g_line2)}</text>',
    ]

    # PANEL B: Claim LA-57 — SWE-bench Multimodal & Pro (24 Git-Worktree Tasks)
    p += [
        '  <rect x="632" y="68" width="584" height="398" rx="12" fill="#0d1729" stroke="#10b981" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="650" y="92" fill="#34d399" font-family="{FONT_SANS}" font-size="12.5" font-weight="800">B. Claim LA-57 · SWE-bench Multimodal &amp; Pro (24 Real Git-Worktree Tasks)</text>',
        f'  <text x="650" y="109" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.0">Theorem 102 · SVG viewBox + Flex-Grid + HiDPI DPR + Mermaid DAG + PCB DRC + Saga</text>',
        '  <rect x="650" y="118" width="548" height="20" rx="4" fill="#1e293b"/>',
        f'  <text x="656" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">Repository Repair Architecture</text>',
        f'  <text x="916" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">Pytest</text>',
        f'  <text x="978" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">MM Verif</text>',
        f'  <text x="1050" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">Broken</text>',
        f'  <text x="1118" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0" font-weight="700">mJ/Task</text>',
    ]
    for idx, r in enumerate(mm_rows):
        ry = 144 + idx * 41
        sid = r["system_id"]
        is_apex = "openie_apex" in sid
        is_null = sid == "visual_symbolic_null"
        bg = "#062c24" if is_apex else ("#0b243b" if is_null else "#231420")
        tc = "#6ee7b7" if is_apex else ("#7dd3fc" if is_null else "#fda4af")
        short_lbl = (
            r["system"]
            .replace("OpenIE-Apex Multimodal-Pro ", "Apex MM-Pro ")
            .replace("Visual-Symbolic Deterministic Null ", "Visual-Symbolic Null ")
            .replace("prism-ml/", "")
            .replace("thinkingmachines/", "")
        )[:37]
        p += [
            f'  <rect x="650" y="{ry}" width="548" height="35" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="656" y="{ry+21}" fill="{tc}" font-family="{FONT_SANS}" font-size="8.5" font-weight="{"700" if (is_apex or is_null) else "500"}">{esc(short_lbl)}</text>',
            f'  <text x="916" y="{ry+21}" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4">{float(r["unit_test_only_pass_pct"]):.0f}%</text>',
            f'  <text x="982" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{float(r["multimodal_verified_pct"]):.1f}%</text>',
            f'  <text x="1052" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">{float(r["broken_visual_or_code_commit_pct"]):.1f}%</text>',
            f'  <text x="1116" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.3">{float(r["mean_energy_mj_per_task"]):.4f}</text>',
        ]
    p += [
        '  <rect x="650" y="394" width="548" height="58" rx="6" fill="#072922" stroke="#10b981" stroke-width="0.9"/>',
        f'  <text x="660" y="414" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Conjunctive Gate C_visual &amp;&amp; C_pytest Blocks 100% of Silent Visual Clipping/Overlap:</text>',
        f'  <text x="660" y="432" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.0">• Visual-Symbolic Null solves 66.7% (16/24) at 0.0018 mJ/task (0 LLM FLOPs, >1.6M x &lt; 27B)</text>',
        f'  <text x="660" y="446" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0">• Wave-2 4 KB AST-Patch-LUT6 hits 100.0% (12/12) at 0.0004 mJ/task (7.38M x &lt; unverified 27B)</text>',
    ]

    # PANEL C: Claim LA-58 — Planetary Digital Well (12 Stdlib Modules) & Window 20 Shunt
    well = mm_meta["planetary_well_codifier"]
    p += [
        '  <rect x="24" y="484" width="584" height="392" rx="12" fill="#0d1729" stroke="#f59e0b" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="509" fill="#fbbf24" font-family="{FONT_SANS}" font-size="12.5" font-weight="800">C. Claim LA-58 · Planetary Digital Well (12 Stdlib) &amp; Window 20 DC Shunt</text>',
        f'  <text x="42" y="526" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.0">Theorems 103–104 · 384/384 (100%) Stdlib Parity · Yosys abc -lut 6 Miter ({well["yosys_miter_cones"]} LUTs)</text>',
        '  <rect x="42" y="536" width="548" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Hardware Host &amp; Telemetry Rail</text>',
        f'  <text x="292" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">W20 (Ours)</text>',
        f'  <text x="378" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">W20b (R5)</text>',
        f'  <text x="468" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Speedup</text>',
        f'  <text x="530" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">J Cut</text>',
    ]
    for idx, wr in enumerate(w20_rows):
        ry = 565 + idx * 72
        bm = wr["board_metered"] == "True"
        bg = "#072922" if bm else "#121f36"
        rail = "Physical TI INA3221 VDD_CPU_SOC_MSS Shunt (measured_j, board_metered=true)" if bm else (
            "Linux powercap amd_rapl Package Counter (reported_j, board_metered=false)"
            if "Zen5" in wr["host_label"]
            else "Darwin RUSAGE_INFO_V6 ri_energy_nj Counter (reported_j, board_metered=false)"
        )
        e20 = float(wr["w20_measured_nj_per_op_vdd_cpu"] if bm else wr["w20_reported_nj_per_op"])
        e20b = float(wr["w20b_measured_nj_per_op_vdd_cpu"] if bm else wr["w20b_reported_nj_per_op"])
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="62" rx="6" fill="{bg}" stroke="{"#10b981" if bm else "#334155"}" stroke-width="1.0"/>',
            f'  <text x="50" y="{ry+21}" fill="#f8fafc" font-family="{FONT_SANS}" font-size="9.5" font-weight="700">{esc(wr["host_label"].replace("_", " "))}</text>',
            f'  <text x="50" y="{ry+43}" fill="{"#6ee7b7" if bm else "#94a3b8"}" font-family="{FONT_MONO}" font-size="7.9">{esc(rail)}</text>',
            f'  <text x="292" y="{ry+22}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{float(wr["w20_bfws4_well_joulebit6_ns_per_op"]):.0f} ns</text>',
            f'  <text x="292" y="{ry+43}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.1">{e20:.0f} nJ</text>',
            f'  <text x="378" y="{ry+22}" fill="#fda4af" font-family="{FONT_MONO}" font-size="8.8">{float(wr["w20b_rung5_json_int8_lb2_ns_per_op"]):.0f} ns</text>',
            f'  <text x="378" y="{ry+43}" fill="#fda4af" font-family="{FONT_MONO}" font-size="8.1">{e20b:.0f} nJ</text>',
            f'  <text x="470" y="{ry+32}" fill="#38bdf8" font-family="{FONT_MONO}" font-size="9.6" font-weight="800">{float(wr["latency_speedup_x"]):.2f}x</text>',
            f'  <text x="530" y="{ry+32}" fill="#34d399" font-family="{FONT_MONO}" font-size="9.6" font-weight="800">{float(wr["energy_reduction_x"]):.2f}x</text>',
        ]
    p += [
        '  <rect x="42" y="788" width="548" height="72" rx="6" fill="#0b1f33" stroke="#38bdf8" stroke-width="0.9"/>',
        f'  <text x="52" y="809" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">12 Python/POSIX Stdlib Modules Projected Rung 5 (LLM JSON) -&gt; Rung 7 (64-bit LUT6):</text>',
        f'  <text x="52" y="828" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.0">• ipaddress · http.HTTPStatus · WCAG 2.2 · stat · urllib · semver · datetime · errno · bisect</text>',
        f'  <text x="52" y="846" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.0">• 7.38M x energy cut vs Rung-5 LLM JSON tool-call &amp; 2,125x vs Rung-6 Python subprocess IPC</text>',
    ]

    # PANEL D: Claim LA-59 — Zero-DSP XC7A200T Gateware + 15-Export WASM + 106 6-Kernel Proofs
    p += [
        '  <rect x="632" y="484" width="584" height="392" rx="12" fill="#0d1729" stroke="#a78bfa" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="650" y="509" fill="#c4b5fd" font-family="{FONT_SANS}" font-size="12.5" font-weight="800">D. Claim LA-59 · Zero-DSP XC7A200T, {wasm_kb:.2f} KB WASM &amp; {n_thms} 6-Kernel Proofs</text>',
        f'  <text x="650" y="526" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.0">Theorems 105–106 · Crossing 100+ LutAdapt Theorems ({n_fleet:,} Linux bwrap Fleet)</text>',
    ]
    cards = [
        (
            "1. Artix-7 XC7A200T-2FBG484-2 4-Bank BFWS + Well Gateware (openXC7)",
            f"0 DSP48E1 slices | {xc7['placed_slice_lutx']} / {xc7['available_slice_lutx']:,} LUT6s ({100.0 * xc7['placed_slice_lutx'] / xc7['available_slice_lutx']:.2f}%) | {xc7['placed_slice_ffx']} FFs",
            f"Post-PAR Fmax = {xc7['post_par_max_freq_mhz']:.2f} MHz ({xc7['post_par_critical_path_total_ns']:.2f} ns) | 64/64 Well & JouleBit6 RTL Parity",
            "#38bdf8",
        ),
        (
            f"2. Browser-Native #![no_std] lut6_core.wasm ({wasm_bytes:,} B = {wasm_kb:.2f} KB <= 16 KB)",
            "15 C-ABI Exports (incl. wasm_eval_bfws4_and_well_codifier) · 0 External Crates · V8 Verified",
            "Theorem 105 32-Bit Bijection: 4-Bank BFWS Cascade + 8-Gate Well-Codifier + JouleBit6 Slab",
            "#34d399",
        ),
        (
            f"3. Historic 100+ Theorem Milestone ({n_thms} LutAdapt / {n_fleet:,} bwrap Fleet)",
            f"con-leche --verified ({lean['con_leche_verified_declarations']:,} decls) · con-ron --verified · lean4lean · nanoda",
            "18/106 Zero-Axiom · 73/106 Choice-Free · 0 sorry · 0 Custom Axioms · 8 New in BfwsWellMultimodal",
            "#c4b5fd",
        ),
        (
            "4. Four-Bank 16 KB System-0 Smartwatch & FPGA Vault",
            "4 x 4,096 B = 16,384 B (25.00% of 64 KB Watch SRAM, 2,048 LUT6 Slots, 4 RAMB36E1 Blocks)",
            "0 Wave-2 Hash-Collision Breaker Trips Across All 25 Official ARC-AGI-3 Games (183 Levels)",
            "#fbbf24",
        ),
    ]
    for idx, (title_c, l1, l2, col_c) in enumerate(cards):
        ry = 540 + idx * 80
        p += [
            f'  <rect x="650" y="{ry}" width="548" height="71" rx="7" fill="#121d31" stroke="{col_c}" stroke-opacity="0.55" stroke-width="1.0"/>',
            f'  <text x="662" y="{ry+21}" fill="{col_c}" font-family="{FONT_SANS}" font-size="9.6" font-weight="800">{esc(title_c)}</text>',
            f'  <text x="662" y="{ry+41}" fill="#f1f5f9" font-family="{FONT_MONO}" font-size="8.2" font-weight="600">{esc(l1)}</text>',
            f'  <text x="662" y="{ry+58}" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.0">{esc(l2)}</text>',
        ]
    p.append("</svg>")

    svg_txt = "\n".join(p) + "\n"
    ET.fromstring(svg_txt)

    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_path = fig_dir / "bfws-arc3-well-and-multimodal.svg"
    png_path = fig_dir / "bfws-arc3-well-and-multimodal.png"
    svg_path.write_text(svg_txt, encoding="utf-8")

    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-w", "2480", str(svg_path), "-o", str(png_path)], check=True)
    print(f"Generated {svg_path} ({svg_path.stat().st_size} B) and {png_path} ({png_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
