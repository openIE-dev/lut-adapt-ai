"""Generate publication-grade vector SVG + 2480px Retina PNG for Phase 39 (`Figure 15`):
`paper/figures/siw2-arc3-and-joulebit-27b.{svg,png}`
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
    arc3_rows = list(csv.DictReader((art / "phase39_arc3_16env_siw2_summary.csv").open(encoding="utf-8")))
    arc3_games = list(csv.DictReader((art / "phase39_arc3_16env_siw2_per_game.csv").open(encoding="utf-8")))
    m3u_27b = json.loads((art / "phase39_bonsai27b_joulebit_m3_ultra.json").read_text(encoding="utf-8"))
    w19_rows = list(csv.DictReader((art / "phase39_window19_cross_silicon.csv").open(encoding="utf-8")))
    xc7 = json.loads((art / "openxc7_siw_joulebit_timing.json").read_text(encoding="utf-8"))
    lean = json.loads((art / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet = json.loads((art / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    wasm_bytes = (art / "wasm" / "lut6_core.wasm").stat().st_size
    wasm_kb = round(wasm_bytes / 1024.0, 2)

    n_thms = int(lean["total_kernel_checked_theorems"])
    n_fleet = int(fleet["total_theorems_and_lemmas"])
    thor_w19 = next(r for r in w19_rows if r["board_metered"] == "True")
    thor_nj = float(thor_w19["w19_measured_nj_per_op_vdd_cpu"])
    thor_gain = float(thor_w19["energy_reduction_x"])

    W, H = 1240, 900
    sub_hdr = (
        f"Claims LA-19 ({n_thms} Thms / {n_fleet:,} bwrap Fleet) & LA-53..LA-55  |  "
        f"ARC-AGI-3 SIW(2) = 9/113 Solved (7 Games, RHAE 2.252)  |  Thor Window 19 = {thor_nj:.0f} nJ ({thor_gain:.2f}x vs LittleBit-2)  |  {wasm_kb:.2f} KB 14-Export WASM"
    )

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p39-t p39-d">',
        '  <title id="p39-t">Landmark-Serialized SIW(2) on 16-Game ARC-AGI-3, Sharded JouleBit-LUT6 + EntroMCU-28 on Real Ternary-Bonsai-2-27B, and Zero-DSP XC7A200T Gateware</title>',
        f'  <desc id="p39-d">Four-panel summary of Claims LA-53 through LA-55 and {n_thms} six-kernel Lean 4.36.0 theorems ({n_fleet} Linux bwrap fleet): (A) Official 16-environment ARC-AGI-3 Landmark-Serialized Iterated Width SIW(2) + Two-Bank 8 KB System-0 LUT6 vault (9/113 levels across 7 games, mean RHAE 2.252, 0 breaker trips); (B) End-to-end sharded JouleBit-LUT6 + EntroMCU-28 speculative streaming on real prism-ml/Ternary-Bonsai-2-27B on Apple M3 Ultra 256 GB across Cold Wave-1 and Warm Wave-2 prompts at 100.0% greedy parity; (C) Cross-silicon Window 19 telemetry across Jetson AGX Thor DC shunt measured_j, AMD Zen 5 RAPL reported_j, and Apple M3 Ultra reported_j; and (D) Zero-DSP48E1 Artix-7 XC7A200T openXC7 gateware, {wasm_kb:.2f} KB 14-export lut6_core.wasm, and {n_thms} six-kernel theorems.</desc>',
        "  <defs>",
        '    <linearGradient id="bg39" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#040814"/><stop offset="52%" stop-color="#091628"/><stop offset="100%" stop-color="#041b1e"/>',
        "    </linearGradient>",
        '    <linearGradient id="g39-em" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#22d3ee"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{W}" height="{H}" fill="url(#bg39)"/>',
        f'  <text x="28" y="34" fill="#f8fafc" font-family="{FONT_SANS}" font-size="16.8" font-weight="800">Landmark-Serialized SIW(2) ARC-AGI-3, Real 27B JouleBit-LUT6 Streaming &amp; Zero-DSP Gateware</text>',
        f'  <text x="28" y="54" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.9">{esc(sub_hdr)}</text>',
    ]

    # PANEL A: Claim LA-53 — 16-Game Official ARC-AGI-3 SIW(2) + Two-Bank 8 KB Vault
    p += [
        '  <rect x="24" y="68" width="584" height="398" rx="12" fill="#0d1729" stroke="#22d3ee" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="92" fill="#22d3ee" font-family="{FONT_SANS}" font-size="12.6" font-weight="800">A. Claim LA-53 · 16-Game Official Black-Box ARC-AGI-3 SIW(2) + 8 KB Vault</text>',
        f'  <text x="42" y="109" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.1">Theorems 91–93 · Landmark Reset + 6px Canvas Subdivision + Two-Bank 8 KB (12.5% Watch SRAM)</text>',
        '  <rect x="42" y="118" width="548" height="20" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Agent Architecture (113 Official Levels)</text>',
        f'  <text x="330" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Games</text>',
        f'  <text x="378" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Solved</text>',
        f'  <text x="434" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Mean RHAE</text>',
        f'  <text x="516" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Vault</text>',
    ]
    sel_arc = [arc3_rows[0], arc3_rows[2], arc3_rows[4], arc3_rows[5], arc3_rows[6], arc3_rows[7], arc3_rows[8]]
    for idx, r in enumerate(sel_arc):
        ry = 143 + idx * 36
        s = r["system"]
        is_siw_w2 = "SIW(2) ->" in s
        is_siw_w1 = "SIW(2) +" in s
        bg = "#062c24" if is_siw_w2 else ("#0b243b" if is_siw_w1 else "#131c2e")
        tc = "#6ee7b7" if is_siw_w2 else ("#7dd3fc" if is_siw_w1 else "#cbd5e1")
        lbl = (
            s.replace("Phase 39 OpenIE-Apex ", "Apex ")
            .replace("Phase 38 OpenIE-Apex ", "Phase 38 ")
            .replace("Phase 37 OpenIE-Apex ", "Phase 37 ")
            .replace("Phase 36 ", "")
        )[:41]
        vb = int(r["active_policy_crystal_bytes"])
        v_txt = f"{vb // 1024} KB" if vb > 0 else "0 B"
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="31" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="48" y="{ry+19}" fill="{tc}" font-family="{FONT_SANS}" font-size="8.8" font-weight="{"700" if (is_siw_w1 or is_siw_w2) else "500"}">{esc(lbl)}</text>',
            f'  <text x="334" y="{ry+19}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{r["games_with_solved_levels"]}/16</text>',
            f'  <text x="380" y="{ry+19}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{r["levels_completed_16"]}/113</text>',
            f'  <text x="442" y="{ry+19}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{float(r["mean_official_rhae_16"]):.4f}</text>',
            f'  <text x="518" y="{ry+19}" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.4">{v_txt}</text>',
        ]
    solved_g = [g for g in arc3_games if g["agent_kind"] == "apex_siw2_lut6_wave2" and int(g["levels_completed"]) > 0]
    g_summary = " · ".join(f'{g["game_id"].split("-")[0]}:{g["levels_completed"]} ({float(g["official_rhae_score"]):.1f})' for g in solved_g)
    p += [
        '  <rect x="42" y="399" width="548" height="54" rx="6" fill="#072922" stroke="#10b981" stroke-width="0.9"/>',
        f'  <text x="52" y="418" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">7 Solved Games (+m0r0 &amp; +s5i5 unlocked, +49.7% Mean RHAE vs IW(2), 0 Breaker Trips):</text>',
        f'  <text x="52" y="438" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.0">{esc(g_summary)}</text>',
    ]

    # PANEL B: Claim LA-54 — Real Ternary-Bonsai-2-27B Sharded JouleBit-LUT6 + EntroMCU-28
    ha = m3u_27b["real_hidden_state_top1_audit"]
    p += [
        '  <rect x="632" y="68" width="584" height="398" rx="12" fill="#0d1729" stroke="#10b981" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="650" y="92" fill="#34d399" font-family="{FONT_SANS}" font-size="12.6" font-weight="800">B. Claim LA-54 · Real Ternary-Bonsai-2-27B JouleBit-LUT6 Streaming (M3 Ultra)</text>',
        f'  <text x="650" y="109" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.1">0 QAT · Real 27B Hidden Top-1 = {ha["joulebit_lut6_045bpw_top1_pct"]}% (cos={ha["joulebit_lut6_045bpw_mean_cosine"]}) · .jbit6 pread = {m3u_27b["pread_jbit6_gbps"]} GiB/s · 100% Parity</text>',
        '  <rect x="650" y="118" width="548" height="20" rx="4" fill="#1e293b"/>',
        f'  <text x="656" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">27B Streaming Arm (333 Greedy Tokens)</text>',
        f'  <text x="936" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Cold W1</text>',
        f'  <text x="1006" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Warm W2</text>',
        f'  <text x="1076" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">GiB/tok</text>',
        f'  <text x="1142" y="132" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">JpCT (J)</text>',
    ]
    for idx, r in enumerate(m3u_27b["rows"]):
        ry = 143 + idx * 41
        s = r["system"]
        is_ours = "(Ours)" in s
        is_bad = "LittleBit-2" in s
        bg = "#062c24" if is_ours else ("#2a121b" if is_bad else "#131c2e")
        tc = "#6ee7b7" if is_ours else ("#fda4af" if is_bad else "#cbd5e1")
        short_s = (
            s.replace("OpenIE Two-Rung Token-LUT6 (8 KB) + ", "Two-Rung 8KB + ")
            .replace(" + EntroMCU-28 (Ours)", " (Ours)")
            .replace("Phase 36 Cross-Prompt ", "Phase 36 ")
            .replace("OpenIE ", "")
        )[:39]
        p += [
            f'  <rect x="650" y="{ry}" width="548" height="35" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="656" y="{ry+21}" fill="{tc}" font-family="{FONT_SANS}" font-size="8.7" font-weight="{"700" if is_ours else "500"}">{esc(short_s)}</text>',
            f'  <text x="938" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{r["cold_wave1_tok_per_sweep"]:.3f}x</text>',
            f'  <text x="1008" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{r["warm_wave2_tok_per_sweep"]:.3f}x</text>',
            f'  <text x="1080" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{r["combined_streamed_gb_per_tok"]:.3f}</text>',
            f'  <text x="1144" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{r["m3u_reported_j_per_correct_tok"]:.4f}</text>',
        ]
    p += [
        '  <rect x="650" y="394" width="548" height="59" rx="6" fill="#072922" stroke="#10b981" stroke-width="0.9"/>',
        f'  <text x="660" y="414" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">Theorems 95–97 Verified on 27B: Beats Cold Wave-1 Fallback (1.049 vs 1.000 tok/sweep)</text>',
        f'  <text x="660" y="432" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.2">&amp; Warm Wave-2 (3.935 GiB/tok, 1.72x vs MLX 2-Bit) · Ungated LittleBit-2 Thrash = 0.0995 J/tok!</text>',
        f'  <text x="660" y="446" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.0">100.00% Exact Greedy Token Parity Across All 8 Prompts (GatedDeltaNet ArraysCache + KV Rollback)</text>',
    ]

    # PANEL C: Cross-Silicon Window 19 Telemetry (Thor DC Shunt measured_j vs Zen 5 RAPL vs M3 Ultra)
    p += [
        '  <rect x="24" y="484" width="584" height="392" rx="12" fill="#0d1729" stroke="#f59e0b" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="509" fill="#fbbf24" font-family="{FONT_SANS}" font-size="12.6" font-weight="800">C. Window 19 Cross-Silicon Shunt &amp; Counter Audit (Zero-MAC FWHT-64 + JouleBit6)</text>',
        f'  <text x="42" y="526" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.1">Theorem 94 Exact 64-Entry Rademacher Slab Parity · Strict measured_j vs reported_j Separation</text>',
        '  <rect x="42" y="536" width="548" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Hardware Host &amp; Telemetry Rail</text>',
        f'  <text x="292" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">W19 (Ours)</text>',
        f'  <text x="378" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">W19b (LB-2)</text>',
        f'  <text x="468" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">Speedup</text>',
        f'  <text x="530" y="550" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.2" font-weight="700">J Cut</text>',
    ]
    for idx, wr in enumerate(w19_rows):
        ry = 565 + idx * 72
        bm = wr["board_metered"] == "True"
        bg = "#072922" if bm else "#121f36"
        rail = "Physical TI INA3221 VDD_CPU_SOC_MSS Shunt (measured_j, board_metered=true)" if bm else (
            "Linux powercap amd_rapl Package Counter (reported_j, board_metered=false)"
            if "Zen5" in wr["host_label"]
            else "Darwin RUSAGE_INFO_V6 ri_energy_nj Counter (reported_j, board_metered=false)"
        )
        e19 = float(wr["w19_measured_nj_per_op_vdd_cpu"] if bm else wr["w19_reported_nj_per_op"])
        e19b = float(wr["w19b_measured_nj_per_op_vdd_cpu"] if bm else wr["w19b_reported_nj_per_op"])
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="62" rx="6" fill="{bg}" stroke="{"#10b981" if bm else "#334155"}" stroke-width="1.0"/>',
            f'  <text x="50" y="{ry+21}" fill="#f8fafc" font-family="{FONT_SANS}" font-size="9.5" font-weight="700">{esc(wr["host_label"].replace("_", " "))}</text>',
            f'  <text x="50" y="{ry+43}" fill="{"#6ee7b7" if bm else "#94a3b8"}" font-family="{FONT_MONO}" font-size="7.9">{esc(rail)}</text>',
            f'  <text x="292" y="{ry+22}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{float(wr["w19_siw2_joulebit6_ns_per_op"]):.0f} ns</text>',
            f'  <text x="292" y="{ry+43}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.1">{e19:.0f} nJ</text>',
            f'  <text x="378" y="{ry+22}" fill="#fda4af" font-family="{FONT_MONO}" font-size="8.8">{float(wr["w19b_littlebit2_mac_ns_per_op"]):.0f} ns</text>',
            f'  <text x="378" y="{ry+43}" fill="#fda4af" font-family="{FONT_MONO}" font-size="8.1">{e19b:.0f} nJ</text>',
            f'  <text x="470" y="{ry+32}" fill="#38bdf8" font-family="{FONT_MONO}" font-size="9.6" font-weight="800">{float(wr["latency_speedup_x"]):.2f}x</text>',
            f'  <text x="530" y="{ry+32}" fill="#34d399" font-family="{FONT_MONO}" font-size="9.6" font-weight="800">{float(wr["energy_reduction_x"]):.2f}x</text>',
        ]
    p += [
        '  <rect x="42" y="788" width="548" height="72" rx="6" fill="#0b1f33" stroke="#38bdf8" stroke-width="0.9"/>',
        f'  <text x="52" y="809" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Zero Binary Multiplications Across All 64 Rows x 60 Rotated FWHT-64 Columns:</text>',
        f'  <text x="52" y="828" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.1">• Eliminates 384 per-slab unpack-and-multiply ops via 10 shared 64-entry Rademacher LUT6 tables</text>',
        f'  <text x="52" y="846" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.1">• Bit-exact integer parity verified between Window 19 and Window 19b on ARM64, x86_64 &amp; wasm32</text>',
    ]

    # PANEL D: Claim LA-55 — Zero-DSP XC7A200T Gateware + 14-Export <=16 KB WASM + 98 6-Kernel Proofs
    p += [
        '  <rect x="632" y="484" width="584" height="392" rx="12" fill="#0d1729" stroke="#a78bfa" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="650" y="509" fill="#c4b5fd" font-family="{FONT_SANS}" font-size="12.6" font-weight="800">D. Claim LA-55 · Zero-DSP XC7A200T Gateware, {wasm_kb:.2f} KB WASM &amp; {n_thms} 6-Kernel Proofs</text>',
        f'  <text x="650" y="526" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.1">Theorem 98 4-Stage Pipeline · License-Free openXC7 .bit · 6/6 Linux bwrap Paranoid Kernels</text>',
    ]
    cards = [
        (
            "1. Artix-7 XC7A200T-2FBG484-2 Unified Gateware (openXC7 Post-PAR)",
            f"0 DSP48E1 slices | {xc7['placed_slice_lutx']} / {xc7['available_slice_lutx']:,} LUT6s ({100.0 * xc7['placed_slice_lutx'] / xc7['available_slice_lutx']:.2f}%) | {xc7['placed_slice_ffx']} FFs",
            f"Post-PAR Fmax = {xc7['post_par_max_freq_mhz']:.2f} MHz ({xc7['post_par_critical_path_total_ns']:.2f} ns) | 77.85 Mbit .bit | 64/64 RTL JouleBit6 Parity",
            "#38bdf8",
        ),
        (
            f"2. Browser-Native #![no_std] lut6_core.wasm ({wasm_bytes:,} B = {wasm_kb:.2f} KB <= 16 KB)",
            "14 C-ABI Exports (incl. wasm_eval_siw2_and_joulebit6) · 0 External Crates · V8 Verified",
            "Two-Bank SIW(2) Landmark Switch + Zero-MUL FWHT-64 JouleBit6 + waiOS BLAKE3 (0xa541323b)",
            "#34d399",
        ),
        (
            f"3. Lean 4.36.0 Six-Kernel Mechanized Trust ({n_thms} LutAdapt / {n_fleet:,} bwrap Fleet)",
            f"con-leche --verified ({lean['con_leche_verified_declarations']:,} decls) · con-ron --verified · lean4lean · nanoda",
            "0 sorry · 0 Custom Axioms · 0 opaque · 0 ofReduceBool · 8 New Proofs in SerializedIwJouleBit27B",
            "#c4b5fd",
        ),
        (
            "4. End-to-End Wearable & Edge Memory Footprint",
            "Two-Bank System-0 Vault = 8,192 B (12.50% of 64 KB Watch Always-On SRAM, 1,024 LUT6 Gates)",
            "1,712x smaller than uLM-28M INT4 (14.02 MB) & >10,948x smaller than decider-0.8b BF16",
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
    svg_path = fig_dir / "siw2-arc3-and-joulebit-27b.svg"
    png_path = fig_dir / "siw2-arc3-and-joulebit-27b.png"
    svg_path.write_text(svg_txt, encoding="utf-8")

    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-w", "2480", str(svg_path), "-o", str(png_path)], check=True)
    print(f"Generated {svg_path} ({svg_path.stat().st_size} B) and {png_path} ({png_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
