"""Generate publication-grade vector SVG + 2400px Retina PNG for Phase 36 (`Figure 12`):
`paper/figures/bonsai-stream-benes-arc2-and-arc-agi-3.{svg,png}`
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

    p36 = json.loads((art / "phase36_bonsai_stream_arc3_meta.json").read_text(encoding="utf-8"))
    lean = json.loads((art / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet = json.loads((art / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    arc2_rows = list(csv.DictReader((art / "arc_agi_2_multilayer_benes_lut6.csv").open(encoding="utf-8")))
    arc3_rows = list(csv.DictReader((art / "arc_agi_3_interactive_mol_rhae.csv").open(encoding="utf-8")))

    b_doc = p36["bonsai_27b_airmol_m3_ultra"]
    b_rows = b_doc["rows"]
    cs = p36["cross_silicon_w14_w15_telemetry"]
    thor = cs["jetson_agx_thor_board_metered"]
    zen5 = cs["amd_zen5_395_rapl"]
    m3u = cs["apple_m3_ultra_darwin"]

    n_thms = int(lean["total_kernel_checked_theorems"])
    n_fleet = int(fleet["total_theorems_and_lemmas"])
    wasm_bytes = (art / "wasm" / "lut6_core.wasm").stat().st_size
    wasm_kb = wasm_bytes / 1024.0

    W, H = 1220, 860
    sub_hdr = (
        f"Claims LA-19 ({n_thms} Thms / {n_fleet:,} bwrap Fleet), LA-42..LA-44  |  "
        f"Thor DC Shunt measured_j (board_metered=true) != Host reported_j  |  0 Raw Benchmark Grids/Frames/Text"
    )

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p36-t p36-d">',
        '  <title id="p36-t">Bonsai-AirMoL 1.75 bpw 27B Streaming Speculation, Multi-Layer Benes ARC-AGI-2 Synthesis, and Official ARC-AGI-3 Interactive RHAE</title>',
        f'  <desc id="p36-d">Four-panel summary of Claims LA-42 through LA-44 and {n_thms} six-kernel Lean 4.36.0 theorems ({n_fleet} fleet): (A) Real prism-ml/Ternary-Bonsai-2-27B EntroMCU-28 1.750 bpw streaming and 54-bit 3-token LUT6 speculation on M3 Ultra 256GB (1.619x I/O cut, 100% greedy parity) plus Jetson AGX Thor DC shunt measured_j ({thor["w14_vs_w13_measured_energy_gain"]}x CpJ); (B) Multi-layer Benes-ARC-LUT6 spatial cascades on ARC-AGI-2 (20.83% on n=24 and 12.50% on n=64, 100% selective accuracy, 0% CHR@80); (C) Official ARC-AGI-3 Interactive Reasoning Benchmark across 8 environments (57 levels) scored by EnvironmentScoreCalculator RHAE; and (D) {n_thms} six-kernel theorems, {wasm_kb:.2f} KB lut6_core.wasm, and R2 precompressed .br sidecar invariant.</desc>',
        "  <defs>",
        '    <linearGradient id="bg36" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#060b16"/>',
        '      <stop offset="50%" stop-color="#0a1426"/>',
        '      <stop offset="100%" stop-color="#071a20"/>',
        "    </linearGradient>",
        '    <linearGradient id="g36-em" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#22d3ee"/>',
        "    </linearGradient>",
        '    <linearGradient id="g36-am" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#fbbf24"/>',
        "    </linearGradient>",
        '    <linearGradient id="g36-ro" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#f43f5e"/><stop offset="100%" stop-color="#fb7185"/>',
        "    </linearGradient>",
        '    <linearGradient id="g36-vi" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#a78bfa"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{W}" height="{H}" fill="url(#bg36)"/>',
        f'  <text x="30" y="36" fill="#f8fafc" font-family="{FONT_SANS}" font-size="17.8" font-weight="800">Bonsai-AirMoL 1.75 bpw 27B Streaming, Multi-Layer Beneš ARC-AGI-2 &amp; Official ARC-AGI-3 RHAE</text>',
        f'  <text x="30" y="56" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.7">{esc(sub_hdr)}</text>',
    ]

    # =========================================================================
    # PANEL A: Claim LA-42 — Real Ternary-Bonsai-2-27B Streaming + Thor Window 14
    # =========================================================================
    p += [
        '  <rect x="24" y="72" width="576" height="372" rx="12" fill="#0d1729" stroke="#22d3ee" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="98" fill="#22d3ee" font-family="{FONT_SANS}" font-size="13.0" font-weight="800">A. Claim LA-42 · Real Ternary-Bonsai-2-27B EntroMCU-28 + 3-Tok LUT6 AirMoL</text>',
        f'  <text x="42" y="116" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.3">M3 Ultra 256GB End-to-End (|V|=248,320, 48 GatedDeltaNet + 16 FullAttn) · 100.0% Exact Greedy Parity</text>',
    ]
    for i, br in enumerate(b_rows):
        ry = 128 + i * 54
        is_air = "Bonsai-AirMoL" in br["mode"]
        is_mcu = "EntroMCU-28 T=1" in br["mode"]
        bar_w = int(310 * (float(br["streamed_linear_gb_per_token"]) / float(b_rows[0]["streamed_linear_gb_per_token"])))
        grad = "url(#g36-em)" if is_air else ("url(#g36-vi)" if is_mcu else "url(#g36-ro)")
        col = "#6ee7b7" if is_air else ("#c4b5fd" if is_mcu else "#fda4af")
        short_m = (
            "OpenIE Bonsai-AirMoL (1.75 bpw + 54b 3-Tok LUT6 + Hybrid Rollback)"
            if is_air
            else ("OpenIE EntroMCU-28 T=1 Lossless Stream (1.75 bpw, 0 pread err)" if is_mcu else "MLX 2-Bit Affine T=1 Baseline (2.25 bpw, 36 B/128w)")
        )
        stat_s = (
            f"{br['streamed_linear_gb_per_token']:.3f} GiB/tok ({br['io_byte_reduction_factor']:.3f}x cut) · "
            f"{br['resident_tok_per_s_m3_ultra']} tok/s · {br['reported_j_per_token_m3_ultra']} J/tok"
        )
        p += [
            f'  <text x="44" y="{ry+12}" fill="#e2e8f0" font-family="{FONT_SANS}" font-size="10.2" font-weight="700">{esc(short_m)}</text>',
            f'  <rect x="44" y="{ry+18}" width="{max(bar_w, 60)}" height="16" rx="4" fill="{grad}"/>',
            f'  <text x="{52 + max(bar_w, 60)}" y="{ry+30}" fill="{col}" font-family="{FONT_MONO}" font-size="8.7" font-weight="700">{esc(stat_s)}</text>',
        ]

    p.append(
        f'  <text x="42" y="304" fill="#fde68a" font-family="{FONT_SANS}" font-size="10.8" font-weight="700">Window 14 (EntroMCU-28 + FWHT-1024 + 4-Tok Burst) vs. Window 13 (MLX 2-Bit T=1) Silicon Energy:</text>'
    )
    hw14 = [
        (
            "Jetson AGX Thor DC Shunt (measured_j)",
            f"{thor['w14_bonsai_airmol_measured_nj']/1e3:.1f} vs {thor['w13_mlx2bit_measured_nj']/1e3:.1f} uJ/tok ({thor['w14_vs_w13_measured_energy_gain']:.2f}x CpJ, {thor['w14_vs_w13_speedup']:.2f}x spd)",
            235,
            "url(#g36-em)",
            "#6ee7b7",
        ),
        (
            "AMD Ryzen AI Max+ 395 RAPL (reported_j)",
            f"{zen5['w14_reported_nj']/1e3:.1f} vs {zen5['w13_reported_nj']/1e3:.1f} uJ/tok ({zen5['w14_vs_w13_rapl_gain']:.2f}x CpJ, {zen5['w14_vs_w13_speedup']:.2f}x spd)",
            195,
            "url(#g36-vi)",
            "#c4b5fd",
        ),
        (
            "Apple M3 Ultra Counter (reported_j)",
            f"{m3u['w14_reported_nj']/1e3:.2f} vs {m3u['w13_reported_nj']/1e3:.2f} uJ/tok ({m3u['w14_vs_w13_reported_gain']:.2f}x CpJ, {m3u['w14_vs_w13_speedup']:.2f}x spd)",
            188,
            "url(#g36-am)",
            "#fde68a",
        ),
    ]
    for idx, (hlbl, htxt, hw_w, hgrad, hcol) in enumerate(hw14):
        hy = 314 + idx * 34
        p += [
            f'  <rect x="44" y="{hy}" width="{hw_w}" height="15" rx="3.5" fill="{hgrad}"/>',
            f'  <text x="50" y="{hy+11}" fill="#060b16" font-family="{FONT_SANS}" font-size="8.8" font-weight="800">{esc(hlbl[:34])}</text>',
            f'  <text x="{52 + hw_w}" y="{hy+12}" fill="{hcol}" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">{esc(htxt)}</text>',
        ]
    p.append(
        f'  <text x="42" y="432" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.0">Saves 1.490 GiB/sweep (1,599,897,600 B) · {b_doc["crystallized_lut6_entries"]}/512 slots in 4 KB LUT6 · {b_doc["wave2_draft_acceptance_pct"]}% 3-gram burst accept</text>'
    )

    # =========================================================================
    # PANEL B: Claim LA-43 — Multi-Layer Beneš-Routed ARC-AGI-2 Synthesis
    # =========================================================================
    p += [
        '  <rect x="620" y="72" width="576" height="372" rx="12" fill="#0d1729" stroke="#a78bfa" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="638" y="98" fill="#c4b5fd" font-family="{FONT_SANS}" font-size="13.0" font-weight="800">B. Claim LA-43 · Multi-Layer Beneš-ARC-LUT6 on ARC-AGI-2 (n=24 &amp; n=64)</text>',
        f'  <text x="638" y="116" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.3">Theorem 70/71: Diameter 2L+1 (3&#x2192;5&#x2192;7) · Strictly C_train(f)=1 Gated · 0 Test-Grid Leakage</text>',
        '  <rect x="638" y="126" width="540" height="22" rx="4" fill="#1e293b"/>',
        f'  <text x="646" y="141" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9" font-weight="700">Slice &amp; Synthesizer Cascade</text>',
        f'  <text x="954" y="141" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9" font-weight="700">Solved</text>',
        f'  <text x="1012" y="141" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9" font-weight="700">SelAcc</text>',
        f'  <text x="1070" y="141" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9" font-weight="700">CHR@80</text>',
        f'  <text x="1126" y="141" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9" font-weight="700">LUT6s</text>',
    ]
    for idx, ar in enumerate(arc2_rows):
        ay = 152 + idx * 33
        is_mol = "OpenIE" in ar["system"]
        is_top = "3-Layer" in ar["system"] or ("2-Layer" in ar["system"] and ar["slice"] == "Matched_n24")
        bg_r = "#08221e" if is_top else ("#0b1929" if is_mol else "#1a1118")
        tc = "#6ee7b7" if is_top else ("#93c5fd" if is_mol else "#fda4af")
        s_tag = "n=24" if ar["slice"] == "Matched_n24" else "n=64"
        sys_s = (
            ar["system"]
            .replace("OpenIE ", "")
            .replace(" (No C_train Gate)", "")
            .replace(" (Diameter 3, C_train==1 Gate)", " [D=3]")
            .replace(" (Diameter 5 + Panel/Ray/Symmetry, C_train==1)", " [D=5]")
            .replace(" (Diameter 7 + Gravity/Component, C_train==1)", " [D=7]")
        )
        lbl = f"{s_tag} · {sys_s}"[:46]
        chr_c = "#6ee7b7" if float(ar["confident_hallucination_rate_80_pct"]) == 0 else "#fda4af"
        p += [
            f'  <rect x="638" y="{ay}" width="540" height="29" rx="4" fill="{bg_r}"/>',
            f'  <text x="646" y="{ay+18}" fill="{tc}" font-family="{FONT_MONO}" font-size="8.8" font-weight="{"700" if is_mol else "400"}">{esc(lbl)}</text>',
            f'  <text x="954" y="{ay+18}" fill="{tc}" font-family="{FONT_MONO}" font-size="9.0" font-weight="700">{esc(ar["solved_pct"])}%</text>',
            f'  <text x="1012" y="{ay+18}" fill="{tc}" font-family="{FONT_MONO}" font-size="9.0">{esc(ar["selective_committed_acc_pct"])}%</text>',
            f'  <text x="1070" y="{ay+18}" fill="{chr_c}" font-family="{FONT_MONO}" font-size="9.0">{esc(ar["confident_hallucination_rate_80_pct"])}%</text>',
            f'  <text x="1128" y="{ay+18}" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.9">{esc(ar["mean_lut6_gates"])}</text>',
        ]
    p.append(
        f'  <text x="638" y="432" fill="#fde68a" font-family="{FONT_MONO}" font-size="9.0">Matched n=24: 12.50% &#x2192; 20.83% (+66.7%) · Expanded n=64: 6.25% &#x2192; 12.50% (2.0x) · 0.00% Unsound</text>'
    )

    # =========================================================================
    # PANEL C: Claim LA-44 — Official ARC-AGI-3 Interactive Reasoning (RHAE)
    # =========================================================================
    p += [
        '  <rect x="24" y="462" width="736" height="374" rx="12" fill="#0d1729" stroke="#10b981" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="488" fill="#6ee7b7" font-family="{FONT_SANS}" font-size="13.0" font-weight="800">C. Claim LA-44 · Official ARC-AGI-3 Interactive Reasoning Benchmark (RHAE)</text>',
        f'  <text x="42" y="506" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.2">arc-agi 0.9.10 + arcengine 0.9.3 (MIT) · 8 Official Environments (57 Levels) · 100% Black-Box 64x64 Frames</text>',
        '  <rect x="42" y="518" width="700" height="24" rx="5" fill="#1e293b"/>',
        f'  <text x="50" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">Interactive Agent Architecture</text>',
        f'  <text x="428" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">Levels</text>',
        f'  <text x="492" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">Mean RHAE</text>',
        f'  <text x="576" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">Peak RHAE</text>',
        f'  <text x="660" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">LUT6 Step%</text>',
    ]
    for idx, r3 in enumerate(arc3_rows):
        ry = 548 + idx * 50
        is_w2 = "Wave 2" in r3["system"]
        is_w1 = "Wave 1" in r3["system"]
        rbg = "#08241f" if is_w2 else ("#0b1b2e" if is_w1 else "#141827")
        stroke3 = "#10b981" if is_w2 else ("#38bdf8" if is_w1 else "#334155")
        tc3 = "#6ee7b7" if is_w2 else ("#93c5fd" if is_w1 else "#cbd5e1")
        short3 = (
            r3["system"]
            .replace(" (arcprize Uniform Available Actions)", "")
            .replace(" (No World-Model Verification)", "")
            .replace("OpenIE Interactive-MoL ", "Interactive-MoL ")
            .replace(", C_env==1)", ")")
        )[:52]
        sub3 = (
            f"Actions: {r3['total_actions_taken']} | Crystal: {r3['active_policy_crystal_bytes']} B ({r3['crystallized_lut6_slots']} slots) | "
            f"Step: {r3['us_per_env_step']} us | Shift Breakers: {r3['mechanic_shift_breaker_trips']}"
        )
        lvl_s = f"{r3['levels_completed']}/{r3['total_levels']}"
        p += [
            f'  <rect x="42" y="{ry}" width="700" height="44" rx="6" fill="{rbg}" stroke="{stroke3}" stroke-width="1.1"/>',
            f'  <text x="50" y="{ry+17}" fill="{tc3}" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">{esc(short3)}</text>',
            f'  <text x="50" y="{ry+34}" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.4">{esc(sub3)}</text>',
            f'  <text x="428" y="{ry+18}" fill="{tc3}" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">{esc(lvl_s)}</text>',
            f'  <text x="496" y="{ry+18}" fill="{tc3}" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">{esc(r3["mean_official_rhae_score"])}</text>',
            f'  <text x="582" y="{ry+18}" fill="#fde68a" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">{esc(r3["peak_game_rhae_score"])}</text>',
            f'  <text x="666" y="{ry+18}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">{esc(r3["lut6_reflex_step_share_pct"])}%</text>',
        ]

    p += [
        '  <rect x="42" y="756" width="700" height="66" rx="8" fill="#081a1a" stroke="#10b981" stroke-opacity="0.5"/>',
        f'  <text x="54" y="776" fill="#6ee7b7" font-family="{FONT_SANS}" font-size="10.4" font-weight="800">Why Downward System-0 Compilation Matters Quadratically on ARC-AGI-3 (Theorem 72 &amp; Window 15):</text>',
        f'  <text x="54" y="794" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.8">Official RHAE = min(115, 100 * (baseline / actions)^2): 4x exploration actions &#x2192; 16x quadratic score penalty!</text>',
        f'  <text x="54" y="811" fill="#93c5fd" font-family="{FONT_MONO}" font-size="8.8">Window 15 (2-Layer ARC2 + ARC3 Reflex): Thor Shunt = {thor["w15_arc2_arc3_reflex_measured_nj"]:.1f} nJ/op (measured_j) · {thor["w15_arc2_arc3_reflex_ns"]:.1f} ns/op</text>',
    ]

    # =========================================================================
    # PANEL D: Claim LA-19 (74 Proofs / 1,438 Fleet), 12.75 KB WASM & R2 .br Sync
    # =========================================================================
    p += [
        '  <rect x="776" y="462" width="420" height="374" rx="12" fill="#0d1729" stroke="#f59e0b" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="794" y="488" fill="#fde68a" font-family="{FONT_SANS}" font-size="13.0" font-weight="800">D. {n_thms} Proofs, {wasm_kb:.2f} KB WASM &amp; Edge .br Sync</text>',
        f'  <text x="794" y="506" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.2">6/6 Kernels · {n_fleet:,} Linux bwrap Fleet · 11 Live WASM Exports</text>',
    ]
    d_cards = [
        (
            f"1 · BonsaiStreamArc3.lean ({n_thms} LutAdapt Thms)",
            f"{lean['purely_constructive_zero_axiom_theorems']}/{n_thms} Zero-Axiom · {lean['choice_free_theorems']}/{n_thms} Choice-Free · 0 sorry",
            f"con-leche & con-ron: {lean['con_leche_verified_declarations']:,} decls (--verified)",
            "#10b981",
        ),
        (
            "2 · Theorems 67..71 (Bonsai-AirMoL & Beneš-ARC2)",
            "Exact 2/9 EntroMCU-28 cut · 54-bit 3-tok bijection",
            "Diameter 2L+1 & constructive 2-layer separation",
            "#22d3ee",
        ),
        (
            "3 · Theorems 72..74 (ARC-AGI-3 RHAE & Edge R2)",
            "Quadratic (B/A)^2 bound · 0 stale shift actions",
            "r2_sync .br sidecar sync -> 0 stale edge shadows",
            "#a78bfa",
        ),
        (
            f"4 · Browser Silicon Lab ({wasm_kb:.2f} KB = {wasm_bytes:,} B)",
            "wasm_eval_benes_arc2_2layer8 & wasm_eval_arc3_reflex",
            "<= 16 KB budget across all 8 #![no_std] targets!",
            "#f59e0b",
        ),
    ]
    for idx, (c1, c2, c3, ccol) in enumerate(d_cards):
        cy = 522 + idx * 72
        p += [
            f'  <rect x="794" y="{cy}" width="384" height="62" rx="8" fill="#091120" stroke="{ccol}" stroke-width="1.3"/>',
            f'  <text x="808" y="{cy+19}" fill="{ccol}" font-family="{FONT_SANS}" font-size="10.5" font-weight="800">{esc(c1)}</text>',
            f'  <text x="808" y="{cy+37}" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="8.9">{esc(c2)}</text>',
            f'  <text x="808" y="{cy+53}" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.8">{esc(c3)}</text>',
        ]
    p.append(
        '  <text x="794" y="822" fill="#6ee7b7" font-family="ui-monospace, Menlo, monospace" font-size="9.0">site-ops/deploy/r2_sync.py: auto-syncs &amp; prunes .br sidecars (6f33577)</text>'
    )
    p.append("</svg>")

    svg_text = "\n".join(p) + "\n"
    ET.fromstring(svg_text)

    out_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "bonsai-stream-benes-arc2-and-arc-agi-3.svg"
    png_path = out_dir / "bonsai-stream-benes-arc2-and-arc-agi-3.png"
    svg_path.write_text(svg_text, encoding="utf-8")

    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-w", "2400", "-o", str(png_path), str(svg_path)], check=True)
    shutil.copy2(svg_path, root / "docs" / "keynote" / svg_path.name)
    print(f"Generated {svg_path} ({svg_path.stat().st_size} B) & {png_path} ({png_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
