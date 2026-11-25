"""Generate publication-grade vector SVG + 2400px Retina PNG for Phase 35 (`Figure 11`):
`paper/figures/bonsai-ternary-decider-and-post-hle.{svg,png}`
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


def esc(s: str) -> str:
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

    p35 = json.loads((art / "phase35_bonsai_decider_post_hle_meta.json").read_text(encoding="utf-8"))
    thor = json.loads((art / "lut6_nucleus_thor_shunt.json").read_text(encoding="utf-8"))
    fw = json.loads((art / "lut6_nucleus_fw128b.json").read_text(encoding="utf-8"))
    m5 = json.loads((art / "lut6_nucleus_m5_max.json").read_text(encoding="utf-8"))
    lean = json.loads((art / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet = json.loads((art / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))

    sys1_rows = list(csv.DictReader((art / "system1_decider_vs_lut6_compile.csv").open(encoding="utf-8")))
    tril_rows = list(csv.DictReader((art / "post_hle_arc_minif2f_critpt_trilogy.csv").open(encoding="utf-8")))

    b = p35["bonsai_2_27b_hadamard_ternary_audit"]
    s1 = p35["system1_decider_vs_lut6_summary"]
    arc = p35["arc_agi_2_summary"]
    m2f = p35["minif2f_lean4_summary"]
    cpt = p35["critpt_physics_summary"]
    wasm_kb = round((art / "wasm" / "lut6_core.wasm").stat().st_size / 1024.0, 2)

    thor_map = {w["window"]: w for w in thor["windows"]}
    fw_map = {w["window"]: w for w in fw["windows"]}
    m5_map = {w["window"]: w for w in m5["windows"]}

    thor_w12 = float(thor_map["Window12_Bonsai_FWHT1024_Antisym40_Block16"]["measured_nj_per_op_vdd_cpu"])
    thor_w13 = float(thor_map["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]["measured_nj_per_op_vdd_cpu"])
    thor_e_gain = thor_w13 / thor_w12
    thor_lat_gain = float(thor_map["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]["ns_per_op"]) / float(
        thor_map["Window12_Bonsai_FWHT1024_Antisym40_Block16"]["ns_per_op"]
    )

    zen_w12 = float(fw_map["Window12_Bonsai_FWHT1024_Antisym40_Block16"]["reported_nj_per_op"])
    zen_w13 = float(fw_map["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]["reported_nj_per_op"])
    zen_e_gain = zen_w13 / zen_w12

    m5_w12 = float(m5_map["Window12_Bonsai_FWHT1024_Antisym40_Block16"]["reported_nj_per_op"])
    m5_w13 = float(m5_map["Window13_Baseline_MLX2Bit_Unpack_Dequant_Block16"]["reported_nj_per_op"])
    m5_e_gain = m5_w13 / m5_w12

    n_thms = int(lean["total_kernel_checked_theorems"])
    n_fleet = int(fleet["total_theorems_and_lemmas"])

    sub_hdr = f"Claims LA-19 ({n_thms} Thms / {n_fleet:,} Fleet), LA-39..LA-41  |  Thor Shunt measured_j (board_metered=true) Strictly Separated from Host reported_j"
    a_sub = (
        f"Audited {b['audited_weights_count']:,} real weights: c=3 count={b['code3_invalid_occurrences']}, "
        f"max|bias+scale|={b['max_abs_bias_plus_scale']}, err={b['max_lossless_roundtrip_f32_err']}"
    )
    a_mlx_lbl = f"MLX Affine-2: 36 B / 128w (2.25 bpw, {b['mlx_linear_total_gb']:.2f} GB)"
    a_ent_lbl = f"OpenIE EntroMCU: 28 B / 128w (1.75 bpw, {b['entromcu_linear_total_gb']:.2f} GB, -22.22%)"
    a_foot = f"Theorem 60: 1024 = 4^5 = 32^2 -> exact >> 5 shift (0 sqrt(2) floats) | Cosine vs MLX F32 = {b['fwht1024_antisym40_vs_mlx_cosine']:.6f}"
    b_sub = f"64 Typed Wearable/Tool Decisions (Wave 1 + Wave 2 Paraphrases + Adversarial OOD) | {s1['footprint_reduction_vs_decider08b']:,.0f}x Smaller"
    b_foot = f"Wave-2 Compiled System-0 LUT6 Speedup vs. Mapika/decider-0.8b (Metal): {s1['wave2_speedup_compiled_lut6_vs_decider08b']:,.1f}x (100% W2 Rung-0 Hit)"
    c_foot = (
        f"ARC-LUT6 mean={arc['arc_lut6_mean_gates_per_admitted_rule']} LUT6s | "
        f"MiniF2F W2 tactic checks cut {m2f['tactic_invocation_reduction_factor']}x ({m2f['wave2_speedup_factor']}x spd) | "
        f"CritPt RMS-CE cut {cpt['rms_calibration_reduction_factor']}x"
    )
    d_sub = f"6/6 Independent Kernels · {n_fleet:,} Linux bwrap Fleet Theorems"

    W, H = 1220, 860
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p35-t p35-d">',
        '  <title id="p35-t">Native Hadamard-Ternary 27B (Bonsai-2-27B) Zero-MUL EntroMCU/T-MAC, System-1 Neural Decider Downward Compilation, and Post-LastExam Open Benchmark Trilogy</title>',
        f'  <desc id="p35-d">Four-panel synthesis of Claims LA-39 through LA-41 and {n_thms} six-kernel Lean 4.36.0 theorems ({n_fleet} fleet): (A) prism-ml/Ternary-Bonsai-2-27B zero-copy audit showing lossless 36 B to 28 B (-22.22%) EntroMCU transcoding, exact shift-5 FWHT-1024, and Thor physical DC shunt measured_j gain ({thor_e_gain:.2f}x); (B) Mapika/decider-0.8b and potion-base-8M neural System-1 deciders vs downward-compiled 4 KB System-0 LUT6; (C) Post-LastExam open benchmark trilogy across ARC-AGI-2 inductive LUT6 synthesis, MiniF2F-Lean4 soundness-zero kernel gate, and CritPt physics calibration; and (D) {n_thms} six-kernel proofs and zero-unsoundness hierarchy.</desc>',
        "  <defs>",
        '    <linearGradient id="bg35" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#060b16"/>',
        '      <stop offset="50%" stop-color="#0a1426"/>',
        '      <stop offset="100%" stop-color="#071a20"/>',
        "    </linearGradient>",
        '    <linearGradient id="g35-em" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#22d3ee"/>',
        "    </linearGradient>",
        '    <linearGradient id="g35-am" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#fbbf24"/>',
        "    </linearGradient>",
        '    <linearGradient id="g35-ro" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#f43f5e"/><stop offset="100%" stop-color="#fb7185"/>',
        "    </linearGradient>",
        '    <linearGradient id="g35-vi" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#38bdf8"/><stop offset="100%" stop-color="#a78bfa"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{W}" height="{H}" fill="url(#bg35)"/>',
        f'  <text x="30" y="36" fill="#f8fafc" font-family="{FONT_SANS}" font-size="18.2" font-weight="800">Hadamard-Ternary 27B Zero-MUL T-MAC, System-1&#x2192;System-0 Compilation &amp; Post-LastExam Trilogy</text>',
        f'  <text x="30" y="56" fill="#94a3b8" font-family="{FONT_MONO}" font-size="10.0">{esc(sub_hdr)}</text>',
        # Panel A
        '  <rect x="24" y="72" width="576" height="372" rx="12" fill="#0d1729" stroke="#22d3ee" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="98" fill="#22d3ee" font-family="{FONT_SANS}" font-size="13.2" font-weight="800">A. Claim LA-39 · Ternary-Bonsai-2-27B (Qwen3.8-27B) + EntroMCU &amp; FWHT-1024</text>',
        f'  <text x="42" y="116" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.6">{esc(a_sub)}</text>',
        '  <rect x="42" y="128" width="540" height="86" rx="8" fill="#091120" stroke="#334155" stroke-width="1"/>',
        f'  <text x="56" y="147" fill="#e2e8f0" font-family="{FONT_SANS}" font-size="11.0" font-weight="700">128-Weight Hadamard-Ternary Group Packing (Lossless Bit-Exact, Saves {b["entromcu_saved_gb"]:.2f} GB on 27B)</text>',
        '  <rect x="56" y="157" width="320" height="18" rx="4" fill="url(#g35-ro)"/>',
        f'  <text x="384" y="170" fill="#fda4af" font-family="{FONT_MONO}" font-size="9.4">{esc(a_mlx_lbl)}</text>',
        '  <rect x="56" y="183" width="249" height="18" rx="4" fill="url(#g35-em)"/>',
        f'  <text x="313" y="196" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="9.4">{esc(a_ent_lbl)}</text>',
        f'  <text x="42" y="236" fill="#e2e8f0" font-family="{FONT_SANS}" font-size="11.0" font-weight="700">Window 12 (FWHT-1024 &gt;&gt; 5 + Antisym-40 T-MAC, 8 MULs/row) vs. Window 13 (MLX 2-Bit Dequant, 1,024 MULs/row)</text>',
    ]

    hw_bars = [
        (
            "Jetson AGX Thor DC Shunt (measured_j)",
            f"{thor_w12/1e3:.1f} vs {thor_w13/1e3:.1f} uJ/op ({thor_e_gain:.2f}x CpJ, {thor_lat_gain:.2f}x spd)",
            min(int(150 * thor_e_gain), 250),
            "url(#g35-em)",
            "#6ee7b7",
        ),
        (
            "AMD Ryzen AI Max+ 395 RAPL (reported_j)",
            f"{zen_w12/1e3:.1f} vs {zen_w13/1e3:.1f} uJ/op ({zen_e_gain:.2f}x CpJ)",
            min(int(150 * zen_e_gain), 250),
            "url(#g35-vi)",
            "#c4b5fd",
        ),
        (
            "Apple M5 Max Counter (reported_j)",
            f"{m5_w12/1e3:.2f} vs {m5_w13/1e3:.2f} uJ/op ({m5_e_gain:.2f}x CpJ)",
            min(int(150 * m5_e_gain), 250),
            "url(#g35-am)",
            "#fde68a",
        ),
    ]
    for i, (lbl, metric_txt, bar_w, grad, col) in enumerate(hw_bars):
        by = 250 + i * 52
        p += [
            f'  <text x="46" y="{by+13}" fill="#cbd5e1" font-family="{FONT_SANS}" font-size="10.6" font-weight="600">{esc(lbl)}</text>',
            f'  <rect x="46" y="{by+20}" width="{max(bar_w, 60)}" height="16" rx="4" fill="{grad}"/>',
            f'  <text x="{54 + max(bar_w, 60)}" y="{by+32}" fill="{col}" font-family="{FONT_MONO}" font-size="9.5" font-weight="700">{esc(metric_txt)}</text>',
        ]
    p.append(f'  <text x="42" y="428" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.3">{esc(a_foot)}</text>')

    # Panel B
    p += [
        '  <rect x="620" y="72" width="576" height="372" rx="12" fill="#0d1729" stroke="#a78bfa" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="638" y="98" fill="#c4b5fd" font-family="{FONT_SANS}" font-size="13.2" font-weight="800">B. Claim LA-40 · Neural System-1 Deciders vs. Compiled 4 KB System-0 LUT6</text>',
        f'  <text x="638" y="116" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.6">{esc(b_sub)}</text>',
    ]
    for idx, r in enumerate(sys1_rows):
        ry = 132 + idx * 72
        is_mol = "OpenIE" in r["system"]
        stroke_c = "#10b981" if is_mol else "#475569"
        bg_c = "#081c1b" if is_mol else "#091120"
        short_sys = r["system"].split(" (")[0]
        sub_sys = "(" + r["system"].split(" (", 1)[1] if " (" in r["system"] else ""
        m1 = f"Acc: {r['overall_acc_pct']}% (W2: {r['wave2_acc_pct']}%) | OOD False: {r['ood_false_action_pct']}% | RMS-CE: {r['rms_calibration_error_pct']}%"
        m2 = f"RAM: {r['active_weight_mb']} MB (64KB Watch: {r['watch_sram_pct_or_overflow']}) | {r['mean_latency_ms']} ms/q | {r['reported_mj_per_decision']} reported_mJ/q"
        p += [
            f'  <rect x="638" y="{ry}" width="540" height="62" rx="8" fill="{bg_c}" stroke="{stroke_c}" stroke-width="1.2"/>',
            f'  <text x="652" y="{ry+20}" fill="#f8fafc" font-family="{FONT_SANS}" font-size="11.0" font-weight="700">{esc(short_sys)} <tspan fill="#94a3b8" font-size="9.5" font-weight="500">{esc(sub_sys)}</tspan></text>',
            f'  <text x="652" y="{ry+40}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="9.5">{esc(m1)}</text>',
            f'  <text x="652" y="{ry+54}" fill="#93c5fd" font-family="{FONT_MONO}" font-size="9.2">{esc(m2)}</text>',
        ]
    p.append(f'  <text x="638" y="430" fill="#fde68a" font-family="{FONT_MONO}" font-size="9.5">{esc(b_foot)}</text>')

    # Panel C
    p += [
        '  <rect x="24" y="462" width="736" height="374" rx="12" fill="#0d1729" stroke="#10b981" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="488" fill="#6ee7b7" font-family="{FONT_SANS}" font-size="13.2" font-weight="800">C. Claim LA-41 · Post-LastExam Open Trilogy: ARC-AGI-2, MiniF2F-Lean4 &amp; CritPt Physics</text>',
        f'  <text x="42" y="506" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.5">100% Open Apache-2.0 / gated=False · SHA-256 Pinned · 0 Raw Prompt/Grid/Proof/Code Strings Persisted</text>',
        '  <rect x="42" y="518" width="700" height="24" rx="5" fill="#1e293b"/>',
        f'  <text x="52" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">Benchmark &amp; System</text>',
        f'  <text x="454" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">Solved</text>',
        f'  <text x="518" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">SelAcc</text>',
        f'  <text x="584" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">CHR@80</text>',
        f'  <text x="652" y="534" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">Unsound</text>',
    ]
    for idx, tr in enumerate(tril_rows[:8]):
        ty = 548 + idx * 32
        is_ref = "OpenIE" in tr["system"]
        row_bg = "#08201d" if is_ref else ("#0b1324" if idx % 2 == 0 else "#0f172a")
        txt_c = "#6ee7b7" if is_ref else "#e2e8f0"
        b_short = tr["benchmark"].split(" (")[0]
        s_short = (
            tr["system"]
            .replace("OpenIE ", "")
            .replace(" (Reference)", "")
            .replace(" (Portfolio + Lean 4.36.0 Soundness-Zero Gate)", " [Lean 4.36 Gate]")
            .replace(" (.lut6 Goal-AST Tactic Crystal + Lean 4.36.0 Gate)", " [.lut6 Crystal]")
        )
        label_str = f"{b_short}: {s_short}"[:54]
        chr_c = "#6ee7b7" if float(tr["confident_hallucination_rate_80_pct"]) == 0 else "#fda4af"
        uns_c = "#6ee7b7" if float(tr["unsound_commits_pct"]) == 0 else "#fda4af"
        fw_w = "700" if is_ref else "400"
        p += [
            f'  <rect x="42" y="{ty}" width="700" height="28" rx="4" fill="{row_bg}"/>',
            f'  <text x="50" y="{ty+18}" fill="{txt_c}" font-family="{FONT_MONO}" font-size="9.1" font-weight="{fw_w}">{esc(label_str)}</text>',
            f'  <text x="454" y="{ty+18}" fill="{txt_c}" font-family="{FONT_MONO}" font-size="9.4">{esc(tr["solved_pct"])}%</text>',
            f'  <text x="518" y="{ty+18}" fill="{txt_c}" font-family="{FONT_MONO}" font-size="9.4">{esc(tr["selective_committed_acc_pct"])}%</text>',
            f'  <text x="584" y="{ty+18}" fill="{chr_c}" font-family="{FONT_MONO}" font-size="9.4">{esc(tr["confident_hallucination_rate_80_pct"])}%</text>',
            f'  <text x="652" y="{ty+18}" fill="{uns_c}" font-family="{FONT_MONO}" font-size="9.4" font-weight="700">{esc(tr["unsound_commits_pct"])}%</text>',
        ]
    p.append(f'  <text x="42" y="822" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.3">{esc(c_foot)}</text>')

    # Panel D
    p += [
        '  <rect x="776" y="462" width="420" height="374" rx="12" fill="#0d1729" stroke="#f59e0b" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="794" y="488" fill="#fde68a" font-family="{FONT_SANS}" font-size="13.2" font-weight="800">D. Claim LA-19 · {n_thms} Proofs &amp; Soundness Ladder</text>',
        f'  <text x="794" y="506" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.5">{esc(d_sub)}</text>',
    ]
    ladder = [
        ("Level 3 · Soundness-Zero Kernel", "MiniF2F-Lean4 (Lean 4.36.0 + 6 Kernels)", "Necessary & Sufficient -> P[unsound] = 0", "#10b981"),
        ("Level 2 · Demonstration-Exact", "ARC-AGI-2 (C_train(f)=1 -> 5x LUT6)", "0 Train-Inconsistent Rules Admitted", "#22d3ee"),
        ("Level 1 · Necessary + Consensus", "LastExam / CritPt (Bounds + Dual-Model)", "0.00% CHR@80, 5.37x-9.49x Lower RMS-CE", "#a78bfa"),
        ("Level 0 · System-0 LUT6 + WASM", f"{wasm_kb:.2f} KB lut6_core.wasm (8 #![no_std] Targets)", "4 KB Page = 6.25% Watch SRAM, 0 FLOPs", "#f59e0b"),
    ]
    for i, (t1, t2, t3, col) in enumerate(ladder):
        ly = 522 + i * 72
        p += [
            f'  <rect x="794" y="{ly}" width="384" height="62" rx="8" fill="#091120" stroke="{col}" stroke-width="1.3"/>',
            f'  <text x="808" y="{ly+19}" fill="{col}" font-family="{FONT_SANS}" font-size="10.8" font-weight="800">{esc(t1)}</text>',
            f'  <text x="808" y="{ly+37}" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="9.4">{esc(t2)}</text>',
            f'  <text x="808" y="{ly+53}" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.1">{esc(t3)}</text>',
        ]
    p.append('  <text x="794" y="822" fill="#6ee7b7" font-family="ui-monospace, Menlo, monospace" font-size="9.3">TernaryHadamardArc.lean: Theorems 59..66 (0 sorry, 0 custom axioms)</text>')
    p.append("</svg>")

    svg_text = "\n".join(p) + "\n"
    ET.fromstring(svg_text)

    out_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "bonsai-ternary-decider-and-post-hle.svg"
    png_path = out_dir / "bonsai-ternary-decider-and-post-hle.png"
    svg_path.write_text(svg_text, encoding="utf-8")

    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-w", "2400", "-o", str(png_path), str(svg_path)], check=True)
    shutil.copy2(svg_path, root / "docs" / "keynote" / svg_path.name)
    print(f"Generated {svg_path} ({svg_path.stat().st_size} B) & {png_path} ({png_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
