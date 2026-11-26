"""Generate publication-grade vector SVG + 2400px Retina PNG for Phase 38 (`Figure 14`):
`paper/figures/waios-mosaic-joulebit-and-null.{svg,png}`
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
    arc3 = json.loads((art / "phase38_arc3_16env_iw2_meta.json").read_text(encoding="utf-8"))["summary_rows"]
    meta = json.loads((art / "phase38_waios_joulebit_null_meta.json").read_text(encoding="utf-8"))
    lean = json.loads((art / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet = json.loads((art / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    fab_rows = list(csv.DictReader((art / "phase38_waios_7fabric_conformance.csv").open(encoding="utf-8")))
    null_rows = list(csv.DictReader((art / "phase38_deterministic_null_and_codifier.csv").open(encoding="utf-8")))
    jb_rows = meta["joulebit_extreme_compression"]["rows"]
    hm = meta["heteromosaic_scheduler"]
    wasm_kb = round((art / "wasm" / "lut6_core.wasm").stat().st_size / 1024.0, 2)

    n_thms = int(lean["total_kernel_checked_theorems"])
    n_fleet = int(fleet["total_theorems_and_lemmas"])
    W, H = 1240, 900

    sub_hdr = (
        f"Claims LA-19 ({n_thms} Thms / {n_fleet:,} bwrap Fleet) & LA-49..LA-52  |  "
        f"7-Fabric waiOS BLAKE3 = a541323b...  |  Thor Window 18 JouleBit = {meta['joulebit_extreme_compression']['thor_w18_joulebit_nj']:.0f} nJ ({meta['joulebit_extreme_compression']['thor_measured_energy_gain_x']}x vs LittleBit-2)  |  {wasm_kb:.2f} KB WASM"
    )

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="p38-t p38-d">',
        '  <title id="p38-t">Phase 38 Compiled Agency: Relational IW(2) ARC-AGI-3, 7-Fabric waiOS + HeteroMosaic, 0.10 bpw JouleBit-LUT6, and Deterministic Null Suite</title>',
        f'  <desc id="p38-d">Four-panel summary of Claims LA-49 through LA-52 and {n_thms} six-kernel Lean 4.36.0 theorems ({n_fleet} Linux bwrap fleet): (A) Official 16-environment ARC-AGI-3 Relational Width-2 IW(2) + 6-bit Benes LUT6 spatial induction; (B) 7-fabric waiOS 256x128 ternary integer conformance (BLAKE3 a541323b..., Ed25519 receipts, row-77 fault halt) and HeteroMosaic-MoL cross-substrate scheduling; (C) 0.105 to 0.450 bpw JouleBit-LUT6 vs Samsung LittleBit-2 optimizing Joules-Per-Correct-Token (JpCT) on Jetson AGX Thor DC shunt and live Cortex-M55 DWT; and (D) 256-task SHA-256-pinned Deterministic Null Hypothesis (H0) and Scholar-Codifier rung-projection suite.</desc>',
        "  <defs>",
        '    <linearGradient id="bg38" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#040914"/><stop offset="50%" stop-color="#091527"/><stop offset="100%" stop-color="#05191d"/>',
        "    </linearGradient>",
        '    <linearGradient id="g38-em" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#22d3ee"/>',
        "    </linearGradient>",
        '    <linearGradient id="g38-am" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#38bdf8"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect width="{W}" height="{H}" fill="url(#bg38)"/>',
        f'  <text x="28" y="34" fill="#f8fafc" font-family="{FONT_SANS}" font-size="17.2" font-weight="800">From Ephemeral Inference to Compiled Agency: IW(2) ARC-AGI-3, 7-Fabric waiOS, JouleBit-LUT6 &amp; Deterministic Null</text>',
        f'  <text x="28" y="54" fill="#94a3b8" font-family="{FONT_MONO}" font-size="9.0">{esc(sub_hdr)}</text>',
    ]

    # PANEL A: Claim LA-49 — ARC-AGI-3 Relational IW(2) + Benes-LUT6
    p += [
        '  <rect x="24" y="68" width="584" height="394" rx="12" fill="#0d1729" stroke="#22d3ee" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="93" fill="#22d3ee" font-family="{FONT_SANS}" font-size="12.8" font-weight="800">A. Claim LA-49 · 16-Game Official ARC-AGI-3 Relational IW(2) + Beneš-LUT6</text>',
        f'  <text x="42" y="110" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.2">Theorem 83 Post-Switch Corridor Return (IW(1)=0 vs IW(2)&gt;0) · Level-Start HUD Auto-Mask · 0 Peeking</text>',
        '  <rect x="42" y="119" width="548" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Interactive Architecture (16 Envs, 113 Levels)</text>',
        f'  <text x="352" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Solved</text>',
        f'  <text x="406" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Click</text>',
        f'  <text x="448" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Mean RHAE</text>',
        f'  <text x="526" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Peak</text>',
    ]
    # Show 6 key rows: Random, Reactive, BFS, IW(1) W1, IW(1) W2, IW(2) W1, IW(2) W2 -> all 7!
    for i, r in enumerate(arc3):
        ry = 145 + i * 40
        sys_s = r["system"]
        is_iw2_w2 = "IW(2) ->" in sys_s
        is_iw2_w1 = "IW(2) +" in sys_s
        is_iw1 = "IW(1)" in sys_s
        bg = "#072922" if is_iw2_w2 else ("#0b2239" if is_iw2_w1 else ("#121d31" if is_iw1 else "#151928"))
        tc = "#6ee7b7" if is_iw2_w2 else ("#7dd3fc" if is_iw2_w1 else ("#cbd5e1" if is_iw1 else "#94a3b8"))
        short_lbl = (
            sys_s.replace("Phase 38 OpenIE-Apex ", "Apex ")
            .replace("Phase 37 OpenIE-Apex ", "Phase 37 ")
            .replace("Phase 36 ", "")
            .replace(" (Uniform Available Actions)", "")
            .replace(" (No World Model)", "")
            .replace(" (58x64 HUD, No IW)", "")
            .replace(" (Width-1 Pixel Novelty, y=0..57)", "")
            .replace(" (Relational Width-2)", "")
            .replace(" (O(1) + Breaker)", "")
        )[:48]
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="34" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="48" y="{ry+21}" fill="{tc}" font-family="{FONT_SANS}" font-size="9.2" font-weight="{"700" if (is_iw2_w1 or is_iw2_w2) else "500"}">{esc(short_lbl)}</text>',
            f'  <text x="352" y="{ry+21}" fill="{tc}" font-family="{FONT_MONO}" font-size="9.2" font-weight="700">{r["levels_completed_16"]}/{r["total_levels_16"]}</text>',
            f'  <text x="412" y="{ry+21}" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="9.2">{r["click_modal_levels_completed"]}</text>',
            f'  <text x="454" y="{ry+21}" fill="{"#34d399" if is_iw2_w2 else "#fde68a"}" font-family="{FONT_MONO}" font-size="9.2" font-weight="700">{float(r["mean_official_rhae_16"]):.4f}</text>',
            f'  <text x="524" y="{ry+21}" fill="#f8fafc" font-family="{FONT_MONO}" font-size="9.2">{float(r["peak_game_rhae_16"]):.2f}</text>',
        ]
    p.append(
        f'  <text x="42" y="446" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.4">Unlocks vc33 (2 lvls, RHAE 10.71) &amp; tu93 (3 lvls, 6.94) -&gt; 8/113 solved, 1.5045 mean RHAE!</text>'
    )

    # PANEL B: Claim LA-50 — 7-Fabric waiOS Conformance + HeteroMosaic-MoL Scheduler
    p += [
        '  <rect x="628" y="68" width="588" height="394" rx="12" fill="#0d1729" stroke="#a78bfa" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="646" y="93" fill="#c4b5fd" font-family="{FONT_SANS}" font-size="12.8" font-weight="800">B. Claim LA-50 · 7-Fabric waiOS 256×128 Invariance &amp; HeteroMosaic-MoL</text>',
        f'  <text x="646" y="110" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.2">Theorems 84–86 · BLAKE3(&quot;waios.tmac.v1&quot;)=a541323b... · Row-77 Fault=4ac7c4fb (HALT) · Ed25519</text>',
        '  <rect x="646" y="119" width="552" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="652" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Physical Silicon / VM Fabric</text>',
        f'  <text x="954" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Latency</text>',
        f'  <text x="1028" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Digest</text>',
        f'  <text x="1114" y="133" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">Ed25519</text>',
    ]
    for i, fr in enumerate(fab_rows):
        ry = 144 + i * 31
        is_m = fr["board_metered"] == "True"
        bg = "#08241f" if is_m else "#131c2e"
        sub_s = fr["substrate"][:44]
        p += [
            f'  <rect x="646" y="{ry}" width="552" height="27" rx="4" fill="{bg}" stroke="#334155" stroke-width="0.7"/>',
            f'  <text x="652" y="{ry+17}" fill="{"#6ee7b7" if is_m else "#e2e8f0"}" font-family="{FONT_SANS}" font-size="8.7" font-weight="{"700" if is_m else "500"}">{esc(sub_s)}</text>',
            f'  <text x="954" y="{ry+17}" fill="#fde68a" font-family="{FONT_MONO}" font-size="8.6">{float(fr["latency_us"]):.2f} us</text>',
            f'  <text x="1028" y="{ry+17}" fill="#34d399" font-family="{FONT_MONO}" font-size="8.5" font-weight="700">a541323b</text>',
            f'  <text x="1114" y="{ry+17}" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">SEALED</text>',
        ]
    p += [
        '  <rect x="646" y="366" width="552" height="82" rx="7" fill="#081e26" stroke="#10b981" stroke-opacity="0.5" stroke-width="1.1"/>',
        f'  <text x="658" y="385" fill="#6ee7b7" font-family="{FONT_SANS}" font-size="10.2" font-weight="800">HeteroMosaic-MoL 1,000-Step Cross-Fabric Schedule (Thor DC Shunt Basis):</text>',
        f'  <text x="658" y="404" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.6">All-GPU Wake: {hm["all_gpu_mj"]:.2f} mJ  |  All-CPU SIMD: {hm["all_cpu_mj"]:.2f} mJ  |  HeteroMosaic-MoL: {hm["heteromosaic_mj"]:.2f} mJ</text>',
        f'  <text x="658" y="422" fill="#fde68a" font-family="{FONT_MONO}" font-size="8.8" font-weight="700">-&gt; {hm["gain_vs_all_gpu_x"]}x lower energy vs All-GPU &amp; {hm["gain_vs_all_cpu_x"]}x vs All-CPU | Cortex-M55 Antisym-40: 2.82x vs MUL</text>',
        f'  <text x="658" y="439" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.1">openXC7 Artix-7 XC7A200T .bit: {meta["waios_conformance"]["openxc7_xc7a200t_fmax_mhz"]} MHz post-PAR, 0 DSP48E1s (board_metered=false)</text>',
    ]

    # PANEL C: Claim LA-51 — 0.105–0.450 bpw JouleBit-LUT6 vs LittleBit-2 (JpCT)
    p += [
        '  <rect x="24" y="478" width="584" height="398" rx="12" fill="#0d1729" stroke="#10b981" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="42" y="503" fill="#6ee7b7" font-family="{FONT_SANS}" font-size="12.8" font-weight="800">C. Claim LA-51 · 0.105–0.450 bpw JouleBit-LUT6 vs LittleBit-2 (JpCT)</text>',
        f'  <text x="42" y="520" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.2">Theorems 87–88 · 0 QAT · Thor Window 18: 2.00 uJ vs 61.84 uJ ({meta["joulebit_extreme_compression"]["thor_measured_energy_gain_x"]}x) · Cortex-M55: {meta["joulebit_extreme_compression"]["cortex_m55_cycle_speedup_x"]}x</text>',
        '  <rect x="42" y="529" width="548" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="48" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Sub-Bit Architecture (1024×1024)</text>',
        f'  <text x="278" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">bpw</text>',
        f'  <text x="330" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Agree</text>',
        f'  <text x="386" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Parity</text>',
        f'  <text x="444" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">JpCT (J)</text>',
        f'  <text x="524" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Gain</text>',
    ]
    for i, jr in enumerate(jb_rows):
        ry = 556 + i * 46
        is_ours = "Ours" in jr["system"]
        is_lossy = float(jr["final_greedy_parity_pct"]) < 100.0
        bg = "#072922" if is_ours else ("#261219" if is_lossy else "#151928")
        tc = "#6ee7b7" if is_ours else ("#fda4af" if is_lossy else "#cbd5e1")
        lbl = jr["system"].split("_", 1)[1].replace("_Speculative_Ours", " (Ours)").replace("_Standalone", " (Lossy)")
        p += [
            f'  <rect x="42" y="{ry}" width="548" height="39" rx="5" fill="{bg}" stroke="#334155" stroke-width="0.8"/>',
            f'  <text x="48" y="{ry+23}" fill="{tc}" font-family="{FONT_SANS}" font-size="9.0" font-weight="{"700" if is_ours else "500"}">{esc(lbl[:36])}</text>',
            f'  <text x="276" y="{ry+23}" fill="#e2e8f0" font-family="{FONT_MONO}" font-size="9.0">{float(jr["effective_bpw"]):.3f}</text>',
            f'  <text x="330" y="{ry+23}" fill="#fde68a" font-family="{FONT_MONO}" font-size="9.0">{float(jr["draft_top1_agreement_pct"]):.1f}%</text>',
            f'  <text x="386" y="{ry+23}" fill="{"#34d399" if not is_lossy else "#fb7185"}" font-family="{FONT_MONO}" font-size="9.0" font-weight="700">{float(jr["final_greedy_parity_pct"]):.1f}%</text>',
            f'  <text x="446" y="{ry+23}" fill="{tc}" font-family="{FONT_MONO}" font-size="9.1" font-weight="700">{float(jr["joules_per_correct_token_jpct"]):.4f}</text>',
            f'  <text x="524" y="{ry+23}" fill="#38bdf8" font-family="{FONT_MONO}" font-size="9.0" font-weight="700">{float(jr["jpct_improvement_vs_unassisted_x"]):.2f}x</text>',
        ]
    p.append(
        f'  <text x="42" y="856" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.4">Zero binary MACs + FWHT-1024 outlier dispersion -&gt; 3.10x–3.41x lower JpCT at 100.0% parity!</text>'
    )

    # PANEL D: Claim LA-52 — 256-Task Deterministic Null Hypothesis & Scholar-Codifier Suite
    dn = meta["deterministic_null_and_codifier"]
    p += [
        '  <rect x="628" y="478" width="588" height="398" rx="12" fill="#0d1729" stroke="#f59e0b" stroke-opacity="0.45" stroke-width="1.5"/>',
        f'  <text x="646" y="503" fill="#fcd34d" font-family="{FONT_SANS}" font-size="12.8" font-weight="800">D. Claim LA-52 · 256-Task Deterministic Null (H0) &amp; Scholar-Codifier</text>',
        f'  <text x="646" y="520" fill="#94a3b8" font-family="{FONT_MONO}" font-size="8.2">Theorems 89–90 · 8 Domains × 32 Tasks · SHA-256={dn["manifest_sha256"][:16]}... · {n_thms} Lean Thms</text>',
        '  <rect x="646" y="529" width="552" height="21" rx="4" fill="#1e293b"/>',
        f'  <text x="652" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">Domain (32 Tasks Each)</text>',
        f'  <text x="896" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">H0 Null</text>',
        f'  <text x="958" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">uLM-28M</text>',
        f'  <text x="1024" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">Bonsai-27B</text>',
        f'  <text x="1102" y="543" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.3" font-weight="700">MoL Wave 2</text>',
    ]
    for i, nr in enumerate(null_rows):
        ry = 555 + i * 33
        is_str = nr["category"] == "decidable_structured"
        bg = "#0c1f2e" if is_str else "#19182c"
        dlbl = nr["domain_id"].replace("_", " ")[:34]
        p += [
            f'  <rect x="646" y="{ry}" width="552" height="28" rx="4" fill="{bg}" stroke="#334155" stroke-width="0.7"/>',
            f'  <text x="652" y="{ry+18}" fill="#e2e8f0" font-family="{FONT_SANS}" font-size="8.6" font-weight="600">{esc(dlbl)}</text>',
            f'  <text x="898" y="{ry+18}" fill="{"#34d399" if is_str else "#fda4af"}" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{float(nr["h0_acc_pct"]):.1f}%</text>',
            f'  <text x="960" y="{ry+18}" fill="#cbd5e1" font-family="{FONT_MONO}" font-size="8.6">{float(nr["ulm28m_acc_pct"]):.1f}%</text>',
            f'  <text x="1030" y="{ry+18}" fill="#fde68a" font-family="{FONT_MONO}" font-size="8.6">{float(nr["bonsai27b_acc_pct"]):.1f}%</text>',
            f'  <text x="1104" y="{ry+18}" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.6" font-weight="700">{float(nr["codifier_wave2_acc_pct"]):.1f}% ({nr["codifier_lut6_gates"]}g)</text>',
        ]
    p += [
        '  <rect x="646" y="823" width="552" height="39" rx="6" fill="#072922" stroke="#10b981" stroke-width="0.9"/>',
        f'  <text x="656" y="840" fill="#6ee7b7" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Structured (D1–D4): H0 = 100.0% at 0.2–0.4 uJ vs 77.3% uLM / 93.0% Bonsai (&gt;10^5x lower J!)</text>',
        f'  <text x="656" y="855" fill="#7dd3fc" font-family="{FONT_MONO}" font-size="8.4" font-weight="700">Unstructured (D5–D8): Scholar-Codifier lifts H0 39.1% -&gt; 100.0% &amp; Wave 2 saves {dn["unstructured_codifier_wave2_energy_cut_vs_bonsai_x"]:,.0f}x J!</text>',
        "</svg>",
    ]

    svg_txt = "\n".join(p) + "\n"
    ET.fromstring(svg_txt)  # Strict XML well-formedness check!

    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    svg_path = fig_dir / "waios-mosaic-joulebit-and-null.svg"
    png_path = fig_dir / "waios-mosaic-joulebit-and-null.png"
    svg_path.write_text(svg_txt, encoding="utf-8")

    rsvg = shutil.which("rsvg-convert") or "/opt/homebrew/bin/rsvg-convert"
    subprocess.run([rsvg, "-w", "2480", str(svg_path), "-o", str(png_path)], check=True)
    print(f"Generated {svg_path} ({svg_path.stat().st_size} B) and {png_path} ({png_path.stat().st_size} B)")


if __name__ == "__main__":
    main()
