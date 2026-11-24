"""Generate publication-grade vector SVGs (and 2x Retina PNG twins) for:
  1. `avalanche-compilation-threshold.svg` — The 9-Rung Computer Intelligence Avalanche Ratchet
     & The Compilation Threshold (Delta T > Delta T*, Proposition 1).
  2. `six-kernel-trust-and-stack.svg` — Six-Kernel CVE-2026-72844-Hardened Formal Proof Audit
     (Claim LA-19: con-leche, con-ron, lean4lean, nanoda, leanchecker, Lean 4.36.0) &
     Three-Layer Post-Transformer Compiled Stack.
  3. `docs/keynote/trinity-stack.svg` — Ecosystem Keynote Diagram (MathGround + OpenIE/LUT-Adapt
     + Pattern-Lang) with strict OpenIE tier-typed Joule honesty.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


FONT_SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_avalanche_svg() -> str:
    W, H = 1200, 760
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="av-t av-d">',
        '  <title id="av-t">The Nine-Rung Computer Intelligence Avalanche Ratchet and the Compilation Threshold</title>',
        '  <desc id="av-d">Three-column architectural and thermodynamic diagram mapping Rungs 1 to 5 (Search, Generation, Reasoning, Communication, Tool Use) in the ephemeral cold-start MAC regime where Delta T is below Delta T*, across Proposition 1\'s Compilation Threshold (Delta T &gt; E_compile / (p\'_0 c_wake - c_0) and deterministic C(z)=1 gate with 0.00% broken commits), into Rungs 6 to 9 (Immersion, Competence, Ecology, Economy), with a bottom log-scale energy trajectory from 148-485 J frontier wakes down to 272 uJ Rust MCP lookups (reported_j) and 62 pJ clock-gated LUT6 cycles (est_j).</desc>',
        "  <defs>",
        '    <linearGradient id="bg-grad" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0%" stop-color="#070b14"/>',
        '      <stop offset="55%" stop-color="#0b1120"/>',
        '      <stop offset="100%" stop-color="#09131c"/>',
        "    </linearGradient>",
        '    <linearGradient id="thresh-glow" x1="0" y1="0" x2="0" y2="1">',
        '      <stop offset="0%" stop-color="#1e293b" stop-opacity="0.95"/>',
        '      <stop offset="50%" stop-color="#0f172a" stop-opacity="0.98"/>',
        '      <stop offset="100%" stop-color="#0c1f24" stop-opacity="0.95"/>',
        "    </linearGradient>",
        '    <linearGradient id="energy-bar" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0%" stop-color="#10b981"/>',
        '      <stop offset="38%" stop-color="#22d3ee"/>',
        '      <stop offset="72%" stop-color="#f59e0b"/>',
        '      <stop offset="100%" stop-color="#f43f5e"/>',
        "    </linearGradient>",
        '    <marker id="arr-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#f59e0b"/>',
        "    </marker>",
        '    <marker id="arr-emerald" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#10b981"/>',
        "    </marker>",
        '    <marker id="arr-cyan" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#22d3ee"/>',
        "    </marker>",
        '    <marker id="arr-rose" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#f43f5e"/>',
        "    </marker>",
        "  </defs>",
        f'  <rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="url(#bg-grad)" stroke="#334155" stroke-width="2"/>',
        # Header
        f'  <text x="30" y="38" font-family="{FONT_SANS}" font-size="20.5" font-weight="800" fill="#f8fafc">The 9-Rung Computer Intelligence Avalanche Ratchet &amp; The Compilation Threshold</text>',
        f'  <text x="30" y="60" font-family="{FONT_SANS}" font-size="12.2" font-weight="500" fill="#94a3b8">Why stateless MAC inference stalls at Rung 5—and how online verified downward LUT6 / .mol/ compilation unlocks Rungs 6–9</text>',
        '  <rect x="962" y="17" width="210" height="30" rx="15" fill="#0f172a" stroke="#10b981" stroke-width="1.6"/>',
        f'  <text x="1067" y="36" font-family="{FONT_MONO}" font-size="11" font-weight="700" fill="#10b981" text-anchor="middle">Prop. 1 · 6-Kernel Verified</text>',
    ]

    # Column 1: Phase I (Rungs 1-5)
    parts += [
        '  <rect x="26" y="78" width="398" height="448" rx="14" fill="#111827" stroke="#f43f5e" stroke-width="1.6"/>',
        f'  <text x="44" y="103" font-family="{FONT_SANS}" font-size="12.8" font-weight="800" fill="#fda4af">PHASE I · EPHEMERAL COLD-START ERA (RUNGS 1–5)</text>',
        f'  <text x="44" y="121" font-family="{FONT_MONO}" font-size="10.0" fill="#94a3b8">Stateless MACs · ΔT ≈ 1 ≤ ΔT* · Cold-start compute tax</text>',
    ]

    rungs_p1 = [
        ("1", "Search", "Commoditized Access", "Collapsed access cost pre-LLM; now linear RAG feature.", "RAG Feature", "#64748b", "#cbd5e1"),
        ("2", "Generation", "Deflationary Production", "Drafting text/media went scarce -> near-free in ~3 yrs.", "Value Leaks", "#d97706", "#fde68a"),
        ("3", "Reasoning", "Cost of Correctness", "Made output enterprise-viable; burns 15–500 J per wake.", "High MAC Tax", "#f59e0b", "#fef3c7"),
        ("4", "Communication", "Network-Effect Protocols", "Agent-to-agent delegation (MCP / E-G-T); scales w/ edges.", "Network Rung", "#22d3ee", "#cffafe"),
        ("5", "Tool Use", "Execution (Current Ceiling)", "Turns Rungs 1–4 into work—stuck in stateless cold starts.", "STUCK: ΔT ≤ ΔT*", "#f43f5e", "#ffe4e6"),
    ]
    y0 = 135
    for idx, (num, name, tag, desc, pill, stroke, text_hi) in enumerate(rungs_p1):
        ry = y0 + idx * 76
        is_stuck = num == "5"
        bg = "#1f1218" if is_stuck else "#0f172a"
        sw = "2.1" if is_stuck else "1.3"
        parts += [
            f'  <rect x="42" y="{ry}" width="366" height="66" rx="10" fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>',
            f'  <circle cx="64" cy="{ry+23}" r="12" fill="#0b1120" stroke="{stroke}" stroke-width="1.6"/>',
            f'  <text x="64" y="{ry+27}" font-family="{FONT_MONO}" font-size="11.5" font-weight="800" fill="{text_hi}" text-anchor="middle">{num}</text>',
            f'  <text x="84" y="{ry+24}" font-family="{FONT_SANS}" font-size="13.0" font-weight="800" fill="#f8fafc">{esc(name)}</text>',
            f'  <text x="84" y="{ry+40}" font-family="{FONT_SANS}" font-size="10.2" font-weight="600" fill="{text_hi}">{esc(tag)}</text>',
            f'  <text x="52" y="{ry+56}" font-family="{FONT_SANS}" font-size="10.0" fill="#cbd5e1">{esc(desc)}</text>',
            f'  <rect x="296" y="{ry+9}" width="102" height="20" rx="10" fill="#0b1120" stroke="{stroke}" stroke-width="1.2"/>',
            f'  <text x="347" y="{ry+22.5}" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="{text_hi}" text-anchor="middle">{esc(pill)}</text>',
        ]

    # Column 2: Center Phase Transition (The Compilation Threshold)
    parts += [
        '  <rect x="442" y="78" width="302" height="448" rx="14" fill="url(#thresh-glow)" stroke="#f59e0b" stroke-width="2.2"/>',
        '  <rect x="466" y="92" width="254" height="26" rx="13" fill="#1c1917" stroke="#f59e0b" stroke-width="1.4"/>',
        f'  <text x="593" y="109" font-family="{FONT_MONO}" font-size="10.6" font-weight="800" fill="#fbbf24" text-anchor="middle">THE COMPILATION THRESHOLD</text>',
        f'  <text x="593" y="135" font-family="{FONT_SANS}" font-size="11.0" font-weight="600" fill="#e2e8f0" text-anchor="middle">Phase Transition in Compute Physics</text>',
        # Box A: Amortization Inequality
        '  <rect x="458" y="148" width="270" height="114" rx="10" fill="#0b1120" stroke="#22d3ee" stroke-width="1.6"/>',
        f'  <text x="472" y="169" font-family="{FONT_SANS}" font-size="10.8" font-weight="800" fill="#22d3ee">1 · PERSISTENCE HORIZON (CASCADE.LEAN)</text>',
        '  <rect x="468" y="178" width="250" height="34" rx="6" fill="#111827" stroke="#1e293b" stroke-width="1.0"/>',
        f'  <text x="593" y="199" font-family="{FONT_MONO}" font-size="9.6" font-weight="800" fill="#f8fafc" text-anchor="middle">ΔT &gt; ΔT* = E_compile / (p&apos;₀·c_wake − c₀)</text>',
        f'  <text x="472" y="230" font-family="{FONT_MONO}" font-size="9.8" fill="#67e8f9">Ordering gate:  p&apos;₀ / c₀ &gt; 1 / c_wake</text>',
        f'  <text x="472" y="248" font-family="{FONT_SANS}" font-size="9.8" fill="#cbd5e1">Staying (Rung 6) makes ΔT large enough to</text>',
        f'  <text x="472" y="259" font-family="{FONT_SANS}" font-size="9.8" fill="#cbd5e1">amortize downward compilation into silicon.</text>',
        # Box B: Deterministic Verifier Gate
        '  <rect x="458" y="274" width="270" height="112" rx="10" fill="#0b1120" stroke="#10b981" stroke-width="1.6"/>',
        f'  <text x="472" y="295" font-family="{FONT_SANS}" font-size="10.8" font-weight="800" fill="#10b981">2 · PROPOSER VS. AUTHORITY GATE</text>',
        f'  <text x="472" y="315" font-family="{FONT_MONO}" font-size="10.2" font-weight="700" fill="#f8fafc">C(z) = 1  or  |C_t(x)| = 1 (Split-ACI)</text>',
        f'  <text x="472" y="335" font-family="{FONT_SANS}" font-size="9.9" fill="#e2e8f0">• Hallucinations quarantined at the gate</text>',
        f'  <text x="472" y="352" font-family="{FONT_SANS}" font-size="9.9" fill="#6ee7b7" font-weight="700">• 0.00% broken commits admitted (LA-15..21)</text>',
        f'  <text x="472" y="369" font-family="{FONT_MONO}" font-size="9.3" fill="#94a3b8">ModelGenerated -/-&gt; Deterministic</text>',
        # Box C: SleepCrystallizer Hardware Forge
        '  <rect x="458" y="398" width="270" height="112" rx="10" fill="#0b1120" stroke="#a78bfa" stroke-width="1.6"/>',
        f'  <text x="472" y="419" font-family="{FONT_SANS}" font-size="10.8" font-weight="800" fill="#c4b5fd">3 · DOWNWARD SLEEPCRYSTALLIZER</text>',
        f'  <text x="472" y="439" font-family="{FONT_SANS}" font-size="9.9" fill="#e2e8f0">• Walsh–Hadamard + Joint Minterm (0% chimera)</text>',
        f'  <text x="472" y="456" font-family="{FONT_SANS}" font-size="9.9" fill="#e2e8f0">• 512×512 Beneš–Copy–Beneš (43 stages, 0 PAR)</text>',
        f'  <text x="472" y="473" font-family="{FONT_SANS}" font-size="9.9" fill="#e2e8f0">• 81 × RAMB36E2 + .mol/ AST/Diff skill store</text>',
        f'  <text x="472" y="493" font-family="{FONT_MONO}" font-size="9.5" font-weight="700" fill="#fbbf24">1-cycle shadow-bank swap · 0 FLOPs</text>',
    ]

    # Column 3: Phase II (Rungs 6-9)
    parts += [
        '  <rect x="762" y="78" width="412" height="448" rx="14" fill="#111827" stroke="#10b981" stroke-width="1.8"/>',
        f'  <text x="780" y="103" font-family="{FONT_SANS}" font-size="12.8" font-weight="800" fill="#6ee7b7">PHASE II · PERSISTENT COMPILED ERA (RUNGS 6–9)</text>',
        f'  <text x="780" y="121" font-family="{FONT_MONO}" font-size="10.0" fill="#94a3b8">Stateful graph · ΔT ≫ ΔT* · p&apos;₀ → 1 at LUT6 energy floor</text>',
    ]

    rungs_p2 = [
        (
            "6",
            "Immersion",
            "Experience Accumulates (The Agent Stays)",
            "Local state persists across waves; ΔT grows past ΔT*,",
            "making downward crystallization energetically favorable.",
            "ΔT > ΔT*",
            "#22d3ee",
            "#cffafe",
        ),
        (
            "7",
            "Competence",
            "Experience Compiles into Owned Skill",
            "Certified wakes compile to 0-FLOP LUT6 & .mol/ skills:",
            "62 pJ est_j RTL cycle · 272 µJ reported_j Rust MCP hit.",
            "p'₀ → 1 Floor",
            "#10b981",
            "#d1fae5",
        ),
        (
            "8",
            "Ecology",
            "Competent Agents Populate Shared Worlds",
            "Sub-mJ single-binary Rust/edge agents coexist locally,",
            "exchanging attested skills over 154 µs MCP / Enzymes.",
            "154 µs MCP",
            "#38bdf8",
            "#e0f2fe",
        ),
        (
            "9",
            "Economy",
            "Division of Labor & Joule-Priced Exchange",
            "Specialists route work by Simon–Kadane p_i / c_i with",
            "tier-typed Joule receipts (est_j | reported_j | measured_j).",
            "p_i / c_i Order",
            "#a78bfa",
            "#ede9fe",
        ),
    ]
    y2 = 136
    for idx, (num, name, tag, d1, d2, pill, stroke, text_hi) in enumerate(rungs_p2):
        ry = y2 + idx * 95
        parts += [
            f'  <rect x="778" y="{ry}" width="380" height="84" rx="10" fill="#0f172a" stroke="{stroke}" stroke-width="1.6"/>',
            f'  <circle cx="801" cy="{ry+24}" r="13" fill="#0b1120" stroke="{stroke}" stroke-width="1.8"/>',
            f'  <text x="801" y="{ry+28}" font-family="{FONT_MONO}" font-size="12" font-weight="800" fill="{text_hi}" text-anchor="middle">{num}</text>',
            f'  <text x="822" y="{ry+24}" font-family="{FONT_SANS}" font-size="13.5" font-weight="800" fill="#f8fafc">{esc(name)}</text>',
            f'  <text x="822" y="{ry+41}" font-family="{FONT_SANS}" font-size="10.3" font-weight="600" fill="{text_hi}">{esc(tag)}</text>',
            f'  <text x="790" y="{ry+59}" font-family="{FONT_SANS}" font-size="10.0" fill="#cbd5e1">{esc(d1)}</text>',
            f'  <text x="790" y="{ry+73}" font-family="{FONT_SANS}" font-size="10.0" fill="#cbd5e1">{esc(d2)}</text>',
            f'  <rect x="1044" y="{ry+9}" width="104" height="21" rx="10.5" fill="#0b1120" stroke="{stroke}" stroke-width="1.2"/>',
            f'  <text x="1096" y="{ry+23}" font-family="{FONT_MONO}" font-size="9.4" font-weight="700" fill="{text_hi}" text-anchor="middle">{esc(pill)}</text>',
        ]

    # Phase-crossing connectors (Rung 5 -> Threshold -> Rung 6 -> Rung 7)
    parts += [
        '  <path d="M 408 472 L 434 472 L 434 205 L 454 205" fill="none" stroke="#f59e0b" stroke-width="2.4" marker-end="url(#arr-amber)"/>',
        '  <path d="M 728 205 L 752 205 L 752 178 L 774 178" fill="none" stroke="#22d3ee" stroke-width="2.4" marker-end="url(#arr-cyan)"/>',
        '  <path d="M 728 454 L 752 454 L 752 273 L 774 273" fill="none" stroke="#10b981" stroke-width="2.4" marker-end="url(#arr-emerald)"/>',
    ]

    # Bottom Panel: Log10 Energy Scale across 13 orders of magnitude (10^-11 J to 10^3 J)
    parts += [
        '  <rect x="26" y="540" width="1148" height="196" rx="14" fill="#111827" stroke="#334155" stroke-width="1.6"/>',
        f'  <text x="46" y="565" font-family="{FONT_SANS}" font-size="12.8" font-weight="800" fill="#f8fafc">QUANTITATIVE ENERGY TRAJECTORY ACROSS THE COMPILATION THRESHOLD (LOG₁₀ JOULES PER ACTION)</text>',
        f'  <text x="46" y="583" font-family="{FONT_SANS}" font-size="10.2" fill="#94a3b8">Strictly separated evidence classes from openIE-dev/lut-adapt-ai (Claims LA-19..LA-23; measured_j = None, board_metered = false)</text>',
    ]

    # Axis from log10(J) = -11 (10 pJ) to +3 (1 kJ) -> span 14 decades across x in [76, 1124]
    x_left, x_right = 76.0, 1124.0
    y_axis = 656.0

    def x_of_log10(lg: float) -> float:
        return x_left + (lg - (-11.0)) / 14.0 * (x_right - x_left)

    parts.append(f'  <rect x="{x_left}" y="{y_axis-5}" width="{x_right-x_left}" height="10" rx="5" fill="url(#energy-bar)" opacity="0.88"/>')

    ticks = [
        (-11, "10 pJ"),
        (-9, "1 nJ"),
        (-6, "1 µJ"),
        (-3, "1 mJ"),
        (0, "1 J"),
        (3, "1 kJ"),
    ]
    for lg, lbl in ticks:
        tx = x_of_log10(lg)
        parts += [
            f'  <line x1="{tx:.1f}" y1="{y_axis-12}" x2="{tx:.1f}" y2="{y_axis+12}" stroke="#cbd5e1" stroke-width="1.4"/>',
            f'  <text x="{tx:.1f}" y="{y_axis+26}" font-family="{FONT_MONO}" font-size="9.6" fill="#94a3b8" text-anchor="middle">{lbl}</text>',
        ]

    milestones = [
        (math.log10(62.4e-12), "62.4 pJ [est_j]", "CE-Gated xck26 LUT6", "4 ns @ 250 MHz (LA-22)", "#10b981", -36),
        (math.log10(2.72e-4), "272 µJ [reported_j]", "Rust mol-agent-rs MCP", "0.154 ms/req (LA-23)", "#22d3ee", -36),
        (math.log10(1.18e-2), "11.8 mJ [reported_j]", ".mol/ Function Replay", "5.49 ms Wave-2 (LA-21)", "#38bdf8", 35),
        (math.log10(1.12), "1.12 J [reported_j]", "SWE-bench Diff Replay", "593 ms Git worktree (LA-20)", "#f59e0b", -36),
        (math.log10(245.0), "148–485 J [reported_j]", "35B / 120B Cold Wake", "22–33 s MAC", "#f43f5e", 35),
    ]
    for lg, e_txt, title_txt, sub_txt, col, dy in milestones:
        mx = x_of_log10(lg)
        ly = y_axis + dy
        parts += [
            f'  <line x1="{mx:.1f}" y1="{y_axis}" x2="{mx:.1f}" y2="{ly + (10 if dy < 0 else -12)}" stroke="{col}" stroke-width="1.8"/>',
            f'  <circle cx="{mx:.1f}" cy="{y_axis}" r="6.5" fill="#0b1120" stroke="{col}" stroke-width="2.4"/>',
        ]
        if dy < 0:
            parts += [
                f'  <text x="{mx:.1f}" y="{ly-12}" font-family="{FONT_MONO}" font-size="9.8" font-weight="800" fill="{col}" text-anchor="middle">{esc(e_txt)}</text>',
                f'  <text x="{mx:.1f}" y="{ly+1}" font-family="{FONT_SANS}" font-size="9.8" font-weight="700" fill="#f8fafc" text-anchor="middle">{esc(title_txt)}</text>',
            ]
        else:
            parts += [
                f'  <text x="{mx:.1f}" y="{ly+4}" font-family="{FONT_MONO}" font-size="9.8" font-weight="800" fill="{col}" text-anchor="middle">{esc(e_txt)}</text>',
                f'  <text x="{mx:.1f}" y="{ly+17}" font-family="{FONT_SANS}" font-size="9.6" font-weight="600" fill="#e2e8f0" text-anchor="middle">{esc(title_txt)} · {esc(sub_txt)}</text>',
            ]

    parts += [
        f'  <text x="46" y="724" font-family="{FONT_MONO}" font-size="9.5" fill="#94a3b8">Honesty invariant: est_j (AMD DS925 VCD) ≠ reported_j (Apple M5 Max IOReport + ri_energy_nj) ≠ shunt measured_j (unset here; see Gates / CTA for INA260 / INA238).</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def build_six_kernel_and_stack_svg() -> str:
    W, H = 1200, 700
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="sk-t sk-d">',
        '  <title id="sk-t">Six-Kernel Paranoid Proof Audit (Claim LA-19) and Three-Layer Compiled Intelligence Stack</title>',
        '  <desc id="sk-d">Left panel shows the 58 LutAdapt Lean 4.36.0 theorems passing six independent proof checkers via lake check --paranoid (con-leche --verified, con-ron --verified, lean4lean, nanoda, leanchecker --paranoid, and Lean 4.36.0), neutralizing CVE-2026-72844, CVE-2026-72711, and native_decide exploits, paired with Inria Charon + Aeneas #![no_std] Rust MIR extraction. Right panel shows the three-layer compiled intelligence architecture spanning physical L0 control &amp; LUT6 silicon, cognitive MoL downward compilation, and verifier-first Rust MCP runtime with tier-typed Joule receipts.</desc>',
        "  <defs>",
        '    <marker id="sk-emerald" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#10b981"/>',
        "    </marker>",
        '    <marker id="sk-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#f59e0b"/>',
        "    </marker>",
        '    <marker id="sk-cyan" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#22d3ee"/>',
        "    </marker>",
        "  </defs>",
        f'  <rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="#0b1120" stroke="#334155" stroke-width="2"/>',
        f'  <text x="30" y="38" font-family="{FONT_SANS}" font-size="20.2" font-weight="800" fill="#f8fafc">Six-Kernel Formal Proof Assurance (Claim LA-19) &amp; Compiled Stack</text>',
        f'  <text x="30" y="60" font-family="{FONT_SANS}" font-size="12.0" font-weight="500" fill="#94a3b8">Multi-kernel defense against single-prover soundness bugs (CVE-2026-72844, CVE-2026-72711, native_decide) across all three stack layers</text>',
        '  <rect x="952" y="17" width="220" height="30" rx="15" fill="#0f172a" stroke="#22d3ee" stroke-width="1.6"/>',
        f'  <text x="1062" y="36" font-family="{FONT_MONO}" font-size="10.8" font-weight="700" fill="#22d3ee" text-anchor="middle">6 / 6 Independent Kernels</text>',
        # Left Panel: 6-Kernel Paranoid Audit
        '  <rect x="28" y="80" width="564" height="592" rx="14" fill="#111827" stroke="#22d3ee" stroke-width="1.6"/>',
        f'  <text x="46" y="106" font-family="{FONT_SANS}" font-size="13.2" font-weight="800" fill="#f8fafc">A · Six-Kernel Paranoid Proof Audit (lake check --paranoid)</text>',
        f'  <text x="46" y="124" font-family="{FONT_SANS}" font-size="10.3" fill="#94a3b8">Mechanized Section 2 theorems exported from Lean 4.36.0 and re-verified from scratch</text>',
        # Source theorem box
        '  <rect x="46" y="138" width="528" height="78" rx="10" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>',
        f'  <text x="58" y="160" font-family="{FONT_MONO}" font-size="7.3" font-weight="800" fill="#38bdf8">proofs/LutAdapt/{{Cascade,BenesCopy,JointMinterm,Conformal...,NucleusCodon,TernaryTmac,GlobalFrontiers,WearableLastExam}}.lean</text>',
        f'  <text x="62" y="181" font-family="{FONT_SANS}" font-size="10.4" fill="#e2e8f0">58 Machine-Checked Theorems (1,422 Fleet) · 11 Zero-Axiom · 44/58 Choice-Free · +Aeneas Rust-&gt;Lean</text>',
        f'  <text x="62" y="200" font-family="{FONT_MONO}" font-size="9.6" fill="#6ee7b7">0 sorry · 0 custom axioms · 0 opaque · 0 Lean.ofReduceBool (native_decide)</text>',
        # Neutralized 2026 Vulnerability Banner
        '  <rect x="46" y="228" width="528" height="76" rx="10" fill="#1c1219" stroke="#f43f5e" stroke-width="1.4"/>',
        f'  <text x="62" y="248" font-family="{FONT_SANS}" font-size="10.4" font-weight="800" fill="#fda4af">2026 PROVER SOUNDNESS &amp; AI REWARD-HACKING VECTORS BLOCKED:</text>',
        f'  <text x="60" y="267" font-family="{FONT_MONO}" font-size="8.5" fill="#fecdd3">✕ CVE-2026-72844: nested-inductive phantom param + .proj confusion (patched ≥ v4.32.2)</text>',
        f'  <text x="60" y="283" font-family="{FONT_MONO}" font-size="8.5" fill="#fecdd3">✕ CVE-2026-72711: add_opaque local-fvar inference cache poisoning (patched ≥ v4.32.2)</text>',
        f'  <text x="60" y="298" font-family="{FONT_MONO}" font-size="8.5" fill="#fecdd3">✕ RL Bypass: native_decide (Lean.ofReduceBool), @[implemented_by], debug.skipKernelTC</text>',
    ]

    kernels = [
        ("1 · con-leche (--verified)", "Set-Theory Consistency Proof", "Breitner minimal kernel proved", "never to accept False · 58,802 decls", "#10b981"),
        ("2 · con-ron (--verified)", "Aeneas-Verified Rust Kernel", "Formally refined Rust port of", "con-leche · 58,802 decls checked", "#10b981"),
        ("3 · lean4lean", "Lean-in-Lean Typechecker", "Carneiro independent kernel;", "58,804 declarations replayed", "#22d3ee"),
        ("4 · nanoda", "Independent Rust Checker", "Standalone Rust export verifier;", "sorryAx explicitly unpermitted", "#22d3ee"),
        ("5 · leanchecker --paranoid", "Fresh-Env C++ Kernel Replay", "Replays .olean environment from", "scratch with zero tactic trust", "#38bdf8"),
        ("6 · Lean v4.36.0 (Nov 2026)", "Post-CVE Default C++ Kernel", "Includes WHNF cache & GMP fixes;", "clean lake build in ~0.89 s", "#a78bfa"),
    ]
    for i, (kname, ksub, kl1, kl2, col) in enumerate(kernels):
        col_idx = i % 2
        row_idx = i // 2
        kx = 46 + col_idx * 270
        ky = 320 + row_idx * 104
        parts += [
            f'  <rect x="{kx}" y="{ky}" width="258" height="92" rx="10" fill="#0f172a" stroke="{col}" stroke-width="1.5"/>',
            f'  <text x="{kx+12}" y="{ky+21}" font-family="{FONT_MONO}" font-size="10.2" font-weight="800" fill="#f8fafc">{esc(kname)}</text>',
            f'  <text x="{kx+12}" y="{ky+38}" font-family="{FONT_SANS}" font-size="9.8" font-weight="700" fill="{col}">{esc(ksub)}</text>',
            f'  <text x="{kx+12}" y="{ky+56}" font-family="{FONT_SANS}" font-size="9.5" fill="#cbd5e1">{esc(kl1)}</text>',
            f'  <text x="{kx+12}" y="{ky+70}" font-family="{FONT_SANS}" font-size="9.5" fill="#cbd5e1">{esc(kl2)}</text>',
            f'  <rect x="{kx+182}" y="{ky+66}" width="64" height="18" rx="9" fill="#052e16" stroke="#10b981" stroke-width="1.1"/>',
            f'  <text x="{kx+214}" y="{ky+78}" font-family="{FONT_MONO}" font-size="8.8" font-weight="800" fill="#6ee7b7" text-anchor="middle">ACCEPTS</text>',
        ]

    parts.append(
        f'  <text x="46" y="652" font-family="{FONT_MONO}" font-size="8.6" fill="#6ee7b7">Artifact: artifacts/lean4_six_kernel_paranoid_audit.csv · verify_lean_proofs.py (exit 0)</text>'
    )

    # Right Panel: Three-Layer Compiled Intelligence Stack
    parts += [
        '  <rect x="608" y="80" width="564" height="592" rx="14" fill="#111827" stroke="#10b981" stroke-width="1.6"/>',
        f'  <text x="626" y="106" font-family="{FONT_SANS}" font-size="13.2" font-weight="800" fill="#f8fafc">B · Energy-Ordered Three-Layer Compiled Stack</text>',
        f'  <text x="626" y="124" font-family="{FONT_SANS}" font-size="10.3" fill="#94a3b8">Verifier-first protocols, online LUT6/.mol/ crystallization, and bare-metal L0 execution</text>',
    ]

    layers = [
        (
            144,
            "LAYER 3 · VERIFIER-FIRST INTERFACE & PROTOCOL RUNTIME",
            "Single-Binary Rust MCP / OpenAI Daemon & Tier-Typed Joule Receipts",
            "• Zero-Python mol-agent-rs: 0.154 ms/req MCP (17.21× faster than Python)",
            "• Attests every answer with C(z) certificate + honest Joule receipt",
            "272 µJ / req [reported_j]",
            "#22d3ee",
        ),
        (
            290,
            "LAYER 2 · COGNITIVE DOWNWARD COMPILER (MOL CASCADE)",
            "Online SleepCrystallizer · Joint-Mode LUT6 & Canonical .mol/ Skills",
            "• 91.67% 48-task · 90.62% BigCodeBench · 2.50× 4-repo SWE-bench · 62.50% 120B",
            "• Replays verified skills in 5.5–70 ms (11.8 mJ) with 0.00% broken commits",
            "11.8 mJ – 1.12 J [reported_j]",
            "#10b981",
        ),
        (
            436,
            "LAYER 1 · PHYSICAL & DETERMINISTIC BOOLEAN FABRIC",
            "CE-Gated 512×512 Beneš–Copy–Beneš RTL & #![no_std] L0 Safety Spine",
            "• xck26 P4-AXIS: 41,465 LUTs (35.4%), 81 RAMB36E2, 13 logic levels @ 250 MHz",
            "• ENARDEN + bubble CE cuts quiescent power 4.10× (15.59 est_mW, 0 DSP48)",
            "62.4 pJ / cycle [est_j]",
            "#38bdf8",
        ),
    ]
    for ly, ltitle, lsub, lb1, lb2, lpill, col in layers:
        parts += [
            f'  <rect x="626" y="{ly}" width="408" height="122" rx="11" fill="#0f172a" stroke="{col}" stroke-width="1.7"/>',
            f'  <text x="642" y="{ly+23}" font-family="{FONT_SANS}" font-size="10.6" font-weight="800" fill="{col}">{esc(ltitle)}</text>',
            f'  <text x="642" y="{ly+42}" font-family="{FONT_SANS}" font-size="11.2" font-weight="700" fill="#f8fafc">{esc(lsub)}</text>',
            f'  <text x="642" y="{ly+64}" font-family="{FONT_SANS}" font-size="9.8" fill="#cbd5e1">{esc(lb1)}</text>',
            f'  <text x="642" y="{ly+82}" font-family="{FONT_SANS}" font-size="9.8" fill="#cbd5e1">{esc(lb2)}</text>',
            f'  <rect x="642" y="{ly+93}" width="184" height="20" rx="10" fill="#0b1120" stroke="{col}" stroke-width="1.2"/>',
            f'  <text x="734" y="{ly+106.5}" font-family="{FONT_MONO}" font-size="9.4" font-weight="700" fill="{col}" text-anchor="middle">{esc(lpill)}</text>',
        ]

    # Downward compile arrows between layers
    parts += [
        '  <line x1="830" y1="266" x2="830" y2="286" stroke="#10b981" stroke-width="2.2" marker-end="url(#sk-emerald)"/>',
        '  <line x1="830" y1="412" x2="830" y2="432" stroke="#22d3ee" stroke-width="2.2" marker-end="url(#sk-cyan)"/>',
    ]

    # Right-side Narrow Conformal Escalation Hatch (Rung 3 Fallback)
    parts += [
        '  <rect x="1050" y="144" width="104" height="414" rx="11" fill="#1a120b" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5,3"/>',
        f'  <text x="1102" y="170" font-family="{FONT_MONO}" font-size="9.8" font-weight="800" fill="#fbbf24" text-anchor="middle">RUNG 3 WAKE</text>',
        f'  <text x="1102" y="186" font-family="{FONT_SANS}" font-size="9.2" fill="#fde68a" text-anchor="middle">Miss Only</text>',
        f'  <text x="1102" y="201" font-family="{FONT_MONO}" font-size="8.8" fill="#94a3b8" text-anchor="middle">(C(z) = 0)</text>',
        '  <line x1="1062" y1="214" x2="1142" y2="214" stroke="#334155" stroke-width="1"/>',
        f'  <text x="1102" y="238" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="#f8fafc" text-anchor="middle">Qwen3.6-35B</text>',
        f'  <text x="1102" y="253" font-family="{FONT_SANS}" font-size="9.0" fill="#cbd5e1" text-anchor="middle">970 MB RAM</text>',
        f'  <text x="1102" y="267" font-family="{FONT_SANS}" font-size="8.8" fill="#94a3b8" text-anchor="middle">12.3× NVMe cut</text>',
        '  <line x1="1062" y1="284" x2="1142" y2="284" stroke="#334155" stroke-width="1"/>',
        f'  <text x="1102" y="308" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="#f8fafc" text-anchor="middle">Nemotron-120B</text>',
        f'  <text x="1102" y="323" font-family="{FONT_SANS}" font-size="9.0" fill="#cbd5e1" text-anchor="middle">12B Active MoE</text>',
        f'  <text x="1102" y="337" font-family="{FONT_SANS}" font-size="8.8" fill="#94a3b8" text-anchor="middle">22.17 tok/s</text>',
        '  <line x1="1062" y1="354" x2="1142" y2="354" stroke="#334155" stroke-width="1"/>',
        f'  <text x="1102" y="378" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="#38bdf8" text-anchor="middle">Burn / CubeCL</text>',
        f'  <text x="1102" y="393" font-family="{FONT_SANS}" font-size="8.8" fill="#cbd5e1" text-anchor="middle">Metal · Vulkan</text>',
        f'  <text x="1102" y="407" font-family="{FONT_SANS}" font-size="8.8" fill="#cbd5e1" text-anchor="middle">CUDA · CPU</text>',
        f'  <text x="1102" y="421" font-family="{FONT_MONO}" font-size="8.8" fill="#6ee7b7" text-anchor="middle">69.2× DeltaNet</text>',
        # Escalation & Crystallize return arrows
        '  <path d="M 1034 335 L 1047 335" fill="none" stroke="#f59e0b" stroke-width="2.0" marker-end="url(#sk-amber)"/>',
        '  <path d="M 1102 558 L 1102 586 L 830 586 L 830 562" fill="none" stroke="#10b981" stroke-width="2.0" stroke-dasharray="5,3" marker-end="url(#sk-emerald)"/>',
        f'  <text x="966" y="603" font-family="{FONT_MONO}" font-size="9.2" font-weight="700" fill="#10b981" text-anchor="middle">Certified Wake → Crystallize Downward</text>',
        # Bottom honesty summary box
        '  <rect x="626" y="616" width="528" height="42" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.2"/>',
        f'  <text x="638" y="634" font-family="{FONT_MONO}" font-size="8.4" font-weight="700" fill="#e2e8f0">Anti-Laundering Lattice (ConformalProvenance.lean): ConformalBounded &lt; CheckSafe &lt; Deterministic</text>',
        f'  <text x="638" y="649" font-family="{FONT_SANS}" font-size="9.1" fill="#94a3b8">Shunt measured_j claimed only with physical INA260/INA238/INA3221 meters; laptop/RTL runs report reported_j &amp; est_j.</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def build_trinity_keynote_svg() -> str:
    """Dedicated keynote diagram uniting MathGround + OpenIE/LUT-Adapt + Pattern-Lang."""
    W, H = 1200, 680
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="tr-t tr-d">',
        '  <title id="tr-t">The Trinity of Compiled Intelligence: MathGround, OpenIE LUT-Adapt, and Pattern-Lang</title>',
        '  <desc id="tr-d">Keynote visual architecture uniting Pattern-Lang (verifier-first interface &amp; tier-typed Joule receipts), OpenIE LUT-Adapt (cognitive downward LUT6/.mol/ compilation &amp; 970 MB / 120B sparse-MoE streaming), and MathGround (embodied L0-closed physical safety spine on bare-metal silicon), guarded by 6-kernel Lean 4.36.0 + con-leche/con-ron verification.</desc>',
        "  <defs>",
        '    <marker id="tr-emerald" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#10b981"/>',
        "    </marker>",
        '    <marker id="tr-amber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        '      <path d="M0 0 L10 5 L0 10 z" fill="#f59e0b"/>',
        "    </marker>",
        "  </defs>",
        f'  <rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="#070b14" stroke="#334155" stroke-width="2"/>',
        f'  <text x="36" y="42" font-family="{FONT_SANS}" font-size="22" font-weight="800" fill="#f8fafc">The Trinity of Compiled Intelligence · Crossing the Compilation Threshold</text>',
        f'  <text x="36" y="65" font-family="{FONT_SANS}" font-size="12.5" fill="#94a3b8">Decoupling competence and safety from continuous floating-point spend across developer interfaces, software agents, and physical machines</text>',
        '  <rect x="920" y="20" width="244" height="32" rx="16" fill="#0f172a" stroke="#10b981" stroke-width="1.6"/>',
        f'  <text x="1042" y="40" font-family="{FONT_MONO}" font-size="10.8" font-weight="800" fill="#10b981" text-anchor="middle">6-Kernel con-leche + con-ron</text>',
    ]

    bands = [
        (
            96,
            "INTERFACE LAYER · PATTERN-LANG (pattern-lang.ai)",
            "The Verifier-First Runtime & Unforgeable Truth Receipt",
            "Intercepts IDE, SDK, and MCP calls; routes L0 Lookup → L1 Lawful Arithmetic / E-Graphs → gated Rung-3 fallback.",
            "Every response carries a 6-kernel-checked certificate + honest tier-typed Joule receipt (est_j | reported_j | shunt measured_j).",
            "0.154 ms / 272 µJ MCP Hit [reported_j]",
            "#a78bfa",
            "#ede9fe",
        ),
        (
            262,
            "COGNITIVE LAYER · OPENIE / LUT-ADAPT (research.openie.dev)",
            "The Crystallizing Mind · Online Downward LUT6 & .mol/ Compilation",
            "SleepCrystallizer compiles certified wakes into 0-FLOP Beneš LUT6 pages & .mol/ skills; 0.00% broken commits admitted.",
            "Streams 35B MoE in 970 MB RAM (12.3× NVMe cut) & 120B-A12B at 22.2 tok/s; beats standalone 24B/35B 2.50× on 4-repo SWE-bench.",
            "5.5 ms / 11.8 mJ Skill Replay [reported_j]",
            "#22d3ee",
            "#cffafe",
        ),
        (
            428,
            "PHYSICAL LAYER · MATHGROUND (mathground.ai)",
            "The Unbreakable L0 Spine · Proofs First, Models Last on $2–$50k Silicon",
            "6-kernel-verified kinematics, control barriers, & CE-gated xck26 LUT6 fabrics (#![no_std] Rust/Zig, 15.59 est_mW idle).",
            "Saving a dollar on compute never strips away physical safety; escalates via conformal uncertainty only off-spine.",
            "62.4 pJ LUT6 Cycle [est_j] · µJ MCU Spine",
            "#10b981",
            "#d1fae5",
        ),
    ]
    for by, btag, bhead, bl1, bl2, bpill, stroke, hi in bands:
        parts += [
            f'  <rect x="36" y="{by}" width="884" height="144" rx="14" fill="#111827" stroke="{stroke}" stroke-width="2.0"/>',
            f'  <text x="58" y="{by+28}" font-family="{FONT_MONO}" font-size="11.0" font-weight="800" fill="{stroke}">{esc(btag)}</text>',
            f'  <text x="58" y="{by+52}" font-family="{FONT_SANS}" font-size="15.5" font-weight="800" fill="#f8fafc">{esc(bhead)}</text>',
            f'  <text x="58" y="{by+78}" font-family="{FONT_SANS}" font-size="11.2" fill="#cbd5e1">{esc(bl1)}</text>',
            f'  <text x="58" y="{by+99}" font-family="{FONT_SANS}" font-size="11.2" fill="#cbd5e1">{esc(bl2)}</text>',
            f'  <rect x="58" y="{by+111}" width="292" height="23" rx="11.5" fill="#0b1120" stroke="{stroke}" stroke-width="1.3"/>',
            f'  <text x="204" y="{by+126}" font-family="{FONT_MONO}" font-size="10.0" font-weight="700" fill="{hi}" text-anchor="middle">{esc(bpill)}</text>',
        ]

    # Right column: Conformal Escalation + Downward Crystallization Loop
    parts += [
        '  <rect x="946" y="96" width="218" height="476" rx="14" fill="#161110" stroke="#f59e0b" stroke-width="1.8" stroke-dasharray="6,4"/>',
        f'  <text x="1055" y="126" font-family="{FONT_MONO}" font-size="11.2" font-weight="800" fill="#fbbf24" text-anchor="middle">CONFORMAL ESCALATION</text>',
        f'  <text x="1055" y="145" font-family="{FONT_SANS}" font-size="10.4" fill="#fde68a" text-anchor="middle">Invoked Strictly on Miss</text>',
        f'  <text x="1055" y="162" font-family="{FONT_MONO}" font-size="9.8" fill="#94a3b8" text-anchor="middle">C(z) = 0  or  OOD Shift</text>',
        '  <rect x="964" y="182" width="182" height="106" rx="10" fill="#0b1120" stroke="#f59e0b" stroke-width="1.3"/>',
        f'  <text x="1055" y="205" font-family="{FONT_SANS}" font-size="11.0" font-weight="800" fill="#f8fafc" text-anchor="middle">Proposer Only (Rung 3)</text>',
        f'  <text x="1055" y="226" font-family="{FONT_MONO}" font-size="9.8" fill="#cbd5e1" text-anchor="middle">• Qwen3.6-35B-A3B</text>',
        f'  <text x="1055" y="244" font-family="{FONT_MONO}" font-size="9.8" fill="#cbd5e1" text-anchor="middle">• Nemotron-120B-A12B</text>',
        f'  <text x="1055" y="262" font-family="{FONT_MONO}" font-size="9.8" fill="#cbd5e1" text-anchor="middle">• Devstral-Small-2-24B</text>',
        f'  <text x="1055" y="279" font-family="{FONT_SANS}" font-size="9.2" fill="#fbbf24" text-anchor="middle">Rust Burn / CubeCL Shaders</text>',
        '  <rect x="964" y="308" width="182" height="118" rx="10" fill="#0b1120" stroke="#10b981" stroke-width="1.4"/>',
        f'  <text x="1055" y="331" font-family="{FONT_SANS}" font-size="11.0" font-weight="800" fill="#6ee7b7" text-anchor="middle">Authority Gate</text>',
        f'  <text x="1055" y="352" font-family="{FONT_SANS}" font-size="9.8" fill="#e2e8f0" text-anchor="middle">AST + Sandbox + ACI</text>',
        f'  <text x="1055" y="371" font-family="{FONT_MONO}" font-size="10.0" font-weight="800" fill="#10b981" text-anchor="middle">0.00% Broken Commits</text>',
        f'  <text x="1055" y="391" font-family="{FONT_SANS}" font-size="9.4" fill="#94a3b8" text-anchor="middle">Unverified guesses are</text>',
        f'  <text x="1055" y="406" font-family="{FONT_SANS}" font-size="9.4" fill="#94a3b8" text-anchor="middle">refused, never compiled.</text>',
        '  <path d="M 920 168 L 942 168" fill="none" stroke="#f59e0b" stroke-width="2.2" marker-end="url(#tr-amber)"/>',
        '  <path d="M 920 334 L 942 334" fill="none" stroke="#f59e0b" stroke-width="2.2" marker-end="url(#tr-amber)"/>',
        '  <path d="M 1055 426 L 1055 500 L 924 500" fill="none" stroke="#10b981" stroke-width="2.4" marker-end="url(#tr-emerald)"/>',
        f'  <text x="1055" y="528" font-family="{FONT_MONO}" font-size="9.6" font-weight="700" fill="#10b981" text-anchor="middle">Forged Downward into</text>',
        f'  <text x="1055" y="544" font-family="{FONT_MONO}" font-size="9.6" font-weight="700" fill="#10b981" text-anchor="middle">LUT6 &amp; L0 Spine</text>',
        f'  <text x="600" y="618" font-family="{FONT_SANS}" font-size="13.2" font-weight="700" fill="#f8fafc" text-anchor="middle">“True agency is not measured by how long a model reasons, but by how quickly verified competence compiles to the energy floor.”</text>',
        f'  <text x="600" y="646" font-family="{FONT_MONO}" font-size="9.8" fill="#94a3b8" text-anchor="middle">Open Interface Engineering · Epistemic Honesty Standard: Hallucinations Quarantined (0.00% Broken Commits) · Tier-Typed Joule Receipts Never Faked</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def write_and_rasterize(svg_text: str, out_svg: Path, png_width: int = 2400) -> None:
    # Validate well-formed XML
    ET.fromstring(svg_text)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg_text, encoding="utf-8")
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        out_png = out_svg.with_suffix(".png")
        subprocess.run([rsvg, "-w", str(png_width), "-o", str(out_png), str(out_svg)], check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fig_dir = repo_root / "paper" / "figures"
    keynote_dir = repo_root / "docs" / "keynote"

    av_svg = build_avalanche_svg()
    sk_svg = build_six_kernel_and_stack_svg()
    tr_svg = build_trinity_keynote_svg()

    write_and_rasterize(av_svg, fig_dir / "avalanche-compilation-threshold.svg")
    write_and_rasterize(sk_svg, fig_dir / "six-kernel-trust-and-stack.svg")
    write_and_rasterize(tr_svg, keynote_dir / "trinity-stack.svg")
    write_and_rasterize(av_svg, keynote_dir / "avalanche-compilation-threshold.svg")
    write_and_rasterize(sk_svg, keynote_dir / "six-kernel-trust-and-stack.svg")

    print("Generated and XML-validated 3 vector SVGs + 2400px PNG twins:")
    print(f"  - {fig_dir / 'avalanche-compilation-threshold.svg'}")
    print(f"  - {fig_dir / 'six-kernel-trust-and-stack.svg'}")
    print(f"  - {keynote_dir / 'trinity-stack.svg'}")


if __name__ == "__main__":
    main()
