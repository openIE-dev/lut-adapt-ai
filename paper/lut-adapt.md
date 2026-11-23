---
title: "Downward LUT6 Compilation in a Mixture of Limits Cascade: Beneš-Routed Boolean Fabrics, Conformal Wake Gates, and Frontier Sparse-MoE NVMe Streaming"
description: "A Mixture of Limits cascade that runs zero-FLOP paged LUT6 lookup circuits first, streams 4-bit dense and sparse-MoE late-2026 models up to 35B from NVMe only on misses, gates commits via deterministic verifiers, and compiles certified wakes downward into Beneš-routed BRAM pages."
author: "David Charlot"
affiliation: "Open Interface Engineering"
genre: "Research paper · systems & hardware architecture"
status: "Draft"
date: "2026-11-22"
board_metered: false
slug: "lut-adapt"
---

# Downward LUT6 Compilation in a Mixture of Limits Cascade: Beneš-Routed Boolean Fabrics, Conformal Wake Gates, and Frontier Sparse-MoE NVMe Streaming

**David Charlot, Open Interface Engineering**  
Research paper · OpenIE `research.openie.dev` · Repository: `lut-adapt-ai`

> **Measurement.** Research paper, not a journal final. No physical FPGA board or isolated silicon rail shunt was metered in this repository (`measured_j = None`, `board_metered = false`). Hardware logic/BRAM utilization (`41,186` `xcup` CLB LUTs, `81` `RAMB36E2` blocks, `0` `DSP48` slices) and single-cycle (`CPI = 1` at `250 MHz` nominal) switching are **simulated** via Yosys `0.71+` (`synth_xilinx -family xcup` targeting AMD Kria KV260 `xck26`) and Verilator `5.054`. Workstation wall-clock times, exact Darwin `F_NOCACHE` `os.pread` NVMe byte counts, peak active RAM (`250.0 MB` on Oct 2026 `LiquidAI/LFM3-1.7B`, `500.0 MB` on Metal-fused `LiquidAI/LFM2.5-8B-A1B`, `970.0 MB` = `1.47%` of FP16 on Metal-fused `Qwen/Qwen3.6-35B-A3B`), `100.0%` bit/token parity, dual-channel Apple Silicon on-chip `reported_j` hardware counters (`L3 proc_pid_rusage(RUSAGE_INFO_V6).ri_energy_nj` CPU `nJ` + `L2 libIOReport` `"Energy Model"` `'GPU Energy'` `nJ`), and whole-machine macOS `PMU` battery/adapter accumulator deltas are **measured / reported** in software on Apple M5 Max. Analytical joule figures (`est_j`) stay strictly separated from on-chip `reported_j`, and `measured_j` remains `None` throughout. The `11,428 pJ` KV260 `INA260` `VCC_SOM` board reading cited for context belongs solely to *Metered Commit Gates* [same house]. No fabricated citations.

> A cascade that runs zero-FLOP lookup circuits first, streams late-2026 hybrid and sparse-MoE foundation models up to `34.66B` parameters from NVMe only on novel inputs, gates code commits behind deterministic verifiers, and compiles certified wakes downward into Beneš-routed `LUT6` pages — cutting NVMe weight traffic by up to `12.30×` at `100%` exact token parity and reaching `0.00%` broken commits admitted across multi-wave engineering benchmarks.

---

## 0. Overview

This paper studies how a machine can turn repeated neural-model work into cheap table lookups without losing exact output parity or promoting unverified guesses into trusted rules. It reports software measurements, register-transfer-level (RTL) synthesis and cycle simulations, and labeled analytical energy estimates.

**In plain words**

1. Most inputs a deployed system sees are variations of patterns it has already solved, so paying full neural-network compute and memory traffic on every token wastes energy and bandwidth.
2. We place a fast, reconfigurable 6-input truth-table (`LUT6`) lookup layer in front of streaming state-space and foundation models, and compile answers downward into lookup memory pages only after a strict check certifies them.
3. When late-2026 hybrid and sparse Mixture-of-Experts (`MoE`) language models up to `34.66B` parameters stream their 4-bit weights from NVMe flash (`Expert-AirMoL`) through custom fused Apple Metal `INT4` + `GatedDeltaNet` shaders, router-sparse expert `pread` plus a zero-FLOP `LUT6` burst drafter cuts disk reads by up to `12.30×` in **`970.0 MB` active RAM (`1.47%` of FP16, `6.77 J/tok` `reported_j`)** with zero change to the output tokens; pairing the cascade with a deterministic contract or `pytest` verifier `C(z)` lifts verified pass rate to **`87.50%`** across both a `48`-task algorithmic suite (vs. `77.08%` for standalone `35B`) and a `16`-task multi-file git-repository patching benchmark (vs. `62.50%` for `35B`), admitting **`0.00%` broken commits** and replaying crystallized Wave-3 skills in **`2.89 ms` (`0.0424 J` `reported_j`, `15,401×` faster and `25,227×` lower `reported_j`)**.

In *Metered Commit Gates* [same house], Part II introduced **Mixture of Limits (MoL)**: try gears in decreasing order of pass probability per joule $p_i / c_i$ (Simon and Kadane, 1975), stop as soon as the task completion predicate $C(z) = 1$ holds, and never promote `ModelGenerated` output to `Deterministic`. A static Lookup gear, however, covers only the patterns written into it before deployment. Conversely, existing lookup-table neural networks — LogicNets (Umuroglu et al., 2020), DiffLogic (Petersen et al., 2022), and Differentiable Weightless Neural Networks (Bacellar et al., 2024) — bake wire connectivity into fixed FPGA routing, requiring minutes to hours of vendor place-and-route (`Vivado`) whenever the input distribution shifts.

This work closes the loop between the top and bottom of the cascade via **online downward `LUT6` compilation**:

- **Rung 0 (Lookup — Paged Beneš + `LUT6`)**: Evaluates 64-bit truth-table `INIT` masks routed through a non-blocking $512 \times 512$ **Beneš–Copy–Beneš** multicast network and cached across $81 \times \texttt{RAMB36E2}$ dual-port block RAMs (`0` floating-point operations, `CPI = 1` cycle context switch in Verilator simulation).
- **Rung 1 (Formula / Stream — `Mamba-3` Complex MIMO SSD)**: Handles misses that require sequence context via a constant-memory $O(1)$ recurrent state (`48 KB` pinned SRAM in our `14.3M`-parameter reference), avoiding $O(L)$ KV-cache growth.
- **Rung 2 (Topological Router — `HC-SOINN`)**: Maintains online prototype topology over latent embeddings, isolating regime boundaries and tripping a circuit breaker (`Refuse` / escalate) when unlabeled distribution shift arrives.
- **Rung 3 (Solver / Model LAST — `JEPA-EBRM`, `AirMoL`, & `Expert-AirMoL` 4-Bit NVMe Streaming)**: Wakes deep latent energy-based verification or streams pretrained late-2026 hybrid and sparse-MoE foundation models (`LiquidAI/LFM3-1.7B`, `google/gemma-4-E2B-it`, `LiquidAI/LFM2.5-8B-A1B`, and `Qwen/Qwen3.6-35B-A3B`, alongside compact `LFM2.5-350M` and `Qwen3.5-0.8B-Base`) only when cheaper rungs abstain or require exact target verification.
- **Downward `SleepCrystallizer`**: Extracts recurring wake traces using Walsh–Hadamard spectral warm-starts, joint-mode minterm truth tables, and Waksman Beneš switch schedules, hot-swapping shadow BRAM banks without halting inference or running FPGA place-and-route.

### 0.1 Contributions

1. **Amortized downward compilation condition (Section 2.1)**: Extends the Simon–Kadane $p_i/c_i$ cascade ordering with an explicit crystallization cost $E_{\text{compile}}$, establishing when compiling certified Rung 3 wakes into Rung 0 `LUT6` pages lowers cumulative energy-to-correct-completion.
2. **Non-blocking $512 \times 512$ Beneš–Copy–Beneš + `81`-`RAMB36E2` RTL architecture (Section 2.2, Section 3.3, Claim LA-3)**: Eliminates both the `102.0%` KV260 LUT overflow of flat crossbars (`119,478` LUTs) and the `926,619`-LUT explosion of flip-flop page multiplexers, synthesizing to `41,186` `xcup` CLB LUTs (`35.2%` of `xck26`), `81` `RAMB36E2`s (`56.2%`), and `0` `DSP48` slices with `1`-cycle (`4.0 ns` at `250 MHz` nominal) page switching in simulation.
3. **Universal $\le 64$-bit joint-mode minterm synthesis (Section 2.3, Claims LA-5 & LA-7)**: Proves joint-mode minterm lookup dominates independent bit-plane DiffLogic argmax on correlated branches (`if/else`, BPE multi-token bursts) with zero off-support chimeras, packing $K = \lfloor 64 / \lceil \log_2 |V| \rceil \rfloor$ subword tokens into a single `uint64` (`16b × 4 = 64b` for `|V|=65,536`; `18b × 3 = 54b` for `|V|=248,320`).
4. **Oracle-free Split-Conformal + Online ACI crystallization gate (Section 2.4, Claim LA-4)**: Pairs a quadratically regularized latent `JEPA-EBRM` score with Split Conformal calibration and Adaptive Conformal Inference (Gibbs and Candès, 2021), achieving `0.00%` false crystallization risk and `1.0000` OOD AUROC on shifted regimes where softmax MSP, margin, Helmholtz logit energy, and 5-pass MC-Dropout suffer `17.62%–24.28%` false risk.
5. **`AirMoL` and `Expert-AirMoL` zero-extra-sweep 4-bit NVMe streaming up to `34.66B` Sparse-MoE (Sections 3.4–3.5, Claims LA-6, LA-7, LA-9)**: Combines Darwin `F_NOCACHE` `os.pread` 4-bit sharding, router-indexed sparse-expert `pread` (`pack4` byte strides fetching only active `k`-of-`E` experts), hybrid state rollback (`ShortConv` / `SlidingAttn` / `Gated DeltaNet` + `GQA KV`), and online token-`LUT6` burst speculation across `LiquidAI/LFM3-1.7B` (Oct 2026; `2.74×` I/O cut in `3.79%` RAM), `LiquidAI/LFM2.5-8B-A1B` (`8.47B`; `6.90×` I/O cut, `3.93×` speedup), and `Qwen/Qwen3.6-35B-A3B` (`34.66B`; **`12.30×` I/O cut**, **`2,530.7 MB` RAM = `3.83%` of FP16**, **`6.50×` speedup**) at `100.0%` exact token parity.
6. **Sub-1-bit `LUT6` factorization, custom Metal `INT4` MoE / `GatedDeltaNet` kernels, multi-file git-repo `pytest` patching, and OpenAI/MCP `.mol/` agent (Section 3.5, Claims LA-10 through LA-14)**: Replaces the `1.98M` runtime binary-matrix MACs of `LittleBit-1 / LittleBit-2` (`0.576 bpw`; Lee et al., 2025; Lee and Kim, 2026) with zero-MAC `Sub1Bit-LUT6` partial-sum tables (`< 4 × 10⁻⁶` max error, `100.0%` exact parity via `MoL` verification in `0` QAT epochs); accelerates `GatedDeltaNet` **`20.66×`** and `INT4` Sparse-MoE **`43.45×`** in custom Apple Metal shaders (`970.0 MB` = `1.47%` of FP16 active RAM on `Qwen3.6-35B-A3B`, `6.77 J/tok` `reported_j`); and reaches **`87.50%` verified pass** on both the `48`-task algorithmic bakeoff (vs. `0.00%–77.08%` standalone, using `35B` on only `22.92%` of tasks) and the `16`-task multi-file git-repository `pytest` benchmark (vs. `0.00%–62.50%` standalone) with **`0.00%` broken commits** and a drop-in OpenAI (`/v1/chat/completions` REST+SSE) and Model Context Protocol (`MCP` JSON-RPC `2.0`) local coding agent (`26/26` `pytest` passing).

### 0.2 Energy to correct completion and downward amortization

Following *Metered Commit Gates* [same house], let $C(z) \in \{0, 1\}$ be the task completion predicate over state $z$ and $\delta \in [0, 1)$ the allowed failure rate. Systems are ranked by minimal expected joules to reach $C(z) = 1$:

$$J^*(S) = \min_{\pi} \mathbb{E}[J_\pi] \quad \text{subject to} \quad \Pr[C(z) = 1] \ge 1 - \delta.$$

In this paper, every joule value is computed under an explicit analytical model (`est_j`) and never mixed with hardware meter readings (`measured_j = None`). Downward compilation modifies the stationary distribution over gears: paying a one-time analytical compilation cost $E_{\text{compile}}$ after certified wakes increases Rung 0's coverage probability $p_0(t)$, shifting mass from $c_3$ (wake model) to $c_0 \ll c_3$ (`LUT6` lookup).

### 0.3 Relation to OpenIE Mixture of Limits and RCE-JEPA [same house]

Prior OpenIE studies — *Metered Commit Gates*, *Task Satiation vs. Aggregate Rebound*, *The Reality Code Engine*, *RCE-JEPA*, and *RCE Compile* — are `[same house]` works cited for architectural continuity, never as independent external confirmation. Two invariants from those studies bind every module in `lut-adapt-ai`:

- **Provenance non-strengthening (`ModelGenerated ↛ Deterministic`)**: When `SleepCrystallizer` compiles `LUT6` truth tables from a deterministic `RCE` check (Section 4.1), the resulting table is `Check-safe` on verified holdout grids (guarded online by an `HC-SOINN` / `RCE` circuit breaker). When `SleepCrystallizer` compiles under the oracle-free `JEPA-EBRM` Split-Conformal + ACI gate (Section 2.4), the resulting table carries a finite-sample statistical coverage bound (`Conformal-bounded`, $1-\alpha$) and is **never** stamped `Deterministic`.
- **Proposer vs. authority in speculative decoding (`The drafter proposes; the target decides`)**: In byte-level and BPE token-level speculative decoding (Claims LA-5, LA-6, LA-7), `PagedBenesLUT6` acts strictly as a zero-FLOP proposer. Every emitted token is verified against the exact greedy distribution of the target model (`Mamba3CodeLM`, `LFM2.5-350M`, or `Qwen3.5-0.8B-Base`), guaranteeing `100.0%` lossless output parity.

### 0.4 How to read this paper

Section 1 defines all notation, acronyms, and measurement labels before use. Section 2 states and proves four mathematical results governing cascade amortization, Beneš multicast routing, joint-mode minterm synthesis, and conformal wake gating. Section 3 describes the reference implementation across Python, Rust, Zig, Go, Julia, SystemVerilog (`xck26`), `AirMoL` / `Expert-AirMoL` NVMe streaming, and `VerifiedCodeCascade`. Section 4 reports Claims `LA-1` through `LA-11` with explicit measurement columns. Section 5 compares prior work in two structured tables. Section 6 records threats to validity, non-claims, and executable falsifiers `LA-F1` through `LA-F11`. Appendix A lists exact reproduction commands.

---

## 1. Notation, terms, and measurement labels

All symbols, hardware primitives, model acronyms, and evidence labels used in later sections are collected below.

| Term | Definition in this paper |
| :--- | :--- |
| **Mixture of Limits (MoL)** | Cascade policy [same house] that orders gears by decreasing $p_i / c_i$, stops when $C(z) = 1$, and records typed refusals and receipts. |
| **`LUT6` (`INIT[63:0]`)** | A 6-input, 1-output Boolean look-up table primitive on Xilinx UltraScale+ silicon, fully specified by a 64-bit truth-table `INIT` mask. |
| **`RAMB36E2`** | Dual-port 36-Kbit Block RAM primitive on AMD/Xilinx Zynq UltraScale+ (`xck26` carries `144` `RAMB36E2` blocks). |
| **Beneš–Copy–Beneš** | Three-cascade non-blocking multicast network ($\mathcal{B}_1 \to \mathcal{C} \to \mathcal{B}_2$) routing $N=512$ input wires to $M=480$ `LUT6` input pins via $11,008$ SRAM switch bits without FPGA re-synthesis. |
| **`Mamba-3` SSD** | State Space Duality sequence model with trapezoidal discretization, complex-valued RoPE state transitions, and MIMO projections (`O(1)` recurrent state size per step). |
| **`HC-SOINN`** | Hierarchy-Context Self-Organizing Incremental Neural Network (Rung 2); builds online topological prototypes and detects novel regimes via distance thresholds. |
| **`JEPA-EBRM`** | Joint-Embedding Predictive Architecture paired with a latent Energy-Based Reasoning Model $E_\phi(x, y)$ (Rung 3 verifier / scorer). |
| **Split Conformal + ACI** | Distribution-free calibration on held-out scores combined with Adaptive Conformal Inference ($\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$; Gibbs and Candès, 2021). |
| **Joint-mode minterm** | Packing a $K$-step discrete burst $(y_1, \dots, y_K)$ into one integer word ($\le 64$ bits) and storing the empirical conditional mode per context, preventing marginal bit-plane chimeras. |
| **`AirMoL` & `Expert-AirMoL`** | 4-bit group-quantized (`INT4` + FP16 scale/zero) NVMe layer-streaming and router-sparse expert-indexed `pack4` streaming engines with Darwin `F_NOCACHE` `os.pread`, pinned $O(1)$ recurrent states, and token-`LUT6` burst speculation. |
| **`LFM3-1.7B` & `LFM2.5-350M`** | Liquid AI's October 2026 `1.73B`-parameter Tri-Hybrid checkpoint (`13` `ShortConv` + `5` `SlidingAttn` + `6` `FullAttn`, `|V| = 128,000`) and March 2026 `354.5M` hybrid checkpoint (`|V| = 65,536`). |
| **`LFM2.5-8B-A1B` & `Qwen3.6-35B-A3B`** | Late-2026 sparse Mixture-of-Experts (`MoE`) hybrid checkpoints: Liquid AI's `8.47B` (`4`-of-`32` active experts, `18` `Conv` + `6` `Attn`) and Qwen's `34.66B` (`8`-of-`256` active experts + shared expert, `30` `Gated DeltaNet` + `10` `Attn`, `|V| = 248,320`). |
| **`Gemma-4-E2B-it` & `Qwen3.5-0.8B`** | Google's 2026 on-device hybrid instruction model (`google/gemma-4-E2B-it`) and Alibaba Qwen Team's February 2026 `752.4M` hybrid checkpoint (`Qwen/Qwen3.5-0.8B-Base`). |
| **`LittleBit` / `LittleBit-2` & `Sub1Bit-LUT6`** | Samsung's `0.1–0.58 bpw` multi-scale sign-value-independent binary latent factorization (`W ≈ G ⊙ ((U₁V₁ᵀ)s₁ + (U₂V₂ᵀ)s₂)` via SVD / Procrustes ITQ + QAT; Lee et al., 2025; Lee and Kim, 2026), vs. OpenIE's zero-MAC `6`-input `LUT6` partial-sum precomputation (`Sub1Bit-LUT6`). |
| **`VerifiedCodeCascade`** | Multi-rung code-synthesis cascade gating every commit behind a `0.35s`-bounded AST + sandbox execution + boundary-invariant contract verifier $C(z) \in \{0,1\}$, counterexample escalation, and online `BenesLUT6` + canonical AST skill crystallization. |
| **`PLD` / `REST`** | Prompt Lookup Decoding (Saxena, 2023) and Retrieval-Based Speculative Decoding (He et al., 2024), zero-FLOP retrieval drafters evaluated as baselines. |
| **`Medusa` / `EAGLE-2`** | Multi-head MLP speculative drafter (Cai et al., 2024) and context-aware autoregressive feature drafter (Li et al., 2024), neural drafters evaluated as baselines. |
| **`Deterministic` / `Check-safe`** | Provenance of an output verified by an exact specification check (`RCE` / $C(z)=1$); distinct from statistical model predictions. |
| **`Conformal-bounded`** | Provenance of a `LUT6` page crystallized under the oracle-free `JEPA-EBRM` ACI gate ($1-\alpha$ marginal coverage); never promoted to `Deterministic`. |
| **`ModelGenerated`** | Provenance of a neural model proposal prior to verification. `ModelGenerated ↛ Deterministic`. |
| **`Measured (software / I/O / OS PMU)`** | Exact byte counts read/written via OS syscalls (`os.pread` with `F_NOCACHE`), resident memory allocations, wall-clock seconds, `100.0%` bit/token agreement, and whole-system macOS `PMU` (`AppleSmartBattery.PowerTelemetryData`) cumulative energy deltas on Apple M5 Max. Not isolated chip-rail joules. |
| **`Simulated (RTL)`** | Gate/BRAM/DSP counts from Yosys `0.71+` (`synth_xilinx -family xcup`) and cycle counts (`CPI = 1`) from Verilator `5.054`. Not physical board timing closure or rail power (`board_metered = false`). |
| **`Estimated (est_j)`** | Analytical joules computed from operation counts or TDP-time stand-ins (`joule_meter.py` fallback when `RAPL`/`NVML` hardware counters are absent on macOS). Comparable only across arms inside the same formula; never silicon watts. |

---

## 2. Mathematical formulations and four results

### 2.1 Result 1: Cascade ordering with amortized downward compilation

Consider gears $i \in \{0, 1, 2, 3\}$ tried sequentially on an input $x$. Gear $i$ closes the task ($C(z)=1$) with probability $p_i$ at expected analytical energy $c_i > 0$. By Simon and Kadane (1975) (derived for Mixture of Limits in *Metered Commit Gates*, Section II.2 [same house]), the expected energy of a permutation $\sigma$ under independent close events is

$$\mathbb{E}[J_\sigma] = \sum_{k=1}^{n} c_{\sigma(k)} \prod_{j < k} \bigl(1 - p_{\sigma(j)}\bigr),$$

which is minimized if and only if gears are ordered by non-increasing ratio $p_{\sigma(1)}/c_{\sigma(1)} \ge \dots \ge p_{\sigma(n)}/c_{\sigma(n)}$.

Now suppose a stationary regime $\mathcal{D}_r$ persists for a remaining horizon of $\Delta T$ queries. Initially Rung 0 (`LUT6`) has coverage $p_0 = 0$ on $\mathcal{D}_r$, so every query escalates to a wake path of expected cost $c_{\text{wake}}$ (for example Rung 3 at cost $c_3$). After collecting $N_{\text{wake}}$ certified wake pairs $(x, y^*)$, `SleepCrystallizer` expends one-time compilation energy $E_{\text{compile}}$ to synthesize a Rung 0 `LUT6` page with certified coverage $p_0' \in (0, 1]$ and per-query lookup cost $c_0 \ll c_{\text{wake}}$.

**Proposition 1 (Net energy condition for downward crystallization).** *Downward compilation strictly reduces expected cumulative energy over the remaining regime horizon $\Delta T$ if and only if*

$$\Delta T \cdot p_0' \bigl(c_{\text{wake}} - c_0\bigr) > E_{\text{compile}}, \qquad \text{and} \qquad \frac{p_0'}{c_0} > \frac{p_{\text{wake}}}{c_{\text{wake}}}.$$

*Proof.* Without crystallization, the remaining $\Delta T$ queries each incur expected energy $c_{\text{wake}}$ (assuming the wake path closes the task with $p_{\text{wake}} = 1$), totaling $\Delta T \cdot c_{\text{wake}}$. With crystallization at cost $E_{\text{compile}}$, Rung 0 precedes the wake path whenever $p_0'/c_0 > 1/c_{\text{wake}} \iff c_0 < p_0' c_{\text{wake}}$. Each subsequent query then costs $c_0 + (1 - p_0') c_{\text{wake}} = c_{\text{wake}} - p_0'(c_{\text{wake}} - c_0)$ in expectation, provided $c_0$ is negligible or already accounted on miss; more precisely, when a Rung 0 miss escalates after spending $c_0$, per-query expected spend is $c_0 + (1 - p_0') c_{\text{wake}}$, saving $p_0' c_{\text{wake}} - c_0$ per step. Summing over $\Delta T$ steps yields net savings $\Delta T (p_0' c_{\text{wake}} - c_0) - E_{\text{compile}} > 0$. When $c_0 \ll p_0' c_{\text{wake}}$ (in our analytical model $c_0 / c_3 < 10^{-4}$), this reduces to $\Delta T \cdot p_0'(c_{\text{wake}} - c_0) > E_{\text{compile}}$. $\blacksquare$

**Consequence on non-stationary drift.** If regimes switch every $\Delta T \le E_{\text{compile}} / (p_0' c_{\text{wake}} - c_0)$ steps (high-frequency adversarial churn), compiling `LUT6` pages wastes energy. `SleepCrystallizer` therefore gates compilation behind a minimum buffer of $N_{\text{wake}}$ certified traces and a holdout agreement threshold $\tau_{\text{agree}} = 0.985$.

---

### 2.2 Result 2: Non-blocking $512 \times 512$ Beneš–Copy–Beneš multicast routing

A Rung 0 layer of $G = 80$ dynamic `LUT6` primitives requires $M = 6G = 480$ input pins drawn from an $N = 512$-bit input bus. Because learned Boolean features exhibit variable fan-out (one input bit may feed zero `LUT6` pins or dozens), a standard $N \times N$ permutation Beneš network cannot route multicast assignments, while a flat $M \times N$ multiplexer crossbar requires $480$ independent $512:1$ muxes (`119,478` CLB LUTs for $384$ ports, overflowing the `117,120`-LUT Kria KV260 `xck26`; Claim LA-3).

**Theorem 2 (Contention-free multicast via Beneš–Copy–Beneš).** *Let $N = 2^n$ ($n=9, N=512$) and $M \le N$ ($M=480$). Any arbitrary many-to-one multicast mapping $\phi: \{0, \dots, M-1\} \to \{0, \dots, N-1\}$ is realizable without wire contention in a three-cascade $2 \times 2$ switch fabric $\mathcal{B}_1 \to \mathcal{C} \to \mathcal{B}_2$ of depth*

$$D(N) = 2(2\log_2 N - 1) + \log_2 N = 5\log_2 N - 2 = 43 \text{ stages}$$

*comprising $43 \times (N/2) = 11,008$ SRAM-configured $2 \times 2$ cells, with $O(N \log N)$ configuration runtime and zero FPGA place-and-route.*

*Proof.* Let $U = \{\phi(0), \dots, \phi(M-1)\} = \{s_1, \dots, s_r\}$ be the $r \le M$ distinct active inputs with fan-out multiplicities $m_1, \dots, m_r \ge 1$ satisfying $\sum_{j=1}^r m_j = M \le N$. Define partial sums $S_0 = 0$ and $S_j = \sum_{i=1}^j m_i$.
1. **Concentration ($\mathcal{B}_1$, $2n-1 = 17$ stages)**: Construct any bijection $\pi_1: \{0, \dots, N-1\} \to \{0, \dots, N-1\}$ mapping each active source $s_j$ to contiguous block head $S_{j-1}$ ($\pi_1(s_j) = S_{j-1}$), sending unused inputs arbitrarily to $\{0,\dots,N-1\} \setminus \{S_0,\dots,S_{r-1}\}$. Since $\pi_1$ is a bijection on $N=2^n$ wires, the classical Beneš–Waksman theorem (Beneš, 1965; Waksman, 1968) decomposes $\pi_1$ into $2n-1$ stages of $N/2$ $2\times 2$ switches via $O(N \log N)$ 2-coloring of bipartite input/output switch graphs.
2. **Contiguous fan-out ($\mathcal{C}$, $n = 9$ stages)**: Each seed at wire $S_{j-1}$ must replicate across the contiguous interval $[S_{j-1}, S_j - 1]$ without crossing $[S_j, S_{j+1}-1]$. By Thompson's monotonic copy network (Thompson, 1978; Lee, 1988), routing each active interval $[S_{j-1}, S_j - 1]$ through $n$ butterfly stages according to the binary expansion of destination index $q \in [S_{j-1}, S_j - 1]$ preserves strict monotonicity of interval endpoints at every stage, guaranteeing zero link contention and producing $m_j$ copies of $s_j$ on outputs $[S_{j-1}, S_j - 1]$.
3. **Output permutation ($\mathcal{B}_2$, $2n-1 = 17$ stages)**: Partition each contiguous copy block $[S_{j-1}, S_j - 1]$ bijectively onto the target pin set $\phi^{-1}(s_j) \subset \{0, \dots, M-1\}$, padding $[M, N-1]$ arbitrarily to form a full bijection $\pi_2$ on $\{0,\dots,N-1\}$. Applying Waksman's algorithm to $\pi_2$ configures $\mathcal{B}_2$ in $O(N \log N)$ time. Composing $\pi_2 \circ \mathcal{C} \circ \pi_1$ delivers input $\phi(q)$ to every pin $q \in \{0,\dots,M-1\}$. $\blacksquare$

---

### 2.3 Result 3: Joint-mode minterm synthesis vs. marginal coordinate argmax

Standard differentiable LUT layers (Petersen et al., 2022) predict multi-bit or multi-token outputs $\mathbf{Y} = (Y_1, \dots, Y_B) \in \mathcal{Y}$ by training independent binary gates $\hat{y}_b(x) = \mathbb{I}[\Pr(Y_b = 1 \mid \psi(x)) > 1/2]$ on each bit-plane $b \in \{1,\dots,B\}$. On multi-modal conditional distributions — such as code after `if (` vs. `return `, or $K$-token BPE bursts — independent bit-plane rounding creates invalid **chimera** tokens.

**Proposition 3 (Dominance and zero-chimera property of joint-mode minterms).** *For any discrete context projector $\psi: \mathcal{X} \to \mathcal{C}$ and discrete burst space $\mathcal{Y}$ encoded into $B \le 64$ bits, define the joint-mode minterm table $T^*(c) = \arg\max_{\mathbf{y} \in \mathcal{Y}} \Pr(\mathbf{Y} = \mathbf{y} \mid \psi(X) = c)$ and the marginal bit-plane predictor $\hat{\mathbf{y}}^{\text{marg}}(c) = (\arg\max_{v \in \{0,1\}} \Pr(Y_b = v \mid \psi(X)=c))_{b=1}^B$. Then:*
1. *Exact-burst accuracy satisfies $\Pr[\mathbf{Y} = T^*(\psi(X))] \ge \Pr[\mathbf{Y} = \hat{\mathbf{y}}^{\text{marg}}(\psi(X))]$, with strict inequality whenever marginal bit majorities disagree with the joint mode.*
2. *Every output of $T^*$ lies inside the empirical support $\text{supp}(\mathbf{Y} \mid \psi(X)=c)$ (zero chimera bursts), whereas $\hat{\mathbf{y}}^{\text{marg}}(c)$ can achieve `0%` exact-burst accuracy even when two valid branches each occur with probability $0.5$.*

*Proof.* Conditioning on $\psi(X) = c$, for any deterministic function $g(c) \in \{0,1\}^B$ we have $\Pr(\mathbf{Y} = g(c) \mid \psi(X)=c) \le \max_{\mathbf{y} \in \mathcal{Y}} \Pr(\mathbf{Y} = \mathbf{y} \mid \psi(X)=c) = \Pr(\mathbf{Y} = T^*(c) \mid \psi(X)=c)$. Taking expectations over $\psi(X)$ proves (1). For (2), $T^*(c)$ selects an element attaining positive probability mass in $\text{supp}(\mathbf{Y} \mid \psi(X)=c)$. By contrast, let $B=3$ and suppose $\mathbf{Y} \mid c$ places mass $0.4, 0.35, 0.25$ on $(1,1,0), (1,0,1), (0,1,1)$ respectively; each marginal bit has $\Pr(Y_b = 1 \mid c) \ge 0.60 > 1/2$, so $\hat{\mathbf{y}}^{\text{marg}}(c) = (1,1,1) \notin \text{supp}(\mathbf{Y} \mid c)$, yielding `0%` exact-burst match versus `40%` for $T^*(c) = (1,1,0)$. $\blacksquare$

**Universal 64-bit BPE token packing.** For any pretrained tokenizer with vocabulary size $|V|$, each token ID requires $b_V = \lceil \log_2 |V| \rceil$ bits, allowing a single `uint64` word (`8` bytes, matching one 64-bit `LUT6 INIT` vector) to pack $K = \lfloor 64 / b_V \rfloor$ tokens losslessly:
- `LiquidAI/LFM2.5-350M` and `LiquidAI/LFM2.5-8B-A1B` ($|V| = 65,536$): $b_V = 16\text{ bits} \implies K = \lfloor 64/16 \rfloor = 4\text{ tokens/word}$ (`64` bits).
- `LiquidAI/LFM3-1.7B` ($|V| = 128,000$): $b_V = 17\text{ bits} \implies K = \lfloor 64/17 \rfloor = 3\text{ tokens/word}$ (`51` bits $\le 64$).
- `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.6-35B-A3B` ($|V| = 248,320$): $b_V = 18\text{ bits} \implies K = \lfloor 64/18 \rfloor = 3\text{ tokens/word}$ (`54` bits $\le 64$).

---

### 2.4 Result 4: Oracle-free latent `JEPA-EBRM` Split-Conformal + Online ACI gate

In open-world perceptual or telemetry tasks where no deterministic `RCE` check exists, gating `SleepCrystallizer` by a neural classifier's softmax Maximum Softmax Probability (MSP) or logit energy (Liu et al., 2020) suffers from severe overconfidence along linear logit directions (Hein et al., 2019), crystallizing hallucinated labels into Rung 0 `LUT6` tables (`17.6%–24.3%` false risk in Claim LA-4).

We define a quadratically regularized latent `JEPA-EBRM` nonconformity score over encoder $z = f_\theta(x) \in \mathbb{R}^d$ and class/action prototypes $\{\mu_y\}$:

$$E_\phi(x, y) = \frac{1}{2}\|f_\theta(x) - \mu_y\|_2^2 + \frac{\lambda_r}{2}\|f_\theta(x)\|_2^2 + \psi_\phi(f_\theta(x), y), \qquad \lambda_r > 0,$$

where $\psi_\phi$ is bounded ($|\psi_\phi(z, y)| \le C_\psi$). Calibration on $n$ exchangeable held-out pairs $\{(X_i, Y_i)\}_{i=1}^n$ sets quantile $\hat{Q}_{1-\alpha} = \text{Quantile}_{\lceil (n+1)(1-\alpha) \rceil / n}(\{E_\phi(X_i, Y_i)\}_{i=1}^n)$. Online during deployment, Adaptive Conformal Inference (ACI; Gibbs and Candès, 2021) updates $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$ with step size $\gamma > 0$, emitting prediction set $\mathcal{C}_t(x) = \{y : E_\phi(x, y) \le \hat{Q}_{1-\alpha_t}\}$. `SleepCrystallizer` accepts $(x, \hat{y})$ for `LUT6` compilation if and only if $|\mathcal{C}_t(x)| = 1$ (singleton, unambiguous, and in-manifold), tagging the page `Conformal-bounded` (`ModelGenerated ↛ Deterministic`).

**Proposition 4 (Coverage, adaptive drift recovery, and radial OOD abstention).**
1. *(Exchangeable finite-sample validity)* Under i.i.d. or exchangeable $(X_{\text{test}}, Y_{\text{test}})$ at fixed $\alpha_t = \alpha$, $\Pr[Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})] \ge 1 - \alpha$ (Vovk et al., 2005; Lei et al., 2018).
2. *(Arbitrary shift recovery)* For any bounded nonconformity sequence under distribution shift, ACI's empirical miscoverage satisfies $\lim_{T\to\infty} T^{-1} \sum_{t=1}^T \text{err}_t = \alpha$ deterministically whenever $\alpha_t$ remains in $(0, 1)$ (Gibbs and Candès, 2021, Proposition 4.1).
3. *(Radial OOD abstention)* Because $|\psi_\phi| \le C_\psi$ and $\lambda_r > 0$, $\min_{y} E_\phi(x, y) \ge \frac{\lambda_r}{2}\|z\|_2^2 - C_\psi \to +\infty$ as $\|z\|_2 \to \infty$. Thus every ray leaving the bounded radius $R(\alpha_t) = \sqrt{2(\hat{Q}_{1-\alpha_t} + C_\psi)/\lambda_r}$ yields $\mathcal{C}_t(x) = \emptyset$ (explicit `Refuse` / abstain, zero false crystallization). $\blacksquare$

---

## 3. Architecture, polyglot reference suite, and KV260 RTL

### 3.1 The four-rung downward-compilation cascade

Figure 1 illustrates the closed-loop execution and downward-compilation dataflow implemented in `src/clai/`.

![Figure 1. Downward compilation across a four-rung Mixture of Limits cascade.](figures/mol-downward-cascade.png)

*Figure 1. Downward compilation across a four-rung Mixture of Limits cascade. Design diagram, not a board power measurement. Receipts keep analytical `est_j` separate from `measured_j = None`.*

Every incoming input enters **Rung 0** (`src/clai/rung0/difflut_net.py`, `src/clai/Speculative/code_corpus_speculative.py`, or `src/clai/rtl_gen/verilog_emitter.py`). On a cache hit whose confidence or deterministic check passes, Rung 0 closes immediately without waking floating-point weights. On a miss or ambiguity:
1. **Rung 1 (`Mamba-3` Complex MIMO SSD, `src/clai/rung2/triton_mamba3_ssd.py`)** updates its constant-size complex recurrent state via trapezoidal discretization and MIMO projections.
2. **Rung 2 (`HC-SOINN`, `src/clai/rung2_5/hc_soinn.py`)** tracks topological clusters over latent embeddings, flagging novel regimes (`regime_shift`) and isolating per-regime `SleepCrystallizer` buffers so new clusters do not overwrite prior `LUT6` banks.
3. **Rung 3 (`JEPA-EBRM` & `AirMoL`, `src/clai/rung3/jepa_ebrm.py`, `src/clai/governance/conformal_ebrm.py`, `src/clai/frontier/`)** performs latent energy minimization or streams 4-bit foundation model shards from NVMe.
4. **`SleepCrystallizer` (`src/clai/compiler/sleep_crystallizer.py`)** warm-starts Boolean gates from 1st/2nd/3rd-order Walsh–Hadamard spectral correlations ($c_S = \mathbb{E}[y \prod_{i \in S} x_i]$), computes exact Waksman switch bits (`src/clai/rung0/benes_lut_fabric.py`), verifies holdout agreement $\ge 98.5\%$, and atomically flips the active `LUT6` bank pointer.

### 3.2 Polyglot systems suite

To verify that the cascade's bit-plane operations, $O(1)$ Mamba-3 recurrence, and zero-allocation lookup paths are portable across systems languages and GPU kernels, five independent implementations were built and tested in `sys/`:

- **Python 3.12 + PyTorch + Custom Triton Kernel (`src/clai/`)**: Reference 4-rung engine, custom Triton `LUT6` evaluation kernel (`triton_lut6.py`), conformal governance, SystemVerilog generator, and `AirMoL` NVMe engine.
- **Rust (`sys/rust/`, `Burn 0.22.1` + `CubeCL` Metal GPU + `Rayon`)**: Full 4-rung GPU/CPU training and sleep crystallization on Apple M5 Max Metal (`burn-mlx` / `wgpu` / `CubeCL`), emitting standalone KV260 Verilog (`artifacts/rust_clai_rung0_kv260.v`).
- **Zig `0.17.1` (`sys/zig/src/mol_baremetal.zig`)**: Bare-metal `std.Io.Threaded` cascade using `@Vector(8, f32)` SIMD and `@popCount` with strictly `0` heap bytes allocated inside the hot inference loop.
- **Go `1.27.4` (`sys/go/mol_dataplane.go`)**: Concurrent worker-pool L7 packet dataplane (`math/bits.OnesCount64`) verified at `0 B/op` and `0 allocs/op` under `go test -bench`.
- **Julia `1.13.4` (`sys/julia/mol_scientific.jl`)**: Exact reverse-mode automatic differentiation through smooth Walsh–Hadamard `DiffLUT6` gates via `Enzyme.jl`, matching central finite differences to `6.11 × 10⁻¹⁰`.

### 3.3 Kria KV260 (`xck26`) SystemVerilog RTL and `RAMB36E2` page cache

Figure 2 shows the hardware datapath implemented in `hw/kv260/rtl/` (`clai_kv260_axi_top.sv`, `clai_benes_hotswap_core.sv`, and `clai_benes_hotswap_core.sv`).

![Figure 2. Beneš–Copy–Beneš multicast fabric and single-cycle BRAM LUT6 page cache.](figures/benes-paged-silicon.png)

*Figure 2. `512×512` Beneš–Copy–Beneš multicast fabric and `81 × RAMB36E2` single-cycle `LUT6` page cache targeting AMD Kria KV260 (`xck26`). Synthesized in Yosys `0.71+` (`synth_xilinx -family xcup`) and cycle-simulated in Verilator `5.054`; `board_metered = false`.*

In real source-code or token workloads, a single static `LUT6` bank (`5,120` `INIT` bits for `80` dynamic `LUT6`s) holds only one local context, while reloading a shadow bank over a 64-bit AXI4-Lite bus stalls switching for `82` clock cycles (`328 ns` at `250 MHz`). Conversely, storing `512` pages in distributed flip-flops with a `512:1` combinational mux explodes to `926,619` CLB LUTs (`791%` of `xck26`).

`clai_benes_hotswap_core.sv` resolves both bottlenecks by mapping `512` resident pages (`5,120` `INIT` bits + `48`-bit context tag + `1` valid bit = `5,169` bits, aligned to `81 × 64`-bit words = `5,184` bits/page) across **`81` parallel `RAMB36E2` primitives** configured in `512 × 64` simple dual-port mode:
- **Port A (`AXI4-Lite` background fill)**: Asynchronous `SleepCrystallizer` page updates from the ARM Cortex-A53 PS without stalling the datapath.
- **Port B (`1`-cycle hash-indexed read)**: Computes a 9-bit XOR-fold hash of the 48-bit context, reads all `5,120` `INIT` bits + `48`-bit tag in **`1` clock cycle (`4.0 ns` at `250 MHz` nominal, `CPI = 1.0`)**, verifies exact 48-bit tag equality (`tag_hit`), and evaluates the `80` Beneš-routed `LUT6`s in the same pipeline beat.

### 3.4 `AirMoL`: Zero-extra-sweep 4-bit NVMe layer streaming on 2026 hybrid models

Inspired by `AirLLM` (2023), `AirMoL` (`src/clai/frontier/air_mol_streamer.py` and `hf_air_mol_pretrained.py`) executes models larger than available RAM budget by sharding weights layer-by-layer onto NVMe flash and loading one layer at a time. Standard `AirLLM` suffers from two compounding I/O taxes: (1) autoregressive `T=1` decoding sweeps the **entire model from NVMe once per generated token**, and (2) Transformer KV-caches spill to disk on long contexts.

![Figure 3. AirMoL 4-bit NVMe layer streaming with universal 64-bit token LUT6 speculation.](figures/airmol-nvme-speculation.png)

*Figure 3. `AirMoL` zero-extra-sweep 4-bit NVMe layer streaming with universal `≤64`-bit joint-mode token `LUT6` speculation on `LiquidAI/LFM2.5-350M` and `Qwen/Qwen3.5-0.8B-Base`. NVMe bytes, peak active RAM, exact token agreement, and wall-clock throughput are measured on Apple M5 Max; joules are unmetered.*

`AirMoL` eliminates neural draft-weight I/O and cuts per-token NVMe sweeps through four mechanisms:
1. **Group-wise 4-bit (`INT4`) NVMe sharding + unbuffered `F_NOCACHE` `os.pread`**: Quantizes linear projections in groups of `128` (`2` weights per `uint8` byte + FP16 scale/zero) and reads each shard via POSIX `os.pread` on file descriptors opened with Darwin `fcntl(fd, F_NOCACHE, 1)` so OS page-cache hits cannot mask physical SSD transfer volume.
2. **Universal $\le 64$-bit token `LUT6` burst drafter (`TokenPagedBenesLUT6Drafter`)**: Proposes $K \in \{3, 4\}$ BPE tokens from 3-token/2-token/1-token context suffixes in `0` FLOPs and `0` NVMe reads.
3. **Single-sweep batch verification (`K + 1` positions)**: Feeds `[last_token, draft_1, ..., draft_K]` through each streamed layer shard **once**. Matrix-matrix verification of $K+1 \le 5$ tokens adds negligible compute relative to NVMe PCIe transfer latency while verifying up to $K+1$ tokens in a single full-model sweep.
4. **Zero-extra-sweep hybrid state rollback**: Modern 2026 hybrid models interleave $O(1)$-state linear recurrence or bounded sliding windows (`ShortConv` + `SlidingAttn W=256` in October 2026 `LFM3-1.7B`; `Short-Conv` in `LFM2.5`; `Gated DeltaNet` in `Qwen3.5` / `Qwen3.6`) with sparse `GQA` full-attention layers (`6` of `24` layers in `LFM3-1.7B` and `Qwen3.5-0.8B-Base`; `10` of `40` layers in `Qwen3.6-35B-A3B`). Before each verification sweep, `AirMoL` snapshots the compact recurrent/window states in RAM and advances step-by-step through the $K+1$ positions inside each streamed layer, recording intermediate state snapshots `[s_0, ..., s_K]`. If draft token $d_{m+1}$ is rejected ($0 \le m \le K$), `AirMoL` restores snapshot `s_m` and truncates the sparse GQA KV-cache to `seq_len + m + 1` in microseconds — requiring **zero extra NVMe sweeps** on partial rejections.

### 3.5 `Expert-AirMoL` Sparse-MoE NVMe streaming, `Sub1Bit-LUT6`, and `VerifiedCodeCascade`

Scaling disk-offloaded inference from sub-`2B` dense hybrids to frontier `8.47B` and `34.66B` sparse Mixture-of-Experts (`MoE`) architectures exposes a second severe `AirLLM` bottleneck: standard layer-wise disk streaming reads **all `E` routed experts per layer** (`32` experts in `LiquidAI/LFM2.5-8B-A1B`, `256` experts in `Qwen/Qwen3.6-35B-A3B`) even though the router activates only $k \ll E$ experts per token (`4/32 = 12.5%` and `8/256 = 3.125%`).

![Figure 4. Expert-AirMoL router-sparse 4-bit NVMe streaming and Verifier-Gated VerifiedCodeCascade.](figures/moe-expert-cascade.png)

*Figure 4. `Expert-AirMoL` router-sparse 4-bit expert-indexed NVMe streaming combined with `BenesLUT6` burst speculation and the multi-rung `C(z)`-verified code synthesis and crystallization loop (`src/clai/frontier/moe_air_mol.py` and `verified_code_cascade.py`).*

`Expert-AirMoL` (`src/clai/frontier/moe_air_mol.py`), `Sub1Bit-LUT6` (`sub1bit_lut_factor.py`), and `VerifiedCodeCascade` (`verified_code_cascade.py`) close the frontier scaling and accuracy loop through three components:
1. **Router-indexed `pack4` expert `pread` + `BenesLUT6` union verification**: Each MoE layer splits into a compact non-expert header shard (norms, `ShortConv` / `Gated DeltaNet` or `GQA` mixer, shared expert, and `k`-of-`E` router gate) and a fixed-stride `INT4` block-64 `.experts.pack4` file on NVMe. During a $K+1$ burst verification step (`K = 3`), `Expert-AirMoL` runs the mixer and router first, forms the active union $\mathcal{U}_l = \bigcup_{t=0}^{K} \text{TopK}(h_{l,t})$ (`|\mathcal{U}_l| \approx 11\text{–}12` of `32` experts on `8.47B`; `|\mathcal{U}_l| \approx 33\text{–}38` of `256` experts on `34.66B`), and issues exact byte-offset `F_NOCACHE` `os.pread` calls **only** for $e \in \mathcal{U}_l$, multiplying router sparsity with `LUT6` sweep amortization (`6.90×` and `12.30×` NVMe read reductions at `100.0%` lossless token parity; Claim LA-9).
2. **Zero-MAC `Sub1Bit-LUT6` latent factorization (`0.576 bpw`)**: Samsung's `LittleBit` and `LittleBit-2` (Lee et al., 2025; Lee and Kim, 2026) factorize weights into multi-scale binary matrices $W \approx G \odot ((U_1 V_1^\top) s_1 + (U_2 V_2^\top) s_2)$ ($U_s \in \{-1,+1\}^{d_{\text{out}}\times r}, V_s \in \{-1,+1\}^{d_{\text{in}}\times r}$), but unpack `int32` bitpacks to execute dense $O(r(d_{\text{in}} + d_{\text{out}}))$ binary matrix multiplications at runtime and require `5` epochs of multi-GPU QAT. Choosing rank $r = 6G_{\text{lut}}$ (`258 = 43 × 6`), `Sub1BitLUT6FactorLinear` precomputes the `64` exact signed partial sums of every `6`-column binary block of $V_s$ and $U_s^\top$ into `6`-input `LUT6` tables, matching `LittleBit-2` Procrustes ITQ within `< 4 × 10⁻⁶` with **`0` binary-matrix MACs** while `MoL` target verification restores `100.0%` exact token parity in `0` QAT epochs (Claim LA-10).
3. **Verifier-gated `C(z)` escalation and online skill crystallization**: For novel synthesis tasks where `100%` parity with a single base model is insufficient when that model itself errs, `VerifiedCodeCascade` places a deterministic `0.35s` POSIX `ITIMER_REAL`-bounded AST + sandbox execution + boundary-invariant contract verifier $C(z) \in \{0,1\}$ across Rungs `0` (`512`-wire `BenesLUT6` spec signature router, `80` `LUT6`s), `1` (canonical AST verified skill store), `2` (`LiquidAI/LFM3-1.7B`), and `3a–3c` (`google/gemma-4-E2B-it` $\to$ `LiquidAI/LFM2.5-8B-A1B` $\to$ `Qwen/Qwen3.6-35B-A3B` with exact failing counterexample injection). Only candidates satisfying $C(z) = 1$ are committed and compiled downward into Rungs `0–1`; even on a Rung `0–1` hit, $C(z)$ executes before commit so specification drift safely triggers escalation with `0.0%` stale-cache admissions (Claim LA-11).

---

## 4. Claims and results

Every claim below links directly to an executable script and persisted artifact in `artifacts/`. Measurement classes follow Section 1 strictly.

### 4.1 Claim LA-1: Analytical `J/VC` collapse on synthetic tasks and real Wireshark `.pcap` traffic

`scripts/run_collapse_curve.py` (5,120 synthetic piecewise-stationary episodes with a regime shift at episode 2,560) and `scripts/run_realworld_domain.py` (5,120 real L7 enterprise packets in `artifacts/enterprise_drift_capture.pcap`, including a stealth VLAN `0x8100` shift at packet 2,500 and TLS anomaly burst at packet 4,200) evaluate downward `LUT6` compilation under deterministic `RCE` holdout verification ($\ge 98.5\%$).

| Workload & Window | Rung 0 `LUT6` Share | Verified Pass Rate | `LUT6` Holdout Parity | Analytical Energy (`est_j / VC`) | Verilator RTL Parity | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Synthetic ep. `1–512` (cold wake) | `0.00%` | `0.00%` $\to$ `51.95%` | `26.56%` | `2,323.18 est_j` | — | Software measured + `est_j` |
| Synthetic ep. `3,073–3,584` (crystallized) | `52.73%` | `60.55%` | `68.75%` | `2.59 est_j` (`895×` ratio) | `100.0%` (`1,000` vecs) | Software + Verilator sim + `est_j` |
| `.pcap` pkts `1–512` (cold IPv4 L7) | `0.00%` | `73.05%` | `100.00%` | `3.3351 est_j` | `100.0%` | Software + Verilator sim + `est_j` |
| `.pcap` pkts `1,537–2,048` (crystallized) | `100.00%` | `100.00%` | `100.00%` | `0.000140 est_j` | `100.0%` | Software + Verilator sim + `est_j` |
| `.pcap` pkts `2,049–2,560` (VLAN `0x8100` drift) | `2.73%` (breaker tripped) | `68.55%` | `100.00%` | `3.0485 est_j` | `100.0%` | Software + Verilator sim + `est_j` |
| `.pcap` pkts `2,561–3,072` (re-crystallized) | `100.00%` | `100.00%` | `100.00%` | `0.000131 est_j` (`25,458×` ratio) | `100.0%` (`5,120/5,120`) | Software + Verilator sim + `est_j` |

At packet `2,500`, the `RCE` / `HC-SOINN` circuit breaker detects the unannounced 4-byte IEEE `802.1Q` header offset (`96.9%` mismatch against the pre-drift `LUT6` bank), flushes the stale bank, wakes Rung 1/3 to adapt, and re-crystallizes a new `LUT6` bank within one sleep window. Note that the `25,458×` ratio compares `est_j` analytical counts across phases of the same model, not metered silicon watts.

### 4.2 Claim LA-2: Five-language polyglot parity and zero-allocation dataplanes

| Implementation | Target / Backend | Key Verified Metric | Evidence / Measurement |
| :--- | :--- | :--- | :--- |
| **Rust (`sys/rust/`)** | `Burn 0.22.1` + `CubeCL` Metal GPU + `Rayon` | `2,331.42` $\to$ `0.19 est_j/VC` (`12,270×` analytical ratio), `95.31%` Rung 0 share, emits valid KV260 `.v` | Measured execution + `est_j` (`artifacts/rust_collapse_curve_metrics.csv`) |
| **Zig `0.17.1` (`sys/zig/`)** | `@Vector(8, f32)` SIMD + `@popCount` | `0` bytes heap allocation in hot loop; `87 ns/pkt` (`11.49M pkts/s`); `100%` `LUT6` parity | Measured on Apple M5 Max |
| **Go `1.27.4` (`sys/go/`)** | Worker pool + `bits.OnesCount64` | `0 B/op`, `0 allocs/op`; `31.86 ns/op` (`31.39M evals/s`) | Measured via `go test -bench` |
| **Julia `1.13.4` (`sys/julia/`)** | `Enzyme.jl` reverse-mode AD | Max gradient discrepancy vs. central finite differences: `6.11 × 10⁻¹⁰` | Measured numerical check |

### 4.3 Claim LA-3: KV260 (`xck26`) Beneš–Copy–Beneš and `RAMB36E2` paged synthesis

`scripts/run_frontier_benchmarks.py` and `scripts/verify_paged_bram_rtl.py` synthesize competing reconfigurable `LUT6` fabrics via Yosys `0.71+` (`synth_xilinx -family xcup`, targeting `xck26`: `117,120` CLB LUTs, `144` `RAMB36E2`s) and verify cycle-accurate switching in Verilator `5.054`.

| RTL Architecture (`80` Dynamic `LUT6`s unless noted) | Resident Pages | Context Switch Latency | Synthesized `xcup` CLB LUTs (`% xck26`) | `RAMB36E2` (`% xck26`) | `DSP48` | Post-VLAN Drift Acc. | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `2A` Flat `512:1` Crossbar (`64` `LUT6`, `384` ports) | `1` | `82 cyc` (`328 ns`) | `119,478` (**`102.0%` OVERFLOW**) | `0` (`0.0%`) | `0` | `98.44%` | Simulated (Yosys `xcup`) |
| `2B` Compact Flat Crossbar (`32` `LUT6`, `192` ports) | `1` | `82 cyc` (`328 ns`) | `59,739` (`51.0%`) | `0` (`0.0%`) | `0` | `95.94%` | Simulated (Yosys `xcup`) |
| `2C` Fixed-Wire DiffLogic / WNN (`0` dynamic ports) | `1` | Vivado re-PAR (mins) | `160` (`0.14%`) | `0` (`0.0%`) | `0` | **`52.34%`** (fails drift) | Simulated (Yosys `xcup`) |
| `2D` Single-Page `512×512` Beneš–Copy–Beneš | `1` | `82 cyc` (`328 ns`) | `40,155` (`34.3%`) | `0` (`0.0%`) | `0` | `98.44%` | Simulated (Yosys + Verilator) |
| `P2` Distributed FF + `512:1` Mux (`512` pages) | `512` | `1 cyc` (`4.0 ns`) | `926,619` (**`791.2%` OVERFLOW**) | `0` (`0.0%`) | `0` | `98.44%` | Simulated / extrapolated |
| **`P3` Beneš–Copy–Beneš + `81` `RAMB36E2` (Ours)** | **`512`** | **`1 cyc` (`4.0 ns`, `CPI=1.0`)** | **`41,186` (`35.2%`)** | **`81` (`56.2%`)** | **`0`** | **`98.44%`** | **Simulated (Yosys + Verilator, `24/24` pass)** |

Adding `512` hardware-resident `LUT6` context pages via `81` `RAMB36E2` blocks adds only `+1,031` CLB LUTs (`+0.88%` of the KV260) over the single-page Beneš core while accelerating context switching `82×` (`82` cycles $\to$ `1` cycle) and leaving `64.8%` of CLB logic, `43.8%` of BRAM, and `100%` of `DSP48` slices free for co-resident accelerators.

### 4.4 Claim LA-4: Oracle-free latent `JEPA-EBRM` Split-Conformal + Online ACI gate

Evaluated in `scripts/run_frontier_benchmarks.py` (`artifacts/frontier3_conformal_baselines.csv`) across clean in-distribution traffic, `30%` noisy-boundary corruption, and `100%` unlabeled out-of-distribution (OOD) shift (`target α = 0.02`), without access to a deterministic ground-truth oracle:

| Crystallization Gate | Passes / Ep | False Crystallization Risk | OOD Detection AUROC | Safe ID Coverage | Crystallized `LUT6` Clean Acc. | Caught Unlabeled Drift? | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `3A` Softmax MSP ($\max p \ge 0.95$) | `1` | `17.62%` | `0.8552` | `90.93%` | `99.22%` | `False` | Measured in software |
| `3B` Temp-Scaled Margin ($p_{(1)} - p_{(2)} \ge 0.85$) | `1` | `17.66%` | `0.8422` | `91.12%` | `98.83%` | `False` | Measured in software |
| `3C` Helmholtz Logit Energy ($-T \log \sum e^{z_k/T}$) | `1` | `24.28%` | `0.8045` | `97.17%` | `99.80%` | `False` | Measured in software |
| `3D` Epistemic MC-Dropout (`5` stochastic passes) | `5` | `18.24%` | `0.8494` | `91.03%` | `99.61%` | `False` | Measured in software |
| **`3E` Latent `JEPA-EBRM` + Split-Conformal ACI (Ours)** | **`1`** | **`0.00%`** | **`1.0000`** | **`95.66%`** | **`99.02%`** | **`True`** | **Measured in software** |

### 4.5 Claim LA-5: Lossless byte-level speculative decoding on `520 KB` polyglot source code

`scripts/run_code_corpus_benchmark.py` (`artifacts/real_code_corpus_baselines.csv`) trains `Mamba3CodeLM` (`1.13M` params, `2.257 MFLOPs/byte`) and `CausalTransformerCodeLM` (`3.55M` params, `7.209 MFLOPs/byte`) on `34` real source files (`426,290` bytes across `.py`, `.rs`, `.sv`, `.zig`, `.go`, `.jl`) and evaluates lossless `64`-byte completions on `48` held-out prompts from `8` unseen files (`93,152` bytes):

| Decoding / Speculation Architecture | Lossless Parity | Mean Bytes / Pass | Passes for `64 B` | Draft FLOPs / Pass | Total MFLOPs / Byte | Analytical `est_mJ / Byte` | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `A` Causal Transformer (`RoPE` + `SwiGLU` + KV-Cache) | `100.0%` | `1.00` | `64.00` | `0` | `7.209` | `58.94 est_mJ` | Software measured + `est_j` |
| `B` `Mamba-3` Complex MIMO (`T=1` autoregressive) | `100.0%` | `1.00` | `64.00` | `0` | `2.257` | `20.84 est_mJ` | Software measured + `est_j` |
| `C` `PLD` In-Prompt Lookup (Saxena, 2023) | `100.0%` | `2.13` | `29.98` | `0` | `1.322` | `11.52 est_mJ` | Software measured + `est_j` |
| `D` `REST` Exact Suffix-Trie Retrieval (He et al., 2024) | `100.0%` | `3.84` | `16.65` | `0` | `0.734` | `6.40 est_mJ` | Software measured + `est_j` |
| `E` `Medusa` Parallel MLP Heads (Cai et al., 2024) | `100.0%` | `2.62` | `24.44` | `428,288` | `1.241` | `9.46 est_mJ` | Software measured + `est_j` |
| `F` `EAGLE-2` Autoregressive Neural Draft (Li et al., 2024) | `100.0%` | `2.75` | `23.29` | `1,122,560` | `1.435` | `9.07 est_mJ` | Software measured + `est_j` |
| **`G` `MoL` Paged Beneš `LUT6` (`4-Byte` Joint Minterm, Ours)** | **`100.0%`** | **`3.84`** | **`16.69`** | **`0`** | **`0.736`** | **`6.41 est_mJ`** | **Software measured + `est_j`** |

`PagedBenesLUT6` matches full-corpus `REST` (`3.84 B/pass`, `3.07×` fewer total FLOPs than `T=1` `Mamba-3` and `9.79×` fewer FLOPs than Causal Transformer) while replacing variable-latency CPU suffix-trie pointer chasing with fixed `64`-bit `LUT6 INIT` pages suitable for `1`-cycle `RAMB36E2` hardware lookup (`Claim LA-3`), and beating neural drafters (`Medusa`, `EAGLE-2`) with `0` draft FLOPs.

### 4.6 Claim LA-6: Controlled `AirMoL` 4-bit NVMe streaming (`14.3M`-parameter matched pair)

`scripts/run_air_mol_benchmark.py` (`artifacts/air_mol_nvme_baselines.csv`) compares parameter-matched `12`-layer models (`14.26M` Transformer vs. `14.30M` `Mamba-3`, `54.55 MB` FP32 $\to$ `7.82 MB` 4-bit NVMe shards = `6.98×` compression with `+0.0006` val-BPB delta) under Darwin `F_NOCACHE` `os.pread` across `64`-byte completions:

| Offloaded Streaming Architecture | Exact Parity | Bytes / Sweep | Sweeps / `64 B` | NVMe Read (`MB`) | NVMe KV Write (`MB`) | Peak Active RAM | Measured `ms / Byte` | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `1` `AirLLM` Transformer FP16 + KV Spill (`T=1`) | `100.0%` | `1.00` | `64.00` | `2,130.02` | `407.62` | `8.81 MB` | `60.64 ms` | Measured (`F_NOCACHE` I/O + RAM) |
| `2` `AirLLM` Transformer 4-Bit Prefetch + KV Spill (`T=1`) | `100.0%` | `1.00` | `64.00` | `887.55` | `407.62` | `8.81 MB` | `64.87 ms` | Measured (`F_NOCACHE` I/O + RAM) |
| `3` `AirMamba-3` 4-Bit + Swapped `EAGLE-2` Drafter | `100.0%` | `2.86` | `22.38` | `175.27` | `0.00` | `7.20 MB` | `19.20 ms` | Measured (`F_NOCACHE` I/O + RAM) |
| `4` `AirMamba-3` 4-Bit + Pinned $O(1)$ State (`T=1`) | `100.0%` | `1.00` | `64.00` | `488.30` | `0.00` | `7.20 MB` | `37.72 ms` | Measured (`F_NOCACHE` I/O + RAM) |
| **`5` `AirMoL` 4-Bit + Cold Paged `LUT6` (Ours)** | **`100.0%`** | **`3.74`** | **`17.12`** | **`130.66`** | **`0.00`** | **`7.51 MB`** | **`11.18 ms`** | **Measured (`F_NOCACHE` I/O + RAM)** |
| **`6` `AirMoL` 4-Bit + Post-Wake Crystallized (Ours)** | **`100.0%`** | **`4.27`** | **`15.00`** | **`114.44`** | **`0.00`** | **`7.51 MB`** | **`11.09 ms`** | **Measured (`F_NOCACHE` I/O + RAM)** |

Combining 4-bit weight sharding, pinned `48 KB` $O(1)$ `Mamba-3` recurrent state (`0 MB` KV spill), and post-wake `LUT6` burst speculation cuts total NVMe bytes transferred (`Read + Write`) from `2,537.6 MB` (`AirLLM` FP16) and `1,295.2 MB` (`AirLLM` 4-bit) down to **`114.4 MB`** (**`22.2×`** and **`11.3×`** total flash I/O reductions) while holding only `7.51 MB` in active RAM.

### 4.7 Claim LA-7: Pretrained 2026 dense & tri-hybrid foundation models under `AirMoL` (`LFM3-1.7B`, `LFM2.5-350M`, & `Qwen3.5-0.8B-Base`)

`scripts/run_hf_lfm3_air_mol.py`, `scripts/run_hf_air_mol_pretrained.py`, and `scripts/run_hf_qwen35_air_mol.py` evaluate three actual open-weights **2026** hybrid foundation models downloaded from HuggingFace and sharded into 4-bit NVMe layer files:
- **`LiquidAI/LFM3-1.7B`** (released October 23, 2026; `1,728,801,536` params, `24` Tri-Hybrid layers = `13` `ShortConv` $O(1)$ + `5` `SlidingAttn` $O(W=256)$ + `6` `FullAttn`, `|V| = 128,000` $\implies$ `17b × 3 = 51b` $\le 64$ in `uint64`; `6,594.85 MB` FP32 $\to$ `1,068.60 MB` 4-bit NVMe shards = `6.17×` compression; `24` BPE tokens/prompt).
- **`LiquidAI/LFM2.5-350M`** (released March 2026; `354,483,968` params, `16` layers = `10` Short-Conv/SSM + `6` GQA full-attention, `|V| = 65,536` $\implies$ `16b × 4 = 64b` in `uint64`; `1,352.25 MB` FP32 $\to$ `226.27 MB` 4-bit NVMe shards = `5.98×` compression; `32` BPE tokens/prompt).
- **`Qwen/Qwen3.5-0.8B-Base`** (released February 2026; `752,393,024` params, `24` layers = `18` Gated DeltaNet linear-attention + `6` gated full-attention, `|V| = 248,320` $\implies$ `18b × 3 = 54b` $\le 64$ in `uint64`; `2,870.15 MB` FP32 $\to$ `540.75 MB` 4-bit NVMe shards = `5.31×` compression; `24` BPE tokens/prompt).

| Pretrained 2026 Checkpoint & `AirMoL` Arm | Lossless Parity | Tokens / Sweep | Code Chars / Sweep | Sweeps / Gen | NVMe Read (`GB`) | Peak Active RAM (`% FP32`) | Measured `tok/s` (`chars/s`) | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `LFM3-1.7B` (Oct 2026) · `AirLLM` 4-Bit (`T=1`) | `100.0%` | `1.00` | `2.30` | `24.00` | `25.045 GB` | `250.0 MB` (`3.79%`) | `0.53` (`1.2`) | Measured (`F_NOCACHE` I/O + RAM) |
| `LFM3-1.7B` (Oct 2026) · `AirLLM` 4-Bit + `PLD` | `100.0%` | `1.37` | `3.16` | `17.50` | `18.262 GB` | `250.0 MB` (`3.79%`) | `0.73` (`1.7`) | Measured (`F_NOCACHE` I/O + RAM) |
| **`LFM3-1.7B` (Oct 2026) · `AirMoL` + Cold `BenesLUT6`** | **`100.0%`** | **`1.55`** | **`3.56`** | **`15.50`** | **`16.175 GB`** | **`250.0 MB` (`3.79%`)** | **`0.81` (`1.9`)** | **Measured (`F_NOCACHE` I/O + RAM)** |
| **`LFM3-1.7B` (Oct 2026) · `AirMoL` + Post-Wake `LUT6`** | **`100.0%`** | **`2.74`** | **`6.31`** | **`8.75`** | **`9.131 GB`** | **`250.0 MB` (`3.79%`)** | **`1.32` (`3.0`)** | **Measured (`F_NOCACHE` I/O + RAM)** |
| `LFM2.5-350M` · `AirLLM` FP16 (`T=1`) | `100.0%` | `1.00` | `3.11` | `32.00` | `25.129 GB` | `128.0 MB` (`9.47%`) | `2.21` (`6.9`) | Measured (`F_NOCACHE` I/O + RAM) |
| `LFM2.5-350M` · `AirLLM` 4-Bit Prefetch (`T=1`) | `100.0%` | `1.00` | `2.73` | `32.00` | `7.071 GB` | `128.0 MB` (`9.47%`) | `2.21` (`6.0`) | Measured (`F_NOCACHE` I/O + RAM) |
| `LFM2.5-350M` · `AirLLM` 4-Bit + `PLD` | `100.0%` | `1.13` | `3.10` | `28.20` | `6.231 GB` | `128.0 MB` (`9.47%`) | `2.71` (`7.4`) | Measured (`F_NOCACHE` I/O + RAM) |
| **`LFM2.5-350M` · `AirMoL` 4-Bit + Cold Token-`LUT6`** | **`100.0%`** | **`1.32`** | **`3.61`** | **`24.20`** | **`5.347 GB`** | **`128.0 MB` (`9.47%`)** | **`2.74` (`7.5`)** | **Measured (`F_NOCACHE` I/O + RAM)** |
| **`LFM2.5-350M` · `AirMoL` 4-Bit + Post-Wake `LUT6`** | **`100.0%`** | **`2.86`** | **`7.80`** | **`11.20`** | **`2.475 GB`** | **`128.0 MB` (`9.47%`)** | **`6.26` (`17.1`)** | **Measured (`F_NOCACHE` I/O + RAM)** |
| `Qwen3.5-0.8B` · `AirLLM` 4-Bit Prefetch (`T=1`) | `100.0%` | `1.00` | `2.72` | `24.00` | `12.674 GB` | `242.5 MB` (`8.45%`) | `0.53` (`1.4`) | Measured (`F_NOCACHE` I/O + RAM) |
| `Qwen3.5-0.8B` · `AirLLM` 4-Bit + `PLD` | `100.0%` | `1.32` | `3.58` | `18.25` | `9.637 GB` | `242.5 MB` (`8.45%`) | `0.50` (`1.4`) | Measured (`F_NOCACHE` I/O + RAM) |
| **`Qwen3.5-0.8B` · `AirMoL` 4-Bit + Cold Token-`LUT6`** | **`100.0%`** | **`1.48`** | **`4.02`** | **`16.25`** | **`8.581 GB`** | **`242.5 MB` (`8.45%`)** | **`0.32` (`0.9`)** | **Measured (`F_NOCACHE` I/O + RAM)** |
| **`Qwen3.5-0.8B` · `AirMoL` 4-Bit + Post-Wake `LUT6`** | **`100.0%`** | **`2.74`** | **`7.46`** | **`8.75`** | **`4.621 GB`** | **`242.5 MB` (`8.45%`)** | **`1.11` (`3.0`)** | **Measured (`F_NOCACHE` I/O + RAM)** |

On October 2026 `LiquidAI/LFM3-1.7B` (accelerated on Apple M5 Max MPS while streaming 4-bit layer shards via `F_NOCACHE` `os.pread`), cold `AirMoL` immediately improves both NVMe volume (`25.045 GB` $\to$ `16.175 GB`) and wall-clock speed (`0.53` $\to$ `0.81 tok/s`), and post-wake `AirMoL` cuts NVMe reads by **`2.74×`** (`25.045 GB` $\to$ **`9.131 GB`**) and reaches **`1.32 tok/s` (`2.49×` speedup)** while holding only **`250.0 MB` (`3.79%` of FP32)** in active weight RAM at `100.0%` exact parity. By contrast, an instructive negative result is preserved on CPU-bound `Qwen3.5-0.8B-Base` **cold** unseen prompts (`1.48 tok/sweep`), where `AirMoL` cuts NVMe bytes (`12.674 GB` $\to$ `8.581 GB`) yet `tok/s` drops (`0.53` $\to$ `0.32 tok/s`) until wake crystallization (`2.74 tok/sweep`) reaches `1.11 tok/s`.

### 4.8 Claim LA-8: Interactive CLI and HTTP/SSE streaming server (`mol_code_cli.py`)

`scripts/mol_code_cli.py` and `src/clai/service/live_code_engine.py` expose six local backends (`--backend {mamba3,lfm2.5-350m,qwen3.5-0.8b,lfm3-1.7b,lfm2.5-8b-a1b,qwen3.6-35b-a3b}`) and `--air-nvme` over both a terminal streaming CLI and a zero-dependency local HTTP/SSE server (`GET /v1/health`, `POST /v1/complete/stream`, `POST /v1/crystallize`), reporting per-step Rung `0` vs. Rung `3` share, `RAMB36E2` slot residency (`438/512` $\to$ `460/512`), bank generation (`Gen 1` $\to$ `Gen 2`), and real-time NVMe shard bytes read (`artifacts/live_cli_server_telemetry.json`; `22/22` tests passing across `tests/`).

### 4.9 Claim LA-9: Frontier Sparse-MoE `Expert-AirMoL` (`LiquidAI/LFM2.5-8B-A1B` `8.47B` & `Qwen/Qwen3.6-35B-A3B` `34.66B`)

`scripts/run_hf_moe_8b_35b_air_mol.py` (`artifacts/hf_2026_moe_8b_35b_air_mol_baselines.csv`) shards two real late-2026 sparse Mixture-of-Experts foundation models into header shards plus fixed-stride `INT4` block-64 `.experts.pack4` NVMe files and streams them under Darwin `F_NOCACHE` `os.pread` on Apple M5 Max:
- **`LiquidAI/LFM2.5-8B-A1B`** (`8,467,856,832` params, `24` layers = `18` $O(1)$ `Conv` + `6` `FullAttn`, `4`-of-`32` active experts, `16,151.16 MB` FP16 $\to$ `4,683.45 MB` 4-bit NVMe shards; whole-system macOS `PMU` energy delta across the full run: `9,237.23 J` over `479` ticks).
- **`Qwen/Qwen3.6-35B-A3B`** (`34,660,610,688` params, `40` layers = `30` $O(1)$ `Gated DeltaNet` + `10` `FullAttn`, `8`-of-`256` active experts + shared expert, `66,109.87 MB` FP16 $\to$ `18,595.19 MB` 4-bit NVMe shards; whole-system macOS `PMU` energy delta across the full run: `20,070.18 J` over `1,077` ticks).

| Sparse-MoE Checkpoint & Streaming Arm | Lossless Parity | Active Experts | Tokens / Sweep | NVMe Read / Tok | I/O Reduction vs. 4-Bit `AirLLM` | Peak Active RAM (`% FP16`) | Measured `tok/s` | Analytical `est_j / tok` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LFM2.5-8B-A1B` · `AirLLM` Dense 4-Bit (`T=1`) | `100.0%` | `100.00%` | `1.00` | `4.574 GB` | `1.00×` | `1,408.3 MB` (`8.72%`) | `0.28` | `368.80 est_j` |
| `LFM2.5-8B-A1B` · `Expert-AirLLM` Router-Sparse (`T=1`) | `100.0%` | `16.93%` | `1.00` | `1.200 GB` | `3.81×` | `1,408.3 MB` (`8.72%`) | `0.58` | `181.26 est_j` |
| `LFM2.5-8B-A1B` · `Expert-AirMoL` + `PLD` | `100.0%` | `28.82%` | `1.78` | `0.883 GB` | `5.18×` | `1,408.3 MB` (`8.72%`) | `0.79` | `132.57 est_j` |
| **`LFM2.5-8B-A1B` · `Expert-AirMoL` + Cold `BenesLUT6`** | **`100.0%`** | **`33.91%`** | **`1.85`** | **`1.002 GB`** | **`4.56×`** | **`1,408.3 MB` (`8.72%`)** | **`0.72`** | **`145.42 est_j`** |
| **`LFM2.5-8B-A1B` · `Expert-AirMoL` + Post-Wake `LUT6`** | **`100.0%`** | **`36.46%`** | **`3.00`** | **`0.663 GB`** | **`6.90×`** | **`1,408.3 MB` (`8.72%`)** | **`1.10` (`3.93×`)** | **`95.67 est_j`** |
| `Qwen3.6-35B-A3B` · `AirLLM` Dense 4-Bit (`T=1`) | `100.0%` | `100.00%` | `1.00` | `18.159 GB` | `1.00×` | `3,214.7 MB` (`4.86%`) | `0.06` | `1,696.81 est_j` |
| `Qwen3.6-35B-A3B` · `Expert-AirLLM` Router-Sparse (`T=1`) | `100.0%` | `6.35%` | `1.00` | `2.356 GB` | `7.71×` | `2,530.7 MB` (`3.83%`) | `0.22` | `471.87 est_j` |
| `Qwen3.6-35B-A3B` · `Expert-AirMoL` + `PLD` | `100.0%` | `11.58%` | `1.89` | `1.708 GB` | `10.63×` | `2,530.7 MB` (`3.83%`) | `0.32` | `326.24 est_j` |
| **`Qwen3.6-35B-A3B` · `Expert-AirMoL` + Cold `BenesLUT6`** | **`100.0%`** | **`12.91%`** | **`2.00`** | **`1.722 GB`** | **`10.54×`** | **`2,530.7 MB` (`3.83%`)** | **`0.31`** | **`334.42 est_j`** |
| **`Qwen3.6-35B-A3B` · `Expert-AirMoL` + Post-Wake `LUT6`** | **`100.0%`** | **`14.97%`** | **`2.57`** | **`1.477 GB`** | **`12.30×`** | **`2,530.7 MB` (`3.83%`)** | **`0.39` (`6.50×`)** | **`267.08 est_j`** |

Even though verifying $K+1 = 4$ tokens per sweep slightly widens the active expert union (`6.35%` $\to$ `14.97%` of `256` experts on `Qwen3.6-35B-A3B`), amortizing layer headers, shared experts, and router dispatch over `2.57` accepted tokens/sweep cuts net NVMe transfer from `18.159 GB/tok` (`~66.1 GB/tok` in FP16) down to **`1.477 GB/tok`** (**`12.30×` vs. 4-Bit `AirLLM`**, **`~44.8×` vs. FP16 `AirLLM`**) and accelerates throughput **`6.50×`** (`0.06` $\to$ **`0.39 tok/s`**) in **`2.53 GB` active RAM (`3.83%` of FP16)** with `100.0%` lossless token parity.

### 4.10 Claim LA-10: Sub-1-bit latent factorization (`LittleBit-1 / LittleBit-2`) vs. zero-MAC `Sub1Bit-LUT6` and `MoL`

`scripts/run_sub1bit_lut_benchmark.py` (`artifacts/sub1bit_littlebit_vs_mol_lut6.csv`) evaluates clean-room sub-1-bit (`0.576 bpw` = `27.8×` vs. FP16, rank `r = 258 = 43 × 6` `LUT6` groups) latent binary factorizations directly on real `LiquidAI/LFM2.5-8B-A1B` `layers.2.feed_forward.experts.0.w1.weight [1792, 2048]`:

| Weight / Draft Compression Method (`[1792, 2048]` Real Expert Matrix) | Effective `bpw` (`× FP16`) | Weight SNR (`dB`) | Output Cosine Sim | `16K`-Slice Top-1 Match | Binary Matmul MACs / Tok | QAT Epochs Needed | Evidence / Measurement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `A` Unrotated `SVD` Binary SVID (`LittleBit-1` PTQ style) | `0.576` (`27.8×`) | `1.76 dB` | `0.5761` | `3.12%` | `1,981,440` | `0` (`5` in paper) | Measured in software |
| `B` Procrustes `ITQ` Hypercube SVID (`LittleBit-2` PTQ style) | `0.576` (`27.8×`) | `2.42 dB` | `0.6487` | `10.94%` | `1,981,440` | `0` (`5` in paper) | Measured in software |
| **`C` OpenIE `Sub1Bit-LUT6` Table Lookup (`0` MACs, Draft)** | **`0.576` (`27.8×`)** | **`2.42 dB`** | **`0.6487`** | **`10.94%`** | **`0`** | **`0`** | **Measured (`< 4e-6` vs. `B`)** |
| `D` OpenIE `Expert-AirMoL` `INT4` Block-64 Target | `4.500` (`3.56×`) | `20.77 dB` | `0.9958` | `82.81%` | `0` (dense `INT4`) | `0` | Measured in software |
| **`E` OpenIE `MoL`: `Sub1Bit-LUT6` Draft + Target Verify (Ours)** | **`0.576` draft** | **`148.84 dB` (FP32)** | **`1.0000`** | **`100.00%`** | **`0` draft MACs** | **`0`** | **Measured lossless parity** |

Rotating latent SVD factors onto hypercube vertices (`LittleBit-2` principle; Lee and Kim, 2026) raises PTQ top-1 match `3.5×` (`3.12%` $\to$ `10.94%`), and OpenIE's `Sub1Bit-LUT6` (`Arm C`) reproduces `Arm B` within `< 4 × 10⁻⁶` max error while eliminating all `1,981,440` runtime binary MACs via `6`-input `LUT6` partial-sum lookup. However, standalone `0.55–0.58 bpw` models remain lossy even after multi-GPU QAT (`0.00%` verified pass on exact algorithmic coding in Claim LA-11), whereas placing `Sub1Bit-LUT6` / `BenesLUT6` at Rung `0` beneath `Expert-AirMoL` target verification (`Arm E`) achieves **`100.00%` lossless fidelity** with **`0` QAT epochs**.

### 4.11 Claim LA-11: `48`-task (`16` specs × `3` waves) late-2026 head-to-head frontier coding & verification bakeoff

`scripts/run_frontier_ai_bakeoff.py` (`artifacts/frontier_ai_head_to_head_bakeoff.csv`, `artifacts/frontier_ai_head_to_head_traces.json`) compares seven real late-2026 systems on `48` algorithmic, protocol, and systems coding tasks (`16` specifications across RFC-1982 serial arithmetic, weighted interval sweep-lines, lexicographic cycle-safe toposort, integer nanosecond token buckets, CRC-8/ATM, RFC-6901 JSON Pointer `~01` unescaping, Bellman–Ford `≤k`-hop paths, `>2⁵³` exact window medians, unrestricted Damerau–Levenshtein, `4×4` Beneš routing, SQL literal parameterization, monotonic-deque subsequence sum, DWARF SLEB128, SHA-256 consistent hashing, and finite-sample conformal quantiles) across **Wave 1** (`16` cold unseen specs), **Wave 2** (`14` refactored prompts with held-out tests + `2` adversarial specification mutations on `serial_lt` and `crc8_check`), and **Wave 3** (`16` production replay tasks). Whole-system macOS `PMU` energy delta across the benchmark was `117,166.40 J` (`6,646` ticks).

| Late-2026 System Arm (`48` Tasks = `16` Specs × `3` Waves) | W1 Cold | W2 Adapt | W3 Warm | Overall Pass | Broken Commits | W3 `Rung 0/1` Share | `35B` Calls | Mean W1 / W3 (`ms`) | `reported_j` / Solve (`est_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A` `LittleBit-2` `Qwen3-8B` `0.55 bpw` QAT (`1`-Shot) | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `100.00%` | `0.00%` | `0.0%` | `22,413` / `22,708` | `8,020.13` (`32,558 est_j`) |
| `B` `LiquidAI/LFM3-1.7B` (Oct 2026 Tri-Hybrid, `1`-Shot) | `43.75%` | `25.00%` | `25.00%` | `31.25%` | `68.75%` | `0.00%` | `0.0%` | `8,460` / `9,791` | `152.77` (`957.21 est_j`) |
| `C` `google/gemma-4-E2B-it` (`2026` Hybrid, `1`-Shot) | `37.50%` | `18.75%` | `18.75%` | `25.00%` | `75.00%` | `0.00%` | `0.0%` | `18,924` / `23,578` | `5,800.68*` (`3,700.44 est_j`) |
| `D` `LiquidAI/LFM2.5-8B-A1B` (`8.47B` MoE, `1`-Shot) | `6.25%` | `6.25%` | `6.25%` | `6.25%` | `93.75%` | `0.00%` | `0.0%` | `53,123` / `53,885` | `33,415.04` (`47,196 est_j`) |
| `E` `Qwen/Qwen3.6-35B-A3B` Metal `AirMoL` (`1`-Shot) | `81.25%` | `75.00%` | `75.00%` | `77.08%` | `22.92%` | `0.00%` | `100.0%` | `49,887` / `44,447` | `1,069.65` (`5,701.32 est_j`) |
| `F` `RouteLLM` Confidence Router (`LFM3` $\to$ `Gemma4` $\to$ `8B`) | `43.75%` | `25.00%` | `25.00%` | `31.25%` | `68.75%` | `0.00%` | `0.0%` | `9,735` / `11,294` | `171.09` (`1,148.96 est_j`) |
| **`G` OpenIE `MoL` `VerifiedCodeCascade` (Ours)** | **`87.50%`** | **`87.50%`** | **`87.50%`** | **`87.50%`** | **`0.00%`** | **`87.50%`** | **`22.92%`** | **`116,626` / `44,052` (`2.89 ms` solved!)** | **`3,655.51*` (`0.0424 J` in W3!)** |

Four findings emerge directly from `artifacts/frontier_ai_head_to_head_traces.json`:
1. **`87.50%` verified solve rate (`42 / 48`) using `35B` on only `22.92%` (`11 / 48`) of tasks**: Eliminating `220`-token mid-AST truncation (`640–768` token budgets), respecting native chat templates (`<|channel>thought...<channel|>` on `Gemma 4`, `<think></think>` on `LFM3` and `Qwen3.6`), and passing structured counterexample + buggy-draft repair lifted `OpenIE MoL` from `43.75%` to **`87.50%` across all three waves**, beating standalone `Qwen3.6-35B-A3B` (`77.08%`, `100%` `35B` calls) by **`+10.42` percentage points** while avoiding `35B` entirely on `77.08%` of tasks (`LFM3-1.7B` solved `7` Wave-1 specs directly; `Gemma-4-E2B-it` + repair solved `3`; `Metal-Qwen3.6-35B-A3B` solved `4` plus mutated `W2_01`).
2. **`0.00%` broken commits admitted**: Every ungated standalone model and confidence router (`RouteLLM`) admitted **`22.92%–100.00%` broken commits**, whereas `MoL` abstained cleanly on the `2` unsolved specs (`F10_true_damerau` and `F11_benes4_route`) with **`0.00%` broken commits**.
3. **`100%` adversarial Wave-2 specification-drift safety & re-crystallization**: Both Wave-2 adversarial mutations (`W2_01_F01_rfc1982_serial_MUTATED` and `W2_05_F05_crc8_atm_MUTATED`) were caught by $C(z)$ (`0` stale admissions) and re-solved into `Skill#15` and `Skill#16` (`16` total verified skills in `artifacts/.mol_verified_skills`), replaying in **`2.70 ms`** in Wave 3.
4. **`15,401×` Wave-3 speedup & `25,227×` lower `reported_j` on crystallized solves**: Even across all `16` Wave-3 tasks (including the `2` re-escalated abstentions), `MoL` (`44,052 ms`) is faster than standalone `Qwen3.6-35B-A3B` (`44,447 ms`); across the **`14` crystallized Wave-3 solves (`87.50%`)**, mean latency collapses to **`2.89 ms`** and on-chip `reported_j` collapses to **`0.0424 J`** (vs. `1,069.65 J` on standalone `35B`). *(Footnote `*`: One `W2_10` `Gemma-4` invocation experienced an unprivileged macOS `IOReport` `'GPU Energy'` hardware-accumulator rollover spike of `+59.2 kJ`, retained unedited in `frontier_ai_head_to_head_bakeoff.csv` and filtered by `joule_meter.py`'s `> 450 W` rollover guard in subsequent runs.)*

### 4.12 Claim LA-12: Native Apple Metal (`MSL`) fused `INT4` Sparse-MoE & `GatedDeltaNet` shaders + dual-channel `reported_j`

`src/clai/frontier/metal_moe_kernels.py` and `scripts/run_metal_moe_benchmark.py` (`artifacts/metal_moe_air_mol_speedup.csv`, `artifacts/metal_moe_air_mol_meta.json`) compile three custom Metal Shading Language (`MSL`) kernels via `torch.mps.compile_shader`: (i) `gated_delta_recurrent_k128`, exploiting exact column independence across $v \in \{0, \dots, V_{\text{dim}}-1\}$ of recurrent state $\mathbf{S}[b, h_v, :, v]$ to hold `128` `FP32` state elements in GPU registers across all $T$ tokens (**`20.66×` faster**: `34.523` $\to$ `1.671 ms`, **`46.2×` lower `reported_j`**: `1.4989` $\to$ `0.0325 J`, `1.19 × 10⁻⁷` max state error); and (ii) `q4_moe_gate_up_silu` + `q4_moe_down_weighted`, dequantizing `4.50 bpw` `INT4` block-64 expert weights directly inside `simd_sum` registers with zero `FP16/FP32` weight allocations (**`43.45×` faster**: `22.408` $\to$ `0.516 ms` on `28` active experts, **`29.8×` lower `reported_j`**: `0.3959` $\to$ `0.0133 J`, `7.2 × 10⁻⁷` max `FP32` error), paired with Queue-Depth-`8` parallel `pread` and `BenesLUT6` sign-sketch router lookahead:

| Sparse-MoE Checkpoint & Cold-NVMe Engine Variant | Exact Token Parity | Tok / Sweep | NVMe Read / Tok | Peak Active RAM | `tok/s` (`× PyTorch`) | On-Chip `reported_j / tok` (`GPU` + `CPU`) | Analytical `est_j / tok` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LFM2.5-8B-A1B` · `1` PyTorch-Ref `Expert-AirMoL` (`T=1`) | `100.0%` | `1.00` | `1.257 GB` | `1,408.3 MB` | `0.394` (`1.00×`) | `10.978 J` (`5.25` + `5.73`) | `266.66 est_j` |
| `LFM2.5-8B-A1B` · `2` Metal-Fused `Expert-AirMoL` (`T=1`) | `100.0%` | `1.00` | `1.257 GB` | **`500.0 MB`** | `0.689` (`1.75×`) | `6.979 J` (`1.30` + `5.68`) | `152.42 est_j` |
| **`LFM2.5-8B-A1B` · `3` Metal-Fused + `BenesLUT6` (`K=3`, Ours)** | **`100.0%`** | **`2.67`** | **`0.791 GB`** | **`500.0 MB`** | **`1.671` (`4.24×`)** | **`3.493 J` (`1.22` + `2.28`)** | **`62.85 est_j`** |
| `Qwen3.6-35B-A3B` · `1` PyTorch-Ref `Expert-AirMoL` (`T=1`) | `100.0%` | `1.00` | `2.353 GB` | `2,434.7 MB` | `0.191` (`1.00×`) | `20.765 J` (`8.76` + `12.00`) | `549.49 est_j` |
| `Qwen3.6-35B-A3B` · `2` Metal-Fused `Expert-AirMoL` (`T=1`) | `100.0%` | `1.00` | `2.353 GB` | **`970.0 MB` (`1.47%`)** | `0.241` (`1.26×`) | `15.440 J` (`3.93` + `11.51`) | `436.26 est_j` |
| **`Qwen3.6-35B-A3B` · `3` Metal-Fused + `BenesLUT6` (`K=3`, Ours)** | **`100.0%`** | **`2.67`** | **`1.447 GB`** | **`970.0 MB` (`1.47%`)** | **`0.621` (`3.25×`)** | **`6.773 J` (`2.31` + `4.47`)** | **`169.03 est_j`** |

Because `q4_moe_*` never unpacks `INT4` experts into `FP16/FP32` buffers, cold-NVMe Peak Active Weight RAM on `Qwen/Qwen3.6-35B-A3B` drops **`2.51×`** from `2,434.7 MB` down to **`970.0 MB` (`1.47%` of the `66.11 GB` FP16 model)** while cutting on-chip `reported_j / tok` **`3.07×`** (`20.765` $\to$ **`6.773 J/tok`**) at `100.0%` exact token parity (and reaching **`15.7 tok/s`** when `16.8 GB` `INT4` expert packs remain pinned in Unified Memory on `>= 36 GiB` workstations).

### 4.13 Claim LA-13: `16`-task (`8` multi-file git repositories × `2` waves) `SWE-Mini` `pytest` patching benchmark

`src/clai/frontier/repo_patch_cascade.py` and `scripts/run_repo_patch_benchmark.py` (`artifacts/repo_patch_swebench_mini.csv`, `artifacts/repo_patch_swebench_mini_traces.json`) evaluate real cross-module repository bug-fixing across `8` isolated git repositories (`billing`, `scheduler`, `auth`, `wal`, `http_client`, `semver`, `ttlcache`, `etl`) × `2` waves (`16` evaluations, including a Wave-2 security specification mutation on `R03_jwt_clock_skew` adding future-`iat` rejection), where $C(z)$ enforces byte-identical `tests/` and executes real subprocess `pytest -q`:

| Multi-File Git-Repository Arm (`16` Tasks = `8` Repos × `2` Waves) | W1 Cold | W2 Warm | Overall Pass | Broken Patches Committed | W2 `.mol/` Hit Share | Mean W1 / W2 (`ms`) | `reported_j` / Verified Fix (`est_j`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A` `LiquidAI/LFM3-1.7B` (`1`-Shot Repo Patch) | `25.00%` | `25.00%` | `25.00%` | `75.00%` | `0.00%` | `4,943` / `5,020` | `371.61 J` (`637.68 est_j`) |
| `B` `google/gemma-4-E2B-it` (`1`-Shot Repo Patch) | `62.50%` | `62.50%` | `62.50%` | `37.50%` | `0.00%` | `6,865` / `7,117` | `133.48 J` (`469.79 est_j`) |
| `C` `LiquidAI/LFM2.5-8B-A1B` (`1`-Shot Repo Patch) | `0.00%` | `0.00%` | `0.00%` | `100.00%` | `0.00%` | `61,430` / `61,416` | `17,882.33 J` (`54,052 est_j`) |
| `D` `Qwen/Qwen3.6-35B-A3B` Metal `AirMoL` (`1`-Shot Repo Patch) | `62.50%` | `62.50%` | `62.50%` | `37.50%` | `0.00%` | `20,270` / `20,502` | `288.39 J` (`3,098.71 est_j`) |
| **`E` OpenIE `MoL` `RepoPatchCascade` (Ours)** | **`87.50%`** | **`87.50%`** | **`87.50%`** | **`0.00%`** | **`75.00%`** | **`59,341` / `30,663` (`594 ms` `.mol/`)** | **`692.38 J` (`1.10 J` on `.mol/` hits!)** |

`RepoPatchCascade` solves **`87.50%` (`14 / 16`)** of multi-file repository bugs (`+25.0` percentage points over standalone `35B` and `Gemma 4`), commits **`0.00%` broken patches**, catches the Wave-2 `R03` specification mutation via `pytest` $C(z)$ and re-crystallizes `Patch#8` (`2,905 ms`), and replays the `6` recurring Wave-2 `.mol/` patches in **`594.0 ms` (`1.10 J` `reported_j`, including `git init` + subprocess `pytest`)** vs. `38,937.5 ms` (`590.38 J` `reported_j`) in Wave 1 (**`65.6×` faster, `537×` lower `reported_j`**).

### 4.14 Claim LA-14: Drop-in OpenAI (`/v1/chat/completions`) and `MCP` (`JSON-RPC 2.0`) local coding agent with persistent `.mol/` skills

`src/clai/service/openai_mcp_agent.py` and `scripts/mol_agent_server.py` (`mol-agent` CLI entrypoint) package the cascade for immediate IDE/agent integration (`Cursor`, `Continue.dev`, `Claude Code`, `Antigravity`, `OpenAI SDK`) with zero third-party web dependencies: (i) OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions` (`stream: true/false` SSE, optional `x_mol_contract` deterministic verifier, and `x_mol_receipt` reporting rung, `skill_id`, `est_j`, `reported_j`, `measured_j: null`); and (ii) Model Context Protocol (`MCP` JSON-RPC `2.0` over `--mcp-stdio` and `POST /mcp`) exposing `mol_synthesize_verified_function`, `mol_repair_repo_pytest`, `mol_lookup_crystallized_skill`, and `mol_energy_telemetry` backed by persistent `.mol/skills_v1.json` + `.mol/benes_lut6_router.npz` (`26 / 26` `pytest` tests passing).

---

## 5. Related work and convergence observations

### 5.1 Boolean LUT fabrics, ternary lookup, and multi-model cascade routers

| System | ID / Venue | What it establishes | Energy or cost as reported | Difference here |
| :--- | :--- | :--- | :--- | :--- |
| **LogicNets** (Umuroglu et al., 2020) | `arXiv:2004.03021` · FPL 2020 | Compiles sparse quantized neural layers directly into FPGA truth tables (`LUT`s). | Authors report FPGA LUT/Fmax synthesis. | Fixed wire connectivity; any drift requires full vendor re-synthesis (`Vivado`), whereas our `512×512` Beneš + `RAMB36E2` core switches pages in `1` cycle (`0` re-PAR). |
| **DiffLogic** (Petersen et al., 2022) & **DWN** (Bacellar et al., 2024) | `arXiv:2210.08277` (NeurIPS 2022) · `arXiv:2410.11112` | Differentiable relaxation of Boolean gates / weightless LUT networks trained end-to-end. | Gate counts, throughput, and classification accuracy. | Offline whole-model classifiers using independent output gates; fail on unlabeled topological drift (`52.34%` in Claim LA-3) and suffer marginal bit-plane chimeras (Proposition 3). |
| **T-MAC** (Wei et al., 2024) & **TeLLMe** (Qiao et al., 2025) | `arXiv:2407.00088` · `arXiv:2504.16266` | Accelerates low-bit / ternary (`BitNet b1.58`, `arXiv:2402.17764`) matrix products via LUT lookup on CPUs/FPGAs. | Latency and FPGA power/efficiency reported by authors. | Accelerates dense linear algebra **inside** a model gear; does not bypass model invocation or compile wake traces downward into `0`-FLOP `LUT6` circuits. |
| **LittleBit** (Lee et al., 2025) & **LittleBit-2** (Lee and Kim, 2026) | `arXiv:2506.13771` (NeurIPS 2025) · `arXiv:2603.00042` (ICML 2026) | Compresses LLMs to `0.1–0.58 bpw` via multi-scale binary latent matrix factorization (`SVD` / Procrustes `ITQ` + `5`-epoch QAT). | Compression ratio (`11.6×–31×`), PPL, and zero-shot accuracy after multi-GPU QAT. | Unpacks `int32` bitpacks to dense `O(r(d_in+d_out))` binary MACs at runtime and loses exact code fidelity (`0.00%` pass at `0.55 bpw` in Claim LA-11); `Sub1Bit-LUT6` uses `0` binary MACs and `MoL` restores `100.0%` exact parity in `0` QAT epochs (Claim LA-10). |
| **FrugalGPT** (Chen et al., 2023) & **RouteLLM** (Ong et al., 2024) | `arXiv:2305.05176` · `arXiv:2406.18665` (ICLR 2025) | Routes queries from weak to strong LLMs using learned scorers or preference data. | API dollar cost vs. benchmark quality. | Model-to-model routing without `C(z)` contract verification (`89.58%` broken commits admitted in Claim LA-11), no `LUT6` floor, and no downward crystallization. |
| **GreenServ** (2026) & **VoltanaLLM** (2025) | `arXiv:2601.17551` (ICPE 2026) · `arXiv:2509.04827` | Multi-LLM bandit routing and disaggregated frequency/state-space serving under GPU energy meters (`Zeus` / `pyNVML`). | **Metered** GPU Wh / joules reported by authors on their serving pools. | Optimizes which LLM or GPU frequency runs; `lut-adapt-ai` compiles recurring patterns out of floating-point execution altogether (our joules remain `est_j`, not GPU-metered). |
| **Metered Commit Gates** & **RCE-JEPA** (Charlot, 2026) `[same house]` | `research.openie.dev/papers/gates/` · `/papers/rce-jepa/` | Defines commit gates (`11,428 pJ` KV260 `INA260` SOM reading), `Mixture of Limits` $p_i/c_i$ law, and `RCE` + `JEPA` world-coding boundary. | KV260 `INA260` SOM `measured_j` (Gates Part I) + OpCounter `est_j`. | Same-house foundation; this paper adds the dynamic `512×512` Beneš + `81`-`RAMB36E2` `LUT6` downward compiler, Split-Conformal ACI wake gate, and `AirMoL` 2026 NVMe streaming. |

### 5.2 Speculative decoding, disk-offloaded LLM inference, and conformal gating

| System | ID / Venue | What it establishes | Energy or cost as reported | Difference here |
| :--- | :--- | :--- | :--- | :--- |
| **Speculative Decoding** (Leviathan et al., 2023) | `arXiv:2211.17192` · ICML 2023 | Lossless target-verified parallel token acceptance from a fast draft distribution. | Wall-clock latency speedup. | Assumes a neural draft model in memory; we replace the neural drafter with `0`-FLOP `uint64` joint-minterm `LUT6` pages. |
| **Medusa** (Cai et al., 2024) & **EAGLE-2** (Li et al., 2024) | `arXiv:2401.10774` (ICML 2024) · `arXiv:2406.16858` (EMNLP 2024) | Parallel MLP heads and context-aware dynamic draft trees using target hidden states. | Wall-clock speedup (`~2×–4×`) on GPU-resident LLMs. | Require floating-point draft weights (`0.43M–1.12M` FLOPs/pass in Claim LA-5) and extra weight reads if offloaded (`175.3 MB` vs. `114.4 MB` in Claim LA-6). |
| **PLD** (Saxena, 2023) & **REST** (He et al., 2024) | Prompt Lookup (2023) · `arXiv:2311.08252` (NAACL 2024) | Zero-FLOP n-gram lookup in the prompt (`PLD`) or a datastore suffix trie (`REST`). | Latency speedup; `0` draft FLOPs. | `PLD` fails when completions are absent from the prompt (`1.13–1.32 tok/sweep` vs. `2.74–2.86` post-wake in Claim LA-7); `REST` uses variable-depth pointer tries rather than fixed `64`-bit `RAMB36E2` `LUT6` pages with online wake crystallization. |
| **HCSpec** (2026) & **CAS-Spec** (2025) | `10.18653/v1/2026.acl-long.353` · `arXiv:2510.26843` (NeurIPS 2025) | Two-tier horizontal draft cascades and self-speculative target-sparsity hierarchies. | Latency speedup; lossless tokens. | Draft hierarchies remain neural floating-point sub-networks inside the Model gear rather than Boolean `LUT6` memory pages. |
| **AirLLM** (lyogavin, 2023) | Open-source layer-wise SSD streaming | Runs large LLMs in tiny RAM by streaming one quantized layer shard per step from disk. | Memory footprint (`4 GB` GPU RAM for `70B`). | Pure `T=1` decoding (`1.00 tok/sweep`) with Transformer KV-cache spill; `AirMoL` adds `≤64`-bit token-`LUT6` burst speculation and `0`-extra-sweep hybrid state rollback (`2.74×–2.86×` fewer NVMe reads). |
| **Split Conformal & ACI** (Vovk et al., 2005; Lei et al., 2018; Gibbs and Candès, 2021) | Algorithmic Learning in a Random World · *JASA* 2018 · `arXiv:2106.00170` (NeurIPS 2021) | Distribution-free marginal coverage and online $\alpha_t$ adaptation under arbitrary shift. | Coverage guarantees (`1 - α`). | Statistical prediction sets for end-user predictions; we wire Split-Conformal + ACI over a radially regularized `JEPA-EBRM` score specifically to prevent false hardware `LUT6` crystallization (`0.00%` false risk). |

---

## 6. Limits, threats to validity, not claimed, and falsifiers

**Internal validity.** Synthetic (`Claim LA-1`) and conformal (`Claim LA-4`) benchmarks use controlled regime shifts where stationary windows exceed the `SleepCrystallizer` buffer threshold ($N_{\text{wake}}$). Under uniform i.i.d. random noise (`entropy = 1 bit/bit`) or per-step adversarial switching ($\Delta T = 1$), Proposition 1 predicts **zero downward crystallization gain** (`p_0' \approx 0`), and `PagedBenesLUT6` falls back cleanly to `1.00 tok/sweep` (plus single-step Rung 0 lookup overhead). On `Qwen3.5-0.8B-Base` cold unseen prompts (`Claim LA-7`), CPU verification of `K+1 = 4` tokens lowered wall-clock throughput (`0.32` vs. `0.53 tok/s`) even while reducing NVMe bytes read (`8.581 GB` vs. `12.674 GB`), demonstrating that batch verification is a net latency win on CPU only when draft acceptance exceeds the target's per-token compute-to-I/O crossover (`2.74 tok/sweep` post-wake reaches `1.11 tok/s`).

**Construct validity.** Exact parity (`100.0%` in Claims LA-5, LA-6, LA-7) proves that `AirMoL` and `PagedBenesLUT6` reproduce the **greedy output of the target model** bit-for-bit; it does not make a `350M` or `0.8B` base model infallible on open-ended software engineering benchmarks (`SWE-bench`). Likewise, `Conformal-bounded` (`1 - α`) bounds false crystallization relative to the calibrated `JEPA-EBRM` manifold, not physical plant safety (`ModelGenerated ↛ Deterministic`).

**External validity.** Foundation-model streaming and coding benchmarks (`Claims LA-7, LA-9, LA-11`) were executed across six pretrained 2026 checkpoints (`LFM2.5-350M`, `Qwen3.5-0.8B-Base`, `LiquidAI/LFM3-1.7B`, `google/gemma-4-E2B-it`, `LiquidAI/LFM2.5-8B-A1B` `8.47B`, and `Qwen/Qwen3.6-35B-A3B` `34.66B`, plus `LittleBit-2` `Qwen3-8B` `0.55 bpw`) on an Apple M5 Max workstation with external NVMe storage. Checkpoints at `70B+` remain labeled projections in `artifacts/air_mol_nvme_baselines.csv`, not measured `70B` runs. In `Claim LA-11`, greedy `220–300`-token budgets caused `9` of `16` multi-constraint algorithmic specs to hit `VERIFIER_ABSTAIN_SAFE` even after escalating to `Qwen3.6-35B-A3B`; `MoL` guarantees `0.00%` broken commits admitted and strict dominance over every constituent model, not `100%` cold solve rate on arbitrary unbounded software repositories.

**Measurement validity.** Every per-task `J/VC`, `est_j/tok`, and `mJ/byte` figure in this paper is an **analytical estimate (`est_j`)**. On macOS, `src/clai/contracts/joule_meter.py` records both an analytical TDP-time / operation-count estimate (`est_j`) and non-privileged whole-machine `ioreg` `AppleSmartBattery.PowerTelemetryData` `PMU` cumulative energy deltas (`9,237.23 J`, `20,070.18 J`, `126,947.30 J`), which include display/SSD/OS background power and are never reported as isolated silicon-rail joules (`measured_j = None`). On FPGA RTL (`Claim LA-3`), Yosys `0.71+` (`synth_xilinx -family xcup`) proves structural `xck26` netlist synthesis (`41,186` LUTs, `81` `RAMB36E2`s, `0` `DSP48`s) and Verilator `5.054` proves cycle-accurate `CPI = 1` behavior, **not** Xilinx Vivado post-place-and-route `250 MHz` timing closure (`WNS >= 0`) and **not** physical KV260 `VCCINT` / `INA260` power (`board_metered = false`).

**Statistical reporting.** Counts, byte totals, peak active RAM, and mean acceptance rates are reported directly from deterministic CSV/JSON artifacts in `artifacts/` without inventing uncomputed confidence intervals.

**Not claimed.** The following are **not** claimed in this paper: physical KV260 bitstream loading or rail-metered joules for the `clai_benes_hotswap_core.sv` netlist; Vivado post-PAR timing closure at `250 MHz`; silicon energy leadership against commercial ASICs (Groq, Etched, Taalas); promotion of `JEPA-EBRM` conformal proposals to `Deterministic`; or full repository-scale `SWE-bench` evaluation beyond the `48`-task deterministic contract benchmark in `Claim LA-11`.

### 6.1 Executable falsifiers

| ID | Claim | Exact Disproof / Falsifier Condition |
| :--- | :--- | :--- |
| **`LA-F1`** | `Claim LA-1` (`.pcap` collapse & RTL parity) | `python3 scripts/run_realworld_domain.py` fails `100.0%` Verilator RTL parity across `5,120` packets or fails to trip the VLAN `0x8100` drift breaker at packet `2,500`. |
| **`LA-F2`** | `Claim LA-2` (Polyglot zero-alloc & AD parity) | `zig build test` allocates `> 0` heap bytes in the hot cascade loop, `go test -bench=.` reports `> 0 allocs/op`, or `julia sys/julia/mol_scientific.jl` exceeds `1e-7` Enzyme vs. finite-difference error. |
| **`LA-F3`** | `Claim LA-3` (KV260 Beneš + `81` `RAMB36E2` fit) | `python3 scripts/verify_paged_bram_rtl.py` synthesizes `> 117,120` `xcup` CLB LUTs, `> 144` `RAMB36E2`s, `> 0` `DSP48`s, or requires `> 1` Verilator clock cycle per cached context switch. |
| **`LA-F4`** | `Claim LA-4` (Conformal `JEPA-EBRM` `0%` false risk) | `python3 scripts/run_frontier_benchmarks.py` records false crystallization risk significantly above target `α = 0.02` on the shifted evaluation stream or promotes `Conformal-bounded` output to `Deterministic`. |
| **`LA-F5`** | `Claim LA-5` (Lossless byte-level speculation) | `python3 scripts/run_code_corpus_benchmark.py` produces any `< 100.0%` exact byte-for-byte match against greedy `Mamba3CodeLM` across the `48` unseen test prompts. |
| **`LA-F6`** | `Claim LA-6` (`AirMoL` 4-Bit NVMe I/O reduction) | `python3 scripts/run_air_mol_benchmark.py` under Darwin `F_NOCACHE` fails `100.0%` lossless match or reads more NVMe bytes than `AirLLM` 4-bit `T=1`. |
| **`LA-F7`** | `Claim LA-7` (2026 `LFM3-1.7B`, `LFM2.5`, & `Qwen3.5` `AirMoL`) | `run_hf_lfm3_air_mol.py`, `run_hf_air_mol_pretrained.py`, or `run_hf_qwen35_air_mol.py` diverges by even `1` BPE token from exact greedy target decoding (`< 100.0%` parity). |
| **`LA-F8`** | `Claim LA-8` (Full repository regression suite) | `pytest -v` exits non-zero on any of the `26` tests in `tests/`. |
| **`LA-F9`** | `Claim LA-9` (`8.47B` & `34.66B` Sparse-MoE `Expert-AirMoL`) | `run_hf_moe_8b_35b_air_mol.py` diverges from `100.0%` exact greedy token parity or fails to reduce `F_NOCACHE` NVMe bytes read per token below both dense 4-bit `AirLLM` and router-sparse `T=1` `Expert-AirLLM`. |
| **`LA-F10`** | `Claim LA-10` (`Sub1Bit-LUT6` zero-MAC equivalence) | `run_sub1bit_lut_benchmark.py` shows `> 1e-3` max error between `Sub1Bit-LUT6` (`0` binary MACs) and dense Procrustes `ITQ` binary matmul, or fails `100.0%` parity when paired with target verification. |
| **`LA-F11`** | `Claim LA-11` (`VerifiedCodeCascade` soundness & dominance) | `run_frontier_ai_bakeoff.py` admits `> 0%` broken commits, admits a stale Wave-1 skill on a mutated Wave-2 specification, or achieves lower verified pass rate than any constituent standalone model. |
| **`LA-F12`** | `Claim LA-12` (Metal `INT4` MoE & `GatedDeltaNet` shaders) | `run_metal_moe_benchmark.py` exceeds `1e-4` `FP32` kernel error or diverges from `100.0%` greedy token parity on `LFM2.5-8B-A1B` or `Qwen3.6-35B-A3B`. |
| **`LA-F13`** | `Claim LA-13` (`SWE-Mini` multi-file git-repo `pytest` cascade) | `run_repo_patch_benchmark.py` admits `> 0%` broken patches, allows modification of `tests/`, or fails to catch the Wave-2 `R03` specification mutation. |
| **`LA-F14`** | `Claim LA-14` (OpenAI `/v1` + `MCP` `.mol/` agent server) | `pytest tests/test_phase26_metal_ioreport_mcp.py` fails OpenAI REST/SSE streaming, `MCP` JSON-RPC `2.0` tool dispatch, `.mol/` reload, or stamps non-null `measured_j`. |

---

## 7. References

*OpenIE citation contract: Every reference below is a real primary source verified by title, venue/arXiv identifier, or model card. Works marked `[same house]` are authored by David Charlot at Open Interface Engineering (`research.openie.dev`) and are cited for disclosure and continuity, never as independent external confirmation.*

1. Bacellar, A., et al. (2024). Differentiable Weightless Neural Networks. *arXiv:2410.11112*.
2. Beneš, V. E. (1965). *Mathematical Theory of Connecting Networks and Telephone Traffic*. Academic Press.
3. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. *ICML 2024*, *arXiv:2401.10774*.
4. Charlot, D. (2026). Metered Commit Gates: Energy-to-Correct-Completion for Agentic Systems (Part I: The Commit Gate; Part II: Mixture of Limits). *Open Interface Engineering*, `https://research.openie.dev/papers/gates/` `[same house]`.
5. Charlot, D. (2026). RCE-JEPA: World Coding, with JEPA as Premise. *Open Interface Engineering*, `https://research.openie.dev/papers/rce-jepa/` `[same house]`.
6. Charlot, D. (2026). RCE Compile: Verified Compilation Across Boundaries. *Open Interface Engineering*, `https://research.openie.dev/papers/rce-compile/` `[same house]`.
7. Charlot, D. (2026). Task Satiation vs. Aggregate Rebound. *Open Interface Engineering*, `https://research.openie.dev/papers/satiation/` `[same house]`.
8. Chen, L., Zaharia, M., and Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. *arXiv:2305.05176*.
9. Dao, T., and Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *ICML 2024*, *arXiv:2405.21060*.
10. Furao, S., and Hasegawa, O. (2006). An incremental network for on-line unsupervised classification and topology learning (SOINN). *Neural Networks*, 19(1):90–106.
11. Gibbs, I., and Candès, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS 2021*, *arXiv:2106.00170*.
12. GreenServ (2026). Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference. *ICPE 2026*, *arXiv:2601.17551*.
13. HCSpec (2026). Two-Tier Horizontal Cascade Speculative Decoding for High-Efficiency Large Language Model Inference. *ACL 2026*, `https://doi.org/10.18653/v1/2026.acl-long.353`.
14. He, Z., Zhong, Z., Cai, T., Lee, J. D., and He, D. (2024). REST: Retrieval-Based Speculative Decoding. *NAACL 2024*, *arXiv:2311.08252*.
15. Hein, M., Andriushchenko, M., and Bitterwolf, J. (2019). Why ReLU Networks Yield High-Confidence Predictions Far Away From the Training Data and How to Mitigate the Problem. *CVPR 2019*, *arXiv:1812.05720*.
16. Horowitz, M. (2014). Computing's Energy Problem (and What We Can Do About It). *IEEE ISSCC 2014*, `https://doi.org/10.1109/ISSCC.2014.6757323`.
17. Kuon, I., and Rose, J. (2007). Measuring the Gap Between FPGAs and ASICs. *IEEE TCAD*, 26(2):203–215.
18. Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process. *IBM Journal of Research and Development*, 5(3):183–191.
19. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview / Meta AI Research*.
20. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., and Huang, F. J. (2006). A Tutorial on Energy-Based Learning. *Predicting Structured Data*, MIT Press.
21. Google Gemma Team (2026). `google/gemma-4-E2B-it` On-Device Hybrid Instruction Model Card (2026). `https://huggingface.co/google/gemma-4-E2B-it`.
22. Lee, H., Kim, M., Baek, S., Kim, J., Hwang, S., Shin, J., Lee, J., and Kim, E. (2025). LittleBit: Ultra Low-Bit Quantization via Latent Matrix Factorization. *NeurIPS 2025*, *arXiv:2506.13771*.
23. Lee, H., and Kim, E. (2026). LittleBit-2: Pushing the Frontier of Extreme LLMs Quantization into 0.1-Bit (`SamsungLabs/LittleBit`). *ICML 2026*, *arXiv:2603.00042*.
24. Lee, T. T. (1988). Nonblocking Copy Networks for Multicast Packet Switching. *IEEE JSAC*, 6(9):1455–1467.
25. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *JASA*, 113(523):1094–1111.
26. Leviathan, Y., Kalman, M., and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*, *arXiv:2211.17192*.
27. Li, Y., Wei, F., Zhang, C., and Zhang, H. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees. *EMNLP 2024*, *arXiv:2406.16858*.
28. Liquid AI (2026a). `LiquidAI/LFM2.5-350M` and `LiquidAI/LFM2.5-8B-A1B` Open-Weights Hybrid & Sparse-MoE Foundation Model Cards. `https://huggingface.co/LiquidAI/LFM2.5-8B-A1B`.
29. Liquid AI (2026b). `LiquidAI/LFM3-1.7B` Tri-Hybrid Open-Weights Foundation Model Card (October 23, 2026). `https://huggingface.co/LiquidAI/LFM3-1.7B`.
30. Liu, W., Wang, X., Owens, J., and Li, Y. (2020). Energy-based Out-of-distribution Detection. *NeurIPS 2020*, *arXiv:2010.03759*.
31. lyogavin (2023). AirLLM: Layer-wise Large Language Model Inference on Memory-Constrained Hardware. `https://github.com/lyogavin/airllm`.
32. Ma, S., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (`BitNet b1.58`). *arXiv:2402.17764*.
33. Ong, I., et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. *ICLR 2025*, *arXiv:2406.18665*.
34. Petersen, F., Borgelt, C., Kuehne, H., and Deussen, O. (2022). Deep Differentiable Logic Gate Networks. *NeurIPS 2022*, *arXiv:2210.08277*.
35. Qiao, Y., Chen, Z., Zhang, Y., Wang, Y., et al. (2025). TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs. *arXiv:2504.16266*.
36. Qwen Team (2026a). `Qwen/Qwen3.5-0.8B-Base` Hybrid Gated DeltaNet + Attention Open-Weights Model Card (February 2026). `https://huggingface.co/Qwen/Qwen3.5-0.8B-Base`.
37. Qwen Team (2026b). `Qwen/Qwen3.6-35B-A3B` Sparse-MoE Hybrid Gated DeltaNet + Attention Open-Weights Model Card (2026). `https://huggingface.co/Qwen/Qwen3.6-35B-A3B`.
38. Saxena, A. (2023). Prompt Lookup Decoding. `https://github.com/apoorvumang/prompt-lookup-decoding`.
39. Simon, H. A., and Kadane, J. B. (1975). Optimal Problem-Solving Search: All-or-None Solutions. *Artificial Intelligence*, 6(3):235–247, `https://doi.org/10.1016/0004-3702(75)90002-8`.
40. Thompson, C. D. (1978). Generalized Connection Networks for Parallel Processor Intercommunication. *IEEE Transactions on Computers*, C-27(12):1119–1125.
41. Umuroglu, Y., Akhauri, Y., Fraser, N. J., and Blott, M. (2020). LogicNets: Co-Designed Neural Networks and Circuits for Extreme-Throughput Applications. *FPL 2020*, *arXiv:2004.03021*.
42. VoltanaLLM (2025). Energy-Efficient and SLO-Aware Disaggregated LLM Serving via Adaptive Frequency Control and State-Space Routing. *arXiv:2509.04827*.
43. Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
44. Waksman, A. (1968). A Permutation Network. *Journal of the ACM*, 15(1):159–163.
45. Wei, J., et al. (2024). T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge. *arXiv:2407.00088*.

---

## Appendix A. Reproducibility

All artifacts, RTL netlists, and CSV/JSON tables reported in Claims `LA-1` through `LA-11` are reproduced from the repository root (`openIE-dev/lut-adapt-ai`):

```bash
# 1. Full 22-test Python + RTL + Conformal + AirMoL + Expert-AirMoL + Bakeoff regression suite
PYTHONPATH=src pytest -v

# 2. Synthetic & real Wireshark .pcap collapse curves + KV260 Verilator parity (Claim LA-1)
python3 scripts/run_collapse_curve.py
python3 scripts/run_realworld_domain.py

# 3. Polyglot verification: Rust Burn/CubeCL Metal, Zig 0.17.1, Go 1.27.4, Julia 1.13.4 (Claim LA-2)
(cd sys/rust && cargo run --release)
(cd sys/zig && zig build test)
(cd sys/go && go test -v -bench=.)
julia sys/julia/mol_scientific.jl

# 4. Frontier baselines & KV260 512x512 Benes + 81-RAMB36E2 Yosys/Verilator synthesis (Claims LA-3, LA-4)
python3 scripts/run_frontier_benchmarks.py
python3 scripts/verify_paged_bram_rtl.py

# 5. 520 KB real source-code speculative decoding & controlled 14.3M AirMoL NVMe benchmarks (Claims LA-5, LA-6)
python3 scripts/run_code_corpus_benchmark.py
python3 scripts/run_air_mol_benchmark.py

# 6. Pretrained 2026 Dense & Tri-Hybrid Models through AirMoL 4-Bit NVMe (Claim LA-7)
python3 scripts/run_hf_lfm3_air_mol.py
python3 scripts/run_hf_air_mol_pretrained.py
python3 scripts/run_hf_qwen35_air_mol.py

# 7. Frontier 8.47B & 34.66B Sparse-MoE Expert-AirMoL NVMe Streaming (Claim LA-9)
python3 scripts/run_hf_moe_8b_35b_air_mol.py

# 8. Sub-1-Bit Latent Factorization (LittleBit-1 / LittleBit-2 vs. Sub1Bit-LUT6 & MoL; Claim LA-10)
python3 scripts/run_sub1bit_lut_benchmark.py

# 9. 48-Task (16 Specs x 3 Waves) Head-to-Head Late-2026 Frontier Coding Bakeoff (Claim LA-11)
python3 scripts/run_frontier_ai_bakeoff.py

# 10. Native Apple Metal INT4 MoE & GatedDeltaNet + reported_j, SWE-Mini Repo Patch, & OpenAI/MCP Agent (Claims LA-12..14)
python3 scripts/run_metal_moe_benchmark.py
python3 scripts/run_repo_patch_benchmark.py
python3 scripts/mol_agent_server.py --host 127.0.0.1 --port 8090 --mol-dir .mol

# 11. Interactive CLI / HTTP-SSE streaming server with NVMe + KV260 BRAM telemetry (Claim LA-8)
python3 scripts/mol_code_cli.py --backend lfm3-1.7b --prompt "def forward(self, x):" --max-bytes 64
python3 scripts/mol_code_cli.py --backend qwen3.6-35b-a3b --prompt "def forward(self, x):" --max-bytes 48
```
