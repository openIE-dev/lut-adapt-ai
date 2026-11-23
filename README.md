# LUT-Adapt-AI (`central-limit-ai` + `central-limit-sys`)

**Author:** David Charlot, Open Interface Engineering (`research.openie.dev`) · **License:** Open Source (`MIT / Apache-2.0`)

> **Measurement.** Research repository, not a journal final. No physical FPGA board or isolated silicon rail shunt was metered in this repository (`measured_j = None`, `board_metered = false`). Hardware utilization (`41,186` `xcup` CLB LUTs in `1`-cycle OOC; **`41,465` `xcup` CLB LUTs = `35.40%`**, **`13` logic levels [`4.00×` shallower]**, **`170 <= 189` `IOB`s**, `81` `RAMB36E2` blocks, `0` `DSP48` slices in the `CE`-gated `4`-stage `250 MHz`-pipelined `P4-AXIS` top), `Verilator` `VCD` + `AMD DS925` dynamic power (**`15.59 est_mW` idle [`4.10×` lower]** and **`25.84 est_mW` `AXI4-Stream` packet-shift [`2.69×` lower]**), and zero-bubble (`CPI = 1.00` at `250 MHz` nominal, `256/256` consecutive beats) switching are **simulated** via Yosys `0.71+` (`synth_xilinx -family xcup` targeting Kria KV260 `xck26-sfvc784-2LV-c`) and Verilator `5.054`. All `27` formal theorems in Section 2 (`proofs/`) are **machine-checked** in **Lean `4.36.0`** (`0` `sorry`, `0` custom axioms). Workstation wall-clock times, exact Darwin `F_NOCACHE` `os.pread` NVMe byte counts, peak active RAM (`250.0 MB` = `3.79%` FP32 on Oct 2026 `LiquidAI/LFM3-1.7B`, `500.0 MB` on Metal-fused `LiquidAI/LFM2.5-8B-A1B`, `970.0 MB` = `1.47%` FP16 on Metal-fused `Qwen/Qwen3.6-35B-A3B`, and **`58,075.4 MB` resident Unified Memory at `22.17 tok/s` on `120B`-scale `nvidia/Nemotron-3-Super-120B-A12B`**), `100.0%` bit/token parity, dual-channel Apple Silicon `reported_j` hardware counters (`L3 proc_pid_rusage(RUSAGE_INFO_V6).ri_energy_nj` CPU `nJ` + `L2 libIOReport` `"Energy Model"` `'GPU Energy'` `nJ`), and whole-machine macOS `PMU` energy deltas are **measured / reported** in software on Apple M5 Max (`measured_j = None`). Analytical joule figures (`est_j`) stay strictly separated from `reported_j` and `measured_j`.

A full-stack, RTL-simulated and software-measured reference implementation of the **4-Rung Mixture of Limits (MoL)** cascade and **Downward LUT6 Compiler**, spanning:

1. **Python / PyTorch 2.15 + Triton (`src/clai/`)**: Complete 64-lane streaming micro-batch runtime with dynamic escalation compaction (`64 -> B_r2 -> B_r2.5 -> B_r3`), custom telescoping Mamba-3 SSD adjoints, open-world continual plasticity, and Xilinx UltraScale+ `LUT6` Verilog synthesis.
2. **Rust 1.100 + Burn 0.22.1 / CubeCL (`sys/rust/`)**: Custom cross-platform `#[cube]` JIT GPU/CPU kernels — dual-word 64-lane bit-plane transpose/Shannon/voter, 16x16 SRAM-tiled Mamba-3 SSD forward & `Autodiff` `Backward` adjoints, plus portable fused `INT4` Block-64 Sparse-MoE (`q4_moe_gate_up_silu_down`: **`35.09×` Metal GPU**, **`2.67×` LLVM CPU**, `1.79e-7` err) and `GatedDeltaNet` (`gated_delta_recurrent_k128`: **`69.20×` Metal GPU**, **`14.87×` LLVM CPU**, `2.24e-8` err) — targeting Apple Metal, Vulkan (`SPIR-V`), NVIDIA CUDA (`PTX`), AMD ROCm (`HIP`), and LLVM SIMD CPU.
3. **Zig 0.17.1 (`sys/zig/`)**: Bare-metal `@Vector(8, u64)` comptime-unrolled 64-minterm SIMD LUT6 engine (512 episodes per vector op, ~135 ns/episode across 2 layers), stack-allocated EBRM relaxation, and KV260 `LUT6` Verilog primitive emitter.
4. **Go 1.27.4 `GOEXPERIMENT=simd,jsonv2` (`sys/go/`)**: Portable hardware SIMD (`simd.Float32s` ARM64 NEON / AMD64 AVX-512) + concurrent goroutine 64-lane LUT6 & HC-SOINN router with `encoding/json/v2` telemetry.
5. **Julia 1.13.4 (`sys/julia/`)**: `KernelAbstractions.jl` `@kernel` 64-lane bit-sliced LUT6 + `@simd ivdep` Mamba-3 complex RoPE stream + zero-allocation LLVM-IR `Enzyme.autodiff` JEPA-EBRM settling (`JuliaC.jl` compatible).
6. **Kria KV260 AXI4-Lite + AXI4-Stream Hardware Subsystem (`hw/kv260/`, `scripts/verify_kv260_subsystem.py`)**: Full AMBA AXI4-Lite CSR control plane + AXI4-Stream division-free byte-stream to 512-wire `EngramBitBus` streamer (`clai_engram_streamer.sv`) + dual-bank Engram BRAM + `1,536`-`LUT6` core + popcount voter + `irq_escalate_to_ps` interrupt (`clai_kv260_axi_top.sv`), Vivado 250 MHz `.xdc` & BD TCL script, synthesized in **Yosys 0.71+** (`0 DSP48`s) and co-simulated in **Verilator 5.054**.
7. **Real-World L7 Packet, AST & SCADA Telemetry Domain (`src/clai/domains/network_ast_telemetry.py`, `scripts/run_realworld_domain.py`)**: Raw variable-length L3/L4/L7 wire frames (HTTP/JSON AST, gRPC Protobuf, DNS, Modbus TCP, SQLi, shell traversal, SYN-Xmas scans, live 802.1Q VLAN / VXLAN protocol drift, and unseen Zero-Day SCADA `FC=0x10` PLC overrides) compiled downward into a hardware-verified `LUT6` line-rate firewall (`artifacts/clai_realworld_firewall_kv260.v`).
8. **KV260 Multi-Page BRAM `LUT6` Cache + `4`-Stage `250 MHz`-Pipelined Beneš Core (`hw/kv260/rtl/clai_benes_hotswap_core.sv`, `clai_benes_pipelined_250mhz.sv`, `scripts/verify_pipelined_250mhz_rtl.py`)**: `512`-page (`81` `RAMB36E2` Block RAMs, or `4,096`-page `URAM288`) Galois-XOR hash-indexed `80×64b` `LUT6 INIT` page cache in both a `1`-cycle OOC core (`41,186` LUTs, `52` logic levels) and a balanced `4`-stage pipelined IOB-safe `AXI4-Stream` wrapper (`clai_benes_axi_stream_250mhz_top`: **`13` logic levels [`4.00×` shallower]**, **`170 <= 189` `xck26` `IOB`s**, **`41,459` LUTs = `35.40%`**, `0` `DSP48`s, **`256 / 256` zero-bubble `CPI = 1.00` Verilator parity**).
9. **Interactive Speculative Code-Completion CLI & Streaming HTTP/SSE Server (`src/clai/service/live_code_engine.py`, `scripts/mol_code_cli.py`)**: Exposes selectable `--backend {mamba3,lfm2.5-350m,qwen3.5-0.8b,lfm3-1.7b,lfm2.5-8b-a1b,qwen3.6-35b-a3b}` across `Mamba3CodeLM` (`--air-nvme`) and late-2026 pretrained NVMe-streamed Hybrid & Sparse-MoE checkpoints (`LiquidAI/LFM3-1.7B`, `LiquidAI/LFM2.5-8B-A1B`, `Qwen/Qwen3.6-35B-A3B`) with `PagedBenesLUT6` + `HardwareBRAMPageCache`, ANSI-colored terminal streaming (`complete`, `repl`), multi-threaded HTTP + SSE endpoints (`serve`), and automatic `FileEditSleepWatcher` online Sleep-Crystallization.
10. **`AirMoL` & `Expert-AirMoL` — Dense, Tri-Hybrid, and `35B` Sparse-MoE NVMe Streaming (`src/clai/frontier/air_mol_streamer.py`, `hf_air_mol_pretrained.py`, `moe_air_mol.py`)**: Combines per-layer `INT4` (`block_size=64`) uncached `F_NOCACHE` NVMe `os.pread` streaming + router-sparse `.experts.pack4` byte-stride expert `pread` + Zero-Extra-Sweep Hybrid Cache Rollback + universal `<= 64`-bit Token-Level `BenesLUT6` burst speculation across October 2026 `LiquidAI/LFM3-1.7B` (`2.74×` NVMe cut, `3.79%` RAM, `2.49×` speedup), `LiquidAI/LFM2.5-8B-A1B` (`8.47B`; `6.90×` NVMe cut, `3.93×` speedup), and `Qwen/Qwen3.6-35B-A3B` (`34.66B` `256`-expert MoE; **`12.30×` NVMe I/O reduction**, **`2.53 GB` RAM = `3.83%` of FP16**, **`6.50×` speedup**) at **`100.0%` lossless token parity**.
11. **`Sub1Bit-LUT6` (`0.576 bpw`) vs. Samsung `LittleBit-1 / LittleBit-2`, `VerifiedCodeCascade`, & `Rung 3d` Native `<think>` Ceiling (`src/clai/frontier/sub1bit_lut_factor.py`, `verified_code_cascade.py`, `scripts/run_rung3d_thinking_ceiling.py`)**: Replaces `LittleBit-2`'s `1.98M` runtime binary-matrix MACs with zero-MAC `6`-input `LUT6` partial-sum lookup (`< 4e-6` match, `100.0%` parity via `MoL` target verification in `0` QAT epochs), and beats standalone `LittleBit-2 0.55bpw`, `LiquidAI/LFM3-1.7B`, `google/gemma-4-E2B-it`, `LiquidAI/LFM2.5-8B-A1B`, `Qwen/Qwen3.6-35B-A3B`, and `RouteLLM` on a `48`-task (`16` specs × `3` waves) engineering bakeoff (**`87.50%` fast `Arm G`**, rising to **`91.67%` overall / `93.75%` W2–W3** with bounded open-weights `Rung 3d` `Qwen3.6-35B-A3B` `<think>` escalation vs. `0.00%–77.08%` standalone, **`0.00%` broken commits admitted**, only **`20.83%` `35B` calls**, and `< 3.8 ms` / `0.0351 J` `reported_j` Wave-3 crystallized replay).
12. **Native Apple Metal (`torch.mps.compile_shader`) `INT4` MoE + `GatedDeltaNet` Shaders, Multi-File Git-Repo `pytest` Patching (`SWE-Mini`), & Drop-In OpenAI/MCP Agent (`src/clai/frontier/metal_moe_kernels.py`, `repo_patch_cascade.py`, `src/clai/service/openai_mcp_agent.py`)**:
    - Custom Metal Shading Language (`MSL`) `gated_delta_recurrent_k128` (**`20.66×` faster**, **`46.2×` lower `reported_j`**) and `q4_moe_gate_up_silu` + `q4_moe_down_weighted` (**`43.45×` faster**, **`29.8×` lower `reported_j`**, `7.2e-7` `FP32` error), cutting `Qwen/Qwen3.6-35B-A3B` cold-NVMe Peak Active RAM to **`970.0 MB` (`1.47%` of FP16)** (`3.25×` faster `Expert-AirMoL` at `6.77 J/tok` `reported_j`, and **`15.7 tok/s`** in `Q4`-Unified workstation mode).
    - `16`-task (`8` multi-file git repos × `2` waves) `SWE-Mini` `pytest` patching benchmark achieving **`87.50%` verified pass** (vs. `0.0%–62.5%` standalone), **`0.00%` broken patches committed**, and **`65.6×` faster / `537×` lower `reported_j` Wave-2 `.mol/` patch replay** (`594 ms`, `1.10 J`).
    - Drop-in `mol-agent` server (`GET /v1/models`, `POST /v1/chat/completions` REST+SSE, and `MCP` JSON-RPC `2.0` over `stdio` & `POST /mcp`) backed by persistent `.mol/` `BenesLUT6` + AST skill stores.
13. **Official Public External Benchmarks via Live `MCP` `JSON-RPC 2.0` (`src/clai/frontier/external_public_benchmarks.py`, `scripts/run_external_public_benchmarks.py`)**:
    - **`bigcode/bigcodebench` `v0.1.3`** (`SHA-256 = ed5a3a39...`, `16` tasks × `2` human-authored prompt waves = `32` `unittest` evaluations, zero gold leakage): **`OpenIE MoL MCP Agent` reaches `87.50%` W1, `93.75%` W2 (`instruct_prompt` human paraphrase), and `90.62%` (`29/32`) overall** vs. `43.75%` (`LFM3-1.7B`), `37.50%` (`Gemma-4-E2B-it`), and `78.12%` (`Qwen3.6-35B-A3B`), with **`0.00%` broken outputs admitted**, **`81.25%` (`13/16`) Wave-2 `.mol/` fast-path hits** (`6,001 ms` vs. `47,321 ms`, `7.89×` faster), and lower `reported_j` per solve (`263.54 J` vs. `277.45 J` on `35B`).
    - **Upstream `princeton-nlp/SWE-bench_Verified`** (`SHA-256 = a45b1fe4...`, `8` real `sympy/sympy` git-worktree checkouts × `2` waves = `16` `pytest` evaluations, zero gold-patch leakage): **`OpenIE MoL MCP Agent` reaches `50.00%` (`8/16`)** vs. `0.00%` (`LFM3-1.7B`, `Gemma-4-E2B-it`) and `25.00%` (`Qwen3.6-35B-A3B`) (**`2.00×` pass rate**), **`0.00%` broken patches admitted**, and **`100%` (`4/4` solved) Wave-2 `.mol/` `git diff` fast-path replay** (`1,590–1,720 ms` at `0.33–0.44 J` `reported_j`).
    - **Multi-Repository Upstream `SWE-bench_Verified` + Multi-Window AST Call-Graph Localizer (`src/clai/frontier/multirepo_swebench.py`, `scripts/run_multirepo_swebench_benchmark.py`)**: Expands zero-leakage git-worktree evaluation across **`4` distinct upstream Python repositories (`django/django`, `pytest-dev/pytest`, `psf/requests`, `sympy/sympy`; `24` evaluations)**, localizing `11/12` (`91.67%`) ground-truth files without reading `patch`, reaching **`20.83%` overall / `25.00%` W2 (`2.50×` standalone `Devstral-2-24B`, `Qwen3.6-35B-A3B`, and `Gemma-4-E2B-it`'s `8.33%`)** with **`0.00%` broken patches admitted** and **`0.59–2.03 s` (`1.12–5.50 J` `reported_j`) Wave-2 `.mol/` unified-diff replay**.
14. **Lean `4.36.0` Formal Proofs, Measured `120B` Hybrid Sparse-MoE Apex, `CE`-Gated `VCD` Power, & Single-Binary Rust `mol-agent-rs` (`proofs/`, `frontier_120b_hybrid_moe.py`, `verify_ce_gated_vcd_power.py`, `sys/rust/src/bin/mol_agent.rs`)**:
    - **Lean `4.36.0` Machine-Checked Formalization (`Claim LA-19`)**: **`27` kernel-verified theorems** across `LutAdapt/{Cascade,BenesCopy,JointMinterm,ConformalProvenance}.lean` (`0` `sorry`, `0` custom axioms, `lake build` in `2.26 s`).
    - **Measured `120B` Hybrid Mamba-Attention Sparse-MoE (`nvidia/Nemotron-3-Super-120B-A12B`) + `Devstral-Small-2-24B` (`Claim LA-21`)**: Real Apple M5 Max (`128 GiB` Unified Memory) Metal `llama-server` benchmark (`58.1 GB` RSS, **`22.17 tok/s`** on `120B-A12B`), lifting the `8`-spec (`16`-eval) hard frontier suite to **`62.50%` verified pass (`0.00%` broken commits)** vs. `43.75%` standalone `120B-A12B` and `31.25%` `24B` (solving `BigCodeBench/8` and replaying Wave-2 `.mol/` skills in **`5.5–69.8 ms`** / **`0.012–0.027 J`**).
    - **`CE`-Gated `250 MHz` `P4-AXIS` Verilator `VCD` Power Audit (`Claim LA-22`)**: `ENARDEN` + valid-bubble `FDRE` clock gating cuts idle power **`4.10×`** (`63.99` $\to$ **`15.59 est_mW`**) and `AXI4-Stream` packet-shift power **`2.69×`** (`69.62` $\to$ **`25.84 est_mW`**) at `100.0%` (`320/320`) Verilator parity.
    - **Zero-Python Single-Binary Rust `mol-agent-rs` (`Claim LA-23`)**: Native `std::net` + `Burn 0.22.1 / CubeCL` server (`GET /v1/models`, `POST /v1/chat/completions` REST+SSE, `POST /mcp`, `--mcp-stdio`) serving `.mol/` `MCP` lookups in **`0.1540 ms/req` (`17.21×` faster than Python `mol-agent`)** and live `CubeCL_Metal_GPU` verification in **`0.9887 ms/req`**.
15. **OpenIE Research Paper, Vector Diagrams & Standalone Web Edition (`paper/`)**: Authored by **David Charlot, Open Interface Engineering** in full `research.openie.dev` house style:
    - **Single-Column LaTeX Edition**: `paper/mol_architecture.tex` $\to$ `paper/mol_architecture.pdf` (`STIX Two Text` + `STIX Two Math` + `Menlo`, yellow `Measurement.` honesty banner, `In plain words`, `Notation` table, `4` formal results + `Lean 4.36.0` proofs, `Claims LA-1..LA-23`, `Related work` convergence table, `Falsifiers LA-F1..LA-F23`, and `[same house]` disclosures).
    - **Canonical Markdown & Diagrams**: `paper/lut-adapt.md` and `paper/figures/{mol-downward-cascade,benes-paged-silicon,airmol-nvme-speculation,moe-expert-cascade}.{svg,png}`.
    - **Standalone `research.openie.dev` Web Suite**: `paper/web/index.html` (Dark/Light/Auto theme, sticky TOC, zoomable figure lightbox, JSON-LD `ScholarlyArticle`), `paper/web/about.html` (`/about/` summary), and `paper/web/living.html` (`/living/` interactive calculators & telemetry explorers).

---

## Architecture: The 4-Rung Thermodynamic Ladder

| Rung | Subsystem | Compute Primitive | Energy / Latency Scale | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Rung 0** | `EngramBitBus` + `DiffLUTNetwork` | 512-wire `uint64` bit-planes $\to$ 64-lane `LUT6` | Nanoseconds / Microjoules (0 MACs) | Bit-exact crystallized Boolean reflex & KV260 FPGA target |
| **Rung 2** | `Rung2Mamba3Streamer` | Complex RoPE + Trapezoidal Rank-$2R$ Chunkwise SSD | Microseconds / Millijoules | $O(1)$-state variable-length byte streaming & parity tracking |
| **Rung 2.5** | `HypersphericalSOINN` | $\mathbb{S}^{D-1}$ Topology-Aware Prototype Graph | Microseconds / Zero-Gradient | Instantaneous post-mutation fission & open-world novel class attachment |
| **Rung 3** | `LatentJEPA_EBRM` | 25M NCM 4-Level RVQ + JEPA + EBRM Settling | Milliseconds / Joules | Iterative $-\nabla_z \mathcal{F}$ energy minimization on novel constraints |
| **Sleep** | `SleepCrystallizer` | 3rd-Order Walsh-Hadamard Spectral Mux + ST-DiffLUT | Amortized offline pass | Distills verified upper-rung traces downward into Rung 0 `LUT6` gates |

---

## Key Algorithmic Fixes Over Raw Prototype

1. **Clean 512-Wire Partitioning (`engram_store.py`)**: Eliminates random hash-bucket XOR scrambling over raw input wires by partitioning `[B, 8]` `uint64` words into:
   - `Words 0..1` (`0..127`): Unscrambled 128-bit streaming 16-byte XOR fold (exact parity sensitivity).
   - `Words 2..5` (`128..383`): 16-slot $\times$ 16-level chunk-normalized monotonic Thermometer bit-planes (zero Hamming cliffs, zero MACs).
   - `Words 6..7` (`384..511`): Associative Host-DRAM Engram N-gram heads with active `write_association_u64` sleep writeback.
2. **3rd-Order Cumulant Parity Discovery + Inter-Layer Straight-Through Binarization (`sleep_crystallizer.py`, `difflut_net.py`)**: Unmixes parity-reversed linear clauses via class-conditioned 3rd-order Walsh-Hadamard cumulants and applies Straight-Through binary wire quantization across LUT layers to eliminate compound discretization drift when snapping to `{0, 1}` `LUT6` INIT masks.
3. **Non-Collapsing Concept Valley Anchors (`jepa_ebrm.py`)**: Replaces trivial `z_pos = 0` training targets with separated orthogonal class-concept anchors $z^*(s_x, y) \neq 0$ and warm JEPA proposals so `settle_batch` genuinely descends $-\nabla_z \mathcal{F}$.
4. **Genuine Burn 0.22.1 `Autodiff` `Backward` Registration (`sys/rust/src/ext.rs`)**: Wires `Mamba3SsdBackward` (`Backward<B, 5>`) into `mamba3_ssd_backward_cube_kernel` instead of detaching via `Autodiff::from_inner`.

---

## Quickstart Commands

```bash
# 1. Python Invariant Suites (35/35) + Lean 4.36.0 Proofs, Multi-Repo SWE-bench, 120B Hybrid MoE, & CE-Gated RTL
PYTHONPATH=src .venv/bin/pytest -v
(cd proofs && lake build) && PYTHONPATH=src .venv/bin/python scripts/verify_lean_proofs.py
PYTHONPATH=src .venv/bin/python scripts/run_multirepo_swebench_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_120b_hybrid_moe_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/verify_ce_gated_vcd_power.py
PYTHONPATH=src .venv/bin/python scripts/run_external_public_benchmarks.py
PYTHONPATH=src .venv/bin/python scripts/verify_pipelined_250mhz_rtl.py
PYTHONPATH=src .venv/bin/python scripts/run_rung3d_thinking_ceiling.py
PYTHONPATH=src .venv/bin/python scripts/run_metal_moe_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_repo_patch_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_frontier_ai_bakeoff.py
PYTHONPATH=src .venv/bin/python scripts/mol_agent_server.py --host 127.0.0.1 --port 8090 --mol-dir .mol
PYTHONPATH=src .venv/bin/python scripts/run_hf_lfm3_air_mol.py
PYTHONPATH=src .venv/bin/python scripts/run_hf_moe_8b_35b_air_mol.py
PYTHONPATH=src .venv/bin/python scripts/run_sub1bit_lut_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_hf_air_mol_pretrained.py
PYTHONPATH=src .venv/bin/python scripts/run_hf_qwen35_air_mol.py
PYTHONPATH=src .venv/bin/python scripts/run_air_mol_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/verify_paged_bram_rtl.py
PYTHONPATH=src .venv/bin/python scripts/mol_code_cli.py complete --backend lfm3-1.7b --lang py --max-bytes 60
PYTHONPATH=src .venv/bin/python scripts/mol_code_cli.py complete --backend qwen3.6-35b-a3b --lang py --max-bytes 60
PYTHONPATH=src .venv/bin/python scripts/mol_code_cli.py serve --backend lfm2.5-350m --host 127.0.0.1 --port 8765 --watch src
/opt/homebrew/bin/tectonic paper/mol_architecture.tex
PYTHONPATH=src .venv/bin/python scripts/run_code_corpus_benchmark.py
PYTHONPATH=src .venv/bin/python scripts/run_frontier_benchmarks.py
PYTHONPATH=src .venv/bin/python scripts/run_collapse_curve.py
PYTHONPATH=src .venv/bin/python scripts/run_realworld_domain.py --emit-pcap artifacts/enterprise_drift_capture.pcap
tcpdump -nn -e -c 8 -r artifacts/enterprise_drift_capture.pcap
PYTHONPATH=src .venv/bin/python scripts/verify_hardware_rtl.py
PYTHONPATH=src .venv/bin/python scripts/verify_kv260_subsystem.py
clang -x c-header -std=c11 -Wall -Wextra -Werror -fsyntax-only hw/kv260/sw/clai_kv260_uio.h

# 2. Rust 1.100 + Burn 0.22.1 / CubeCL (Tests + Zero-Python `mol-agent-rs` + Cross-Platform Q4 MoE & DeltaNet)
cd sys/rust && cargo test --release && cargo run --release --bin mol_agent -- --self-test-bench && cargo run --release --bin bench_cubecl_moe_deltanet && cargo run --release && cargo run --release --bin collapse_curve

# 3. Zig 0.17.1 (@Vector(8, u64) 512-Episode SIMD + KV260 Verilog Emitter)
cd sys/zig && zig build test && zig build run --release=fast

# 4. Go 1.27.4 (Hardware ARM64 NEON / AVX-512 SIMD + JSON v2 Telemetry)
cd sys/go && GOEXPERIMENT=simd,jsonv2 go run ./cmd/clai

# 5. Julia 1.13.4 (KernelAbstractions.jl + Enzyme.jl Zero-Alloc LLVM AD)
cd sys/julia && julia --project=. -t auto -O3 -e 'using CentralLimitAI; run_collapse_microbatch()'
```
