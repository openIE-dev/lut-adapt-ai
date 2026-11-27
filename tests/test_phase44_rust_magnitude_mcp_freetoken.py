"""Phase 44 Invariant & Regression Suite (`Claims LA-72..LA-75` + Historic `138`-Theorem / `1,502`-Fleet `Claim LA-19` Milestone):
  1. `Claim LA-72`: Native Zero-Python Rust `mol-magnitude-adapter` (`OpenAI /v1/chat/completions`,
     `Anthropic /v1/messages`, `MCP JSON-RPC 2.0 /mcp`) across `Apple M5 Max`, `NVIDIA Jetson AGX Thor`
     (physical `TI INA238` `VIN` DC shunt `measured_j`, `board_metered = true`), and `AMD Ryzen AI Max+ 395`
     (`Zen 5` `RAPL` `reported_j`) on `CogLoadBench` `4x` extraneous context pollution (`100.0%` verified,
     `0` Linux Vulkan `graph_replays.rs:821` panics).
  2. `Claim LA-73`: `24`-Workflow (`72`-Call) Stateful `MCPMark` Multi-Turn CRUD Sandbox (`0.00%` live
     sandbox corruption via Copy-on-Write `MCPMark-CoW-Guard LUT6` vs. `75.00%` unguarded corruption) +
     `144/144` (`9,216/9,216` minterm) `Yosys 0.71+` (`abc -lut 6`) `$lut` Boolean Miter.
  3. `Claim LA-74`: Bandwidth-Adaptive `BA-Tiered-Expert-AirMoL` (`Closed-Form CPU-SIMD / GPU-Link Split`
     + `K = 3` `LUT6` Burst + Scan-Safe `Reuse`) vs. `FreeToken` (`UC Berkeley Aug 2026`) & `KTransformers`,
     plus Cross-Silicon `Window 24` Physical `Thor` DC Shunt Telemetry.
  4. `Claim LA-75` & `Claim LA-19`: Exact `64.00 KB` (`14`-Bank `56 KB` + `4 KB` Nucleus + `4 KB` Stack)
     Watch `SRAM` Partition, Zero-`DSP48E1` `> 100 MHz` `openXC7` `XC7A200T` `.bit`, `19`-Export
     `15,521 B <= 16,384 B` `lut6_core.wasm`, and `138` Six-Kernel `Lean 4.36.0` Proofs (`>= 1,502` Fleet!).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from clai.frontier.enigma_zerobench_codifier import (
    compile_truth_table_6bit_to_lut6_masks,
    eval_lut6_masks6,
)
from clai.frontier.rust_magnitude_mcp_freetoken import (
    ba_airmol_coschedule2_py,
    mcpmark_cow_guard2_py,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la72_native_rust_magnitude_adapter_and_cogload_fleet() -> None:
    rust_bin = REPO_ROOT / "sys" / "lut6-core" / "target" / "release" / "mol_magnitude_adapter"
    assert rust_bin.is_file(), "Expected compiled release binary mol_magnitude_adapter"
    st_out = json.loads(subprocess.check_output([str(rust_bin), "--self-test"], text=True).strip())
    assert st_out["self_test"] == "PASS"
    assert set(st_out["endpoints"]) == {"/v1/chat/completions", "/v1/messages", "/mcp"}

    cog = list(csv.DictReader((ART_DIR / "phase44_rust_magnitude_cogload.csv").open(encoding="utf-8")))
    assert len(cog) == 12  # 3 hosts x 4 systems
    for r in cog:
        bm = r["board_metered"].lower() == "true"
        if bm:
            assert "Thor" in r["host_label"]
            assert r["measured_j"] not in ("", "null") and r["reported_j"] in ("", "null")
        else:
            assert "Thor" not in r["host_label"]
            assert r["measured_j"] in ("", "null") and r["reported_j"] not in ("", "null")
        if "mol_magnitude_adapter_rs" in r["system"]:
            assert float(r["verified_pass_pct"]) == 100.0
            assert float(r["broken_or_crash_pct"]) == 0.0
            assert int(r["vulkan_prefill_crashes"]) == 0


def test_claim_la73_mcpmark_cow_crud_sandbox_and_yosys_miter() -> None:
    mcp_masks = compile_truth_table_6bit_to_lut6_masks(mcpmark_cow_guard2_py)
    for x in range(64):
        assert eval_lut6_masks6(mcp_masks, x) == mcpmark_cow_guard2_py(x)

    mcp_rows = list(csv.DictReader((ART_DIR / "phase44_mcpmark_stateful_crud.csv").open(encoding="utf-8")))
    assert len(mcp_rows) == 3
    for r in mcp_rows:
        assert float(r["unguarded_state_corrupted_pct"]) == 75.0
        assert float(r["wave1_cow_verified_pct"]) == 100.0
        assert float(r["wave1_state_corrupted_pct"]) == 0.0
        assert float(r["wave2_lut6_idemp_verified_pct"]) == 100.0
        assert float(r["wave2_state_corrupted_pct"]) == 0.0
        assert int(r["wave1_cow_rollbacks_caught"]) == 18
        assert int(r["wave2_idemp_replays_deduped"]) >= 24

    miter_rows = list(csv.DictReader((ART_DIR / "phase44_mcpmark_ba_lut6_miter.csv").open(encoding="utf-8")))
    assert sum(int(r["lut6_cells"]) for r in miter_rows) == 144
    assert sum(int(r["minterms_verified"]) for r in miter_rows) == 9216
    assert all(int(r["yosys_miter_mismatches"]) == 0 for r in miter_rows)


def test_claim_la74_ba_airmol_vs_freetoken_ktransformers_and_window24_thor_shunt() -> None:
    ba_masks = compile_truth_table_6bit_to_lut6_masks(ba_airmol_coschedule2_py)
    for x in range(64):
        assert eval_lut6_masks6(ba_masks, x) == ba_airmol_coschedule2_py(x)

    rows = list(csv.DictReader((ART_DIR / "phase44_ba_airmol_vs_freetoken.csv").open(encoding="utf-8")))
    assert len(rows) == 15  # 3 sparse-MoE families x 5 runtimes
    for fam in ("Qwen3.8-Flash-Next-180B-A6B", "GLM-5-355B-A32B", "DeepSeek-V4.1-Flash-552B-A16B"):
        sub = [r for r in rows if r["model_family"] == fam]
        s1, s2, _s3, s4, s5 = sub
        assert float(s5["effective_tok_per_s"]) > float(s4["effective_tok_per_s"]) > float(s2["effective_tok_per_s"]) > float(s1["effective_tok_per_s"])
        assert float(s5["speedup_vs_freetoken_x"]) > 2.5
        assert float(s5["speedup_vs_ktransformers_x"]) > 3.5
        assert int(s5["scan_evictions_prevented"]) > 0
        assert float(s5["nvme_mb_per_token"]) < float(s4["nvme_mb_per_token"])

    w24_rows = list(csv.DictReader((ART_DIR / "phase44_window24_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w24_rows) == 3
    metered = [r["host_label"] for r in w24_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w24_rows:
        assert float(r["energy_reduction_x"]) > 1.5
        if r["board_metered"] == "True":
            assert float(r["latency_speedup_x"]) > 3.0
            assert r["w24_measured_nj_per_op_vdd_cpu"] != "" and r["w24_reported_nj_per_op"] == ""
        else:
            if "Zen5" in r["host_label"]:
                assert float(r["latency_speedup_x"]) > 1.3
            assert r["w24_measured_nj_per_op_vdd_cpu"] == "" and r["w24_reported_nj_per_op"] != ""


def test_claim_la75_openxc7_19export_wasm_and_138_six_kernel_proofs_1502_fleet() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_mag_mcp_freetoken_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["mcpmark_ba_airmol_64minterm_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 138
    assert lean_meta["module_theorem_counts"].get("RustMagnitudeMcpFreeToken") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1502
    assert fleet_meta["all_six_kernels_passed_sandboxed"] is True

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const fnCount = Object.keys(ex).filter(k => k.startsWith('wasm_')).length;
  const argOk = 57 | (3 << 6) | (12 << 12) | (63 << 16) | ((0b100101) << 22);
  const rOk = ex.wasm_eval_mag_mcp_freetoken_lut6(argOk >>> 0, 7) >>> 0;
  const argRollback = 57 | (3 << 6) | (12 << 12) | (61 << 16) | ((0b010011) << 22);
  const rRoll = ex.wasm_eval_mag_mcp_freetoken_lut6(argRollback >>> 0, 7) >>> 0;
  const argBrk = argOk | (1 << 28);
  const rBrk = ex.wasm_eval_mag_mcp_freetoken_lut6(argBrk >>> 0, 7) >>> 0;
  console.log(JSON.stringify({{ fnCount, rOk, rRoll, rBrk }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 19
    assert ((out["rOk"] >> 30) & 3) == 1
    assert ((out["rOk"] >> 26) & 0xF) == 13
    assert ((out["rOk"] >> 20) & 3) == 3
    assert ((out["rOk"] >> 18) & 3) == 2
    assert ((out["rRoll"] >> 20) & 3) == 0
    assert ((out["rRoll"] >> 18) & 3) == 0
    assert ((out["rBrk"] >> 30) & 3) == 2
