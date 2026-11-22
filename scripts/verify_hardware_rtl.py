"""Hardware RTL Synthesis (Yosys) & Bit-Exact Silicon Co-Simulation (Verilator) Verifier.

Proves two hardware invariants on the exported `clai_rung0_kv260.v` netlist:
  1. Yosys Xilinx UltraScale+ (`xcup` / Kria KV260) synthesis maps the core into exact LUT6
     cells with 0 DSP48 multipliers.
  2. Verilator 5.x RTL simulation of `clai_rung0_kv260.v` produces 100% bit-identical 512-wire
     outputs (`8,192 / 8,192` bits) to the 64-lane uint64 DiffLUTNetwork engine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from clai.contracts.joule_meter import select_torch_device
from clai.contracts.rce_gate import ModularConstraintEnv
from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus
from clai.rung0.triton_lut6 import transpose_engram_words_to_bitplanes, unpack_uint64_to_episodes
from clai.rung0.verilog_export import export_xilinx_lut6_verilog


def _find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    return str(fallback) if fallback.exists() else None


def verify_yosys_xilinx_synthesis(verilog_path: Path) -> dict:
    yosys = _find_tool("yosys")
    if not yosys:
        return {"skipped": True}

    import json

    cmd = [
        yosys,
        "-Q",
        "-T",
        "-p",
        f"read_verilog -DSYNTHESIS {verilog_path}; synth_xilinx -family xcup -top clai_rung0_core; stat -json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout
    json_start = out.rfind("\n{\n")
    if json_start == -1:
        json_start = out.find("{")
    stats = json.loads(out[json_start:])
    cells = stats.get("design", {}).get("num_cells_by_type", {})
    total_luts = sum(int(v) for k, v in cells.items() if "LUT" in k.upper())
    total_dsps = sum(int(v) for k, v in cells.items() if "DSP" in k.upper())
    print(
        f"[Yosys UltraScale+ KV260 Synthesis] Synthesized Xilinx LUT Primitives: {total_luts} | "
        f"DSP48 Multipliers: {total_dsps} (Zero-MAC Verified!)"
    )
    assert total_dsps == 0, f"Expected 0 DSP48 blocks, found {total_dsps}"
    assert total_luts > 0, f"Expected synthesized Xilinx LUT cells > 0, got cells={cells}"
    return {"skipped": False, "lut_cells": total_luts, "dsp_cells": total_dsps}


def verify_verilator_bit_exact_rtl(
    net: DiffLUTNetwork, engram: EngramBitBus, verilog_path: Path, work_dir: Path
) -> dict:
    verilator = _find_tool("verilator")
    if not verilator:
        return {"skipped": True}

    dev = next(net.parameters()).device
    env = ModularConstraintEnv(seed=99)
    n_vecs = 16
    episodes, _ = env.sample_microbatch(n_vecs)
    words_u64 = engram.fetch_batch_words_u64(episodes, dev)  # [16, 8] int64

    # Expected 512-bit final LUT outputs from 64-lane uint64 packed hardware chain
    planes_in, orig_b = transpose_engram_words_to_bitplanes(words_u64)
    planes_out = net.forward_packed_chain(planes_in)
    expected_bits = unpack_uint64_to_episodes(planes_out, orig_b).cpu().to(torch.int64)  # [16, 512]
    input_bits = engram.unpack_words_to_float_bus(words_u64).cpu().to(torch.int64)  # [16, 512]

    def bits512_to_hex(row: torch.Tensor) -> str:
        val = 0
        for b_idx in range(512):
            if int(row[b_idx].item()) == 1:
                val |= 1 << b_idx
        return f"512'h{val:0128X}"

    tb_lines = [
        "`timescale 1ns/1ps",
        "module tb_clai_rung0;",
        "    logic [511:0] bus_in;",
        "    wire  [511:0] lut_votes_out;",
        "    integer errors = 0;",
        "",
        "    clai_rung0_core dut (",
        "        .bus_in(bus_in),",
        "        .lut_votes_out(lut_votes_out)",
        "    );",
        "",
        "    initial begin",
    ]
    for i in range(n_vecs):
        in_hex = bits512_to_hex(input_bits[i])
        exp_hex = bits512_to_hex(expected_bits[i])
        tb_lines.extend(
            [
                f"        bus_in = {in_hex};",
                "        #1;",
                f"        if (lut_votes_out !== {exp_hex}) begin",
                f'            $display("MISMATCH at vec {i}: got %h", lut_votes_out);',
                "            errors = errors + 1;",
                "        end",
            ]
        )
    tb_lines.extend(
        [
            '        $display("[Verilator RTL Co-Sim] Vectors=%0d | Total Bit-Vector Errors=%0d", '
            f"{n_vecs}, errors);",
            "        if (errors != 0) $fatal(1);",
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )

    tb_path = work_dir / "tb_clai_rung0.sv"
    tb_path.write_text("\n".join(tb_lines), encoding="utf-8")
    obj_dir = work_dir / "verilator_obj"

    env_vars = os.environ.copy()
    env_vars["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env_vars.get('PATH', '')}"

    subprocess.run(
        [
            verilator,
            "--binary",
            "-j",
            "4",
            "-Wno-UNOPTFLAT",
            "-Wno-DECLFILENAME",
            "--Mdir",
            str(obj_dir),
            "--top-module",
            "tb_clai_rung0",
            str(verilog_path),
            str(tb_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env_vars,
    )

    sim_bin = obj_dir / "Vtb_clai_rung0"
    sim_res = subprocess.run([str(sim_bin)], check=True, capture_output=True, text=True)
    print(sim_res.stdout.strip())
    return {"skipped": False, "vectors": n_vecs, "wires_checked": n_vecs * 512, "errors": 0}


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    dev = select_torch_device()
    torch.manual_seed(42)
    engram = EngramBitBus(seed=42)
    net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    with torch.no_grad():
        for l_idx, layer in enumerate(net.layers):
            gen = torch.Generator(device=dev)
            gen.manual_seed(100 + l_idx)
            layer.wire_logits.normal_(0.0, 1.0, generator=gen)
            layer.table_logits.normal_(0.0, 1.0, generator=gen)
    net.compile_all()

    v_path = out_dir / "clai_rung0_kv260_verify.v"
    v_path.write_text(export_xilinx_lut6_verilog(net), encoding="utf-8")

    syn_stats = verify_yosys_xilinx_synthesis(v_path)
    sim_stats = verify_verilator_bit_exact_rtl(net, engram, v_path, out_dir)
    print(f"[CLAI Hardware Signoff] Synthesis={syn_stats} | RTL_Sim={sim_stats}")


if __name__ == "__main__":
    main()
