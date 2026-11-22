"""Synthesizable Xilinx UltraScale+ KV260 LUT6 #(.INIT(64'h...)) primitive & voter RTL emitter."""

from __future__ import annotations

from clai.rung0.difflut_net import DiffLUTNetwork


def export_xilinx_lut6_verilog(
    net: DiffLUTNetwork,
    module_name: str = "clai_rung0_core",
    include_sim_primitive: bool = True,
) -> str:
    """
    Exports compiled DiffLUTNetwork directly to Xilinx UltraScale+ LUT6 primitives.
    Includes a guarded `ifndef SYNTHESIS behavioral LUT6 cell so the exact same .v file
    both synthesizes directly to Xilinx LUT6 silicon in Vivado/Yosys (`synth_xilinx`)
    and simulates bit-accurately in Verilator / Icarus Verilog.
    """
    in_bits = net.layers[0].in_wires
    out_bits = net.layers[-1].num_luts
    lines: list[str] = ["`timescale 1ns/1ps", ""]

    if include_sim_primitive:
        lines.extend(
            [
                "`ifndef SYNTHESIS",
                "`ifndef XILINX_LUT6_SIM_CELL",
                "`define XILINX_LUT6_SIM_CELL",
                "module LUT6 #(parameter [63:0] INIT = 64'h0000000000000000) (",
                "    output wire O,",
                "    input  wire I0,",
                "    input  wire I1,",
                "    input  wire I2,",
                "    input  wire I3,",
                "    input  wire I4,",
                "    input  wire I5",
                ");",
                "    assign O = INIT[{I5, I4, I3, I2, I1, I0}];",
                "endmodule",
                "`endif",
                "`endif",
                "",
            ]
        )

    lines.extend(
        [
            f"module {module_name} (",
            f"    input  wire [{in_bits - 1}:0] bus_in,",
            f"    output wire [{out_bits - 1}:0] lut_votes_out",
            ");",
        ]
    )

    prev_sig = "bus_in"
    for l_idx, layer in enumerate(net.layers):
        cur_sig = f"stage_{l_idx}"
        lines.append(f"    wire [{layer.num_luts - 1}:0] {cur_sig};")
        wires = layer.compiled_wires.cpu().numpy()
        tables = layer.compiled_tables.cpu().numpy()
        for n_idx in range(layer.num_luts):
            init_val = 0
            for bit_pos in range(64):
                if int(tables[n_idx, bit_pos]) == 1:
                    init_val |= 1 << bit_pos
            w = wires[n_idx]
            lines.append(
                f"    LUT6 #(.INIT(64'h{init_val:016X})) u_l{l_idx}_n{n_idx} ("
                f".O({cur_sig}[{n_idx}]), "
                f".I0({prev_sig}[{w[0]}]), .I1({prev_sig}[{w[1]}]), .I2({prev_sig}[{w[2]}]), "
                f".I3({prev_sig}[{w[3]}]), .I4({prev_sig}[{w[4]}]), .I5({prev_sig}[{w[5]}]));"
            )
        prev_sig = cur_sig

    lines.append(f"    assign lut_votes_out = {prev_sig};")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)
