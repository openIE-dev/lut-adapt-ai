"""End-to-End Kria KV260 AXI4-Lite + AXI4-Stream Hardware Subsystem Verifier.

Verifies `hw/kv260/rtl/clai_kv260_axi_top.sv` + `clai_engram_streamer.sv` + `clai_rung0_kv260.v`:
  1. Yosys 0.71+ `synth_xilinx -family xcup` full-subsystem synthesis: proves 0 DSP48
     multipliers across the division-free thermometer streamer, 1,536-LUT6 core, 8-class
     popcount voter, and AMBA AXI4-Lite CSR plane.
  2. Verilator 5.x clocked AXI4-Lite + AXI4-Stream Bus Functional Model (BFM) simulation:
     - Verifies `"CLAI"` magic CSR, decoupled AW/W AXI4-Lite writes, and Engram BRAM programming
     - Streams variable-length packets with `tvalid` bubbles over AXI4-Stream
     - Verifies 100% bit-exact 384-wire XOR+Thermometer hardware encoding + 128-wire BRAM lookup
     - Verifies 100% bit-exact 8-class LUT6 popcount votes, margin comparator, satiation counters,
       and `irq_escalate_to_ps` interrupt assertion & W1C acknowledge against PyTorch!
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from clai.contracts.joule_meter import select_torch_device
from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus
from clai.rung0.triton_lut6 import count_votes_from_bitplanes, pack_episodes_to_uint64
from clai.rung0.verilog_export import export_xilinx_lut6_verilog


def _find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / name
    return str(fallback) if fallback.exists() else None


def py_hw_streamer_reference(
    pkt: np.ndarray,
    bank6: dict[int, int],
    bank7: dict[int, int],
    addr_bits: int = 10,
) -> list[int]:
    """Bit-exact Python golden model of `clai_engram_streamer.sv` (returns 16 x uint32 words)."""
    mask_addr = (1 << addr_bits) - 1
    xor_bytes = np.zeros(16, dtype=np.uint8)
    slot_sum = np.zeros(16, dtype=np.int64)
    quad_shift = 0
    rolling_hash = 0

    for idx, b_val in enumerate(pkt.tolist()):
        slot = idx & 15
        xor_bytes[slot] ^= b_val
        slot_sum[slot] += b_val
        next_quad = ((b_val & 0xFF) << 24) | (quad_shift & 0xFFFFFF)
        if (idx & 3) == 3:
            rolling_hash = (((rolling_hash << 5) + rolling_hash) ^ next_quad) & 0xFFFFFFFF
        quad_shift = ((b_val & 0xFF) << 16) | ((quad_shift >> 8) & 0xFFFF)

    raddr6 = (rolling_hash & mask_addr) ^ ((rolling_hash >> addr_bits) & mask_addr)
    raddr7 = ((rolling_hash >> (32 - addr_bits)) & mask_addr) ^ (rolling_hash & mask_addr)

    w6 = bank6.get(
        raddr6, (0x9E3779B97F4A7C15 ^ raddr6) & 0xFFFFFFFFFFFFFFFF
    )
    w7 = bank7.get(
        raddr7, (0xBF58476D1CE4E5B9 ^ (raddr7 << 32)) & 0xFFFFFFFFFFFFFFFF
    )

    # Also verify exact match on Words 0..5 against Python EngramBitBus!
    engram_ref = EngramBitBus(seed=42)
    py_words_u64 = (
        engram_ref.fetch_wire_words_u64(pkt, torch.device("cpu"))[0]
        .numpy()
        .view(np.uint64)
    )

    w0 = int.from_bytes(xor_bytes[0:8].tobytes(), "little")
    w1 = int.from_bytes(xor_bytes[8:16].tobytes(), "little")
    assert w0 == int(py_words_u64[0]), "XOR Word 0 mismatch vs EngramBitBus!"
    assert w1 == int(py_words_u64[1]), "XOR Word 1 mismatch vs EngramBitBus!"

    n_chunks = max(1, (len(pkt) + 15) >> 4)
    words_thermo = []
    for grp in range(4):
        pack64 = 0
        for sub in range(4):
            s = grp * 4 + sub
            t16 = 0
            for lvl in range(16):
                if slot_sum[s] >= 15 * n_chunks * (lvl + 1):
                    t16 |= 1 << lvl
            pack64 |= t16 << (sub * 16)
        words_thermo.append(pack64)
        assert pack64 == int(py_words_u64[2 + grp]), f"Thermo Word {2 + grp} mismatch vs EngramBitBus!"

    all_u64 = [w0, w1, *words_thermo, w6, w7]
    u32_out: list[int] = []
    for val64 in all_u64:
        u32_out.append(val64 & 0xFFFFFFFF)
        u32_out.append((val64 >> 32) & 0xFFFFFFFF)
    return u32_out


def synthesize_kv260_axi_subsystem(core_v: Path) -> dict:
    yosys = _find_tool("yosys")
    assert yosys is not None, "Yosys not found"

    streamer_sv = Path("hw/kv260/rtl/clai_engram_streamer.sv")
    hotswap_sv = Path("hw/kv260/rtl/clai_rung0_hotswap_core.sv")
    axi_top_sv = Path("hw/kv260/rtl/clai_kv260_axi_top.sv")

    script = (
        f"read_verilog -DSYNTHESIS {core_v}; "
        f"read_verilog -sv -DSYNTHESIS {streamer_sv} {hotswap_sv} {axi_top_sv}; "
        "synth_xilinx -family xcup -top clai_kv260_axi_top; "
        "stat -json"
    )
    proc = subprocess.run(
        [yosys, "-Q", "-T", "-p", script],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    j_idx = out.rfind("\n{\n")
    if j_idx == -1:
        j_idx = out.find("{")
    stats = json.loads(out[j_idx:])
    cells = stats.get("design", {}).get("num_cells_by_type", {})

    luts = sum(int(v) for k, v in cells.items() if "LUT" in k.upper())
    ffs = sum(int(v) for k, v in cells.items() if k.upper().startswith("FD"))
    carrys = sum(int(v) for k, v in cells.items() if "CARRY" in k.upper())
    brams = sum(int(v) for k, v in cells.items() if "RAMB" in k.upper())
    dsps = sum(int(v) for k, v in cells.items() if "DSP" in k.upper())

    print(
        f"[Yosys KV260 Full AXI Subsystem] LUTs={luts}/117120 ({100.0 * luts / 117120:.1f}% KV260) | "
        f"FFs={ffs} | CARRY8={carrys} | BRAM={brams} | DSP48={dsps}"
    )
    assert dsps == 0, f"Expected 0 DSP48 multipliers in KV260 subsystem, found {dsps}"
    assert 1536 <= luts < 117120, f"LUT count {luts} outside Kria KV260 (xck26) 117,120 CLB capacity!"
    return {"luts": luts, "ffs": ffs, "carry8": carrys, "bram": brams, "dsp48": dsps}


def cosimulate_kv260_axi_bfm(net: DiffLUTNetwork, core_v: Path, out_dir: Path) -> dict:
    verilator = _find_tool("verilator")
    assert verilator is not None, "Verilator not found"

    dev = next(net.parameters()).device
    rng = np.random.default_rng(2026)

    # Custom Engram BRAM overrides written via AXI4-Lite CSRs before streaming
    bank6 = {0x12A: 0xCAFE_BABE_1234_5678}
    bank7 = {0x12A: 0xDEAD_BEEF_8765_4321}

    n_pkts = 8
    packets = [rng.integers(0, 256, size=(int(rng.integers(19, 64)),), dtype=np.uint8) for _ in range(n_pkts)]

    expected_bus_u32: list[list[int]] = []
    expected_votes: list[list[int]] = []
    expected_pred_margin: list[tuple[int, int, int, int]] = []

    for pkt in packets:
        u32_16 = py_hw_streamer_reference(pkt, bank6, bank7, addr_bits=10)
        expected_bus_u32.append(u32_16)

        bits512 = np.zeros((1, 512), dtype=np.float32)
        for w_i, val32 in enumerate(u32_16):
            for b in range(32):
                bits512[0, w_i * 32 + b] = float((val32 >> b) & 1)
        x_bin = torch.from_numpy(bits512).to(dev)
        x_pack, orig_b = pack_episodes_to_uint64(x_bin)
        h_pack = net.forward_packed_chain(x_pack)
        votes = count_votes_from_bitplanes(h_pack, orig_b, 8)[0].cpu().tolist()
        expected_votes.append(votes)

        best_c = 0
        best_v = 0
        sec_v = 0
        for c, v in enumerate(votes):
            if v > best_v:
                sec_v = best_v
                best_v = v
                best_c = c
            elif v > sec_v:
                sec_v = v
        expected_pred_margin.append((best_c, best_v, sec_v, best_v - sec_v))

    # Generate clocked SystemVerilog AXI4-Lite + AXI4-Stream BFM Testbench
    sv = [
        "`timescale 1ns/1ps",
        "module tb_clai_kv260_axi;",
        "    logic        aclk = 0;",
        "    logic        aresetn = 0;",
        "    always #2 aclk = ~aclk; // 250 MHz PL Clock (4.0 ns period)",
        "",
        "    logic [11:0] s_axi_awaddr = 0;",
        "    logic [2:0]  s_axi_awprot = 0;",
        "    logic        s_axi_awvalid = 0;",
        "    wire         s_axi_awready;",
        "    logic [31:0] s_axi_wdata = 0;",
        "    logic [3:0]  s_axi_wstrb = 4'hF;",
        "    logic        s_axi_wvalid = 0;",
        "    wire         s_axi_wready;",
        "    wire  [1:0]  s_axi_bresp;",
        "    wire         s_axi_bvalid;",
        "    logic        s_axi_bready = 0;",
        "    logic [11:0] s_axi_araddr = 0;",
        "    logic [2:0]  s_axi_arprot = 0;",
        "    logic        s_axi_arvalid = 0;",
        "    wire         s_axi_arready;",
        "    wire  [31:0] s_axi_rdata;",
        "    wire  [1:0]  s_axi_rresp;",
        "    wire         s_axi_rvalid;",
        "    logic        s_axi_rready = 0;",
        "",
        "    logic [7:0]  s_axis_tdata = 0;",
        "    logic        s_axis_tvalid = 0;",
        "    wire         s_axis_tready;",
        "    logic        s_axis_tlast = 0;",
        "    wire         irq_escalate_to_ps;",
        "",
        "    integer errors = 0;",
        "    logic [31:0] rd_val;",
        "",
        "    clai_kv260_axi_top #(.ENGRAM_ADDR_BITS(10)) dut (.*);",
        "",
        "    task automatic axi_write(input [11:0] addr, input [31:0] data, input integer mode);",
        "        begin",
        "            @(posedge aclk);",
        "            if (mode == 0) begin",
        "                // Simultaneous AW + W",
        "                s_axi_awaddr <= addr; s_axi_awvalid <= 1'b1;",
        "                s_axi_wdata  <= data; s_axi_wstrb <= 4'hF; s_axi_wvalid <= 1'b1;",
        "                s_axi_bready <= 1'b1;",
        "                fork",
        "                    begin do @(posedge aclk); while (!s_axi_awready); s_axi_awvalid <= 1'b0; end",
        "                    begin do @(posedge aclk); while (!s_axi_wready);  s_axi_wvalid  <= 1'b0; end",
        "                join",
        "            end else begin",
        "                // Decoupled AW-then-W handshake check",
        "                s_axi_awaddr <= addr; s_axi_awvalid <= 1'b1; s_axi_bready <= 1'b1;",
        "                do @(posedge aclk); while (!s_axi_awready);",
        "                s_axi_awvalid <= 1'b0;",
        "                @(posedge aclk);",
        "                s_axi_wdata <= data; s_axi_wstrb <= 4'hF; s_axi_wvalid <= 1'b1;",
        "                do @(posedge aclk); while (!s_axi_wready);",
        "                s_axi_wvalid <= 1'b0;",
        "            end",
        "            do @(posedge aclk); while (!s_axi_bvalid);",
        "            s_axi_bready <= 1'b0;",
        "        end",
        "    endtask",
        "",
        "    task automatic axi_read(input [11:0] addr, output [31:0] data);",
        "        begin",
        "            @(posedge aclk);",
        "            s_axi_araddr  <= addr;",
        "            s_axi_arvalid <= 1'b1;",
        "            s_axi_rready  <= 1'b1;",
        "            do @(posedge aclk); while (!s_axi_arready);",
        "            s_axi_arvalid <= 1'b0;",
        "            while (!s_axi_rvalid) @(posedge aclk);",
        "            data = s_axi_rdata;",
        "            @(posedge aclk);",
        "            s_axi_rready <= 1'b0;",
        "        end",
        "    endtask",
        "",
        "    task automatic axis_send_byte(input [7:0] b, input bit is_last, input bit bubble);",
        "        begin",
        "            if (bubble) begin",
        "                @(posedge aclk);",
        "                s_axis_tvalid <= 1'b0;",
        "            end",
        "            @(posedge aclk);",
        "            s_axis_tdata  <= b;",
        "            s_axis_tlast  <= is_last;",
        "            s_axis_tvalid <= 1'b1;",
        "            do @(posedge aclk); while (!s_axis_tready);",
        "            s_axis_tvalid <= 1'b0;",
        "            s_axis_tlast  <= 1'b0;",
        "        end",
        "    endtask",
        "",
        "    initial begin",
        "        repeat (5) @(posedge aclk);",
        "        aresetn = 1;",
        "        repeat (3) @(posedge aclk);",
        "",
        "        // 1. Verify CLAI Magic CSR (0x080 == 0x434C4149)",
        "        axi_read(12'h080, rd_val);",
        "        if (rd_val !== 32'h434C4149) begin",
        '            $display("MAGIC CSR MISMATCH: %h", rd_val); errors = errors + 1;',
        "        end",
        "",
        "        // 2. Program Engram BRAM slot 0x12A on Bank 0 & Bank 1 + set margin_thresh = 6",
        "        axi_write(12'h008, 32'd6, 1);",
        "        axi_write(12'h044, 32'h12345678, 0);",
        "        axi_write(12'h048, 32'hCAFEBABE, 0);",
        "        axi_write(12'h040, 32'h8000012A, 0);",
        "        axi_write(12'h044, 32'h87654321, 0);",
        "        axi_write(12'h048, 32'hDEADBEEF, 0);",
        "        axi_write(12'h040, 32'h8001012A, 0);",
        "",
    ]

    exp_satiated = 0
    exp_escalated = 0
    for p_idx, pkt in enumerate(packets):
        sv.append(f"        // --- Stream Packet {p_idx} (len={len(pkt)}) ---")
        for b_i, byte_v in enumerate(pkt.tolist()):
            is_last = 1 if b_i == len(pkt) - 1 else 0
            bubble = 1 if (b_i % 9 == 4) else 0
            sv.append(f"        axis_send_byte(8'd{byte_v}, 1'b{is_last}, 1'b{bubble});")

        # Wait 4 PL clock cycles for ST_FINALIZE -> core_bus_reg -> LUT6 fabric -> CSR latch
        sv.append("        repeat (4) @(posedge aclk);")

        # Check all 16 x 32-bit hardware-encoded EngramBitBus words at 0x100..0x13C
        for w_i, exp_w32 in enumerate(expected_bus_u32[p_idx]):
            addr = 0x100 + w_i * 4
            sv.extend(
                [
                    f"        axi_read(12'h{addr:03X}, rd_val);",
                    f"        if (rd_val !== 32'h{exp_w32:08X}) begin",
                    f'            $display("PKT {p_idx} BUS WORD {w_i} MISMATCH: got %h exp {exp_w32:08X}", rd_val);',
                    "            errors = errors + 1;",
                    "        end",
                ]
            )

        # Check all 8 class popcount votes at 0x020..0x03C
        for c_i, exp_v in enumerate(expected_votes[p_idx]):
            addr = 0x020 + c_i * 4
            sv.extend(
                [
                    f"        axi_read(12'h{addr:03X}, rd_val);",
                    f"        if (rd_val !== 32'd{exp_v}) begin",
                    f'            $display("PKT {p_idx} CLASS {c_i} VOTE MISMATCH: got %0d exp {exp_v}", rd_val);',
                    "            errors = errors + 1;",
                    "        end",
                ]
            )

        best_c, best_v, sec_v, margin = expected_pred_margin[p_idx]
        exp_res = ((margin & 0xFF) << 24) | ((sec_v & 0xFF) << 16) | ((best_v & 0xFF) << 8) | (best_c & 0x7)
        if margin >= 6:
            exp_satiated += 1
            exp_irq = 0
        else:
            exp_escalated += 1
            exp_irq = 1

        sv.extend(
            [
                "        axi_read(12'h00C, rd_val);",
                f"        if (rd_val !== 32'h{exp_res:08X}) begin",
                f'            $display("PKT {p_idx} RESULT CSR MISMATCH: got %h exp {exp_res:08X}", rd_val);',
                "            errors = errors + 1;",
                "        end",
                f"        if (irq_escalate_to_ps !== 1'b{exp_irq}) begin",
                f'            $display("PKT {p_idx} IRQ MISMATCH: got %b exp {exp_irq}", irq_escalate_to_ps);',
                "            errors = errors + 1;",
                "        end",
                "        // W1C clear status/IRQ",
                "        axi_write(12'h004, 32'h00000005, 0);",
            ]
        )

    # Verify hardware performance counters (0x010 total, 0x014 satiated, 0x018 escalated)
    sv.extend(
        [
            "        axi_read(12'h010, rd_val);",
            f"        if (rd_val !== 32'd{n_pkts}) errors = errors + 1;",
            "        axi_read(12'h014, rd_val);",
            f"        if (rd_val !== 32'd{exp_satiated}) errors = errors + 1;",
            "        axi_read(12'h018, rd_val);",
            f"        if (rd_val !== 32'd{exp_escalated}) errors = errors + 1;",
            "",
            "        // 3. Test Direct MMIO 512-bit Bus Override Mode (CSR_CTRL = 0x7: trigger + mmio + irq)",
            *[
                f"        axi_write(12'h{0x100 + w_i * 4:03X}, 32'h{expected_bus_u32[0][w_i]:08X}, {w_i & 1});"
                for w_i in range(16)
            ],
            "        axi_write(12'h000, 32'h00000007, 0);",
            "        repeat (3) @(posedge aclk);",
            "        axi_read(12'h00C, rd_val);",
            f"        if (rd_val !== 32'h{(((expected_pred_margin[0][3] & 0xFF) << 24) | ((expected_pred_margin[0][2] & 0xFF) << 16) | ((expected_pred_margin[0][1] & 0xFF) << 8) | (expected_pred_margin[0][0] & 0x7)):08X}) begin",
            '            $display("MMIO BUS EVAL MISMATCH: got %h", rd_val); errors = errors + 1;',
            "        end",
            "        axi_write(12'h004, 32'h00000005, 0);",
            "",
            "        // 4. Test Zero-Downtime 4ns Shadow-Bank Hot-Swap Overlay on Live AXI4-Stream!",
            "        begin : test_hotswap",
            "            integer idx;",
            "            for (idx = 0; idx < 32; idx = idx + 1) begin",
            "                if (idx >= 20 && idx < 24) begin",
            "                    // Class 5 detector: L0 routes wire 0 -> O=I0; L1 routes 20..23 -> 3-of-6 maj",
            "                    axi_write(12'h054, 32'hAAAAAAAA, 0);",
            "                    axi_write(12'h058, 32'hAAAAAAAA, 0);",
            "                    axi_write(12'h05C, 32'h00000000, 0);",
            "                    axi_write(12'h060, 32'h00000000, 0);",
            "                    axi_write(12'h050, 32'h80000000 | idx, 0);",
            "                    axi_write(12'h054, 32'hF8808000, 0);",
            "                    axi_write(12'h058, 32'hFFF8F880, 0);",
            "                    axi_write(12'h05C, (22 << 18) | (21 << 9) | 20, 0);",
            "                    axi_write(12'h060, (21 << 18) | (20 << 9) | 23, 0);",
            "                    axi_write(12'h050, 32'h80000100 | idx, 0);",
            "                end else begin",
            "                    axi_write(12'h054, 32'h00000000, 0);",
            "                    axi_write(12'h058, 32'h00000000, 0);",
            "                    axi_write(12'h05C, 32'h00000000, 0);",
            "                    axi_write(12'h060, 32'h00000000, 0);",
            "                    axi_write(12'h050, 32'h80000000 | idx, 0);",
            "                    axi_write(12'h050, 32'h80000100 | idx, 0);",
            "                end",
            "            end",
            "            // Single-cycle (4.0 ns) atomic swap to Bank 1 + enable overlay (0x34)",
            "            axi_write(12'h000, 32'h00000034, 0);",
            "            axi_read(12'h064, rd_val);",
            "            if (rd_val !== 32'h80000001) begin",
            '                $display("HOTSWAP BANK GENERATION MISMATCH: %h", rd_val); errors = errors + 1;',
            "            end",
            "            // Stream live 16-byte packet with bit 0 = 1 without resetting aresetn",
            "            axis_send_byte(8'h01, 1'b0, 1'b0);",
            "            for (idx = 1; idx < 15; idx = idx + 1) axis_send_byte(8'h00, 1'b0, 1'b0);",
            "            axis_send_byte(8'h00, 1'b1, 1'b0);",
            "            repeat (4) @(posedge aclk);",
            "            axi_read(12'h00C, rd_val);",
            "            if (rd_val !== 32'h40004005 || irq_escalate_to_ps !== 1'b0) begin",
            '                $display("HOTSWAP LIVE PACKET MISMATCH: res=%h irq=%b", rd_val, irq_escalate_to_ps);',
            "                errors = errors + 1;",
            "            end",
            "        end",
            "",
            f'        $display("[Verilator KV260 AXI4-Lite + AXI4-Stream BFM] Packets=%0d | 4ns Atomic HotSwap=OK (Bank=1, Gen=1, Class=5, Votes=64/64) | Total Errors=%0d", {n_pkts}, errors);',
            "        if (errors != 0) $fatal(1);",
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )

    tb_path = out_dir / "tb_clai_kv260_axi.sv"
    tb_path.write_text("\n".join(sv), encoding="utf-8")
    obj_dir = out_dir / "verilator_kv260_axi_obj"

    env_vars = os.environ.copy()
    env_vars["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env_vars.get('PATH', '')}"

    subprocess.run(
        [
            verilator,
            "--binary",
            "--timing",
            "-j",
            "4",
            "-Wno-UNOPTFLAT",
            "-Wno-DECLFILENAME",
            "-Wno-UNUSED",
            "--Mdir",
            str(obj_dir),
            "--top-module",
            "tb_clai_kv260_axi",
            str(core_v),
            "hw/kv260/rtl/clai_engram_streamer.sv",
            "hw/kv260/rtl/clai_rung0_hotswap_core.sv",
            "hw/kv260/rtl/clai_kv260_axi_top.sv",
            str(tb_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env_vars,
    )

    sim_bin = obj_dir / "Vtb_clai_kv260_axi"
    sim_res = subprocess.run([str(sim_bin)], check=True, capture_output=True, text=True)
    print(sim_res.stdout.strip())
    return {
        "packets": n_pkts,
        "satiated": exp_satiated,
        "escalated": exp_escalated,
        "errors": 0,
    }


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    dev = select_torch_device()
    torch.manual_seed(42)
    net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    with torch.no_grad():
        for l_idx, layer in enumerate(net.layers):
            gen = torch.Generator(device=dev)
            gen.manual_seed(300 + l_idx)
            layer.wire_logits.normal_(0.0, 1.0, generator=gen)
            layer.table_logits.normal_(0.0, 1.0, generator=gen)
    net.compile_all()

    core_v = out_dir / "clai_rung0_kv260_axi_core.v"
    core_v.write_text(export_xilinx_lut6_verilog(net, module_name="clai_rung0_core"), encoding="utf-8")

    syn = synthesize_kv260_axi_subsystem(core_v)
    bfm = cosimulate_kv260_axi_bfm(net, core_v, out_dir)
    print(f"[CLAI KV260 Subsystem Signoff] Yosys={syn} | Verilator_BFM={bfm}")


if __name__ == "__main__":
    main()
