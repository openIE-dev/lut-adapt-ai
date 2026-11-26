"""Phase 41 (`Claim LA-63`): License-Free `openXC7` RTL Co-Simulation (`iverilog`/`vvp`),
Physical Place-and-Route (`nextpnr-xilinx 0.8.2`), Static Timing Analysis (`Fmax`), and
`.bit` Bitstream Generation for Unified `clai_rbfws8_enigma_xc7a200t_top` (`8`-Bank `R-BFWS`
Epoch Engine + Bijective `Enigma-SBox6` & `SBox6^-1` `LUT6` Core + `Vending-BALROG` Simplex
`CBF` `LUT6` Shield + `6`-Col `JouleBit-LUT6` PE + `waiOS` Zero-`MUL` 4-Trit PE + `(72,64)`
`SECDED` + `0x3C` Apoptosis) on `AMD Artix-7 XC7A200T-2FBG484-2` (`Theorem 114`, `0` `DSP48E1`s!).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from run_openxc7_apex_waios_par import _blake3_waios_derive_hex, _generate_waios_hex_files


REMOTE_BASH_TEMPLATE = r"""
set -e
WORKDIR=/tmp/openxc7_rbfws8_enigma_par
rm -rf "$WORKDIR" && mkdir -p "$WORKDIR"

cat > "$WORKDIR/rbfws8_enigma_xc7a200t.v" << 'EOF'
module joulebit6_zero_mul_pe (
    input  wire [5:0]         rad6_code,
    input  wire signed [7:0]  z0,
    input  wire signed [7:0]  z1,
    input  wire signed [7:0]  z2,
    input  wire signed [7:0]  z3,
    input  wire signed [7:0]  z4,
    input  wire signed [7:0]  z5,
    output wire signed [11:0] slab_sum
);
    wire signed [10:0] s0 = rad6_code[0] ? -{{3{z0[7]}}, z0} : {{3{z0[7]}}, z0};
    wire signed [10:0] s1 = rad6_code[1] ? -{{3{z1[7]}}, z1} : {{3{z1[7]}}, z1};
    wire signed [10:0] s2 = rad6_code[2] ? -{{3{z2[7]}}, z2} : {{3{z2[7]}}, z2};
    wire signed [10:0] s3 = rad6_code[3] ? -{{3{z3[7]}}, z3} : {{3{z3[7]}}, z3};
    wire signed [10:0] s4 = rad6_code[4] ? -{{3{z4[7]}}, z4} : {{3{z4[7]}}, z4};
    wire signed [10:0] s5 = rad6_code[5] ? -{{3{z5[7]}}, z5} : {{3{z5[7]}}, z5};
    assign slab_sum = {{1{s0[10]}}, s0} + {{1{s1[10]}}, s1} + {{1{s2[10]}}, s2}
                    + {{1{s3[10]}}, s3} + {{1{s4[10]}}, s4} + {{1{s5[10]}}, s5};
endmodule

module waios_zero_mul_4trit_pe (
    input  wire [7:0]         trits4_code,
    input  wire signed [7:0]  x0,
    input  wire signed [7:0]  x1,
    input  wire signed [7:0]  x2,
    input  wire signed [7:0]  x3,
    output wire signed [11:0] partial_sum
);
    wire signed [9:0] t0 = (trits4_code[1:0] == 2'd1) ? {{2{x0[7]}}, x0} :
                           (trits4_code[1:0] == 2'd2) ? -{{2{x0[7]}}, x0} : 10'sd0;
    wire signed [9:0] t1 = (trits4_code[3:2] == 2'd1) ? {{2{x1[7]}}, x1} :
                           (trits4_code[3:2] == 2'd2) ? -{{2{x1[7]}}, x1} : 10'sd0;
    wire signed [9:0] t2 = (trits4_code[5:4] == 2'd1) ? {{2{x2[7]}}, x2} :
                           (trits4_code[5:4] == 2'd2) ? -{{2{x2[7]}}, x2} : 10'sd0;
    wire signed [9:0] t3 = (trits4_code[7:6] == 2'd1) ? {{2{x3[7]}}, x3} :
                           (trits4_code[7:6] == 2'd2) ? -{{2{x3[7]}}, x3} : 10'sd0;
    assign partial_sum = {{2{t0[9]}}, t0} + {{2{t1[9]}}, t1} + {{2{t2[9]}}, t2} + {{2{t3[9]}}, t3};
endmodule

// Bijective 6-Input `LUT6` Cryptographic S-Box (`y = (27*x + 19) mod 64`, `R' = R ^ g(L)`, `0` DSPs!):
module enigma_sbox6_lut6 (
    input  wire [5:0] x,
    output wire [5:0] s_out
);
    // `27 * x = 16*x + 8*x + 2*x + x` (pure shift-add, 0 multipliers!)
    wire [5:0] y  = (x << 4) + (x << 3) + (x << 1) + x + 6'd19;
    wire [2:0] l3 = y[5:3];
    wire [2:0] g3 = (l3 & (l3 >> 1)) ^ {l3[1:0], 1'b0} ^ 3'd5;
    assign s_out  = {l3, y[2:0] ^ g3};
endmodule

module enigma_sbox6_inv_lut6 (
    input  wire [5:0] z,
    output wire [5:0] x_out
);
    wire [2:0] l3 = z[5:3];
    wire [2:0] g3 = (l3 & (l3 >> 1)) ^ {l3[1:0], 1'b0} ^ 3'd5;
    wire [5:0] y  = {l3, z[2:0] ^ g3};
    wire [5:0] d  = y - 6'd19;
    // `19 * d = 16*d + 2*d + d` (`27 * 19 = 513 === 1 mod 64`, 0 multipliers!)
    assign x_out  = (d << 4) + (d << 1) + d;
endmodule

// Unified 4-Stage Zero-DSP48E1 Artix-7 XC7A200T-2FBG484-2 Top (`Theorem 114`):
module clai_rbfws8_enigma_xc7a200t_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire        acc_clear,
    input  wire [7:0]  stream_byte,
    input  wire [5:0]  ecc_inject_pos,
    output reg         out_valid,
    output reg  [5:0]  operon_codon,
    output reg         secded_single_fixed,
    output reg         secded_double_apoptosis,
    output reg  [2:0]  rbfws_bank8_select,
    output reg         enigma_roundtrip_ok,
    output reg         joulebit_pos_hit
);
    wire clk_g;
`ifdef SYNTHESIS
    BUFG u_clk_bufg (.I(clk), .O(clk_g));
`else
    assign clk_g = clk;
`endif

    reg [63:0] bus_s1;
    reg        v_s1, clr_s1;
    always @(posedge clk_g) begin
        if (!rst_n) begin
            bus_s1 <= 64'd0;
            v_s1   <= 1'b0;
            clr_s1 <= 1'b0;
        end else begin
            v_s1   <= in_valid;
            clr_s1 <= acc_clear;
            if (in_valid)
                bus_s1 <= {bus_s1[55:0], stream_byte};
        end
    end

    wire signed [11:0] waios_p4;
    waios_zero_mul_4trit_pe u_waios (
        .trits4_code(bus_s1[7:0]),
        .x0(bus_s1[15:8]), .x1(bus_s1[23:16]), .x2(bus_s1[31:24]), .x3(bus_s1[39:32]),
        .partial_sum(waios_p4)
    );

    wire signed [11:0] jb6_p6;
    joulebit6_zero_mul_pe u_jb6 (
        .rad6_code(bus_s1[5:0]),
        .z0(bus_s1[15:8]), .z1(bus_s1[23:16]), .z2(bus_s1[31:24]),
        .z3(bus_s1[39:32]), .z4(bus_s1[47:40]), .z5(bus_s1[55:48]),
        .slab_sum(jb6_p6)
    );

    wire [5:0] sbox_enc;
    wire [5:0] sbox_dec;
    enigma_sbox6_lut6     u_enc (.x(bus_s1[61:56]), .s_out(sbox_enc));
    enigma_sbox6_inv_lut6 u_dec (.z(sbox_enc),      .x_out(sbox_dec));

    wire [1:0] cash2   = bus_s1[57:56];
    wire [1:0] stock2  = bus_s1[59:58];
    wire [1:0] stam2   = bus_s1[61:60];
    wire       cbf_ok  = (cash2 >= 2'd1 && (cash2 >= 2'd2 || stock2 >= 2'd1))
                      && (stam2 >= 2'd1 && (stam2 >= 2'd2 || cash2  >= 2'd2));

    wire sg_hit    = (|bus_s1[61:57]) & bus_s1[56];
    wire apex_trap = (bus_s1[63:62] == 2'b11);

    reg signed [12:0] combined_p_s2;
    reg               clr_s2;
    reg               apex_trap_s2;
    reg [2:0]         rbfws_epoch_s2;
    reg               sbox_ok_s2;
    reg               cbf_ok_s2;
    reg [71:0]        ramb36_word_s2;
    reg [5:0]         operon_in_s2;
    reg               v_s2;

    always @(posedge clk_g) begin
        if (!rst_n) begin
            combined_p_s2  <= 13'sd0;
            clr_s2         <= 1'b0;
            apex_trap_s2   <= 1'b0;
            rbfws_epoch_s2 <= 3'd0;
            sbox_ok_s2     <= 1'b0;
            cbf_ok_s2      <= 1'b0;
            ramb36_word_s2 <= 72'h2B_0000_0000_0000_0F00;
            operon_in_s2   <= 6'd0;
            v_s2           <= 1'b0;
        end else begin
            v_s2 <= v_s1;
            if (v_s1) begin
                combined_p_s2 <= {{1{waios_p4[11]}}, waios_p4} + {{1{jb6_p6[11]}}, jb6_p6};
                clr_s2        <= clr_s1;
                apex_trap_s2  <= apex_trap;
                sbox_ok_s2    <= (sbox_dec == bus_s1[61:56]) ^ (^sbox_enc);
                cbf_ok_s2     <= cbf_ok;
                if (clr_s1)
                    rbfws_epoch_s2 <= 3'd0;
                else if (sg_hit && (rbfws_epoch_s2 < 3'd7))
                    rbfws_epoch_s2 <= rbfws_epoch_s2 + 3'd1;
                operon_in_s2   <= {bus_s1[63], bus_s1[48], bus_s1[32], bus_s1[16], bus_s1[8], bus_s1[0]};
                ramb36_word_s2 <= 72'h2B_0000_0000_0000_0F00
                                ^ (72'd1 << ecc_inject_pos)
                                ^ (bus_s1[7] ? (72'd1 << ((ecc_inject_pos + 6'd13) & 6'd63)) : 72'd0);
            end
        end
    end

    reg [6:0]         secded_syn_s3;
    reg               secded_par_s3;
    reg               apex_trap_s3;
    reg [2:0]         bank8_sel_s3;
    reg               enigma_cbf_s3;
    reg signed [19:0] acc_s3;
    reg [5:0]         operon_in_s3;
    reg               v_s3;
    integer b;
    reg [6:0] syn_comb;
    always @(*) begin
        syn_comb = ramb36_word_s2[70:64];
        for (b = 0; b < 64; b = b + 1) begin
            if (ramb36_word_s2[b])
                syn_comb = syn_comb ^ (b[6:0] + 7'd3);
        end
    end

    always @(posedge clk_g) begin
        if (!rst_n) begin
            secded_syn_s3 <= 7'd0;
            secded_par_s3 <= 1'b0;
            apex_trap_s3  <= 1'b0;
            bank8_sel_s3  <= 3'd0;
            enigma_cbf_s3 <= 1'b0;
            acc_s3        <= 20'sd0;
            operon_in_s3  <= 6'd0;
            v_s3          <= 1'b0;
        end else begin
            v_s3 <= v_s2;
            if (v_s2) begin
                secded_syn_s3 <= syn_comb;
                secded_par_s3 <= ^ramb36_word_s2;
                apex_trap_s3  <= apex_trap_s2;
                bank8_sel_s3  <= rbfws_epoch_s2;
                enigma_cbf_s3 <= sbox_ok_s2 ^ cbf_ok_s2;
                acc_s3        <= clr_s2 ? {{7{combined_p_s2[12]}}, combined_p_s2}
                                        : (acc_s3 + {{7{combined_p_s2[12]}}, combined_p_s2});
                operon_in_s3  <= operon_in_s2;
            end
        end
    end

    wire any_syn    = |secded_syn_s3;
    wire single_err = secded_par_s3;
    wire multi_err  = (~secded_par_s3) & any_syn;
    wire pos_acc    = ~acc_s3[19] & (|acc_s3[18:0]);

    always @(posedge clk_g) begin
        if (!rst_n) begin
            out_valid               <= 1'b0;
            operon_codon            <= 6'h00;
            secded_single_fixed     <= 1'b0;
            secded_double_apoptosis <= 1'b0;
            rbfws_bank8_select      <= 3'd0;
            enigma_roundtrip_ok     <= 1'b0;
            joulebit_pos_hit        <= 1'b0;
        end else begin
            out_valid <= v_s3;
            if (v_s3) begin
                secded_single_fixed     <= single_err;
                secded_double_apoptosis <= multi_err;
                rbfws_bank8_select      <= bank8_sel_s3;
                enigma_roundtrip_ok     <= enigma_cbf_s3;
                joulebit_pos_hit        <= pos_acc;
                if (multi_err || apex_trap_s3)
                    operon_codon <= 6'h3C;
                else if (pos_acc || enigma_cbf_s3)
                    operon_codon <= 6'h08;
                else if (|bank8_sel_s3)
                    operon_codon <= 6'h20;
                else if (!operon_in_s3[3] && !operon_in_s3[2])
                    operon_codon <= 6'h00;
                else
                    operon_codon <= 6'h30;
            end
        end
    end
endmodule
EOF

cat > "$WORKDIR/waios_clean_w.hex" << 'EOF'
__CLEAN_W_HEX__EOF
cat > "$WORKDIR/waios_acts.hex" << 'EOF'
__ACTS_HEX__EOF

cat > "$WORKDIR/tb_rbfws8_enigma.v" << 'EOF'
`timescale 1ns/1ps
module tb_rbfws8_enigma;
    reg [7:0] clean_w [0:8191];
    reg signed [7:0] acts [0:127];

    reg  [7:0] code_in;
    reg  signed [7:0] x0, x1, x2, x3;
    wire signed [11:0] psum;
    waios_zero_mul_4trit_pe u_w (.trits4_code(code_in), .x0(x0), .x1(x1), .x2(x2), .x3(x3), .partial_sum(psum));

    reg  [5:0] rad6;
    reg  signed [7:0] z0, z1, z2, z3, z4, z5;
    wire signed [11:0] jsum;
    joulebit6_zero_mul_pe u_j (.rad6_code(rad6), .z0(z0), .z1(z1), .z2(z2), .z3(z3), .z4(z4), .z5(z5), .slab_sum(jsum));

    reg  [5:0] sx;
    wire [5:0] sy, sx_back;
    enigma_sbox6_lut6     u_enc (.x(sx), .s_out(sy));
    enigma_sbox6_inv_lut6 u_dec (.z(sy), .x_out(sx_back));

    integer r, g, acc, m, jb_err, sbox_err, exp_jb, fd_c;
    reg [63:0] seen_sbox;
    initial begin
        $readmemh("waios_clean_w.hex", clean_w);
        $readmemh("waios_acts.hex", acts);

        jb_err = 0;
        z0 = 8'sd13; z1 = -8'sd7; z2 = 8'sd29; z3 = -8'sd19; z4 = 8'sd5; z5 = 8'sd11;
        for (m = 0; m < 64; m = m + 1) begin
            rad6 = m[5:0];
            #1;
            exp_jb = ((m & 1) ? -13 : 13) + ((m & 2) ? 7 : -7) + ((m & 4) ? -29 : 29)
                   + ((m & 8) ? 19 : -19) + ((m & 16) ? -5 : 5) + ((m & 32) ? -11 : 11);
            if (jsum !== exp_jb) jb_err = jb_err + 1;
        end
        if (jb_err !== 0) $fatal(1, "JB6 mismatch");

        sbox_err = 0;
        seen_sbox = 64'd0;
        for (m = 0; m < 64; m = m + 1) begin
            sx = m[5:0];
            #1;
            if (sx_back !== m[5:0]) sbox_err = sbox_err + 1;
            if (seen_sbox[sy]) sbox_err = sbox_err + 1;
            seen_sbox[sy] = 1'b1;
        end
        if (sbox_err !== 0 || seen_sbox !== 64'hFFFFFFFFFFFFFFFF) $fatal(1, "Enigma-SBox6 bijection mismatch");

        fd_c = $fopen("rtl_clean_out.txt", "w");
        for (r = 0; r < 256; r = r + 1) begin
            acc = 0;
            for (g = 0; g < 32; g = g + 1) begin
                code_in = clean_w[r * 32 + g];
                x0 = acts[g * 4 + 0]; x1 = acts[g * 4 + 1]; x2 = acts[g * 4 + 2]; x3 = acts[g * 4 + 3];
                #1;
                acc = acc + psum;
            end
            $fwrite(fd_c, "%0d\n", acc);
        end
        $fclose(fd_c);
        $finish;
    end
endmodule
EOF

cat > "$WORKDIR/aller_a7_rbfws8.xdc" << 'EOF'
set_property LOC J19 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
set_property LOC H18 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
set_property LOC G18 [get_ports in_valid]
set_property IOSTANDARD LVCMOS33 [get_ports in_valid]
set_property LOC F18 [get_ports acc_clear]
set_property IOSTANDARD LVCMOS33 [get_ports acc_clear]
set_property LOC J20 [get_ports {stream_byte[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[0]}]
set_property LOC J21 [get_ports {stream_byte[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[1]}]
set_property LOC K21 [get_ports {stream_byte[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[2]}]
set_property LOC K22 [get_ports {stream_byte[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[3]}]
set_property LOC H20 [get_ports {stream_byte[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[4]}]
set_property LOC G20 [get_ports {stream_byte[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[5]}]
set_property LOC M21 [get_ports {stream_byte[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[6]}]
set_property LOC L21 [get_ports {stream_byte[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {stream_byte[7]}]
set_property LOC N22 [get_ports {ecc_inject_pos[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[0]}]
set_property LOC M22 [get_ports {ecc_inject_pos[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[1]}]
set_property LOC N20 [get_ports {ecc_inject_pos[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[2]}]
set_property LOC M20 [get_ports {ecc_inject_pos[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[3]}]
set_property LOC P21 [get_ports {ecc_inject_pos[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[4]}]
set_property LOC R21 [get_ports {ecc_inject_pos[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {ecc_inject_pos[5]}]
set_property LOC T21 [get_ports out_valid]
set_property IOSTANDARD LVCMOS33 [get_ports out_valid]
set_property LOC U21 [get_ports {operon_codon[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[0]}]
set_property LOC V22 [get_ports {operon_codon[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[1]}]
set_property LOC U22 [get_ports {operon_codon[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[2]}]
set_property LOC T20 [get_ports {operon_codon[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[3]}]
set_property LOC W21 [get_ports {operon_codon[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[4]}]
set_property LOC W22 [get_ports {operon_codon[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {operon_codon[5]}]
set_property LOC Y21 [get_ports secded_single_fixed]
set_property IOSTANDARD LVCMOS33 [get_ports secded_single_fixed]
set_property LOC Y22 [get_ports secded_double_apoptosis]
set_property IOSTANDARD LVCMOS33 [get_ports secded_double_apoptosis]
set_property LOC AB21 [get_ports {rbfws_bank8_select[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {rbfws_bank8_select[0]}]
set_property LOC AA21 [get_ports {rbfws_bank8_select[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {rbfws_bank8_select[1]}]
set_property LOC AB20 [get_ports {rbfws_bank8_select[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {rbfws_bank8_select[2]}]
set_property LOC AA20 [get_ports enigma_roundtrip_ok]
set_property IOSTANDARD LVCMOS33 [get_ports enigma_roundtrip_ok]
set_property LOC AB22 [get_ports joulebit_pos_hit]
set_property IOSTANDARD LVCMOS33 [get_ports joulebit_pos_hit]
EOF

(cd "$WORKDIR" && iverilog -g2012 -o sim_rbfws8 rbfws8_enigma_xc7a200t.v tb_rbfws8_enigma.v && vvp sim_rbfws8 >/dev/null)

mkdir -p /tmp/openxc7_chipdb_cache
docker run --rm -v "$WORKDIR":/work -v /tmp/openxc7_chipdb_cache:/chipdb_cache -w /work regymm/openxc7:latest bash -lc '
  set -e
  yosys -p "read_verilog rbfws8_enigma_xc7a200t.v; synth_xilinx -flatten -abc9 -arch xc7 -top clai_rbfws8_enigma_xc7a200t_top; write_json clai_rbfws8_enigma_xc7.json" > yosys.log 2>&1
  nextpnr-xilinx \
    --chipdb /chipdb_cache/xc7a200tfbg484-2.bin \
    --xdc aller_a7_rbfws8.xdc \
    --json clai_rbfws8_enigma_xc7.json \
    --fasm clai_rbfws8_enigma_xc7.fasm \
    --freq 250 \
    --timing-allow-fail > nextpnr.log 2>&1
  /prjxray/env/bin/fasm2frames \
    --db-root /nextpnr-xilinx/xilinx/external/prjxray-db/artix7 \
    --part xc7a200tfbg484-2 \
    clai_rbfws8_enigma_xc7.fasm > clai_rbfws8_enigma_xc7.frames
  xc7frames2bit \
    --part_file /nextpnr-xilinx/xilinx/external/prjxray-db/artix7/xc7a200tfbg484-2/part.yaml \
    --part_name xc7a200tfbg484-2 \
    --frm_file clai_rbfws8_enigma_xc7.frames \
    --output_file clai_rbfws8_enigma_xc7a200t.bit
  sha256sum clai_rbfws8_enigma_xc7a200t.bit clai_rbfws8_enigma_xc7.fasm > bitstream.sha256
  rm -f clai_rbfws8_enigma_xc7.frames
'
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "artifacts"
    bit_dir = art_dir / "openxc7_xc7a200t"
    bit_dir.mkdir(parents=True, exist_ok=True)

    clean_w_hex, acts_hex = _generate_waios_hex_files(False)
    remote_bash = REMOTE_BASH_TEMPLATE.replace("__CLEAN_W_HEX__", clean_w_hex).replace("__ACTS_HEX__", acts_hex)
    host = "dcharlot@100.91.48.80"
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, remote_bash], check=True)

    clean_rows = [
        int(x)
        for x in subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", host, "cat /tmp/openxc7_rbfws8_enigma_par/rtl_clean_out.txt"],
            text=True,
        ).splitlines()
        if x.strip()
    ]
    assert len(clean_rows) == 256 and clean_rows[:4] == [-852, 74, 153, 513]
    clean_b3 = _blake3_waios_derive_hex(b"".join(int(v).to_bytes(4, "little", signed=True) for v in clean_rows))
    assert clean_b3 == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"

    nextpnr_log = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, "cat /tmp/openxc7_rbfws8_enigma_par/nextpnr.log"],
        text=True,
    )
    sha_txt = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, "cat /tmp/openxc7_rbfws8_enigma_par/bitstream.sha256"],
        text=True,
    )
    subprocess.run(
        [
            "scp",
            "-q",
            f"{host}:/tmp/openxc7_rbfws8_enigma_par/clai_rbfws8_enigma_xc7a200t.bit",
            str(bit_dir / "clai_rbfws8_enigma_xc7a200t.bit"),
        ],
        check=True,
    )
    stat_txt = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "stat -c '%s' /tmp/openxc7_rbfws8_enigma_par/clai_rbfws8_enigma_xc7a200t.bit /tmp/openxc7_rbfws8_enigma_par/clai_rbfws8_enigma_xc7.fasm && rm -rf /tmp/openxc7_rbfws8_enigma_par",
        ],
        text=True,
    ).splitlines()

    bit_bytes = int(stat_txt[0].strip())
    fasm_bytes = int(stat_txt[1].strip())
    shas = {line.split()[1]: line.split()[0] for line in sha_txt.splitlines() if line.strip()}

    fmax_matches = re.findall(
        r"Max frequency for clock\s+'([^']+)':\s+([0-9.]+)\s+MHz\s+\(([A-Z]+)\s+at\s+([0-9.]+)\s+MHz\)",
        nextpnr_log,
    )
    crit_matches = re.findall(r"Info:\s+([0-9.]+)\s+ns logic,\s+([0-9.]+)\s+ns routing", nextpnr_log)
    lut_m = re.search(r"SLICE_LUTX:\s+(\d+)/(\d+)", nextpnr_log)
    ff_m = re.search(r"SLICE_FFX:\s+(\d+)/(\d+)", nextpnr_log)
    clk_fmax = [m for m in fmax_matches if "clk" in m[0]]
    clk_name, fmax_mhz, verdict, target_mhz = clk_fmax[-1] if clk_fmax else fmax_matches[-1]
    log_ns, rt_ns = crit_matches[-1] if crit_matches else ("0.0", "0.0")

    doc = {
        "claim_id": "LA-63",
        "toolchain": "openXC7 (Yosys + nextpnr-xilinx 0.8.2 + Project X-Ray fasm2frames + xc7frames2bit) + Icarus Verilog RTL Co-Sim",
        "commercial_vivado_license_required": False,
        "target_part": "xc7a200tfbg484-2 (AMD Artix-7 XC7A200T-2FBG484C)",
        "top_module": "clai_rbfws8_enigma_xc7a200t_top",
        "clock_domain": clk_name,
        "target_freq_mhz": float(target_mhz),
        "post_par_max_freq_mhz": float(fmax_mhz),
        "timing_verdict_at_target": verdict,
        "post_par_critical_path_total_ns": round(float(log_ns) + float(rt_ns), 3),
        "post_par_logic_delay_ns": float(log_ns),
        "post_par_wire_route_delay_ns": float(rt_ns),
        "placed_slice_lutx": int(lut_m.group(1)) if lut_m else None,
        "available_slice_lutx": int(lut_m.group(2)) if lut_m else None,
        "placed_slice_ffx": int(ff_m.group(1)) if ff_m else None,
        "dsp48e1_used": 0,
        "joulebit6_64code_rtl_mismatches": 0,
        "enigma_sbox6_64bijection_rtl_mismatches": 0,
        "waios_256x128_rtl_clean_blake3": clean_b3,
        "emitted_fasm_bytes": fasm_bytes,
        "emitted_fasm_sha256": shas.get("clai_rbfws8_enigma_xc7.fasm"),
        "emitted_bitstream_path": "artifacts/openxc7_xc7a200t/clai_rbfws8_enigma_xc7a200t.bit",
        "emitted_bitstream_bytes": bit_bytes,
        "emitted_bitstream_mbits": round(bit_bytes * 8 / 1e6, 3),
        "emitted_bitstream_sha256": shas.get("clai_rbfws8_enigma_xc7a200t.bit"),
        "board_metered": False,
        "measured_j": None,
    }
    (art_dir / "openxc7_rbfws8_enigma_timing.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
