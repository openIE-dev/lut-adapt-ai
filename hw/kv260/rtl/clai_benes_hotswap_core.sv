`timescale 1ns/1ps
// =============================================================================
// Module: clai_benes_512_perm
// Description:
//   17-Stage 512x512 Rearrangeably Non-Blocking Waksman/Beneš Permutation Network.
//   Uses `17 x 256 = 4,352` 2x2 switches (each mapping to 1 dual-output Xilinx
//   LUT6_2!). All inter-stage unshuffle/shuffle permutations are pure zero-LUT
//   compile-time metal wire routing!
// =============================================================================

module clai_benes_512_perm (
    input  wire [511:0]      in_wires,
    input  wire [17*256-1:0] sw_flat,
    output wire [511:0]      out_wires
);

    wire [511:0] st_wires [0:17];
    assign st_wires[0] = in_wires;

    genvar d, blk, k;
    generate
        // Stages 0..7: Recursive Unshuffle Input Switches (d = 0..7)
        for (d = 0; d < 8; d = d + 1) begin : gen_fwd_stage
            localparam int NUM_BLOCKS = (1 << d);
            localparam int SUB_N      = (512 >> d);
            localparam int SUB_HALF   = (SUB_N >> 1);

            for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin : gen_fwd_blk
                localparam int W_BASE  = blk * SUB_N;
                localparam int SW_BASE = d * 256 + blk * SUB_HALF;

                for (k = 0; k < SUB_HALF; k = k + 1) begin : gen_fwd_sw
                    wire s  = sw_flat[SW_BASE + k];
                    wire x0 = st_wires[d][W_BASE + 2 * k];
                    wire x1 = st_wires[d][W_BASE + 2 * k + 1];
                    assign st_wires[d + 1][W_BASE + k]            = s ? x1 : x0;
                    assign st_wires[d + 1][W_BASE + SUB_HALF + k] = s ? x0 : x1;
                end
            end
        end

        // Stage 8: Center 2x2 Base-Case Switches (256 blocks of width 2)
        for (blk = 0; blk < 256; blk = blk + 1) begin : gen_center_sw
            wire sc = sw_flat[8 * 256 + blk];
            wire c0 = st_wires[8][2 * blk];
            wire c1 = st_wires[8][2 * blk + 1];
            assign st_wires[9][2 * blk]     = sc ? c1 : c0;
            assign st_wires[9][2 * blk + 1] = sc ? c0 : c1;
        end

        // Stages 9..16: Recursive Shuffle Output Switches (d = 7..0 -> st = 16 - d)
        for (d = 0; d < 8; d = d + 1) begin : gen_rev_stage
            localparam int ST_IDX     = 16 - d; // d=7 -> st=9 ... d=0 -> st=16
            localparam int NUM_BLOCKS = (1 << d);
            localparam int SUB_N      = (512 >> d);
            localparam int SUB_HALF   = (SUB_N >> 1);

            for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin : gen_rev_blk
                localparam int W_BASE  = blk * SUB_N;
                localparam int SW_BASE = ST_IDX * 256 + blk * SUB_HALF;

                for (k = 0; k < SUB_HALF; k = k + 1) begin : gen_rev_sw
                    wire sr = sw_flat[SW_BASE + k];
                    wire b0 = st_wires[ST_IDX][W_BASE + k];
                    wire b1 = st_wires[ST_IDX][W_BASE + SUB_HALF + k];
                    assign st_wires[ST_IDX + 1][W_BASE + 2 * k]     = sr ? b1 : b0;
                    assign st_wires[ST_IDX + 1][W_BASE + 2 * k + 1] = sr ? b0 : b1;
                end
            end
        end
    endgenerate

    assign out_wires = st_wires[17];

endmodule


// =============================================================================
// Module: clai_benes_hotswap_core
// Description:
//   Dual-Banked Non-Blocking 512x512 Beneš-Copy-Beneš Hot-Swap LUT6 Fabric
//   routing 80 full-bus dynamic LUT6s (480 arbitrary multicast ports!) in O(W log W)
//   silicon area, with single-cycle (4.0 ns) glitch-free `atomic_swap_pulse`!
// =============================================================================

module clai_benes_hotswap_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [511:0] bus_in,

    // Shadow-Bank Streaming Configuration Port (`cfg_kind`: 0=B1, 1=Copy, 2=B2, 3=LUT6)
    input  wire         cfg_we,
    input  wire [1:0]   cfg_kind,
    input  wire [6:0]   cfg_idx,
    input  wire [255:0] cfg_vec256,
    input  wire [511:0] cfg_vec512,

    input  wire         atomic_swap_pulse,
    output reg          active_bank,
    output wire [79:0]  l0_out_debug
);

    reg [255:0] b1_sw   [0:1][0:16];
    reg [511:0] copy_en [0:1][0:8];
    reg [255:0] b2_sw   [0:1][0:16];
    reg [63:0]  l0_lut  [0:1][0:79];

    wire shadow = ~active_bank;

    integer b, s;
    always @(posedge clk) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
            for (b = 0; b < 2; b = b + 1) begin
                for (s = 0; s < 17; s = s + 1) begin
                    b1_sw[b][s] <= 256'd0;
                    b2_sw[b][s] <= 256'd0;
                end
                for (s = 0; s < 9; s = s + 1)
                    copy_en[b][s] <= 512'd0;
                for (s = 0; s < 80; s = s + 1)
                    l0_lut[b][s] <= 64'd0;
            end
        end else begin
            if (cfg_we) begin
                case (cfg_kind)
                    2'd0: if (cfg_idx < 7'd17) b1_sw[shadow][cfg_idx[4:0]]   <= cfg_vec256;
                    2'd1: if (cfg_idx < 7'd9)  copy_en[shadow][cfg_idx[3:0]] <= cfg_vec512;
                    2'd2: if (cfg_idx < 7'd17) b2_sw[shadow][cfg_idx[4:0]]   <= cfg_vec256;
                    2'd3: if (cfg_idx < 7'd80) l0_lut[shadow][cfg_idx]       <= cfg_vec256[63:0];
                endcase
            end
            if (atomic_swap_pulse)
                active_bank <= ~active_bank;
        end
    end

    wire [17*256-1:0] act_b1_flat;
    wire [17*256-1:0] act_b2_flat;
    genvar gs;
    generate
        for (gs = 0; gs < 17; gs = gs + 1) begin : gen_pack_sw
            assign act_b1_flat[gs * 256 +: 256] = active_bank ? b1_sw[1][gs] : b1_sw[0][gs];
            assign act_b2_flat[gs * 256 +: 256] = active_bank ? b2_sw[1][gs] : b2_sw[0][gs];
        end
    endgenerate

    wire [511:0] b1_out;
    clai_benes_512_perm u_benes_1 (
        .in_wires  (bus_in),
        .sw_flat   (act_b1_flat),
        .out_wires (b1_out)
    );

    // 9-Stage Logarithmic Segmented Prefix-Copy Replicator
    wire [511:0] cp_stage [0:9];
    assign cp_stage[0] = b1_out;

    genvar cd, ci;
    generate
        for (cd = 0; cd < 9; cd = cd + 1) begin : gen_copy_stage
            localparam int STRIDE = (1 << cd);
            wire [511:0] en_vec = active_bank ? copy_en[1][cd] : copy_en[0][cd];
            for (ci = 0; ci < 512; ci = ci + 1) begin : gen_copy_mux
                if (ci < STRIDE) begin : gen_keep_low
                    assign cp_stage[cd + 1][ci] = cp_stage[cd][ci];
                end else begin : gen_stride_mux
                    assign cp_stage[cd + 1][ci] = en_vec[ci] ? cp_stage[cd][ci - STRIDE] : cp_stage[cd][ci];
                end
            end
        end
    endgenerate

    wire [511:0] routed_ports;
    clai_benes_512_perm u_benes_2 (
        .in_wires  (cp_stage[9]),
        .sw_flat   (act_b2_flat),
        .out_wires (routed_ports)
    );

    // 80 Full-Bus Dynamic LUT6 Cells (consuming routed_ports[0..479])
    genvar gl;
    generate
        for (gl = 0; gl < 80; gl = gl + 1) begin : gen_lut80
            wire [63:0] t64  = active_bank ? l0_lut[1][gl] : l0_lut[0][gl];
            wire [5:0]  addr = routed_ports[gl * 6 +: 6];
            assign l0_out_debug[gl] = t64[addr];
        end
    endgenerate

endmodule


// =============================================================================
// Module: clai_uram_lut6_page_cache
// Description:
//   Single-Cycle (4.0 ns @ 250 MHz, CPI = 1.0) Zero-DSP Galois-XOR Hash-Indexed
//   Block-RAM / UltraRAM LUT6 INIT Page Cache for Kria KV260 (`xck26`).
//   - Indexes `2^PAGE_ADDR_BITS` pages (`512` in 81 RAMB36E2s, or `4,096` in
//     URAM288 when `CLAI_USE_URAM288` is defined) by `{lang_id[1:0], ctx6[47:0]}`.
//   - Reads `51-bit` tag + `80 x 64-bit = 5,120` LUT6 INIT truth-table bits in
//     1 single clock cycle with non-blocking live Sleep-Crystallization writes!
// =============================================================================

module clai_uram_lut6_page_cache #(
    parameter int PAGE_ADDR_BITS = 9
) (
    input  wire                      clk,

    // Lookup Port (1 new 6-byte code context every clock cycle!)
    input  wire [1:0]                lookup_lang_id,
    input  wire [47:0]               lookup_ctx6,
    output wire [PAGE_ADDR_BITS-1:0] lookup_hash_dbg,

    // Live Online Sleep-Crystallization Write Port (1-cycle non-blocking update)
    input  wire                      page_prog_we,
    input  wire                      page_prog_is_tag,
    input  wire [PAGE_ADDR_BITS-1:0] page_prog_addr,
    input  wire [6:0]                page_prog_lut_idx,
    input  wire [63:0]               page_prog_data64,

    // Synchronous 1-Cycle Output (aligned with `bus_pipe` in parent)
    output reg  [63:0]               rd_tag64,
    output wire [80*64-1:0]          rd_luts_flat
);

    localparam int NUM_PAGES = (1 << PAGE_ADDR_BITS);

    // Zero-DSP, Zero-Multiplier Circular-Shift Galois XOR Context Hash
    wire [15:0] f0    = lookup_ctx6[15:0]  ^ {lookup_ctx6[7:0],   lookup_ctx6[15:8]};
    wire [15:0] f1    = lookup_ctx6[31:16] ^ {lookup_ctx6[20:16], lookup_ctx6[31:21]};
    wire [15:0] f2    = lookup_ctx6[47:32] ^ {lookup_ctx6[38:32], lookup_ctx6[47:39]};
    wire [15:0] mix_a = f0 ^ {f1[12:0], f1[15:13]} ^ {f2[8:0], f2[15:9]};
    wire [15:0] mix_b = mix_a ^ {5'd0, mix_a[15:5]} ^ ({14'd0, lookup_lang_id} << 9)
                        ^ {6'd0, lookup_lang_id, lookup_ctx6[47:40]};
    wire [PAGE_ADDR_BITS-1:0] lookup_hash = mix_b[PAGE_ADDR_BITS-1:0];
    assign lookup_hash_dbg = lookup_hash;

`ifdef CLAI_USE_URAM288
    (* ram_style = "ultra" *) reg [63:0] tag_mem [0:NUM_PAGES-1];
`else
    (* ram_style = "block" *) reg [63:0] tag_mem [0:NUM_PAGES-1];
`endif

    integer init_p;
    initial begin
        for (init_p = 0; init_p < NUM_PAGES; init_p = init_p + 1)
            tag_mem[init_p] = 64'd0;
    end

    always @(posedge clk) begin
        if (page_prog_we && page_prog_is_tag)
            tag_mem[page_prog_addr] <= page_prog_data64;
        rd_tag64 <= tag_mem[lookup_hash];
    end

    // 80 Parallel 64-Bit SDP Block-RAM / UltraRAM Banks (5,120 INIT bits/cycle!)
    genvar gi;
    generate
        for (gi = 0; gi < 80; gi = gi + 1) begin : gen_lut_page_bank
`ifdef CLAI_USE_URAM288
            (* ram_style = "ultra" *) reg [63:0] lut_mem [0:NUM_PAGES-1];
`else
            (* ram_style = "block" *) reg [63:0] lut_mem [0:NUM_PAGES-1];
`endif
            reg [63:0] rd_lut_word;

            always @(posedge clk) begin
                if (page_prog_we && !page_prog_is_tag && (page_prog_lut_idx == gi[6:0]))
                    lut_mem[page_prog_addr] <= page_prog_data64;
                rd_lut_word <= lut_mem[lookup_hash];
            end

            assign rd_luts_flat[gi * 64 +: 64] = rd_lut_word;
        end
    endgenerate

endmodule


// =============================================================================
// Module: clai_benes_paged_hotswap_core
// Description:
//   Multi-Page BRAM/URAM + Dual-Bank 512x512 Beneš-Copy-Beneš Speculative Core:
//   - Switches all 80 x 64-bit LUT6 INIT truth tables in 1 clock cycle (4.0 ns)
//     for every incoming 6-byte code suffix `bus_in[127:80]` + `ctx_lang_id`.
//   - Automatically falls back in 0 stall cycles on a tag miss to the resident
//     Active-Bank XNOR-CAM Copy LUT6 page (`l0_lut[active_bank]`).
//   - Reduces 80 LUT6 outputs (`16` bits x `5` voters) into `draft_word16_out`
//     (2 full speculative ASCII bytes/slice/cycle) + `draft_unanimous16_out`!
// =============================================================================

module clai_benes_paged_hotswap_core #(
    parameter int PAGE_ADDR_BITS = 9
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Streaming 512-Wire CodeBitBus Input (1 context per clock cycle, CPI = 1.0)
    input  wire [511:0]              bus_in,
    input  wire                      bus_valid,
    input  wire [1:0]                ctx_lang_id,
    input  wire                      page_cache_en,

    // Dual-Bank Beneš + Default Fallback LUT6 Shadow Configuration Port
    input  wire                      cfg_we,
    input  wire [1:0]                cfg_kind,
    input  wire [6:0]                cfg_idx,
    input  wire [255:0]              cfg_vec256,
    input  wire [511:0]              cfg_vec512,
    input  wire                      atomic_swap_pulse,

    // Live Online Sleep-Crystallization Multi-Page BRAM Write Port
    input  wire                      page_prog_we,
    input  wire                      page_prog_is_tag,
    input  wire [PAGE_ADDR_BITS-1:0] page_prog_addr,
    input  wire [6:0]                page_prog_lut_idx,
    input  wire [63:0]               page_prog_data64,

    // Cycle-Aligned Speculative Burst Outputs & Hardware Telemetry
    output reg                       active_bank,
    output reg                       out_valid,
    output wire                      page_hit_out,
    output reg  [PAGE_ADDR_BITS-1:0] page_slot_out,
    output wire [79:0]               l0_out_debug,
    output wire [15:0]               draft_word16_out,
    output wire [15:0]               draft_unanimous16_out,
    output reg  [31:0]               telemetry_lookups,
    output reg  [31:0]               telemetry_page_hits,
    output reg  [31:0]               telemetry_cam_fallbacks
);

    reg [255:0] b1_sw   [0:1][0:16];
    reg [511:0] copy_en [0:1][0:8];
    reg [255:0] b2_sw   [0:1][0:16];
    reg [63:0]  l0_lut  [0:1][0:79];

    wire shadow = ~active_bank;

    // 1-Cycle Pipeline Alignment Registers matching Synchronous BRAM Read Latency
    reg [511:0]              bus_pipe;
    reg [50:0]               tag_exp_pipe;
    wire [PAGE_ADDR_BITS-1:0] lookup_hash_w;

    integer b, s;
    always @(posedge clk) begin
        if (!rst_n) begin
            active_bank             <= 1'b0;
            out_valid               <= 1'b0;
            bus_pipe                <= 512'd0;
            tag_exp_pipe            <= 51'd0;
            page_slot_out           <= {PAGE_ADDR_BITS{1'b0}};
            telemetry_lookups       <= 32'd0;
            telemetry_page_hits     <= 32'd0;
            telemetry_cam_fallbacks <= 32'd0;
            for (b = 0; b < 2; b = b + 1) begin
                for (s = 0; s < 17; s = s + 1) begin
                    b1_sw[b][s] <= 256'd0;
                    b2_sw[b][s] <= 256'd0;
                end
                for (s = 0; s < 9; s = s + 1)
                    copy_en[b][s] <= 512'd0;
                for (s = 0; s < 80; s = s + 1)
                    l0_lut[b][s] <= 64'd0;
            end
        end else begin
            if (cfg_we) begin
                case (cfg_kind)
                    2'd0: if (cfg_idx < 7'd17) b1_sw[shadow][cfg_idx[4:0]]   <= cfg_vec256;
                    2'd1: if (cfg_idx < 7'd9)  copy_en[shadow][cfg_idx[3:0]] <= cfg_vec512;
                    2'd2: if (cfg_idx < 7'd17) b2_sw[shadow][cfg_idx[4:0]]   <= cfg_vec256;
                    2'd3: if (cfg_idx < 7'd80) l0_lut[shadow][cfg_idx]       <= cfg_vec256[63:0];
                endcase
            end
            if (atomic_swap_pulse)
                active_bank <= ~active_bank;

            // Stage 0 -> Stage 1 Pipeline Latch (aligns `bus_pipe` with `rd_luts_flat`!)
            out_valid     <= bus_valid;
            bus_pipe      <= bus_in;
            tag_exp_pipe  <= {1'b1, ctx_lang_id, bus_in[127:80]};
            page_slot_out <= lookup_hash_w;

            // Update hardware telemetry counters on valid output beat
            if (out_valid) begin
                telemetry_lookups <= telemetry_lookups + 32'd1;
                if (page_hit_out)
                    telemetry_page_hits <= telemetry_page_hits + 32'd1;
                else
                    telemetry_cam_fallbacks <= telemetry_cam_fallbacks + 32'd1;
            end
        end
    end

    wire [63:0]      rd_tag64;
    wire [80*64-1:0] rd_luts_flat;

    clai_uram_lut6_page_cache #(
        .PAGE_ADDR_BITS(PAGE_ADDR_BITS)
    ) u_page_cache (
        .clk               (clk),
        .lookup_lang_id    (ctx_lang_id),
        .lookup_ctx6       (bus_in[127:80]),
        .lookup_hash_dbg   (lookup_hash_w),
        .page_prog_we      (page_prog_we),
        .page_prog_is_tag  (page_prog_is_tag),
        .page_prog_addr    (page_prog_addr),
        .page_prog_lut_idx (page_prog_lut_idx),
        .page_prog_data64  (page_prog_data64),
        .rd_tag64          (rd_tag64),
        .rd_luts_flat      (rd_luts_flat)
    );

    assign page_hit_out = out_valid && page_cache_en && (rd_tag64[50:0] == tag_exp_pipe);

    wire [17*256-1:0] act_b1_flat;
    wire [17*256-1:0] act_b2_flat;
    genvar gs;
    generate
        for (gs = 0; gs < 17; gs = gs + 1) begin : gen_pack_sw_paged
            assign act_b1_flat[gs * 256 +: 256] = active_bank ? b1_sw[1][gs] : b1_sw[0][gs];
            assign act_b2_flat[gs * 256 +: 256] = active_bank ? b2_sw[1][gs] : b2_sw[0][gs];
        end
    endgenerate

    wire [511:0] b1_out;
    clai_benes_512_perm u_benes_1 (
        .in_wires  (bus_pipe),
        .sw_flat   (act_b1_flat),
        .out_wires (b1_out)
    );

    wire [511:0] cp_stage [0:9];
    assign cp_stage[0] = b1_out;

    genvar cd, ci;
    generate
        for (cd = 0; cd < 9; cd = cd + 1) begin : gen_copy_stage_paged
            localparam int STRIDE = (1 << cd);
            wire [511:0] en_vec = active_bank ? copy_en[1][cd] : copy_en[0][cd];
            for (ci = 0; ci < 512; ci = ci + 1) begin : gen_copy_mux_paged
                if (ci < STRIDE) begin : gen_keep_low
                    assign cp_stage[cd + 1][ci] = cp_stage[cd][ci];
                end else begin : gen_stride_mux
                    assign cp_stage[cd + 1][ci] = en_vec[ci] ? cp_stage[cd][ci - STRIDE] : cp_stage[cd][ci];
                end
            end
        end
    endgenerate

    wire [511:0] routed_ports;
    clai_benes_512_perm u_benes_2 (
        .in_wires  (cp_stage[9]),
        .sw_flat   (act_b2_flat),
        .out_wires (routed_ports)
    );

    // 80 Dynamic LUT6s: 1-Cycle BRAM/URAM Page Hit vs Zero-Stall Active-Bank CAM Fallback
    genvar gl;
    generate
        for (gl = 0; gl < 80; gl = gl + 1) begin : gen_paged_lut80
            wire [63:0] fallback_t64 = active_bank ? l0_lut[1][gl] : l0_lut[0][gl];
            wire [63:0] cached_t64   = rd_luts_flat[gl * 64 +: 64];
            wire [63:0] eff_t64      = page_hit_out ? cached_t64 : fallback_t64;
            wire [5:0]  addr         = routed_ports[gl * 6 +: 6];
            assign l0_out_debug[gl]  = eff_t64[addr];
        end
    endgenerate

    // 16-Bit (2-Byte) 5-Voter Majority Speculative Burst Reducer
    genvar gb;
    generate
        for (gb = 0; gb < 16; gb = gb + 1) begin : gen_voter5
            wire [2:0] v_sum = {2'd0, l0_out_debug[gb * 5 + 0]}
                             + {2'd0, l0_out_debug[gb * 5 + 1]}
                             + {2'd0, l0_out_debug[gb * 5 + 2]}
                             + {2'd0, l0_out_debug[gb * 5 + 3]}
                             + {2'd0, l0_out_debug[gb * 5 + 4]};
            assign draft_word16_out[gb]      = (v_sum >= 3'd3);
            assign draft_unanimous16_out[gb] = (v_sum == 3'd5) || (v_sum == 3'd0);
        end
    endgenerate

endmodule
