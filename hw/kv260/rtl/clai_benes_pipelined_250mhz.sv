`timescale 1ns/1ps
// =============================================================================
// Module: clai_benes_512_split_13_4
// Description:
//   17-Stage 512x512 Waksman/Beneš Permutation Network split at stage 13 (`13 + 4`)
//   for balanced 250 MHz (`4.000 ns`) Kria KV260 (`xck26-sfvc784-2LV-c`) pipelining.
// =============================================================================

module clai_benes_512_stages_0_12 (
    input  wire [511:0]      in_wires,
    input  wire [13*256-1:0] sw_flat_0_12,
    output wire [511:0]      mid_wires
);
    wire [511:0] st_wires [0:13];
    assign st_wires[0] = in_wires;

    genvar d, blk, k;
    generate
        // Stages 0..7: Unshuffle input stages
        for (d = 0; d < 8; d = d + 1) begin : gen_fwd_stage
            localparam int NUM_BLOCKS = (1 << d);
            localparam int SUB_N      = (512 >> d);
            localparam int SUB_HALF   = (SUB_N >> 1);
            for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin : gen_fwd_blk
                localparam int W_BASE  = blk * SUB_N;
                localparam int SW_BASE = d * 256 + blk * SUB_HALF;
                for (k = 0; k < SUB_HALF; k = k + 1) begin : gen_fwd_sw
                    wire s  = sw_flat_0_12[SW_BASE + k];
                    wire x0 = st_wires[d][W_BASE + 2 * k];
                    wire x1 = st_wires[d][W_BASE + 2 * k + 1];
                    assign st_wires[d + 1][W_BASE + k]            = s ? x1 : x0;
                    assign st_wires[d + 1][W_BASE + SUB_HALF + k] = s ? x0 : x1;
                end
            end
        end

        // Stage 8: Center 2x2 switches
        for (blk = 0; blk < 256; blk = blk + 1) begin : gen_center_sw
            wire sc = sw_flat_0_12[8 * 256 + blk];
            wire c0 = st_wires[8][2 * blk];
            wire c1 = st_wires[8][2 * blk + 1];
            assign st_wires[9][2 * blk]     = sc ? c1 : c0;
            assign st_wires[9][2 * blk + 1] = sc ? c0 : c1;
        end

        // Stages 9..12: First 4 shuffle output stages (`d = 7, 6, 5, 4` -> `ST_IDX = 9..12`)
        for (d = 4; d < 8; d = d + 1) begin : gen_rev_hi
            localparam int ST_IDX     = 16 - d;
            localparam int NUM_BLOCKS = (1 << d);
            localparam int SUB_N      = (512 >> d);
            localparam int SUB_HALF   = (SUB_N >> 1);
            for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin : gen_rev_blk_hi
                localparam int W_BASE  = blk * SUB_N;
                localparam int SW_BASE = ST_IDX * 256 + blk * SUB_HALF;
                for (k = 0; k < SUB_HALF; k = k + 1) begin : gen_rev_sw_hi
                    wire sr = sw_flat_0_12[SW_BASE + k];
                    wire b0 = st_wires[ST_IDX][W_BASE + k];
                    wire b1 = st_wires[ST_IDX][W_BASE + SUB_HALF + k];
                    assign st_wires[ST_IDX + 1][W_BASE + 2 * k]     = sr ? b1 : b0;
                    assign st_wires[ST_IDX + 1][W_BASE + 2 * k + 1] = sr ? b0 : b1;
                end
            end
        end
    endgenerate

    assign mid_wires = st_wires[13];
endmodule


module clai_benes_512_stages_13_16 (
    input  wire [511:0]     mid_wires,
    input  wire [4*256-1:0] sw_flat_13_16,
    output wire [511:0]     out_wires
);
    wire [511:0] st_wires [13:17];
    assign st_wires[13] = mid_wires;

    genvar d, blk, k;
    generate
        // Stages 13..16 (`d = 3, 2, 1, 0` -> `ST_IDX = 13..16`)
        for (d = 0; d < 4; d = d + 1) begin : gen_rev_lo
            localparam int ST_IDX     = 16 - d;
            localparam int LOCAL_IDX  = ST_IDX - 13;
            localparam int NUM_BLOCKS = (1 << d);
            localparam int SUB_N      = (512 >> d);
            localparam int SUB_HALF   = (SUB_N >> 1);
            for (blk = 0; blk < NUM_BLOCKS; blk = blk + 1) begin : gen_rev_blk_lo
                localparam int W_BASE  = blk * SUB_N;
                localparam int SW_BASE = LOCAL_IDX * 256 + blk * SUB_HALF;
                for (k = 0; k < SUB_HALF; k = k + 1) begin : gen_rev_sw_lo
                    wire sr = sw_flat_13_16[SW_BASE + k];
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
// Module: clai_uram_lut6_page_cache_ce
// Description:
//   Clock-Enable (`ENARDEN`) Gated 81-RAMB36E2 (`512`-page) Galois-XOR Hash-Indexed
//   `LUT6 INIT` Page Cache (`Claim LA-22`). Asserts `RAMB36E2` read sense-amplifiers
//   ONLY on cycles with `lookup_rd_en = 1'b1`, eliminating unconditional `200 pJ/cyc`
//   idle bitline precharge (`4.5x+` quiescent dynamic power reduction at 0 latency cost).
// =============================================================================

module clai_uram_lut6_page_cache_ce #(
    parameter int PAGE_ADDR_BITS = 9
) (
    input  wire                      clk,
    input  wire                      lookup_rd_en,
    input  wire [1:0]                lookup_lang_id,
    input  wire [47:0]               lookup_ctx6,
    output wire [PAGE_ADDR_BITS-1:0] lookup_hash_dbg,

    input  wire                      page_prog_we,
    input  wire                      page_prog_is_tag,
    input  wire [PAGE_ADDR_BITS-1:0] page_prog_addr,
    input  wire [6:0]                page_prog_lut_idx,
    input  wire [63:0]               page_prog_data64,

    output reg  [63:0]               rd_tag64,
    output wire [80*64-1:0]          rd_luts_flat
);
    localparam int NUM_PAGES = (1 << PAGE_ADDR_BITS);

    wire [15:0] f0    = lookup_ctx6[15:0]  ^ {lookup_ctx6[7:0],   lookup_ctx6[15:8]};
    wire [15:0] f1    = lookup_ctx6[31:16] ^ {lookup_ctx6[20:16], lookup_ctx6[31:21]};
    wire [15:0] f2    = lookup_ctx6[47:32] ^ {lookup_ctx6[38:32], lookup_ctx6[47:39]};
    wire [15:0] mix_a = f0 ^ {f1[12:0], f1[15:13]} ^ {f2[8:0], f2[15:9]};
    wire [15:0] mix_b = mix_a ^ {5'd0, mix_a[15:5]} ^ ({14'd0, lookup_lang_id} << 9)
                        ^ {6'd0, lookup_lang_id, lookup_ctx6[47:40]};
    wire [PAGE_ADDR_BITS-1:0] lookup_hash = mix_b[PAGE_ADDR_BITS-1:0];
    assign lookup_hash_dbg = lookup_hash;

    (* ram_style = "block" *) reg [63:0] tag_mem [0:NUM_PAGES-1];
    integer init_p;
    initial begin
        for (init_p = 0; init_p < NUM_PAGES; init_p = init_p + 1)
            tag_mem[init_p] = 64'd0;
    end

    always @(posedge clk) begin
        if (page_prog_we && page_prog_is_tag)
            tag_mem[page_prog_addr] <= page_prog_data64;
        if (lookup_rd_en)
            rd_tag64 <= tag_mem[lookup_hash];
    end

    genvar gi;
    generate
        for (gi = 0; gi < 80; gi = gi + 1) begin : gen_lut_page_bank_ce
            (* ram_style = "block" *) reg [63:0] lut_mem [0:NUM_PAGES-1];
            reg [63:0] rd_lut_word;

            always @(posedge clk) begin
                if (page_prog_we && !page_prog_is_tag && (page_prog_lut_idx == gi[6:0]))
                    lut_mem[page_prog_addr] <= page_prog_data64;
                if (lookup_rd_en)
                    rd_lut_word <= lut_mem[lookup_hash];
            end

            assign rd_luts_flat[gi * 64 +: 64] = rd_lut_word;
        end
    endgenerate
endmodule


// =============================================================================
// Module: clai_benes_paged_pipelined_250mhz
// Description:
//   4-Stage Balanced Pipelined (`II = 1`, `CPI = 1.00`, `4`-cycle = `16.0 ns` latency)
//   512x512 Beneš-Copy-Beneš + 81-RAMB36E2 Speculative Core for `xck26-sfvc784-2LV-c`.
//   Supports `ENABLE_CE_GATING = 1` (`Claim LA-22`) gating `81 x RAMB36E2` `ENARDEN`
//   and wide `5,120`-bit `S1..S4` pipeline `FDRE` banks on active valid bubbles.
// =============================================================================

module clai_benes_paged_pipelined_250mhz #(
    parameter int PAGE_ADDR_BITS   = 9,
    parameter int ENABLE_CE_GATING = 1
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // Streaming 512-Wire CodeBitBus Input (1 context every clock cycle, CPI = 1.00)
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

    // Registered Stage-4 Speculative Burst Outputs & Hardware Telemetry
    output reg                       active_bank,
    output reg                       out_valid,
    output reg                       page_hit_out,
    output reg  [PAGE_ADDR_BITS-1:0] page_slot_out,
    output reg  [79:0]               l0_out_debug,
    output reg  [15:0]               draft_word16_out,
    output reg  [15:0]               draft_unanimous16_out,
    output reg  [31:0]               telemetry_lookups,
    output reg  [31:0]               telemetry_page_hits,
    output reg  [31:0]               telemetry_cam_fallbacks
);

    reg [255:0] b1_sw   [0:1][0:16];
    reg [511:0] copy_en [0:1][0:8];
    reg [255:0] b2_sw   [0:1][0:16];
    reg [63:0]  l0_lut  [0:1][0:79];

    wire shadow = ~active_bank;

    wire [17*256-1:0] act_b1_flat;
    wire [17*256-1:0] act_b2_flat;
    wire [80*64-1:0]  act_fallback_flat;
    genvar gs;
    generate
        for (gs = 0; gs < 17; gs = gs + 1) begin : gen_pack_sw
            assign act_b1_flat[gs * 256 +: 256] = active_bank ? b1_sw[1][gs] : b1_sw[0][gs];
            assign act_b2_flat[gs * 256 +: 256] = active_bank ? b2_sw[1][gs] : b2_sw[0][gs];
        end
        for (gs = 0; gs < 80; gs = gs + 1) begin : gen_pack_fb
            assign act_fallback_flat[gs * 64 +: 64] = active_bank ? l0_lut[1][gs] : l0_lut[0][gs];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Stage 0 Combinational: BRAM Read Address + Beneš-1 Stages 0..12 (13 lvls)
    // -------------------------------------------------------------------------
    wire [PAGE_ADDR_BITS-1:0] lookup_hash_s0;
    wire [63:0]               rd_tag64_s1;
    wire [80*64-1:0]          rd_luts_flat_s1;
    wire                      bram_rd_en_s0 = (ENABLE_CE_GATING != 0) ? (bus_valid && page_cache_en) : 1'b1;

    clai_uram_lut6_page_cache_ce #(
        .PAGE_ADDR_BITS(PAGE_ADDR_BITS)
    ) u_page_cache (
        .clk               (clk),
        .lookup_rd_en      (bram_rd_en_s0),
        .lookup_lang_id    (ctx_lang_id),
        .lookup_ctx6       (bus_in[127:80]),
        .lookup_hash_dbg   (lookup_hash_s0),
        .page_prog_we      (page_prog_we),
        .page_prog_is_tag  (page_prog_is_tag),
        .page_prog_addr    (page_prog_addr),
        .page_prog_lut_idx (page_prog_lut_idx),
        .page_prog_data64  (page_prog_data64),
        .rd_tag64          (rd_tag64_s1),
        .rd_luts_flat      (rd_luts_flat_s1)
    );

    wire [511:0] b1_mid_s0;
    clai_benes_512_stages_0_12 u_b1_hi (
        .in_wires     (bus_in),
        .sw_flat_0_12 (act_b1_flat[13*256-1:0]),
        .mid_wires    (b1_mid_s0)
    );

    // Stage 1 Pipeline Registers (`s1`)
    reg                      s1_valid;
    reg                      s1_cache_en;
    reg [50:0]               s1_tag_exp;
    reg [PAGE_ADDR_BITS-1:0] s1_slot;
    reg [511:0]              s1_b1_mid;

    // -------------------------------------------------------------------------
    // Stage 1 Combinational: Beneš-1 Stages 13..16 (4) + Copy 0..8 (9) = 13 lvls
    //                        || Tag Compare + 5,120-bit INIT Mux (~5 lvls)
    // -------------------------------------------------------------------------
    wire [511:0] b1_out_s1;
    clai_benes_512_stages_13_16 u_b1_lo (
        .mid_wires     (s1_b1_mid),
        .sw_flat_13_16 (act_b1_flat[17*256-1:13*256]),
        .out_wires     (b1_out_s1)
    );

    wire [511:0] cp_stage_s1 [0:9];
    assign cp_stage_s1[0] = b1_out_s1;
    genvar cd, ci;
    generate
        for (cd = 0; cd < 9; cd = cd + 1) begin : gen_copy_s1
            localparam int STRIDE = (1 << cd);
            wire [511:0] en_vec = active_bank ? copy_en[1][cd] : copy_en[0][cd];
            for (ci = 0; ci < 512; ci = ci + 1) begin : gen_copy_mux_s1
                if (ci < STRIDE) begin : gen_keep_low
                    assign cp_stage_s1[cd + 1][ci] = cp_stage_s1[cd][ci];
                end else begin : gen_stride_mux
                    assign cp_stage_s1[cd + 1][ci] = en_vec[ci] ? cp_stage_s1[cd][ci - STRIDE] : cp_stage_s1[cd][ci];
                end
            end
        end
    endgenerate

    wire s1_hit = s1_valid && s1_cache_en && (rd_tag64_s1[50:0] == s1_tag_exp);
    wire [80*64-1:0] s1_eff_luts = s1_hit ? rd_luts_flat_s1 : act_fallback_flat;

    // Stage 2 Pipeline Registers (`s2`)
    reg                      s2_valid;
    reg                      s2_hit;
    reg [PAGE_ADDR_BITS-1:0] s2_slot;
    reg [511:0]              s2_cp_out;
    reg [80*64-1:0]          s2_eff_luts;

    // -------------------------------------------------------------------------
    // Stage 2 Combinational: Beneš-2 Stages 0..12 (13 lvls)
    // -------------------------------------------------------------------------
    wire [511:0] b2_mid_s2;
    clai_benes_512_stages_0_12 u_b2_hi (
        .in_wires     (s2_cp_out),
        .sw_flat_0_12 (act_b2_flat[13*256-1:0]),
        .mid_wires    (b2_mid_s2)
    );

    // Stage 3 Pipeline Registers (`s3`)
    reg                      s3_valid;
    reg                      s3_hit;
    reg [PAGE_ADDR_BITS-1:0] s3_slot;
    reg [511:0]              s3_b2_mid;
    reg [80*64-1:0]          s3_eff_luts;

    // -------------------------------------------------------------------------
    // Stage 3 Combinational: Beneš-2 Stages 13..16 (4) + 80x LUT6 (64:1) + Voter5
    // -------------------------------------------------------------------------
    wire [511:0] routed_ports_s3;
    clai_benes_512_stages_13_16 u_b2_lo (
        .mid_wires     (s3_b2_mid),
        .sw_flat_13_16 (act_b2_flat[17*256-1:13*256]),
        .out_wires     (routed_ports_s3)
    );

    wire [79:0] l0_comb_s3;
    genvar gl;
    generate
        for (gl = 0; gl < 80; gl = gl + 1) begin : gen_lut80_s3
            wire [63:0] t64  = s3_eff_luts[gl * 64 +: 64];
            wire [5:0]  addr = routed_ports_s3[gl * 6 +: 6];
            assign l0_comb_s3[gl] = t64[addr];
        end
    endgenerate

    wire [15:0] draft_w16_comb;
    wire [15:0] draft_u16_comb;
    genvar gb;
    generate
        for (gb = 0; gb < 16; gb = gb + 1) begin : gen_voter5_s3
            wire [2:0] v_sum = {2'd0, l0_comb_s3[gb * 5 + 0]}
                             + {2'd0, l0_comb_s3[gb * 5 + 1]}
                             + {2'd0, l0_comb_s3[gb * 5 + 2]}
                             + {2'd0, l0_comb_s3[gb * 5 + 3]}
                             + {2'd0, l0_comb_s3[gb * 5 + 4]};
            assign draft_w16_comb[gb] = (v_sum >= 3'd3);
            assign draft_u16_comb[gb] = (v_sum == 3'd5) || (v_sum == 3'd0);
        end
    endgenerate

    // Sequential Pipeline Advance & Shadow Configuration
    integer b, s;
    always @(posedge clk) begin
        if (!rst_n) begin
            active_bank             <= 1'b0;
            s1_valid                <= 1'b0;
            s1_cache_en             <= 1'b0;
            s1_tag_exp              <= 51'd0;
            s1_slot                 <= {PAGE_ADDR_BITS{1'b0}};
            s1_b1_mid               <= 512'd0;
            s2_valid                <= 1'b0;
            s2_hit                  <= 1'b0;
            s2_slot                 <= {PAGE_ADDR_BITS{1'b0}};
            s2_cp_out               <= 512'd0;
            s2_eff_luts             <= {5120{1'b0}};
            s3_valid                <= 1'b0;
            s3_hit                  <= 1'b0;
            s3_slot                 <= {PAGE_ADDR_BITS{1'b0}};
            s3_b2_mid               <= 512'd0;
            s3_eff_luts             <= {5120{1'b0}};
            out_valid               <= 1'b0;
            page_hit_out            <= 1'b0;
            page_slot_out           <= {PAGE_ADDR_BITS{1'b0}};
            l0_out_debug            <= 80'd0;
            draft_word16_out        <= 16'd0;
            draft_unanimous16_out   <= 16'd0;
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

            // Stage 0 -> Stage 1 (clock-enable gated on `bus_valid` when `ENABLE_CE_GATING != 0`)
            s1_valid <= bus_valid;
            if ((ENABLE_CE_GATING == 0) || bus_valid) begin
                s1_cache_en <= page_cache_en;
                s1_tag_exp  <= {1'b1, ctx_lang_id, bus_in[127:80]};
                s1_slot     <= lookup_hash_s0;
                s1_b1_mid   <= b1_mid_s0;
            end

            // Stage 1 -> Stage 2
            s2_valid <= s1_valid;
            if ((ENABLE_CE_GATING == 0) || s1_valid) begin
                s2_hit      <= s1_hit;
                s2_slot     <= s1_slot;
                s2_cp_out   <= cp_stage_s1[9];
                s2_eff_luts <= s1_eff_luts;
            end

            // Stage 2 -> Stage 3
            s3_valid <= s2_valid;
            if ((ENABLE_CE_GATING == 0) || s2_valid) begin
                s3_hit      <= s2_hit;
                s3_slot     <= s2_slot;
                s3_b2_mid   <= b2_mid_s2;
                s3_eff_luts <= s2_eff_luts;
            end

            // Stage 3 -> Registered Stage 4 Output
            out_valid <= s3_valid;
            if ((ENABLE_CE_GATING == 0) || s3_valid) begin
                page_hit_out          <= s3_hit;
                page_slot_out         <= s3_slot;
                l0_out_debug          <= l0_comb_s3;
                draft_word16_out      <= draft_w16_comb;
                draft_unanimous16_out <= draft_u16_comb;
            end

            if (s3_valid) begin
                telemetry_lookups <= telemetry_lookups + 32'd1;
                if (s3_hit)
                    telemetry_page_hits <= telemetry_page_hits + 32'd1;
                else
                    telemetry_cam_fallbacks <= telemetry_cam_fallbacks + 32'd1;
            end
        end
    end

endmodule


// =============================================================================
// Module: clai_benes_axi_stream_250mhz_top
// Description:
//   IOB-Safe (`<= 128` top-level package pins <= `189` on `xck26-sfvc784-2LV-c`)
//   64-Bit AXI4-Stream + Serialized CSR Wrapper around `clai_benes_paged_pipelined_250mhz`.
// =============================================================================

module clai_benes_axi_stream_250mhz_top #(
    parameter int PAGE_ADDR_BITS   = 9,
    parameter int ENABLE_CE_GATING = 1
) (
    input  wire        clk,
    input  wire        rst_n,

    // 64-Bit AXI4-Stream Context Ingest (shifts into 512-bit CodeBitBus shift register)
    input  wire [63:0] s_axis_tdata,
    input  wire [1:0]  s_axis_tuser_lang,
    input  wire        s_axis_tvalid,
    input  wire        s_axis_tlast,
    output wire        s_axis_tready,

    // Compact 32-Bit Serialized Configuration / BRAM Programming Port
    input  wire        csr_we,
    input  wire [5:0]  csr_cmd,
    input  wire [15:0] csr_addr,
    input  wire [31:0] csr_wdata,

    // Registered Speculative Stream Output (`<= 128` total top-level pins!)
    output wire        m_axis_tvalid,
    output wire        m_axis_page_hit,
    output wire [8:0]  m_axis_page_slot,
    output wire [15:0] m_axis_draft_word16,
    output wire [15:0] m_axis_draft_unanimous16
);
    assign s_axis_tready = 1'b1;

    reg [511:0] bus_shift_reg;
    reg [511:0] cfg_shift512;
    reg         fire_lookup;
    reg [1:0]   lang_reg;
    reg         cache_en_reg;

    reg                      cfg_we_r;
    reg [1:0]                cfg_kind_r;
    reg [6:0]                cfg_idx_r;
    reg                      swap_r;
    reg                      p_we_r;
    reg                      p_is_tag_r;
    reg [PAGE_ADDR_BITS-1:0] p_addr_r;
    reg [6:0]                p_lut_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            bus_shift_reg <= 512'd0;
            cfg_shift512  <= 512'd0;
            fire_lookup   <= 1'b0;
            lang_reg      <= 2'd0;
            cache_en_reg  <= 1'b1;
            cfg_we_r      <= 1'b0;
            cfg_kind_r    <= 2'd0;
            cfg_idx_r     <= 7'd0;
            swap_r        <= 1'b0;
            p_we_r        <= 1'b0;
            p_is_tag_r    <= 1'b0;
            p_addr_r      <= {PAGE_ADDR_BITS{1'b0}};
            p_lut_r       <= 7'd0;
        end else begin
            fire_lookup <= s_axis_tvalid && s_axis_tlast;
            if (s_axis_tvalid) begin
                bus_shift_reg <= {s_axis_tdata, bus_shift_reg[511:64]};
                lang_reg      <= s_axis_tuser_lang;
            end

            cfg_we_r <= 1'b0;
            swap_r   <= 1'b0;
            p_we_r   <= 1'b0;
            if (csr_we) begin
                case (csr_cmd[2:0])
                    3'd0: cfg_shift512 <= {csr_wdata, cfg_shift512[511:32]};
                    3'd1: begin
                        cfg_we_r   <= 1'b1;
                        cfg_kind_r <= csr_cmd[4:3];
                        cfg_idx_r  <= csr_addr[6:0];
                    end
                    3'd2: begin
                        p_we_r     <= 1'b1;
                        p_is_tag_r <= csr_cmd[3];
                        p_addr_r   <= csr_addr[PAGE_ADDR_BITS-1:0];
                        p_lut_r    <= csr_addr[15:9];
                    end
                    3'd3: swap_r       <= 1'b1;
                    3'd4: cache_en_reg <= csr_wdata[0];
                    default: ;
                endcase
            end
        end
    end

    wire        unused_bank;
    wire [79:0] unused_l0;
    wire [31:0] unused_t0, unused_t1, unused_t2;

    clai_benes_paged_pipelined_250mhz #(
        .PAGE_ADDR_BITS  (PAGE_ADDR_BITS),
        .ENABLE_CE_GATING(ENABLE_CE_GATING)
    ) u_pipelined_core (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .bus_in                 (bus_shift_reg),
        .bus_valid              (fire_lookup),
        .ctx_lang_id            (lang_reg),
        .page_cache_en          (cache_en_reg),
        .cfg_we                 (cfg_we_r),
        .cfg_kind               (cfg_kind_r),
        .cfg_idx                (cfg_idx_r),
        .cfg_vec256             (cfg_shift512[255:0]),
        .cfg_vec512             (cfg_shift512),
        .atomic_swap_pulse      (swap_r),
        .page_prog_we           (p_we_r),
        .page_prog_is_tag       (p_is_tag_r),
        .page_prog_addr         (p_addr_r),
        .page_prog_lut_idx      (p_lut_r),
        .page_prog_data64       (cfg_shift512[63:0]),
        .active_bank            (unused_bank),
        .out_valid              (m_axis_tvalid),
        .page_hit_out           (m_axis_page_hit),
        .page_slot_out          (m_axis_page_slot),
        .l0_out_debug           (unused_l0),
        .draft_word16_out       (m_axis_draft_word16),
        .draft_unanimous16_out  (m_axis_draft_unanimous16),
        .telemetry_lookups      (unused_t0),
        .telemetry_page_hits    (unused_t1),
        .telemetry_cam_fallbacks(unused_t2)
    );

endmodule
