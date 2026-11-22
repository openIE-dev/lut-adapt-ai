`timescale 1ns/1ps
// =============================================================================
// Module: clai_engram_streamer
// Target: AMD/Xilinx Kria KV260 (Zynq UltraScale+ xck26-sfvc784-2LV-c)
// Description:
//   Zero-MAC, Zero-Divider AXI4-Stream Byte-Packet -> 512-Wire EngramBitBus
//   Hardware Encoder.
//   - Words 0..1 (Wires   0..127): 128-bit streaming 16-byte XOR fold
//   - Words 2..5 (Wires 128..383): 16-slot x 16-level chunk-normalized monotonic
//     thermometer bars via exact division-free identity:
//       floor(sum[s] / n_chunks) >= 15*(lvl+1) <==> sum[s] >= (15*n_chunks)*(lvl+1)
//     computed purely with shift-add-subtract LUT logic (0 DSP48s, 0 dividers).
//   - Words 6..7 (Wires 384..511): Dual-bank 64-bit Associative Engram BRAM
//     updated live by Zynq PS SleepCrystallizer over AXI4-Lite.
// =============================================================================

module clai_engram_streamer #(
    parameter int ENGRAM_ADDR_BITS = 10
) (
    input  wire                         clk,
    input  wire                         rst_n,

    // AXI4-Stream Slave (from Zynq PS AXI DMA)
    input  wire [7:0]                   s_axis_tdata,
    input  wire                         s_axis_tvalid,
    output wire                         s_axis_tready,
    input  wire                         s_axis_tlast,

    // SleepCrystallizer Engram BRAM Write Port (from AXI4-Lite CSRs)
    input  wire                         bram_we,
    input  wire                         bram_bank_sel, // 0 -> Word 6, 1 -> Word 7
    input  wire [ENGRAM_ADDR_BITS-1:0]  bram_waddr,
    input  wire [63:0]                  bram_wdata,

    // Encoded 512-Wire Output Bus (valid for 1 cycle on `bus_valid`)
    output reg  [511:0]                 bus_out,
    output reg                          bus_valid,
    output reg  [15:0]                  last_pkt_len
);

    localparam int NUM_BUCKETS = (1 << ENGRAM_ADDR_BITS);

    typedef enum logic [1:0] {
        ST_STREAM   = 2'd0,
        ST_FINALIZE = 2'd1
    } state_t;

    state_t state;

    reg [7:0]  xor_bytes [0:15];
    reg [15:0] slot_sum  [0:15];
    reg [15:0] byte_cnt;
    reg [23:0] quad_shift;
    reg [31:0] rolling_hash;

    // Dual-bank 64-bit Associative Engram BRAM (maps to UltraScale+ RAMB36E2)
    (* ram_style = "block" *) reg [63:0] engram_bank6 [0:NUM_BUCKETS-1];
    (* ram_style = "block" *) reg [63:0] engram_bank7 [0:NUM_BUCKETS-1];

    reg [63:0] bram_r6;
    reg [63:0] bram_r7;

    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_BUCKETS; init_i = init_i + 1) begin
            engram_bank6[init_i] = 64'h9E3779B97F4A7C15 ^ {32'd0, init_i[31:0]};
            engram_bank7[init_i] = 64'hBF58476D1CE4E5B9 ^ {init_i[31:0], 32'd0};
        end
    end

    assign s_axis_tready = (state == ST_STREAM);

    wire accept_byte = s_axis_tvalid && s_axis_tready;
    wire [3:0] cur_slot = byte_cnt[3:0];
    wire [15:0] next_byte_cnt = byte_cnt + 16'd1;

    // Update rolling 4-gram hash with zero DSP48s (shift-add + XOR)
    wire [31:0] next_quad = {s_axis_tdata, quad_shift};
    wire [31:0] hash_step = ((rolling_hash << 5) + rolling_hash) ^ next_quad;
    wire [31:0] eff_hash  = (byte_cnt[1:0] == 2'b11) ? hash_step : rolling_hash;

    wire [ENGRAM_ADDR_BITS-1:0] raddr6 = eff_hash[ENGRAM_ADDR_BITS-1:0] ^ eff_hash[2*ENGRAM_ADDR_BITS-1:ENGRAM_ADDR_BITS];
    wire [ENGRAM_ADDR_BITS-1:0] raddr7 = eff_hash[31:32-ENGRAM_ADDR_BITS] ^ eff_hash[ENGRAM_ADDR_BITS-1:0];

    // Synchronous BRAM Read / Write
    always @(posedge clk) begin
        if (bram_we) begin
            if (!bram_bank_sel)
                engram_bank6[bram_waddr] <= bram_wdata;
            else
                engram_bank7[bram_waddr] <= bram_wdata;
        end
        if (accept_byte && s_axis_tlast) begin
            bram_r6 <= engram_bank6[raddr6];
            bram_r7 <= engram_bank7[raddr7];
        end
    end

    // Division-free, DSP-free Thermometer Combinational Array (evaluated in ST_FINALIZE)
    // n_chunks = ceil(last_pkt_len / 16) = (last_pkt_len + 15) >> 4, min 1
    wire [15:0] raw_chunks = (last_pkt_len + 16'd15) >> 4;
    wire [15:0] n_chunks   = (raw_chunks == 16'd0) ? 16'd1 : raw_chunks;
    // base_step = 15 * n_chunks = (n_chunks << 4) - n_chunks (zero multipliers!)
    wire [19:0] base_step  = ({4'd0, n_chunks} << 4) - {4'd0, n_chunks};

    wire [19:0] thresh [0:15];
    assign thresh[0]  = base_step;                         // 1 * step
    assign thresh[1]  = base_step << 1;                    // 2 * step
    assign thresh[2]  = (base_step << 1) + base_step;      // 3 * step
    assign thresh[3]  = base_step << 2;                    // 4 * step
    assign thresh[4]  = (base_step << 2) + base_step;      // 5 * step
    assign thresh[5]  = (base_step << 2) + (base_step << 1);// 6 * step
    assign thresh[6]  = (base_step << 3) - base_step;      // 7 * step
    assign thresh[7]  = base_step << 3;                    // 8 * step
    assign thresh[8]  = (base_step << 3) + base_step;      // 9 * step
    assign thresh[9]  = (base_step << 3) + (base_step << 1);// 10 * step
    assign thresh[10] = (base_step << 3) + (base_step << 1) + base_step; // 11 * step
    assign thresh[11] = (base_step << 3) + (base_step << 2);// 12 * step
    assign thresh[12] = (base_step << 4) - (base_step << 1) - base_step; // 13 * step
    assign thresh[13] = (base_step << 4) - (base_step << 1);// 14 * step
    assign thresh[14] = (base_step << 4) - base_step;      // 15 * step
    assign thresh[15] = base_step << 4;                    // 16 * step

    wire [255:0] thermo_flat;
    genvar gs, gl;
    generate
        for (gs = 0; gs < 16; gs = gs + 1) begin : gen_slot
            for (gl = 0; gl < 16; gl = gl + 1) begin : gen_lvl
                assign thermo_flat[gs * 16 + gl] = ({4'd0, slot_sum[gs]} >= thresh[gl]);
            end
        end
    endgenerate

    wire [63:0] xor_word0 = {
        xor_bytes[7],  xor_bytes[6],  xor_bytes[5],  xor_bytes[4],
        xor_bytes[3],  xor_bytes[2],  xor_bytes[1],  xor_bytes[0]
    };
    wire [63:0] xor_word1 = {
        xor_bytes[15], xor_bytes[14], xor_bytes[13], xor_bytes[12],
        xor_bytes[11], xor_bytes[10], xor_bytes[9],  xor_bytes[8]
    };

    integer k;
    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_STREAM;
            byte_cnt     <= 16'd0;
            quad_shift   <= 24'd0;
            rolling_hash <= 32'd0;
            bus_out      <= 512'd0;
            bus_valid    <= 1'b0;
            last_pkt_len <= 16'd0;
            for (k = 0; k < 16; k = k + 1) begin
                xor_bytes[k] <= 8'd0;
                slot_sum[k]  <= 16'd0;
            end
        end else begin
            bus_valid <= 1'b0;

            case (state)
                ST_STREAM: begin
                    if (accept_byte) begin
                        xor_bytes[cur_slot] <= xor_bytes[cur_slot] ^ s_axis_tdata;
                        slot_sum[cur_slot]  <= slot_sum[cur_slot] + {8'd0, s_axis_tdata};
                        quad_shift          <= {s_axis_tdata, quad_shift[23:8]};
                        rolling_hash        <= eff_hash;

                        if (s_axis_tlast) begin
                            last_pkt_len <= next_byte_cnt;
                            state        <= ST_FINALIZE;
                        end else begin
                            byte_cnt <= next_byte_cnt;
                        end
                    end
                end

                ST_FINALIZE: begin
                    bus_out   <= {bram_r7, bram_r6, thermo_flat, xor_word1, xor_word0};
                    bus_valid <= 1'b1;

                    // Reset accumulators for next variable-length stream packet in 1 cycle!
                    byte_cnt     <= 16'd0;
                    quad_shift   <= 24'd0;
                    rolling_hash <= 32'd0;
                    for (k = 0; k < 16; k = k + 1) begin
                        xor_bytes[k] <= 8'd0;
                        slot_sum[k]  <= 16'd0;
                    end
                    state <= ST_STREAM;
                end

                default: state <= ST_STREAM;
            endcase
        end
    end

endmodule
