`timescale 1ns/1ps
// =============================================================================
// Module: clai_kv260_axi_top
// Target: AMD/Xilinx Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
// Clock : 250 MHz PL Fabric Clock (4.000 ns period, 0 DSP48 multipliers)
// Description:
//   Complete AMBA AXI4-Lite + AXI4-Stream Hardware Subsystem wrapping:
//     1. `clai_engram_streamer` (line-rate byte-stream -> 512-wire Engram bus)
//     2. `clai_rung0_core`      (crystallized 3-layer 1,536-LUT6 combinational fabric)
//     3. 8-Class 64-Input Adder-Tree Popcount Voter + Top-2 Margin Comparator
//     4. Metered Escalation Interrupt (`irq_escalate_to_ps`) & Telemetry CSRs
// =============================================================================

module clai_kv260_axi_top #(
    parameter int C_S_AXI_ADDR_WIDTH = 12,
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int ENGRAM_ADDR_BITS   = 10
) (
    input  wire                              aclk,
    input  wire                              aresetn,

    // -------------------------------------------------------------------------
    // AMBA AXI4-Lite Slave Interface (from Zynq UltraScale+ PS HPM0_FPD)
    // -------------------------------------------------------------------------
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    input  wire [2:0]                        s_axi_awprot,
    input  wire                              s_axi_awvalid,
    output reg                               s_axi_awready,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                              s_axi_wvalid,
    output reg                               s_axi_wready,

    output reg  [1:0]                        s_axi_bresp,
    output reg                               s_axi_bvalid,
    input  wire                              s_axi_bready,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    input  wire [2:0]                        s_axi_arprot,
    input  wire                              s_axi_arvalid,
    output reg                               s_axi_arready,

    output reg  [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    output reg  [1:0]                        s_axi_rresp,
    output reg                               s_axi_rvalid,
    input  wire                              s_axi_rready,

    // -------------------------------------------------------------------------
    // AXI4-Stream Slave Packet Input (from Zynq PS AXI DMA)
    // -------------------------------------------------------------------------
    input  wire [7:0]                        s_axis_tdata,
    input  wire                              s_axis_tvalid,
    output wire                              s_axis_tready,
    input  wire                              s_axis_tlast,

    // -------------------------------------------------------------------------
    // Escalation Interrupt to Zynq UltraScale+ PS GIC (`pl_ps_irq0`)
    // Asserted when Rung 0 vote_margin < csr_margin_thresh (Rung 2/3 wake-up!)
    // -------------------------------------------------------------------------
    output wire                              irq_escalate_to_ps
);

    wire _unused_prot = ^s_axi_awprot ^ ^s_axi_arprot;

    // -------------------------------------------------------------------------
    // Control & Telemetry Registers
    // -------------------------------------------------------------------------
    reg        csr_use_mmio_bus;
    reg        csr_irq_enable;
    reg        csr_overlay_enable;
    reg [7:0]  csr_margin_thresh;
    reg        mmio_eval_pulse;
    reg        atomic_swap_pulse;

    reg        hs_cfg_we;
    reg        hs_layer_sel;
    reg [5:0]  hs_lut_idx;
    reg [63:0] hs_table_init;
    reg [53:0] hs_wires_pack;

    reg [31:0] mmio_bus_regs [0:15];
    reg [511:0] latched_stream_bus;

    reg        bram_we_pulse;
    reg        bram_bank_sel;
    reg [ENGRAM_ADDR_BITS-1:0] bram_waddr;
    reg [63:0] bram_wdata;

    reg        done_sticky;
    reg        last_satiated;
    reg        irq_pending;
    reg [2:0]  latched_pred_class;
    reg [7:0]  latched_best_votes;
    reg [7:0]  latched_second_votes;
    reg [7:0]  latched_margin;
    reg [7:0]  latched_class_votes [0:7];

    reg [31:0] cnt_total_episodes;
    reg [31:0] cnt_satiated;
    reg [31:0] cnt_escalated;

    assign irq_escalate_to_ps = irq_pending && csr_irq_enable;

    // -------------------------------------------------------------------------
    // 1. Streaming Hardware Engram Encoder (AXI4-Stream -> 512-Wire Bus)
    // -------------------------------------------------------------------------
    wire [511:0] stream_bus_out;
    wire         stream_bus_valid;
    wire [15:0]  last_pkt_len;

    clai_engram_streamer #(
        .ENGRAM_ADDR_BITS(ENGRAM_ADDR_BITS)
    ) u_engram_streamer (
        .clk           (aclk),
        .rst_n         (aresetn),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .bram_we       (bram_we_pulse),
        .bram_bank_sel (bram_bank_sel),
        .bram_waddr    (bram_waddr),
        .bram_wdata    (bram_wdata),
        .bus_out       (stream_bus_out),
        .bus_valid     (stream_bus_valid),
        .last_pkt_len  (last_pkt_len)
    );

    // -------------------------------------------------------------------------
    // 2. Multiplexed 512-Wire Input Bus -> Crystallized 1,536-LUT6 Core
    // -------------------------------------------------------------------------
    wire [511:0] mmio_bus_flat = {
        mmio_bus_regs[15], mmio_bus_regs[14], mmio_bus_regs[13], mmio_bus_regs[12],
        mmio_bus_regs[11], mmio_bus_regs[10], mmio_bus_regs[9],  mmio_bus_regs[8],
        mmio_bus_regs[7],  mmio_bus_regs[6],  mmio_bus_regs[5],  mmio_bus_regs[4],
        mmio_bus_regs[3],  mmio_bus_regs[2],  mmio_bus_regs[1],  mmio_bus_regs[0]
    };

    // Pipeline stage registering the active 512-wire bus prior to LUT6 array
    reg [511:0] core_bus_reg;
    reg         core_eval_valid;

    always @(posedge aclk) begin
        if (!aresetn) begin
            core_bus_reg       <= 512'd0;
            latched_stream_bus <= 512'd0;
            core_eval_valid    <= 1'b0;
        end else begin
            core_eval_valid <= 1'b0;
            if (!csr_use_mmio_bus && stream_bus_valid) begin
                core_bus_reg       <= stream_bus_out;
                latched_stream_bus <= stream_bus_out;
                core_eval_valid    <= 1'b1;
            end else if (csr_use_mmio_bus && mmio_eval_pulse) begin
                core_bus_reg    <= mmio_bus_flat;
                core_eval_valid <= 1'b1;
            end
        end
    end

    wire [511:0] static_votes_raw;
    wire [511:0] overlay_votes_raw;
    wire         hs_active_bank;
    wire [31:0]  hs_swap_generation;

    clai_rung0_core u_lut6_fabric (
        .bus_in        (core_bus_reg),
        .lut_votes_out (static_votes_raw)
    );

    clai_rung0_hotswap_core u_lut6_hotswap_overlay (
        .clk               (aclk),
        .rst_n             (aresetn),
        .bus_in            (core_bus_reg),
        .cfg_we            (hs_cfg_we),
        .cfg_layer_sel     (hs_layer_sel),
        .cfg_lut_idx       (hs_lut_idx),
        .cfg_table_init    (hs_table_init),
        .cfg_wires_pack    (hs_wires_pack),
        .atomic_swap_pulse (atomic_swap_pulse),
        .active_bank       (hs_active_bank),
        .swap_generation   (hs_swap_generation),
        .lut_votes_out     (overlay_votes_raw)
    );

    wire [511:0] lut_votes_raw = csr_overlay_enable ? overlay_votes_raw : static_votes_raw;

    // -------------------------------------------------------------------------
    // 3. 8-Class x 64-Wire Parallel Popcount Voter + Top-2 Margin Comparator
    // -------------------------------------------------------------------------
    reg [7:0] comb_votes [0:7];
    reg [2:0] comb_best_cls;
    reg [7:0] comb_best_v;
    reg [7:0] comb_second_v;
    reg [7:0] comb_margin;

    integer c_idx, b_idx;
    reg [7:0] pop_acc;
    always @(*) begin
        for (c_idx = 0; c_idx < 8; c_idx = c_idx + 1) begin
            pop_acc = 8'd0;
            for (b_idx = 0; b_idx < 64; b_idx = b_idx + 1) begin
                pop_acc = pop_acc + {7'd0, lut_votes_raw[c_idx * 64 + b_idx]};
            end
            comb_votes[c_idx] = pop_acc;
        end

        comb_best_cls = 3'd0;
        comb_best_v   = 8'd0;
        comb_second_v = 8'd0;
        for (c_idx = 0; c_idx < 8; c_idx = c_idx + 1) begin
            if (comb_votes[c_idx] > comb_best_v) begin
                comb_second_v = comb_best_v;
                comb_best_v   = comb_votes[c_idx];
                comb_best_cls = c_idx[2:0];
            end else if (comb_votes[c_idx] > comb_second_v) begin
                comb_second_v = comb_votes[c_idx];
            end
        end
        comb_margin = comb_best_v - comb_second_v;
    end

    // -------------------------------------------------------------------------
    // 4. Spec-Compliant AMBA AXI4-Lite Slave State Machine & CSR Updates
    // -------------------------------------------------------------------------
    reg aw_captured;
    reg w_captured;
    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_latched;
    reg [31:0]                   wdata_latched;
    reg [3:0]                    wstrb_latched;

    wire wr_fire = (aw_captured || (s_axi_awvalid && s_axi_awready)) &&
                   (w_captured  || (s_axi_wvalid  && s_axi_wready));
    wire [C_S_AXI_ADDR_WIDTH-1:0] eff_awaddr = aw_captured ? awaddr_latched : s_axi_awaddr;
    wire [31:0]                   eff_wdata  = w_captured  ? wdata_latched  : s_axi_wdata;
    wire [3:0]                    eff_wstrb  = w_captured  ? wstrb_latched  : s_axi_wstrb;

    integer m_i;
    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready      <= 1'b0;
            s_axi_wready       <= 1'b0;
            s_axi_bvalid       <= 1'b0;
            s_axi_bresp        <= 2'b00;
            aw_captured        <= 1'b0;
            w_captured         <= 1'b0;
            awaddr_latched     <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            wdata_latched      <= 32'd0;
            wstrb_latched      <= 4'd0;

            csr_use_mmio_bus   <= 1'b0;
            csr_irq_enable     <= 1'b1;
            csr_overlay_enable <= 1'b0;
            csr_margin_thresh  <= 8'd4;
            mmio_eval_pulse    <= 1'b0;
            atomic_swap_pulse  <= 1'b0;
            hs_cfg_we          <= 1'b0;
            hs_layer_sel       <= 1'b0;
            hs_lut_idx         <= 6'd0;
            hs_table_init      <= 64'd0;
            hs_wires_pack      <= 54'd0;
            bram_we_pulse      <= 1'b0;
            bram_bank_sel      <= 1'b0;
            bram_waddr         <= {ENGRAM_ADDR_BITS{1'b0}};
            bram_wdata         <= 64'd0;

            done_sticky        <= 1'b0;
            last_satiated      <= 1'b0;
            irq_pending        <= 1'b0;
            latched_pred_class <= 3'd0;
            latched_best_votes <= 8'd0;
            latched_second_votes <= 8'd0;
            latched_margin     <= 8'd0;
            cnt_total_episodes <= 32'd0;
            cnt_satiated       <= 32'd0;
            cnt_escalated      <= 32'd0;

            for (m_i = 0; m_i < 8; m_i = m_i + 1)
                latched_class_votes[m_i] <= 8'd0;
            for (m_i = 0; m_i < 16; m_i = m_i + 1)
                mmio_bus_regs[m_i] <= 32'd0;
        end else begin
            mmio_eval_pulse   <= 1'b0;
            atomic_swap_pulse <= 1'b0;
            hs_cfg_we         <= 1'b0;
            bram_we_pulse     <= 1'b0;

            // Latch LUT6 evaluation result & update thermodynamic counters
            if (core_eval_valid) begin
                done_sticky          <= 1'b1;
                latched_pred_class   <= comb_best_cls;
                latched_best_votes   <= comb_best_v;
                latched_second_votes <= comb_second_v;
                latched_margin       <= comb_margin;
                for (m_i = 0; m_i < 8; m_i = m_i + 1)
                    latched_class_votes[m_i] <= comb_votes[m_i];

                cnt_total_episodes <= cnt_total_episodes + 32'd1;
                if (comb_margin >= csr_margin_thresh) begin
                    last_satiated <= 1'b1;
                    cnt_satiated  <= cnt_satiated + 32'd1;
                end else begin
                    last_satiated <= 1'b0;
                    cnt_escalated <= cnt_escalated + 32'd1;
                    irq_pending   <= 1'b1;
                end
            end

            // AXI4-Lite Write Address Acceptance
            if (!aw_captured && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                if (s_axi_awvalid && s_axi_awready) begin
                    awaddr_latched <= s_axi_awaddr;
                    aw_captured    <= 1'b1;
                    s_axi_awready  <= 1'b0;
                end
            end else begin
                s_axi_awready <= 1'b0;
            end

            // AXI4-Lite Write Data Acceptance
            if (!w_captured && !s_axi_bvalid) begin
                s_axi_wready <= 1'b1;
                if (s_axi_wvalid && s_axi_wready) begin
                    wdata_latched <= s_axi_wdata;
                    wstrb_latched <= s_axi_wstrb;
                    w_captured    <= 1'b1;
                    s_axi_wready  <= 1'b0;
                end
            end else begin
                s_axi_wready <= 1'b0;
            end

            // Commit Write Transaction
            if (wr_fire && !s_axi_bvalid) begin
                aw_captured  <= 1'b0;
                w_captured   <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00; // OKAY

                if (eff_awaddr[11:6] == 6'b0001_00) begin
                    // 0x100..0x13C: 16 x 32-bit MMIO Bus Words
                    if (eff_wstrb[0]) mmio_bus_regs[eff_awaddr[5:2]][7:0]   <= eff_wdata[7:0];
                    if (eff_wstrb[1]) mmio_bus_regs[eff_awaddr[5:2]][15:8]  <= eff_wdata[15:8];
                    if (eff_wstrb[2]) mmio_bus_regs[eff_awaddr[5:2]][23:16] <= eff_wdata[23:16];
                    if (eff_wstrb[3]) mmio_bus_regs[eff_awaddr[5:2]][31:24] <= eff_wdata[31:24];
                end else begin
                    case (eff_awaddr[11:2])
                        10'h000: begin // 0x000 CSR_CTRL
                            if (eff_wstrb[0]) begin
                                mmio_eval_pulse    <= eff_wdata[0];
                                csr_use_mmio_bus   <= eff_wdata[1];
                                csr_irq_enable     <= eff_wdata[2];
                                csr_overlay_enable <= eff_wdata[4];
                                atomic_swap_pulse  <= eff_wdata[5];
                                if (eff_wdata[3]) begin
                                    cnt_total_episodes <= 32'd0;
                                    cnt_satiated       <= 32'd0;
                                    cnt_escalated      <= 32'd0;
                                    done_sticky        <= 1'b0;
                                end
                            end
                        end
                        10'h001: begin // 0x004 CSR_STATUS (W1C on bit 0 done & bit 2 irq_pending)
                            if (eff_wstrb[0]) begin
                                if (eff_wdata[0]) done_sticky <= 1'b0;
                                if (eff_wdata[2]) irq_pending <= 1'b0;
                            end
                        end
                        10'h002: begin // 0x008 CSR_MARGIN_THRESH
                            if (eff_wstrb[0]) csr_margin_thresh <= eff_wdata[7:0];
                        end
                        10'h010: begin // 0x040 CSR_BRAM_CTRL
                            bram_waddr    <= eff_wdata[ENGRAM_ADDR_BITS-1:0];
                            bram_bank_sel <= eff_wdata[16];
                            bram_we_pulse <= eff_wdata[31];
                        end
                        10'h011: bram_wdata[31:0]  <= eff_wdata; // 0x044 CSR_BRAM_WDATA_LO
                        10'h012: bram_wdata[63:32] <= eff_wdata; // 0x048 CSR_BRAM_WDATA_HI
                        10'h014: begin // 0x050 CSR_HS_CTRL
                            hs_lut_idx   <= eff_wdata[5:0];
                            hs_layer_sel <= eff_wdata[8];
                            hs_cfg_we    <= eff_wdata[31];
                        end
                        10'h015: hs_table_init[31:0]  <= eff_wdata;       // 0x054 CSR_HS_TABLE_LO
                        10'h016: hs_table_init[63:32] <= eff_wdata;       // 0x058 CSR_HS_TABLE_HI
                        10'h017: hs_wires_pack[26:0]  <= eff_wdata[26:0]; // 0x05C CSR_HS_WIRES_LO
                        10'h018: hs_wires_pack[53:27] <= eff_wdata[26:0]; // 0x060 CSR_HS_WIRES_HI
                        default: ;
                    endcase
                end
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI4-Lite Read Channel
    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (!s_axi_rvalid && !s_axi_arready && s_axi_arvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                s_axi_rresp   <= 2'b00;

                if (s_axi_araddr[11:6] == 6'b0001_00) begin
                    if (csr_use_mmio_bus)
                        s_axi_rdata <= mmio_bus_regs[s_axi_araddr[5:2]];
                    else
                        s_axi_rdata <= latched_stream_bus[s_axi_araddr[5:2] * 32 +: 32];
                end else if (s_axi_araddr[11:5] == 7'b0000_001) begin
                    // 0x020..0x03C: Individual Class Popcount Votes [0..7]
                    s_axi_rdata <= {24'd0, latched_class_votes[s_axi_araddr[4:2]]};
                end else begin
                    case (s_axi_araddr[11:2])
                        10'h000: s_axi_rdata <= {27'd0, csr_overlay_enable, 1'b0, csr_irq_enable, csr_use_mmio_bus, 1'b0};
                        10'h001: s_axi_rdata <= {12'd0, last_pkt_len, 1'b0, irq_pending, last_satiated, done_sticky};
                        10'h002: s_axi_rdata <= {24'd0, csr_margin_thresh};
                        10'h003: s_axi_rdata <= {latched_margin, latched_second_votes, latched_best_votes, 5'd0, latched_pred_class};
                        10'h004: s_axi_rdata <= cnt_total_episodes;
                        10'h005: s_axi_rdata <= cnt_satiated;
                        10'h006: s_axi_rdata <= cnt_escalated;
                        10'h019: s_axi_rdata <= {hs_active_bank, hs_swap_generation[30:0]}; // 0x064 CSR_HS_GENERATION
                        10'h020: s_axi_rdata <= 32'h434C_4149; // "CLAI" Magic Signature
                        default: s_axi_rdata <= 32'd0;
                    endcase
                end
            end else begin
                s_axi_arready <= 1'b0;
                if (s_axi_rvalid && s_axi_rready)
                    s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
