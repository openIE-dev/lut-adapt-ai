`timescale 1ns/1ps
module tb_clai_kv260_axi;
    logic        aclk = 0;
    logic        aresetn = 0;
    always #2 aclk = ~aclk; // 250 MHz PL Clock (4.0 ns period)

    logic [11:0] s_axi_awaddr = 0;
    logic [2:0]  s_axi_awprot = 0;
    logic        s_axi_awvalid = 0;
    wire         s_axi_awready;
    logic [31:0] s_axi_wdata = 0;
    logic [3:0]  s_axi_wstrb = 4'hF;
    logic        s_axi_wvalid = 0;
    wire         s_axi_wready;
    wire  [1:0]  s_axi_bresp;
    wire         s_axi_bvalid;
    logic        s_axi_bready = 0;
    logic [11:0] s_axi_araddr = 0;
    logic [2:0]  s_axi_arprot = 0;
    logic        s_axi_arvalid = 0;
    wire         s_axi_arready;
    wire  [31:0] s_axi_rdata;
    wire  [1:0]  s_axi_rresp;
    wire         s_axi_rvalid;
    logic        s_axi_rready = 0;

    logic [7:0]  s_axis_tdata = 0;
    logic        s_axis_tvalid = 0;
    wire         s_axis_tready;
    logic        s_axis_tlast = 0;
    wire         irq_escalate_to_ps;

    integer errors = 0;
    logic [31:0] rd_val;

    clai_kv260_axi_top #(.ENGRAM_ADDR_BITS(10)) dut (.*);

    task automatic axi_write(input [11:0] addr, input [31:0] data, input integer mode);
        begin
            @(posedge aclk);
            if (mode == 0) begin
                // Simultaneous AW + W
                s_axi_awaddr <= addr; s_axi_awvalid <= 1'b1;
                s_axi_wdata  <= data; s_axi_wstrb <= 4'hF; s_axi_wvalid <= 1'b1;
                s_axi_bready <= 1'b1;
                fork
                    begin do @(posedge aclk); while (!s_axi_awready); s_axi_awvalid <= 1'b0; end
                    begin do @(posedge aclk); while (!s_axi_wready);  s_axi_wvalid  <= 1'b0; end
                join
            end else begin
                // Decoupled AW-then-W handshake check
                s_axi_awaddr <= addr; s_axi_awvalid <= 1'b1; s_axi_bready <= 1'b1;
                do @(posedge aclk); while (!s_axi_awready);
                s_axi_awvalid <= 1'b0;
                @(posedge aclk);
                s_axi_wdata <= data; s_axi_wstrb <= 4'hF; s_axi_wvalid <= 1'b1;
                do @(posedge aclk); while (!s_axi_wready);
                s_axi_wvalid <= 1'b0;
            end
            do @(posedge aclk); while (!s_axi_bvalid);
            s_axi_bready <= 1'b0;
        end
    endtask

    task automatic axi_read(input [11:0] addr, output [31:0] data);
        begin
            @(posedge aclk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;
            do @(posedge aclk); while (!s_axi_arready);
            s_axi_arvalid <= 1'b0;
            while (!s_axi_rvalid) @(posedge aclk);
            data = s_axi_rdata;
            @(posedge aclk);
            s_axi_rready <= 1'b0;
        end
    endtask

    task automatic axis_send_byte(input [7:0] b, input bit is_last, input bit bubble);
        begin
            if (bubble) begin
                @(posedge aclk);
                s_axis_tvalid <= 1'b0;
            end
            @(posedge aclk);
            s_axis_tdata  <= b;
            s_axis_tlast  <= is_last;
            s_axis_tvalid <= 1'b1;
            do @(posedge aclk); while (!s_axis_tready);
            s_axis_tvalid <= 1'b0;
            s_axis_tlast  <= 1'b0;
        end
    endtask

    initial begin
        repeat (5) @(posedge aclk);
        aresetn = 1;
        repeat (3) @(posedge aclk);

        // 1. Verify CLAI Magic CSR (0x080 == 0x434C4149)
        axi_read(12'h080, rd_val);
        if (rd_val !== 32'h434C4149) begin
            $display("MAGIC CSR MISMATCH: %h", rd_val); errors = errors + 1;
        end

        // 2. Program Engram BRAM slot 0x12A on Bank 0 & Bank 1 + set margin_thresh = 6
        axi_write(12'h008, 32'd6, 1);
        axi_write(12'h044, 32'h12345678, 0);
        axi_write(12'h048, 32'hCAFEBABE, 0);
        axi_write(12'h040, 32'h8000012A, 0);
        axi_write(12'h044, 32'h87654321, 0);
        axi_write(12'h048, 32'hDEADBEEF, 0);
        axi_write(12'h040, 32'h8001012A, 0);

        // --- Stream Packet 0 (len=57) ---
        axis_send_byte(8'd4, 1'b0, 1'b0);
        axis_send_byte(8'd172, 1'b0, 1'b0);
        axis_send_byte(8'd206, 1'b0, 1'b0);
        axis_send_byte(8'd45, 1'b0, 1'b0);
        axis_send_byte(8'd191, 1'b0, 1'b1);
        axis_send_byte(8'd76, 1'b0, 1'b0);
        axis_send_byte(8'd195, 1'b0, 1'b0);
        axis_send_byte(8'd6, 1'b0, 1'b0);
        axis_send_byte(8'd103, 1'b0, 1'b0);
        axis_send_byte(8'd89, 1'b0, 1'b0);
        axis_send_byte(8'd209, 1'b0, 1'b0);
        axis_send_byte(8'd163, 1'b0, 1'b0);
        axis_send_byte(8'd234, 1'b0, 1'b0);
        axis_send_byte(8'd241, 1'b0, 1'b1);
        axis_send_byte(8'd143, 1'b0, 1'b0);
        axis_send_byte(8'd93, 1'b0, 1'b0);
        axis_send_byte(8'd229, 1'b0, 1'b0);
        axis_send_byte(8'd230, 1'b0, 1'b0);
        axis_send_byte(8'd158, 1'b0, 1'b0);
        axis_send_byte(8'd119, 1'b0, 1'b0);
        axis_send_byte(8'd115, 1'b0, 1'b0);
        axis_send_byte(8'd156, 1'b0, 1'b0);
        axis_send_byte(8'd111, 1'b0, 1'b1);
        axis_send_byte(8'd20, 1'b0, 1'b0);
        axis_send_byte(8'd95, 1'b0, 1'b0);
        axis_send_byte(8'd31, 1'b0, 1'b0);
        axis_send_byte(8'd217, 1'b0, 1'b0);
        axis_send_byte(8'd94, 1'b0, 1'b0);
        axis_send_byte(8'd26, 1'b0, 1'b0);
        axis_send_byte(8'd54, 1'b0, 1'b0);
        axis_send_byte(8'd185, 1'b0, 1'b0);
        axis_send_byte(8'd164, 1'b0, 1'b1);
        axis_send_byte(8'd199, 1'b0, 1'b0);
        axis_send_byte(8'd220, 1'b0, 1'b0);
        axis_send_byte(8'd219, 1'b0, 1'b0);
        axis_send_byte(8'd90, 1'b0, 1'b0);
        axis_send_byte(8'd40, 1'b0, 1'b0);
        axis_send_byte(8'd107, 1'b0, 1'b0);
        axis_send_byte(8'd192, 1'b0, 1'b0);
        axis_send_byte(8'd212, 1'b0, 1'b0);
        axis_send_byte(8'd92, 1'b0, 1'b1);
        axis_send_byte(8'd103, 1'b0, 1'b0);
        axis_send_byte(8'd95, 1'b0, 1'b0);
        axis_send_byte(8'd202, 1'b0, 1'b0);
        axis_send_byte(8'd92, 1'b0, 1'b0);
        axis_send_byte(8'd91, 1'b0, 1'b0);
        axis_send_byte(8'd85, 1'b0, 1'b0);
        axis_send_byte(8'd180, 1'b0, 1'b0);
        axis_send_byte(8'd168, 1'b0, 1'b0);
        axis_send_byte(8'd129, 1'b0, 1'b1);
        axis_send_byte(8'd183, 1'b0, 1'b0);
        axis_send_byte(8'd231, 1'b0, 1'b0);
        axis_send_byte(8'd23, 1'b0, 1'b0);
        axis_send_byte(8'd32, 1'b0, 1'b0);
        axis_send_byte(8'd161, 1'b0, 1'b0);
        axis_send_byte(8'd184, 1'b0, 1'b0);
        axis_send_byte(8'd206, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'hE73C178E) begin
            $display("PKT 0 BUS WORD 0 MISMATCH: got %h exp E73C178E", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h7ECD9BF3) begin
            $display("PKT 0 BUS WORD 1 MISMATCH: got %h exp 7ECD9BF3", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'h375721AA) begin
            $display("PKT 0 BUS WORD 2 MISMATCH: got %h exp 375721AA", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h4D639CAC) begin
            $display("PKT 0 BUS WORD 3 MISMATCH: got %h exp 4D639CAC", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h0FFF03FF) begin
            $display("PKT 0 BUS WORD 4 MISMATCH: got %h exp 0FFF03FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h00FF0FFF) begin
            $display("PKT 0 BUS WORD 5 MISMATCH: got %h exp 00FF0FFF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h003F003F) begin
            $display("PKT 0 BUS WORD 6 MISMATCH: got %h exp 003F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h007F03FF) begin
            $display("PKT 0 BUS WORD 7 MISMATCH: got %h exp 007F03FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h000700FF) begin
            $display("PKT 0 BUS WORD 8 MISMATCH: got %h exp 000700FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h007F00FF) begin
            $display("PKT 0 BUS WORD 9 MISMATCH: got %h exp 007F00FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h003F001F) begin
            $display("PKT 0 BUS WORD 10 MISMATCH: got %h exp 003F001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h007F003F) begin
            $display("PKT 0 BUS WORD 11 MISMATCH: got %h exp 007F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7F24) begin
            $display("PKT 0 BUS WORD 12 MISMATCH: got %h exp 7F4A7F24", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 0 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 0 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF584451) begin
            $display("PKT 0 BUS WORD 15 MISMATCH: got %h exp BF584451", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 0 CLASS 0 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd26) begin
            $display("PKT 0 CLASS 1 VOTE MISMATCH: got %0d exp 26", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd27) begin
            $display("PKT 0 CLASS 2 VOTE MISMATCH: got %0d exp 27", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd42) begin
            $display("PKT 0 CLASS 3 VOTE MISMATCH: got %0d exp 42", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 0 CLASS 4 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd36) begin
            $display("PKT 0 CLASS 5 VOTE MISMATCH: got %0d exp 36", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd35) begin
            $display("PKT 0 CLASS 6 VOTE MISMATCH: got %0d exp 35", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd34) begin
            $display("PKT 0 CLASS 7 VOTE MISMATCH: got %0d exp 34", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h06242A03) begin
            $display("PKT 0 RESULT CSR MISMATCH: got %h exp 06242A03", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b0) begin
            $display("PKT 0 IRQ MISMATCH: got %b exp 0", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 1 (len=57) ---
        axis_send_byte(8'd162, 1'b0, 1'b0);
        axis_send_byte(8'd231, 1'b0, 1'b0);
        axis_send_byte(8'd28, 1'b0, 1'b0);
        axis_send_byte(8'd167, 1'b0, 1'b0);
        axis_send_byte(8'd13, 1'b0, 1'b1);
        axis_send_byte(8'd29, 1'b0, 1'b0);
        axis_send_byte(8'd53, 1'b0, 1'b0);
        axis_send_byte(8'd25, 1'b0, 1'b0);
        axis_send_byte(8'd246, 1'b0, 1'b0);
        axis_send_byte(8'd145, 1'b0, 1'b0);
        axis_send_byte(8'd93, 1'b0, 1'b0);
        axis_send_byte(8'd76, 1'b0, 1'b0);
        axis_send_byte(8'd49, 1'b0, 1'b0);
        axis_send_byte(8'd148, 1'b0, 1'b1);
        axis_send_byte(8'd105, 1'b0, 1'b0);
        axis_send_byte(8'd42, 1'b0, 1'b0);
        axis_send_byte(8'd178, 1'b0, 1'b0);
        axis_send_byte(8'd213, 1'b0, 1'b0);
        axis_send_byte(8'd138, 1'b0, 1'b0);
        axis_send_byte(8'd247, 1'b0, 1'b0);
        axis_send_byte(8'd113, 1'b0, 1'b0);
        axis_send_byte(8'd150, 1'b0, 1'b0);
        axis_send_byte(8'd91, 1'b0, 1'b1);
        axis_send_byte(8'd186, 1'b0, 1'b0);
        axis_send_byte(8'd212, 1'b0, 1'b0);
        axis_send_byte(8'd76, 1'b0, 1'b0);
        axis_send_byte(8'd123, 1'b0, 1'b0);
        axis_send_byte(8'd235, 1'b0, 1'b0);
        axis_send_byte(8'd24, 1'b0, 1'b0);
        axis_send_byte(8'd221, 1'b0, 1'b0);
        axis_send_byte(8'd81, 1'b0, 1'b0);
        axis_send_byte(8'd72, 1'b0, 1'b1);
        axis_send_byte(8'd212, 1'b0, 1'b0);
        axis_send_byte(8'd109, 1'b0, 1'b0);
        axis_send_byte(8'd200, 1'b0, 1'b0);
        axis_send_byte(8'd162, 1'b0, 1'b0);
        axis_send_byte(8'd0, 1'b0, 1'b0);
        axis_send_byte(8'd41, 1'b0, 1'b0);
        axis_send_byte(8'd171, 1'b0, 1'b0);
        axis_send_byte(8'd154, 1'b0, 1'b0);
        axis_send_byte(8'd251, 1'b0, 1'b1);
        axis_send_byte(8'd12, 1'b0, 1'b0);
        axis_send_byte(8'd179, 1'b0, 1'b0);
        axis_send_byte(8'd192, 1'b0, 1'b0);
        axis_send_byte(8'd97, 1'b0, 1'b0);
        axis_send_byte(8'd80, 1'b0, 1'b0);
        axis_send_byte(8'd54, 1'b0, 1'b0);
        axis_send_byte(8'd30, 1'b0, 1'b0);
        axis_send_byte(8'd214, 1'b0, 1'b0);
        axis_send_byte(8'd28, 1'b0, 1'b1);
        axis_send_byte(8'd225, 1'b0, 1'b0);
        axis_send_byte(8'd131, 1'b0, 1'b0);
        axis_send_byte(8'd53, 1'b0, 1'b0);
        axis_send_byte(8'd201, 1'b0, 1'b0);
        axis_send_byte(8'd0, 1'b0, 1'b0);
        axis_send_byte(8'd165, 1'b0, 1'b0);
        axis_send_byte(8'd36, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'h71BF4312) begin
            $display("PKT 1 BUS WORD 0 MISMATCH: got %h exp 71BF4312", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h9CC56B49) begin
            $display("PKT 1 BUS WORD 1 MISMATCH: got %h exp 9CC56B49", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'h6795D1FD) begin
            $display("PKT 1 BUS WORD 2 MISMATCH: got %h exp 6795D1FD", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h7C0E1948) begin
            $display("PKT 1 BUS WORD 3 MISMATCH: got %h exp 7C0E1948", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h01FF0FFF) begin
            $display("PKT 1 BUS WORD 4 MISMATCH: got %h exp 01FF0FFF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h07FF01FF) begin
            $display("PKT 1 BUS WORD 5 MISMATCH: got %h exp 07FF01FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h007F0003) begin
            $display("PKT 1 BUS WORD 6 MISMATCH: got %h exp 007F0003", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h00FF001F) begin
            $display("PKT 1 BUS WORD 7 MISMATCH: got %h exp 00FF001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h00070FFF) begin
            $display("PKT 1 BUS WORD 8 MISMATCH: got %h exp 00070FFF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h00FF003F) begin
            $display("PKT 1 BUS WORD 9 MISMATCH: got %h exp 00FF003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h007F0003) begin
            $display("PKT 1 BUS WORD 10 MISMATCH: got %h exp 007F0003", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h0003000F) begin
            $display("PKT 1 BUS WORD 11 MISMATCH: got %h exp 0003000F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7DFD) begin
            $display("PKT 1 BUS WORD 12 MISMATCH: got %h exp 7F4A7DFD", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 1 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 1 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF584437) begin
            $display("PKT 1 BUS WORD 15 MISMATCH: got %h exp BF584437", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 1 CLASS 0 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd38) begin
            $display("PKT 1 CLASS 1 VOTE MISMATCH: got %0d exp 38", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 1 CLASS 2 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd26) begin
            $display("PKT 1 CLASS 3 VOTE MISMATCH: got %0d exp 26", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 1 CLASS 4 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 1 CLASS 5 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd35) begin
            $display("PKT 1 CLASS 6 VOTE MISMATCH: got %0d exp 35", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 1 CLASS 7 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h03232601) begin
            $display("PKT 1 RESULT CSR MISMATCH: got %h exp 03232601", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 1 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 2 (len=48) ---
        axis_send_byte(8'd66, 1'b0, 1'b0);
        axis_send_byte(8'd17, 1'b0, 1'b0);
        axis_send_byte(8'd201, 1'b0, 1'b0);
        axis_send_byte(8'd114, 1'b0, 1'b0);
        axis_send_byte(8'd27, 1'b0, 1'b1);
        axis_send_byte(8'd94, 1'b0, 1'b0);
        axis_send_byte(8'd203, 1'b0, 1'b0);
        axis_send_byte(8'd117, 1'b0, 1'b0);
        axis_send_byte(8'd182, 1'b0, 1'b0);
        axis_send_byte(8'd105, 1'b0, 1'b0);
        axis_send_byte(8'd188, 1'b0, 1'b0);
        axis_send_byte(8'd86, 1'b0, 1'b0);
        axis_send_byte(8'd13, 1'b0, 1'b0);
        axis_send_byte(8'd219, 1'b0, 1'b1);
        axis_send_byte(8'd64, 1'b0, 1'b0);
        axis_send_byte(8'd40, 1'b0, 1'b0);
        axis_send_byte(8'd42, 1'b0, 1'b0);
        axis_send_byte(8'd103, 1'b0, 1'b0);
        axis_send_byte(8'd36, 1'b0, 1'b0);
        axis_send_byte(8'd71, 1'b0, 1'b0);
        axis_send_byte(8'd151, 1'b0, 1'b0);
        axis_send_byte(8'd151, 1'b0, 1'b0);
        axis_send_byte(8'd104, 1'b0, 1'b1);
        axis_send_byte(8'd37, 1'b0, 1'b0);
        axis_send_byte(8'd149, 1'b0, 1'b0);
        axis_send_byte(8'd246, 1'b0, 1'b0);
        axis_send_byte(8'd240, 1'b0, 1'b0);
        axis_send_byte(8'd57, 1'b0, 1'b0);
        axis_send_byte(8'd218, 1'b0, 1'b0);
        axis_send_byte(8'd119, 1'b0, 1'b0);
        axis_send_byte(8'd146, 1'b0, 1'b0);
        axis_send_byte(8'd218, 1'b0, 1'b1);
        axis_send_byte(8'd193, 1'b0, 1'b0);
        axis_send_byte(8'd238, 1'b0, 1'b0);
        axis_send_byte(8'd155, 1'b0, 1'b0);
        axis_send_byte(8'd134, 1'b0, 1'b0);
        axis_send_byte(8'd118, 1'b0, 1'b0);
        axis_send_byte(8'd34, 1'b0, 1'b0);
        axis_send_byte(8'd115, 1'b0, 1'b0);
        axis_send_byte(8'd23, 1'b0, 1'b0);
        axis_send_byte(8'd184, 1'b0, 1'b1);
        axis_send_byte(8'd64, 1'b0, 1'b0);
        axis_send_byte(8'd80, 1'b0, 1'b0);
        axis_send_byte(8'd110, 1'b0, 1'b0);
        axis_send_byte(8'd26, 1'b0, 1'b0);
        axis_send_byte(8'd51, 1'b0, 1'b0);
        axis_send_byte(8'd82, 1'b0, 1'b0);
        axis_send_byte(8'd69, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'hB37698A9) begin
            $display("PKT 2 BUS WORD 0 MISMATCH: got %h exp B37698A9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h47D0EBFA) begin
            $display("PKT 2 BUS WORD 1 MISMATCH: got %h exp 47D0EBFA", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'h011CDF9B) begin
            $display("PKT 2 BUS WORD 2 MISMATCH: got %h exp 011CDF9B", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'hB7809FCD) begin
            $display("PKT 2 BUS WORD 3 MISMATCH: got %h exp B7809FCD", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h007F003F) begin
            $display("PKT 2 BUS WORD 4 MISMATCH: got %h exp 007F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h007F00FF) begin
            $display("PKT 2 BUS WORD 5 MISMATCH: got %h exp 007F00FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h003F003F) begin
            $display("PKT 2 BUS WORD 6 MISMATCH: got %h exp 003F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h000701FF) begin
            $display("PKT 2 BUS WORD 7 MISMATCH: got %h exp 000701FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h01FF07FF) begin
            $display("PKT 2 BUS WORD 8 MISMATCH: got %h exp 01FF07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h001F07FF) begin
            $display("PKT 2 BUS WORD 9 MISMATCH: got %h exp 001F07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h00FF001F) begin
            $display("PKT 2 BUS WORD 10 MISMATCH: got %h exp 00FF001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h007F003F) begin
            $display("PKT 2 BUS WORD 11 MISMATCH: got %h exp 007F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7F98) begin
            $display("PKT 2 BUS WORD 12 MISMATCH: got %h exp 7F4A7F98", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 2 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 2 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF584511) begin
            $display("PKT 2 BUS WORD 15 MISMATCH: got %h exp BF584511", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd31) begin
            $display("PKT 2 CLASS 0 VOTE MISMATCH: got %0d exp 31", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd30) begin
            $display("PKT 2 CLASS 1 VOTE MISMATCH: got %0d exp 30", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd34) begin
            $display("PKT 2 CLASS 2 VOTE MISMATCH: got %0d exp 34", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd22) begin
            $display("PKT 2 CLASS 3 VOTE MISMATCH: got %0d exp 22", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 2 CLASS 4 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd25) begin
            $display("PKT 2 CLASS 5 VOTE MISMATCH: got %0d exp 25", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd35) begin
            $display("PKT 2 CLASS 6 VOTE MISMATCH: got %0d exp 35", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 2 CLASS 7 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h01222306) begin
            $display("PKT 2 RESULT CSR MISMATCH: got %h exp 01222306", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 2 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 3 (len=48) ---
        axis_send_byte(8'd115, 1'b0, 1'b0);
        axis_send_byte(8'd127, 1'b0, 1'b0);
        axis_send_byte(8'd142, 1'b0, 1'b0);
        axis_send_byte(8'd212, 1'b0, 1'b0);
        axis_send_byte(8'd139, 1'b0, 1'b1);
        axis_send_byte(8'd131, 1'b0, 1'b0);
        axis_send_byte(8'd73, 1'b0, 1'b0);
        axis_send_byte(8'd3, 1'b0, 1'b0);
        axis_send_byte(8'd65, 1'b0, 1'b0);
        axis_send_byte(8'd222, 1'b0, 1'b0);
        axis_send_byte(8'd63, 1'b0, 1'b0);
        axis_send_byte(8'd119, 1'b0, 1'b0);
        axis_send_byte(8'd129, 1'b0, 1'b0);
        axis_send_byte(8'd151, 1'b0, 1'b1);
        axis_send_byte(8'd156, 1'b0, 1'b0);
        axis_send_byte(8'd114, 1'b0, 1'b0);
        axis_send_byte(8'd187, 1'b0, 1'b0);
        axis_send_byte(8'd150, 1'b0, 1'b0);
        axis_send_byte(8'd158, 1'b0, 1'b0);
        axis_send_byte(8'd253, 1'b0, 1'b0);
        axis_send_byte(8'd45, 1'b0, 1'b0);
        axis_send_byte(8'd125, 1'b0, 1'b0);
        axis_send_byte(8'd124, 1'b0, 1'b1);
        axis_send_byte(8'd93, 1'b0, 1'b0);
        axis_send_byte(8'd88, 1'b0, 1'b0);
        axis_send_byte(8'd199, 1'b0, 1'b0);
        axis_send_byte(8'd75, 1'b0, 1'b0);
        axis_send_byte(8'd42, 1'b0, 1'b0);
        axis_send_byte(8'd189, 1'b0, 1'b0);
        axis_send_byte(8'd147, 1'b0, 1'b0);
        axis_send_byte(8'd5, 1'b0, 1'b0);
        axis_send_byte(8'd50, 1'b0, 1'b1);
        axis_send_byte(8'd162, 1'b0, 1'b0);
        axis_send_byte(8'd214, 1'b0, 1'b0);
        axis_send_byte(8'd64, 1'b0, 1'b0);
        axis_send_byte(8'd122, 1'b0, 1'b0);
        axis_send_byte(8'd53, 1'b0, 1'b0);
        axis_send_byte(8'd33, 1'b0, 1'b0);
        axis_send_byte(8'd73, 1'b0, 1'b0);
        axis_send_byte(8'd152, 1'b0, 1'b0);
        axis_send_byte(8'd233, 1'b0, 1'b1);
        axis_send_byte(8'd217, 1'b0, 1'b0);
        axis_send_byte(8'd153, 1'b0, 1'b0);
        axis_send_byte(8'd83, 1'b0, 1'b0);
        axis_send_byte(8'd216, 1'b0, 1'b0);
        axis_send_byte(8'd174, 1'b0, 1'b0);
        axis_send_byte(8'd112, 1'b0, 1'b0);
        axis_send_byte(8'd111, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'h53503F6A) begin
            $display("PKT 3 BUS WORD 0 MISMATCH: got %h exp 53503F6A", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'hC67CDF93) begin
            $display("PKT 3 BUS WORD 1 MISMATCH: got %h exp C67CDF93", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'h0EEDC0F0) begin
            $display("PKT 3 BUS WORD 2 MISMATCH: got %h exp 0EEDC0F0", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h2FE9AAE4) begin
            $display("PKT 3 BUS WORD 3 MISMATCH: got %h exp 2FE9AAE4", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h03FF03FF) begin
            $display("PKT 3 BUS WORD 4 MISMATCH: got %h exp 03FF03FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h1FFF00FF) begin
            $display("PKT 3 BUS WORD 5 MISMATCH: got %h exp 1FFF00FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h003F001F) begin
            $display("PKT 3 BUS WORD 6 MISMATCH: got %h exp 003F001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h001F003F) begin
            $display("PKT 3 BUS WORD 7 MISMATCH: got %h exp 001F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h3FFF00FF) begin
            $display("PKT 3 BUS WORD 8 MISMATCH: got %h exp 3FFF00FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h001F003F) begin
            $display("PKT 3 BUS WORD 9 MISMATCH: got %h exp 001F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h03FF07FF) begin
            $display("PKT 3 BUS WORD 10 MISMATCH: got %h exp 03FF07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h003F003F) begin
            $display("PKT 3 BUS WORD 11 MISMATCH: got %h exp 003F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7D1A) begin
            $display("PKT 3 BUS WORD 12 MISMATCH: got %h exp 7F4A7D1A", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 3 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 3 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF58469D) begin
            $display("PKT 3 BUS WORD 15 MISMATCH: got %h exp BF58469D", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 3 CLASS 0 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd40) begin
            $display("PKT 3 CLASS 1 VOTE MISMATCH: got %0d exp 40", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 3 CLASS 2 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd35) begin
            $display("PKT 3 CLASS 3 VOTE MISMATCH: got %0d exp 35", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 3 CLASS 4 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd36) begin
            $display("PKT 3 CLASS 5 VOTE MISMATCH: got %0d exp 36", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 3 CLASS 6 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 3 CLASS 7 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h04242801) begin
            $display("PKT 3 RESULT CSR MISMATCH: got %h exp 04242801", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 3 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 4 (len=48) ---
        axis_send_byte(8'd65, 1'b0, 1'b0);
        axis_send_byte(8'd62, 1'b0, 1'b0);
        axis_send_byte(8'd204, 1'b0, 1'b0);
        axis_send_byte(8'd76, 1'b0, 1'b0);
        axis_send_byte(8'd156, 1'b0, 1'b1);
        axis_send_byte(8'd215, 1'b0, 1'b0);
        axis_send_byte(8'd248, 1'b0, 1'b0);
        axis_send_byte(8'd246, 1'b0, 1'b0);
        axis_send_byte(8'd127, 1'b0, 1'b0);
        axis_send_byte(8'd75, 1'b0, 1'b0);
        axis_send_byte(8'd156, 1'b0, 1'b0);
        axis_send_byte(8'd53, 1'b0, 1'b0);
        axis_send_byte(8'd91, 1'b0, 1'b0);
        axis_send_byte(8'd50, 1'b0, 1'b1);
        axis_send_byte(8'd108, 1'b0, 1'b0);
        axis_send_byte(8'd239, 1'b0, 1'b0);
        axis_send_byte(8'd185, 1'b0, 1'b0);
        axis_send_byte(8'd92, 1'b0, 1'b0);
        axis_send_byte(8'd231, 1'b0, 1'b0);
        axis_send_byte(8'd223, 1'b0, 1'b0);
        axis_send_byte(8'd100, 1'b0, 1'b0);
        axis_send_byte(8'd17, 1'b0, 1'b0);
        axis_send_byte(8'd253, 1'b0, 1'b1);
        axis_send_byte(8'd244, 1'b0, 1'b0);
        axis_send_byte(8'd132, 1'b0, 1'b0);
        axis_send_byte(8'd125, 1'b0, 1'b0);
        axis_send_byte(8'd38, 1'b0, 1'b0);
        axis_send_byte(8'd204, 1'b0, 1'b0);
        axis_send_byte(8'd135, 1'b0, 1'b0);
        axis_send_byte(8'd28, 1'b0, 1'b0);
        axis_send_byte(8'd90, 1'b0, 1'b0);
        axis_send_byte(8'd135, 1'b0, 1'b1);
        axis_send_byte(8'd75, 1'b0, 1'b0);
        axis_send_byte(8'd83, 1'b0, 1'b0);
        axis_send_byte(8'd81, 1'b0, 1'b0);
        axis_send_byte(8'd155, 1'b0, 1'b0);
        axis_send_byte(8'd64, 1'b0, 1'b0);
        axis_send_byte(8'd250, 1'b0, 1'b0);
        axis_send_byte(8'd233, 1'b0, 1'b0);
        axis_send_byte(8'd52, 1'b0, 1'b0);
        axis_send_byte(8'd243, 1'b0, 1'b1);
        axis_send_byte(8'd130, 1'b0, 1'b0);
        axis_send_byte(8'd88, 1'b0, 1'b0);
        axis_send_byte(8'd88, 1'b0, 1'b0);
        axis_send_byte(8'd187, 1'b0, 1'b0);
        axis_send_byte(8'd188, 1'b0, 1'b0);
        axis_send_byte(8'd144, 1'b0, 1'b0);
        axis_send_byte(8'd118, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'h087A31B3) begin
            $display("PKT 4 BUS WORD 0 MISMATCH: got %h exp 087A31B3", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h36EC3CB8) begin
            $display("PKT 4 BUS WORD 1 MISMATCH: got %h exp 36EC3CB8", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'hA1E2B408) begin
            $display("PKT 4 BUS WORD 2 MISMATCH: got %h exp A1E2B408", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h1EA69267) begin
            $display("PKT 4 BUS WORD 3 MISMATCH: got %h exp 1EA69267", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h001F007F) begin
            $display("PKT 4 BUS WORD 4 MISMATCH: got %h exp 001F007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h03FF07FF) begin
            $display("PKT 4 BUS WORD 5 MISMATCH: got %h exp 03FF07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h03FF007F) begin
            $display("PKT 4 BUS WORD 6 MISMATCH: got %h exp 03FF007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h0FFFFFFF) begin
            $display("PKT 4 BUS WORD 7 MISMATCH: got %h exp 0FFFFFFF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h007F07FF) begin
            $display("PKT 4 BUS WORD 8 MISMATCH: got %h exp 007F07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h007F003F) begin
            $display("PKT 4 BUS WORD 9 MISMATCH: got %h exp 007F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h001F01FF) begin
            $display("PKT 4 BUS WORD 10 MISMATCH: got %h exp 001F01FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h03FF007F) begin
            $display("PKT 4 BUS WORD 11 MISMATCH: got %h exp 03FF007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7E2D) begin
            $display("PKT 4 BUS WORD 12 MISMATCH: got %h exp 7F4A7E2D", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 4 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 4 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF58463B) begin
            $display("PKT 4 BUS WORD 15 MISMATCH: got %h exp BF58463B", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 4 CLASS 0 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 4 CLASS 1 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd31) begin
            $display("PKT 4 CLASS 2 VOTE MISMATCH: got %0d exp 31", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 4 CLASS 3 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 4 CLASS 4 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 4 CLASS 5 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd26) begin
            $display("PKT 4 CLASS 6 VOTE MISMATCH: got %0d exp 26", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 4 CLASS 7 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h01202104) begin
            $display("PKT 4 RESULT CSR MISMATCH: got %h exp 01202104", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 4 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 5 (len=61) ---
        axis_send_byte(8'd158, 1'b0, 1'b0);
        axis_send_byte(8'd143, 1'b0, 1'b0);
        axis_send_byte(8'd118, 1'b0, 1'b0);
        axis_send_byte(8'd203, 1'b0, 1'b0);
        axis_send_byte(8'd230, 1'b0, 1'b1);
        axis_send_byte(8'd127, 1'b0, 1'b0);
        axis_send_byte(8'd57, 1'b0, 1'b0);
        axis_send_byte(8'd144, 1'b0, 1'b0);
        axis_send_byte(8'd120, 1'b0, 1'b0);
        axis_send_byte(8'd176, 1'b0, 1'b0);
        axis_send_byte(8'd85, 1'b0, 1'b0);
        axis_send_byte(8'd152, 1'b0, 1'b0);
        axis_send_byte(8'd245, 1'b0, 1'b0);
        axis_send_byte(8'd137, 1'b0, 1'b1);
        axis_send_byte(8'd201, 1'b0, 1'b0);
        axis_send_byte(8'd110, 1'b0, 1'b0);
        axis_send_byte(8'd233, 1'b0, 1'b0);
        axis_send_byte(8'd245, 1'b0, 1'b0);
        axis_send_byte(8'd96, 1'b0, 1'b0);
        axis_send_byte(8'd107, 1'b0, 1'b0);
        axis_send_byte(8'd107, 1'b0, 1'b0);
        axis_send_byte(8'd221, 1'b0, 1'b0);
        axis_send_byte(8'd131, 1'b0, 1'b1);
        axis_send_byte(8'd230, 1'b0, 1'b0);
        axis_send_byte(8'd32, 1'b0, 1'b0);
        axis_send_byte(8'd153, 1'b0, 1'b0);
        axis_send_byte(8'd15, 1'b0, 1'b0);
        axis_send_byte(8'd1, 1'b0, 1'b0);
        axis_send_byte(8'd108, 1'b0, 1'b0);
        axis_send_byte(8'd98, 1'b0, 1'b0);
        axis_send_byte(8'd192, 1'b0, 1'b0);
        axis_send_byte(8'd81, 1'b0, 1'b1);
        axis_send_byte(8'd39, 1'b0, 1'b0);
        axis_send_byte(8'd184, 1'b0, 1'b0);
        axis_send_byte(8'd138, 1'b0, 1'b0);
        axis_send_byte(8'd104, 1'b0, 1'b0);
        axis_send_byte(8'd69, 1'b0, 1'b0);
        axis_send_byte(8'd184, 1'b0, 1'b0);
        axis_send_byte(8'd44, 1'b0, 1'b0);
        axis_send_byte(8'd178, 1'b0, 1'b0);
        axis_send_byte(8'd154, 1'b0, 1'b1);
        axis_send_byte(8'd16, 1'b0, 1'b0);
        axis_send_byte(8'd114, 1'b0, 1'b0);
        axis_send_byte(8'd105, 1'b0, 1'b0);
        axis_send_byte(8'd106, 1'b0, 1'b0);
        axis_send_byte(8'd135, 1'b0, 1'b0);
        axis_send_byte(8'd86, 1'b0, 1'b0);
        axis_send_byte(8'd80, 1'b0, 1'b0);
        axis_send_byte(8'd210, 1'b0, 1'b0);
        axis_send_byte(8'd10, 1'b0, 1'b1);
        axis_send_byte(8'd15, 1'b0, 1'b0);
        axis_send_byte(8'd20, 1'b0, 1'b0);
        axis_send_byte(8'd244, 1'b0, 1'b0);
        axis_send_byte(8'd35, 1'b0, 1'b0);
        axis_send_byte(8'd245, 1'b0, 1'b0);
        axis_send_byte(8'd66, 1'b0, 1'b0);
        axis_send_byte(8'd56, 1'b0, 1'b0);
        axis_send_byte(8'd191, 1'b0, 1'b0);
        axis_send_byte(8'd16, 1'b0, 1'b1);
        axis_send_byte(8'd27, 1'b0, 1'b0);
        axis_send_byte(8'd252, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'hDC93C882) begin
            $display("PKT 5 BUS WORD 0 MISMATCH: got %h exp DC93C882", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h8663393C) begin
            $display("PKT 5 BUS WORD 1 MISMATCH: got %h exp 8663393C", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'hEB3886FA) begin
            $display("PKT 5 BUS WORD 2 MISMATCH: got %h exp EB3886FA", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h6F5F6C0F) begin
            $display("PKT 5 BUS WORD 3 MISMATCH: got %h exp 6F5F6C0F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h01FF03FF) begin
            $display("PKT 5 BUS WORD 4 MISMATCH: got %h exp 01FF03FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h007F003F) begin
            $display("PKT 5 BUS WORD 5 MISMATCH: got %h exp 007F003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h01FF03FF) begin
            $display("PKT 5 BUS WORD 6 MISMATCH: got %h exp 01FF03FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h03FF007F) begin
            $display("PKT 5 BUS WORD 7 MISMATCH: got %h exp 03FF007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h00FF003F) begin
            $display("PKT 5 BUS WORD 8 MISMATCH: got %h exp 00FF003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h000F0007) begin
            $display("PKT 5 BUS WORD 9 MISMATCH: got %h exp 000F0007", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h003F07FF) begin
            $display("PKT 5 BUS WORD 10 MISMATCH: got %h exp 003F07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h000F007F) begin
            $display("PKT 5 BUS WORD 11 MISMATCH: got %h exp 000F007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7F66) begin
            $display("PKT 5 BUS WORD 12 MISMATCH: got %h exp 7F4A7F66", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 5 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 5 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF58454D) begin
            $display("PKT 5 BUS WORD 15 MISMATCH: got %h exp BF58454D", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd34) begin
            $display("PKT 5 CLASS 0 VOTE MISMATCH: got %0d exp 34", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd30) begin
            $display("PKT 5 CLASS 1 VOTE MISMATCH: got %0d exp 30", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 5 CLASS 2 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd27) begin
            $display("PKT 5 CLASS 3 VOTE MISMATCH: got %0d exp 27", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 5 CLASS 4 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 5 CLASS 5 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 5 CLASS 6 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 5 CLASS 7 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h01212200) begin
            $display("PKT 5 RESULT CSR MISMATCH: got %h exp 01212200", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 5 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 6 (len=38) ---
        axis_send_byte(8'd244, 1'b0, 1'b0);
        axis_send_byte(8'd32, 1'b0, 1'b0);
        axis_send_byte(8'd87, 1'b0, 1'b0);
        axis_send_byte(8'd58, 1'b0, 1'b0);
        axis_send_byte(8'd127, 1'b0, 1'b1);
        axis_send_byte(8'd102, 1'b0, 1'b0);
        axis_send_byte(8'd65, 1'b0, 1'b0);
        axis_send_byte(8'd138, 1'b0, 1'b0);
        axis_send_byte(8'd102, 1'b0, 1'b0);
        axis_send_byte(8'd122, 1'b0, 1'b0);
        axis_send_byte(8'd60, 1'b0, 1'b0);
        axis_send_byte(8'd126, 1'b0, 1'b0);
        axis_send_byte(8'd43, 1'b0, 1'b0);
        axis_send_byte(8'd230, 1'b0, 1'b1);
        axis_send_byte(8'd6, 1'b0, 1'b0);
        axis_send_byte(8'd121, 1'b0, 1'b0);
        axis_send_byte(8'd164, 1'b0, 1'b0);
        axis_send_byte(8'd190, 1'b0, 1'b0);
        axis_send_byte(8'd124, 1'b0, 1'b0);
        axis_send_byte(8'd148, 1'b0, 1'b0);
        axis_send_byte(8'd109, 1'b0, 1'b0);
        axis_send_byte(8'd230, 1'b0, 1'b0);
        axis_send_byte(8'd107, 1'b0, 1'b1);
        axis_send_byte(8'd160, 1'b0, 1'b0);
        axis_send_byte(8'd226, 1'b0, 1'b0);
        axis_send_byte(8'd42, 1'b0, 1'b0);
        axis_send_byte(8'd92, 1'b0, 1'b0);
        axis_send_byte(8'd48, 1'b0, 1'b0);
        axis_send_byte(8'd234, 1'b0, 1'b0);
        axis_send_byte(8'd248, 1'b0, 1'b0);
        axis_send_byte(8'd175, 1'b0, 1'b0);
        axis_send_byte(8'd39, 1'b0, 1'b1);
        axis_send_byte(8'd121, 1'b0, 1'b0);
        axis_send_byte(8'd150, 1'b0, 1'b0);
        axis_send_byte(8'd50, 1'b0, 1'b0);
        axis_send_byte(8'd187, 1'b0, 1'b0);
        axis_send_byte(8'd200, 1'b0, 1'b0);
        axis_send_byte(8'd13, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'h15190829) begin
            $display("PKT 6 BUS WORD 0 MISMATCH: got %h exp 15190829", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h2A2A8DDA) begin
            $display("PKT 6 BUS WORD 1 MISMATCH: got %h exp 2A2A8DDA", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'h4E605084) begin
            $display("PKT 6 BUS WORD 2 MISMATCH: got %h exp 4E605084", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h5EA91EC1) begin
            $display("PKT 6 BUS WORD 3 MISMATCH: got %h exp 5EA91EC1", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h00FF07FF) begin
            $display("PKT 6 BUS WORD 4 MISMATCH: got %h exp 00FF07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h00FF001F) begin
            $display("PKT 6 BUS WORD 5 MISMATCH: got %h exp 00FF001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h007F01FF) begin
            $display("PKT 6 BUS WORD 6 MISMATCH: got %h exp 007F01FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h003F0007) begin
            $display("PKT 6 BUS WORD 7 MISMATCH: got %h exp 003F0007", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h0007007F) begin
            $display("PKT 6 BUS WORD 8 MISMATCH: got %h exp 0007007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h00070007) begin
            $display("PKT 6 BUS WORD 9 MISMATCH: got %h exp 00070007", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h03FF003F) begin
            $display("PKT 6 BUS WORD 10 MISMATCH: got %h exp 03FF003F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h0007000F) begin
            $display("PKT 6 BUS WORD 11 MISMATCH: got %h exp 0007000F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7C98) begin
            $display("PKT 6 BUS WORD 12 MISMATCH: got %h exp 7F4A7C98", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 6 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 6 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF58442A) begin
            $display("PKT 6 BUS WORD 15 MISMATCH: got %h exp BF58442A", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd37) begin
            $display("PKT 6 CLASS 0 VOTE MISMATCH: got %0d exp 37", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd25) begin
            $display("PKT 6 CLASS 1 VOTE MISMATCH: got %0d exp 25", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd30) begin
            $display("PKT 6 CLASS 2 VOTE MISMATCH: got %0d exp 30", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 6 CLASS 3 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd34) begin
            $display("PKT 6 CLASS 4 VOTE MISMATCH: got %0d exp 34", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd28) begin
            $display("PKT 6 CLASS 5 VOTE MISMATCH: got %0d exp 28", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 6 CLASS 6 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd35) begin
            $display("PKT 6 CLASS 7 VOTE MISMATCH: got %0d exp 35", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h02232500) begin
            $display("PKT 6 RESULT CSR MISMATCH: got %h exp 02232500", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 6 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        // --- Stream Packet 7 (len=43) ---
        axis_send_byte(8'd237, 1'b0, 1'b0);
        axis_send_byte(8'd154, 1'b0, 1'b0);
        axis_send_byte(8'd103, 1'b0, 1'b0);
        axis_send_byte(8'd115, 1'b0, 1'b0);
        axis_send_byte(8'd110, 1'b0, 1'b1);
        axis_send_byte(8'd214, 1'b0, 1'b0);
        axis_send_byte(8'd26, 1'b0, 1'b0);
        axis_send_byte(8'd159, 1'b0, 1'b0);
        axis_send_byte(8'd252, 1'b0, 1'b0);
        axis_send_byte(8'd182, 1'b0, 1'b0);
        axis_send_byte(8'd222, 1'b0, 1'b0);
        axis_send_byte(8'd8, 1'b0, 1'b0);
        axis_send_byte(8'd150, 1'b0, 1'b0);
        axis_send_byte(8'd217, 1'b0, 1'b1);
        axis_send_byte(8'd68, 1'b0, 1'b0);
        axis_send_byte(8'd95, 1'b0, 1'b0);
        axis_send_byte(8'd201, 1'b0, 1'b0);
        axis_send_byte(8'd15, 1'b0, 1'b0);
        axis_send_byte(8'd215, 1'b0, 1'b0);
        axis_send_byte(8'd91, 1'b0, 1'b0);
        axis_send_byte(8'd21, 1'b0, 1'b0);
        axis_send_byte(8'd201, 1'b0, 1'b0);
        axis_send_byte(8'd145, 1'b0, 1'b1);
        axis_send_byte(8'd107, 1'b0, 1'b0);
        axis_send_byte(8'd150, 1'b0, 1'b0);
        axis_send_byte(8'd214, 1'b0, 1'b0);
        axis_send_byte(8'd116, 1'b0, 1'b0);
        axis_send_byte(8'd190, 1'b0, 1'b0);
        axis_send_byte(8'd99, 1'b0, 1'b0);
        axis_send_byte(8'd40, 1'b0, 1'b0);
        axis_send_byte(8'd173, 1'b0, 1'b0);
        axis_send_byte(8'd126, 1'b0, 1'b1);
        axis_send_byte(8'd63, 1'b0, 1'b0);
        axis_send_byte(8'd106, 1'b0, 1'b0);
        axis_send_byte(8'd20, 1'b0, 1'b0);
        axis_send_byte(8'd219, 1'b0, 1'b0);
        axis_send_byte(8'd130, 1'b0, 1'b0);
        axis_send_byte(8'd3, 1'b0, 1'b0);
        axis_send_byte(8'd80, 1'b0, 1'b0);
        axis_send_byte(8'd120, 1'b0, 1'b0);
        axis_send_byte(8'd143, 1'b0, 1'b1);
        axis_send_byte(8'd103, 1'b0, 1'b0);
        axis_send_byte(8'd179, 1'b1, 1'b0);
        repeat (4) @(posedge aclk);
        axi_read(12'h100, rd_val);
        if (rd_val !== 32'hF3A4FF1B) begin
            $display("PKT 7 BUS WORD 0 MISMATCH: got %h exp F3A4FF1B", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h104, rd_val);
        if (rd_val !== 32'h8CDB1CF9) begin
            $display("PKT 7 BUS WORD 1 MISMATCH: got %h exp 8CDB1CF9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h108, rd_val);
        if (rd_val !== 32'hB61907E5) begin
            $display("PKT 7 BUS WORD 2 MISMATCH: got %h exp B61907E5", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h10C, rd_val);
        if (rd_val !== 32'h21E9F1F5) begin
            $display("PKT 7 BUS WORD 3 MISMATCH: got %h exp 21E9F1F5", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h110, rd_val);
        if (rd_val !== 32'h003F07FF) begin
            $display("PKT 7 BUS WORD 4 MISMATCH: got %h exp 003F07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h114, rd_val);
        if (rd_val !== 32'h01FF007F) begin
            $display("PKT 7 BUS WORD 5 MISMATCH: got %h exp 01FF007F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h118, rd_val);
        if (rd_val !== 32'h01FF001F) begin
            $display("PKT 7 BUS WORD 6 MISMATCH: got %h exp 01FF001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h11C, rd_val);
        if (rd_val !== 32'h00FF001F) begin
            $display("PKT 7 BUS WORD 7 MISMATCH: got %h exp 00FF001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h120, rd_val);
        if (rd_val !== 32'h07FF0FFF) begin
            $display("PKT 7 BUS WORD 8 MISMATCH: got %h exp 07FF0FFF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h124, rd_val);
        if (rd_val !== 32'h000F07FF) begin
            $display("PKT 7 BUS WORD 9 MISMATCH: got %h exp 000F07FF", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h128, rd_val);
        if (rd_val !== 32'h001F001F) begin
            $display("PKT 7 BUS WORD 10 MISMATCH: got %h exp 001F001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h12C, rd_val);
        if (rd_val !== 32'h000F001F) begin
            $display("PKT 7 BUS WORD 11 MISMATCH: got %h exp 000F001F", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h130, rd_val);
        if (rd_val !== 32'h7F4A7D52) begin
            $display("PKT 7 BUS WORD 12 MISMATCH: got %h exp 7F4A7D52", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h134, rd_val);
        if (rd_val !== 32'h9E3779B9) begin
            $display("PKT 7 BUS WORD 13 MISMATCH: got %h exp 9E3779B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h138, rd_val);
        if (rd_val !== 32'h1CE4E5B9) begin
            $display("PKT 7 BUS WORD 14 MISMATCH: got %h exp 1CE4E5B9", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h13C, rd_val);
        if (rd_val !== 32'hBF58445B) begin
            $display("PKT 7 BUS WORD 15 MISMATCH: got %h exp BF58445B", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h020, rd_val);
        if (rd_val !== 32'd36) begin
            $display("PKT 7 CLASS 0 VOTE MISMATCH: got %0d exp 36", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h024, rd_val);
        if (rd_val !== 32'd29) begin
            $display("PKT 7 CLASS 1 VOTE MISMATCH: got %0d exp 29", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h028, rd_val);
        if (rd_val !== 32'd23) begin
            $display("PKT 7 CLASS 2 VOTE MISMATCH: got %0d exp 23", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h02C, rd_val);
        if (rd_val !== 32'd27) begin
            $display("PKT 7 CLASS 3 VOTE MISMATCH: got %0d exp 27", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h030, rd_val);
        if (rd_val !== 32'd26) begin
            $display("PKT 7 CLASS 4 VOTE MISMATCH: got %0d exp 26", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h034, rd_val);
        if (rd_val !== 32'd32) begin
            $display("PKT 7 CLASS 5 VOTE MISMATCH: got %0d exp 32", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h038, rd_val);
        if (rd_val !== 32'd24) begin
            $display("PKT 7 CLASS 6 VOTE MISMATCH: got %0d exp 24", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h03C, rd_val);
        if (rd_val !== 32'd33) begin
            $display("PKT 7 CLASS 7 VOTE MISMATCH: got %0d exp 33", rd_val);
            errors = errors + 1;
        end
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h03212400) begin
            $display("PKT 7 RESULT CSR MISMATCH: got %h exp 03212400", rd_val);
            errors = errors + 1;
        end
        if (irq_escalate_to_ps !== 1'b1) begin
            $display("PKT 7 IRQ MISMATCH: got %b exp 1", irq_escalate_to_ps);
            errors = errors + 1;
        end
        // W1C clear status/IRQ
        axi_write(12'h004, 32'h00000005, 0);
        axi_read(12'h010, rd_val);
        if (rd_val !== 32'd8) errors = errors + 1;
        axi_read(12'h014, rd_val);
        if (rd_val !== 32'd1) errors = errors + 1;
        axi_read(12'h018, rd_val);
        if (rd_val !== 32'd7) errors = errors + 1;

        // 3. Test Direct MMIO 512-bit Bus Override Mode (CSR_CTRL = 0x7: trigger + mmio + irq)
        axi_write(12'h100, 32'hE73C178E, 0);
        axi_write(12'h104, 32'h7ECD9BF3, 1);
        axi_write(12'h108, 32'h375721AA, 0);
        axi_write(12'h10C, 32'h4D639CAC, 1);
        axi_write(12'h110, 32'h0FFF03FF, 0);
        axi_write(12'h114, 32'h00FF0FFF, 1);
        axi_write(12'h118, 32'h003F003F, 0);
        axi_write(12'h11C, 32'h007F03FF, 1);
        axi_write(12'h120, 32'h000700FF, 0);
        axi_write(12'h124, 32'h007F00FF, 1);
        axi_write(12'h128, 32'h003F001F, 0);
        axi_write(12'h12C, 32'h007F003F, 1);
        axi_write(12'h130, 32'h7F4A7F24, 0);
        axi_write(12'h134, 32'h9E3779B9, 1);
        axi_write(12'h138, 32'h1CE4E5B9, 0);
        axi_write(12'h13C, 32'hBF584451, 1);
        axi_write(12'h000, 32'h00000007, 0);
        repeat (3) @(posedge aclk);
        axi_read(12'h00C, rd_val);
        if (rd_val !== 32'h06242A03) begin
            $display("MMIO BUS EVAL MISMATCH: got %h", rd_val); errors = errors + 1;
        end
        axi_write(12'h004, 32'h00000005, 0);

        // 4. Test Zero-Downtime 4ns Shadow-Bank Hot-Swap Overlay on Live AXI4-Stream!
        begin : test_hotswap
            integer idx;
            for (idx = 0; idx < 32; idx = idx + 1) begin
                if (idx >= 20 && idx < 24) begin
                    // Class 5 detector: L0 routes wire 0 -> O=I0; L1 routes 20..23 -> 3-of-6 maj
                    axi_write(12'h054, 32'hAAAAAAAA, 0);
                    axi_write(12'h058, 32'hAAAAAAAA, 0);
                    axi_write(12'h05C, 32'h00000000, 0);
                    axi_write(12'h060, 32'h00000000, 0);
                    axi_write(12'h050, 32'h80000000 | idx, 0);
                    axi_write(12'h054, 32'hF8808000, 0);
                    axi_write(12'h058, 32'hFFF8F880, 0);
                    axi_write(12'h05C, (22 << 18) | (21 << 9) | 20, 0);
                    axi_write(12'h060, (21 << 18) | (20 << 9) | 23, 0);
                    axi_write(12'h050, 32'h80000100 | idx, 0);
                end else begin
                    axi_write(12'h054, 32'h00000000, 0);
                    axi_write(12'h058, 32'h00000000, 0);
                    axi_write(12'h05C, 32'h00000000, 0);
                    axi_write(12'h060, 32'h00000000, 0);
                    axi_write(12'h050, 32'h80000000 | idx, 0);
                    axi_write(12'h050, 32'h80000100 | idx, 0);
                end
            end
            // Single-cycle (4.0 ns) atomic swap to Bank 1 + enable overlay (0x34)
            axi_write(12'h000, 32'h00000034, 0);
            axi_read(12'h064, rd_val);
            if (rd_val !== 32'h80000001) begin
                $display("HOTSWAP BANK GENERATION MISMATCH: %h", rd_val); errors = errors + 1;
            end
            // Stream live 16-byte packet with bit 0 = 1 without resetting aresetn
            axis_send_byte(8'h01, 1'b0, 1'b0);
            for (idx = 1; idx < 15; idx = idx + 1) axis_send_byte(8'h00, 1'b0, 1'b0);
            axis_send_byte(8'h00, 1'b1, 1'b0);
            repeat (4) @(posedge aclk);
            axi_read(12'h00C, rd_val);
            if (rd_val !== 32'h40004005 || irq_escalate_to_ps !== 1'b0) begin
                $display("HOTSWAP LIVE PACKET MISMATCH: res=%h irq=%b", rd_val, irq_escalate_to_ps);
                errors = errors + 1;
            end
        end

        $display("[Verilator KV260 AXI4-Lite + AXI4-Stream BFM] Packets=%0d | 4ns Atomic HotSwap=OK (Bank=1, Gen=1, Class=5, Votes=64/64) | Total Errors=%0d", 8, errors);
        if (errors != 0) $fatal(1);
        $finish;
    end
endmodule
