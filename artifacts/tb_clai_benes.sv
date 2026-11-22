`timescale 1ns/1ps
module tb_clai_benes;
    logic clk = 0;
    logic rst_n = 0;
    always #2 clk = ~clk;
    logic [511:0] bus_in = 0;
    logic         cfg_we = 0;
    logic [1:0]   cfg_kind = 0;
    logic [6:0]   cfg_idx = 0;
    logic [255:0] cfg_vec256 = 0;
    logic [511:0] cfg_vec512 = 0;
    logic         atomic_swap_pulse = 0;
    wire          active_bank;
    wire  [79:0]  l0_out_debug;
    integer errors = 0;

    clai_benes_hotswap_core dut (.*);

    initial begin
        repeat (4) @(posedge clk); rst_n = 1; @(posedge clk);
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd0; cfg_vec256 <= 256'h201780001FF00740000000001000000007FF0000207F00F80000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd1; cfg_vec256 <= 256'h0300000800004000000000F240000000020003F00000008001FF00FE00000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd2; cfg_vec256 <= 256'h80020080003E8040000400000000000000000000000000002200000000F00A00;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd3; cfg_vec256 <= 256'h00000008000000004200000000000000400000000000700000000C0000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd4; cfg_vec256 <= 256'h700000000000000080000000000000009000000000C000004020000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd5; cfg_vec256 <= 256'h4080000000000000C000000000000008C0000000080000008000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd6; cfg_vec256 <= 256'h8000000000000000800000000000000080000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd7; cfg_vec256 <= 256'h0000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd8; cfg_vec256 <= 256'h0000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd9; cfg_vec256 <= 256'h8000000000000000800000000000000080000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd10; cfg_vec256 <= 256'h8080000000000000880000000000002088000000200000008000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd11; cfg_vec256 <= 256'h60C000040000000040000000000000084000000000000000C480000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd12; cfg_vec256 <= 256'h9C02001000000000D0000000000000A0F8000000008000107000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd13; cfg_vec256 <= 256'hBFE8078C000000087380000000005D0079C00000000043007E00003000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd14; cfg_vec256 <= 256'hEFFBFEC0002080701F07800073F300001FE3F0080F8F00003FFE000000000200;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd15; cfg_vec256 <= 256'h7C007FC01FF860001F07F80160007F0000080003FF0000800100BF8000E20000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd16; cfg_vec256 <= 256'h3FF000C04020000FF800FFC01C0020200000FF80F02040401800FC0C3FFF0000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd0; cfg_vec512 <= 512'h00000200000200002000100001000200004000020000100000100000200008000040000200004001000008004000200000400001000002000020000200000002;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd1; cfg_vec512 <= 512'h00000C00000C0000C000600006000C000180000C0000600000600000C00030000180000C00018006000030018000C0000180000600000C0000C0000C0000000C;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd2; cfg_vec512 <= 512'h0000F00000F0000F000780007800F0001E0000F0000780000780000F0003C0001E0000F0001E00780003C01E000F00001E0000780000F0000F0000F0000000F0;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd3; cfg_vec512 <= 512'h00FF0000FF000FF007F8007F807F001FE000FF0007F80007F8000FF003FC001FE000FF001FE01F8003FC03E00FF0001FE0007F8000FF000FF000FF000000FF00;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd4; cfg_vec512 <= 512'hFF0000FF0000F00008000780000000E0001F0000F80007F80007F0000C0003E0001F0000E00000007C00000010000FE0001F80007F0000F0000F0000FFFF0000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd5; cfg_vec512 <= 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd6; cfg_vec512 <= 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd7; cfg_vec512 <= 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd8; cfg_vec512 <= 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd0; cfg_vec256 <= 256'hE594481B5C7C510DE079F048EE8417AE0CFF8ED6BD00CB7273BC2A0438A20000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd1; cfg_vec256 <= 256'h1C9F9C0BCD2A681625250FF69C5AD400FDE8CE29C9D1E4CEB573C206099C6000;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd2; cfg_vec256 <= 256'h93FAD395046345208F395C7059F824806CDFFB9F5FC0820009AFE3439EA475A0;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd3; cfg_vec256 <= 256'h8909205075BF74B089A3EBE895582E6814BADC009EC2F0809B4ECAD0A72751D0;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd4; cfg_vec256 <= 256'hA3C0204035981E04B92839603C4467E882A8DC08B2F056C8886CC508EC40CFF8;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd5; cfg_vec256 <= 256'h3C0AB4106422E096045A4C1A500EAC201C7838A60CB4509820760C661C18564C;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd6; cfg_vec256 <= 256'h8E8228080AEA44CA0442C08CC0CA8480EC004244808EA0066460200CC86C2040;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd7; cfg_vec256 <= 256'h00A820200A8A00A020AAA82A000080A0082808000A0882000000008800008A82;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd8; cfg_vec256 <= 256'h2B69E96B0959B963A99965E91939556F671929E96579918D8D4F03AB6537B5A9;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd9; cfg_vec256 <= 256'h53D6CE336E9362076F035AFF625216C77F4266D29B4AAE4336BF479FD3DB6EE2;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd10; cfg_vec256 <= 256'hB9EFDFFF98FEA97EAE4BCF0CBF88A8596BFDA91D0AF9CA592C3A9F4C6E7A39AF;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd11; cfg_vec256 <= 256'hDAD4F6D59CD1B48A84F9C0EEF68E8EB7B988ABBB82BF89A89DE697A5CCF290F4;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd12; cfg_vec256 <= 256'h26A7423759635D8E35E04FFA22BD4AC95D0C191216222A7C70C030030CC21003;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd13; cfg_vec256 <= 256'h2E4D3BCD3C43A9933FD37C34137635DE2C2474311D209D6B1063520C151DCBB5;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd14; cfg_vec256 <= 256'h05A71EE8267C4AC308CE5FB10BAA445403FB4C0082395E4D01551573842EFBF3;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd15; cfg_vec256 <= 256'h00D3E3630FEC8B83BB24071016D65E200050C62D2A910799A4C6EC49E547D322;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd16; cfg_vec256 <= 256'h00000188CE7F940CE73622CA0A4FE2BBF2FB13E4A58172613AED04530A457383;
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd0; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd1; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd2; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd3; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd4; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd5; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd6; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd7; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd8; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd9; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd10; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd11; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd12; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd13; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd14; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd15; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd16; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd17; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd18; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd19; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd20; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd21; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd22; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd23; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd24; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd25; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd26; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd27; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd28; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd29; cfg_vec256 <= {192'd0, 64'h6996966996696996};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd30; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd31; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd32; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd33; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd34; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd35; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd36; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd37; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd38; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd39; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd40; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd41; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd42; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd43; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd44; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd45; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd46; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd47; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd48; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd49; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd50; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd51; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd52; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd53; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd54; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd55; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd56; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd57; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd58; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd59; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd60; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd61; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd62; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd63; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd64; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd65; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd66; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd67; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd68; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd69; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd70; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd71; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd72; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd73; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd74; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd75; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd76; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd77; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd78; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd79; cfg_vec256 <= {192'd0, 64'h0000000000000000};
        @(posedge clk); cfg_we <= 0; atomic_swap_pulse <= 1;
        @(posedge clk); atomic_swap_pulse <= 0;
        @(posedge clk);
        if (active_bank !== 1'b1) errors = errors + 1;
        bus_in = 512'hCEA3DA56CFC973CEA27AA60C2777D43686A59BFB3D3DAED78A540E92C3216735A702E13BA5AF091FF04784A3CBFC93EEB854964CE09570107F692AD9B0732651;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 4 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 0 MISMATCH: got %0d exp 4", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h8C01AC998A91493DE71EB745790CE2783D5C32CC612F7DBA5B0DBC7EA6A74BA3769566821423CC9CB505BF2D4C9C3C8312D79FD640B364112639D0C3254CA602;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 3 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 1 MISMATCH: got %0d exp 3", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h5DDB66F0F409BA8099336F363C0A8AE0219009A75F3ECF5AE6075037415419E1A37D318763A6CE2EB5C7CC3EC22838CCBEEDD2FA8AD3724CC3B27BBC4F615DEF;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 5 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 2 MISMATCH: got %0d exp 5", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h5E18985AA938BED3F84085773B6A1EE819A431380C42DB71F8CB5032ACEAE96DCFD25025E448C39FDC5D0EAB588076E7D4B0ECD93E493FBE708B4A3332A1DD1E;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 6 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 3 MISMATCH: got %0d exp 6", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h62ED65BE96987385B11E8582BBADE77F96966CE84FB9255102D0BD1B83D9FFE994DDE0756D8C295B3CFC64C73338AB61C364C4CE254EB7153F14B1C95D2D6737;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 6 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 4 MISMATCH: got %0d exp 6", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h3BF414A38204BF5D2642B2B2DE6A20AF2F77DAF39EF68F914EEBA113B5975221427D4F58AAC7E72F51E99752798A9890FABFCBD9C5C7832BAF603DE2D8657316;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 5 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 5 MISMATCH: got %0d exp 5", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'h824542D6D588AB4D9FCFC9307052D187E8050D389DB0E048CD9E0D2F1DABFA37FDC7F4BDA15B1168B6606D8A9A4DAFE31B256D6497844835D00AB045E7669995;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 6 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 6 MISMATCH: got %0d exp 6", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        bus_in = 512'hEDC34C06017AAB395B433B2F31CECC421A25CFC6DEF80FA4F7D7C8EAF88D9BBF245632F3939F995A17D3C9FA07BF5D7D942382907EF13833957372748D6EEAF6;
        #1;
        if ($countones(l0_out_debug[29:20]) !== 4 || |l0_out_debug[19:0] !== 1'b0) begin
            $display("BENES VEC 7 MISMATCH: got %0d exp 4", $countones(l0_out_debug[29:20]));
            errors = errors + 1;
        end
        $display("[Verilator 512x512 Benes-Copy-Benes HotSwap Co-Sim] 80 LUT6s (480 Multicast Ports) | Errors=%0d", errors);
        if (errors != 0) $fatal(1);
        $finish;
    end
endmodule
