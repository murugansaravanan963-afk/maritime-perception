`timescale 1ns / 1ps

module dfl_exp_lut #(
    parameter integer Q_BITS = 8,
    parameter integer LUT_WIDTH = 16,
    parameter integer LUT_DEPTH = 65,
    parameter LUT_FILE = "dfl_exp_lut.mem"
)(
    input  wire [Q_BITS-1:0] d,
    output reg  [LUT_WIDTH-1:0] exp_out
);

    // Maritime6-960 logic ROM. Values are generated from the supplied
    // rtl_vectors/dfl_exp_lut.mem. d >= LUT_DEPTH maps to zero.
    // A logic ROM is retained so 16 parallel reads/core do not replicate BRAMs.
    always @(*) begin
        case (d)
            8'd0: exp_out = 16'h8000;
            8'd1: exp_out = 16'h6b94;
            8'd2: exp_out = 16'h5a6a;
            8'd3: exp_out = 16'h4bfd;
            8'd4: exp_out = 16'h3fde;
            8'd5: exp_out = 16'h35ad;
            8'd6: exp_out = 16'h2d1d;
            8'd7: exp_out = 16'h25ea;
            8'd8: exp_out = 16'h1fde;
            8'd9: exp_out = 16'h1ac8;
            8'd10: exp_out = 16'h1682;
            8'd11: exp_out = 16'h12eb;
            8'd12: exp_out = 16'h0fe6;
            8'd13: exp_out = 16'h0d5d;
            8'd14: exp_out = 16'h0b3b;
            8'd15: exp_out = 16'h0970;
            8'd16: exp_out = 16'h07ef;
            8'd17: exp_out = 16'h06ab;
            8'd18: exp_out = 16'h059b;
            8'd19: exp_out = 16'h04b6;
            8'd20: exp_out = 16'h03f5;
            8'd21: exp_out = 16'h0354;
            8'd22: exp_out = 16'h02cc;
            8'd23: exp_out = 16'h025a;
            8'd24: exp_out = 16'h01fa;
            8'd25: exp_out = 16'h01a9;
            8'd26: exp_out = 16'h0165;
            8'd27: exp_out = 16'h012c;
            8'd28: exp_out = 16'h00fc;
            8'd29: exp_out = 16'h00d4;
            8'd30: exp_out = 16'h00b2;
            8'd31: exp_out = 16'h0096;
            8'd32: exp_out = 16'h007e;
            8'd33: exp_out = 16'h006a;
            8'd34: exp_out = 16'h0059;
            8'd35: exp_out = 16'h004b;
            8'd36: exp_out = 16'h003f;
            8'd37: exp_out = 16'h0035;
            8'd38: exp_out = 16'h002c;
            8'd39: exp_out = 16'h0025;
            8'd40: exp_out = 16'h001f;
            8'd41: exp_out = 16'h001a;
            8'd42: exp_out = 16'h0016;
            8'd43: exp_out = 16'h0013;
            8'd44: exp_out = 16'h0010;
            8'd45: exp_out = 16'h000d;
            8'd46: exp_out = 16'h000b;
            8'd47: exp_out = 16'h0009;
            8'd48: exp_out = 16'h0008;
            8'd49: exp_out = 16'h0007;
            8'd50: exp_out = 16'h0006;
            8'd51: exp_out = 16'h0005;
            8'd52: exp_out = 16'h0004;
            8'd53: exp_out = 16'h0003;
            8'd54: exp_out = 16'h0003;
            8'd55: exp_out = 16'h0002;
            8'd56: exp_out = 16'h0002;
            8'd57: exp_out = 16'h0002;
            8'd58: exp_out = 16'h0001;
            8'd59: exp_out = 16'h0001;
            8'd60: exp_out = 16'h0001;
            8'd61: exp_out = 16'h0001;
            8'd62: exp_out = 16'h0001;
            8'd63: exp_out = 16'h0001;
            8'd64: exp_out = 16'h0000;
            default: exp_out = {LUT_WIDTH{1'b0}};
        endcase
    end

`ifndef SYNTHESIS
    initial begin
        if (Q_BITS != 8) $error("dfl_exp_lut requires Q_BITS=8");
        if (LUT_WIDTH != 16) $error("dfl_exp_lut requires LUT_WIDTH=16");
        if (LUT_DEPTH != 65) $error("Maritime6-960 LUT requires LUT_DEPTH=65");
    end
`endif
endmodule
