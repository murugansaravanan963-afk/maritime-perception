`timescale 1ns / 1ps

module downscale_block #(
    parameter integer Q_BITS = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     in_valid,

    input  wire [Q_BITS-1:0]         q0,
    input  wire [Q_BITS-1:0]         q1,
    input  wire [Q_BITS-1:0]         q2,
    input  wire [Q_BITS-1:0]         q3,
    input  wire [Q_BITS-1:0]         q4,
    input  wire [Q_BITS-1:0]         q5,
    input  wire [Q_BITS-1:0]         q6,
    input  wire [Q_BITS-1:0]         q7,
    input  wire [Q_BITS-1:0]         q8,
    input  wire [Q_BITS-1:0]         q9,
    input  wire [Q_BITS-1:0]         q10,
    input  wire [Q_BITS-1:0]         q11,
    input  wire [Q_BITS-1:0]         q12,
    input  wire [Q_BITS-1:0]         q13,
    input  wire [Q_BITS-1:0]         q14,
    input  wire [Q_BITS-1:0]         q15,

    output reg                      out_valid,

    output reg [Q_BITS-1:0]         d0,
    output reg [Q_BITS-1:0]         d1,
    output reg [Q_BITS-1:0]         d2,
    output reg [Q_BITS-1:0]         d3,
    output reg [Q_BITS-1:0]         d4,
    output reg [Q_BITS-1:0]         d5,
    output reg [Q_BITS-1:0]         d6,
    output reg [Q_BITS-1:0]         d7,
    output reg [Q_BITS-1:0]         d8,
    output reg [Q_BITS-1:0]         d9,
    output reg [Q_BITS-1:0]         d10,
    output reg [Q_BITS-1:0]         d11,
    output reg [Q_BITS-1:0]         d12,
    output reg [Q_BITS-1:0]         d13,
    output reg [Q_BITS-1:0]         d14,
    output reg [Q_BITS-1:0]         d15
);

    // ------------------------------------------------------------
    // Stage 1: comparator tree
    // ------------------------------------------------------------

    reg [Q_BITS-1:0] max1_0;
    reg [Q_BITS-1:0] max1_1;
    reg [Q_BITS-1:0] max1_2;
    reg [Q_BITS-1:0] max1_3;
    reg [Q_BITS-1:0] max1_4;
    reg [Q_BITS-1:0] max1_5;
    reg [Q_BITS-1:0] max1_6;
    reg [Q_BITS-1:0] max1_7;

    reg [Q_BITS-1:0] q_s1 [0:15];

    reg valid_s1;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max1_0 <= 0;
            max1_1 <= 0;
            max1_2 <= 0;
            max1_3 <= 0;
            max1_4 <= 0;
            max1_5 <= 0;
            max1_6 <= 0;
            max1_7 <= 0;

            valid_s1 <= 1'b0;

            for (i = 0; i < 16; i = i + 1)
                q_s1[i] <= 0;
        end
        else begin
            valid_s1 <= in_valid;

            if (in_valid) begin
                max1_0 <= (q0  >= q1 ) ? q0  : q1;
                max1_1 <= (q2  >= q3 ) ? q2  : q3;
                max1_2 <= (q4  >= q5 ) ? q4  : q5;
                max1_3 <= (q6  >= q7 ) ? q6  : q7;
                max1_4 <= (q8  >= q9 ) ? q8  : q9;
                max1_5 <= (q10 >= q11) ? q10 : q11;
                max1_6 <= (q12 >= q13) ? q12 : q13;
                max1_7 <= (q14 >= q15) ? q14 : q15;

                q_s1[0]  <= q0;
                q_s1[1]  <= q1;
                q_s1[2]  <= q2;
                q_s1[3]  <= q3;
                q_s1[4]  <= q4;
                q_s1[5]  <= q5;
                q_s1[6]  <= q6;
                q_s1[7]  <= q7;
                q_s1[8]  <= q8;
                q_s1[9]  <= q9;
                q_s1[10] <= q10;
                q_s1[11] <= q11;
                q_s1[12] <= q12;
                q_s1[13] <= q13;
                q_s1[14] <= q14;
                q_s1[15] <= q15;
            end
        end
    end


    // ------------------------------------------------------------
    // Stage 2
    // ------------------------------------------------------------

    reg [Q_BITS-1:0] max2_0;
    reg [Q_BITS-1:0] max2_1;
    reg [Q_BITS-1:0] max2_2;
    reg [Q_BITS-1:0] max2_3;

    reg [Q_BITS-1:0] q_s2 [0:15];

    reg valid_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max2_0 <= 0;
            max2_1 <= 0;
            max2_2 <= 0;
            max2_3 <= 0;

            valid_s2 <= 1'b0;

            for (i = 0; i < 16; i = i + 1)
                q_s2[i] <= 0;
        end
        else begin
            valid_s2 <= valid_s1;

            if (valid_s1) begin
                max2_0 <= (max1_0 >= max1_1) ? max1_0 : max1_1;
                max2_1 <= (max1_2 >= max1_3) ? max1_2 : max1_3;
                max2_2 <= (max1_4 >= max1_5) ? max1_4 : max1_5;
                max2_3 <= (max1_6 >= max1_7) ? max1_6 : max1_7;

                for (i = 0; i < 16; i = i + 1)
                    q_s2[i] <= q_s1[i];
            end
        end
    end


    // ------------------------------------------------------------
    // Stage 3
    // ------------------------------------------------------------

    reg [Q_BITS-1:0] max3_0;
    reg [Q_BITS-1:0] max3_1;

    reg [Q_BITS-1:0] q_s3 [0:15];

    reg valid_s3;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max3_0 <= 0;
            max3_1 <= 0;

            valid_s3 <= 1'b0;

            for (i = 0; i < 16; i = i + 1)
                q_s3[i] <= 0;
        end
        else begin
            valid_s3 <= valid_s2;

            if (valid_s2) begin
                max3_0 <= (max2_0 >= max2_1) ? max2_0 : max2_1;
                max3_1 <= (max2_2 >= max2_3) ? max2_2 : max2_3;

                for (i = 0; i < 16; i = i + 1)
                    q_s3[i] <= q_s2[i];
            end
        end
    end


    // ------------------------------------------------------------
    // Stage 4: final maximum
    // ------------------------------------------------------------

    reg [Q_BITS-1:0] q_s4 [0:15];
    reg [Q_BITS-1:0] qmax;

    reg valid_s4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            qmax <= 0;
            valid_s4 <= 1'b0;

            for (i = 0; i < 16; i = i + 1)
                q_s4[i] <= 0;
        end
        else begin
            valid_s4 <= valid_s3;

            if (valid_s3) begin
                qmax <= (max3_0 >= max3_1) ? max3_0 : max3_1;

                for (i = 0; i < 16; i = i + 1)
                    q_s4[i] <= q_s3[i];
            end
        end
    end


    // ------------------------------------------------------------
    // Stage 5:
    //
    // d[i] = qmax - q[i]
    //
    // Because qmax >= q[i]:
    //
    //     0 <= d[i] <= 255     for Q_BITS = 8
    //
    // Therefore d is UNSIGNED and still fits in Q_BITS bits.
    // ------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;

            d0  <= 0;
            d1  <= 0;
            d2  <= 0;
            d3  <= 0;
            d4  <= 0;
            d5  <= 0;
            d6  <= 0;
            d7  <= 0;
            d8  <= 0;
            d9  <= 0;
            d10 <= 0;
            d11 <= 0;
            d12 <= 0;
            d13 <= 0;
            d14 <= 0;
            d15 <= 0;
        end
        else begin
            out_valid <= valid_s4;

            if (valid_s4) begin
                d0  <= qmax - q_s4[0];
                d1  <= qmax - q_s4[1];
                d2  <= qmax - q_s4[2];
                d3  <= qmax - q_s4[3];
                d4  <= qmax - q_s4[4];
                d5  <= qmax - q_s4[5];
                d6  <= qmax - q_s4[6];
                d7  <= qmax - q_s4[7];
                d8  <= qmax - q_s4[8];
                d9  <= qmax - q_s4[9];
                d10 <= qmax - q_s4[10];
                d11 <= qmax - q_s4[11];
                d12 <= qmax - q_s4[12];
                d13 <= qmax - q_s4[13];
                d14 <= qmax - q_s4[14];
                d15 <= qmax - q_s4[15];
            end
        end
    end

endmodule