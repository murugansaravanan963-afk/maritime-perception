`timescale 1ns / 1ps

module dfl_divider #(
    parameter integer SUM_WIDTH = 20,
    parameter integer WEIGHTED_SUM_WIDTH = 23,
    parameter integer OFF_BITS = 12,
    parameter integer OFF_FRAC = 8,

    parameter integer NUM_WIDTH =
        WEIGHTED_SUM_WIDTH + OFF_FRAC
)(
    input wire clk,
    input wire rst_n,

    input wire in_valid,

    input wire [SUM_WIDTH-1:0]
        sum_exp,

    input wire [WEIGHTED_SUM_WIDTH-1:0]
        weighted_sum,

    output wire busy,

    output reg out_valid,

    output reg [OFF_BITS-1:0]
        offset
);


    // ============================================================
    // Radix-4 exact restoring divider
    //
    // Two radix-2 restoring steps are performed per clock.
    //
    // For odd NUM_WIDTH:
    //
    //     first iteration processes one bit
    //     remaining iterations process two bits
    //
    // Example NUM_WIDTH = 31:
    //
    //     iteration 0 : 1 bit
    //     iteration 1 : 2 bits
    //     ...
    //     iteration 15: 2 bits
    //
    // Total = 31 quotient bits in 16 busy clocks.
    //
    // IMPORTANT:
    // This is NOT approximate division.
    // The arithmetic sequence is identical to the original
    // radix-2 restoring divider.
    // ============================================================


    // ============================================================
    // Derived constants
    // ============================================================

    localparam integer NUM_ITER =
        (NUM_WIDTH + 1) / 2;


    localparam integer COUNT_WIDTH =
        (NUM_ITER <= 1)
        ? 1
        : $clog2(NUM_ITER + 1);


    localparam integer FIRST_SINGLE =
        NUM_WIDTH % 2;


    // ============================================================
    // Internal state
    // ============================================================

    reg busy_r;


    reg [NUM_WIDTH-1:0]
        numerator_r;


    reg [SUM_WIDTH-1:0]
        denominator_r;


    reg [NUM_WIDTH-1:0]
        quotient_r;


    reg [SUM_WIDTH:0]
        remainder_r;


    reg [COUNT_WIDTH-1:0]
        count_r;


    // ============================================================
    // Denominator extension
    // ============================================================

    wire [SUM_WIDTH:0]
        denominator_ext;


    assign denominator_ext =
        {
            1'b0,
            denominator_r
        };


    // ============================================================
    // Restoring step #1
    //
    // Consume numerator_r[NUM_WIDTH-1].
    // ============================================================

    wire [SUM_WIDTH:0]
        remainder_shift_1;


    wire
        quotient_bit_1;


    wire [SUM_WIDTH:0]
        remainder_after_1;


    assign remainder_shift_1 =
        {
            remainder_r[SUM_WIDTH-1:0],
            numerator_r[NUM_WIDTH-1]
        };


    assign quotient_bit_1 =
        (
            remainder_shift_1 >=
            denominator_ext
        );


    assign remainder_after_1 =
        quotient_bit_1
        ?
        (
            remainder_shift_1 -
            denominator_ext
        )
        :
        remainder_shift_1;


    // ============================================================
    // Restoring step #2
    //
    // Consume numerator_r[NUM_WIDTH-2].
    //
    // This step begins from remainder_after_1, therefore it is
    // exactly equivalent to executing the original radix-2
    // divider for a second clock.
    // ============================================================

    wire [SUM_WIDTH:0]
        remainder_shift_2;


    wire
        quotient_bit_2;


    wire [SUM_WIDTH:0]
        remainder_after_2;


    assign remainder_shift_2 =
        {
            remainder_after_1[SUM_WIDTH-1:0],
            numerator_r[NUM_WIDTH-2]
        };


    assign quotient_bit_2 =
        (
            remainder_shift_2 >=
            denominator_ext
        );


    assign remainder_after_2 =
        quotient_bit_2
        ?
        (
            remainder_shift_2 -
            denominator_ext
        )
        :
        remainder_shift_2;


    // ============================================================
    // Busy output
    // ============================================================

    assign busy =
        busy_r;


    // ============================================================
    // Sequential divider
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            busy_r <=
                1'b0;


            out_valid <=
                1'b0;


            offset <=
                {OFF_BITS{1'b0}};


            numerator_r <=
                {NUM_WIDTH{1'b0}};


            denominator_r <=
                {SUM_WIDTH{1'b0}};


            quotient_r <=
                {NUM_WIDTH{1'b0}};


            remainder_r <=
                {(SUM_WIDTH+1){1'b0}};


            count_r <=
                {COUNT_WIDTH{1'b0}};

        end


        else begin

            // ====================================================
            // Default one-clock output pulse
            // ====================================================

            out_valid <=
                1'b0;


            // ====================================================
            // IDLE
            // ====================================================

            if (!busy_r) begin

                if (in_valid) begin

                    // ------------------------------------------------
                    // Preserve original divide-by-zero behavior.
                    // ------------------------------------------------

                    if (
                        sum_exp ==
                        {SUM_WIDTH{1'b0}}
                    ) begin

                        offset <=
                            {OFF_BITS{1'b0}};


                        out_valid <=
                            1'b1;


                        busy_r <=
                            1'b0;

                    end


                    else begin

                        // --------------------------------------------
                        // numerator =
                        //
                        // weighted_sum << OFF_FRAC
                        // --------------------------------------------

                        numerator_r <=
                            {
                                weighted_sum,
                                {OFF_FRAC{1'b0}}
                            };


                        denominator_r <=
                            sum_exp;


                        quotient_r <=
                            {NUM_WIDTH{1'b0}};


                        remainder_r <=
                            {(SUM_WIDTH+1){1'b0}};


                        count_r <=
                            {COUNT_WIDTH{1'b0}};


                        busy_r <=
                            1'b1;

                    end

                end

            end


            // ====================================================
            // BUSY
            // ====================================================

            else begin

                // =================================================
                // Odd-width first iteration
                //
                // NUM_WIDTH = 31 in our current DFL configuration.
                //
                // First cycle consumes only the first bit.
                //
                // This leaves an even 30 bits for the remaining
                // 15 radix-4 iterations.
                // =================================================

                if (
                    FIRST_SINGLE
                    &&
                    count_r == 0
                ) begin

                    // --------------------------------------------
                    // Shift numerator by one bit.
                    // --------------------------------------------

                    numerator_r <=
                        {
                            numerator_r[NUM_WIDTH-2:0],
                            1'b0
                        };


                    // --------------------------------------------
                    // Save one-step remainder.
                    // --------------------------------------------

                    remainder_r <=
                        remainder_after_1;


                    // --------------------------------------------
                    // Append one quotient bit.
                    // --------------------------------------------

                    quotient_r <=
                        {
                            quotient_r[NUM_WIDTH-2:0],
                            quotient_bit_1
                        };


                    // --------------------------------------------
                    // NUM_WIDTH=1 corner case.
                    // --------------------------------------------

                    if (
                        NUM_ITER == 1
                    ) begin

                        offset <=
                            {
                                quotient_r[OFF_BITS-2:0],
                                quotient_bit_1
                            };


                        out_valid <=
                            1'b1;


                        busy_r <=
                            1'b0;


                        count_r <=
                            {COUNT_WIDTH{1'b0}};

                    end


                    else begin

                        count_r <=
                            count_r + 1'b1;

                    end

                end


                // =================================================
                // Normal radix-4 iteration
                //
                // Execute two exact restoring steps.
                // =================================================

                else begin

                    // --------------------------------------------
                    // Shift numerator by two bits.
                    // --------------------------------------------

                    numerator_r <=
                        {
                            numerator_r[NUM_WIDTH-3:0],
                            2'b00
                        };


                    // --------------------------------------------
                    // Save remainder after two restoring steps.
                    // --------------------------------------------

                    remainder_r <=
                        remainder_after_2;


                    // --------------------------------------------
                    // Append two quotient bits.
                    // --------------------------------------------

                    quotient_r <=
                        {
                            quotient_r[NUM_WIDTH-3:0],
                            quotient_bit_1,
                            quotient_bit_2
                        };


                    // ============================================
                    // Last radix-4 iteration
                    // ============================================

                    if (
                        count_r ==
                        NUM_ITER - 1
                    ) begin

                        // ----------------------------------------
                        // quotient_r is updated non-blockingly.
                        //
                        // Therefore construct the final lower
                        // OFF_BITS explicitly using the two
                        // quotient bits generated this cycle.
                        // ----------------------------------------

                        offset <=
                            {
                                quotient_r[
                                    OFF_BITS-3:0
                                ],
                                quotient_bit_1,
                                quotient_bit_2
                            };


                        out_valid <=
                            1'b1;


                        busy_r <=
                            1'b0;


                        count_r <=
                            {COUNT_WIDTH{1'b0}};

                    end


                    else begin

                        count_r <=
                            count_r + 1'b1;

                    end

                end

            end

        end

    end


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            SUM_WIDTH < 1
        ) begin

            $error(
                "dfl_divider: SUM_WIDTH must be >= 1"
            );

        end


        if (
            WEIGHTED_SUM_WIDTH < 1
        ) begin

            $error(
                "dfl_divider: WEIGHTED_SUM_WIDTH must be >= 1"
            );

        end


        if (
            OFF_BITS < 3
        ) begin

            $error(
                "dfl_divider: radix-4 implementation requires OFF_BITS >= 3"
            );

        end


        if (
            OFF_FRAC < 0
        ) begin

            $error(
                "dfl_divider: OFF_FRAC must be >= 0"
            );

        end


        if (
            NUM_WIDTH <
            OFF_BITS
        ) begin

            $error(
                "dfl_divider: NUM_WIDTH must be >= OFF_BITS"
            );

        end


        if (
            NUM_WIDTH < 3
        ) begin

            $error(
                "dfl_divider: radix-4 implementation requires NUM_WIDTH >= 3"
            );

        end

    end

`endif

endmodule