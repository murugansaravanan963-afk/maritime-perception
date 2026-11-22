`timescale 1ns / 1ps

module score_argmax_threshold #(
    parameter integer NUM_CLASSES = 6,
    parameter integer SCORE_BITS  = 8,

    // Must be wide enough to represent NUM_CLASSES-1.
    parameter integer CLASS_BITS  =
        (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES),

    parameter integer SCORE_Q_THR = 64
)(
    input logic clk,
    input logic rst_n,

    input logic in_valid,

    // ============================================================
    // Quantized class scores
    // ============================================================

    input logic [SCORE_BITS-1:0]
        score_q [0:NUM_CLASSES-1],

    // ============================================================
    // Result
    // ============================================================

    output logic out_valid,

    output logic [CLASS_BITS-1:0] cls,
    output logic [SCORE_BITS-1:0] score,

    output logic pass
);

    // ============================================================
    // Generic pipelined reduction tree
    //
    // TREE_SIZE:
    // smallest power of 2 >= NUM_CLASSES.
    //
    // Examples:
    //
    // NUM_CLASSES = 5
    // TREE_SIZE    = 8
    //
    // NUM_CLASSES = 6
    // TREE_SIZE    = 32
    //
    // NUM_CLASSES = 80
    // TREE_SIZE    = 128
    //
    // Invalid padded leaves are explicitly marked invalid so that
    // they can never win, even when all real scores are zero.
    // ============================================================

    localparam integer TREE_LEVELS =
        (NUM_CLASSES <= 1) ?
            0 :
            $clog2(NUM_CLASSES);

    localparam integer TREE_SIZE =
        (1 << TREE_LEVELS);


    // ============================================================
    // Tree storage
    //
    // level_score[level][node]
    // level_class[level][node]
    // level_valid[level][node]
    //
    // Level 0 contains the input leaves.
    //
    // Each later level halves the number of active nodes.
    // ============================================================

    logic [SCORE_BITS-1:0]
        level_score [0:TREE_LEVELS][0:TREE_SIZE-1];

    logic [CLASS_BITS-1:0]
        level_class [0:TREE_LEVELS][0:TREE_SIZE-1];

    logic
        level_valid [0:TREE_LEVELS][0:TREE_SIZE-1];


    // ============================================================
    // Valid pipeline
    //
    // Level 0 is registered, then every tree level adds one clock.
    //
    // Final threshold/output register adds one more clock.
    // ============================================================

    logic [TREE_LEVELS:0]
        valid_pipe;


    integer i;
    integer level;
    integer node;


    // ============================================================
    // Level 0
    //
    // Register all real classes.
    //
    // Remaining leaves up to TREE_SIZE are padding.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            valid_pipe <= '0;

            for (
                i = 0;
                i < TREE_SIZE;
                i = i + 1
            ) begin

                level_score[0][i] <= '0;
                level_class[0][i] <= '0;
                level_valid[0][i] <= 1'b0;

            end

        end
        else begin

            // ----------------------------------------------------
            // Valid propagation
            // ----------------------------------------------------

            valid_pipe[0] <=
                in_valid;

            for (
                i = 1;
                i <= TREE_LEVELS;
                i = i + 1
            ) begin

                valid_pipe[i] <=
                    valid_pipe[i-1];

            end


            // ----------------------------------------------------
            // Register leaves
            // ----------------------------------------------------

            if (in_valid) begin

                for (
                    i = 0;
                    i < TREE_SIZE;
                    i = i + 1
                ) begin

                    if (i < NUM_CLASSES) begin

                        level_score[0][i] <=
                            score_q[i];

                        level_class[0][i] <=
                            i;

                        level_valid[0][i] <=
                            1'b1;

                    end
                    else begin

                        // Padding leaf.
                        //
                        // Important:
                        // score=0 alone is not enough because it
                        // could tie with a real class whose score
                        // is also zero.
                        //
                        // valid=0 guarantees padding never wins.

                        level_score[0][i] <=
                            '0;

                        level_class[0][i] <=
                            '0;

                        level_valid[0][i] <=
                            1'b0;

                    end

                end

            end

        end

    end


    // ============================================================
    // Reduction levels
    //
    // At each node compare:
    //
    // LEFT  = node 2*n
    // RIGHT = node 2*n+1
    //
    // Winner rule:
    //
    // 1. valid beats invalid
    // 2. higher score wins
    // 3. equal score -> lower class index wins
    //
    // This preserves the original RTL / Golden tie behavior.
    // ============================================================

    generate

        genvar g_level;
        genvar g_node;

        for (
            g_level = 1;
            g_level <= TREE_LEVELS;
            g_level = g_level + 1
        ) begin : GEN_LEVEL

            localparam integer NODES_THIS_LEVEL =
                TREE_SIZE >> g_level;


            for (
                g_node = 0;
                g_node < NODES_THIS_LEVEL;
                g_node = g_node + 1
            ) begin : GEN_NODE

                always_ff @(posedge clk or negedge rst_n) begin

                    if (!rst_n) begin

                        level_score[g_level][g_node] <=
                            '0;

                        level_class[g_level][g_node] <=
                            '0;

                        level_valid[g_level][g_node] <=
                            1'b0;

                    end
                    else begin

                        // ----------------------------------------
                        // Left invalid, right valid
                        // -> right wins.
                        // ----------------------------------------

                        if (
                            !level_valid[g_level-1][2*g_node]
                            &&
                            level_valid[g_level-1][2*g_node+1]
                        ) begin

                            level_score[g_level][g_node] <=
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node+1
                                ];

                            level_class[g_level][g_node] <=
                                level_class[
                                    g_level-1
                                ][
                                    2*g_node+1
                                ];

                            level_valid[g_level][g_node] <=
                                1'b1;

                        end


                        // ----------------------------------------
                        // Left valid, right invalid
                        // -> left wins.
                        // ----------------------------------------

                        else if (
                            level_valid[g_level-1][2*g_node]
                            &&
                            !level_valid[g_level-1][2*g_node+1]
                        ) begin

                            level_score[g_level][g_node] <=
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node
                                ];

                            level_class[g_level][g_node] <=
                                level_class[
                                    g_level-1
                                ][
                                    2*g_node
                                ];

                            level_valid[g_level][g_node] <=
                                1'b1;

                        end


                        // ----------------------------------------
                        // Both valid
                        // ----------------------------------------

                        else if (
                            level_valid[g_level-1][2*g_node]
                            &&
                            level_valid[g_level-1][2*g_node+1]
                        ) begin

                            // Higher score -> left wins.

                            if (
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node
                                ]
                                >
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node+1
                                ]
                            ) begin

                                level_score[g_level][g_node] <=
                                    level_score[
                                        g_level-1
                                    ][
                                        2*g_node
                                    ];

                                level_class[g_level][g_node] <=
                                    level_class[
                                        g_level-1
                                    ][
                                        2*g_node
                                    ];

                            end


                            // Higher score -> right wins.

                            else if (
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node+1
                                ]
                                >
                                level_score[
                                    g_level-1
                                ][
                                    2*g_node
                                ]
                            ) begin

                                level_score[g_level][g_node] <=
                                    level_score[
                                        g_level-1
                                    ][
                                        2*g_node+1
                                    ];

                                level_class[g_level][g_node] <=
                                    level_class[
                                        g_level-1
                                    ][
                                        2*g_node+1
                                    ];

                            end


                            // ------------------------------------
                            // Tie:
                            //
                            // lower class index wins.
                            // ------------------------------------

                            else begin

                                if (
                                    level_class[
                                        g_level-1
                                    ][
                                        2*g_node
                                    ]
                                    <=
                                    level_class[
                                        g_level-1
                                    ][
                                        2*g_node+1
                                    ]
                                ) begin

                                    level_score[g_level][g_node] <=
                                        level_score[
                                            g_level-1
                                        ][
                                            2*g_node
                                        ];

                                    level_class[g_level][g_node] <=
                                        level_class[
                                            g_level-1
                                        ][
                                            2*g_node
                                        ];

                                end
                                else begin

                                    level_score[g_level][g_node] <=
                                        level_score[
                                            g_level-1
                                        ][
                                            2*g_node+1
                                        ];

                                    level_class[g_level][g_node] <=
                                        level_class[
                                            g_level-1
                                        ][
                                            2*g_node+1
                                        ];

                                end

                            end


                            level_valid[g_level][g_node] <=
                                1'b1;

                        end


                        // ----------------------------------------
                        // Both invalid.
                        // ----------------------------------------

                        else begin

                            level_score[g_level][g_node] <=
                                '0;

                            level_class[g_level][g_node] <=
                                '0;

                            level_valid[g_level][g_node] <=
                                1'b0;

                        end

                    end

                end

            end

        end

    endgenerate


    // ============================================================
    // Final output / threshold stage
    //
    // SCORE_Q_THR follows Golden behavior:
    //
    // pass = max_score >= threshold
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_valid <=
                1'b0;

            cls <=
                '0;

            score <=
                '0;

            pass <=
                1'b0;

        end
        else begin

            out_valid <=
                valid_pipe[TREE_LEVELS];


            if (
                valid_pipe[TREE_LEVELS]
            ) begin

                cls <=
                    level_class[
                        TREE_LEVELS
                    ][0];

                score <=
                    level_score[
                        TREE_LEVELS
                    ][0];


                if (
                    level_valid[
                        TREE_LEVELS
                    ][0]
                    &&
                    (
                        level_score[
                            TREE_LEVELS
                        ][0]
                        >=
                        SCORE_Q_THR
                    )
                )
                    pass <=
                        1'b1;

                else
                    pass <=
                        1'b0;

            end

        end

    end


    // ============================================================
    // Simulation-only parameter checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (NUM_CLASSES < 1) begin

            $error(
                "score_argmax_threshold: NUM_CLASSES must be >= 1"
            );

        end


        if (
            (1 << CLASS_BITS) <
            NUM_CLASSES
        ) begin

            $error(
                "score_argmax_threshold: CLASS_BITS too small for NUM_CLASSES"
            );

        end


        if (
            SCORE_Q_THR >=
            (1 << SCORE_BITS)
        ) begin

            $error(
                "score_argmax_threshold: SCORE_Q_THR does not fit SCORE_BITS"
            );

        end

    end

`endif

endmodule