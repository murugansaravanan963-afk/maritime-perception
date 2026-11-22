`timescale 1ns / 1ps

module nms_engine #(
    parameter integer COORD_BITS = 14,

    parameter integer NUM_CLASSES = 6,
    parameter integer CLASS_BITS =
        (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES),

    parameter integer SCORE_BITS  = 8,
    parameter integer ANCHOR_BITS = 15,

    parameter integer MAX_CAND = 4096,
    parameter integer MAX_DET  = 100,

    parameter integer IOU_NUM = 9,
    parameter integer IOU_DEN = 20,

    parameter integer AREA_BITS = 28
)(
    input logic clk,
    input logic rst_n,

    input logic start,

    input logic [$clog2(MAX_CAND+1)-1:0] cand_count,

    // ============================================================
    // Synchronous Candidate RAM
    // ============================================================

    output logic [$clog2(MAX_CAND)-1:0] cand_rd_addr,

    input logic [COORD_BITS-1:0] cand_x1,
    input logic [COORD_BITS-1:0] cand_y1,
    input logic [COORD_BITS-1:0] cand_x2,
    input logic [COORD_BITS-1:0] cand_y2,

    input logic [CLASS_BITS-1:0] cand_cls,
    input logic [SCORE_BITS-1:0] cand_score,
    input logic [ANCHOR_BITS-1:0] cand_anchor,

    // ============================================================
    // Detection output
    // ============================================================

    output logic det_valid,

    output logic [COORD_BITS-1:0] det_x1,
    output logic [COORD_BITS-1:0] det_y1,
    output logic [COORD_BITS-1:0] det_x2,
    output logic [COORD_BITS-1:0] det_y2,

    output logic [CLASS_BITS-1:0] det_cls,
    output logic [SCORE_BITS-1:0] det_score,
    output logic [ANCHOR_BITS-1:0] det_anchor,

    output logic [$clog2(MAX_DET+1)-1:0] n_det,

    output logic done
);

    localparam integer CAND_ADDR_BITS =
        (MAX_CAND <= 1) ? 1 : $clog2(MAX_CAND);


    // ============================================================
    // FSM
    // ============================================================

    typedef enum logic [4:0] {
        S_IDLE,

        // Sequential flag-memory clear
        S_CLEAR_FLAGS,

        S_CLASS_START,

        S_FIND_REQ,
        S_FIND_FLAG_READ,
        S_FIND_CHECK,
        S_FIND_DONE,

        S_BEST_REQ,
        S_KEEP_BEST,

        S_SUPPRESS_START,
        S_SUPPRESS_REQ,
        S_SUPPRESS_FLAG_READ,
        S_SUPPRESS_CHECK,
        S_SUPPRESS_WAIT,

        S_NEXT_CLASS,

        S_DONE
    } state_t;

    state_t state;


    // ============================================================
    // Current class / candidate index
    // ============================================================

    logic [CLASS_BITS-1:0] current_class;

    logic [CAND_ADDR_BITS-1:0] scan_idx;


    // ============================================================
    // MAX 10 synchronous flag memories
    //
    // Original:
    //
    //   processed[scan_idx]
    //   suppressed[scan_idx]
    //
    // were implemented as thousands of FFs + a huge dynamic mux.
    //
    // New:
    //
    //   4096 x 1 synchronous memories
    //
    // Logical meaning is unchanged:
    //
    //   processed  = candidate already emitted
    //   suppressed = candidate removed by IoU
    // ============================================================

    (* ramstyle = "M9K" *)
    logic processed_mem [0:MAX_CAND-1];

    (* ramstyle = "M9K" *)
    logic suppressed_mem [0:MAX_CAND-1];


    // Registered synchronous-memory outputs

    logic processed_rd_data;
    logic suppressed_rd_data;


    // Registered flags used by FSM

    logic processed_flag_reg;
    logic suppressed_flag_reg;


    // Per-frame sequential clear address

    logic [CAND_ADDR_BITS-1:0] clear_idx;


    // ============================================================
    // Synchronous flag-memory read
    //
    // scan_idx is registered.
    //
    // This removes the old:
    //
    // scan_idx
    //    ↓
    // 4096:1 FF mux
    //    ↓
    // flag register
    //
    // path.
    // ============================================================

    always_ff @(posedge clk) begin

        processed_rd_data <=
            processed_mem[scan_idx];

        suppressed_rd_data <=
            suppressed_mem[scan_idx];

    end


    // ============================================================
    // Best candidate
    // ============================================================

    logic best_found;

    logic [CAND_ADDR_BITS-1:0] best_idx;

    logic [SCORE_BITS-1:0] best_score;
    logic [ANCHOR_BITS-1:0] best_anchor;


    // ============================================================
    // Saved best box
    // ============================================================

    logic [COORD_BITS-1:0] best_x1;
    logic [COORD_BITS-1:0] best_y1;
    logic [COORD_BITS-1:0] best_x2;
    logic [COORD_BITS-1:0] best_y2;


    // ============================================================
    // IoU comparator
    // ============================================================

    logic iou_in_valid;
    logic iou_out_valid;
    logic iou_suppress;


    iou_compare #(
        .COORD_BITS(COORD_BITS),

        .IOU_NUM(IOU_NUM),
        .IOU_DEN(IOU_DEN),

        .AREA_BITS(AREA_BITS)
    ) u_iou_compare (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(iou_in_valid),

        .ax1(best_x1),
        .ay1(best_y1),
        .ax2(best_x2),
        .ay2(best_y2),

        .bx1(cand_x1),
        .by1(cand_y1),
        .bx2(cand_x2),
        .by2(cand_y2),

        .out_valid(iou_out_valid),

        .suppress(iou_suppress)
    );


    // ============================================================
    // Candidate currently undergoing IoU comparison
    // ============================================================

    logic [CAND_ADDR_BITS-1:0]
        suppress_candidate_idx;


    // ============================================================
    // Candidate RAM address
    // ============================================================

    always_comb begin

        cand_rd_addr = scan_idx;

    end


    // ============================================================
    // Main FSM
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            state <= S_IDLE;

            current_class <= '0;

            scan_idx <= '0;
            clear_idx <= '0;

            processed_flag_reg <= 1'b0;
            suppressed_flag_reg <= 1'b0;

            best_found <= 1'b0;

            best_idx <= '0;
            best_score <= '0;
            best_anchor <= '0;

            best_x1 <= '0;
            best_y1 <= '0;
            best_x2 <= '0;
            best_y2 <= '0;

            suppress_candidate_idx <= '0;

            iou_in_valid <= 1'b0;

            det_valid <= 1'b0;

            det_x1 <= '0;
            det_y1 <= '0;
            det_x2 <= '0;
            det_y2 <= '0;

            det_cls <= '0;
            det_score <= '0;
            det_anchor <= '0;

            n_det <= '0;

            done <= 1'b0;

        end
        else begin

            // ====================================================
            // Default one-cycle pulses
            // ====================================================

            det_valid <= 1'b0;
            done <= 1'b0;
            iou_in_valid <= 1'b0;


            case (state)

                // =================================================
                // IDLE
                // =================================================

                S_IDLE:
                begin

                    if (start) begin

                        current_class <= '0;

                        scan_idx <= '0;
                        clear_idx <= '0;

                        best_found <= 1'b0;

                        best_idx <= '0;
                        best_score <= '0;
                        best_anchor <= '0;

                        processed_flag_reg <= 1'b0;
                        suppressed_flag_reg <= 1'b0;

                        n_det <= '0;


                        if (cand_count == 0) begin

                            state <= S_DONE;

                        end
                        else begin

                            // ------------------------------------
                            // Instead of clearing 8192 FFs in
                            // parallel, clear only candidates that
                            // actually exist in this frame.
                            // ------------------------------------

                            state <= S_CLEAR_FLAGS;

                        end

                    end

                end


                // =================================================
                // Sequentially clear current-frame flag entries
                //
                // Adds cand_count clocks once at NMS start.
                //
                // For Mode-B sparse candidate counts this should
                // remain small compared with the full NMS scan.
                // =================================================

                S_CLEAR_FLAGS:
                begin

                    processed_mem[clear_idx] <= 1'b0;
                    suppressed_mem[clear_idx] <= 1'b0;


                    if (
                        clear_idx ==
                        cand_count - 1'b1
                    ) begin

                        clear_idx <= '0;

                        state <= S_CLASS_START;

                    end
                    else begin

                        clear_idx <=
                            clear_idx + 1'b1;

                    end

                end


                // =================================================
                // Start current class
                // =================================================

                S_CLASS_START:
                begin

                    scan_idx <= '0;

                    best_found <= 1'b0;

                    best_idx <= '0;
                    best_score <= '0;
                    best_anchor <= '0;

                    state <= S_FIND_REQ;

                end


                // =================================================
                // FIND candidate request
                //
                // Candidate RAM and flag RAM both receive scan_idx.
                // =================================================

                S_FIND_REQ:
                begin

                    if (
                        scan_idx <
                        cand_count
                    ) begin

                        state <= S_FIND_FLAG_READ;

                    end
                    else begin

                        state <= S_FIND_DONE;

                    end

                end


                // =================================================
                // Capture synchronous flag-memory result
                // =================================================

                S_FIND_FLAG_READ:
                begin

                    processed_flag_reg <=
                        processed_rd_data;

                    suppressed_flag_reg <=
                        suppressed_rd_data;

                    state <= S_FIND_CHECK;

                end


                // =================================================
                // FIND:
                // Same candidate-selection rule as original RTL.
                // =================================================

                S_FIND_CHECK:
                begin

                    if (
                        (
                            cand_cls ==
                            current_class
                        )
                        &&
                        !processed_flag_reg
                        &&
                        !suppressed_flag_reg
                    ) begin

                        // First valid candidate

                        if (!best_found) begin

                            best_found <= 1'b1;

                            best_idx <= scan_idx;
                            best_score <= cand_score;
                            best_anchor <= cand_anchor;

                        end

                        // Higher score wins.
                        //
                        // Equal score:
                        // lower anchor index wins.
                        //
                        // Original tie-break rule preserved.

                        else if (
                            (
                                cand_score >
                                best_score
                            )
                            ||
                            (
                                (
                                    cand_score ==
                                    best_score
                                )
                                &&
                                (
                                    cand_anchor <
                                    best_anchor
                                )
                            )
                        ) begin

                            best_idx <= scan_idx;
                            best_score <= cand_score;
                            best_anchor <= cand_anchor;

                        end

                    end


                    if (
                        scan_idx ==
                        cand_count - 1'b1
                    ) begin

                        state <= S_FIND_DONE;

                    end
                    else begin

                        scan_idx <=
                            scan_idx + 1'b1;

                        state <= S_FIND_REQ;

                    end

                end


                // =================================================
                // Best search complete
                // =================================================

                S_FIND_DONE:
                begin

                    if (!best_found) begin

                        state <= S_NEXT_CLASS;

                    end
                    else begin

                        scan_idx <= best_idx;

                        state <= S_BEST_REQ;

                    end

                end


                // =================================================
                // Candidate RAM synchronous read request
                // =================================================

                S_BEST_REQ:
                begin

                    state <= S_KEEP_BEST;

                end


                // =================================================
                // Keep best candidate
                // =================================================

                S_KEEP_BEST:
                begin

                    // Save box for IoU comparisons

                    best_x1 <= cand_x1;
                    best_y1 <= cand_y1;
                    best_x2 <= cand_x2;
                    best_y2 <= cand_y2;


                    // Same logical operation as:
                    //
                    // processed[best_idx] = 1

                    processed_mem[best_idx] <= 1'b1;


                    // Emit selected detection

                    if (
                        n_det <
                        MAX_DET
                    ) begin

                        det_valid <= 1'b1;

                        det_x1 <= cand_x1;
                        det_y1 <= cand_y1;
                        det_x2 <= cand_x2;
                        det_y2 <= cand_y2;

                        det_cls <= cand_cls;
                        det_score <= cand_score;
                        det_anchor <= cand_anchor;

                        n_det <=
                            n_det + 1'b1;

                    end


                    if (
                        (n_det + 1'b1) >=
                        MAX_DET
                    ) begin

                        state <= S_DONE;

                    end
                    else begin

                        state <= S_SUPPRESS_START;

                    end

                end


                // =================================================
                // Start suppression scan
                // =================================================

                S_SUPPRESS_START:
                begin

                    scan_idx <= '0;

                    state <= S_SUPPRESS_REQ;

                end


                // =================================================
                // Suppression candidate request
                // =================================================

                S_SUPPRESS_REQ:
                begin

                    if (
                        scan_idx >=
                        cand_count
                    ) begin

                        state <= S_CLASS_START;

                    end
                    else begin

                        state <= S_SUPPRESS_FLAG_READ;

                    end

                end


                // =================================================
                // Capture synchronous flag-memory result
                // =================================================

                S_SUPPRESS_FLAG_READ:
                begin

                    processed_flag_reg <=
                        processed_rd_data;

                    suppressed_flag_reg <=
                        suppressed_rd_data;

                    state <= S_SUPPRESS_CHECK;

                end


                // =================================================
                // Suppression check
                // =================================================

                S_SUPPRESS_CHECK:
                begin

                    if (
                        (
                            cand_cls ==
                            current_class
                        )
                        &&
                        !processed_flag_reg
                        &&
                        !suppressed_flag_reg
                    ) begin

                        suppress_candidate_idx <=
                            scan_idx;

                        iou_in_valid <=
                            1'b1;

                        state <= S_SUPPRESS_WAIT;

                    end
                    else begin

                        if (
                            scan_idx ==
                            cand_count - 1'b1
                        ) begin

                            state <= S_CLASS_START;

                        end
                        else begin

                            scan_idx <=
                                scan_idx + 1'b1;

                            state <= S_SUPPRESS_REQ;

                        end

                    end

                end


                // =================================================
                // Wait for pipelined IoU result
                // =================================================

                S_SUPPRESS_WAIT:
                begin

                    if (iou_out_valid) begin

                        // Same logical operation as:
                        //
                        // suppressed[index] = 1

                        if (iou_suppress) begin

                            suppressed_mem[
                                suppress_candidate_idx
                            ] <= 1'b1;

                        end


                        if (
                            suppress_candidate_idx ==
                            cand_count - 1'b1
                        ) begin

                            state <= S_CLASS_START;

                        end
                        else begin

                            scan_idx <=
                                suppress_candidate_idx
                                + 1'b1;

                            state <= S_SUPPRESS_REQ;

                        end

                    end

                end


                // =================================================
                // Current class finished
                // =================================================

                S_NEXT_CLASS:
                begin

                    if (
                        current_class ==
                        NUM_CLASSES - 1
                    ) begin

                        state <= S_DONE;

                    end
                    else begin

                        current_class <=
                            current_class + 1'b1;

                        state <= S_CLASS_START;

                    end

                end


                // =================================================
                // Done
                // =================================================

                S_DONE:
                begin

                    done <= 1'b1;

                    state <= S_IDLE;

                end


                default:
                begin

                    state <= S_IDLE;

                end

            endcase

        end

    end


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (NUM_CLASSES < 1)
            $error(
                "nms_engine: NUM_CLASSES must be >= 1"
            );

        if ((1 << CLASS_BITS) < NUM_CLASSES)
            $error(
                "nms_engine: CLASS_BITS too small"
            );

        if (MAX_CAND < 1)
            $error(
                "nms_engine: MAX_CAND must be >= 1"
            );

        if (MAX_DET < 1)
            $error(
                "nms_engine: MAX_DET must be >= 1"
            );

        if (IOU_DEN <= 0)
            $error(
                "nms_engine: IOU_DEN must be > 0"
            );

        if (IOU_NUM < 0)
            $error(
                "nms_engine: IOU_NUM must be >= 0"
            );

        if (IOU_NUM > IOU_DEN)
            $error(
                "nms_engine: IoU threshold must be <= 1"
            );

    end


    always_ff @(posedge clk) begin

        if (rst_n) begin

            if (cand_count > MAX_CAND)
                $error(
                    "nms_engine: cand_count exceeds MAX_CAND"
                );

            if (n_det > MAX_DET)
                $error(
                    "nms_engine: n_det exceeds MAX_DET"
                );

        end

    end

`endif

endmodule