module mode_b_result_builder #(
    parameter integer COORD_BITS  = 14,
    parameter integer CLASS_BITS  = 3,
    parameter integer SCORE_BITS  = 8,
    parameter integer ANCHOR_BITS = 15,
    parameter integer MAX_DET     = 100
)(
    input  logic clk, rst_n, frame_start,
    input  logic [15:0] frame_id,
    input  logic first_byte_pulse, last_payload_pulse,

    input  logic det_valid,
    input  logic [COORD_BITS-1:0]  det_x1, det_y1, det_x2, det_y2,
    input  logic [CLASS_BITS-1:0]  det_cls,
    input  logic [SCORE_BITS-1:0]  det_score,
    input  logic [ANCHOR_BITS-1:0] det_anchor_idx,
    input  logic [$clog2(MAX_DET+1)-1:0] n_det,
    input  logic overflow,
    input  logic compute_done,
    input  logic packet_done,   // parser finished incl. CRC check
    input  logic crc_error,
    input  logic format_error,

    output logic result_ready,
    input  logic tx_start,
    input  logic tx_ready,
    output logic tx_valid,
    output logic [7:0] tx_data,
    output logic tx_last,

    output logic [31:0] cycles_first_byte_to_done,
    output logic [31:0] cycles_last_byte_to_done
);

    localparam integer DET_IDX_W = (MAX_DET <= 1) ? 1 : $clog2(MAX_DET);

    function automatic [15:0] crc_next(input [15:0] crc_in, input [7:0] d);
        integer i;
        reg [15:0] c;
        begin
            c = crc_in ^ {d, 8'h00};
            for (i = 0; i < 8; i = i + 1)
                c = c[15] ? ((c << 1) ^ 16'h1021) : (c << 1);
            crc_next = c;
        end
    endfunction

    // Packet detection record is exactly 12 bytes / 96 bits:
    // x1, y1, x2, y2, cls, score, anchor_idx (all packet fields padded to 16/8 bits).
    // Intel/Quartus-specific hint. The coding style also permits normal RAM inference.
    (* ramstyle = "M9K" *) logic [95:0] det_mem [0:MAX_DET-1];

    logic [31:0] cyc, first_cyc, last_cyc;
    logic [DET_IDX_W:0] det_wr_count;

    logic [3:0] header_idx;
    logic [DET_IDX_W-1:0] det_idx;
    logic [3:0] det_byte_idx;
    logic [15:0] crc;
    logic [95:0] det_word;
    logic det_word_valid;

    // The compute core can finish BEFORE the two input-CRC bytes arrive.
    // The result (and its crc_error status bit) is only released once
    // both compute_done and packet_done have been seen for this packet.
    logic comp_seen, pkt_seen, result_fired;
    wire  comp_now = comp_seen | compute_done;
    wire  pkt_now  = pkt_seen  | packet_done;

    typedef enum logic [2:0] {
        TX_IDLE,
        TX_HEADER,
        TX_DET_LOAD,
        TX_DET,
        TX_CRC_HI,
        TX_CRC_LO
    } tx_state_t;
    tx_state_t tx_state;

    wire [7:0] status = {
        3'b0,
        (n_det == MAX_DET),
        overflow,
        format_error,
        crc_error,
        ~(crc_error | format_error)
    };

    // Header byte mux: 12 body bytes before detection records.
    logic [7:0] header_byte;
    always_comb begin
        header_byte = 8'h00;
        case (header_idx)
            4'd0:  header_byte = status;
            4'd1:  header_byte = frame_id[15:8];
            4'd2:  header_byte = frame_id[7:0];
            4'd3:  header_byte = {{(8-$clog2(MAX_DET+1)){1'b0}}, n_det};
            4'd4:  header_byte = cycles_first_byte_to_done[31:24];
            4'd5:  header_byte = cycles_first_byte_to_done[23:16];
            4'd6:  header_byte = cycles_first_byte_to_done[15:8];
            4'd7:  header_byte = cycles_first_byte_to_done[7:0];
            4'd8:  header_byte = cycles_last_byte_to_done[31:24];
            4'd9:  header_byte = cycles_last_byte_to_done[23:16];
            4'd10: header_byte = cycles_last_byte_to_done[15:8];
            default: header_byte = cycles_last_byte_to_done[7:0];
        endcase
    end

    // Detection byte mux. No division or modulo is required.
    logic [7:0] det_byte;
    always_comb begin
        case (det_byte_idx)
            4'd0:  det_byte = det_word[95:88];
            4'd1:  det_byte = det_word[87:80];
            4'd2:  det_byte = det_word[79:72];
            4'd3:  det_byte = det_word[71:64];
            4'd4:  det_byte = det_word[63:56];
            4'd5:  det_byte = det_word[55:48];
            4'd6:  det_byte = det_word[47:40];
            4'd7:  det_byte = det_word[39:32];
            4'd8:  det_byte = det_word[31:24];
            4'd9:  det_byte = det_word[23:16];
            4'd10: det_byte = det_word[15:8];
            default: det_byte = det_word[7:0];
        endcase
    end

    always_comb begin
        tx_valid = 1'b0;
        tx_data  = 8'h00;
        tx_last  = 1'b0;

        case (tx_state)
            TX_HEADER: begin
                tx_valid = 1'b1;
                tx_data  = header_byte;
            end
            TX_DET: begin
                tx_valid = det_word_valid;
                tx_data  = det_byte;
            end
            TX_CRC_HI: begin
                tx_valid = 1'b1;
                tx_data  = crc[15:8];
            end
            TX_CRC_LO: begin
                tx_valid = 1'b1;
                tx_data  = crc[7:0];
                tx_last  = 1'b1;
            end
            default: begin end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cyc                       <= 32'd0;
            first_cyc                 <= 32'd0;
            last_cyc                  <= 32'd0;
            cycles_first_byte_to_done <= 32'd0;
            cycles_last_byte_to_done  <= 32'd0;
            det_wr_count              <= '0;
            result_ready              <= 1'b0;
            tx_state                  <= TX_IDLE;
            header_idx                <= 4'd0;
            det_idx                   <= '0;
            det_byte_idx              <= 4'd0;
            crc                       <= 16'hffff;
            det_word                  <= '0;
            det_word_valid            <= 1'b0;
            comp_seen                 <= 1'b0;
            pkt_seen                  <= 1'b0;
            result_fired              <= 1'b0;
        end else begin
            cyc <= cyc + 1'b1;

            if (frame_start) begin
                det_wr_count <= '0;
                result_ready <= 1'b0;
            end

            if (first_byte_pulse) begin
                first_cyc    <= cyc;
                comp_seen    <= 1'b0;
                pkt_seen     <= 1'b0;
                result_fired <= 1'b0;
            end
            else begin
                if (compute_done) comp_seen <= 1'b1;
                if (packet_done)  pkt_seen  <= 1'b1;

                if (comp_now && pkt_now && !result_fired) begin
                    result_ready <= 1'b1;
                    result_fired <= 1'b1;
                end
            end
            if (last_payload_pulse)
                last_cyc <= cyc;

            // One packed write per detection. Padding preserves the original
            // 12-byte packet format exactly.
            if (det_valid && det_wr_count < MAX_DET) begin
                det_mem[det_wr_count[DET_IDX_W-1:0]] <= {
                    {{(16-COORD_BITS){1'b0}}, det_x1},
                    {{(16-COORD_BITS){1'b0}}, det_y1},
                    {{(16-COORD_BITS){1'b0}}, det_x2},
                    {{(16-COORD_BITS){1'b0}}, det_y2},
                    {{(8-CLASS_BITS){1'b0}}, det_cls},
                    {{(8-SCORE_BITS){1'b0}}, det_score},
                    {{(16-ANCHOR_BITS){1'b0}}, det_anchor_idx}
                };
                det_wr_count <= det_wr_count + 1'b1;
            end

            // Latency counters keep their original meaning:
            // measured at compute_done, not at result release.
            if (compute_done) begin
                cycles_first_byte_to_done <= cyc - first_cyc;
                cycles_last_byte_to_done  <= cyc - last_cyc;
            end

            case (tx_state)
                TX_IDLE: begin
                    det_word_valid <= 1'b0;
                    if (tx_start && result_ready) begin
                        header_idx  <= 4'd0;
                        det_idx     <= '0;
                        det_byte_idx<= 4'd0;
                        crc         <= 16'hffff;
                        tx_state    <= TX_HEADER;
                    end
                end

                TX_HEADER: begin
                    if (tx_ready) begin
                        crc <= crc_next(crc, header_byte);
                        if (header_idx == 4'd11) begin
                            if (n_det == 0) begin
                                tx_state <= TX_CRC_HI;
                            end else begin
                                tx_state <= TX_DET_LOAD;
                            end
                        end else begin
                            header_idx <= header_idx + 1'b1;
                        end
                    end
                end

                // Explicit registered read stage encourages block-RAM inference.
                TX_DET_LOAD: begin
                    det_word       <= det_mem[det_idx];
                    det_word_valid <= 1'b1;
                    det_byte_idx   <= 4'd0;
                    tx_state       <= TX_DET;
                end

                TX_DET: begin
                    if (tx_ready && det_word_valid) begin
                        crc <= crc_next(crc, det_byte);
                        if (det_byte_idx == 4'd11) begin
                            det_word_valid <= 1'b0;
                            if (det_idx + 1'b1 >= n_det) begin
                                tx_state <= TX_CRC_HI;
                            end else begin
                                det_idx  <= det_idx + 1'b1;
                                tx_state <= TX_DET_LOAD;
                            end
                        end else begin
                            det_byte_idx <= det_byte_idx + 1'b1;
                        end
                    end
                end

                TX_CRC_HI: begin
                    if (tx_ready)
                        tx_state <= TX_CRC_LO;
                end

                TX_CRC_LO: begin
                    if (tx_ready) begin
                        result_ready <= 1'b0;
                        tx_state     <= TX_IDLE;
                    end
                end

                default: tx_state <= TX_IDLE;
            endcase
        end
    end

endmodule
