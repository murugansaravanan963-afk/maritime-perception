module yolov8_mode_b_packet_top(
 input logic clk,rst_n,
 input logic rx_valid,output logic rx_ready,input logic [7:0] rx_data,
 output logic result_ready,input logic result_read_start,
 input logic tx_ready,output logic tx_valid,
 output logic [7:0] tx_data,output logic tx_last
);
 `include "rtl_params.vh"

 logic fs,cv,cr,pdone,crc_err,fmt_err,fbp,lpp;
 logic [15:0] fid,ncand;

 logic [ANCHOR_BITS-1:0] ai;
 logic [CLASS_BITS-1:0] cl;
 logic [7:0] sc;
 logic [7:0] bb[0:63];

 logic dv;
 logic [COORD_BITS-1:0] x1,y1,x2,y2;
 logic [CLASS_BITS-1:0] dcl;
 logic [7:0] dsc;
 logic [ANCHOR_BITS-1:0] dai;

 logic [$clog2(MAX_DET+1)-1:0] nd;
 logic ov,done;

 logic [31:0] c1,c2;


 // ============================================================
 // MODE-B V2 EMPTY-FRAME BYPASS
 // ============================================================

 logic empty_frame_done;
 logic builder_done;
 logic builder_last_payload;

 logic [$clog2(MAX_DET+1)-1:0] builder_n_det;
 logic builder_overflow;


 // packet_done occurs after CRC bytes have been received.
 //
 // A valid packet with zero candidates has no DFL / Decode / NMS
 // work to perform. Complete it directly at the protocol layer.
    // A format-error packet also gets an immediate (empty) reply with
    // the format_error status bit, so the host is never left waiting.
    assign empty_frame_done =
        pdone
        && ((ncand == 16'd0) || fmt_err);


 // Normal frame:
 //     compute_done comes from the existing compute core.
 //
 // Empty frame:
 //     completion comes directly from packet validation.
 assign builder_done =
        done | empty_frame_done;


 // Empty frames have no payload byte.
 //
 // Use frame_start as the logical payload boundary so that
 // cycles_last_byte_to_done remains defined.
 //
 // Normal non-empty behavior is unchanged.
 assign builder_last_payload =
        lpp | (fs && (ncand == 16'd0));


 // Do not allow stale NMS state to appear in an empty-frame reply.
 assign builder_n_det =
        ((ncand == 16'd0) || fmt_err) ? '0 : nd;

 assign builder_overflow =
        ((ncand == 16'd0) || fmt_err) ? 1'b0 : ov;


 // ============================================================
 // MODE-B PACKET PARSER
 // ============================================================

 mode_b_packet_parser #(
   .CLASS_BITS(CLASS_BITS),
   .ANCHOR_BITS(ANCHOR_BITS)
 ) u_p(
   .clk(clk),
   .rst_n(rst_n),

   .byte_valid(rx_valid),
   .byte_ready(rx_ready),
   .byte_data(rx_data),

   .frame_start(fs),
   .frame_id(fid),
   .n_candidates(ncand),

   .cand_valid(cv),
   .cand_ready(cr),

   .cand_anchor_idx(ai),
   .cand_cls(cl),
   .cand_score(sc),
   .cand_bbox(bb),

   .packet_done(pdone),
   .crc_error(crc_err),
   .format_error(fmt_err),

   .first_byte_pulse(fbp),
   .last_payload_pulse(lpp)
 );


 // ============================================================
 // FROZEN MODE-B COMPUTE CORE
 //
 // Non-empty frames continue through the existing:
 //
 // DFL
 //   -> sparse decode
 //   -> candidate collection
 //   -> NMS
 //
 // This path is unchanged.
 // ============================================================

 mode_b_compute_top #(
   .LUT_DEPTH(LUT_DEPTH),
   .LUT_FILE("dfl_exp_lut.mem"),

   .NUM_CLASSES(NUM_CLASSES),
   .CLASS_BITS(CLASS_BITS),

   .COORD_BITS(COORD_BITS),
   .ANCHOR_BITS(ANCHOR_BITS),

   .IMG_W(IMG_W),
   .IMG_H(IMG_H),

   .MAX_CAND(MAX_CAND),
   .MAX_DET(MAX_DET),

   .IOU_NUM(IOU_NUM),
   .IOU_DEN(IOU_DEN),

   .AREA_BITS(AREA_BITS)
 ) u_c(
   .clk(clk),
   .rst_n(rst_n),

   .frame_start(fs),
   .n_candidates(ncand),

   .in_valid(cv),
   .in_ready(cr),

   .in_anchor_idx(ai),
   .in_cls(cl),
   .in_score(sc),
   .in_bbox(bb),

   .det_valid(dv),

   .det_x1(x1),
   .det_y1(y1),
   .det_x2(x2),
   .det_y2(y2),

   .det_cls(dcl),
   .det_score(dsc),
   .det_anchor_idx(dai),

   .n_det(nd),
   .overflow(ov),

   .done(done)
 );


 // ============================================================
 // RESULT PACKET BUILDER
 // ============================================================

 mode_b_result_builder #(
   .COORD_BITS(COORD_BITS),
   .CLASS_BITS(CLASS_BITS),
   .ANCHOR_BITS(ANCHOR_BITS),
   .MAX_DET(MAX_DET)
 ) u_r(
   .clk(clk),
   .rst_n(rst_n),

   .frame_start(fs),
   .frame_id(fid),

   .first_byte_pulse(fbp),
   .last_payload_pulse(builder_last_payload),

   .det_valid(dv),

   .det_x1(x1),
   .det_y1(y1),
   .det_x2(x2),
   .det_y2(y2),

   .det_cls(dcl),
   .det_score(dsc),
   .det_anchor_idx(dai),

   .n_det(builder_n_det),
   .overflow(builder_overflow),

   .compute_done(builder_done),
   .packet_done(pdone),

   .crc_error(crc_err),
   .format_error(fmt_err),

   .result_ready(result_ready),

   .tx_start(result_read_start),
   .tx_ready(tx_ready),

   .tx_valid(tx_valid),
   .tx_data(tx_data),
   .tx_last(tx_last),

   .cycles_first_byte_to_done(c1),
   .cycles_last_byte_to_done(c2)
 );

endmodule