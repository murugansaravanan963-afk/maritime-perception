`timescale 1ns/1ps
module mode_b_socket_tb;
  logic clk=0, rst_n=0, rx_valid=0, rx_ready;
  logic [7:0] rx_data=0;
  logic result_ready, result_read_start=0, tx_ready=1, tx_valid, tx_last;
  logic [7:0] tx_data;
  integer fd, fo, r, nbytes=0, i;
  reg [7:0] pkt [0:300000];

  always #3.333 clk = ~clk;

  yolov8_mode_b_packet_top dut(
    .clk(clk),.rst_n(rst_n),
    .rx_valid(rx_valid),.rx_ready(rx_ready),.rx_data(rx_data),
    .result_ready(result_ready),.result_read_start(result_read_start),
    .tx_ready(tx_ready),.tx_valid(tx_valid),.tx_data(tx_data),.tx_last(tx_last)
  );

  task automatic send_byte(input [7:0] b);
    begin
      @(negedge clk); rx_data=b; rx_valid=1'b1;
      do @(posedge clk); while(!rx_ready);
      @(negedge clk); rx_valid=1'b0;
    end
  endtask

  initial begin
    fd=$fopen("rx_packet.hex","r");
    if(fd==0) $fatal(1,"Cannot open rx_packet.hex");
    while(!$feof(fd)) begin
      r=$fscanf(fd,"%h",pkt[nbytes]);
      if(r==1) nbytes=nbytes+1;
    end
    $fclose(fd);
    if(nbytes<18) $fatal(1,"rx_packet.hex too short: %0d bytes",nbytes);
    $display("[SOCKET_TB] Loaded %0d Mode-B bytes",nbytes);

    repeat(8) @(posedge clk); rst_n=1'b1; repeat(3) @(posedge clk);
    for(i=0;i<nbytes;i=i+1) send_byte(pkt[i]);

    wait(result_ready===1'b1);
    $display("[SOCKET_TB] RTL result ready");
    fo=$fopen("tx_reply.hex","w");
    if(fo==0) $fatal(1,"Cannot create tx_reply.hex");
    @(negedge clk); result_read_start=1'b1;
    @(negedge clk); result_read_start=1'b0;
    while(1) begin
      @(posedge clk);
      if(tx_valid) begin
        $fwrite(fo,"%02x\n",tx_data);
        if(tx_last) begin
          $fclose(fo);
          $display("[SOCKET_TB] Reply complete -> tx_reply.hex");
          $finish;
        end
      end
    end
  end
endmodule
