module mode_b_crc16_ccitt;
    function automatic [15:0] next_crc(input [15:0] crc_in, input [7:0] data);
        integer i;
        reg [15:0] c;
        begin
            c = crc_in ^ ({data,8'h00});
            for (i=0;i<8;i=i+1)
                c = c[15] ? ((c << 1) ^ 16'h1021) : (c << 1);
            next_crc = c;
        end
    endfunction
endmodule
