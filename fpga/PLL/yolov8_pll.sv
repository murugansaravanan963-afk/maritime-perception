`timescale 1 ps / 1 ps

module yolov8_pll (
    input  wire inclk0,
    input  wire areset,

    output wire c0,
    output wire locked
);

    wire [4:0] clk_bus;
    wire       locked_int;

    // ============================================================
    // MAX 10 ALTPLL
    //
    // Input:
    //     50 MHz
    //     period = 20 ns = 20000 ps
    //
    // Output c0:
    //     100 MHz
    //     multiply = 2
    //     divide   = 1
    //     phase    = 0
    //     duty     = 50 %
    // ============================================================

    altpll altpll_component (
        .inclk       ({1'b0, inclk0}),
        .areset      (areset),

        .clk         (clk_bus),
        .locked      (locked_int),

        .activeclock (),
        .clkbad      (),
        .clkena      ({6{1'b1}}),
        .clkloss     (),
        .clkswitch   (1'b0),

        .configupdate(1'b0),

        .enable0     (),
        .enable1     (),

        .extclk      (),

        .extclkena   ({4{1'b1}}),

        .fbin        (1'b1),
        .fbmimicbidir(),

        .fbout       (),
        .fref        (),

        .icdrclk     (),

        .pfdena      (1'b1),

        .phasecounterselect ({4{1'b0}}),
        .phasedone   (),
        .phasestep   (1'b0),
        .phaseupdown (1'b0),

        .pllena      (1'b1),

        .scanaclr    (1'b0),
        .scanclk     (1'b0),
        .scanclkena  (1'b1),
        .scandata    (1'b0),
        .scandataout (),
        .scandone    (),
        .scanread    (1'b0),
        .scanwrite   (1'b0),

        .sclkout0    (),
        .sclkout1    (),

        .vcooverrange(),
        .vcounderrange()
    );


    // ============================================================
    // ALTPLL parameters
    // ============================================================

    defparam
        altpll_component.bandwidth_type =
            "AUTO",

        altpll_component.clk0_divide_by =
            1,

        altpll_component.clk0_duty_cycle =
            50,

        altpll_component.clk0_multiply_by =
            2,

        altpll_component.clk0_phase_shift =
            "0",

        altpll_component.compensate_clock =
            "CLK0",

        altpll_component.inclk0_input_frequency =
            20000,

        altpll_component.intended_device_family =
            "MAX 10",

        altpll_component.operation_mode =
            "NORMAL",

        altpll_component.pll_type =
            "AUTO",

        altpll_component.port_activeclock =
            "PORT_UNUSED",

        altpll_component.port_areset =
            "PORT_USED",

        altpll_component.port_clkbad0 =
            "PORT_UNUSED",

        altpll_component.port_clkbad1 =
            "PORT_UNUSED",

        altpll_component.port_clkloss =
            "PORT_UNUSED",

        altpll_component.port_clkswitch =
            "PORT_UNUSED",

        altpll_component.port_configupdate =
            "PORT_UNUSED",

        altpll_component.port_fbin =
            "PORT_UNUSED",

        altpll_component.port_inclk0 =
            "PORT_USED",

        altpll_component.port_inclk1 =
            "PORT_UNUSED",

        altpll_component.port_locked =
            "PORT_USED",

        altpll_component.port_pfdena =
            "PORT_UNUSED",

        altpll_component.port_phasecounterselect =
            "PORT_UNUSED",

        altpll_component.port_phasedone =
            "PORT_UNUSED",

        altpll_component.port_phasestep =
            "PORT_UNUSED",

        altpll_component.port_phaseupdown =
            "PORT_UNUSED",

        altpll_component.port_pllena =
            "PORT_UNUSED",

        altpll_component.port_scanaclr =
            "PORT_UNUSED",

        altpll_component.port_scanclk =
            "PORT_UNUSED",

        altpll_component.port_scanclkena =
            "PORT_UNUSED",

        altpll_component.port_scandata =
            "PORT_UNUSED",

        altpll_component.port_scandataout =
            "PORT_UNUSED",

        altpll_component.port_scandone =
            "PORT_UNUSED",

        altpll_component.port_scanread =
            "PORT_UNUSED",

        altpll_component.port_scanwrite =
            "PORT_UNUSED",

        altpll_component.port_clk0 =
            "PORT_USED",

        altpll_component.port_clk1 =
            "PORT_UNUSED",

        altpll_component.port_clk2 =
            "PORT_UNUSED",

        altpll_component.port_clk3 =
            "PORT_UNUSED",

        altpll_component.port_clk4 =
            "PORT_UNUSED",

        altpll_component.port_clk5 =
            "PORT_UNUSED",

        altpll_component.width_clock =
            5;


    assign c0 =
        clk_bus[0];


    assign locked =
        locked_int;

endmodule