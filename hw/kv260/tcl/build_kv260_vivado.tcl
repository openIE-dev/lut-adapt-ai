# =============================================================================
# Vivado Batch Synthesis, Zynq UltraScale+ BD Integration & Bitstream Generator
# Target: Kria KV260 SOM (`xck26-sfvc784-2LV-c` / `xilinx.com:kv260_som:part0:1.4`)
# Usage : vivado -mode batch -source hw/kv260/tcl/build_kv260_vivado.tcl
# =============================================================================

set proj_name "clai_kv260_rung0"
set build_dir "artifacts/vivado_kv260"
set part_name "xck26-sfvc784-2LV-c"

file mkdir $build_dir
create_project -force $proj_name $build_dir -part $part_name

# 1. Add Crystallized Rung 0 LUT6 Netlist + AXI4-Stream / AXI4-Lite Subsystem RTL
add_files -norecurse [list \
    "artifacts/clai_rung0_kv260.v" \
    "hw/kv260/rtl/clai_engram_streamer.sv" \
    "hw/kv260/rtl/clai_rung0_hotswap_core.sv" \
    "hw/kv260/rtl/clai_kv260_axi_top.sv" \
]
add_files -fileset constrs_1 -norecurse "hw/kv260/constraints/kv260_clai.xdc"
update_compile_order -fileset sources_1

# 2. Build Zynq UltraScale+ MPSoC Block Design (`zynq_ultra_ps_e` + `axi_dma` + `clai_kv260_axi_top`)
create_bd_design "kv260_clai_bd"

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__IRQ0      {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {250} \
] [get_bd_cells zynq_ps]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps_250m
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_pkt_dma
set_property -dict [list \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_sg   {0} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
] [get_bd_cells axi_pkt_dma]

create_bd_cell -type module -reference clai_kv260_axi_top clai_rung0_subsys

# Wire AXI4-Lite Control Plane (HPM0_FPD -> DMA + CLAI CSRs)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_slave  {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_xbar   {/zynq_ps/pl_clk0 (250 MHz)} \
    Master     {/zynq_ps/M_AXI_HPM0_FPD} \
    Slave      {/clai_rung0_subsys/s_axi} \
    ddr_seg    {Auto} \
    intc_ip    {New AXI SmartConnect} \
    master_apm {0} \
} [get_bd_intf_pins clai_rung0_subsys/s_axi]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_slave  {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_xbar   {/zynq_ps/pl_clk0 (250 MHz)} \
    Master     {/zynq_ps/M_AXI_HPM0_FPD} \
    Slave      {/axi_pkt_dma/S_AXI_LITE} \
    ddr_seg    {Auto} \
    intc_ip    {/axi_smc} \
    master_apm {0} \
} [get_bd_intf_pins axi_pkt_dma/S_AXI_LITE]

# Wire High-Bandwidth DDR Read Port for AXI DMA -> HP0_FPD
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_slave  {/zynq_ps/pl_clk0 (250 MHz)} \
    Clk_xbar   {/zynq_ps/pl_clk0 (250 MHz)} \
    Master     {/axi_pkt_dma/M_AXI_MM2S} \
    Slave      {/zynq_ps/S_AXI_HP0_FPD} \
    ddr_seg    {Auto} \
    intc_ip    {New AXI SmartConnect} \
    master_apm {0} \
} [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]

# Wire AXI4-Stream Packet Bus (DMA MM2S -> CLAI Engram Streamer) & Escalation Interrupt
connect_bd_intf_net [get_bd_intf_pins axi_pkt_dma/M_AXIS_MM2S] [get_bd_intf_pins clai_rung0_subsys/s_axis]
connect_bd_net [get_bd_pins clai_rung0_subsys/irq_escalate_to_ps] [get_bd_pins zynq_ps/pl_ps_irq0]

assign_bd_address -offset 0xA0000000 -range 0x00001000 [get_bd_addr_segs {zynq_ps/Data/SEG_clai_rung0_subsys_reg0}]
validate_bd_design
save_bd_design

make_wrapper -files [get_files $build_dir/$proj_name.srcs/sources_1/bd/kv260_clai_bd/kv260_clai_bd.bd] -top
add_files -norecurse $build_dir/$proj_name.gen/sources_1/bd/kv260_clai_bd/hdl/kv260_clai_bd_wrapper.v
set_property top kv260_clai_bd_wrapper [current_fileset]

# 3. Synthesize, Place & Route, Report Timing/Utilization/Power, & Export Overlay (.bit + .hwh)
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

open_run impl_1
report_utilization -file "$build_dir/kv260_utilization.rpt"
report_timing_summary -file "$build_dir/kv260_timing.rpt"
report_power -file "$build_dir/kv260_power.rpt"

file copy -force "$build_dir/$proj_name.runs/impl_1/kv260_clai_bd_wrapper.bit" "artifacts/clai_kv260.bit"
file copy -force "$build_dir/$proj_name.gen/sources_1/bd/kv260_clai_bd/hw_handoff/kv260_clai_bd.hwh" "artifacts/clai_kv260.hwh"
puts "\[CLAI KV260\] Overlay ready: artifacts/clai_kv260.bit + artifacts/clai_kv260.hwh"
