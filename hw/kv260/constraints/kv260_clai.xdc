# =============================================================================
# Target: AMD/Xilinx Kria KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
# Module: clai_kv260_axi_top (Rung 0 Zero-MAC LUT6 Subsystem)
# =============================================================================

# 1. Primary 250 MHz PL Fabric Clock (`pl_clk0` from Zynq UltraScale+ PS)
create_clock -period 4.000 -name clk_pl_250m [get_ports aclk]
set_clock_uncertainty 0.150 [get_clocks clk_pl_250m]

# 2. Asynchronous Active-Low Reset False Path (synchronized upstream by proc_sys_reset)
set_false_path -from [get_ports aresetn]

# 3. Ensure Zero DSP48 Inference Across Entire Rung 0 Subsystem
set_property MAX_DSP 0 [current_design]

# 4. Pack `core_bus_reg` Close to CLB LUT6 Fabric for Single-Cycle 3-Stage Propagation
set_property SHREG_EXTRACT NO [get_cells -hier -filter {NAME =~ *core_bus_reg*}]
set_property RAM_STYLE BLOCK  [get_cells -hier -filter {NAME =~ *engram_bank*}]

# 5. Bitstream Compression & Thermal Power Optimization for Kria K26 SOM
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.OVERTEMPSHUTDOWN ENABLE [current_design]
