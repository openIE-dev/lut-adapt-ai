/* memory.x — Raspberry Pi Pico RP2040 (QSPI Flash + 264 KB Striped SRAM Banks 0..3 + Scratch X/Y) */
ENTRY(Reset);

MEMORY
{
    FLASH (rx)      : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    SRAM (rwx)      : ORIGIN = 0x20000000, LENGTH = 256K
    SCRATCH_X (rwx) : ORIGIN = 0x20040000, LENGTH = 4K
    SCRATCH_Y (rwx) : ORIGIN = 0x20041000, LENGTH = 4K
}

_stack_start = ORIGIN(SCRATCH_Y) + LENGTH(SCRATCH_Y);

SECTIONS
{
    .vector_table ORIGIN(FLASH) : ALIGN(8)
    {
        KEEP(*(.vector_table));
    } > FLASH

    .text : ALIGN(4)
    {
        *(.text .text.*);
        *(.rodata .rodata.*);
        . = ALIGN(8);
    } > FLASH

    /* Copy hot 4 KB .lut6 crystals into striped SRAM at boot to eliminate QSPI XIP cache stalls */
    .data.sram_lut6 (NOLOAD) : ALIGN(8)
    {
        *(.data.sram_lut6 .data.sram_lut6.*);
        . = ALIGN(8);
    } > SRAM

    .data : ALIGN(8)
    {
        _sdata = .;
        *(.data .data.*);
        . = ALIGN(8);
        _edata = .;
    } > SRAM AT > FLASH
    _sidata = LOADADDR(.data);

    .bss (NOLOAD) : ALIGN(8)
    {
        _sbss = .;
        *(.bss .bss.* COMMON);
        . = ALIGN(8);
        _ebss = .;
    } > SRAM
}
