/* memory.x — Alif Ensemble E3/E4 M55_HE Core (Zero-Wait DTCM + On-Chip MRAM) */
ENTRY(Reset);

MEMORY
{
    /* 5.5 MB Non-Volatile On-Chip MRAM (CPU view via ATOC) */
    MRAM (rx)  : ORIGIN = 0x80000000, LENGTH = 512K
    /* 256 KB M55_HE Tightly Coupled Memory: ITCM @ 0x00000000, DTCM @ 0x20000000 */
    ITCM (rwx) : ORIGIN = 0x00000000, LENGTH = 256K
    DTCM (rw)  : ORIGIN = 0x20000000, LENGTH = 256K
}

_stack_start = ORIGIN(DTCM) + LENGTH(DTCM);

SECTIONS
{
    .vector_table ORIGIN(MRAM) : ALIGN(8)
    {
        KEEP(*(.vector_table));
    } > MRAM

    .text : ALIGN(4)
    {
        *(.text .text.*);
        *(.rodata .rodata.*);
        . = ALIGN(8);
    } > MRAM

    .lut6_dtcm (NOLOAD) : ALIGN(8)
    {
        *(.lut6_dtcm .lut6_dtcm.*);
        . = ALIGN(8);
    } > DTCM

    .data : ALIGN(8)
    {
        _sdata = .;
        *(.data .data.*);
        . = ALIGN(8);
        _edata = .;
    } > DTCM AT > MRAM
    _sidata = LOADADDR(.data);

    .bss (NOLOAD) : ALIGN(8)
    {
        _sbss = .;
        *(.bss .bss.* COMMON);
        . = ALIGN(8);
        _ebss = .;
    } > DTCM
}
