//! Bare-Metal Raspberry Pi Pico `RP2040` (`thumbv6m-none-eabi`, Cortex-M0+)
//! Striped-SRAM Cycle & `SIO` GPIO Shunt-Trigger Harness (`Rust 1.100.0`, `edition = "2024"`).

#![no_std]
#![no_main]

use core::arch::asm;
use core::hint::black_box;
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

use lut6_core::bootstrap::SEED_AGENCY;
use lut6_core::ratchet_demo::{SEED_KNOWLEDGE, synthesize_wobble_patch};
use lut6_core::{CompactLut6Page, Nucleus, PageLedger, SyncCell};

// Armv6-M SysTick 24-bit CPU-clock cycle counter + RP2040 1 MHz microsecond timer + SIO GPIO
const SYST_CSR: *mut u32 = 0xE000_E010 as *mut u32;
const SYST_RVR: *mut u32 = 0xE000_E014 as *mut u32;
const SYST_CVR: *mut u32 = 0xE000_E018 as *mut u32;
const SIO_GPIO_OUT_SET: *mut u32 = 0xD000_0014 as *mut u32;
const SIO_GPIO_OUT_CLR: *mut u32 = 0xD000_0018 as *mut u32;
const SIO_GPIO_OE_SET: *mut u32 = 0xD000_0024 as *mut u32;
const SHUNT_TRIG_GPIO2: u32 = 1 << 2;

#[repr(C)]
pub struct VectorTable {
    pub initial_sp: unsafe extern "C" fn(),
    pub reset: unsafe extern "C" fn() -> !,
}

unsafe extern "C" {
    fn _stack_start();
    static mut _sbss: u32;
    static mut _ebss: u32;
    static mut _sdata: u32;
    static mut _edata: u32;
    static _sidata: u32;
}

#[unsafe(link_section = ".vector_table")]
#[used]
pub static VECTOR_TABLE: VectorTable = VectorTable {
    initial_sp: _stack_start,
    reset: Reset,
};

#[unsafe(link_section = ".data.sram_lut6")]
static SRAM_KNOWLEDGE: SyncCell<CompactLut6Page> = SyncCell::new(SEED_KNOWLEDGE);

#[unsafe(link_section = ".data.sram_lut6")]
static SRAM_AGENCY: SyncCell<CompactLut6Page> = SyncCell::new(SEED_AGENCY);

#[repr(C)]
pub struct PicoBenchReport {
    pub cycles_innate_hit_64: u32,
    pub cycles_miss_escalate_64: u32,
    pub cycles_refine_gate_once: u32,
    pub cycles_wobble_hit_64: u32,
    pub cycles_batch64_once: u32,
    pub verified_hits: u32,
}

#[unsafe(no_mangle)]
pub static PICO_BENCH_RESULTS: SyncCell<PicoBenchReport> = SyncCell::new(PicoBenchReport {
    cycles_innate_hit_64: 0,
    cycles_miss_escalate_64: 0,
    cycles_refine_gate_once: 0,
    cycles_wobble_hit_64: 0,
    cycles_batch64_once: 0,
    verified_hits: 0,
});

#[inline(always)]
fn systick_restart() {
    unsafe {
        write_volatile(SYST_CSR, 0);
        write_volatile(SYST_RVR, 0x00FF_FFFF);
        write_volatile(SYST_CVR, 0);
        write_volatile(SYST_CSR, 0x5); // Processor clock, enabled
    }
}

#[inline(always)]
fn systick_elapsed() -> u32 {
    let cur = unsafe { read_volatile(SYST_CVR) } & 0x00FF_FFFF;
    0x00FF_FFFFu32.wrapping_sub(cur)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Reset() -> ! {
    unsafe {
        let mut bss = &raw mut _sbss;
        let ebss = &raw mut _ebss;
        while bss < ebss {
            write_volatile(bss, 0);
            bss = bss.add(1);
        }
        let mut data = &raw mut _sdata;
        let edata = &raw mut _edata;
        let mut src = &raw const _sidata;
        while data < edata {
            write_volatile(data, read_volatile(src));
            data = data.add(1);
            src = src.add(1);
        }
        write_volatile(SIO_GPIO_OE_SET, SHUNT_TRIG_GPIO2);
        write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2);
    }

    let (k_root, a_root) = unsafe {
        write_volatile(SRAM_KNOWLEDGE.get(), SEED_KNOWLEDGE);
        write_volatile(SRAM_AGENCY.get(), SEED_AGENCY);
        (&*SRAM_KNOWLEDGE.get(), &*SRAM_AGENCY.get())
    };

    let mut nucleus = Nucleus {
        knowledge_root: k_root,
        agency_root: a_root,
        knowledge_overlay: None,
        agency_overlay: None,
        k_ledger: PageLedger::default(),
        a_ledger: PageLedger::default(),
        k_anchors: [(0x08, 0xFF, 0xA1), (0x0B, 0xFF, 0xA1)],
        a_anchors: [
            ((1u64 << 40) | (1u64 << 48), 0x3F, 0x20),
            (1u64 << 63, 0x3F, 0x3C),
        ],
    };

    unsafe { write_volatile(SIO_GPIO_OUT_SET, SHUNT_TRIG_GPIO2) };
    systick_restart();
    for _ in 0..64 {
        black_box(nucleus.step(black_box(0x08), black_box(1)));
    }
    let c_innate = systick_elapsed();
    unsafe { write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2) };

    unsafe { write_volatile(SIO_GPIO_OUT_SET, SHUNT_TRIG_GPIO2) };
    systick_restart();
    for _ in 0..64 {
        black_box(nucleus.step(black_box(0x34), black_box(0)));
    }
    let c_miss = systick_elapsed();
    unsafe { write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2) };

    let patch = synthesize_wobble_patch(black_box(0x34), black_box(0x5E));
    unsafe { write_volatile(SIO_GPIO_OUT_SET, SHUNT_TRIG_GPIO2) };
    systick_restart();
    let ok = nucleus.refine_knowledge(patch, 16);
    let c_refine = systick_elapsed();
    unsafe { write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2) };
    if !ok {
        loop {
            unsafe { asm!("bkpt #0", options(nomem, nostack)) };
        }
    }

    unsafe { write_volatile(SIO_GPIO_OUT_SET, SHUNT_TRIG_GPIO2) };
    systick_restart();
    for _ in 0..64 {
        black_box(nucleus.step(black_box(0x37), black_box(0)));
    }
    let c_wobble = systick_elapsed();
    unsafe { write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2) };

    let batch_in = [0x08u64; 64];
    unsafe { write_volatile(SIO_GPIO_OUT_SET, SHUNT_TRIG_GPIO2) };
    systick_restart();
    black_box(k_root.eval_batch64(black_box(&batch_in)));
    let c_batch = systick_elapsed();
    unsafe { write_volatile(SIO_GPIO_OUT_CLR, SHUNT_TRIG_GPIO2) };

    unsafe {
        write_volatile(
            PICO_BENCH_RESULTS.get(),
            PicoBenchReport {
                cycles_innate_hit_64: c_innate,
                cycles_miss_escalate_64: c_miss,
                cycles_refine_gate_once: c_refine,
                cycles_wobble_hit_64: c_wobble,
                cycles_batch64_once: c_batch,
                verified_hits: nucleus.k_ledger.hits,
            },
        );
    }

    loop {
        unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) };
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) };
    }
}
