//! Bare-Metal Alif Ensemble E3/E4 `M55_HE` (`thumbv8m.main-none-eabihf`, Cortex-M55 + Helium)
//! Cycle & Shunt-Trigger Harness (`Rust 1.100.0`, `edition = "2024"`, zero external crates).
//!
//! Copies `SEED_KNOWLEDGE` and `SEED_AGENCY` from non-volatile on-chip `MRAM` (`0x8000_0000`)
//! into 8-byte-aligned zero-wait-state `DTCM` (`0x2000_0000`), enables Armv8.1-M `DWT->CYCCNT`,
//! frames the four canonical Dual-Reality windows + 64-lane vertical bit-sliced `eval_batch64`
//! inside `LPGPIO_0` (`0x4200_2000`) pulses for inline `PPK2` / `INA228` physical rail integration,
//! commits `BENCH_RESULTS` to `DTCM` for zero-UART SWD readout, and enters `WFI` sleep.

#![no_std]
#![no_main]

use core::arch::asm;
use core::hint::black_box;
use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

use lut6_core::bootstrap::SEED_AGENCY;
use lut6_core::ratchet_demo::{SEED_KNOWLEDGE, synthesize_wobble_patch};
use lut6_core::{CompactLut6Page, Nucleus, PageLedger, SyncCell};

// Armv8.1-M Debug Control Block & Data Watchpoint/Trace MMIO (`DWT->CYCCNT`)
const DCB_DEMCR: *mut u32 = 0xE000_EDFC as *mut u32;
const DEMCR_TRCENA: u32 = 1 << 24;
const DWT_CTRL: *mut u32 = 0xE000_1000 as *mut u32;
const DWT_CYCCNT: *mut u32 = 0xE000_1004 as *mut u32;
const DWT_CTRL_CYCCNTENA: u32 = 1 << 0;

// Alif Ensemble LPGPIO (Always-On domain @ 0x4200_2000 on E3/E4 & OpenMV AE3)
const LPGPIO_BASE: usize = 0x4200_2000;
const GPIO_SWPORTA_DR: *mut u32 = (LPGPIO_BASE + 0x00) as *mut u32;
const GPIO_SWPORTA_DDR: *mut u32 = (LPGPIO_BASE + 0x04) as *mut u32;
const SHUNT_TRIG_PIN: u32 = 1 << 0;

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

/// Hot DTCM mirrors (`0x2000_0000`) for zero-wait-state single-cycle 64-bit LUT6 loads.
#[unsafe(link_section = ".lut6_dtcm")]
static DTCM_KNOWLEDGE: SyncCell<CompactLut6Page> = SyncCell::new(SEED_KNOWLEDGE);

#[unsafe(link_section = ".lut6_dtcm")]
static DTCM_AGENCY: SyncCell<CompactLut6Page> = SyncCell::new(SEED_AGENCY);

/// Telemetry struct readable straight over SWD (`probe-rs read` / GDB) without UART power draw.
#[repr(C)]
pub struct BenchReport {
    pub cycles_overhead_1k: u32,
    pub cycles_innate_hit_1k: u32,
    pub cycles_miss_escalate_1k: u32,
    pub cycles_refine_gate_once: u32,
    pub cycles_wobble_hit_1k: u32,
    pub cycles_batch64_1k_evals: u32,
    pub verified_hits: u32,
}

#[unsafe(no_mangle)]
pub static BENCH_RESULTS: SyncCell<BenchReport> = SyncCell::new(BenchReport {
    cycles_overhead_1k: 0,
    cycles_innate_hit_1k: 0,
    cycles_miss_escalate_1k: 0,
    cycles_refine_gate_once: 0,
    cycles_wobble_hit_1k: 0,
    cycles_batch64_1k_evals: 0,
    verified_hits: 0,
});

#[inline(always)]
fn dwt_enable_and_reset() {
    unsafe {
        let demcr = read_volatile(DCB_DEMCR);
        write_volatile(DCB_DEMCR, demcr | DEMCR_TRCENA);
        write_volatile(DWT_CYCCNT, 0);
        let ctrl = read_volatile(DWT_CTRL);
        write_volatile(DWT_CTRL, ctrl | DWT_CTRL_CYCCNTENA);
        asm!("dsb sy", "isb sy", options(nomem, nostack));
    }
}

#[inline(always)]
fn dwt_cycles() -> u32 {
    unsafe { read_volatile(DWT_CYCCNT) }
}

#[inline(always)]
fn trig_high() {
    unsafe {
        let v = read_volatile(GPIO_SWPORTA_DR);
        write_volatile(GPIO_SWPORTA_DR, v | SHUNT_TRIG_PIN);
        asm!("dsb sy", options(nomem, nostack));
    }
}

#[inline(always)]
fn trig_low() {
    unsafe {
        asm!("dsb sy", options(nomem, nostack));
        let v = read_volatile(GPIO_SWPORTA_DR);
        write_volatile(GPIO_SWPORTA_DR, v & !SHUNT_TRIG_PIN);
    }
}

#[inline(always)]
fn quiet_gap_cycles(n: u32) {
    let start = dwt_cycles();
    while dwt_cycles().wrapping_sub(start) < n {
        unsafe { asm!("nop", options(nomem, nostack, preserves_flags)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Reset() -> ! {
    // 0. Initialize .bss and .data in DTCM
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
    }

    // 1. Enable Armv8.1-M DWT cycle counter & configure LPGPIO_0 shunt trigger
    dwt_enable_and_reset();
    unsafe {
        let ddr = read_volatile(GPIO_SWPORTA_DDR);
        write_volatile(GPIO_SWPORTA_DDR, ddr | SHUNT_TRIG_PIN);
    }
    trig_low();

    // 2. Hydrate DTCM crystals from non-volatile MRAM seed copies
    let (k_root, a_root) = unsafe {
        write_volatile(DTCM_KNOWLEDGE.get(), SEED_KNOWLEDGE);
        write_volatile(DTCM_AGENCY.get(), SEED_AGENCY);
        asm!("dsb sy", "isb sy", options(nomem, nostack));
        (&*DTCM_KNOWLEDGE.get(), &*DTCM_AGENCY.get())
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

    // Calibration: 1,000-iteration black_box loop overhead
    let t0 = dwt_cycles();
    for i in 0..1000u64 {
        black_box(i);
    }
    let overhead_1k = dwt_cycles().wrapping_sub(t0);

    // WINDOW 1: 1,000 Certified Innate Hits (`0x08 -> 0xA1` in zero-wait DTCM)
    trig_high();
    let t1 = dwt_cycles();
    for _ in 0..1000 {
        let act = nucleus.step(black_box(0x08), black_box(1));
        black_box(act);
    }
    let innate_1k = dwt_cycles().wrapping_sub(t1);
    trig_low();
    quiet_gap_cycles(16_000);

    // WINDOW 2: 1,000 Frontier Misses -> Agency Escalation (`0x34 -> WakeRefiner`)
    trig_high();
    let t2 = dwt_cycles();
    for _ in 0..1000 {
        let act = nucleus.step(black_box(0x34), black_box(0));
        black_box(act);
    }
    let miss_1k = dwt_cycles().wrapping_sub(t2);
    trig_low();
    quiet_gap_cycles(16_000);

    // WINDOW 3: Single On-Device Refinement Admission (64-pt Integer Walsh FWHT + Anchor Gate)
    let candidate_patch = synthesize_wobble_patch(black_box(0x34), black_box(0x5E));
    trig_high();
    let t3 = dwt_cycles();
    let admitted = nucleus.refine_knowledge(candidate_patch, 16);
    let refine_cycles = dwt_cycles().wrapping_sub(t3);
    trig_low();
    if !admitted {
        loop {
            unsafe { asm!("bkpt #0", options(nomem, nostack)) };
        }
    }
    quiet_gap_cycles(16_000);

    // WINDOW 4: 1,000 Post-Compilation Wobble Hits (`0x37` noisy variant -> `0x5E`)
    trig_high();
    let t4 = dwt_cycles();
    for _ in 0..1000 {
        let act = nucleus.step(black_box(0x37), black_box(0));
        black_box(act);
    }
    let wobble_1k = dwt_cycles().wrapping_sub(t4);
    trig_low();
    quiet_gap_cycles(16_000);

    // WINDOW 5: Vertical Bit-Sliced Batch-64 (`16 * 64 = 1,024` LUT6 page evaluations)
    let mut batch_in = [0x08u64; 64];
    let mut lane = 0usize;
    while lane < 64 {
        batch_in[lane] = 0x08 + ((lane as u64) & 0x3);
        lane += 1;
    }
    trig_high();
    let t5 = dwt_cycles();
    for _ in 0..16 {
        let out64 = k_root.eval_batch64(black_box(&batch_in));
        black_box(out64);
    }
    let batch64_1k = dwt_cycles().wrapping_sub(t5);
    trig_low();

    unsafe {
        write_volatile(
            BENCH_RESULTS.get(),
            BenchReport {
                cycles_overhead_1k: overhead_1k,
                cycles_innate_hit_1k: innate_1k.saturating_sub(overhead_1k),
                cycles_miss_escalate_1k: miss_1k.saturating_sub(overhead_1k),
                cycles_refine_gate_once: refine_cycles,
                cycles_wobble_hit_1k: wobble_1k.saturating_sub(overhead_1k),
                cycles_batch64_1k_evals: batch64_1k,
                verified_hits: nucleus.k_ledger.hits,
            },
        );
        asm!("dsb sy", options(nomem, nostack));
    }

    // Metabolic Homeostasis (`CODON_STOP_HALT`): drop M55_HE into ultra-low-power WFI sleep
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
