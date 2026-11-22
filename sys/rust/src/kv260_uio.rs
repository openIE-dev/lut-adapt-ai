//! Native Rust Zynq UltraScale+ PS Linux `/dev/uio` Driver & Cycle-Accurate Loopback HAL
//! for `clai_kv260_axi_top` (`0xA000_0000` on AMD/Xilinx Kria KV260).
//!
//! Supports both:
//!   1. Real `/dev/uioX` POSIX `mmap` + blocking GIC interrupt wait (`LinuxUioBackend`)
//!   2. Cycle-accurate in-memory RTL behavioral mirror (`LoopbackKv260Hal`) for macOS/CI testing.

#![allow(clippy::needless_range_loop)]

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;

use crate::collapse::CompiledMetalRung0;

pub const CLAI_KV260_MAGIC: u32 = 0x434C_4149; // "CLAI"
pub const CSR_CTRL: usize = 0x000;
pub const CSR_STATUS: usize = 0x004;
pub const CSR_MARGIN_THRESH: usize = 0x008;
pub const CSR_RESULT: usize = 0x00C;
pub const CSR_TOTAL_EPISODES: usize = 0x010;
pub const CSR_SATIATED_CNT: usize = 0x014;
pub const CSR_ESCALATED_CNT: usize = 0x018;
pub const CSR_HS_CTRL: usize = 0x050;
pub const CSR_HS_TABLE_LO: usize = 0x054;
pub const CSR_HS_TABLE_HI: usize = 0x058;
pub const CSR_HS_WIRES_LO: usize = 0x05C;
pub const CSR_HS_WIRES_HI: usize = 0x060;
pub const CSR_HS_GENERATION: usize = 0x064;
pub const CSR_MAGIC: usize = 0x080;
pub const CSR_BUS512_BASE: usize = 0x100;

pub trait Kv260RegisterHal {
    fn read_reg(&self, byte_offset: usize) -> u32;
    fn write_reg(&mut self, byte_offset: usize, val: u32);
    fn wait_escalation_irq(&mut self) -> io::Result<u32>;
}

/// Zero-copy POSIX `/dev/uioX` mmap backend for Kria KV260 Ubuntu 24.04 / PetaLinux.
pub struct LinuxUioHal {
    uio_file: File,
    map_ptr: *mut u32,
}

unsafe impl Send for LinuxUioHal {}

impl LinuxUioHal {
    pub fn open(path: &str) -> io::Result<Self> {
        let uio_file = OpenOptions::new().read(true).write(true).open(path)?;
        let ptr = unsafe {
            libc_mmap(
                std::ptr::null_mut(),
                0x1000,
                0x3, // PROT_READ | PROT_WRITE
                0x1, // MAP_SHARED
                uio_file.as_raw_fd(),
                0,
            )
        };
        if ptr.is_null() || (ptr as isize) == -1 {
            return Err(io::Error::last_os_error());
        }
        let hal = Self {
            uio_file,
            map_ptr: ptr as *mut u32,
        };
        let magic = hal.read_reg(CSR_MAGIC);
        if magic != CLAI_KV260_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid KV260 CLAI magic: 0x{magic:08X}"),
            ));
        }
        Ok(hal)
    }
}

impl Drop for LinuxUioHal {
    fn drop(&mut self) {
        unsafe {
            let _ = libc_munmap(self.map_ptr as *mut core::ffi::c_void, 0x1000);
        }
    }
}

unsafe extern "C" {
    #[link_name = "mmap"]
    fn libc_mmap(
        addr: *mut core::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut core::ffi::c_void;

    #[link_name = "munmap"]
    fn libc_munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
}

impl Kv260RegisterHal for LinuxUioHal {
    #[inline]
    fn read_reg(&self, byte_offset: usize) -> u32 {
        assert!(byte_offset < 0x1000 && byte_offset.is_multiple_of(4));
        unsafe { std::ptr::read_volatile(self.map_ptr.add(byte_offset >> 2)) }
    }

    #[inline]
    fn write_reg(&mut self, byte_offset: usize, val: u32) {
        assert!(byte_offset < 0x1000 && byte_offset.is_multiple_of(4));
        unsafe { std::ptr::write_volatile(self.map_ptr.add(byte_offset >> 2), val) }
    }

    fn wait_escalation_irq(&mut self) -> io::Result<u32> {
        self.uio_file.write_all(&1u32.to_ne_bytes())?;
        let mut buf = [0u8; 4];
        self.uio_file.read_exact(&mut buf)?;
        self.write_reg(CSR_STATUS, 0x5); // W1C acknowledge
        Ok(u32::from_ne_bytes(buf))
    }
}

/// Bit-exact behavioral mirror of `clai_kv260_axi_top` + `clai_rung0_hotswap_core` for host CI.
pub struct LoopbackKv260Hal {
    regs: [u32; 1024],
    l0_table: [[u64; 64]; 2],
    l0_wires: [[[u16; 6]; 64]; 2],
    l1_table: [[u64; 64]; 2],
    l1_wires: [[[u16; 6]; 64]; 2],
    active_bank: usize,
    swap_gen: u32,
    irq_cnt: u32,
}

impl Default for LoopbackKv260Hal {
    fn default() -> Self {
        Self::new()
    }
}

impl LoopbackKv260Hal {
    pub fn new() -> Self {
        let mut s = Self {
            regs: [0u32; 1024],
            l0_table: [[0xAAAA_AAAA_AAAA_AAAAu64; 64]; 2],
            l0_wires: [[[0u16; 6]; 64]; 2],
            l1_table: [[0xFFF8_F880_F880_8000u64; 64]; 2],
            l1_wires: [[[0u16; 6]; 64]; 2],
            active_bank: 0,
            swap_gen: 0,
            irq_cnt: 0,
        };
        s.regs[CSR_MAGIC >> 2] = CLAI_KV260_MAGIC;
        s.regs[CSR_MARGIN_THRESH >> 2] = 4;
        s
    }

    pub fn inject_bus512_and_evaluate(&mut self, words: &[u64; 8]) {
        for h in 0..8 {
            self.regs[(CSR_BUS512_BASE >> 2) + h * 2] = words[h] as u32;
            self.regs[(CSR_BUS512_BASE >> 2) + h * 2 + 1] = (words[h] >> 32) as u32;
        }
        let mut bits = [0u8; 512];
        for h in 0..8 {
            for b in 0..64 {
                bits[h * 64 + b] = ((words[h] >> b) & 1) as u8;
            }
        }
        let b_idx = self.active_bank;
        let mut stage0 = [0u8; 64];
        for n in 0..64 {
            let mut addr = 0u32;
            for p in 0..6 {
                addr |= (bits[(self.l0_wires[b_idx][n][p] as usize) & 511] as u32) << p;
            }
            stage0[n] = ((self.l0_table[b_idx][n] >> addr) & 1) as u8;
        }
        let mut votes = [0u32; 8];
        for n in 0..64 {
            let mut addr = 0u32;
            for p in 0..6 {
                addr |= (stage0[(self.l1_wires[b_idx][n][p] as usize) & 63] as u32) << p;
            }
            let out_bit = ((self.l1_table[b_idx][n] >> addr) & 1) as u32;
            votes[n / 8] += out_bit * 8;
        }
        let mut best_c = 0u32;
        let mut best_v = 0u32;
        let mut sec_v = 0u32;
        for (c, &v) in votes.iter().enumerate() {
            if v > best_v {
                sec_v = best_v;
                best_v = v;
                best_c = c as u32;
            } else if v > sec_v {
                sec_v = v;
            }
        }
        let margin = best_v - sec_v;
        self.regs[CSR_RESULT >> 2] =
            ((margin & 0xFF) << 24) | ((sec_v & 0xFF) << 16) | ((best_v & 0xFF) << 8) | best_c;
        self.regs[CSR_TOTAL_EPISODES >> 2] += 1;
        if margin >= self.regs[CSR_MARGIN_THRESH >> 2] {
            self.regs[CSR_SATIATED_CNT >> 2] += 1;
            self.regs[CSR_STATUS >> 2] = 0x3;
        } else {
            self.regs[CSR_ESCALATED_CNT >> 2] += 1;
            self.regs[CSR_STATUS >> 2] = 0x5;
            self.irq_cnt += 1;
        }
    }
}

impl Kv260RegisterHal for LoopbackKv260Hal {
    fn read_reg(&self, byte_offset: usize) -> u32 {
        if byte_offset == CSR_HS_GENERATION {
            ((self.active_bank as u32) << 31) | (self.swap_gen & 0x7FFF_FFFF)
        } else {
            self.regs[byte_offset >> 2]
        }
    }

    fn write_reg(&mut self, byte_offset: usize, val: u32) {
        self.regs[byte_offset >> 2] = val;
        if byte_offset == CSR_HS_CTRL && (val & 0x8000_0000) != 0 {
            let shadow = 1 - self.active_bank;
            let lut_idx = (val & 0x3F) as usize;
            let layer_sel = ((val >> 8) & 1) as usize;
            let t_lo = self.regs[CSR_HS_TABLE_LO >> 2] as u64;
            let t_hi = self.regs[CSR_HS_TABLE_HI >> 2] as u64;
            let w_lo = self.regs[CSR_HS_WIRES_LO >> 2];
            let w_hi = self.regs[CSR_HS_WIRES_HI >> 2];
            let wires = [
                (w_lo & 0x1FF) as u16,
                ((w_lo >> 9) & 0x1FF) as u16,
                ((w_lo >> 18) & 0x1FF) as u16,
                (w_hi & 0x1FF) as u16,
                ((w_hi >> 9) & 0x1FF) as u16,
                ((w_hi >> 18) & 0x1FF) as u16,
            ];
            if layer_sel == 0 {
                self.l0_table[shadow][lut_idx] = (t_hi << 32) | t_lo;
                self.l0_wires[shadow][lut_idx] = wires;
            } else {
                self.l1_table[shadow][lut_idx] = (t_hi << 32) | t_lo;
                self.l1_wires[shadow][lut_idx] = wires;
            }
        } else if byte_offset == CSR_CTRL && (val & (1 << 5)) != 0 {
            self.active_bank = 1 - self.active_bank;
            self.swap_gen += 1;
        }
    }

    fn wait_escalation_irq(&mut self) -> io::Result<u32> {
        self.regs[CSR_STATUS >> 2] &= !0x5;
        Ok(self.irq_cnt)
    }
}

/// High-level Zynq PS Controller driving Shadow-Bank Hot-Swaps on `Kv260RegisterHal`.
pub struct Kv260PsController<H: Kv260RegisterHal> {
    pub hal: H,
}

impl<H: Kv260RegisterHal> Kv260PsController<H> {
    pub fn new(hal: H) -> Self {
        assert_eq!(hal.read_reg(CSR_MAGIC), CLAI_KV260_MAGIC);
        Self { hal }
    }

    pub fn read_latched_engram_bus(&self) -> [u64; 8] {
        let mut out = [0u64; 8];
        for h in 0..8 {
            let lo = self.hal.read_reg(CSR_BUS512_BASE + h * 8) as u64;
            let hi = self.hal.read_reg(CSR_BUS512_BASE + h * 8 + 4) as u64;
            out[h] = (hi << 32) | lo;
        }
        out
    }

    pub fn write_shadow_lut6(
        &mut self,
        layer_sel: u8,
        lut_idx: u8,
        table_init: u64,
        wires: [u16; 6],
    ) {
        let w_lo = ((wires[0] as u32) & 0x1FF)
            | (((wires[1] as u32) & 0x1FF) << 9)
            | (((wires[2] as u32) & 0x1FF) << 18);
        let w_hi = ((wires[3] as u32) & 0x1FF)
            | (((wires[4] as u32) & 0x1FF) << 9)
            | (((wires[5] as u32) & 0x1FF) << 18);
        self.hal.write_reg(CSR_HS_TABLE_LO, table_init as u32);
        self.hal
            .write_reg(CSR_HS_TABLE_HI, (table_init >> 32) as u32);
        self.hal.write_reg(CSR_HS_WIRES_LO, w_lo);
        self.hal.write_reg(CSR_HS_WIRES_HI, w_hi);
        self.hal.write_reg(
            CSR_HS_CTRL,
            0x8000_0000 | (((layer_sel & 1) as u32) << 8) | ((lut_idx & 0x3F) as u32),
        );
    }

    /// Downloads top 8 LUT6s/class from `CompiledMetalRung0` into Shadow Bank & flips in 4.0 ns!
    pub fn hotswap_compiled_rung0(&mut self, r0: &CompiledMetalRung0) -> u32 {
        let luts_per_cls = 512 / 8;
        for layer_sel in 0..=1u8 {
            for c in 0..8usize {
                for k in 0..8usize {
                    let dst_idx = (c * 8 + k) as u8;
                    let src_idx = c * luts_per_cls + k;
                    let t_init = r0.host_tables[layer_sel as usize][src_idx];
                    let raw_w = r0.host_wires[layer_sel as usize][src_idx];
                    let wires: [u16; 6] = if layer_sel == 0 {
                        std::array::from_fn(|p| (raw_w[p] as u16) & 511)
                    } else {
                        std::array::from_fn(|p| (c * 8 + ((raw_w[p] as usize) % 8)) as u16)
                    };
                    self.write_shadow_lut6(layer_sel, dst_idx, t_init, wires);
                }
            }
        }
        // Enable overlay (1<<4) + IRQ (1<<2) + 1-cycle Atomic Swap (1<<5) = 0x34
        self.hal.write_reg(CSR_CTRL, 0x34);
        self.hal.read_reg(CSR_HS_GENERATION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv260_uio_shadow_hotswap_recovers_live_escalation() {
        let mut ctrl = Kv260PsController::new(LoopbackKv260Hal::new());

        // 1. Before hot-swap: all classes tie -> margin = 0 -> escalates via IRQ!
        let pkt_bus = [1u64, 0, 0, 0, 0, 0, 0, 0];
        ctrl.hal.inject_bus512_and_evaluate(&pkt_bus);
        assert_eq!(ctrl.hal.read_reg(CSR_ESCALATED_CNT), 1);
        let irq_n = ctrl.hal.wait_escalation_irq().unwrap();
        assert_eq!(irq_n, 1);
        assert_eq!(ctrl.read_latched_engram_bus(), pkt_bus);

        // 2. Program Class 3 detector into Shadow Bank & trigger 4ns atomic bank swap
        for idx in 0..64u8 {
            if (24..32).contains(&idx) {
                ctrl.write_shadow_lut6(0, idx, 0xAAAA_AAAA_AAAA_AAAA, [0; 6]);
                ctrl.write_shadow_lut6(1, idx, 0xFFF8_F880_F880_8000, [24, 25, 26, 27, 28, 29]);
            } else {
                ctrl.write_shadow_lut6(0, idx, 0, [0; 6]);
                ctrl.write_shadow_lut6(1, idx, 0, [0; 6]);
            }
        }
        ctrl.hal.write_reg(CSR_CTRL, 0x34);
        assert_eq!(ctrl.hal.read_reg(CSR_HS_GENERATION), 0x8000_0001);

        // 3. Re-evaluate packet: Class 3 wins 64 to 0 -> satiated at Rung 0!
        ctrl.hal.inject_bus512_and_evaluate(&pkt_bus);
        assert_eq!(ctrl.hal.read_reg(CSR_SATIATED_CNT), 1);
        assert_eq!(ctrl.hal.read_reg(CSR_RESULT) & 0x7, 3);
        assert_eq!((ctrl.hal.read_reg(CSR_RESULT) >> 24) & 0xFF, 64);
    }
}
