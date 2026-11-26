//! `lut6-core`: Zero-allocation `#![no_std]` canonical `4,096`-byte `LUT6` knowledge basis,
//! vertical 64-lane bit-sliced evaluator, 64-point integer Walsh–Hadamard harmonic filter,
//! 64-codon biological operon ABI, and non-regressing 3-part Bootstrap Nucleus (`Rust 1.100.0`,
//! `edition = "2024"`).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(target_endian = "little"))]
compile_error!("lut6-core requires a little-endian target matching canonical .lut6 byte layout.");

pub mod apex_harness;
pub mod bfws_well;
pub mod bootstrap;
pub mod cbfws_term_math;
pub mod enigma_horizon;
pub mod generated_proof_vectors;
pub mod hadamard_bonsai;
pub mod ratchet_demo;
pub mod seu_resilience;
pub mod siw_joulebit;
pub mod tmac_tail;
pub mod waios_mosaic;
pub mod wearable_epistemic;
#[cfg(target_arch = "wasm32")]
pub mod wasm_api;

#[cfg(all(target_arch = "wasm32", not(test), not(feature = "std")))]
#[panic_handler]
fn wasm_panic_handler(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

pub const MAX_WIRES: usize = 64;
pub const MAX_GATES: usize = 256;
pub const PASS_THROUGH_I0: u64 = 0xAAAA_AAAA_AAAA_AAAA;
pub const FLAG_IS_OVERLAY: u16 = 0x0001;
pub const FLAG_IS_ACTIVE: u16 = 0x0002;
pub const FLAG_HAS_TMAC_TAIL: u16 = 0x0004;

/// Zero-overhead interior mutability wrapper for bare-metal `DTCM` / striped `SRAM` statics
/// on stable Rust `1.100.0` (zero nightly feature flags).
#[repr(transparent)]
pub struct SyncCell<T>(core::cell::UnsafeCell<T>);

// SAFETY: Used in single-core bare-metal or explicitly synchronized benchmark contexts.
unsafe impl<T: Sync> Sync for SyncCell<T> {}

impl<T> SyncCell<T> {
    pub const fn new(val: T) -> Self {
        Self(core::cell::UnsafeCell::new(val))
    }

    #[inline(always)]
    pub const fn get(&self) -> *mut T {
        self.0.get()
    }
}

/// Decision outcome enforcing Conformal Wake Gating & explicit Kill Criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalOutcome {
    /// Certified Rung-6 fast path: zero-FLOP Boolean resolution.
    Act { output_bits: u64, votes: u16 },
    /// Uncertified bit-space: trigger conformal wake gate to escalate to upstream Rung-5.
    Escalate { partial_bits: u64, votes: u16 },
    /// Pre-registered falsification / verifier-rejection budget exceeded.
    Killed,
}

/// Infallible kernel execution signature using Rust 1.100.0's newly stabilized `!` never type.
pub type InfallibleEval = Result<EvalOutcome, !>;

/// Exact `4,096`-byte canonical knowledge basis page (`align(8)` for single-cycle 64-bit loads).
/// Matches 1:1 with a 36-Kbit FPGA `RAMB36E2` block or MCU `DTCM`/`MRAM`/`SRAM` sector.
/// Pin to MCU `SRAM`/`DTCM` in Rust 2024 via `#[unsafe(link_section = ".lut6_dtcm")]`.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct CompactLut6Page {
    // --- 64-Byte Header (bytes 0..64) ---
    pub magic: [u8; 4],           // 0..4:   *b"LUT6"
    pub schema_ver: u16,          // 4..6:   0x0001
    pub flags: u16,               // 6..8:   Bit 0 = is_overlay, Bit 1 = active
    pub domain_hash: u64,         // 8..16:  Canonicalizer / VQ-codebook compatibility ID
    pub parent_commit: [u8; 16],  // 16..32: Lineage SHA-256[0..16] slice
    pub conformal_mask: u64,      // 32..40: Output bits acting as conformal vote/parity gates
    pub override_mask: u64,       // 40..48: (Overlay only) Output bits overriding base page
    pub conformal_thresh: u16,    // 48..50: Minimum parity bits required to Act
    pub num_layers: u8,           // 50..51: Cascade depth (1..=6)
    pub gates_per_layer: u8,      // 51..52: Width per layer (<= 64, maps 1:1 to u64 bitboard)
    pub kill_max_rejects: u32,    // 52..56: Falsifier kill budget
    pub crc32: u32,               // 56..60: IEEE 802.3 CRC32 over header[0..56] + body[64..4096]
    pub _reserved: u32,           // 60..64: Header alignment pad

    // --- 1,984-Byte Routing & Proof Metadata (bytes 64..2048) ---
    pub routes_lo: [u32; 256],    // 64..1088:   Inputs i0..i4 (five 6-bit wire IDs = 30 bits)
    pub routes_hi: [u8; 256],     // 1088..1344: Input i5 (one 6-bit wire ID)
    pub proof_digest: [u8; 32],   // 1344..1376: Lean / SAT miter equivalence certificate digest
    pub _pad: [u8; 672],          // 1376..2048: Reserved for T-MAC / EntroMCU ternary tail

    // --- 2,048-Byte Truth-Table Basis (bytes 2048..4096) ---
    pub init_masks: [u64; 256],   // 2048..4096: 256 x 64-bit zero-FLOP Boolean truth tables
}

const _: () = {
    assert!(core::mem::size_of::<CompactLut6Page>() == 4096);
    assert!(core::mem::align_of::<CompactLut6Page>() == 8);
    assert!(core::mem::offset_of!(CompactLut6Page, routes_lo) == 64);
    assert!(core::mem::offset_of!(CompactLut6Page, init_masks) == 2048);
};

/// Mutable runtime telemetry kept in MCU scratch SRAM (separate from read-only MRAM/Flash page).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PageLedger {
    pub hits: u32,
    pub escalations: u32,
    pub verifier_rejects: u32,
}

#[inline(always)]
const fn crc32_update_byte(mut crc: u32, byte: u8) -> u32 {
    crc ^= byte as u32;
    let mut bit = 0;
    while bit < 8 {
        let mask = if (crc & 1) != 0 { 0xEDB8_8320 } else { 0 };
        crc = (crc >> 1) ^ mask;
        bit += 1;
    }
    crc
}

#[inline(always)]
const fn crc32_update_u16(mut crc: u32, v: u16) -> u32 {
    let b = v.to_le_bytes();
    crc = crc32_update_byte(crc, b[0]);
    crc32_update_byte(crc, b[1])
}

#[inline(always)]
const fn crc32_update_u32(mut crc: u32, v: u32) -> u32 {
    let b = v.to_le_bytes();
    crc = crc32_update_byte(crc, b[0]);
    crc = crc32_update_byte(crc, b[1]);
    crc = crc32_update_byte(crc, b[2]);
    crc32_update_byte(crc, b[3])
}

#[inline(always)]
const fn crc32_update_u64(mut crc: u32, v: u64) -> u32 {
    let b = v.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        crc = crc32_update_byte(crc, b[i]);
        i += 1;
    }
    crc
}

/// Evaluates a single `LUT6` (`init_mask: u64`) across **64 parallel batch inputs** simultaneously
/// using vertical bit-sliced Shannon cofactor muxing (`0` branches, `0` table gather stalls).
#[cfg_attr(target_arch = "wasm32", inline(never))]
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub const fn eval_lut6_bitsliced64(
    mask: u64,
    x0: u64,
    x1: u64,
    x2: u64,
    x3: u64,
    x4: u64,
    x5: u64,
) -> u64 {
    if mask == 0 {
        return 0;
    }
    if mask == u64::MAX {
        return u64::MAX;
    }
    if mask == PASS_THROUGH_I0 {
        return x0;
    }

    // Wobble-degenerate fast path: if truth table is invariant on wobble bits `i0, i1`,
    // skip levels 0 & 1 completely (15 bitwise muxes for all 64 parallel lanes = 0.23 ops/lane!).
    let inv_i0 = (mask & 0x5555_5555_5555_5555) == ((mask >> 1) & 0x5555_5555_5555_5555);
    let inv_i1 = (mask & 0x3333_3333_3333_3333) == ((mask >> 2) & 0x3333_3333_3333_3333);
    if inv_i0 && inv_i1 {
        let mut l3 = [0u64; 8];
        let mut k = 0usize;
        while k < 8 {
            let p0 = ((mask >> (8 * k)) & 1) != 0;
            let p1 = ((mask >> (8 * k + 4)) & 1) != 0;
            l3[k] = match (p0, p1) {
                (false, false) => 0,
                (true, false) => !x2,
                (false, true) => x2,
                (true, true) => u64::MAX,
            };
            k += 1;
        }
        let m4_0 = l3[0] ^ ((l3[0] ^ l3[1]) & x3);
        let m4_1 = l3[2] ^ ((l3[2] ^ l3[3]) & x3);
        let m4_2 = l3[4] ^ ((l3[4] ^ l3[5]) & x3);
        let m4_3 = l3[6] ^ ((l3[6] ^ l3[7]) & x3);
        let m5_0 = m4_0 ^ ((m4_0 ^ m4_1) & x4);
        let m5_1 = m4_2 ^ ((m4_2 ^ m4_3) & x4);
        return m5_0 ^ ((m5_0 ^ m5_1) & x5);
    }

    // General 6-input LUT6 vertical cofactor reduction (31 branchless muxes across 64 lanes)
    let mut l1 = [0u64; 32];
    let mut k = 0usize;
    while k < 32 {
        let pair = (mask >> (2 * k)) & 0x3;
        l1[k] = match pair {
            0 => 0,
            1 => !x0,
            2 => x0,
            _ => u64::MAX,
        };
        k += 1;
    }
    let mut l2 = [0u64; 16];
    let mut i = 0usize;
    while i < 16 {
        let a = l1[2 * i];
        let b = l1[2 * i + 1];
        l2[i] = a ^ ((a ^ b) & x1);
        i += 1;
    }
    let mut l3 = [0u64; 8];
    i = 0;
    while i < 8 {
        let a = l2[2 * i];
        let b = l2[2 * i + 1];
        l3[i] = a ^ ((a ^ b) & x2);
        i += 1;
    }
    let mut l4 = [0u64; 4];
    i = 0;
    while i < 4 {
        let a = l3[2 * i];
        let b = l3[2 * i + 1];
        l4[i] = a ^ ((a ^ b) & x3);
        i += 1;
    }
    let m5_0 = l4[0] ^ ((l4[0] ^ l4[1]) & x4);
    let m5_1 = l4[2] ^ ((l4[2] ^ l4[3]) & x4);
    m5_0 ^ ((m5_0 ^ m5_1) & x5)
}

/// 6-pass SWAR `64x64` bit-matrix transpose (`O(64 * 6)` register ops, `0` branches).
/// Converts `[u64; 64]` horizontal scalar states (`inputs[lane]`, wire `w` at bit `w`)
/// into vertical bit-sliced wire planes (`planes[wire]`, lane `l` at bit `l`) and back
/// (involution: `transpose_64x64(&transpose_64x64(m)) == m`).
pub const fn transpose_64x64(matrix: &[u64; 64]) -> [u64; 64] {
    let mut a = *matrix;
    let masks: [u64; 6] = [
        0x5555_5555_5555_5555,
        0x3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF,
    ];
    let mut stage = 0usize;
    while stage < 6 {
        let step = 1usize << stage;
        let shift = step as u32;
        let m = masks[stage];
        let inv_m = !m;
        let mut base = 0usize;
        while base < 64 {
            let mut j = 0usize;
            while j < step {
                let lo = a[base + j];
                let hi = a[base + j + step];
                a[base + j] = (lo & m) | ((hi & m) << shift);
                a[base + j + step] = ((lo & inv_m) >> shift) | (hi & inv_m);
                j += 1;
            }
            base += step << 1;
        }
        stage += 1;
    }
    a
}

impl CompactLut6Page {
    #[inline(always)]
    pub const fn is_valid(&self) -> bool {
        self.magic[0] == b'L'
            && self.magic[1] == b'U'
            && self.magic[2] == b'T'
            && self.magic[3] == b'6'
            && (self.gates_per_layer as usize) <= MAX_WIRES
            && (self.num_layers as usize) * (self.gates_per_layer as usize) <= MAX_GATES
    }

    /// Computes canonical IEEE 802.3 CRC32 over `header[0..56]` + `body[64..4096]`
    /// matching `zlib.crc32(header_wo_crc + body)` in `scripts/pack_lut6_crystal.py`.
    pub const fn compute_crc32(&self) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        let mut i = 0usize;
        while i < 4 {
            crc = crc32_update_byte(crc, self.magic[i]);
            i += 1;
        }
        crc = crc32_update_u16(crc, self.schema_ver);
        crc = crc32_update_u16(crc, self.flags);
        crc = crc32_update_u64(crc, self.domain_hash);
        i = 0;
        while i < 16 {
            crc = crc32_update_byte(crc, self.parent_commit[i]);
            i += 1;
        }
        crc = crc32_update_u64(crc, self.conformal_mask);
        crc = crc32_update_u64(crc, self.override_mask);
        crc = crc32_update_u16(crc, self.conformal_thresh);
        crc = crc32_update_byte(crc, self.num_layers);
        crc = crc32_update_byte(crc, self.gates_per_layer);
        crc = crc32_update_u32(crc, self.kill_max_rejects);

        i = 0;
        while i < 256 {
            crc = crc32_update_u32(crc, self.routes_lo[i]);
            i += 1;
        }
        i = 0;
        while i < 256 {
            crc = crc32_update_byte(crc, self.routes_hi[i]);
            i += 1;
        }
        i = 0;
        while i < 32 {
            crc = crc32_update_byte(crc, self.proof_digest[i]);
            i += 1;
        }
        i = 0;
        while i < 672 {
            crc = crc32_update_byte(crc, self._pad[i]);
            i += 1;
        }
        i = 0;
        while i < 256 {
            crc = crc32_update_u64(crc, self.init_masks[i]);
            i += 1;
        }
        !crc
    }

    #[inline(always)]
    pub const fn verify_crc32(&self) -> bool {
        self.is_valid() && (self.crc32 == 0 || self.crc32 == self.compute_crc32())
    }

    /// Pure `const fn` 64-wire bitboard evaluator: runs identically at compile time (`rustc`)
    /// for zero-regression proof checks and at runtime in bare-metal `SRAM`/`DTCM`.
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub const fn eval_raw(&self, mut state: u64) -> u64 {
        let layers = self.num_layers as usize;
        let width = self.gates_per_layer as usize;
        let mut gate_idx = 0usize;

        let mut l = 0;
        while l < layers {
            let mut next_state: u64 = 0;
            let mut g = 0;
            while g < width {
                let r_lo = self.routes_lo[gate_idx];
                let r_hi = self.routes_hi[gate_idx] as u32;

                let b0 = ((state >> (r_lo & 0x3F)) & 1) as u32;
                let b1 = ((state >> ((r_lo >> 6) & 0x3F)) & 1) as u32;
                let b2 = ((state >> ((r_lo >> 12) & 0x3F)) & 1) as u32;
                let b3 = ((state >> ((r_lo >> 18) & 0x3F)) & 1) as u32;
                let b4 = ((state >> ((r_lo >> 24) & 0x3F)) & 1) as u32;
                let b5 = ((state >> (r_hi & 0x3F)) & 1) as u32;

                let lut_addr = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5);
                let out_bit = (self.init_masks[gate_idx] >> lut_addr) & 1;

                next_state |= out_bit << g;
                gate_idx += 1;
                g += 1;
            }
            state = next_state;
            l += 1;
        }
        state
    }

    /// 4-way unrolled scalar `LUT6` bitboard evaluator (`eval_raw_unrolled4`):
    /// Breaks loop-carried `next_state` shift/OR dependency chains across 4 parallel gate lanes
    /// per iteration for superscalar Armv9 / Zen 5 / Cortex-M55 dual-issue pipelines while
    /// maintaining bit-exact equivalence to `eval_raw`.
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    pub const fn eval_raw_unrolled4(&self, mut state: u64) -> u64 {
        let layers = self.num_layers as usize;
        let width = self.gates_per_layer as usize;
        let mut gate_idx = 0usize;

        let mut l = 0usize;
        while l < layers {
            let mut acc0: u64 = 0;
            let mut acc1: u64 = 0;
            let mut acc2: u64 = 0;
            let mut acc3: u64 = 0;
            let mut g = 0usize;
            while g + 4 <= width {
                let r0 = self.routes_lo[gate_idx];
                let h0 = self.routes_hi[gate_idx] as u32;
                let a0 = (((state >> (r0 & 0x3F)) & 1) as u32)
                    | ((((state >> ((r0 >> 6) & 0x3F)) & 1) as u32) << 1)
                    | ((((state >> ((r0 >> 12) & 0x3F)) & 1) as u32) << 2)
                    | ((((state >> ((r0 >> 18) & 0x3F)) & 1) as u32) << 3)
                    | ((((state >> ((r0 >> 24) & 0x3F)) & 1) as u32) << 4)
                    | ((((state >> (h0 & 0x3F)) & 1) as u32) << 5);
                acc0 |= ((self.init_masks[gate_idx] >> a0) & 1) << g;

                let r1 = self.routes_lo[gate_idx + 1];
                let h1 = self.routes_hi[gate_idx + 1] as u32;
                let a1 = (((state >> (r1 & 0x3F)) & 1) as u32)
                    | ((((state >> ((r1 >> 6) & 0x3F)) & 1) as u32) << 1)
                    | ((((state >> ((r1 >> 12) & 0x3F)) & 1) as u32) << 2)
                    | ((((state >> ((r1 >> 18) & 0x3F)) & 1) as u32) << 3)
                    | ((((state >> ((r1 >> 24) & 0x3F)) & 1) as u32) << 4)
                    | ((((state >> (h1 & 0x3F)) & 1) as u32) << 5);
                acc1 |= ((self.init_masks[gate_idx + 1] >> a1) & 1) << (g + 1);

                let r2 = self.routes_lo[gate_idx + 2];
                let h2 = self.routes_hi[gate_idx + 2] as u32;
                let a2 = (((state >> (r2 & 0x3F)) & 1) as u32)
                    | ((((state >> ((r2 >> 6) & 0x3F)) & 1) as u32) << 1)
                    | ((((state >> ((r2 >> 12) & 0x3F)) & 1) as u32) << 2)
                    | ((((state >> ((r2 >> 18) & 0x3F)) & 1) as u32) << 3)
                    | ((((state >> ((r2 >> 24) & 0x3F)) & 1) as u32) << 4)
                    | ((((state >> (h2 & 0x3F)) & 1) as u32) << 5);
                acc2 |= ((self.init_masks[gate_idx + 2] >> a2) & 1) << (g + 2);

                let r3 = self.routes_lo[gate_idx + 3];
                let h3 = self.routes_hi[gate_idx + 3] as u32;
                let a3 = (((state >> (r3 & 0x3F)) & 1) as u32)
                    | ((((state >> ((r3 >> 6) & 0x3F)) & 1) as u32) << 1)
                    | ((((state >> ((r3 >> 12) & 0x3F)) & 1) as u32) << 2)
                    | ((((state >> ((r3 >> 18) & 0x3F)) & 1) as u32) << 3)
                    | ((((state >> ((r3 >> 24) & 0x3F)) & 1) as u32) << 4)
                    | ((((state >> (h3 & 0x3F)) & 1) as u32) << 5);
                acc3 |= ((self.init_masks[gate_idx + 3] >> a3) & 1) << (g + 3);

                gate_idx += 4;
                g += 4;
            }
            let mut next_state = (acc0 | acc1) | (acc2 | acc3);
            while g < width {
                let r_lo = self.routes_lo[gate_idx];
                let r_hi = self.routes_hi[gate_idx] as u32;
                let lut_addr = (((state >> (r_lo & 0x3F)) & 1) as u32)
                    | ((((state >> ((r_lo >> 6) & 0x3F)) & 1) as u32) << 1)
                    | ((((state >> ((r_lo >> 12) & 0x3F)) & 1) as u32) << 2)
                    | ((((state >> ((r_lo >> 18) & 0x3F)) & 1) as u32) << 3)
                    | ((((state >> ((r_lo >> 24) & 0x3F)) & 1) as u32) << 4)
                    | ((((state >> (r_hi & 0x3F)) & 1) as u32) << 5);
                next_state |= ((self.init_masks[gate_idx] >> lut_addr) & 1) << g;
                gate_idx += 1;
                g += 1;
            }
            state = next_state;
            l += 1;
        }
        state
    }

    /// Vertical bit-sliced evaluator over 64 pre-transposed wire planes (`planes[wire]`, 64 lanes).
    #[inline]
    pub const fn eval_planes64(&self, mut planes: [u64; 64]) -> [u64; 64] {
        let layers = self.num_layers as usize;
        let width = self.gates_per_layer as usize;
        let mut gate_idx = 0usize;

        let mut l = 0usize;
        while l < layers {
            let mut next_planes = [0u64; 64];
            let mut g = 0usize;
            while g < width {
                let r_lo = self.routes_lo[gate_idx];
                let r_hi = self.routes_hi[gate_idx] as usize;
                let x0 = planes[(r_lo & 0x3F) as usize];
                let x1 = planes[((r_lo >> 6) & 0x3F) as usize];
                let x2 = planes[((r_lo >> 12) & 0x3F) as usize];
                let x3 = planes[((r_lo >> 18) & 0x3F) as usize];
                let x4 = planes[((r_lo >> 24) & 0x3F) as usize];
                let x5 = planes[r_hi & 0x3F];
                next_planes[g] =
                    eval_lut6_bitsliced64(self.init_masks[gate_idx], x0, x1, x2, x3, x4, x5);
                gate_idx += 1;
                g += 1;
            }
            planes = next_planes;
            l += 1;
        }
        planes
    }

    /// Evaluates **64 independent canonical inputs** simultaneously via `64x64` SWAR transpose
    /// + vertical bit-sliced LUT6 evaluation, guaranteeing bit-exact parity with 64 `eval_raw` calls.
    #[inline]
    pub const fn eval_batch64(&self, inputs: &[u64; 64]) -> [u64; 64] {
        let planes = transpose_64x64(inputs);
        let out_planes = self.eval_planes64(planes);
        transpose_64x64(&out_planes)
    }

    /// Evaluates base crystal + optional surgical overlay using Rust 2024 `let_chains`.
    #[inline]
    pub fn evaluate(
        &self,
        canonical_input: u64,
        overlay: Option<&Self>,
        ledger: &mut PageLedger,
    ) -> InfallibleEval {
        if ledger.verifier_rejects >= self.kill_max_rejects {
            return Ok(EvalOutcome::Killed);
        }

        let mut out = self.eval_raw(canonical_input);

        // Rust 2024 `let_chains`: apply overlay patch only if valid, domain-matched, and confident
        if let Some(patch) = overlay
            && patch.is_valid()
            && patch.domain_hash == self.domain_hash
        {
            let patch_out = patch.eval_raw(canonical_input);
            let patch_votes = (patch_out & patch.conformal_mask).count_ones() as u16;
            if patch_votes >= patch.conformal_thresh {
                out = (out & !patch.override_mask) | (patch_out & patch.override_mask);
            }
        }

        let votes = (out & self.conformal_mask).count_ones() as u16;
        let payload_bits = out & !self.conformal_mask;

        if votes >= self.conformal_thresh {
            ledger.hits = ledger.hits.wrapping_add(1);
            Ok(EvalOutcome::Act {
                output_bits: payload_bits,
                votes,
            })
        } else {
            ledger.escalations = ledger.escalations.wrapping_add(1);
            Ok(EvalOutcome::Escalate {
                partial_bits: payload_bits,
                votes,
            })
        }
    }

    /// Two-Tier Crystal Evaluation (`Rung 6 LUT6 -> Rung 5.5 Zero-MUL T-MAC Tail -> Rung 5 Escalate`):
    /// Consults the zero-FLOP `LUT6` truth tables first; on a conformal wake miss, if `FLAG_HAS_TMAC_TAIL`
    /// is set on the page, evaluates the embedded `672`-byte `EntroMCU` / `T-MAC` ternary tail in
    /// stack SRAM (`0` multiplications) before waking upstream `Rung 5`.
    #[inline]
    pub fn evaluate_two_tier(
        &self,
        canonical_input: u64,
        signed_features: &[i8; 64],
        overlay: Option<&Self>,
        ledger: &mut PageLedger,
    ) -> InfallibleEval {
        let Ok(r6) = self.evaluate(canonical_input, overlay, ledger);
        match r6 {
            EvalOutcome::Act { .. } | EvalOutcome::Killed => Ok(r6),
            EvalOutcome::Escalate {
                partial_bits,
                votes,
            } => {
                if (self.flags & FLAG_HAS_TMAC_TAIL) != 0 {
                    let (tmac_payload, tmac_votes, _) =
                        tmac_tail::eval_tmac_antisym_from_pad(&self._pad, signed_features);
                    if tmac_votes >= self.conformal_thresh {
                        ledger.escalations = ledger.escalations.wrapping_sub(1);
                        ledger.hits = ledger.hits.wrapping_add(1);
                        return Ok(EvalOutcome::Act {
                            output_bits: tmac_payload,
                            votes: tmac_votes,
                        });
                    }
                }
                Ok(EvalOutcome::Escalate {
                    partial_bits,
                    votes,
                })
            }
        }
    }
}

// ============================================================================
// Signal vs. Noise: 64-Point Integer Walsh–Hadamard Harmonic Filter
// ============================================================================

/// Minimal Agency primitives: the finite set of actions Nucleus #2 needs to know.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgencyAction {
    EmitCertified { payload: u32 },     // Act directly from Knowledge Store (#1)
    QueryPage { page_slot: u8 },        // Chain/compose another local .lut6 page (#1)
    InvokeExternalTool { tool_id: u8 }, // Escalate to deterministic external tool (#2)
    WakeRefiner { frontier_bits: u64 }, // Escalate to Rung-5 compiler to synthesize a patch (#3)
    ApoptosisRollback,                  // Evict corrupt overlay (`0x3C`) and restore immutable seed
    SleepHalt,                          // Dormant Stop Codon (`0x00`) when ATP/Joule budget is 0
}

/// Fast 64-point Integer Walsh–Hadamard Transform (FWHT) on a single LUT6 `u64` truth table.
/// Separates low-order structural signal (`|S| <= 2`) from high-sequency parity noise (`|S| >= 5`)
/// in `0` FLOPs (6 integer butterfly passes; Parseval identity `sum_S spec[S]^2 == 4096`).
pub const fn lut6_walsh_spectrum(init_mask: u64) -> [i8; 64] {
    let mut w = [0i16; 64];
    let mut i = 0usize;
    while i < 64 {
        w[i] = if ((init_mask >> i) & 1) == 1 { -1 } else { 1 };
        i += 1;
    }
    let mut len = 1usize;
    while len < 64 {
        let mut base = 0usize;
        while base < 64 {
            let mut j = 0usize;
            while j < len {
                let u = w[base + j];
                let v = w[base + j + len];
                w[base + j] = u + v;
                w[base + j + len] = u - v;
                j += 1;
            }
            base += len << 1;
        }
        len <<= 1;
    }
    let mut out = [0i8; 64];
    let mut k = 0usize;
    while k < 64 {
        out[k] = w[k] as i8; // Bounded in [-64, +64]
        k += 1;
    }
    out
}

/// Computes high-harmonic noise energy (`degree >= 5`, i.e., `>= 5` bits set in Walsh index `s`)
/// out of `4,096` total Parseval energy (`64^2`).
/// - Any 2-bit wobble-degenerate truth table (`wobble_codon_mask`) has exact `0` high-harmonic energy.
/// - A single 1-hot point-memorized bit (`1u64 << addr`) leaks `28` (`7 * 4`, `11.1%` of AC energy).
/// - Pure 6-bit XOR parity (`0x6996_9669_9669_6996`) concentrates all `4,096` energy at `|S| = 6`.
pub const fn high_harmonic_energy(init_mask: u64) -> u32 {
    let spec = lut6_walsh_spectrum(init_mask);
    let mut noise_sq: u32 = 0;
    let mut s = 0usize;
    while s < 64 {
        if s.count_ones() >= 5 {
            let c = spec[s] as i32;
            noise_sq += (c * c) as u32;
        }
        s += 1;
    }
    noise_sq
}

/// Total Parseval spectral energy (`sum_{s=0}^{63} spec[s]^2`), invariant `== 4096` for all `u64`.
pub const fn parseval_total_energy(init_mask: u64) -> u32 {
    let spec = lut6_walsh_spectrum(init_mask);
    let mut total: u32 = 0;
    let mut s = 0usize;
    while s < 64 {
        let c = spec[s] as i32;
        total += (c * c) as u32;
        s += 1;
    }
    total
}

// ============================================================================
// The 3-Part Bootstrap Nucleus
// ============================================================================

/// The 3-Part Self-Refining Bootstrap Nucleus:
/// 1. `knowledge_root` (immutable seed store) + `knowledge_overlay`
/// 2. `agency_root`    (immutable seed 64-codon Operon FSM) + `agency_overlay`
/// 3. `refine_*`       (admission gate enforcing harmonic purity + 100% anchor retention)
pub struct Nucleus<'a, const ANCHORS: usize> {
    pub knowledge_root: &'a CompactLut6Page,        // #1 Seed Knowledge (ROM/MRAM)
    pub agency_root: &'a CompactLut6Page,           // #2 Seed Agency (ROM/MRAM)
    pub knowledge_overlay: Option<CompactLut6Page>, // #3 Refined Knowledge Patch
    pub agency_overlay: Option<CompactLut6Page>,    // #3 Refined Agency Patch
    pub k_ledger: PageLedger,
    pub a_ledger: PageLedger,
    /// Certified `(input, expected_output_mask, expected_bits)` vectors that #3 MUST NEVER break.
    pub k_anchors: [(u64, u64, u64); ANCHORS],
    pub a_anchors: [(u64, u64, u64); ANCHORS],
}

impl<'a, const ANCHORS: usize> Nucleus<'a, ANCHORS> {
    /// Step the Nucleus: consult Knowledge (#1), then route via 64-Codon Agency Operon (#2).
    #[inline]
    pub fn step(&mut self, canonical_state: u64, tool_status: u8) -> AgencyAction {
        self.step_with_riboswitch(canonical_state, tool_status, 0xFF, 0, false, false)
    }

    /// Full 64-bit `AgencyStateWord` step exposing ATP/Joule thermometer budget (`joule_budget`),
    /// ribosome chain depth (`hop_ttl`), and fault/reject flags (`tool_fault`, `verifier_reject`).
    #[inline]
    pub fn step_with_riboswitch(
        &mut self,
        canonical_state: u64,
        tool_status: u8,
        joule_budget: u8,
        hop_ttl: u8,
        tool_fault: bool,
        verifier_reject: bool,
    ) -> AgencyAction {
        if !verifier_reject {
            let Ok(k_res) = self.knowledge_root.evaluate(
                canonical_state,
                self.knowledge_overlay.as_ref(),
                &mut self.k_ledger,
            );

            // Fast path: #1 is certified -> return in nanoseconds/microseconds without waking tools
            if let EvalOutcome::Act { output_bits, .. } = k_res {
                return AgencyAction::EmitCertified {
                    payload: output_bits as u32,
                };
            }

            let (partial, votes, killed) = match k_res {
                EvalOutcome::Escalate {
                    partial_bits,
                    votes,
                } => (partial_bits, votes, false),
                EvalOutcome::Killed => {
                    // Auto-evict compromised overlay when rejection budget exceeded
                    self.knowledge_overlay = None;
                    (0, 0, true)
                }
                EvalOutcome::Act { .. } => unreachable!(),
            };

            let agency_in = bootstrap::pack_agency_state_word(
                partial as u32,
                votes as u8,
                tool_status,
                joule_budget,
                hop_ttl,
                tool_fault,
                killed,
            );
            return self.dispatch_agency(agency_in, canonical_state);
        }

        // Explicit verifier reject (`w63 = 1`) -> route straight to Agency Operon for Apoptosis
        self.k_ledger.verifier_rejects = self.k_ledger.verifier_rejects.wrapping_add(1);
        let agency_in = bootstrap::pack_agency_state_word(
            canonical_state as u32,
            0,
            tool_status,
            joule_budget,
            hop_ttl,
            tool_fault,
            true,
        );
        self.dispatch_agency(agency_in, canonical_state)
    }

    #[inline]
    fn dispatch_agency(&mut self, agency_in: u64, canonical_state: u64) -> AgencyAction {
        let Ok(a_res) = self.agency_root.evaluate(
            agency_in,
            self.agency_overlay.as_ref(),
            &mut self.a_ledger,
        );

        match a_res {
            EvalOutcome::Act { output_bits, .. } => {
                let codon = (output_bits & 0x3F) as u8;
                let locus = ((output_bits >> 6) & 0xFF) as u8;
                match codon {
                    0x00..=0x03 => AgencyAction::SleepHalt,
                    0x04..=0x0F => AgencyAction::EmitCertified {
                        payload: ((output_bits >> 14) & 0xFFFF_FFFF) as u32,
                    },
                    0x10..=0x1F => AgencyAction::QueryPage { page_slot: locus },
                    0x20..=0x2F => AgencyAction::InvokeExternalTool { tool_id: locus },
                    0x3C..=0x3F => {
                        // APOPTOSIS_ROLLBACK: evict active overlay and restore immutable seed
                        self.knowledge_overlay = None;
                        self.k_ledger.verifier_rejects = 0;
                        AgencyAction::ApoptosisRollback
                    }
                    _ => AgencyAction::WakeRefiner {
                        frontier_bits: canonical_state,
                    },
                }
            }
            _ => AgencyAction::WakeRefiner {
                frontier_bits: canonical_state,
            },
        }
    }

    /// Axiom #3: Refine Knowledge (#1) ONLY if:
    /// (a) High-order harmonic noise (`|S| >= 5`) stays `<= max_noise_energy`, and
    /// (b) `100%` of seed anchor behaviors are strictly retained (zero regression).
    pub fn refine_knowledge(
        &mut self,
        candidate: CompactLut6Page,
        max_noise_energy: u32,
    ) -> bool {
        if !Self::passes_retention_gate(
            self.knowledge_root,
            &candidate,
            &self.k_anchors,
            max_noise_energy,
        ) {
            return false;
        }
        self.knowledge_overlay = Some(candidate);
        true
    }

    /// Axiom #3: Refine Agency (#2) under identical harmonic + non-regression immune gates.
    pub fn refine_agency(&mut self, candidate: CompactLut6Page, max_noise_energy: u32) -> bool {
        if !Self::passes_retention_gate(
            self.agency_root,
            &candidate,
            &self.a_anchors,
            max_noise_energy,
        ) {
            return false;
        }
        self.agency_overlay = Some(candidate);
        true
    }

    pub fn passes_retention_gate(
        root: &CompactLut6Page,
        patch: &CompactLut6Page,
        anchors: &[(u64, u64, u64)],
        max_noise_energy: u32,
    ) -> bool {
        if !patch.is_valid() || patch.domain_hash != root.domain_hash {
            return false;
        }
        // 1. Harmonic Filter: reject brittle truth tables leaking degree-5/6 parity noise
        let active_gates = (patch.num_layers as usize) * (patch.gates_per_layer as usize);
        let mut g = 0usize;
        while g < active_gates {
            if high_harmonic_energy(patch.init_masks[g]) > max_noise_energy {
                return false;
            }
            g += 1;
        }
        // 2. Retention Invariant: verify zero regression across all protected seed vectors
        let mut dummy = PageLedger::default();
        for &(inp, mask, expected) in anchors {
            let Ok(res) = root.evaluate(inp, Some(patch), &mut dummy);
            match res {
                EvalOutcome::Act { output_bits, .. } if (output_bits & mask) == expected => {}
                _ => return false,
            }
        }
        true
    }
}
