//! 64-Bit Biological Codon & Operon ABI (`AgencyStateWord -> OperonFrame`) and compile-time
//! verified `SEED_AGENCY` crystal (`Nucleus #2`).

use crate::{CompactLut6Page, PASS_THROUGH_I0};

pub const BIO_DOMAIN_HASH: u64 = 0x4249_4F5F_434F_5245; // "BIO_CORE"

// Canonical 64-Codon Instruction Quadrants (`bits [4..5]`, wobble buffer on `bits [0..1]`)
pub const CODON_STOP_HALT: u8 = 0x00;           // 0x00..=0x03: Sleep MCU (WFI / STOP)
pub const CODON_EMIT_DIRECT: u8 = 0x04;         // 0x04..=0x0F: Direct actuator / PIO commit
pub const CODON_SPLICE_PAGE: u8 = 0x10;         // 0x10..=0x17: Local .lut6 skill composition
pub const CODON_FOLD_STATE: u8 = 0x18;          // 0x18..=0x1F: Planner-free FSM state fold
pub const CODON_BUS_PIO_SPI: u8 = 0x20;         // 0x20..=0x23: Deterministic PIO / SPI burst
pub const CODON_BUS_UART_MCP: u8 = 0x24;        // 0x24..=0x27: Framed RPC over UART / PCIe
pub const CODON_WAKE_LOCAL_NPU: u8 = 0x28;      // 0x28..=0x2B: Ungate Ethos-U55/U85 / T-MAC
pub const CODON_VERIFY_CHECK: u8 = 0x2C;        // 0x2C..=0x2F: Deterministic CRC / RCE check
pub const CODON_TRANSCRIBE_FRONTIER: u8 = 0x30; // 0x30..=0x3B: Wake Rung-5 compiler to patch
pub const CODON_APOPTOSIS_ROLLBACK: u8 = 0x3C;  // 0x3C..=0x3F: Evict overlay & restore seed

/// Packs the 64-bit `AgencyStateWord` input bus for `agency_root` (`Nucleus #2`):
/// - `[0..31]`:  mRNA Reading Frame (`partial_bits` / concept signature from `knowledge_root`)
/// - `[32..39]`: Proofreading Fidelity (`votes` count `0..64` from `knowledge_root`)
/// - `[40..47]`: Allosteric Sensor (`tool_rdy` readiness mask for 8 external tool channels)
/// - `[48..55]`: ATP / Joule Budget (8-bit Distributive Thermometer code)
/// - `[56..61]`: Ribosome Step (`hop_ttl` `0..63` chain depth bound)
/// - `[62]`:     Tool Fault flag
/// - `[63]`:     Verifier Reject pending flag
#[inline(always)]
pub const fn pack_agency_state_word(
    reading_frame: u32,
    proofreading_votes: u8,
    tool_rdy: u8,
    joule_thermometer: u8,
    hop_ttl: u8,
    tool_fault: bool,
    verifier_reject: bool,
) -> u64 {
    (reading_frame as u64)
        | ((proofreading_votes as u64) << 32)
        | ((tool_rdy as u64) << 40)
        | ((joule_thermometer as u64) << 48)
        | (((hop_ttl & 0x3F) as u64) << 56)
        | ((tool_fault as u64) << 62)
        | ((verifier_reject as u64) << 63)
}

/// Decodes a 64-bit `OperonFrame` emitted by `agency_root` (`Nucleus #2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedOperonFrame {
    pub codon_opcode: u8,     // [0..5]
    pub locus_id: u8,         // [6..13]
    pub signal_peptide: u32,  // [14..45]
    pub promoter_locks: u16,  // [46..55]
    pub conformal_votes: u8,  // [56..63]
}

impl DecodedOperonFrame {
    #[inline(always)]
    pub const fn from_raw(frame: u64) -> Self {
        Self {
            codon_opcode: (frame & 0x3F) as u8,
            locus_id: ((frame >> 6) & 0xFF) as u8,
            signal_peptide: ((frame >> 14) & 0xFFFF_FFFF) as u32,
            promoter_locks: ((frame >> 46) & 0x3FF) as u16,
            conformal_votes: ((frame >> 56) & 0xFF) as u8,
        }
    }
}

/// Compile-time macro building a 64-bit LUT6 `INIT` mask from any `const fn(bool..bool) -> bool`.
#[macro_export]
macro_rules! synth_lut6 {
    ($pred:ident) => {{
        let mut mask = 0u64;
        let mut addr = 0usize;
        while addr < 64 {
            let i0 = (addr & 1) != 0;
            let i1 = ((addr >> 1) & 1) != 0;
            let i2 = ((addr >> 2) & 1) != 0;
            let i3 = ((addr >> 3) & 1) != 0;
            let i4 = ((addr >> 4) & 1) != 0;
            let i5 = ((addr >> 5) & 1) != 0;
            if $pred(i0, i1, i2, i3, i4, i5) {
                mask |= 1u64 << addr;
            }
            addr += 1;
        }
        mask
    }};
}

/// Runtime helper building a 64-bit LUT6 `INIT` mask from any Boolean predicate `f(i0..i5)`.
pub fn synth_lut6_fn(pred: fn(bool, bool, bool, bool, bool, bool) -> bool) -> u64 {
    let mut mask = 0u64;
    let mut addr = 0usize;
    while addr < 64 {
        let i0 = (addr & 1) != 0;
        let i1 = ((addr >> 1) & 1) != 0;
        let i2 = ((addr >> 2) & 1) != 0;
        let i3 = ((addr >> 3) & 1) != 0;
        let i4 = ((addr >> 4) & 1) != 0;
        let i5 = ((addr >> 5) & 1) != 0;
        if pred(i0, i1, i2, i3, i4, i5) {
            mask |= 1u64 << addr;
        }
        addr += 1;
    }
    mask
}

#[inline(always)]
pub const fn pack_route6(w0: u8, w1: u8, w2: u8, w3: u8, w4: u8, w5: u8) -> (u32, u8) {
    let lo = (w0 as u32)
        | ((w1 as u32) << 6)
        | ((w2 as u32) << 12)
        | ((w3 as u32) << 18)
        | ((w4 as u32) << 24);
    (lo, w5)
}

const fn pred_quadrant_bit0(
    tool_rdy: bool,
    joule_ok: bool,
    ttl_max: bool,
    fault: bool,
    reject: bool,
    _partial_votes: bool,
) -> bool {
    reject || fault || ttl_max || (joule_ok && !tool_rdy)
}

const fn pred_quadrant_bit1(
    tool_rdy: bool,
    joule_ok: bool,
    ttl_max: bool,
    fault: bool,
    reject: bool,
    partial_votes: bool,
) -> bool {
    reject || fault || ttl_max || (joule_ok && (tool_rdy || !partial_votes))
}

const fn pred_apoptosis_subcodon(
    _tool_rdy: bool,
    _joule_ok: bool,
    _ttl_max: bool,
    _fault: bool,
    reject: bool,
    _partial_votes: bool,
) -> bool {
    reject
}

/// Builds the immutable 4 KB `SEED_AGENCY` Crystal (`Nucleus #2`):
/// Implements Riboswitch energy-gating (`w48`), Allosteric Tool-Readiness dispatch (`w40`),
/// Hop-TTL cutoff (`w61`), and Apoptosis rollback (`w63 -> 0x3C`) in a single 1-layer,
/// 64-gate zero-FLOP pass, sealed with canonical IEEE 802.3 `crc32`.
pub const fn build_seed_agency_crystal(domain_hash: u64) -> CompactLut6Page {
    let mut routes_lo = [0u32; 256];
    let mut routes_hi = [0u8; 256];
    let mut init_masks = [0u64; 256];

    // Default all 64 layer-0 gates to zero; wire [14..46] to pass input [0..32] signal peptide
    let mut g = 14usize;
    while g < 46 {
        let src_wire = (g - 14) as u8;
        let (lo, hi) = pack_route6(src_wire, 0, 0, 0, 0, 0);
        routes_lo[g] = lo;
        routes_hi[g] = hi;
        init_masks[g] = PASS_THROUGH_I0;
        g += 1;
    }

    // Wire inputs used by seed Operon control gates:
    // i0=w40 (Ext Tool 0 Ready), i1=w48 (Joule Budget > Min), i2=w61 (Hop TTL Maxed),
    // i3=w62 (Tool Fault), i4=w63 (Verifier Reject), i5=w35 (Partial Votes >= 8)
    let (c_lo, c_hi) = pack_route6(40, 48, 61, 62, 63, 35);

    // Gate 4 (Codon Quadrant Bit 0): Transcription/Rollback (1) if Fault/Reject/TTL-Max,
    // else Splicing (1) if local votes partial & no external tool needed (`0xFFFF_FFF4_FFFF_FFF0`).
    routes_lo[4] = c_lo;
    routes_hi[4] = c_hi;
    init_masks[4] = synth_lut6!(pred_quadrant_bit0);

    // Gate 5 (Codon Quadrant Bit 1): Exogenous Tool (10) or Transcription/Apoptosis (11)
    // (`0xFFFF_FFF8_FFFF_FFF8`).
    routes_lo[5] = c_lo;
    routes_hi[5] = c_hi;
    init_masks[5] = synth_lut6!(pred_quadrant_bit1);

    // Gates 2 & 3 (Apoptosis Sub-Codon `0x3C` when verifier reject bit w63 is high: `0xFFFF_0000_FFFF_0000`)
    let apop_mask = synth_lut6!(pred_apoptosis_subcodon);
    routes_lo[2] = c_lo;
    routes_hi[2] = c_hi;
    init_masks[2] = apop_mask;
    routes_lo[3] = c_lo;
    routes_hi[3] = c_hi;
    init_masks[3] = apop_mask;

    // Gates 56..63 (8 Conformal Vote Gates): Deterministic seed agency is always self-certified
    let mut v = 56usize;
    while v < 64 {
        routes_lo[v] = c_lo;
        routes_hi[v] = c_hi;
        init_masks[v] = 0xFFFF_FFFF_FFFF_FFFF;
        v += 1;
    }

    let mut page = CompactLut6Page {
        magic: *b"LUT6",
        schema_ver: 1,
        flags: 0x0002,
        domain_hash,
        parent_commit: [0u8; 16],
        conformal_mask: 0xFF00_0000_0000_0000,
        override_mask: 0,
        conformal_thresh: 6,
        num_layers: 1,
        gates_per_layer: 64,
        kill_max_rejects: 3,
        crc32: 0,
        _reserved: 0,
        routes_lo,
        routes_hi,
        proof_digest: [0u8; 32],
        _pad: [0u8; 672],
        init_masks,
    };
    page.crc32 = page.compute_crc32();
    page
}

pub static SEED_AGENCY: CompactLut6Page = build_seed_agency_crystal(BIO_DOMAIN_HASH);

// Compile-time biological operon verification (`rustc` refuses to build if seed invariants break!):
const _: () = {
    assert!(SEED_AGENCY.is_valid());
    assert!(SEED_AGENCY.verify_crc32());
    assert!(SEED_AGENCY.init_masks[2] == 0xFFFF_0000_FFFF_0000);
    assert!(SEED_AGENCY.init_masks[4] == 0xFFFF_FFF4_FFFF_FFF4);
    assert!(SEED_AGENCY.init_masks[5] == 0xFFFF_FFF8_FFFF_FFFC);

    // 1. Healthy Joule budget (w48=1) + Tool 0 Ready (w40=1) -> Exogenous Tool Codon `0x20` (`BUS_PIO_SPI`)
    let state_tool_ready = (1u64 << 40) | (1u64 << 48) | 0xCAFE_BABE;
    let operon_1 = SEED_AGENCY.eval_raw(state_tool_ready);
    assert!((operon_1 & 0x3F) == CODON_BUS_PIO_SPI as u64);
    assert!(((operon_1 >> 14) & 0xFFFF_FFFF) == 0xCAFE_BABE); // Signal peptide passed intact!

    // 2. Verifier Reject bit high (w63=1) -> immediately triggers Apoptosis Rollback Codon `0x3C`
    let state_reject = 1u64 << 63;
    let operon_2 = SEED_AGENCY.eval_raw(state_reject);
    assert!((operon_2 & 0x3F) == CODON_APOPTOSIS_ROLLBACK as u64);

    // 3. Zero Joule budget (w48=0) & no faults -> emits Dormant Stop Codon `0x00` (sleep MCU!)
    let operon_sleep = SEED_AGENCY.eval_raw(0x0000_0000_1234_5678);
    assert!((operon_sleep & 0x3F) == CODON_STOP_HALT as u64);

    // 4. Hop TTL maxed (w61=1) or Tool Fault (w62=1) -> emits Transcription Frontier `0x30`
    let operon_ttl = SEED_AGENCY.eval_raw((1u64 << 61) | (1u64 << 48));
    assert!((operon_ttl & 0x3F) == CODON_TRANSCRIBE_FRONTIER as u64);
};
