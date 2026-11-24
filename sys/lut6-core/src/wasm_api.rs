//! Zero-dependency `#![no_std]` WebAssembly (`wasm32-unknown-unknown`) C-ABI exports for
//! browser-native execution of real `.lut6` binary crystals (`yosys_operon_seed.lut6` and
//! `mol_skills_tmac_tail.lut6`), Antisymmetric `40`-entry `T-MAC`, Walsh–Hadamard `FWHT`
//! harmonic filtering, and `(72, 64)` `RAMB36` Hamming `SECDED` cosmic-ray fault injection.

use crate::generated_proof_vectors::{mol_skills_tmac_crystal, yosys_operon_crystal};
use crate::seu_resilience::{SecdedOutcome, decode_secded_72, encode_secded_72};
use crate::tmac_tail::{
    eval_tmac_antisym_from_pad, eval_tmac_from_pad, wire_state_to_bipolar_i8,
};
use crate::{EvalOutcome, PageLedger, high_harmonic_energy};

/// Returns the verified IEEE 802.3 CRC32 of embedded crystal `which` (`0 = Operon`, `1 = Skills+TMAC`),
/// or `0` if `verify_crc32()` fails.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_crystal_crc32(which: u32) -> u32 {
    let page = if which == 0 {
        yosys_operon_crystal()
    } else {
        mol_skills_tmac_crystal()
    };
    if page.verify_crc32() {
        page.compute_crc32()
    } else {
        0
    }
}

/// Evaluates the Yosys `abc -lut 6` synthesized 64-codon Operon crystal (`0xd1cc025a`) on a
/// 64-bit wire vector `(input_lo | (input_hi << 32))`, returning the lower 32 output bits.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_operon_seed(input_lo: u32, input_hi: u32) -> u32 {
    let state = (input_lo as u64) | ((input_hi as u64) << 32);
    yosys_operon_crystal().eval_raw_unrolled4(state) as u32
}

/// Evaluates the Two-Tier `LUT6 + 672 B EntroMCU/T-MAC` crystal (`0x3b12a681`) in `.wasm`.
/// Packed 32-bit return word:
///   - `bits [25..24]`: Tier (`1` = Rung-6 `LUT6` hit, `2` = Rung-5.5 zero-MUL `T-MAC` hit, `0` = OOD Escalate)
///   - `bits [23..16]`: Conformal vote count (`0..16`)
///   - `bits [15..0]`:  Certified 16-bit skill payload
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_two_tier_skill(
    input_lo: u32,
    input_hi: u32,
    use_antisym40: u32,
) -> u32 {
    let skills_page = mol_skills_tmac_crystal();
    let state = (input_lo as u64) | ((input_hi as u64) << 32);
    let mut ledger = PageLedger::default();
    let Ok(r6) = skills_page.evaluate(state, None, &mut ledger);
    if let EvalOutcome::Act { output_bits, votes } = r6 {
        return (1u32 << 24) | (((votes as u32) & 0xFF) << 16) | ((output_bits as u32) & 0xFFFF);
    }
    let feats = wire_state_to_bipolar_i8(state);
    let (pay, votes, _) = if use_antisym40 != 0 {
        eval_tmac_antisym_from_pad(&skills_page._pad, &feats)
    } else {
        eval_tmac_from_pad(&skills_page._pad, &feats)
    };
    if votes >= skills_page.conformal_thresh {
        (2u32 << 24) | (((votes as u32) & 0xFF) << 16) | ((pay as u32) & 0xFFFF)
    } else {
        (((votes as u32) & 0xFF) << 16) | ((pay as u32) & 0xFFFF)
    }
}

/// Computes the exact degree-`>=5` Walsh–Hadamard high-harmonic noise energy (`0..=4096`)
/// of any 64-bit `LUT6` `INIT` truth table `(mask_lo | (mask_hi << 32))`.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_walsh_high_harmonic_energy(mask_lo: u32, mask_hi: u32) -> u32 {
    let mask = (mask_lo as u64) | ((mask_hi as u64) << 32);
    high_harmonic_energy(mask)
}

/// Injects `0`, `1`, or `2` cosmic-ray bit-flips (`flip_a, flip_b ∈ -1..71`, where `0..63` flip
/// `LUT6` `INIT` bits and `64..71` flip `ECC` parity bits) into a `(72, 64)` `RAMB36` word.
/// Packed return:
///   - `bits [25..24]`: Status (`0` = Clean, `1` = Single-Bit Self-Healed (`100%` exact),
///                              `2` = Double-Bit Detected -> `0x3C` Apoptosis Rollback)
///   - `bits [23..16]`: Hamming syndrome (`0..127`)
///   - `bits [15..8]`:  Raw unguarded minterms corrupted (`0..64`)
///   - `bits [7..0]`:   Post-`SECDED`/Apoptosis corrupted minterms (`0` = zero silent corruption!)
#[unsafe(no_mangle)]
pub extern "C" fn wasm_test_secded_bitflip(
    mask_lo: u32,
    mask_hi: u32,
    flip_a: i32,
    flip_b: i32,
) -> u32 {
    let orig = (mask_lo as u64) | ((mask_hi as u64) << 32);
    let orig_ecc = encode_secded_72(orig);

    let mut word = orig;
    let mut ecc = orig_ecc;
    for fb in [flip_a, flip_b] {
        if (0..64).contains(&fb) {
            word ^= 1u64 << (fb as u32);
        } else if (64..72).contains(&fb) {
            ecc ^= 1u8 << ((fb - 64) as u32);
        }
    }
    let raw_corrupted = (orig ^ word).count_ones() & 0xFF;
    match decode_secded_72(word, ecc) {
        SecdedOutcome::Clean(recovered) => {
            let post_err = (orig ^ recovered).count_ones() & 0xFF;
            (raw_corrupted << 8) | post_err
        }
        SecdedOutcome::CorrectedSingle { data, syndrome } => {
            let post_err = (orig ^ data).count_ones() & 0xFF;
            (1u32 << 24) | ((syndrome as u32) << 16) | (raw_corrupted << 8) | post_err
        }
        SecdedOutcome::DetectedUncorrectable { syndrome } => {
            // Triggers `0x3C` ApoptosisRollback to immutable ROM/MRAM seed (`orig`), so `0` silent corruption!
            (2u32 << 24) | ((syndrome as u32) << 16) | (raw_corrupted << 8)
        }
    }
}
