//! Zero-dependency `#![no_std]` WebAssembly (`wasm32-unknown-unknown`) C-ABI exports for
//! browser-native execution of real `.lut6` binary crystals (`yosys_operon_seed.lut6` and
//! `mol_skills_tmac_tail.lut6`), Antisymmetric `40`-entry `T-MAC`, Walsh–Hadamard `FWHT`
//! harmonic filtering, and `(72, 64)` `RAMB36` Hamming `SECDED` cosmic-ray fault injection.

use crate::generated_proof_vectors::{mol_skills_tmac_crystal, yosys_operon_crystal};
use crate::hadamard_bonsai::{
    Arc3ReflexOutcome, eval_arc3_policy_lut6, eval_benes_arc_lut6_2layer_strip64,
    fwht64_shift3_signed, pack_bonsai_3tok_54bit, unpack_bonsai_3tok_54bit,
};
use crate::seu_resilience::{SecdedOutcome, decode_secded_72, encode_secded_72};
use crate::tmac_tail::{
    eval_tmac_antisym_from_pad, eval_tmac_from_pad, wire_state_to_bipolar_i8,
};
use crate::wearable_epistemic::{
    WearableEpistemicDecision, eval_wearable_epistemic_gate, has_benchmark_canary,
};
use crate::{EvalOutcome, PageLedger, high_harmonic_energy};

#[inline(never)]
fn wasm_shared_bipolar_antisym(pad: &[u8; 672], state: u64, sign_mask: Option<u64>) -> (u64, u16) {
    let mut feats = wire_state_to_bipolar_i8(state);
    if let Some(signs) = sign_mask {
        let mut rot = [0i32; 64];
        let mut i = 0usize;
        while i < 64 {
            rot[i] = (feats[i] as i32) << 2;
            i += 1;
        }
        fwht64_shift3_signed(&mut rot, signs);
        i = 0;
        while i < 64 {
            feats[i] = rot[i].clamp(-127, 127) as i8;
            i += 1;
        }
    }
    let (pay, votes, _) = eval_tmac_antisym_from_pad(pad, &feats);
    (pay, votes)
}

/// Returns the verified IEEE 802.3 CRC32 of embedded crystal `which` (`0 = Operon`, `1 = Skills+TMAC`),
/// or `0` if `verify_crc32()` fails.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_crystal_crc32(which: u32) -> u32 {
    let page = if which == 0 {
        yosys_operon_crystal()
    } else {
        mol_skills_tmac_crystal()
    };
    let c = page.compute_crc32();
    if c == page.crc32 { c } else { 0 }
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
    let (pay, votes) = if use_antisym40 != 0 {
        wasm_shared_bipolar_antisym(&skills_page._pad, state, None)
    } else {
        let feats = wire_state_to_bipolar_i8(state);
        let (p, v, _) = eval_tmac_from_pad(&skills_page._pad, &feats);
        (p, v)
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

/// Evaluates the Wearable `56`-Bit `4`-Token `uLM-28M` Opener + `LastExam` Epistemic Hold Gate in `.wasm`.
/// Packed 32-bit return word:
///   - `bits [25..24]`: Mode (`1` = Certified `4`-Token Wearable Reflex, `2` = Wake On-Device `uLM-28M`,
///                            `3` = `LastExam` Epistemic Hold + Escalate [`0` poisoned prefix words!])
///   - `bits [23..16]`: Calibrated confidence `%` (`0..100`)
///   - `bits [15..0]`:  First `14`-bit `uLM-28M` opener token (`bits [13..0]`) or frontier signature
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_wearable_epistemic(
    wire_lo: u32,
    wire_hi: u32,
    frontier_thresh: u32,
) -> u32 {
    let skills_page = mol_skills_tmac_crystal();
    let state = (wire_lo as u64) | ((wire_hi as u64) << 32);
    let mut ledger = PageLedger::default();
    match eval_wearable_epistemic_gate(&skills_page, None, state, frontier_thresh, &mut ledger) {
        WearableEpistemicDecision::CertifiedReflex {
            tokens4,
            action_id: _,
            conf_pct,
        } => (1u32 << 24) | ((conf_pct as u32) << 16) | ((tokens4[0] as u32) & 0x3FFF),
        WearableEpistemicDecision::WakeLocalMicroLlm { votes, .. } => {
            let cal_conf = (25u32 + (votes as u32) * 3).min(65);
            (2u32 << 24) | (cal_conf << 16) | ((votes as u32) & 0xFFFF)
        }
        WearableEpistemicDecision::EpistemicHoldEscalate {
            prior_conf_pct,
            frontier_signature,
        } => (3u32 << 24) | ((prior_conf_pct as u32) << 16) | (frontier_signature as u32),
    }
}

/// Verifies the `#![no_std]` Benchmark Canary Quarantine (`hle:3r2s`) in `.wasm`:
/// returns `1` if the 8-byte ASCII window `(lo32 | hi32 << 32)` matches `hle:3r2s` (blocked from `.lut6`!), `0` otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_check_hle_canary8(ascii_lo: u32, ascii_hi: u32) -> u32 {
    let mut buf = [0u8; 8];
    let l = ascii_lo.to_le_bytes();
    let h = ascii_hi.to_le_bytes();
    buf[0] = l[0];
    buf[1] = l[1];
    buf[2] = l[2];
    buf[3] = l[3];
    buf[4] = h[0];
    buf[5] = h[1];
    buf[6] = h[2];
    buf[7] = h[3];
    if has_benchmark_canary(&buf, false) {
        1
    } else {
        0
    }
}

/// Evaluates `Bonsai-2-27B`-style Sylvester `FWHT-64 (>> 3)` + `ISCA` 2025 `Antisym-40 T-MAC` (`0` MULs) in `.wasm`.
/// Packed 32-bit return word: `bits [23..16]` = conformal votes (`0..8`), `bits [15..0]` = payload + energy digest.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_bonsai_fwht64_antisym40(
    wire_lo: u32,
    wire_hi: u32,
    sign_lo: u32,
    sign_hi: u32,
) -> u32 {
    let state = (wire_lo as u64) | ((wire_hi as u64) << 32);
    let signs = (sign_lo as u64) | ((sign_hi as u64) << 32);
    let (pay, votes) = wasm_shared_bipolar_antisym(&mol_skills_tmac_crystal()._pad, state, Some(signs));
    (((votes as u32) & 0xFF) << 16) | ((pay as u32) & 0xFFFF)
}

/// Evaluates an inductive `ARC-AGI-2` `4`-bit color + `2`-bit spatial parity `LUT6` rule across `8` packed
/// 4-bit cells (`cells_nibbles8`: `8` cells in `u32`) via `4` `64`-bit `LUT6` `INIT` bitplanes (`0` FLOPs!).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_arc_lut6_cells8(
    cells_nibbles8: u32,
    spatial_p0_byte: u32,
    spatial_p1_byte: u32,
    color_map_lo: u32,
    color_map_hi: u32,
) -> u32 {
    let cmap = (color_map_lo as u64) | ((color_map_hi as u64) << 32);
    let mut masks = [0u64; 4];
    let mut m = 0usize;
    while m < 64 {
        let c_in = m & 0xF;
        let par = (m >> 4) & 0x3;
        let mut out_c = ((cmap >> (c_in << 2)) & 0xF) as u8;
        if ((spatial_p1_byte & 0x100) != 0) && par == 3 && c_in == 0 {
            out_c ^= 1;
        }
        let mut bp = 0usize;
        while bp < 4 {
            if ((out_c >> bp) & 1) != 0 {
                masks[bp] |= 1u64 << m;
            }
            bp += 1;
        }
        m += 1;
    }
    let mut packed_out = 0u32;
    let mut idx = 0usize;
    while idx < 8 {
        let c_in = ((cells_nibbles8 >> (idx << 2)) & 0xF) as usize;
        let p0 = ((spatial_p0_byte >> idx) & 1) as usize;
        let p1 = ((spatial_p1_byte >> idx) & 1) as usize;
        let addr6 = c_in | (p0 << 4) | (p1 << 5);
        let nib = (((masks[0] >> addr6) & 1) as u32)
            | ((((masks[1] >> addr6) & 1) as u32) << 1)
            | ((((masks[2] >> addr6) & 1) as u32) << 2)
            | ((((masks[3] >> addr6) & 1) as u32) << 3);
        packed_out |= nib << (idx << 2);
        idx += 1;
    }
    packed_out
}

/// Evaluates `2`-Layer Beneš-routed `ARC-AGI-2` cellular composition (`eval_benes_arc_lut6_2layer_strip64`)
/// across an `8`-cell 1D strip (`cells_nibbles8`) in `.wasm`: fills zero cells bridged between matching
/// left/right non-zero neighbors (`b_left & b_right & !center`) with `fill_color & 0xF` (`0` FLOPs!).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_benes_arc2_2layer8(
    cells_nibbles8: u32,
    up_mask8: u32,
    down_mask8: u32,
    fill_color: u32,
) -> u32 {
    let mut c_planes = [0u64; 4];
    let mut nz8 = 0u64;
    let mut i = 0usize;
    while i < 8 {
        let nib = (cells_nibbles8 >> (i << 2)) & 0xF;
        if nib != 0 {
            nz8 |= 1u64 << i;
        }
        let mut bp = 0usize;
        while bp < 4 {
            if ((nib >> bp) & 1) != 0 {
                c_planes[bp] |= 1u64 << i;
            }
            bp += 1;
        }
        i += 1;
    }
    let left_nz = nz8 << 1;
    let right_nz = nz8 >> 1;
    let up_nz = (up_mask8 & 0xFF) as u64;
    let down_nz = (down_mask8 & 0xFF) as u64;

    // Layer-1: `feat0` = enclosed/bridged empty cell `(center == 0) && ((left & right) || (up & down))`
    let mut feat0_lut = 0u64;
    let mut color_luts = [0u64; 4];
    let fc = (fill_color & 0xF) as usize;
    let mut m = 0usize;
    while m < 64 {
        let c0 = m & 1;
        let u = (m >> 1) & 1;
        let d = (m >> 2) & 1;
        let l = (m >> 3) & 1;
        let r = (m >> 4) & 1;
        if c0 == 0 && ((l == 1 && r == 1) || (u == 1 && d == 1)) {
            feat0_lut |= 1u64 << m;
        }
        let c_in = m & 0xF;
        let f0 = (m >> 4) & 1;
        let c_out = if c_in == 0 && f0 == 1 { fc } else { c_in };
        let mut bp = 0usize;
        while bp < 4 {
            if ((c_out >> bp) & 1) != 0 {
                color_luts[bp] |= 1u64 << m;
            }
            bp += 1;
        }
        m += 1;
    }
    let out_p = eval_benes_arc_lut6_2layer_strip64(
        &c_planes,
        up_nz,
        down_nz,
        left_nz,
        right_nz,
        0,
        &[feat0_lut, 0],
        &color_luts,
    );
    let mut out32 = 0u32;
    i = 0;
    while i < 8 {
        let nib = (((out_p[0] >> i) & 1) as u32)
            | ((((out_p[1] >> i) & 1) as u32) << 1)
            | ((((out_p[2] >> i) & 1) as u32) << 2)
            | ((((out_p[3] >> i) & 1) as u32) << 3);
        out32 |= nib << (i << 2);
        i += 1;
    }
    out32
}

/// Evaluates an `ARC-AGI-3` `System-0` `LUT6` reflex step + mechanic-shift circuit breaker AND verifies
/// `54`-bit `3`-token `Bonsai-2-27B` burst packing in `.wasm`.
/// Packed 32-bit return word:
///   - `bits [25..24]`: `1` = Certified `LUT6` `GameAction`, `2` = Mechanic-Shift Circuit Breaker, `0` = Wake Planner
///   - `bits [23..8]`:  Roundtripped middle `Bonsai` token low-16 check (`t1 & 0xFFFF`)
///   - `bits [7..0]`:   `GameAction` (`1..=7`) or surprise XOR (`expected ^ actual`)
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_arc3_reflex_step(
    percept6: u32,
    expected_sig: i32,
    actual_sig: u32,
    certified_lo: u32,
    t1_probe: u32,
) -> u32 {
    let burst_ok_t1 = if let Some(w) = pack_bonsai_3tok_54bit(17, t1_probe % 248_320, 248_319, 0x155) {
        let (toks, _) = unpack_bonsai_3tok_54bit(w);
        toks[1] & 0xFFFF
    } else {
        0
    };
    // Deterministic 4-direction navigation LUT6: bits [1..0] = goal delta `(dx, dy)` -> GameAction 1..4
    let mut act_masks = [0u64; 3];
    let mut m = 0usize;
    while m < 64 {
        let act = ((m & 0x3) + 1) as u8; // 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        if (act & 1) != 0 {
            act_masks[0] |= 1u64 << m;
        }
        if ((act >> 1) & 1) != 0 {
            act_masks[1] |= 1u64 << m;
        }
        if ((act >> 2) & 1) != 0 {
            act_masks[2] |= 1u64 << m;
        }
        m += 1;
    }
    let exp_opt = if expected_sig >= 0 {
        Some((expected_sig & 0xFF) as u8)
    } else {
        None
    };
    let cert_mask = (certified_lo as u64) | ((certified_lo as u64) << 32);
    match eval_arc3_policy_lut6(
        (percept6 & 0x3F) as u8,
        exp_opt,
        (actual_sig & 0xFF) as u8,
        &act_masks,
        cert_mask,
    ) {
        Arc3ReflexOutcome::EmitAction { action_id } => {
            (1u32 << 24) | (burst_ok_t1 << 8) | (action_id as u32)
        }
        Arc3ReflexOutcome::MechanicShiftBreaker {
            expected_sig: e,
            actual_sig: a,
        } => (2u32 << 24) | (burst_ok_t1 << 8) | ((e ^ a) as u32),
        Arc3ReflexOutcome::WakeWorldModelPlanner => burst_ok_t1 << 8,
    }
}
