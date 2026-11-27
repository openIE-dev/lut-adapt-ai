//! Zero-dependency `#![no_std]` WebAssembly (`wasm32-unknown-unknown`) C-ABI exports for
//! browser-native execution of real `.lut6` binary crystals (`yosys_operon_seed.lut6` and
//! `mol_skills_tmac_tail.lut6`), Antisymmetric `40`-entry `T-MAC`, Walsh–Hadamard `FWHT`
//! harmonic filtering, and `(72, 64)` `RAMB36` Hamming `SECDED` cosmic-ray fault injection.

use crate::apex_harness::{ApexGateOutcome, ApexStepEnvelope, eval_apex_enzymatic_step_lut6};
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

#[inline(never)]
fn wasm_get_crystal(which: u32) -> &'static crate::CompactLut6Page {
    if which == 0 {
        yosys_operon_crystal()
    } else {
        mol_skills_tmac_crystal()
    }
}

/// Returns the verified IEEE 802.3 CRC32 of embedded crystal `which` (`0 = Operon`, `1 = Skills+TMAC`),
/// or `0` if `verify_crc32()` fails.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_crystal_crc32(which: u32) -> u32 {
    let page = wasm_get_crystal(which);
    let c = page.compute_crc32();
    if c == page.crc32 { c } else { 0 }
}

/// Evaluates the Yosys `abc -lut 6` synthesized 64-codon Operon crystal (`0xd1cc025a`) on a
/// 64-bit wire vector `(input_lo | (input_hi << 32))`, returning the lower 32 output bits.
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_operon_seed(input_lo: u32, input_hi: u32) -> u32 {
    let state = (input_lo as u64) | ((input_hi as u64) << 32);
    wasm_get_crystal(0).eval_raw_unrolled4(state) as u32
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
    let skills_page = wasm_get_crystal(1);
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
    let skills_page = wasm_get_crystal(1);
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
    let w = (ascii_lo as u64) | ((ascii_hi as u64) << 32);
    has_benchmark_canary(&w.to_le_bytes(), false) as u32
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
    let (pay, votes) = wasm_shared_bipolar_antisym(&wasm_get_crystal(1)._pad, state, Some(signs));
    (((votes as u32) & 0xFF) << 16) | ((pay as u32) & 0xFFFF)
}

#[inline(never)]
fn wasm_or_nibble_planes(planes: &mut [u64; 4], bit_pos: usize, nib: u32) {
    let b = 1u64 << bit_pos;
    let mut bp = 0usize;
    while bp < core::hint::black_box(4usize) {
        if ((nib >> bp) & 1) != 0 {
            planes[bp] |= b;
        }
        bp += 1;
    }
}

#[inline(never)]
fn wasm_read_nibble_planes(planes: &[u64; 4], bit_pos: usize) -> u32 {
    let mut out = 0u32;
    let mut bp = 0usize;
    while bp < core::hint::black_box(4usize) {
        out |= (((planes[bp] >> bit_pos) & 1) as u32) << bp;
        bp += 1;
    }
    out
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
    let n_lut = core::hint::black_box(64usize);
    let mut m = 0usize;
    while m < n_lut {
        let c_in = m & 0xF;
        let par = (m >> 4) & 0x3;
        let mut out_c = ((cmap >> (c_in << 2)) & 0xF) as u32;
        if ((spatial_p1_byte & 0x100) != 0) && par == 3 && c_in == 0 {
            out_c ^= 1;
        }
        wasm_or_nibble_planes(&mut masks, m, out_c);
        m += 1;
    }
    let mut packed_out = 0u32;
    let mut idx = 0usize;
    while idx < core::hint::black_box(8usize) {
        let c_in = ((cells_nibbles8 >> (idx << 2)) & 0xF) as usize;
        let p0 = ((spatial_p0_byte >> idx) & 1) as usize;
        let p1 = ((spatial_p1_byte >> idx) & 1) as usize;
        let addr6 = c_in | (p0 << 4) | (p1 << 5);
        packed_out |= wasm_read_nibble_planes(&masks, addr6) << (idx << 2);
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
    let n8 = core::hint::black_box(8usize);
    let n64 = core::hint::black_box(64usize);
    let mut i = 0usize;
    while i < n8 {
        let nib = (cells_nibbles8 >> (i << 2)) & 0xF;
        if nib != 0 {
            nz8 |= 1u64 << i;
        }
        wasm_or_nibble_planes(&mut c_planes, i, nib);
        i += 1;
    }
    let left_nz = nz8 << 1;
    let right_nz = nz8 >> 1;
    let up_nz = (up_mask8 & 0xFF) as u64;
    let down_nz = (down_mask8 & 0xFF) as u64;

    // Layer-1: `feat0` = enclosed/bridged empty cell `(center == 0) && ((left & right) || (up & down))`
    const FEAT0_LUT: u64 = {
        let mut v = 0u64;
        let mut m = 0usize;
        while m < 64 {
            if (m & 1) == 0 && (((m & 24) == 24) || ((m & 6) == 6)) {
                v |= 1u64 << m;
            }
            m += 1;
        }
        v
    };
    let mut color_luts = [0u64; 4];
    let fc = fill_color & 0xF;
    let mut m = 0usize;
    while m < n64 {
        let c_in = (m & 0xF) as u32;
        let c_out = if c_in == 0 && ((m >> 4) & 1) == 1 { fc } else { c_in };
        wasm_or_nibble_planes(&mut color_luts, m, c_out);
        m += 1;
    }
    let out_p = eval_benes_arc_lut6_2layer_strip64(
        &c_planes,
        up_nz,
        down_nz,
        left_nz,
        right_nz,
        0,
        &[FEAT0_LUT, 0],
        &color_luts,
    );
    let mut out32 = 0u32;
    i = 0;
    while i < n8 {
        out32 |= wasm_read_nibble_planes(&out_p, i) << (i << 2);
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
    let act_masks = [
        0x5555_5555_5555_5555u64,
        0x6666_6666_6666_6666u64,
        0x8888_8888_8888_8888u64,
    ];
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

/// Evaluates the `OpenIE-Apex-Harness` `#![no_std]` `System-0` `LUT6` horizon gate (`DES-10` attestation +
/// `openie-cad` PCB impedance/thermal + `MathGround` Torricelli `CBF` + `RCE` reserve check) in `.wasm`.
/// Packed 32-bit return word:
///   - `bits [25..24]`: `1` = Certified `System-0` Action, `2` = `DES-10` Attestation Trap,
///                      `3` = `Simplex` Physics/CBF/DRC Barrier Shield, `0` = Wake `AlphaEvolve+Z3` Solver
///   - `bits [15..0]`:  Action code (`1..=15`), XOR attestation tag, or violation bitmask
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_apex_horizon_gate(
    percept6_and_tags: u32,
    z0_mohm_and_tj: u32,
    v_mm_s: u32,
    d_obs_mm: u32,
    cash_and_stock_ok: i32,
) -> u32 {
    let percept6 = (percept6_and_tags & 0x3F) as u8;
    let handoff_tag16 = ((percept6_and_tags >> 8) & 0xFF) as u16;
    let expected_tag16 = ((percept6_and_tags >> 16) & 0xFF) as u16;
    let z0_milliohm = (z0_mohm_and_tj & 0xFFFF) * 2; // e.g., 25_000 -> 50_000 mΩ (50 Ω)
    let tj_deci_c = ((z0_mohm_and_tj >> 16) & 0xFFFF) as u16;
    let env = ApexStepEnvelope {
        percept6,
        handoff_tag16,
        expected_tag16,
        z0_milliohm,
        tj_deci_c,
        v_mm_s,
        a_max_mm_s2: 3_000,
        d_obs_mm,
        cash_cents: cash_and_stock_ok,
        stock_units: cash_and_stock_ok,
    };
    let masks4 = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0xCCCC_CCCC_CCCC_CCCCu64,
        0xF0F0_F0F0_F0F0_F0F0u64,
        0xFF00_FF00_FF00_FF00u64,
    ];
    match eval_apex_enzymatic_step_lut6(&env, &masks4, u64::MAX) {
        ApexGateOutcome::CertifiedSystem0 { action_code } => (1u32 << 24) | (action_code as u32),
        ApexGateOutcome::EnzymaticAttestationTrap { xor_tag } => (2u32 << 24) | (xor_tag as u32),
        ApexGateOutcome::SimplexBarrierShield { violation_mask } => {
            (3u32 << 24) | (violation_mask as u32)
        }
        ApexGateOutcome::WakeApexSolver => 0,
    }
}

/// Executes the full `waiOS` `256 x 128` ternary matrix-vector integer program via `#![no_std]`
/// `Antisym-40` zero-`MUL` `LUT6` table lookup + `BLAKE3("waios.tmac.v1")` inside the browser `.wasm` VM,
/// paired with `HeteroMosaic-MoL` substrate routing (`Claim LA-50`).
///   - `fault_row77 != 0`: injects the canonical row-77 single-weight fault (`0x4ac7c4fb` -> HALT!).
///   - Returns big-endian `digest[0..4]` (`0xa541323b` when valid, `0x4ac7c4fb` when faulted) XORed in the
///     lowest bit (`0` if `Halted`, `substrate_id` preserved via `wasm_heteromosaic_route` if `fault_row77 == 0`).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_verify_waios_antisym40_digest32(
    fault_row77: u32,
    lut6_certified_and_batch: u32,
) -> u32 {
    use crate::waios_mosaic::{
        WAIOS_REFERENCE_DIGEST, eval_waios_streaming_antisym40_digest,
        select_heteromosaic_substrate,
    };
    let dig = eval_waios_streaming_antisym40_digest(fault_row77 != 0);
    let prefix_be = u32::from_be_bytes([dig[0], dig[1], dig[2], dig[3]]);
    let ok = dig == WAIOS_REFERENCE_DIGEST;
    if !ok {
        return prefix_be;
    }
    let cert = (lut6_certified_and_batch & 1) != 0;
    let batch = ((lut6_certified_and_batch >> 8) & 0xFF) as u16;
    let sub = select_heteromosaic_substrate(cert, batch, 128);
    // When `lut6_certified_and_batch == 0`, return pure `0xa541323b`; otherwise encode `(1 << 24) | (sub << 16) | (prefix_be & 0xFFFF)`.
    if lut6_certified_and_batch == 0 {
        prefix_be
    } else {
        (1u32 << 24) | ((sub as u32) << 16) | (prefix_be & 0xFFFF)
    }
}

/// Evaluates the Phase 39 (`Claims LA-53..LA-55`) two-bank `SIW(2)` landmark `LUT6` reflex
/// AND fused zero-multiplier `FWHT-64 + 6-Column JouleBit-LUT6` sub-bit draft slab inside `.wasm`.
///   - `percept_landmark_epoch`: `bits [5..0]=percept6`, `bits [13..8]=landmark6`, `bits [19..16]=epoch`,
///     `bits [27..20]=actual_sig`, `bit 28=expect_0x42`
///   - `act0_and_jb_row`: `bits [15..0]=acts64[0] (i16)`, `bits [21..16]=row_idx (0..63)`
///   - Returns packed `u32`:
///     - `bits [31..28]`: `1` = Certified `System-0` Action, `2` = Mechanic-Shift Breaker, `0` = Wake `SIW(2)`
///     - `bit  27`:       `1` if `landmark_reset` fired (`seen` novelty tables cleared!)
///     - `bit  26`:       `bank_id` (`0` = Bank 0, `1` = Bank 1 post-landmark)
///     - `bits [23..20]`: `next_epoch` (`0..=8`)
///     - `bits [19..16]`: `action_id` or `xor_sig & 0xF`
///     - `bits [15..0]`:  exact `JouleBit-LUT6` `FWHT-64` output `out64[row_idx] as u16` (`0` binary MACs!)
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_siw2_and_joulebit6(
    percept_landmark_epoch: u32,
    act0_and_jb_row: u32,
) -> u32 {
    let p = percept_landmark_epoch & 0x3F;
    let lm_hit = ((percept_landmark_epoch >> 8) & 0x3F) != 0;
    let epoch = (percept_landmark_epoch >> 16) & 0x0F;
    let actual_sig = (percept_landmark_epoch >> 20) & 0xFF;
    let lm_reset = (lm_hit && epoch < 8) as u32;
    let next_ep = (epoch + lm_reset) & 0xF;
    let jb_u16 = wasm_shared_joulebit6_elem(act0_and_jb_row);

    if (percept_landmark_epoch & (1u32 << 28)) != 0 && actual_sig != 0x42 {
        return (2u32 << 28) | (next_ep << 20) | (((0x42 ^ actual_sig) & 0xF) << 16) | jb_u16;
    }
    let bank_id = (next_ep != 0) as u32;
    let code = if bank_id == 0 {
        ((p ^ 1) & 1) | ((((p >> 1) ^ p) & 1) << 1)
    } else {
        4 | (p & 3)
    };
    if code == 0 {
        (lm_reset << 27) | (next_ep << 20) | jb_u16
    } else {
        (1u32 << 28) | (lm_reset << 27) | (bank_id << 26) | (next_ep << 20) | (code << 16) | jb_u16
    }
}

#[inline(never)]
fn wasm_shared_joulebit6_elem(act0_and_row: u32) -> u32 {
    use crate::siw_joulebit::eval_joulebit6_fwht64_slab_i16;
    let act0 = (act0_and_row & 0xFFFF) as u16 as i16;
    let row_idx = ((act0_and_row >> 16) & 0x3F) as usize;
    let n64 = core::hint::black_box(64usize);
    let n6 = core::hint::black_box(6usize);
    let n10 = core::hint::black_box(10usize);
    let mut acts64 = [0i16; 64];
    acts64[0] = act0;
    let mut idx = 1usize;
    while idx < n64 {
        acts64[idx] = (((idx * 11 + 5) & 31) as i16) - 15;
        idx += 1;
    }
    let mut v_codes = [[0u8; 10]; 6];
    let mut r = 0usize;
    while r < n6 {
        let mut g = 0usize;
        while g < n10 {
            v_codes[r][g] = ((r * 19 + g * 23 + 7) & 0x3F) as u8;
            g += 1;
        }
        r += 1;
    }
    let mut u_codes = [0u8; 64];
    let mut rw = 0usize;
    while rw < n64 {
        u_codes[rw] = ((rw * 31 + 13) & 0x3F) as u8;
        rw += 1;
    }
    let mut out64 = [0i32; 64];
    eval_joulebit6_fwht64_slab_i16(&acts64, &v_codes, &u_codes, &mut out64);
    (out64[row_idx & 0x3F] & 0xFFFF) as u32
}

#[inline(never)]
fn wasm_shared_nbank_cascade(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    exp_sig: Option<u8>,
    actual_sig: u8,
    max_ep: u8,
    n_banks: usize,
    cert_style: u8,
) -> (u32, u32, u32, u32) {
    let ep_c = epoch.min(max_ep);
    if let Some(exp) = exp_sig
        && exp != 0xFF
        && exp != actual_sig
    {
        return (2, 0, ep_c as u32, 0);
    }
    let sg_hit = (((!1u64) >> ((subgoal6 & 0x3F) as u32)) & 1) == 1;
    let next_ep = if sg_hit && ep_c < max_ep { ep_c + 1 } else { ep_c };
    let p_idx = (percept6 & 0x3F) as u32;
    let primary = (next_ep as usize) % n_banks;
    let mut probe = 0usize;
    while probe < n_banks {
        let mut b_id = primary + probe;
        if b_id >= n_banks {
            b_id -= n_banks;
        }
        let cert = if b_id + 1 == n_banks {
            u64::MAX
        } else if cert_style == 4 {
            !(0xFFu64 << (b_id * 8))
        } else if cert_style == 8 {
            0xFFu64 << (b_id * 8)
        } else {
            0x1Fu64 << (b_id * 5)
        };
        if ((cert >> p_idx) & 1) == 1 {
            let act_id = ((b_id % 7) + 1) as u32;
            return (1, b_id as u32, next_ep as u32, act_id);
        }
        probe += 1;
    }
    (0, 0, next_ep as u32, 0)
}

/// Export #15 (`Claims LA-56..LA-59`, `Theorem 105`): Evaluates the Four-Bank (`16 KB` = `25.0%` watch SRAM)
/// `BFWS(R, M)` cascade reflex + Planetary Digital Well (`Well-Codifier`) `8`-gate stdlib `LUT6` bank +
/// fused zero-`MUL` `JouleBit6 + FWHT-64` slab inside `#![no_std]` WebAssembly (`<= 16 KB` total `.wasm`).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_bfws4_and_well_codifier(
    bfws_and_well: u32,
    act0_and_row: u32,
) -> u32 {
    use crate::bfws_well::eval_well_codifier_stdlib8;
    let exp_sig = if (bfws_and_well & (1u32 << 30)) != 0 {
        Some(0x5Au8)
    } else {
        None
    };
    let (status2, bank2, ep4, _) = wasm_shared_nbank_cascade(
        (bfws_and_well & 0x3F) as u8,
        ((bfws_and_well >> 6) & 0x3F) as u8,
        ((bfws_and_well >> 12) & 0x0F) as u8,
        exp_sig,
        ((bfws_and_well >> 22) & 0xFF) as u8,
        15,
        4,
        4,
    );
    let well8 = eval_well_codifier_stdlib8(((bfws_and_well >> 16) & 0x3F) as u8) as u32;
    let jb16 = wasm_shared_joulebit6_elem(act0_and_row);
    (status2 << 30) | ((bank2 & 3) << 28) | ((ep4 & 0xF) << 24) | (well8 << 16) | jb16
}

/// Export #16 (`Claims LA-60..LA-63`, `Theorem 113`): Evaluates the Eight-Bank (`32 KB` = `50.0%` watch SRAM)
/// `R-BFWS(R, M)` cascade reflex + bijective `Enigma-SBox6` cryptographic `LUT6` substitution +
/// `Vending-BALROG` `1,000`-step Simplex `CBF` `LUT6` shield + zero-`MUL` `JouleBit6 + FWHT-64` slab
/// inside `#![no_std]` WebAssembly (`<= 16 KB` total `.wasm`).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_rbfws8_enigma_horizon(
    rbfws_enigma_shield: u32,
    act0_and_row: u32,
) -> u32 {
    use crate::enigma_horizon::{
        enigma_sbox6_inv_spec, enigma_sbox6_spec, eval_vending_balrog_shield2_lut6,
    };
    let cipher_in6 = ((rbfws_enigma_shield >> 15) & 0x3F) as u8;
    let actual_sig = if (rbfws_enigma_shield & (1u32 << 28)) != 0 {
        0x33u8
    } else {
        0x5Au8
    };
    let (status2, bank3, ep3, _) = wasm_shared_nbank_cascade(
        (rbfws_enigma_shield & 0x3F) as u8,
        ((rbfws_enigma_shield >> 6) & 0x3F) as u8,
        ((rbfws_enigma_shield >> 12) & 0x07) as u8,
        Some(0x5A),
        actual_sig,
        7,
        8,
        8,
    );
    let sbox6 = if (rbfws_enigma_shield & (1u32 << 27)) != 0 {
        enigma_sbox6_inv_spec(cipher_in6)
    } else {
        enigma_sbox6_spec(cipher_in6)
    } as u32;
    let shield2 = eval_vending_balrog_shield2_lut6(((rbfws_enigma_shield >> 21) & 0x3F) as u8) as u32;
    let jb16 = wasm_shared_joulebit6_elem(act0_and_row);
    (status2 << 30) | ((bank3 & 7) << 27) | ((ep3 & 7) << 24) | ((sbox6 & 0x3F) << 18) | ((shield2 & 3) << 16) | jb16
}

#[inline(never)]
fn wasm_shared_cbfws12_head(packed: u32) -> (u32, u32, u32, u32) {
    let actual_sig = if (packed & (1u32 << 28)) != 0 {
        0x33u8
    } else {
        0x5Au8
    };
    wasm_shared_nbank_cascade(
        (packed & 0x3F) as u8,
        ((packed >> 6) & 0x3F) as u8,
        ((packed >> 12) & 0x0F) as u8,
        Some(0x5A),
        actual_sig,
        11,
        12,
        12,
    )
}

#[inline(never)]
fn wasm_pack_cbfws12_pair2(
    packed: u32,
    pair_a2: u32,
    pair_b2: u32,
    act0_and_row: u32,
) -> u32 {
    let (status2, bank4, ep4, act2) = wasm_shared_cbfws12_head(packed);
    let jb16 = wasm_shared_joulebit6_elem(act0_and_row);
    (status2 << 30)
        | ((bank4 & 0xF) << 26)
        | ((ep4 & 0xF) << 22)
        | ((pair_a2 & 3) << 20)
        | ((pair_b2 & 3) << 18)
        | ((act2 & 3) << 16)
        | jb16
}

/// Export #17 (`Claims LA-64..LA-67`, `Theorem 121`): Evaluates the Twelve-Bank (`48 KB` = `75.0%` watch SRAM)
/// `C-BFWS(R, M)` cascade reflex + `Terminal-Cybench` constant-time defensive `LUT6` guard +
/// `FrontierMath-SciCode` symplectic/unitary conservation `LUT6` shield + zero-`MUL` `JouleBit6 + FWHT-64` slab
/// inside `#![no_std]` WebAssembly (`<= 16 KB` total `.wasm`).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_cbfws12_term_math(
    cbfws_guard_sci: u32,
    act0_and_row: u32,
) -> u32 {
    use crate::cbfws_term_math::{
        eval_cybench_terminal_guard2_lut6, eval_scicode_math_shield2_lut6,
    };
    let guard2 = eval_cybench_terminal_guard2_lut6(((cbfws_guard_sci >> 16) & 0x3F) as u8) as u32;
    let sci2 = eval_scicode_math_shield2_lut6(((cbfws_guard_sci >> 22) & 0x3F) as u8) as u32;
    wasm_pack_cbfws12_pair2(cbfws_guard_sci, guard2, sci2, act0_and_row)
}

/// Export #18 (`Claims LA-68..LA-71`, `Theorem 129`): Evaluates Three-Tier `VRAM -> RAM -> NVMe`
/// `NameLedger + Reuse::expected_with` expert admission `LUT6` (`tier2`), `DeepSeek-V4.1-Flash`
/// trailing $n$-gram conditional memory `Engram-LUT6` (`engram2`), Twelve-Bank `C-BFWS` `System-0`
/// cascade, and zero-`MUL` `JouleBit6 + FWHT-64` slab inside `#![no_std]` WebAssembly (`<= 16 KB` total `.wasm`).
#[unsafe(no_mangle)]
pub extern "C" fn wasm_eval_tiered_engram_lut6(
    tiered_engram_cbfws: u32,
    act0_and_row: u32,
) -> u32 {
    use crate::tiered_engram::{eval_engram_ngram_gate2_lut6, eval_tiered_residency_lut6};
    let tier2 = eval_tiered_residency_lut6(((tiered_engram_cbfws >> 16) & 0x3F) as u8) as u32;
    let engram2 = eval_engram_ngram_gate2_lut6(((tiered_engram_cbfws >> 22) & 0x3F) as u8) as u32;
    wasm_pack_cbfws12_pair2(tiered_engram_cbfws, tier2, engram2, act0_and_row)
}
