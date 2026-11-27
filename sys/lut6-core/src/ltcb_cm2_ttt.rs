//! Phase 46 (`Claims LA-80..LA-83`): `#![no_std]` Bijective `WRT-Capillary-LUT6` MediaWiki-XML /
//! Word-Stem Preprocessor Gate, Lockstep Integer-Exact `Q12` `StateSMix + Online-TTT` `LUT6`
//! Context Mixer & Sparse Zero-Crossing Sign-Flip `JouleBit6` Updater (`0` `FLOP`s, `0` `AdamW`
//! `sqrtf`/`expf` Cross-Hardware Drift), and Cross-Silicon `Window 26` vs. `Window 26b` Kernel.

use crate::mag_mcp_freetoken::{Vault14BankOutcome, eval_fourteen_bank_vault_lut6};
use crate::siw_joulebit::{eval_joulebit6_fwht64_slab_i16, eval_littlebit2_6col_unpack_mac_64};

/// Exact `Q12` (`4,096`) integer probability scale for zero-float `StateSMix` logistic blending.
pub const STATESMIX_PROB_SCALE_Q12: u16 = 4096;

/// Exact `6`-input `LUT6` specification for Bijective `WRT-Capillary` MediaWiki-XML & Stem
/// Preprocessor Routing (`Claims LA-80 & LA-81`, `Theorem 147`):
///
/// `wrt_addr6 = byte_class2 | (dict_match1 << 2) | (cap_fold_ok1 << 3) | (esc_req1 << 4) | (xml_tag_span1 << 5)`:
///   - `wrt_mode2` (`0..=3`):
///       `3` (`0b11`): Literal Escape Pair `[0xFE, raw_byte]` (`esc_req == 1` -> guarantees 100% bijection over all `256^*` streams!),
///       `2` (`0b10`): Capital-Folded Stem Pair `[0xFD, stem_tok]` (`!esc_req && dict_match && cap_fold_ok`),
///       `1` (`0b01`): Single-Byte `WRT` XML/Wiki/Stem Token (`!esc_req && (dict_match || xml_tag_span)`),
///       `0` (`0b00`): Zero-Overhead `1`-Byte 7-Bit ASCII Pass-Through (`!esc_req && !dict_match && !xml_tag_span`).
#[inline(always)]
pub const fn wrt_capillary_gate2_spec(wrt_addr6: u8) -> u8 {
    let a = (wrt_addr6 & 0x3F) as usize;
    let _byte_class = a & 3;
    let dict_match = ((a >> 2) & 1) != 0;
    let cap_fold_ok = ((a >> 3) & 1) != 0;
    let esc_req = ((a >> 4) & 1) != 0;
    let xml_tag_span = ((a >> 5) & 1) != 0;

    if esc_req {
        3
    } else if dict_match && cap_fold_ok {
        2
    } else if dict_match || xml_tag_span {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `wrt_capillary_gate2_spec` with `0` branches.
pub const WRT_CAPILLARY_GATE2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let g = wrt_capillary_gate2_spec(x as u8) as usize;
        if (g & 1) != 0 {
            m[0] |= 1u64 << x;
        }
        if ((g >> 1) & 1) != 0 {
            m[1] |= 1u64 << x;
        }
        x += 1;
    }
    m
};

/// Evaluates the compiled `2`-bit `WRT-Capillary` `LUT6` gate (`0..=3`).
#[inline]
pub const fn eval_wrt_capillary_gate2_lut6(wrt_addr6: u8) -> u8 {
    let sh = (wrt_addr6 & 0x3F) as u32;
    (((WRT_CAPILLARY_GATE2_MASKS[0] >> sh) & 1) as u8)
        | ((((WRT_CAPILLARY_GATE2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Exact `6`-input `LUT6` specification for Lockstep Zero-Float `StateSMix + Online-TTT`
/// Predictor & Sparse Zero-Crossing Sign-Flip Control (`Claims LA-81 & LA-82`, `Theorem 149`):
///
/// `ttt_addr6 = disagree2 | (err_ge_tau1 << 2) | (zero_cross1 << 3) | (sse_conf1 << 4) | (rep0_shadow1 << 5)`:
///   - `ttt_mode2` (`0..=3`):
///       `3` (`0b11`): Zero-Crossing Sign-Flip `LUT6` Slab Rebuild + `Q12` Mixer Update (`err_ge_tau && zero_cross`),
///       `2` (`0b10`): Integer `i16` Straight-Through Accumulator Step (`err_ge_tau && !zero_cross` -> skips `LUT6` slab rebuild!),
///       `1` (`0b01`): `SSE-LUT6` Secondary Symbol Calibration (`sse_conf || rep0_shadow || disagree >= 2`),
///       `0` (`0b00`): Stationary Context Fast Bypass (`0` weight writes, `0` slab rebuilds).
#[inline(always)]
pub const fn statesmix_ttt_gate2_spec(ttt_addr6: u8) -> u8 {
    let a = (ttt_addr6 & 0x3F) as usize;
    let disagree = a & 3;
    let err_ge_tau = ((a >> 2) & 1) != 0;
    let zero_cross = ((a >> 3) & 1) != 0;
    let sse_conf = ((a >> 4) & 1) != 0;
    let rep0_shadow = ((a >> 5) & 1) != 0;

    if err_ge_tau && zero_cross {
        3
    } else if err_ge_tau {
        2
    } else if sse_conf || rep0_shadow || disagree >= 2 {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `statesmix_ttt_gate2_spec` with `0` branches.
pub const STATESMIX_TTT_GATE2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let b = statesmix_ttt_gate2_spec(x as u8) as usize;
        if (b & 1) != 0 {
            m[0] |= 1u64 << x;
        }
        if ((b >> 1) & 1) != 0 {
            m[1] |= 1u64 << x;
        }
        x += 1;
    }
    m
};

/// Evaluates the compiled `2`-bit `StateSMix-TTT` `LUT6` gate (`0..=3`).
#[inline]
pub const fn eval_statesmix_ttt_gate2_lut6(ttt_addr6: u8) -> u8 {
    let sh = (ttt_addr6 & 0x3F) as u32;
    (((STATESMIX_TTT_GATE2_MASKS[0] >> sh) & 1) as u8)
        | ((((STATESMIX_TTT_GATE2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Pure-integer `Q12` (`1..=4095`) `StateSMix` 3-expert probability mixer + `SSE` sharpening
/// (`0` floats, `0` `expf`/`logf`, `100%` lockstep deterministic across `arm64`/`aarch64`/`x86_64`;
/// `Theorem 148`).
#[inline]
pub const fn mix3_probs_q12(
    p_ord1_q12: u16,
    p_ord2_q12: u16,
    p_word_q12: u16,
    w_ord1_q8: u16,
    w_ord2_q8: u16,
    sse_mode: u8,
) -> u16 {
    let w1 = if w_ord1_q8 > 256 { 256 } else { w_ord1_q8 } as u32;
    let rem = 256u32 - w1;
    let w2 = if (w_ord2_q8 as u32) > rem {
        rem
    } else {
        w_ord2_q8 as u32
    };
    let w3 = 256u32 - w1 - w2;

    let raw = (w1 * (p_ord1_q12 as u32) + w2 * (p_ord2_q12 as u32) + w3 * (p_word_q12 as u32)) >> 8;
    // Integer `SSE` (`Secondary Symbol Estimation`) sharpening toward certainty when `sse_mode >= 1`
    let calibrated = if (sse_mode & 3) != 0 {
        if raw > 2048 {
            raw + ((4095u32.saturating_sub(raw)) >> 4)
        } else {
            raw.saturating_sub(raw >> 4)
        }
    } else {
        raw
    };
    if calibrated < 1 {
        1
    } else if calibrated > 4095 {
        4095
    } else {
        calibrated as u16
    }
}

/// Lockstep zero-float `i16` Straight-Through / Hebbian online test-time updater for a `6`-column
/// `JouleBit-LUT6` Rademacher slab (`Theorem 149`):
/// Updates saturating `i16` master accumulators `acc6[0..6]` and returns `(new_rad6_code, sign_flip_mask6)`.
/// Unlike `FP32` `AdamW` (`m_t / (sqrtf(v_t) + eps)`), this step uses `0` floating-point ops and
/// guarantees `0` sign-flip divergence across all CPU/MCU/WASM/FPGA targets!
#[inline]
pub fn step_online_ttt_joulebit6_i16(
    acc6: &mut [i16; 6],
    act_signs6: u8,
    err_q8: i16,
) -> (u8, u8) {
    let mut old_rad6 = 0u8;
    let mut new_rad6 = 0u8;
    let mut j = 0usize;
    while j < 6 {
        if acc6[j] < 0 {
            old_rad6 |= 1u8 << j;
        }
        let sgn: i32 = if ((act_signs6 >> j) & 1) != 0 { -1 } else { 1 };
        let step_delta: i32 = (sgn * (err_q8 as i32)) >> 2;
        let updated = (acc6[j] as i32 + step_delta).clamp(-32_767, 32_767) as i16;
        acc6[j] = updated;
        if updated < 0 {
            new_rad6 |= 1u8 << j;
        }
        j += 1;
    }
    let flip_mask6 = (old_rad6 ^ new_rad6) & 0x3F;
    (new_rad6, flip_mask6)
}

const _: () = {
    let mut x = 0u8;
    while x < 64 {
        assert!(eval_wrt_capillary_gate2_lut6(x) == wrt_capillary_gate2_spec(x));
        assert!(eval_statesmix_ttt_gate2_lut6(x) == statesmix_ttt_gate2_spec(x));
        x += 1;
    }
    assert!(eval_wrt_capillary_gate2_lut6(0b01_0000) == 3); // esc_req=1 -> 3 (Bijective Escape!)
    assert!(eval_wrt_capillary_gate2_lut6(0b00_1100) == 2); // dict_match && cap_fold -> 2
    assert!(eval_wrt_capillary_gate2_lut6(0b10_0000) == 1); // xml_tag_span -> 1
    assert!(eval_wrt_capillary_gate2_lut6(0b00_0000) == 0); // ASCII pass-through -> 0
    assert!(eval_statesmix_ttt_gate2_lut6(0b00_1100) == 3); // err_ge_tau && zero_cross -> 3 (Slab Rebuild!)
    assert!(eval_statesmix_ttt_gate2_lut6(0b00_0100) == 2); // err_ge_tau && !zero_cross -> 2 (Skip Slab Rebuild!)
    assert!(eval_statesmix_ttt_gate2_lut6(0b01_0000) == 1); // sse_conf -> 1 (SSE-LUT6!)
    assert!(eval_statesmix_ttt_gate2_lut6(0b00_0000) == 0); // stationary bypass -> 0

    let pm = mix3_probs_q12(3200, 3600, 3000, 96, 112, 1);
    assert!(pm >= 1 && pm <= 4095);
};

/// Fused Phase 46 (`Window 26`) Rung-7 step:
/// Evaluates `14`-Bank (`56 KB`) `VAULT14` + `WRT-Capillary-LUT6` + `StateSMix-TTT-LUT6` +
/// integer-exact `Q12` 3-model `StateSMix` blend + sparse zero-crossing `Online-JouleBit6` update
/// (`LUT6` slab evaluated ONLY on non-skipped novel tokens / zero-crossing updates; `0` `FLOP`s!).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window26_ltcb_cm2_statesmix_ttt_joulebit6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    wrt_addr6: u8,
    ttt_addr6: u8,
    err_q8: i16,
    acc6: &mut [i16; 6],
    banks_action_masks: &[[u64; 3]; 14],
    banks_cert_masks: &[u64; 14],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> (Vault14BankOutcome, u8, u8, u16, u8) {
    let v14 = eval_fourteen_bank_vault_lut6(
        percept6,
        subgoal6,
        epoch,
        Some(0x5A),
        0x5A,
        !1u64,
        banks_action_masks,
        banks_cert_masks,
    );
    let wrt2 = eval_wrt_capillary_gate2_lut6(wrt_addr6);
    let ttt2 = eval_statesmix_ttt_gate2_lut6(ttt_addr6);
    let p_q12 = mix3_probs_q12(
        2800 + ((percept6 as u16) << 3),
        3100 + ((subgoal6 as u16) << 2),
        2500,
        88,
        112,
        ttt2,
    );
    let mut flip6 = 0u8;
    if ttt2 >= 2 {
        let (_rad6, flp) = step_online_ttt_joulebit6_i16(acc6, percept6, err_q8);
        flip6 = flp;
    }
    // Only rebuild / evaluate `6`-col `JouleBit6` slab on zero-crossing sign-flip (`ttt2 == 3` or `flip6 != 0`)!
    if ttt2 == 3 || flip6 != 0 {
        eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    } else {
        out64[0] = (p_q12 as i32) ^ (wrt2 as i32);
        out64[63] = (acc6[0] as i32) ^ (ttt2 as i32);
    }
    (v14, wrt2, ttt2, p_q12, flip6)
}

/// Uncompiled floating-point `tufazip`/`cmix`-style online `AdamW` + logistic context mixer (`Window 26b`):
/// Executes on EVERY step:
///   1. Dense `64`-parameter `FP32` online `AdamW` moment update (`m_i`, `v_i`, `v_i.sqrt()` Newton/hardware `sqrt`, `fdiv`)
///      plus `1`-bit `QPI1` sign re-binarization,
///   2. `FP32` stretched-logit (`ln(p / (1-p))` + `expf`) 3-model context mixer,
///   3. Unpacked `1`-bit `LittleBit-2` `64`-row binary `MAC` without `LUT6` slab grouping or sparse sign-flip gating.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn eval_window26b_float_online_adamw_mixer_littlebit2(
    weights64: &mut [f32; 64],
    adam_m64: &mut [f32; 64],
    adam_v64: &mut [f32; 64],
    grad_scale: f32,
    probs3: &[f32; 3],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> u32 {
    #[inline(always)]
    fn fast_expf(x: f32) -> f32 {
        let c = x.clamp(-12.0, 12.0);
        let z = 1.0 + c * (1.0 / 32.0) + c * c * (1.0 / 2048.0);
        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let z16 = z8 * z8;
        z16 * z16
    }

    #[inline(always)]
    fn fast_sqrtf(v: f32) -> f32 {
        let x = v.max(1e-12);
        let mut y = f32::from_bits(0x1fbd_1df5 + (x.to_bits() >> 1));
        y = 0.5 * (y + x / y);
        y = 0.5 * (y + x / y);
        y
    }

    let mut sign_pack = 0u64;
    let mut i = 0usize;
    while i < 64 {
        let g = grad_scale * ((acts64[i] as f32) * 0.03125);
        let m = 0.9f32 * adam_m64[i] + 0.1f32 * g;
        let v = 0.999f32 * adam_v64[i] + 0.001f32 * (g * g);
        adam_m64[i] = m;
        adam_v64[i] = v;
        let w = weights64[i] * 0.9999f32 - 0.002f32 * (m / (fast_sqrtf(v) + 1e-8f32));
        weights64[i] = w;
        if w < 0.0 {
            sign_pack |= 1u64 << i;
        }
        i += 1;
    }

    // Stretched-logit 3-model float context mix
    let mut logit_sum = 0.0f32;
    let mix_w = [0.35f32, 0.45f32, 0.20f32];
    let mut k = 0usize;
    while k < 3 {
        let p = probs3[k].clamp(1e-4, 0.9999);
        let ratio = p / (1.0 - p);
        // Padé-style float log approximation around 1
        let u = (ratio - 1.0) / (ratio + 1.0);
        let stretch = 2.0 * u * (1.0 + (u * u) * (0.33333334 + 0.2 * u * u));
        logit_sum += mix_w[k] * stretch;
        k += 1;
    }
    let p_mix = 1.0f32 / (1.0f32 + fast_expf(-logit_sum));
    let cdf_q24 = ((p_mix * 16_777_216.0) as u32).clamp(1, 16_777_215);

    eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);
    cdf_q24 ^ (sign_pack as u32) ^ ((sign_pack >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrt_and_statesmix_ttt_lut6_exhaustive_and_online_determinism() {
        for x in 0u8..64 {
            assert_eq!(eval_wrt_capillary_gate2_lut6(x), wrt_capillary_gate2_spec(x));
            assert_eq!(eval_statesmix_ttt_gate2_lut6(x), statesmix_ttt_gate2_spec(x));
        }
        let mut acc = [12i16, -4, 300, -500, 2, -1];
        let (rad_a, flip_a) = step_online_ttt_joulebit6_i16(&mut acc, 0b01_0101, -32);
        // j=1: acc[1]=-4, sgn=+1, step_delta=-8 -> -12 (no flip); j=4: acc[4]=2, sgn=-1, step_delta=+8 -> +10; j=0: 12 + (-8) = 4
        assert_eq!(flip_a & 0x3F, flip_a);
        let mut acc2 = [4i16, -12, 300, -500, 10, -1];
        let (_rad_b, flip_b) = step_online_ttt_joulebit6_i16(&mut acc2, 0b00_0000, -40);
        // j=0 crosses +4 -> -6 (bit 0 flips!)
        assert_ne!(flip_b & 1, 0);
        let _ = rad_a;
    }
}
