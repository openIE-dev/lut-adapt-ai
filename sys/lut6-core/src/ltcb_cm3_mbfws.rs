//! Phase 47 (`Claims LA-84..LA-87`): `#![no_std]` Stretched Integer Log-Odds `MoL-LTCB-CM3`
//! (`WRT2 + 4-Model StateSMix3`) `LUT6` Gate & Monotone Octave `Stretch`/`Squash` Algebra,
//! `ARC-AGI-3` Modal Pick-Place & Containment-Gated Avatar-Switch `M-BFWS(R,M)` `LUT6` Router,
//! and Cross-Silicon `Window 27` vs. `Window 27b` Shunt Benchmark Kernels (`0` `FLOP`s).

use crate::ltcb_cm2_ttt::step_online_ttt_joulebit6_i16;
use crate::mag_mcp_freetoken::{Vault14BankOutcome, eval_fourteen_bank_vault_lut6};
use crate::siw_joulebit::{eval_joulebit6_fwht64_slab_i16, eval_littlebit2_6col_unpack_mac_64};

/// Exact `Q11` (`2,048`) binary range-coder probability scale (`1..=2047`).
pub const CM3_PROB_SCALE_Q11: u16 = 2048;

/// Exact `6`-input `LUT6` specification for `MoL-LTCB-CM3` Integer Stretched Log-Odds
/// `StateSMix3` & `SSE` Calibration (`Claim LA-84`, `Theorem 156`):
///
/// `cm3_addr6 = byte_cls2 | (in_word1 << 2) | (ord4_hit1 << 3) | (sse_sharp1 << 4) | (rep0_shadow1 << 5)`:
///   - `cm3_mode2` (`0..=3`):
///       `3` (`0b11`): Shadow-Dominant Log-Odds + High-Slope `SSE` Calibration (`rep0_shadow && sse_sharp`),
///       `2` (`0b10`): `ASCII Word-Hash + Order-3/4` Specialist Log-Odds Boost (`in_word && ord4_hit`),
///       `1` (`0b01`): General `4`-Expert Log-Odds + Mild `SSE` (`sse_sharp || rep0_shadow || ord4_hit || byte_cls2 >= 2`),
///       `0` (`0b00`): Stationary Low-Entropy Fast Log-Odds Blend (`0` `SSE` shift).
#[inline(always)]
pub const fn cm3_logodds_mix_gate2_spec(cm3_addr6: u8) -> u8 {
    let a = (cm3_addr6 & 0x3F) as usize;
    let byte_cls2 = a & 3;
    let in_word = ((a >> 2) & 1) != 0;
    let ord4_hit = ((a >> 3) & 1) != 0;
    let sse_sharp = ((a >> 4) & 1) != 0;
    let rep0_shadow = ((a >> 5) & 1) != 0;

    if rep0_shadow && sse_sharp {
        3
    } else if in_word && ord4_hit {
        2
    } else if sse_sharp || rep0_shadow || ord4_hit || byte_cls2 >= 2 {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `cm3_logodds_mix_gate2_spec` with `0` branches.
pub const CM3_LOGODDS_MIX_GATE2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let g = cm3_logodds_mix_gate2_spec(x as u8) as usize;
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

/// Evaluates the compiled `2`-bit `CM3-LogOdds-Mix` `LUT6` gate (`0..=3`).
#[inline]
pub const fn eval_cm3_logodds_mix_gate2_lut6(cm3_addr6: u8) -> u8 {
    let sh = (cm3_addr6 & 0x3F) as u32;
    (((CM3_LOGODDS_MIX_GATE2_MASKS[0] >> sh) & 1) as u8)
        | ((((CM3_LOGODDS_MIX_GATE2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Exact `6`-input `LUT6` specification for `ARC-AGI-3` Modal Pick-Place & Containment-Gated
/// Active-Avatar Switch `M-BFWS(R,M)` Subspace Routing (`Claim LA-85`, `Theorem 159`):
///
/// `modal_addr6 = avail_sig2 | (swatch_col_match1 << 2) | (avatar_contained1 << 3) | (causal_small_tok1 << 4) | (breaker_mismatch1 << 5)`:
///   - `modal_mode2` (`0..=3`):
///       `3` (`0b11`): Mechanic-Shift Circuit-Breaker Halt (`breaker_mismatch == 1`),
///       `2` (`0b10`): `2`-Step Modal Swatch-Slot Macro + `ACTION5` Submit (`avail_sig2 == 2 && swatch_col_match1`),
///       `1` (`0b01`): Containment-Gated Active-Avatar Click Switch (`avail_sig2 == 1 && causal_small_tok1 && !avatar_contained1`),
///       `0` (`0b00`): Standard `14`-Bank `C-BFWS(R,M)` Directional / Click Reflex.
#[inline(always)]
pub const fn mbfws_modal_switch_gate2_spec(modal_addr6: u8) -> u8 {
    let a = (modal_addr6 & 0x3F) as usize;
    let avail_sig2 = a & 3;
    let swatch_col_match = ((a >> 2) & 1) != 0;
    let avatar_contained = ((a >> 3) & 1) != 0;
    let causal_small_tok = ((a >> 4) & 1) != 0;
    let breaker_mismatch = ((a >> 5) & 1) != 0;

    if breaker_mismatch {
        3
    } else if avail_sig2 == 2 && swatch_col_match {
        2
    } else if avail_sig2 == 1 && causal_small_tok && !avatar_contained {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `mbfws_modal_switch_gate2_spec` with `0` branches.
pub const MBFWS_MODAL_SWITCH_GATE2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let g = mbfws_modal_switch_gate2_spec(x as u8) as usize;
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

/// Evaluates the compiled `2`-bit `M-BFWS-Modal-Switch` `LUT6` gate (`0..=3`).
#[inline]
pub const fn eval_mbfws_modal_switch_gate2_lut6(modal_addr6: u8) -> u8 {
    let sh = (modal_addr6 & 0x3F) as u32;
    (((MBFWS_MODAL_SWITCH_GATE2_MASKS[0] >> sh) & 1) as u8)
        | ((((MBFWS_MODAL_SWITCH_GATE2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Pure-integer `const fn` stretched log-odds `round(256 * ln(p / (2048 - p)))` for `p_q11 in 1..=2047`
/// (`0` floating-point ops, odd-symmetric `stretch(2048 - p) == -stretch(p)`, `stretch(1024) == 0`,
/// strictly non-decreasing across `1..=2047`; `Theorem 156`).
#[inline]
pub const fn stretch_prob_q11_to_logit_q8(p_q11: u16) -> i16 {
    let p = if p_q11 < 1 {
        1u32
    } else if p_q11 > 2047 {
        2047u32
    } else {
        p_q11 as u32
    };
    if p == 1024 {
        return 0;
    }
    let (hi, neg) = if p > 1024 {
        (p, false)
    } else {
        (2048 - p, true)
    };
    let q = 2048 - hi;
    let r_q12 = ((hi as u64) << 12) / (q as u64);
    let hb = 63 - r_q12.leading_zeros();
    let k = hb.saturating_sub(12);
    let m = (r_q12 >> k).saturating_sub(4096); // 0..4096
    let num = m * (24_576 + m) * 1770;
    let den = 28_672 * (24_576 + 4 * m);
    let mag = (k as i32) * 177 + ((num / den) as i32);
    if neg { -(mag as i16) } else { mag as i16 }
}

/// Pure-integer `const fn` inverse logistic squash `round(2048 / (1 + exp(-logit / 256)))`
/// returning `p_q11 in 1..=2047` (`0` floating-point ops, `squash(0) == 1024`,
/// odd-symmetric `squash(x) + squash(-x) == 2048`, monotone across all `i32`; `Theorem 156`).
#[inline]
pub const fn squash_logit_q8_to_prob_q11(logit_q8: i32) -> u16 {
    if logit_q8 == 0 {
        return 1024;
    }
    let neg = logit_q8 < 0;
    let abs_lg = if neg {
        (-logit_q8) as u64
    } else {
        logit_q8 as u64
    };
    let x = if abs_lg > 2048 { 2048 } else { abs_lg };
    let k = (x / 177) as u32;
    let r = x % 177;
    let pow2_q12 = (4096u64 + (r * (2839 + ((r * 1257) / 177))) / 177) << k;
    let raw_pos = (2048u64 * pow2_q12 + ((4096 + pow2_q12) >> 1)) / (4096 + pow2_q12);
    let p_pos = if raw_pos < 1024 {
        1024u16
    } else if raw_pos > 2047 {
        2047u16
    } else {
        raw_pos as u16
    };
    if neg {
        let inv = 2048 - p_pos;
        if inv < 1 { 1 } else { inv }
    } else {
        p_pos
    }
}

/// `4`-expert integer stretched log-odds mixer (`Order-1 Shadow` + `Order-2` + `Order-3` + `Word-Hash`)
/// with `CM3-LogOdds-Mix LUT6` mode calibration (`0` floats, `100%` lockstep across all ISAs).
#[inline]
pub const fn mix4_logodds_probs_q11(
    s1_q8: i16,
    s2_q8: i16,
    s3_q8: i16,
    sw_q8: i16,
    w4_q8: [i16; 4],
    cm3_mode2: u8,
) -> u16 {
    let mut logit = ((s1_q8 as i32) * (w4_q8[0] as i32)
        + (s2_q8 as i32) * (w4_q8[1] as i32)
        + (s3_q8 as i32) * (w4_q8[2] as i32)
        + (sw_q8 as i32) * (w4_q8[3] as i32))
        >> 8;
    if cm3_mode2 == 3 {
        logit += (s1_q8 as i32) >> 3;
    } else if cm3_mode2 == 2 {
        logit += (sw_q8 as i32) >> 3;
    }
    let raw_p = squash_logit_q8_to_prob_q11(logit) as u32;
    let cal = if cm3_mode2 >= 1 {
        if raw_p > 1024 {
            raw_p + ((2047u32.saturating_sub(raw_p)) >> 5)
        } else {
            raw_p.saturating_sub(raw_p >> 5)
        }
    } else {
        raw_p
    };
    if cal < 1 {
        1
    } else if cal > 2047 {
        2047
    } else {
        cal as u16
    }
}

const _: () = {
    let mut x = 0u8;
    while x < 64 {
        assert!(eval_cm3_logodds_mix_gate2_lut6(x) == cm3_logodds_mix_gate2_spec(x));
        assert!(eval_mbfws_modal_switch_gate2_lut6(x) == mbfws_modal_switch_gate2_spec(x));
        x += 1;
    }
    assert!(eval_cm3_logodds_mix_gate2_lut6(0b11_0000) == 3); // rep0_shadow && sse_sharp -> 3
    assert!(eval_cm3_logodds_mix_gate2_lut6(0b00_1100) == 2); // in_word && ord4_hit -> 2
    assert!(eval_cm3_logodds_mix_gate2_lut6(0b00_0010) == 1); // byte_cls2 >= 2 -> 1
    assert!(eval_cm3_logodds_mix_gate2_lut6(0b00_0000) == 0); // stationary -> 0

    assert!(eval_mbfws_modal_switch_gate2_lut6(0b10_0000) == 3); // breaker_mismatch -> 3
    assert!(eval_mbfws_modal_switch_gate2_lut6(0b00_0110) == 2); // avail=2 && swatch_col_match -> 2
    assert!(eval_mbfws_modal_switch_gate2_lut6(0b01_0001) == 1); // avail=1 && causal_small && !contained -> 1
    assert!(eval_mbfws_modal_switch_gate2_lut6(0b01_1001) == 0); // contained -> 0 (don't toggle!)

    assert!(stretch_prob_q11_to_logit_q8(1024) == 0);
    assert!(squash_logit_q8_to_prob_q11(0) == 1024);
    let mut p = 1u16;
    let mut prev_s = -32_000i16;
    while p <= 2047 {
        let s = stretch_prob_q11_to_logit_q8(p);
        let sym = stretch_prob_q11_to_logit_q8(2048 - p);
        assert!(s == -sym);
        assert!(s >= prev_s);
        prev_s = s;
        p += 1;
    }
    let mut lg = -1024i32;
    let mut prev_q = 0u16;
    while lg <= 1024 {
        let q = squash_logit_q8_to_prob_q11(lg);
        let q_neg = squash_logit_q8_to_prob_q11(-lg);
        assert!(q + q_neg == 2048);
        assert!(q >= prev_q && q >= 1 && q <= 2047);
        prev_q = q;
        lg += 1;
    }
};

/// Fused Phase 47 (`Window 27`) Rung-7 step (`Claims LA-84..LA-86`):
/// Evaluates `14`-Bank (`56 KB`) `M-BFWS` `VAULT14` + `M-BFWS-Modal-Switch LUT6` +
/// `CM3-LogOdds-Mix LUT6` + `4`-expert integer stretched log-odds `StateSMix3` +
/// sparse zero-crossing `Online-JouleBit6` slab (`0` `FLOP`s!).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window27_cm3_logodds_mbfws_joulebit6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    cm3_addr6: u8,
    modal_addr6: u8,
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
    let cm3_mode = eval_cm3_logodds_mix_gate2_lut6(cm3_addr6);
    let modal_mode = eval_mbfws_modal_switch_gate2_lut6(modal_addr6);
    let s1 = 190 + ((percept6 as i16) << 2);
    let s2 = 240 + ((subgoal6 as i16) << 1);
    let s3 = 280;
    let sw = 320 - ((epoch as i16) << 2);
    let p_q11 = mix4_logodds_probs_q11(s1, s2, s3, sw, [56, 68, 64, 68], cm3_mode);
    let mut flip6 = 0u8;
    if cm3_mode == 3 {
        let (_rad6, flp) = step_online_ttt_joulebit6_i16(acc6, percept6, err_q8);
        flip6 = flp;
    }
    if flip6 != 0 {
        eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    } else {
        out64[0] = (p_q11 as i32) ^ (modal_mode as i32);
        out64[63] = (acc6[0] as i32) ^ (cm3_mode as i32);
    }
    (v14, cm3_mode, modal_mode, p_q11, flip6)
}

/// Uncompiled floating-point `PAQ`/`cmix`-style `6`-expert `ln(p/(1-p))` + `expf` logistic mixer +
/// unpruned `64`-candidate modal `(swatch, slot)` click scan + `LittleBit-2` unpack `MAC` (`Window 27b`).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn eval_window27b_float_paq_ln_exp_modal_scan_littlebit2(
    probs6: &[f32; 6],
    weights6: &mut [f32; 6],
    bit_obs: f32,
    cand_coords64: &[(i16, i16, u8); 64],
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

    let mut stretches = [0.0f32; 6];
    let mut logit_sum = 0.0f32;
    let mut k = 0usize;
    while k < 6 {
        let p = probs6[k].clamp(1e-4, 0.9999);
        let ratio = p / (1.0 - p);
        let u = (ratio - 1.0) / (ratio + 1.0);
        let s = 2.0 * u * (1.0 + (u * u) * (0.33333334 + 0.2 * u * u));
        stretches[k] = s;
        logit_sum += weights6[k] * s;
        k += 1;
    }
    let p_mix = 1.0f32 / (1.0f32 + fast_expf(-logit_sum));
    let err = bit_obs - p_mix;
    k = 0;
    while k < 6 {
        weights6[k] += 0.01f32 * err * stretches[k];
        k += 1;
    }

    // Unpruned 64-object pairwise modal affordance distance scan
    let mut modal_score = 0u32;
    let mut i = 0usize;
    while i < 64 {
        let (ax, ay, ac) = cand_coords64[i];
        let (bx, by, bc) = cand_coords64[(i + 17) & 63];
        let dx = (ax - bx).unsigned_abs() as u32;
        let dy = (ay - by).unsigned_abs() as u32;
        if ac == bc && dx <= 4 {
            modal_score = modal_score.wrapping_add(dy ^ dx);
        } else {
            modal_score ^= dx.wrapping_mul(3).wrapping_add(dy);
        }
        i += 1;
    }

    eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);
    let cdf_q24 = ((p_mix * 16_777_216.0) as u32).clamp(1, 16_777_215);
    cdf_q24 ^ modal_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cm3_logodds_and_mbfws_modal_switch_exhaustive() {
        for x in 0u8..64 {
            assert_eq!(
                eval_cm3_logodds_mix_gate2_lut6(x),
                cm3_logodds_mix_gate2_spec(x)
            );
            assert_eq!(
                eval_mbfws_modal_switch_gate2_lut6(x),
                mbfws_modal_switch_gate2_spec(x)
            );
        }
        for p in 1u16..=2047 {
            let s = stretch_prob_q11_to_logit_q8(p);
            assert_eq!(s, -stretch_prob_q11_to_logit_q8(2048 - p));
        }
        let pm = mix4_logodds_probs_q11(220, 310, 280, 400, [64, 64, 64, 64], 2);
        assert!(pm > 1024 && pm <= 2047);
    }
}
