//! Phase 43 (`Claims LA-68..LA-71`): `#![no_std]` Three-Tier `GPU VRAM (Tier 0) -> System RAM
//! (Tier 1) -> Uncached NVMe SSD (Tier 2)` Predictive `NameLedger` + `Reuse::expected_with`
//! Admission `LUT6` Controller (`Tiered-Expert-AirMoL`), `DeepSeek-V4.1-Flash` Zero-`FLOP`
//! Trailing $n$-Gram Conditional Memory (`Engram-LUT6`), and `Window 23` Cross-Silicon Kernel.

use crate::cbfws_term_math::{Cbfws12BankOutcome, eval_cbfws_twelve_bank_lut6};
use crate::siw_joulebit::{eval_joulebit6_fwht64_slab_i16, eval_littlebit2_6col_unpack_mac_64};

/// Exact `6`-input `LUT6` specification for Three-Tier `VRAM -> RAM -> NVMe` expert residency &
/// expected-reuse admission (`Claim LA-69`, `Theorems 123-124`):
/// `tier_addr6 = residency2 | (incoming_gt_victim1 << 2) | (lookahead_dist2 << 3) | (route_conf1 << 5)`
///   - `residency2` (`bits [1..0]`):
///       `0` = `Tier 0` (`GPU VRAM` resident),
///       `1` = `Tier 1` (`System RAM` warm cache),
///       `2` = `Tier 2` (`Uncached NVMe SSD`, free `RAM` landing slot available),
///       `3` = `Tier 2` (`Uncached NVMe SSD`, `RAM` landing pool full -> `Reuse` check required).
///   - `incoming_gt_victim1` (`bit 2`): `1` iff `Reuse::expected_with(incoming) > Reuse::expected_with(victim)`.
///   - `lookahead_dist2` (`bits [4..3]`): `0` = demand step `k=0`, `1..=3` = `K`-token `LUT6` lookahead.
///   - `route_conf1` (`bit 5`): `1` iff conformal `LUT6` / `JouleBit-LUT6` route lookahead gate passes.
///
/// Output `action2` (`0..=3`):
///   - `3` (`0b11`): `Tier 0` `VRAM` hit (`0` `RAM`/`NVMe` movement),
///   - `2` (`0b10`): `LinkMover` `RAM -> VRAM` async promotion (`0` `NVMe` reads),
///   - `1` (`0b01`): Admit `NVMe -> RAM` read-ahead into landing pool (`slot_free || incoming > victim`),
///   - `0` (`0b00`): Thrash-guarded suppress / stream-bypass (never evicts higher-expected-reuse warm expert!).
#[inline(always)]
pub const fn tiered_residency_action2_spec(tier_addr6: u8) -> u8 {
    let a = (tier_addr6 & 0x3F) as usize;
    let residency = a & 3;
    let incoming_gt_victim = ((a >> 2) & 1) != 0;
    let lookahead_dist = (a >> 3) & 3;
    let route_conf = ((a >> 5) & 1) != 0;

    if residency == 0 {
        3
    } else if lookahead_dist > 0 && !route_conf {
        0
    } else if residency == 1 {
        2
    } else if residency == 2 || incoming_gt_victim {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `tiered_residency_action2_spec` with `0` branches.
pub const TIERED_RESIDENCY2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let act = tiered_residency_action2_spec(x as u8) as usize;
        if (act & 1) != 0 {
            m[0] |= 1u64 << x;
        }
        if ((act >> 1) & 1) != 0 {
            m[1] |= 1u64 << x;
        }
        x += 1;
    }
    m
};

/// Evaluates the compiled `2`-bit Three-Tier `NameLedger + Reuse` residency `LUT6` controller (`0..=3`).
#[inline]
pub const fn eval_tiered_residency_lut6(tier_addr6: u8) -> u8 {
    let sh = (tier_addr6 & 0x3F) as u32;
    (((TIERED_RESIDENCY2_MASKS[0] >> sh) & 1) as u8)
        | ((((TIERED_RESIDENCY2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Exact `6`-input `LUT6` specification for `DeepSeek-V4.1-Flash` trailing $n$-gram conditional
/// memory (`ple.layers` `Engram`, `Claim LA-70`, `Theorem 127`):
/// Replaces `RMS(q) * RMS(k)`, `signed_sqrt(gate)`, and `1 / (1 + exp(-x))` float operations on
/// recurring compressed token $n$-grams (`engram.token_map`) with a zero-`FLOP`, zero-`MUL` `LUT6` gate!
///
/// `engram_addr6 = ngram_hit1 | (qk_sign1 << 1) | (qk_mag2 << 2) | (rms_valid1 << 4) | (sram_cached1 << 5)`:
///   - `gate2 == 0` (`0b00`): Gate closed / bypass (`0` residual added),
///   - `gate2 == 1` (`0b01`): Mild negative alignment $\sigma(-\sqrt{|g|}) \approx 1/4$ (`v >> 2`),
///   - `gate2 == 2` (`0b10`): Neutral alignment $\sigma(0) = 1/2$ (`v >> 1`),
///   - `gate2 == 3` (`0b11`): Concave positive alignment $\sigma(+\sqrt{g}) \approx 7/8$ (`v - (v >> 3)`).
#[inline(always)]
pub const fn engram_gate2_spec(engram_addr6: u8) -> u8 {
    let a = (engram_addr6 & 0x3F) as usize;
    let ngram_hit = (a & 1) != 0;
    let qk_pos = ((a >> 1) & 1) != 0;
    let qk_mag = (a >> 2) & 3;
    let rms_valid = ((a >> 4) & 1) != 0;
    let sram_cached = ((a >> 5) & 1) != 0;

    if !ngram_hit || !rms_valid || !sram_cached {
        0
    } else if !qk_pos {
        if qk_mag >= 2 { 0 } else { 1 }
    } else if qk_mag == 0 {
        2
    } else {
        3
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `engram_gate2_spec` with `0` branches and `0` `FLOP`s.
pub const ENGRAM_GATE2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let g = engram_gate2_spec(x as u8) as usize;
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

/// Evaluates the compiled `2`-bit `Engram-LUT6` conditional-memory gate (`0..=3`, `0` `FLOP`s).
#[inline]
pub const fn eval_engram_ngram_gate2_lut6(engram_addr6: u8) -> u8 {
    let sh = (engram_addr6 & 0x3F) as u32;
    (((ENGRAM_GATE2_MASKS[0] >> sh) & 1) as u8)
        | ((((ENGRAM_GATE2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Applies the zero-`MUL`, zero-`FLOP` shift-add `Engram-LUT6` conditional memory value residual
/// across a `16`-element `i16` slice, returning the parity digest of updated hidden states.
#[inline]
pub const fn apply_engram_lut6_residual16(gate2: u8, values16: &[i16; 16], accum16: &mut [i32; 16]) -> i32 {
    let mut xor_acc = 0i32;
    let mut i = 0usize;
    while i < 16 {
        let v = values16[i] as i32;
        let scaled = match gate2 & 3 {
            0 => 0,
            1 => v >> 2,
            2 => v >> 1,
            _ => v - (v >> 3),
        };
        let next = accum16[i].wrapping_add(scaled);
        accum16[i] = next;
        xor_acc ^= next;
        i += 1;
    }
    xor_acc
}

/// Compile-time exhaustive `64/64` minterm proof gate (`rustc` `const` evaluation).
const _: () = {
    let mut x = 0u8;
    while x < 64 {
        assert!(eval_tiered_residency_lut6(x) == tiered_residency_action2_spec(x));
        assert!(eval_engram_ngram_gate2_lut6(x) == engram_gate2_spec(x));
        x += 1;
    }
    // Spot-check key architectural invariants:
    // 1. VRAM hit (`residency=0`) -> `3` regardless of lookahead bits
    assert!(eval_tiered_residency_lut6(0b00_0000) == 3);
    // 2. Warm RAM + confident 2-token lookahead (`residency=1, k=2, conf=1`) -> `2` (`LinkMover`)
    assert!(eval_tiered_residency_lut6(0b11_0001) == 2);
    // 3. Full RAM pool + lower incoming reuse (`residency=3, incoming_gt_victim=0, conf=1`) -> `0` (no thrash!)
    assert!(eval_tiered_residency_lut6(0b10_1011) == 0);
    // 4. Full RAM pool + strictly higher incoming reuse (`residency=3, incoming_gt_victim=1, conf=1`) -> `1` (admit!)
    assert!(eval_tiered_residency_lut6(0b10_1111) == 1);
    // 5. Engram cached positive strong alignment (`hit=1, pos=1, mag=3, rms=1, sram=1` = `0b11_1111 = 63`) -> `3`
    assert!(eval_engram_ngram_gate2_lut6(63) == 3);
    // 6. Engram strongly negative alignment (`hit=1, pos=0, mag=2, rms=1, sram=1` = `0b11_1001 = 57`) -> `0`
    assert!(eval_engram_ngram_gate2_lut6(57) == 0);
};

/// Fused Phase 43 (`Window 23`) Rung-7 step:
/// Evaluates `12`-Bank `C-BFWS` `System-0` reflex + Three-Tier `NameLedger + Reuse` `LUT6` admission
/// + `DeepSeek-V4.1-Flash` `Engram-LUT6` zero-`FLOP` shift-add residual + `JouleBit6 + FWHT-64` slab
/// (`0` `FLOP`s, `0` `MUL`s, `0` `DSP48E1`s)!
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window23_tiered_engram_joulebit6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    tier_addr6: u8,
    engram_addr6: u8,
    engram_vals16: &[i16; 16],
    banks_action_masks: &[[u64; 3]; 12],
    banks_cert_masks: &[u64; 12],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> (Cbfws12BankOutcome, u8, u8, i32) {
    let cbfws = eval_cbfws_twelve_bank_lut6(
        percept6,
        subgoal6,
        epoch,
        Some(0x5A),
        0x5A,
        !1u64,
        banks_action_masks,
        banks_cert_masks,
    );
    let tier2 = eval_tiered_residency_lut6(tier_addr6);
    let eng_gate2 = eval_engram_ngram_gate2_lut6(engram_addr6);
    let mut eng_acc = [0i32; 16];
    let eng_digest = apply_engram_lut6_residual16(eng_gate2, engram_vals16, &mut eng_acc);
    eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    (cbfws, tier2, eng_gate2, eng_digest)
}

/// `#![no_std]` Newton-Raphson `sqrt(x)` (4 iterations) for the uncompiled `Window 23b` `f32` `Engram` baseline.
#[inline(always)]
fn nostd_sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = f32::from_bits((x.to_bits() >> 1) + 0x1FC0_0000);
    let mut i = 0usize;
    while i < 4 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// `#![no_std]` Padé `[2/2]` `sigmoid(x) = 1 / (1 + exp(-x))` for the uncompiled `Window 23b` `f32` `Engram` baseline.
#[inline(always)]
fn nostd_sigmoid_f32(x: f32) -> f32 {
    let xc = x.clamp(-8.0, 8.0);
    // exp(-xc) via (1 - u/2 + u^2/12) / (1 + u/2 + u^2/12) with 2x halving (`u = xc / 4`)
    let u = xc * 0.25;
    let u2 = u * u;
    let num = 1.0 - 0.5 * u + (1.0 / 12.0) * u2;
    let den = 1.0 + 0.5 * u + (1.0 / 12.0) * u2;
    let e_neg_quarter = num / den;
    let e_neg_half = e_neg_quarter * e_neg_quarter;
    let e_neg = e_neg_half * e_neg_half;
    1.0 / (1.0 + e_neg)
}

/// Uncompiled software baseline (`Window 23b`):
/// Executes:
///   1. Linear `64`-slot expert residency directory scan (`Reuse` victim search loop),
///   2. Uncompiled `f32` `DeepSeek-V4.1-Flash` `Engram` RMS normalization (`q_rms`, `k_rms`), `q · k`
///      dot product, `signed_sqrt(gate)`, and `sigmoid(signed_sqrt(gate)) * value` (`64` head dims),
///   3. Uncompiled `LittleBit-2` `6`-col unpack + binary `MAC`s (`eval_littlebit2_6col_unpack_mac_64`).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn eval_window23b_float_engram_scan_littlebit2_baseline(
    directory_reuse64: &[u32; 64],
    incoming_reuse: u32,
    q64: &[f32; 64],
    k64: &[f32; 64],
    v64: &[f32; 64],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> u8 {
    // 1. Linear 64-slot residency victim scan
    let mut min_reuse = u32::MAX;
    let mut min_slot = 0usize;
    let mut s = 0usize;
    while s < 64 {
        let r = directory_reuse64[s];
        if r < min_reuse {
            min_reuse = r;
            min_slot = s;
        }
        s += 1;
    }
    let admit_flag = if incoming_reuse > min_reuse {
        (min_slot as u32) | 0x40
    } else {
        0
    };

    // 2. Float RMS(q), RMS(k), q·k, signed_sqrt(g), sigmoid, and value scale (`64`-dim Engram head)
    let mut q_sq = 0.0f32;
    let mut k_sq = 0.0f32;
    let mut qk_dot = 0.0f32;
    let mut d = 0usize;
    while d < 64 {
        let qv = q64[d];
        let kv = k64[d];
        q_sq += qv * qv;
        k_sq += kv * kv;
        qk_dot += qv * kv;
        d += 1;
    }
    let denom = nostd_sqrt_f32((q_sq * 0.015625) * (k_sq * 0.015625)) + 1e-6;
    let raw_g = (qk_dot * 0.015625) / denom;
    let signed_sqrt_g = if raw_g >= 0.0 {
        nostd_sqrt_f32(raw_g)
    } else {
        -nostd_sqrt_f32(-raw_g)
    };
    let gate_prob = nostd_sigmoid_f32(signed_sqrt_g);
    let mut val_bits = 0u32;
    d = 0;
    while d < 64 {
        let out_v = gate_prob * v64[d];
        val_bits ^= out_v.to_bits();
        d += 1;
    }

    // 3. Uncompiled LittleBit-2 6-col + FWHT-64 unpack + binary MAC
    eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);

    ((admit_flag ^ val_bits ^ (val_bits >> 16)) & 0x3F) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiered_residency_and_engram_lut6_exhaustive_parity() {
        for addr in 0..64u8 {
            assert_eq!(
                eval_tiered_residency_lut6(addr),
                tiered_residency_action2_spec(addr)
            );
            assert_eq!(eval_engram_ngram_gate2_lut6(addr), engram_gate2_spec(addr));
        }

        // VRAM hit -> 3
        assert_eq!(eval_tiered_residency_lut6(0), 3);
        // Unconfident speculative lookahead (`k=1, conf=0`) on cold NVMe -> 0 (suppressed!)
        assert_eq!(eval_tiered_residency_lut6(0b00_1010), 0);
        // Confident speculative lookahead (`k=2, conf=1`) on free RAM slot (`residency=2`) -> 1 (admit!)
        assert_eq!(eval_tiered_residency_lut6(0b11_0010), 1);

        let vals = [64i16; 16];
        let mut acc = [0i32; 16];
        let _ = apply_engram_lut6_residual16(3, &vals, &mut acc);
        assert_eq!(acc[0], 56); // 64 - (64 >> 3) = 56 (exact 7/8ths concave gain, 0 MULs!)
    }
}
