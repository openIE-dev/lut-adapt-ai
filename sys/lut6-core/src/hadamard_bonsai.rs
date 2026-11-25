//! `#![no_std]` Hadamard-Ternary (`prism-ml/Ternary-Bonsai-2-27B` `block_size = 1024`) +
//! `EntroMCU` (`28 B / 128 trits = 1.750 bpw`) + `ISCA` 2025 `Antisymmetric-40` Zero-`MUL` `T-MAC`
//! and `ARC-AGI-2` 4-Bit Cellular/Color `LUT6` Bit-Sliced Kernels (`Claims LA-39 & LA-41`).

use crate::eval_lut6_bitsliced64;
use crate::tmac_tail::{
    build_tmac_antisym_lut41, lookup_tmac_antisym41, pack_4trits, pack_5trits, unpack_5trits,
};

pub const BONSAI_BLOCK_1024: usize = 1024;
pub const BONSAI_GROUP_128: usize = 128;
pub const BONSAI_GROUPS_PER_1024: usize = BONSAI_BLOCK_1024 / 4; // 256 4-trit T-MAC groups
pub const ENTROMCU_GROUP128_BYTES: usize = 28; // 26 B (128 trits @ 5/byte) + 2 B F16 scale = 1.750 bpw
pub const MLX_AFFINE2_GROUP128_BYTES: usize = 36; // 32 B (U32 2-bit) + 2 B scale + 2 B redundant -scale = 2.250 bpw
pub const BONSAI_VOCAB_SIZE: u32 = 248_320; // < 2^18 = 262,144 => 3 tokens fit in 54 bits + 10-bit header!

/// Losslessly packs `3` `prism-ml/Ternary-Bonsai-2-27B` token IDs (`|V| = 248,320 < 2^18 = 262,144`)
/// plus a `10`-bit conformal/speculation header into a single `64`-bit `LUT6` output word (`54 + 10 = 64`).
#[inline]
pub const fn pack_bonsai_3tok_54bit(
    t0: u32,
    t1: u32,
    t2: u32,
    conf_header10: u16,
) -> Option<u64> {
    if t0 >= BONSAI_VOCAB_SIZE || t1 >= BONSAI_VOCAB_SIZE || t2 >= BONSAI_VOCAB_SIZE {
        return None;
    }
    let word = (t0 as u64)
        | ((t1 as u64) << 18)
        | ((t2 as u64) << 36)
        | (((conf_header10 as u64) & 0x3FF) << 54);
    Some(word)
}

/// Unpacks a `64`-bit `Bonsai-AirMoL` speculative burst word into `([t0, t1, t2], conf_header10)`.
#[inline]
pub const fn unpack_bonsai_3tok_54bit(word: u64) -> ([u32; 3], u16) {
    let mask18 = (1u64 << 18) - 1;
    let t0 = (word & mask18) as u32;
    let t1 = ((word >> 18) & mask18) as u32;
    let t2 = ((word >> 36) & mask18) as u32;
    let hdr10 = ((word >> 54) & 0x3FF) as u16;
    ([t0, t1, t2], hdr10)
}

/// Exact `64`-point Sylvester Fast Walsh–Hadamard Transform (`6` butterfly stages, `0` MULs, `0` floats)
/// with explicit input sign mask `sign_bits` (`bit i = 1 => negate x[i]`) and exact `>> 3` (`1 / sqrt(64) = 1/8`).
#[inline]
pub const fn fwht64_shift3_signed(x: &mut [i32; 64], sign_bits: u64) {
    let mut i = 0usize;
    while i < 64 {
        if ((sign_bits >> i) & 1) != 0 {
            x[i] = -x[i];
        }
        i += 1;
    }
    let mut len = 1usize;
    while len < 64 {
        let step = len << 1;
        let mut base = 0usize;
        while base < 64 {
            let mut j = 0usize;
            while j < len {
                let u = x[base + j];
                let v = x[base + j + len];
                x[base + j] = u + v;
                x[base + j + len] = u - v;
                j += 1;
            }
            base += step;
        }
        len = step;
    }
    i = 0;
    while i < 64 {
        x[i] >>= 3;
        i += 1;
    }
}

/// Exact `1024`-point Sylvester Fast Walsh–Hadamard Transform (`10` butterfly stages = `10,240` integer
/// add/subs, **`0` multiplications, `0` floats**) matching `prism-ml/Ternary-Bonsai-2-27B`'s
/// `normalized-sylvester-walsh-hadamard` (`block_size = 1024 = 4^5`, so `1 / sqrt(1024) = 1 / 32 = >> 5`!).
#[inline]
pub const fn fwht1024_shift5_signed(x: &mut [i32; BONSAI_BLOCK_1024], sign_words: &[u64; 16]) {
    let mut w = 0usize;
    while w < 16 {
        let bits = sign_words[w];
        let base_idx = w << 6;
        let mut b = 0usize;
        while b < 64 {
            if ((bits >> b) & 1) != 0 {
                x[base_idx + b] = -x[base_idx + b];
            }
            b += 1;
        }
        w += 1;
    }

    let mut len = 1usize;
    while len < BONSAI_BLOCK_1024 {
        let step = len << 1;
        let mut base = 0usize;
        while base < BONSAI_BLOCK_1024 {
            let mut j = 0usize;
            while j < len {
                let u = x[base + j];
                let v = x[base + j + len];
                x[base + j] = u + v;
                x[base + j + len] = u - v;
                j += 1;
            }
            base += step;
        }
        len = step;
    }

    let mut idx = 0usize;
    while idx < BONSAI_BLOCK_1024 {
        x[idx] >>= 5;
        idx += 1;
    }
}

/// Losslessly transcodes one `Bonsai-2-27B` MLX `128`-weight affine 2-bit group (`8 x u32` = `32 B` +
/// `2 B` `scale_f16` + `2 B` redundant `bias_f16 == -scale_f16` = `36 B` total, `2.250 bpw`) into
/// canonical `28 B` `EntroMCU` (`26 B` base-3 `5 trits/byte` + `2 B` `scale_f16` = `1.750 bpw`, `-22.22%`).
/// Returns `None` if any 2-bit lane holds the invalid 4th codepoint `c == 3`.
pub const fn transcode_mlx2bit_group128_to_entromcu28(
    words8: &[u32; 8],
    scale_f16_bits: u16,
) -> Option<[u8; ENTROMCU_GROUP128_BYTES]> {
    let mut trits = [0i8; 130];
    let mut w = 0usize;
    while w < 8 {
        let word = words8[w];
        let mut lane = 0usize;
        while lane < 16 {
            let code = ((word >> (lane * 2)) & 0x3) as u8;
            if code > 2 {
                return None;
            }
            trits[w * 16 + lane] = (code as i8) - 1;
            lane += 1;
        }
        w += 1;
    }

    let mut out = [0u8; ENTROMCU_GROUP128_BYTES];
    let mut b = 0usize;
    while b < 26 {
        let base = b * 5;
        out[b] = pack_5trits(
            trits[base],
            trits[base + 1],
            trits[base + 2],
            trits[base + 3],
            trits[base + 4],
        );
        b += 1;
    }
    let sb = scale_f16_bits.to_le_bytes();
    out[26] = sb[0];
    out[27] = sb[1];
    Some(out)
}

/// Unpacks a `28 B` `EntroMCU` group (`1.750 bpw`) back into `128` signed trits `∈ {-1, 0, +1}` + `scale_f16_bits`.
pub const fn unpack_entromcu28_group128(
    packed: &[u8; ENTROMCU_GROUP128_BYTES],
) -> ([i8; BONSAI_GROUP128_LEN], u16) {
    let mut trits = [0i8; BONSAI_GROUP128_LEN];
    let mut b = 0usize;
    while b < 25 {
        let five = unpack_5trits(packed[b]);
        let base = b * 5;
        trits[base] = five[0];
        trits[base + 1] = five[1];
        trits[base + 2] = five[2];
        trits[base + 3] = five[3];
        trits[base + 4] = five[4];
        b += 1;
    }
    let tail = unpack_5trits(packed[25]);
    trits[125] = tail[0];
    trits[126] = tail[1];
    trits[127] = tail[2];
    let scale_bits = u16::from_le_bytes([packed[26], packed[27]]);
    (trits, scale_bits)
}

const BONSAI_GROUP128_LEN: usize = 128;

/// Pre-indexes `16 x 1,024` ternary weights (`∈ {-1, 0, +1}`) into `16 x 256` 4-trit `T-MAC` codes (`0..80`).
pub fn pack_bonsai_block16_codes(
    weights: &[[i8; BONSAI_BLOCK_1024]; 16],
) -> [[u8; BONSAI_GROUPS_PER_1024]; 16] {
    let mut codes = [[40u8; BONSAI_GROUPS_PER_1024]; 16];
    for (r, row) in weights.iter().enumerate() {
        for g in 0..BONSAI_GROUPS_PER_1024 {
            let c0 = g << 2;
            codes[r][g] = pack_4trits(row[c0], row[c0 + 1], row[c0 + 2], row[c0 + 3]);
        }
    }
    codes
}

/// Evaluates a `16 x 1,024` (`16,384`-weight) `Bonsai-2-27B` Hadamard-Ternary block using:
///   1. `fwht1024_shift5_signed` (`0` MULs, `0` floats),
///   2. `256` shared `Antisymmetric-40` `4`-trit `i16` lookup tables (`0` MULs),
///   3. `1` integer group-scale multiply per `128` weights (`8` multiplies/row vs `1,024` = `128x` reduction!).
#[inline(never)]
pub fn eval_bonsai1024_block16_antisym40(
    x_raw: &[i8; BONSAI_BLOCK_1024],
    sign_words: &[u64; 16],
    tmac_codes: &[[u8; BONSAI_GROUPS_PER_1024]; 16],
    group_scales_q8: &[[i16; 8]; 16],
) -> [i32; 16] {
    let mut rot = [0i32; BONSAI_BLOCK_1024];
    for i in 0..BONSAI_BLOCK_1024 {
        rot[i] = (x_raw[i] as i32) << 2; // Q2 headroom so `>> 5` preserves sub-LSB precision
    }
    fwht1024_shift5_signed(&mut rot, sign_words);

    let mut luts = [[0i16; 41]; BONSAI_GROUPS_PER_1024];
    for g in 0..BONSAI_GROUPS_PER_1024 {
        let c0 = g << 2;
        let a0 = rot[c0].clamp(-127, 127) as i8;
        let a1 = rot[c0 + 1].clamp(-127, 127) as i8;
        let a2 = rot[c0 + 2].clamp(-127, 127) as i8;
        let a3 = rot[c0 + 3].clamp(-127, 127) as i8;
        luts[g] = build_tmac_antisym_lut41(a0, a1, a2, a3);
    }

    let mut out = [0i32; 16];
    for r in 0..16 {
        let row_codes = &tmac_codes[r];
        let row_scales = &group_scales_q8[r];
        let mut row_acc = 0i32;
        // 8 groups of 128 weights (`32` 4-trit T-MAC lookups per 128-weight group, `0` inner MULs!)
        for grp128 in 0..8 {
            let g_base = grp128 << 5; // * 32
            let mut sub_sum = 0i32;
            let mut k = 0usize;
            while k < 32 {
                let g0 = g_base + k;
                sub_sum += (lookup_tmac_antisym41(&luts[g0], row_codes[g0]) as i32)
                    + (lookup_tmac_antisym41(&luts[g0 + 1], row_codes[g0 + 1]) as i32)
                    + (lookup_tmac_antisym41(&luts[g0 + 2], row_codes[g0 + 2]) as i32)
                    + (lookup_tmac_antisym41(&luts[g0 + 3], row_codes[g0 + 3]) as i32);
                k += 4;
            }
            // Single scale multiply per 128 weights (128x fewer multiplications than dense!)
            row_acc += sub_sum * (row_scales[grp128] as i32);
        }
        out[r] = row_acc;
    }
    out
}

/// Evaluates an inductive `ARC-AGI-2` 4-bit color + 2-bit spatial/neighbor parity `LUT6` operator
/// across **`64` grid cells in parallel** (`4` bit-sliced `LUT6` evaluations, `0` FLOPs, `0` branches).
#[inline]
pub const fn eval_arc_lut6_strip64(
    c_planes: &[u64; 4],
    spatial_p0: u64,
    spatial_p1: u64,
    out_masks: &[u64; 4],
) -> [u64; 4] {
    [
        eval_lut6_bitsliced64(
            out_masks[0],
            c_planes[0],
            c_planes[1],
            c_planes[2],
            c_planes[3],
            spatial_p0,
            spatial_p1,
        ),
        eval_lut6_bitsliced64(
            out_masks[1],
            c_planes[0],
            c_planes[1],
            c_planes[2],
            c_planes[3],
            spatial_p0,
            spatial_p1,
        ),
        eval_lut6_bitsliced64(
            out_masks[2],
            c_planes[0],
            c_planes[1],
            c_planes[2],
            c_planes[3],
            spatial_p0,
            spatial_p1,
        ),
        eval_lut6_bitsliced64(
            out_masks[3],
            c_planes[0],
            c_planes[1],
            c_planes[2],
            c_planes[3],
            spatial_p0,
            spatial_p1,
        ),
    ]
}

/// Multi-layer (`2`-layer Beneš-routed) `ARC-AGI-2` spatial/relational `LUT6` cascade across `64` cells:
///   - Layer 1 (`2` `LUT6`s): extracts `2` structural predicate bitplanes `(feat0, feat1)` from `(center_nz,
///     up_nz, down_nz, left_nz, right_nz, diag_parity)` (`Cross-5` + parity = `6` wires),
///   - Layer 2 (`4` `LUT6`s): combines `4`-bit cell color `c_planes[0..4]` with `(feat0, feat1)` (`6` wires)
///     into `4` output color bitplanes (`6` `LUT6`s = `48 B` total, `0` FLOPs, radius-`2` / `5x5` reach when cascaded!).
#[inline]
pub const fn eval_benes_arc_lut6_2layer_strip64(
    c_planes: &[u64; 4],
    up_nz: u64,
    down_nz: u64,
    left_nz: u64,
    right_nz: u64,
    diag_parity: u64,
    layer1_feat_masks: &[u64; 2],
    layer2_color_masks: &[u64; 4],
) -> [u64; 4] {
    let center_nz = c_planes[0] | c_planes[1] | c_planes[2] | c_planes[3];
    let feat0 = eval_lut6_bitsliced64(
        layer1_feat_masks[0],
        center_nz,
        up_nz,
        down_nz,
        left_nz,
        right_nz,
        diag_parity,
    );
    let feat1 = eval_lut6_bitsliced64(
        layer1_feat_masks[1],
        center_nz,
        up_nz,
        down_nz,
        left_nz,
        right_nz,
        diag_parity,
    );
    eval_arc_lut6_strip64(c_planes, feat0, feat1, layer2_color_masks)
}

/// Decision of an `ARC-AGI-3` Interactive Reasoning `System-0` `LUT6` reflex step (`Claim LA-44`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arc3ReflexOutcome {
    /// Certified zero-FLOP `GameAction` (`1..=7`) emitted when world-model transition matches & `LUT6` covers state.
    EmitAction { action_id: u8 },
    /// Mechanic-shift circuit breaker: actual frame signature diverged from verified transition prediction!
    MechanicShiftBreaker { expected_sig: u8, actual_sig: u8 },
    /// Uncertified local state: wake online `System-1` transition-verified planner.
    WakeWorldModelPlanner,
}

/// Evaluates an `ARC-AGI-3` `6`-wire local perception/goal-delta `LUT6` policy with an instant
/// online transition-surprise circuit breaker (`Theorem 73`: `0` stale reflex actions on level rule shifts!).
#[inline]
pub const fn eval_arc3_policy_lut6(
    percept6: u8,
    expected_sig: Option<u8>,
    actual_sig: u8,
    action_masks3: &[u64; 3],
    certified_mask: u64,
) -> Arc3ReflexOutcome {
    if let Some(exp) = expected_sig
        && exp != actual_sig
    {
        return Arc3ReflexOutcome::MechanicShiftBreaker {
            expected_sig: exp,
            actual_sig,
        };
    }
    let idx = (percept6 & 0x3F) as u32;
    if ((certified_mask >> idx) & 1) == 0 {
        return Arc3ReflexOutcome::WakeWorldModelPlanner;
    }
    let a0 = ((action_masks3[0] >> idx) & 1) as u8;
    let a1 = (((action_masks3[1] >> idx) & 1) as u8) << 1;
    let a2 = (((action_masks3[2] >> idx) & 1) as u8) << 2;
    let act = a0 | a1 | a2;
    if act == 0 || act > 7 {
        Arc3ReflexOutcome::WakeWorldModelPlanner
    } else {
        Arc3ReflexOutcome::EmitAction { action_id: act }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bonsai_3tok_54bit_bijection_and_arc3_breaker() {
        let w = pack_bonsai_3tok_54bit(0, 124_160, 248_319, 0x2A5).unwrap();
        let (toks, hdr) = unpack_bonsai_3tok_54bit(w);
        assert_eq!(toks, [0, 124_160, 248_319]);
        assert_eq!(hdr, 0x2A5);
        // Out-of-vocab (>= 248,320) must return None
        assert!(pack_bonsai_3tok_54bit(248_320, 1, 2, 0).is_none());

        // 2-Layer Benes-ARC-LUT6: fill background (`c == 0`) with color `2` iff enclosed by `up & down & left & right`
        let mut enclosed_mask = 0u64;
        let mut color_masks = [0u64; 4];
        for m in 0..64usize {
            let c_nz = m & 1;
            let n4 = (m >> 1) & 0xF;
            if c_nz == 0 && n4 == 0xF {
                enclosed_mask |= 1u64 << m;
            }
            let c_in = m & 0xF;
            let f0 = (m >> 4) & 1;
            let c_out = if c_in == 0 && f0 == 1 { 2 } else { c_in };
            for bp in 0..4 {
                if ((c_out >> bp) & 1) != 0 {
                    color_masks[bp] |= 1u64 << m;
                }
            }
        }
        let cp = [0x0u64, 0x0, 0x0, 0x0]; // all background 0
        let out_planes = eval_benes_arc_lut6_2layer_strip64(
            &cp,
            0b1010,
            0b1010,
            0b1010,
            0b1000, // only bit 3 has all 4 neighbors non-zero!
            0,
            &[enclosed_mask, 0],
            &color_masks,
        );
        assert_eq!(out_planes[0], 0);
        assert_eq!(out_planes[1], 0b1000); // color 2 (`bit 1`) at cell 3 only!

        // ARC-AGI-3 transition circuit breaker
        let masks3 = [0xFFFF_FFFF_FFFF_FFFF, 0, 0]; // action 1
        assert_eq!(
            eval_arc3_policy_lut6(5, Some(0x42), 0x42, &masks3, u64::MAX),
            Arc3ReflexOutcome::EmitAction { action_id: 1 }
        );
        assert_eq!(
            eval_arc3_policy_lut6(5, Some(0x42), 0x99, &masks3, u64::MAX),
            Arc3ReflexOutcome::MechanicShiftBreaker {
                expected_sig: 0x42,
                actual_sig: 0x99
            }
        );
    }

    #[test]
    fn test_fwht1024_shift5_exact_orthogonality() {
        let mut v = [0i32; BONSAI_BLOCK_1024];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = (((i * 37 + 11) % 29) as i32 - 14) * 1024;
        }
        let orig = v;
        let signs = [0xA5A5_5A5A_F0F0_0F0Fu64; 16];
        fwht1024_shift5_signed(&mut v, &signs);
        // Apply transpose/inverse: FWHT without pre-sign, then post-sign
        let zero_signs = [0u64; 16];
        fwht1024_shift5_signed(&mut v, &zero_signs);
        for w in 0..16 {
            for b in 0..64 {
                if ((signs[w] >> b) & 1) != 0 {
                    v[w * 64 + b] = -v[w * 64 + b];
                }
            }
        }
        assert_eq!(v, orig, "FWHT-1024 >> 5 must be exact self-inverse on 1024-scaled integers");
    }

    #[test]
    fn test_mlx2bit_to_entromcu28_lossless_and_antisym_parity() {
        let mut trits = [0i8; 128];
        let mut words8 = [0u32; 8];
        for i in 0..128 {
            let t = ((i * 13 + 7) % 3) as i8 - 1;
            trits[i] = t;
            let code = (t + 1) as u32;
            words8[i / 16] |= code << ((i % 16) * 2);
        }
        let packed = transcode_mlx2bit_group128_to_entromcu28(&words8, 0x3C00).unwrap();
        assert_eq!(packed.len(), 28);
        let (unpacked, sc) = unpack_entromcu28_group128(&packed);
        assert_eq!(unpacked, trits);
        assert_eq!(sc, 0x3C00);

        // Invalid code == 3 must be rejected
        let mut bad_words = words8;
        bad_words[3] |= 0x3;
        assert!(transcode_mlx2bit_group128_to_entromcu28(&bad_words, 0x3C00).is_none());

        // Verify 16 x 1,024 Bonsai block exact parity vs naive reference
        let mut w = [[0i8; BONSAI_BLOCK_1024]; 16];
        for r in 0..16 {
            for c in 0..BONSAI_BLOCK_1024 {
                w[r][c] = (((r * 53 + c * 19 + 2) % 3) as i8) - 1;
            }
        }
        let mut x = [0i8; BONSAI_BLOCK_1024];
        for (i, s) in x.iter_mut().enumerate() {
            *s = (((i * 23 + 5) % 31) as i8) - 15;
        }
        let signs = [0x1234_5678_9ABC_DEF0u64; 16];
        let scales = [[3i16, 5, -2, 7, 1, -4, 6, 2]; 16];
        let codes = pack_bonsai_block16_codes(&w);
        let got = eval_bonsai1024_block16_antisym40(&x, &signs, &codes, &scales);

        let mut rot = [0i32; BONSAI_BLOCK_1024];
        for i in 0..BONSAI_BLOCK_1024 {
            rot[i] = (x[i] as i32) << 2;
        }
        fwht1024_shift5_signed(&mut rot, &signs);
        for r in 0..16 {
            let mut exp = 0i32;
            for grp in 0..8 {
                let mut gsum = 0i32;
                for c in (grp * 128)..((grp + 1) * 128) {
                    gsum += (w[r][c] as i32) * (rot[c].clamp(-127, 127));
                }
                exp += gsum * (scales[r][grp] as i32);
            }
            assert_eq!(got[r], exp);
        }
    }
}
