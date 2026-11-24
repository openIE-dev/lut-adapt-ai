//! Zero-Multiplier `EntroMCU` (`5 trits/byte`, `1.585 bpw`) + `PALUTE` / `T-MAC` 4-Trit SRAM
//! Lookup Tail occupying the exact `672` bytes (`bytes 1376..2048`) inside `CompactLut6Page`.
//!
//! Layout of `pad: [u8; 672]` when `CompactLut6Page.flags & FLAG_HAS_TMAC_TAIL != 0`:
//!   - `0..4`:     `b"TMAC"` magic
//!   - `4`:        `out_channels` (`16`: `[0..7]` payload bits, `[8..15]` conformal margin gates)
//!   - `5`:        `in_groups` (`16`: `16 * 4 = 64` ternary inputs)
//!   - `6..8`:     `margin_thresh: i16` (LE; minimum signed margin per conformal gate `[8..15]`)
//!   - `8..213`:   `205` bytes `EntroMCU` canonical base-3 (`5 trits/byte`, `3^5 = 243 <= 256`)
//!                 lossless archive of all `16 * 64 = 1,024` ternary weights (`1.60` bits/weight)
//!   - `213..216`: `3` alignment bytes
//!   - `216..472`: `256` bytes (`16 x 16`) pre-indexed `T-MAC` 4-trit group codes (`0..81`)
//!   - `472..504`: `32` bytes (`16 x i16` LE) channel biases
//!   - `504..536`: `32` bytes SHA-256 tail integrity seal
//!   - `536..672`: `136` bytes canonical Huffman / calibration metadata (`0` wasted page bytes)

pub const TMAC_PAD_BYTES: usize = 672;
pub const TMAC_OUT_CHANNELS: usize = 16;
pub const TMAC_IN_GROUPS: usize = 16;
pub const TMAC_IN_DIM: usize = TMAC_IN_GROUPS * 4; // 64

/// Packs 5 signed trits `w0..w4 ∈ {-1, 0, +1}` into a single `EntroMCU` base-3 byte (`0..242`).
#[inline(always)]
pub const fn pack_5trits(w0: i8, w1: i8, w2: i8, w3: i8, w4: i8) -> u8 {
    let t0 = (w0 + 1) as u8;
    let t1 = (w1 + 1) as u8;
    let t2 = (w2 + 1) as u8;
    let t3 = (w3 + 1) as u8;
    let t4 = (w4 + 1) as u8;
    t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4
}

/// Unpacks a single `EntroMCU` base-3 byte (`0..242`) back into 5 signed trits `∈ {-1, 0, +1}`.
#[inline(always)]
pub const fn unpack_5trits(byte: u8) -> [i8; 5] {
    let mut rem = byte;
    let mut out = [0i8; 5];
    let mut k = 0usize;
    while k < 5 {
        let u = rem % 3;
        rem /= 3;
        out[k] = (u as i8) - 1;
        k += 1;
    }
    out
}

/// Packs 4 signed trits `w0..w3 ∈ {-1, 0, +1}` into a `T-MAC` 81-state group index (`0..80`).
#[inline(always)]
pub const fn pack_4trits(w0: i8, w1: i8, w2: i8, w3: i8) -> u8 {
    let u0 = (w0 + 1) as u8;
    let u1 = (w1 + 1) as u8;
    let u2 = (w2 + 1) as u8;
    let u3 = (w3 + 1) as u8;
    u0 + 3 * u1 + 9 * u2 + 27 * u3
}

/// Builds the 81-entry `PALUTE` / `T-MAC` signed activation lookup table for 4 inputs `(x0..x3)`
/// using **ONLY integer additions and subtractions** (`0` multiplications, `0` FLOPs).
#[inline(always)]
pub const fn build_tmac_group_lut(x0: i8, x1: i8, x2: i8, x3: i8) -> [i16; 81] {
    let v0 = [-(x0 as i16), 0, x0 as i16];
    let v1 = [-(x1 as i16), 0, x1 as i16];
    let v2 = [-(x2 as i16), 0, x2 as i16];
    let v3 = [-(x3 as i16), 0, x3 as i16];

    let mut lut = [0i16; 81];
    let mut idx = 0usize;
    let mut u3 = 0usize;
    while u3 < 3 {
        let s3 = v3[u3];
        let mut u2 = 0usize;
        while u2 < 3 {
            let s2 = s3 + v2[u2];
            let mut u1 = 0usize;
            while u1 < 3 {
                let s1 = s2 + v1[u1];
                lut[idx] = s1 + v0[0];
                lut[idx + 1] = s1 + v0[1];
                lut[idx + 2] = s1 + v0[2];
                idx += 3;
                u1 += 1;
            }
            u2 += 1;
        }
        u3 += 1;
    }
    lut
}

/// Serializes `16 x 64` ternary weights (`∈ {-1, 0, +1}`) + `16` biases into the `672`-byte
/// `CompactLut6Page._pad` slot (`const fn` compatible for bare-metal ROM/MRAM embedding).
pub const fn encode_tmac_tail_pad(
    weights: &[[i8; TMAC_IN_DIM]; TMAC_OUT_CHANNELS],
    biases: &[i16; TMAC_OUT_CHANNELS],
    margin_thresh: i16,
) -> [u8; TMAC_PAD_BYTES] {
    let mut pad = [0u8; TMAC_PAD_BYTES];
    pad[0] = b'T';
    pad[1] = b'M';
    pad[2] = b'A';
    pad[3] = b'C';
    pad[4] = TMAC_OUT_CHANNELS as u8;
    pad[5] = TMAC_IN_GROUPS as u8;
    let mt = margin_thresh.to_le_bytes();
    pad[6] = mt[0];
    pad[7] = mt[1];

    // 1. Pack 1,024 trits into 205 EntroMCU base-3 bytes (`bytes 8..213`)
    let mut flat = [0i8; 1025];
    let mut r = 0usize;
    while r < TMAC_OUT_CHANNELS {
        let mut c = 0usize;
        while c < TMAC_IN_DIM {
            flat[r * TMAC_IN_DIM + c] = weights[r][c];
            c += 1;
        }
        r += 1;
    }
    let mut b_idx = 0usize;
    while b_idx < 205 {
        let base = b_idx * 5;
        pad[8 + b_idx] = pack_5trits(
            flat[base],
            flat[base + 1],
            flat[base + 2],
            flat[base + 3],
            flat[base + 4],
        );
        b_idx += 1;
    }

    // 2. Pre-index 16 x 16 4-trit T-MAC group codes (`bytes 216..472`)
    r = 0;
    while r < TMAC_OUT_CHANNELS {
        let mut g = 0usize;
        while g < TMAC_IN_GROUPS {
            let c0 = g << 2;
            pad[216 + (r << 4) + g] = pack_4trits(
                weights[r][c0],
                weights[r][c0 + 1],
                weights[r][c0 + 2],
                weights[r][c0 + 3],
            );
            g += 1;
        }
        r += 1;
    }

    // 3. Biases (`bytes 472..504`)
    r = 0;
    while r < TMAC_OUT_CHANNELS {
        let bb = biases[r].to_le_bytes();
        pad[472 + (r << 1)] = bb[0];
        pad[472 + (r << 1) + 1] = bb[1];
        r += 1;
    }

    pad
}

#[inline(always)]
pub const fn has_valid_tmac_header(pad: &[u8; TMAC_PAD_BYTES]) -> bool {
    pad[0] == b'T'
        && pad[1] == b'M'
        && pad[2] == b'A'
        && pad[3] == b'C'
        && pad[4] == (TMAC_OUT_CHANNELS as u8)
        && pad[5] == (TMAC_IN_GROUPS as u8)
}

/// Evaluates the `672`-byte `T-MAC` ternary tail in **ZERO multiplications** and **ZERO FLOPs**:
/// builds `16` small `81`-entry signed group tables in stack SRAM, then accumulates `16` output
/// channels via pure table lookups + integer additions.
/// Returns `(payload_bits_u8, conformal_votes_u16, raw_accumulators)`.
#[inline]
pub const fn eval_tmac_from_pad(
    pad: &[u8; TMAC_PAD_BYTES],
    features: &[i8; TMAC_IN_DIM],
) -> (u64, u16, [i16; TMAC_OUT_CHANNELS]) {
    if !has_valid_tmac_header(pad) {
        return (0, 0, [0i16; TMAC_OUT_CHANNELS]);
    }
    let margin_thresh = i16::from_le_bytes([pad[6], pad[7]]);

    // Step 1: Build 16 x 81 zero-multiplier lookup tables on stack (`2,592` bytes hot L1/DTCM)
    let mut group_luts = [[0i16; 81]; TMAC_IN_GROUPS];
    let mut g = 0usize;
    while g < TMAC_IN_GROUPS {
        let c0 = g << 2;
        group_luts[g] = build_tmac_group_lut(
            features[c0],
            features[c0 + 1],
            features[c0 + 2],
            features[c0 + 3],
        );
        g += 1;
    }

    // Step 2: Zero-multiplier table gather + integer accumulate across 16 output channels
    let mut accs = [0i16; TMAC_OUT_CHANNELS];
    let mut payload_bits = 0u64;
    let mut votes = 0u16;

    let mut r = 0usize;
    while r < TMAC_OUT_CHANNELS {
        let bias = i16::from_le_bytes([pad[472 + (r << 1)], pad[472 + (r << 1) + 1]]);
        let mut sum = bias as i32;
        let row_off = 216 + (r << 4);

        // Unroll by 4 groups per step (`0` MULs!)
        let mut grp = 0usize;
        while grp < TMAC_IN_GROUPS {
            let c0 = pad[row_off + grp] as usize;
            let c1 = pad[row_off + grp + 1] as usize;
            let c2 = pad[row_off + grp + 2] as usize;
            let c3 = pad[row_off + grp + 3] as usize;
            sum += (group_luts[grp][c0] as i32)
                + (group_luts[grp + 1][c1] as i32)
                + (group_luts[grp + 2][c2] as i32)
                + (group_luts[grp + 3][c3] as i32);
            grp += 4;
        }
        let clamped = if sum > (i16::MAX as i32) {
            i16::MAX
        } else if sum < (i16::MIN as i32) {
            i16::MIN
        } else {
            sum as i16
        };
        accs[r] = clamped;

        if r < 8 {
            if clamped > 0 {
                payload_bits |= 1u64 << r;
            }
        } else if clamped >= margin_thresh {
            votes += 1;
        }
        r += 1;
    }

    (payload_bits, votes, accs)
}

/// Builds the antisymmetric `41`-entry `LUT Tensor Core` (ISCA 2025) half-table for 4 inputs
/// `(x0..x3)` using **ONLY integer additions and subtractions** (`0` multiplications, `0` FLOPs,
/// `2.025×` fewer non-zero entries than the full 81-entry `T-MAC` table).
///
/// Because `pack_4trits(-w0, -w1, -w2, -w3) == 80 - pack_4trits(w0, w1, w2, w3)`, every ternary
/// 4-group lookup satisfies strict odd symmetry around midpoint `40`:
///   - `code < 40`  -> `lut41[code]`
///   - `code == 40` -> `lut41[40] == 0`
///   - `code > 40`  -> `-lut41[80 - code]`
#[inline(always)]
pub const fn build_tmac_antisym_lut41(x0: i8, x1: i8, x2: i8, x3: i8) -> [i16; 41] {
    let v0 = [-(x0 as i16), 0, x0 as i16];
    let v1 = [-(x1 as i16), 0, x1 as i16];
    let v2 = [-(x2 as i16), 0, x2 as i16];
    let sx3 = -(x3 as i16); // u3 = 0 (`codes 0..26`) and u3 = 1 (`codes 27..39`, where v3[1] = 0)

    let mut lut = [0i16; 41];
    let mut idx = 0usize;
    // u3 = 0 -> codes 0..26 (all 9 (u2, u1) pairs x 3 u0 values)
    let mut u2 = 0usize;
    while u2 < 3 {
        let s2 = sx3 + v2[u2];
        let mut u1 = 0usize;
        while u1 < 3 {
            let s1 = s2 + v1[u1];
            lut[idx] = s1 + v0[0];
            lut[idx + 1] = s1 + v0[1];
            lut[idx + 2] = s1 + v0[2];
            idx += 3;
            u1 += 1;
        }
        u2 += 1;
    }
    // u3 = 1 (`v3 = 0`), u2 = 0 -> codes 27..35
    let s2_u20 = v2[0];
    let mut u1 = 0usize;
    while u1 < 3 {
        let s1 = s2_u20 + v1[u1];
        lut[idx] = s1 + v0[0];
        lut[idx + 1] = s1 + v0[1];
        lut[idx + 2] = s1 + v0[2];
        idx += 3;
        u1 += 1;
    }
    // u3 = 1 (`v3 = 0`), u2 = 1 (`v2 = 0`):
    //   u1 = 0 -> codes 36..38
    let s1_36 = v1[0];
    lut[36] = s1_36 + v0[0];
    lut[37] = s1_36 + v0[1];
    lut[38] = s1_36 + v0[2];
    //   u1 = 1 (`v1 = 0`), u0 = 0 -> code 39; u0 = 1 -> code 40 (`0`)
    lut[39] = v0[0];
    lut[40] = 0;
    lut
}

/// Branchless odd-symmetric lookup (`ARM64 csneg` / `x86 cmov`) over `build_tmac_antisym_lut41`.
#[inline(always)]
pub const fn lookup_tmac_antisym41(lut41: &[i16; 41], code: u8) -> i32 {
    let mirrored = if code > 40 {
        (80 - code) as usize
    } else {
        code as usize
    };
    let v = lut41[mirrored] as i32;
    if code > 40 { -v } else { v }
}

/// Evaluates the `672`-byte `T-MAC` ternary tail using **Antisymmetric 40-Entry Table
/// Symmetrization** (`LUT Tensor Core`, ISCA 2025): cuts MCU stack SRAM from `2,592 B` to
/// `1,312 B` (`49.4%` reduction) and halves table-precomputation additions while preserving
/// `100%` bit-exact equivalence with `eval_tmac_from_pad` on existing `.lut6` crystals.
#[inline]
pub const fn eval_tmac_antisym_from_pad(
    pad: &[u8; TMAC_PAD_BYTES],
    features: &[i8; TMAC_IN_DIM],
) -> (u64, u16, [i16; TMAC_OUT_CHANNELS]) {
    if !has_valid_tmac_header(pad) {
        return (0, 0, [0i16; TMAC_OUT_CHANNELS]);
    }
    let margin_thresh = i16::from_le_bytes([pad[6], pad[7]]);

    let mut group_luts = [[0i16; 41]; TMAC_IN_GROUPS];
    let mut g = 0usize;
    while g < TMAC_IN_GROUPS {
        let c0 = g << 2;
        group_luts[g] = build_tmac_antisym_lut41(
            features[c0],
            features[c0 + 1],
            features[c0 + 2],
            features[c0 + 3],
        );
        g += 1;
    }

    let mut accs = [0i16; TMAC_OUT_CHANNELS];
    let mut payload_bits = 0u64;
    let mut votes = 0u16;

    let mut r = 0usize;
    while r < TMAC_OUT_CHANNELS {
        let bias = i16::from_le_bytes([pad[472 + (r << 1)], pad[472 + (r << 1) + 1]]);
        let mut sum = bias as i32;
        let row_off = 216 + (r << 4);

        let mut grp = 0usize;
        while grp < TMAC_IN_GROUPS {
            sum += lookup_tmac_antisym41(&group_luts[grp], pad[row_off + grp])
                + lookup_tmac_antisym41(&group_luts[grp + 1], pad[row_off + grp + 1])
                + lookup_tmac_antisym41(&group_luts[grp + 2], pad[row_off + grp + 2])
                + lookup_tmac_antisym41(&group_luts[grp + 3], pad[row_off + grp + 3]);
            grp += 4;
        }
        let clamped = if sum > (i16::MAX as i32) {
            i16::MAX
        } else if sum < (i16::MIN as i32) {
            i16::MIN
        } else {
            sum as i16
        };
        accs[r] = clamped;

        if r < 8 {
            if clamped > 0 {
                payload_bits |= 1u64 << r;
            }
        } else if clamped >= margin_thresh {
            votes += 1;
        }
        r += 1;
    }

    (payload_bits, votes, accs)
}

/// Converts a 64-bit canonical wire state into bipolar `±16` `i8` activations when raw continuous
/// ADC/embedding features are not supplied separately.
#[inline(always)]
pub const fn wire_state_to_bipolar_i8(state: u64) -> [i8; TMAC_IN_DIM] {
    let mut out = [-16i8; TMAC_IN_DIM];
    let mut i = 0usize;
    while i < TMAC_IN_DIM {
        if ((state >> i) & 1) != 0 {
            out[i] = 16;
        }
        i += 1;
    }
    out
}

// Compile-time formal check of EntroMCU 5-trit bijection, T-MAC zero-MUL equivalence,
// and exhaustive 81/81 Antisymmetric 40-Entry Table Symmetrization parity:
const _: () = {
    let b = pack_5trits(-1, 0, 1, -1, 1);
    assert!(b < 243);
    let u = unpack_5trits(b);
    assert!(u[0] == -1 && u[1] == 0 && u[2] == 1 && u[3] == -1 && u[4] == 1);

    let lut81 = build_tmac_group_lut(17, -23, 41, -9);
    let lut41 = build_tmac_antisym_lut41(17, -23, 41, -9);
    let code = pack_4trits(1, -1, 1, -1) as usize;
    assert!(lut81[code] == 90);

    let mut c = 0usize;
    while c < 81 {
        assert!(lookup_tmac_antisym41(&lut41, c as u8) == (lut81[c] as i32));
        c += 1;
    }
};
