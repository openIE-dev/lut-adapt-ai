//! `#![no_std]` `waiOS` Cross-Fabric Conformance, `HeteroMosaic` Multi-Substrate Scheduler,
//! Relational Width-`2` `IW(2)` Object-Pair `LUT6` Predicate, & `JouleBit-0.10` (`0.105 bpw`)
//! Shared-Table `LUT6` Kernel (`Claims LA-49..LA-52`).
//!
//! Zero external crates, zero heap allocations, zero floating-point operations.
//! Computes the canonical `waiOS` `256 x 128` ternary matrix-vector integer program via
//! ISCA '25 `Antisym-40` zero-multiply 4-trit table lookup and hashes the 1,024-byte `i32`
//! little-endian output under `BLAKE3` key derivation (`"waios.tmac.v1"`), matching the
//! cross-fabric reference digest `a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0`
//! bit-for-bit and halting on the single-weight row-77 fault (`4ac7c4fb...`).

use crate::eval_lut6_bitsliced64;

pub const WAIOS_ROWS: usize = 256;
pub const WAIOS_COLS: usize = 128;
pub const WAIOS_GROUPS4: usize = WAIOS_COLS / 4; // 32 groups of 4 trits
pub const WAIOS_DOMAIN: &[u8] = b"waios.tmac.v1";

/// Canonical 32-byte `BLAKE3("waios.tmac.v1")` reference digest:
/// `a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0`
pub const WAIOS_REFERENCE_DIGEST: [u8; 32] = [
    0xa5, 0x41, 0x32, 0x3b, 0x13, 0x65, 0x3e, 0xd3, 0x65, 0x15, 0xff, 0x24, 0xef, 0x08, 0xde, 0xb7,
    0xdf, 0xa7, 0x71, 0xe5, 0x70, 0xa4, 0x34, 0x71, 0x27, 0x08, 0xfb, 0x72, 0x55, 0x2f, 0x46, 0xf0,
];

/// Canonical fault digest when row 77's first nonzero-activation weight flips by +1:
/// `4ac7c4fbc2f2a031ac4f3c0e485867fb689d24d4872bdb1b0a6bc80cf88b69dc`
pub const WAIOS_CHANGED_DIGEST: [u8; 32] = [
    0x4a, 0xc7, 0xc4, 0xfb, 0xc2, 0xf2, 0xa0, 0x31, 0xac, 0x4f, 0x3c, 0x0e, 0x48, 0x58, 0x67, 0xfb,
    0x68, 0x9d, 0x24, 0xd4, 0x87, 0x2b, 0xdb, 0x1b, 0x0a, 0x6b, 0xc8, 0x0c, 0xf8, 0x8b, 0x69, 0xdc,
];

const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;
const WAIOS_SEED: u64 = 0xdead_beef_cafe_babe;

/// Packed `waiOS` `256 x 128` input encoded into `32` 4-trit `Antisym-40` codes (`0..=80`) per row.
#[derive(Clone, Copy)]
pub struct WaiosPackedAudit {
    pub codes4: [[u8; WAIOS_GROUPS4]; WAIOS_ROWS],
    pub acts: [i8; WAIOS_COLS],
}

/// Generates the deterministic `waiOS` `256 x 128` ternary matrix + `128` `i8` activations from
/// `WAIOS_SEED`, optionally applying the canonical row-77 single-weight fault (`fault_row77 = true`).
pub fn build_waios_packed_audit(fault_row77: bool) -> WaiosPackedAudit {
    let mut s = WAIOS_SEED;
    let mut weights = [0i8; WAIOS_ROWS * WAIOS_COLS];
    let mut i = 0usize;
    while i < WAIOS_ROWS * WAIOS_COLS {
        s = s.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let m = (s >> 32) % 3;
        weights[i] = if m == 0 {
            -1
        } else if m == 1 {
            0
        } else {
            1
        };
        i += 1;
    }
    let mut acts = [0i8; WAIOS_COLS];
    let mut c = 0usize;
    while c < WAIOS_COLS {
        s = s.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        acts[c] = (((s >> 40) % 251) as i16 - 125) as i8;
        c += 1;
    }

    if fault_row77 {
        let mut nz_col = 0usize;
        while nz_col < WAIOS_COLS {
            if acts[nz_col] != 0 {
                break;
            }
            nz_col += 1;
        }
        let idx = 77 * WAIOS_COLS + nz_col;
        let from = weights[idx];
        weights[idx] = if from == 1 { 0 } else { from + 1 };
    }

    let mut codes4 = [[40u8; WAIOS_GROUPS4]; WAIOS_ROWS];
    let mut r = 0usize;
    while r < WAIOS_ROWS {
        let base = r * WAIOS_COLS;
        let mut g = 0usize;
        while g < WAIOS_GROUPS4 {
            let c0 = base + (g << 2);
            let u0 = (weights[c0] + 1) as u8;
            let u1 = (weights[c0 + 1] + 1) as u8;
            let u2 = (weights[c0 + 2] + 1) as u8;
            let u3 = (weights[c0 + 3] + 1) as u8;
            codes4[r][g] = u0 + 3 * u1 + 9 * u2 + 27 * u3;
            g += 1;
        }
        r += 1;
    }
    WaiosPackedAudit { codes4, acts }
}

/// Evaluates the `waiOS` `256 x 128` ternary matrix-vector product via **ISCA '25 `Antisym-40`
/// zero-multiply table lookup** (`0` multiplications across all `32,768` weights!).
#[inline(never)]
pub fn eval_waios_256x128_antisym40(audit: &WaiosPackedAudit, out: &mut [i32; WAIOS_ROWS]) {
    let mut luts41 = [[0i32; 41]; WAIOS_GROUPS4];
    let mut g = 0usize;
    while g < WAIOS_GROUPS4 {
        let c0 = g << 2;
        let x0 = audit.acts[c0] as i32;
        let x1 = audit.acts[c0 + 1] as i32;
        let x2 = audit.acts[c0 + 2] as i32;
        let x3 = audit.acts[c0 + 3] as i32;
        let t = &mut luts41[g];
        let mut idx = 0usize;

        // u3 = -1 -> codes 0..26 (pure additions/subtractions, 0 MULs!)
        let d2 = [-x2, 0, x2];
        let d1 = [-x1, 0, x1];
        let mut i2 = 0usize;
        while i2 < 3 {
            let s2 = -x3 + d2[i2];
            let mut i1 = 0usize;
            while i1 < 3 {
                let s1 = s2 + d1[i1];
                t[idx] = s1 - x0;
                t[idx + 1] = s1;
                t[idx + 2] = s1 + x0;
                idx += 3;
                i1 += 1;
            }
            i2 += 1;
        }
        // u3 = 0, u2 = -1 -> codes 27..35
        let mut j1 = 0usize;
        while j1 < 3 {
            let s1 = -x2 + d1[j1];
            t[idx] = s1 - x0;
            t[idx + 1] = s1;
            t[idx + 2] = s1 + x0;
            idx += 3;
            j1 += 1;
        }
        // u3 = 0, u2 = 0 -> codes 36..40
        t[36] = -x1 - x0;
        t[37] = -x1;
        t[38] = -x1 + x0;
        t[39] = -x0;
        t[40] = 0;
        g += 1;
    }

    let mut r = 0usize;
    while r < WAIOS_ROWS {
        let row_codes = &audit.codes4[r];
        let mut acc = 0i32;
        let mut grp = 0usize;
        while grp < WAIOS_GROUPS4 {
            let cd = row_codes[grp] as usize;
            if cd > 40 {
                acc -= luts41[grp][80 - cd];
            } else {
                acc += luts41[grp][cd];
            }
            grp += 1;
        }
        out[r] = acc;
        r += 1;
    }
}

/// Baseline scalar multiply-accumulate `waiOS` `256 x 128` loop (`32,768` runtime `MUL`s).
#[inline(never)]
pub fn eval_waios_256x128_scalar_mul(
    weights: &[i8; WAIOS_ROWS * WAIOS_COLS],
    acts: &[i8; WAIOS_COLS],
    out: &mut [i32; WAIOS_ROWS],
) {
    let mut r = 0usize;
    while r < WAIOS_ROWS {
        let base = r * WAIOS_COLS;
        let mut acc = 0i32;
        let mut c = 0usize;
        while c < WAIOS_COLS {
            acc += (weights[base + c] as i32) * (acts[c] as i32);
            c += 1;
        }
        out[r] = acc;
        r += 1;
    }
}

// ============================================================================
// Zero-Dependency `#![no_std]` BLAKE3 (`derive_key` mode for `<= 1,024`-byte chunks)
// ============================================================================

const B3_IV: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
    0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];
const B3_PERM: [u8; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const B3_QUADS: [u8; 32] = [
    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14,
];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const ROOT: u32 = 8;
const DERIVE_KEY_CONTEXT: u32 = 32;
const DERIVE_KEY_MATERIAL: u32 = 64;

#[inline(never)]
fn b3_compress(cv: &[u32; 8], block: &[u32; 16], block_len: u32, flags: u32) -> [u32; 8] {
    let mut s = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        B3_IV[0], B3_IV[1], B3_IV[2], B3_IV[3],
        0, 0, block_len, flags,
    ];
    let mut m = *block;
    let mut round = 0usize;
    while round < 7 {
        let mut qi = 0usize;
        while qi < 8 {
            let (a, b, c, d) = (
                B3_QUADS[qi * 4] as usize,
                B3_QUADS[qi * 4 + 1] as usize,
                B3_QUADS[qi * 4 + 2] as usize,
                B3_QUADS[qi * 4 + 3] as usize,
            );
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(m[qi * 2]);
            s[d] = (s[d] ^ s[a]).rotate_right(16);
            s[c] = s[c].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[c]).rotate_right(12);
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(m[qi * 2 + 1]);
            s[d] = (s[d] ^ s[a]).rotate_right(8);
            s[c] = s[c].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[c]).rotate_right(7);
            qi += 1;
        }
        if round < 6 {
            let mut next_m = [0u32; 16];
            let mut k = 0usize;
            while k < 16 {
                next_m[k] = m[B3_PERM[k] as usize];
                k += 1;
            }
            m = next_m;
        }
        round += 1;
    }
    let mut out = [0u32; 8];
    let mut i = 0usize;
    while i < 8 {
        out[i] = s[i] ^ s[i + 8];
        i += 1;
    }
    out
}

fn b3_block_words(slice: &[u8]) -> [u32; 16] {
    let mut pad = [0u8; 64];
    let n = if slice.len() > 64 { 64 } else { slice.len() };
    let mut i = 0usize;
    while i < n {
        pad[i] = slice[i];
        i += 1;
    }
    let mut w = [0u32; 16];
    let mut k = 0usize;
    while k < 16 {
        let j = k * 4;
        w[k] = u32::from_le_bytes([pad[j], pad[j + 1], pad[j + 2], pad[j + 3]]);
        k += 1;
    }
    w
}

fn b3_hash_chunk1024(input: &[u8], key: &[u32; 8], base_flags: u32) -> [u32; 8] {
    assert!(input.len() <= 1024);
    let blocks = if input.is_empty() {
        1
    } else {
        (input.len() + 63) / 64
    };
    let mut cv = *key;
    let mut b = 0usize;
    while b + 1 < blocks {
        let start = b * 64;
        let f = base_flags | if b == 0 { CHUNK_START } else { 0 };
        cv = b3_compress(&cv, &b3_block_words(&input[start..start + 64]), 64, f);
        b += 1;
    }
    let last_start = (blocks - 1) * 64;
    let last_slice = if input.is_empty() {
        &[][..]
    } else {
        &input[last_start..]
    };
    let f_last = base_flags
        | if blocks == 1 { CHUNK_START } else { 0 }
        | CHUNK_END
        | ROOT;
    b3_compress(&cv, &b3_block_words(last_slice), last_slice.len() as u32, f_last)
}

/// Computes `BLAKE3::derive_key("waios.tmac.v1", output_le_bytes)` over the `256` `i32` outputs.
pub fn waios_digest_of_outputs(out: &[i32; WAIOS_ROWS]) -> [u8; 32] {
    let ctx_cv = b3_hash_chunk1024(WAIOS_DOMAIN, &B3_IV, DERIVE_KEY_CONTEXT);
    let mut bytes = [0u8; WAIOS_ROWS * 4];
    let mut i = 0usize;
    while i < WAIOS_ROWS {
        let b = out[i].to_le_bytes();
        let off = i * 4;
        bytes[off] = b[0];
        bytes[off + 1] = b[1];
        bytes[off + 2] = b[2];
        bytes[off + 3] = b[3];
        i += 1;
    }
    let root_words = b3_hash_chunk1024(&bytes, &ctx_cv, DERIVE_KEY_MATERIAL);
    let mut digest = [0u8; 32];
    let mut k = 0usize;
    while k < 8 {
        let b = root_words[k].to_le_bytes();
        digest[k * 4] = b[0];
        digest[k * 4 + 1] = b[1];
        digest[k * 4 + 2] = b[2];
        digest[k * 4 + 3] = b[3];
        k += 1;
    }
    digest
}

/// Compact streaming `waiOS` `256 x 128` `Antisym-40` + `BLAKE3("waios.tmac.v1")` evaluator
/// (`0` `32 KB`/`8 KB` stack buffers — ideal for `<= 16 KB` `.wasm` and tiny `MCU` SRAM!).
#[inline(never)]
pub fn eval_waios_streaming_antisym40_digest(fault_row77: bool) -> [u8; 32] {
    let mut s_act = WAIOS_SEED;
    let mut k = 0usize;
    while k < WAIOS_ROWS * WAIOS_COLS {
        s_act = s_act.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        k += 1;
    }
    let mut acts = [0i8; WAIOS_COLS];
    let mut nz_col = WAIOS_COLS;
    let mut c = 0usize;
    while c < WAIOS_COLS {
        s_act = s_act.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let a = (((s_act >> 40) % 251) as i16 - 125) as i8;
        if nz_col == WAIOS_COLS && a != 0 {
            nz_col = c;
        }
        acts[c] = a;
        c += 1;
    }

    let mut out = [0i32; WAIOS_ROWS];
    let mut s_w = WAIOS_SEED;
    let mut r = 0usize;
    while r < WAIOS_ROWS {
        let mut acc = 0i32;
        let mut g = 0usize;
        while g < WAIOS_GROUPS4 {
            let c0 = g << 2;
            let mut code = 0u8;
            let mut pow3 = 1u8;
            let mut j = 0usize;
            while j < 4 {
                s_w = s_w.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
                let mut m = ((s_w >> 32) % 3) as u8;
                if fault_row77 && r == 77 && (c0 + j) == nz_col {
                    m = if m == 2 { 1 } else { m + 1 };
                }
                code += m * pow3;
                pow3 *= 3;
                j += 1;
            }
            // Antisym-40 reflection (`code > 40 -> negate mirror 80 - code; code == 40 -> 0`)
            if code != 40 {
                let (mut mc, neg) = if code > 40 { (80 - code, true) } else { (code, false) };
                let mut ps = 0i32;
                let mut t = 0usize;
                while t < 4 {
                    let xv = acts[c0 + t] as i32;
                    let ut = mc % 3;
                    mc /= 3;
                    if ut == 0 {
                        ps -= xv;
                    } else if ut == 2 {
                        ps += xv;
                    }
                    t += 1;
                }
                if neg { acc -= ps; } else { acc += ps; }
            }
            g += 1;
        }
        out[r] = acc;
        r += 1;
    }
    waios_digest_of_outputs(&out)
}

/// Outcome of the `waiOS` cross-fabric conformance & `HeteroMosaic` substrate scheduler gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaiosMosaicVerdict {
    /// Digest matches `WAIOS_REFERENCE_DIGEST`; routes workload to minimum-joule substrate (`0..=3`).
    SealedMatch {
        substrate_id: u8,
        digest_prefix_be: u32,
    },
    /// Digest diverged (`e.g.`, `0x4ac7c4fb` on row-77 1-weight fault): halt immediately, seal nothing!
    HaltedMismatch { digest_prefix_be: u32 },
}

/// `HeteroMosaic-MoL` substrate selector:
///   - `0`: `Rung-0 LUT6` (`SRAM`/`BRAM` zero-FLOP hit, `~34–201 nJ/op`)
///   - `1`: `MCU / Cortex-M55` `Antisym-40 T-MAC` (`small stationary integer batch`)
///   - `2`: `Host CPU` `Antisym-40 + FWHT` (`medium sparse router / horizon step`)
///   - `3`: `iGPU / Metal / Vulkan` `CubeCL` (`high-parallelism batch >= 16`)
#[inline]
pub const fn select_heteromosaic_substrate(
    lut6_certified: bool,
    batch_tokens: u16,
    active_dim: u16,
) -> u8 {
    if lut6_certified {
        0
    } else if batch_tokens <= 2 && active_dim <= 256 {
        1
    } else if batch_tokens < 16 {
        2
    } else {
        3
    }
}

/// Evaluates `IW(2)` relational object-pair `LUT6` predicate over `6` binary spatial/causal features:
/// `b0 = same_row`, `b1 = same_col`, `b2 = manhattan_le_2`, `b3 = color_match`,
/// `b4 = count_delta_pos`, `b5 = remote_switch_toggled`.
#[inline]
pub const fn eval_iw2_relational_lut6(rel_features6: u8, init_mask: u64) -> bool {
    let idx = (rel_features6 & 0x3F) as u32;
    let x0 = if (idx & 1) != 0 { u64::MAX } else { 0 };
    let x1 = if (idx & 2) != 0 { u64::MAX } else { 0 };
    let x2 = if (idx & 4) != 0 { u64::MAX } else { 0 };
    let x3 = if (idx & 8) != 0 { u64::MAX } else { 0 };
    let x4 = if (idx & 16) != 0 { u64::MAX } else { 0 };
    let x5 = if (idx & 32) != 0 { u64::MAX } else { 0 };
    (eval_lut6_bitsliced64(init_mask, x0, x1, x2, x3, x4, x5) & 1) != 0
}

/// `Window 18` (`Claim LA-51`): `JouleBit-0.10` (`0.105 bpw`) rank-`6` shared-table `LUT6` binary-latent
/// projection (`64` rows x `96` cols: `16` 6-input groups -> `16 x 64` shared partial sums + `1` rank-`6`
/// `64`-entry `LUT6` output table, **zero binary multiplications**!) vs. `LittleBit-2` runtime unpack+MAC.
#[inline(never)]
pub fn eval_joulebit_0p10bpw_lut6_64x96(
    acts96: &[i16; 96],
    v_lut6_idx: &[[u8; 16]; 6],
    u_lut6_idx: &[u8; 64],
    out64: &mut [i32; 64],
) {
    // Stage 1: Build shared 6-bit sign partial sums for 16 input groups (`0` MULs!)
    let mut z_lat = [0i32; 6];
    let mut g = 0usize;
    while g < 16 {
        let b = g * 6;
        let a0 = acts96[b] as i32;
        let a1 = acts96[b + 1] as i32;
        let a2 = acts96[b + 2] as i32;
        let a3 = acts96[b + 3] as i32;
        let a4 = acts96[b + 4] as i32;
        let a5 = acts96[b + 5] as i32;
        let mut r = 0usize;
        while r < 6 {
            let m = v_lut6_idx[r][g];
            let s0 = if (m & 1) != 0 { -a0 } else { a0 };
            let s1 = if (m & 2) != 0 { -a1 } else { a1 };
            let s2 = if (m & 4) != 0 { -a2 } else { a2 };
            let s3 = if (m & 8) != 0 { -a3 } else { a3 };
            let s4 = if (m & 16) != 0 { -a4 } else { a4 };
            let s5 = if (m & 32) != 0 { -a5 } else { a5 };
            z_lat[r] += s0 + s1 + s2 + s3 + s4 + s5;
            r += 1;
        }
        g += 1;
    }

    // Stage 2: Precompute ALL `2^6 = 64` signed combinations of `z_lat[0..6]` ONCE (`0` MULs!),
    // then gather all `64` output channels in `1` `LUT6` table lookup per row!
    let mut u_table = [0i32; 64];
    let mut code = 0usize;
    while code < 64 {
        let mut s = 0i32;
        if (code & 1) != 0 { s -= z_lat[0]; } else { s += z_lat[0]; }
        if (code & 2) != 0 { s -= z_lat[1]; } else { s += z_lat[1]; }
        if (code & 4) != 0 { s -= z_lat[2]; } else { s += z_lat[2]; }
        if (code & 8) != 0 { s -= z_lat[3]; } else { s += z_lat[3]; }
        if (code & 16) != 0 { s -= z_lat[4]; } else { s += z_lat[4]; }
        if (code & 32) != 0 { s -= z_lat[5]; } else { s += z_lat[5]; }
        u_table[code] = s >> 3;
        code += 1;
    }

    let mut row = 0usize;
    while row < 64 {
        out64[row] = u_table[(u_lut6_idx[row] & 0x3F) as usize];
        row += 1;
    }
}

/// Baseline `LittleBit-2` `0.105 bpw` runtime bitpack-unpack + dense `64 x 96` MAC execution (`6,144` MACs).
#[inline(never)]
pub fn eval_littlebit2_0p10bpw_dense_unpack_64x96(
    acts96: &[i16; 96],
    v_lut6_idx: &[[u8; 16]; 6],
    u_lut6_idx: &[u8; 64],
    out64: &mut [i32; 64],
) {
    // Unpacks low-rank binary factors into dense `[64][96]` effective weights and multiplies (`LittleBit` runtime path)
    let mut v_dense = [[0i16; 96]; 6];
    let mut r = 0usize;
    while r < 6 {
        let mut g = 0usize;
        while g < 16 {
            let m = v_lut6_idx[r][g];
            let mut k = 0usize;
            while k < 6 {
                v_dense[r][g * 6 + k] = if ((m >> k) & 1) != 0 { -1 } else { 1 };
                k += 1;
            }
            g += 1;
        }
        r += 1;
    }
    let mut row = 0usize;
    while row < 64 {
        let um = u_lut6_idx[row] & 0x3F;
        let mut acc_row = 0i32;
        let mut c = 0usize;
        while c < 96 {
            let mut w_rc = 0i32;
            let mut rk = 0usize;
            while rk < 6 {
                let us: i32 = if ((um >> rk) & 1) != 0 { -1 } else { 1 };
                w_rc += us * (v_dense[rk][c] as i32);
                rk += 1;
            }
            acc_row += w_rc * (acts96[c] as i32);
            c += 1;
        }
        out64[row] = acc_row >> 3;
        row += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waios_256x128_antisym40_matches_canonical_blake3_digest() {
        let clean = build_waios_packed_audit(false);
        let mut out = [0i32; WAIOS_ROWS];
        eval_waios_256x128_antisym40(&clean, &mut out);
        // Check first 4 rows against canonical `waiOS` `SEED = 0xdeadbeefcafebabe` `kernel.ts` output!
        assert_eq!([out[0], out[1], out[2], out[3]], [-852, 74, 153, 513]);
        let digest = waios_digest_of_outputs(&out);
        assert_eq!(digest, WAIOS_REFERENCE_DIGEST);

        // Single-weight fault in row 77 MUST produce exact `WAIOS_CHANGED_DIGEST`!
        let faulted = build_waios_packed_audit(true);
        let mut out_fault = [0i32; WAIOS_ROWS];
        eval_waios_256x128_antisym40(&faulted, &mut out_fault);
        let digest_fault = waios_digest_of_outputs(&out_fault);
        assert_eq!(digest_fault, WAIOS_CHANGED_DIGEST);
        assert_ne!(digest_fault, WAIOS_REFERENCE_DIGEST);
    }

    #[test]
    fn test_joulebit_0p10bpw_lut6_matches_dense_unpack_bit_for_bit() {
        let mut acts = [0i16; 96];
        for (i, slot) in acts.iter_mut().enumerate() {
            *slot = ((i as i16 * 13 + 7) % 63) - 31;
        }
        let mut v_idx = [[0u8; 16]; 6];
        for r in 0..6 {
            for g in 0..16 {
                v_idx[r][g] = ((r * 17 + g * 29 + 5) & 0x3F) as u8;
            }
        }
        let mut u_idx = [0u8; 64];
        for (row, slot) in u_idx.iter_mut().enumerate() {
            *slot = ((row * 37 + 11) & 0x3F) as u8;
        }
        let mut out_lut6 = [0i32; 64];
        let mut out_dense = [0i32; 64];
        eval_joulebit_0p10bpw_lut6_64x96(&acts, &v_idx, &u_idx, &mut out_lut6);
        eval_littlebit2_0p10bpw_dense_unpack_64x96(&acts, &v_idx, &u_idx, &mut out_dense);
        assert_eq!(out_lut6, out_dense);
    }
}
