//! `#![no_std]` Serialized Iterated Width (`SIW(2)`) Landmark `LUT6` Two-Bank Vault &
//! Fused Zero-Multiplier `FWHT-64 + 6-Column JouleBit-LUT6` Sub-Bit Drafter (`Claims LA-53..LA-55`).

use crate::hadamard_bonsai::fwht64_shift3_signed;

/// Outcome of a `SIW(2)` two-bank (`8 KB`) `System-0` reflex + landmark transition step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Siw2ReflexOutcome {
    /// Emits a certified `System-0` action from `Bank 0` or `Bank 1` (`bank_id` in `0..=1`).
    CertifiedAction {
        action_id: u8,
        bank_id: u8,
        landmark_reset: bool,
        next_epoch: u8,
    },
    /// Halts immediately on `Theorem 73` transition-signature mismatch (`0` stale actions).
    MechanicShiftBreaker { xor_sig: u8, next_epoch: u8 },
    /// Escalates to the `SIW(2)` landmark novelty planner (clearing novelty tables if `landmark_reset`).
    WakeSiw2Planner { landmark_reset: bool, next_epoch: u8 },
}

/// Evaluates the `#![no_std]` `SIW(2)` landmark detector (`landmark_lut6_mask` over `landmark6`)
/// and two-bank (`Bank 0` / `Bank 1`) `System-0` trajectory reflex (`Theorems 91..93`).
#[inline]
pub fn eval_siw2_landmark_lut6(
    percept6: u8,
    landmark6: u8,
    epoch: u8,
    expected_sig: Option<u8>,
    actual_sig: u8,
    landmark_lut6_mask: u64,
    bank0_masks: &[u64; 3],
    bank1_masks: &[u64; 3],
    certified_mask_bank0: u64,
    certified_mask_bank1: u64,
) -> Siw2ReflexOutcome {
    let lm_idx = (landmark6 & 0x3F) as u32;
    let lm_hit = ((landmark_lut6_mask >> lm_idx) & 1) != 0;
    let landmark_reset = lm_hit && epoch < 8;
    let next_epoch = if landmark_reset { epoch + 1 } else { epoch };

    if let Some(exp) = expected_sig {
        if exp != actual_sig {
            return Siw2ReflexOutcome::MechanicShiftBreaker {
                xor_sig: exp ^ actual_sig,
                next_epoch,
            };
        }
    }

    let p_idx = (percept6 & 0x3F) as u32;
    let bank_id = if next_epoch == 0 { 0u8 } else { 1u8 };
    let (masks, cert_mask) = if bank_id == 0 {
        (bank0_masks, certified_mask_bank0)
    } else {
        (bank1_masks, certified_mask_bank1)
    };

    if ((cert_mask >> p_idx) & 1) == 0 {
        return Siw2ReflexOutcome::WakeSiw2Planner {
            landmark_reset,
            next_epoch,
        };
    }

    let b0 = ((masks[0] >> p_idx) & 1) as u8;
    let b1 = ((masks[1] >> p_idx) & 1) as u8;
    let b2 = ((masks[2] >> p_idx) & 1) as u8;
    let code = b0 | (b1 << 1) | (b2 << 2);
    if code == 0 {
        Siw2ReflexOutcome::WakeSiw2Planner {
            landmark_reset,
            next_epoch,
        }
    } else {
        Siw2ReflexOutcome::CertifiedAction {
            action_id: code,
            bank_id,
            landmark_reset,
            next_epoch,
        }
    }
}

/// Zero-multiplier in-place `64`-point Sylvester Fast Walsh-Hadamard Transform (`64 = 4^3`),
/// normalized by arithmetic right-shift `>> 3` (reuses `fwht64_shift3_signed` with `sign_mask = 0`).
#[inline]
pub fn fwht64_inplace_i32(buf: &mut [i32; 64]) {
    fwht64_shift3_signed(buf, 0);
}

#[inline(never)]
fn build_rad6_slab(vals: &[i32; 6], shift_out: u32, slab: &mut [i32; 64]) {
    let n64 = core::hint::black_box(64usize);
    let n6 = core::hint::black_box(6usize);
    let mut m = 0usize;
    while m < n64 {
        let mut s = 0i32;
        let mut k = 0usize;
        while k < n6 {
            let v = vals[k];
            s += if ((m >> k) & 1) != 0 { -v } else { v };
            k += 1;
        }
        slab[m] = s >> shift_out;
        m += 1;
    }
}

/// Evaluates a `64 x 64` `JouleBit-LUT6` sub-bit draft block (`rank = 6`, `10` `6`-column slabs
/// over the first `60` `FWHT-64`-rotated coordinates) with **ZERO** runtime binary multiplications
/// (`Theorem 94`)!
pub fn eval_joulebit6_fwht64_slab_i16(
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) {
    let mut z = [0i32; 64];
    let mut c = 0usize;
    while c < 64 {
        z[c] = acts64[c] as i32;
        c += 1;
    }
    fwht64_inplace_i32(&mut z);

    let mut h_lat = [0i32; 6];
    let mut slab = [0i32; 64];
    let mut g = 0usize;
    while g < 10 {
        let b = g * 6;
        let grp = [z[b], z[b + 1], z[b + 2], z[b + 3], z[b + 4], z[b + 5]];
        build_rad6_slab(&grp, 0, &mut slab);
        let mut r = 0usize;
        while r < 6 {
            h_lat[r] += slab[(v_codes[r][g] & 0x3F) as usize];
            r += 1;
        }
        g += 1;
    }

    build_rad6_slab(&h_lat, 3, &mut slab);
    let mut row = 0usize;
    while row < 64 {
        out64[row] = slab[(u_codes[row] & 0x3F) as usize];
        row += 1;
    }
}

/// Baseline `LittleBit-2` style runtime sign-unpack + dense binary MAC over the same `FWHT-64`
/// rotated activation vector (`64 x 60` runtime unpack + multiply-accumulate; `Theorem 94` parity check).
pub fn eval_littlebit2_6col_unpack_mac_64(
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) {
    let mut z = [0i32; 64];
    let mut c = 0usize;
    while c < 64 {
        z[c] = acts64[c] as i32;
        c += 1;
    }
    fwht64_inplace_i32(&mut z);

    let mut v_dense = [[0i32; 60]; 6];
    let mut r = 0usize;
    while r < 6 {
        let mut g = 0usize;
        while g < 10 {
            let code = v_codes[r][g] & 0x3F;
            let mut k = 0usize;
            while k < 6 {
                v_dense[r][g * 6 + k] = if ((code >> k) & 1) != 0 { -1 } else { 1 };
                k += 1;
            }
            g += 1;
        }
        r += 1;
    }

    let mut row = 0usize;
    while row < 64 {
        let u_mask = u_codes[row] & 0x3F;
        let mut acc = 0i32;
        let mut col = 0usize;
        while col < 60 {
            let mut w_rc = 0i32;
            let mut rk = 0usize;
            while rk < 6 {
                let u_sign = if ((u_mask >> rk) & 1) != 0 { -1i32 } else { 1i32 };
                w_rc += u_sign * v_dense[rk][col];
                rk += 1;
            }
            acc += w_rc * z[col];
            col += 1;
        }
        out64[row] = acc >> 3;
        row += 1;
    }
}
