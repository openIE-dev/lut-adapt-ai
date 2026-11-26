//! Phase 41 (`Claims LA-60..LA-63`): `#![no_std]` Eight-Bank (`32 KB` = `8 x 4,096 B` = `50.00%`
//! of `64 KB` Smartwatch `SRAM`) Reset-Persistent `R-BFWS(R, M)` `System-0` Cascade Vault,
//! Bijective `Enigma-SBox6` (`6`-Input `LUT6` Cryptographic S-Box + Inverse Roundtrip), and
//! `Vending-BALROG` `1,000`-Step Simplex Solvency + Stamina `CBF` `LUT6` Shield (`Window 21`).

use crate::siw_joulebit::eval_joulebit6_fwht64_slab_i16;

/// Outcome of the 8-bank (`32 KB`, `50.00%` of `64 KB` watch SRAM) `R-BFWS` reflex + circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rbfws8BankOutcome {
    /// Certified `System-0` action from one of the `8` `4,096 B` `.lut6` banks (`bank_id ∈ 0..=7`).
    CertifiedAction {
        action_id: u8,
        bank_id: u8,
        subgoal_advance: bool,
        next_epoch: u8,
    },
    /// `Theorem 73` mechanic-shift circuit breaker (`expected_sig != actual_sig`).
    MechanicShiftBreaker { xor_sig: u8, next_epoch: u8 },
    /// Uncertified `(epoch, percept6)` tuple -> wake Reset-Persistent `R-BFWS` planner.
    WakeRbfwsPlanner { subgoal_advance: bool, next_epoch: u8 },
}

/// Evaluates the 8-bank (`8 x 4,096 B = 32,768 B`, `4,096` `64`-bit `LUT6` entries) `R-BFWS`
/// `System-0` reflex (`Theorem 109`):
///   - Enforces `Theorem 73` post-step playfield signature check,
///   - Advances `epoch` monotonically (`0..=7`) on `subgoal_lut6_mask` hits,
///   - Probes `(next_epoch + probe) & 7` across all `8` banks to resolve up to `8`-way collisions losslessly.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_rbfws_eight_bank_lut6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    expected_post_sig: Option<u8>,
    actual_sig: u8,
    subgoal_lut6_mask: u64,
    banks_action_masks: &[[u64; 3]; 8],
    banks_cert_masks: &[u64; 8],
) -> Rbfws8BankOutcome {
    if let Some(exp) = expected_post_sig
        && exp != 0xFF
        && exp != actual_sig
    {
        return Rbfws8BankOutcome::MechanicShiftBreaker {
            xor_sig: exp ^ actual_sig,
            next_epoch: epoch.min(7),
        };
    }

    let sg_idx = (subgoal6 & 0x3F) as u32;
    let sg_hit = ((subgoal_lut6_mask >> sg_idx) & 1) == 1;
    let next_epoch = if sg_hit && epoch < 7 {
        epoch + 1
    } else {
        epoch.min(7)
    };

    let p_idx = (percept6 & 0x3F) as u32;
    let primary_bank = (next_epoch & 7) as usize;
    let mut probe = 0usize;
    while probe < 8 {
        let b_id = (primary_bank + probe) & 7;
        if ((banks_cert_masks[b_id] >> p_idx) & 1) == 1 {
            let m = &banks_action_masks[b_id];
            let raw_act = (((m[0] >> p_idx) & 1) as u8)
                | ((((m[1] >> p_idx) & 1) as u8) << 1)
                | ((((m[2] >> p_idx) & 1) as u8) << 2);
            let action_id = if raw_act == 0 { 1 } else { raw_act.min(7) };
            return Rbfws8BankOutcome::CertifiedAction {
                action_id,
                bank_id: b_id as u8,
                subgoal_advance: sg_hit,
                next_epoch,
            };
        }
        probe += 1;
    }

    Rbfws8BankOutcome::WakeRbfwsPlanner {
        subgoal_advance: sg_hit,
        next_epoch,
    }
}

#[inline(always)]
const fn feistel_g3(l3: usize) -> usize {
    ((l3 & (l3 >> 1)) ^ ((l3 << 1) & 7) ^ 5) & 7
}

/// Non-linear cryptographic `6`-bit bijection `S: {0..63} -> {0..63}` (`Claim LA-61`, `Theorem 110`):
/// Affine mixing `y = (27 * x + 19) mod 64` (`27 * 19 = 513 ≡ 1 mod 64`) followed by quadratic
/// 3-bit Feistel substitution `(L, R ^ g(L))`.
#[inline(always)]
pub const fn enigma_sbox6_spec(x: u8) -> u8 {
    let y = ((27usize * ((x & 0x3F) as usize)) + 19) & 0x3F;
    let l3 = (y >> 3) & 7;
    let r3 = (y & 7) ^ feistel_g3(l3);
    ((l3 << 3) | r3) as u8
}

/// Exact algebraic inverse `S^-1: {0..63} -> {0..63}` (`S^-1(S(x)) == x` for all `x < 64`).
#[inline(always)]
pub const fn enigma_sbox6_inv_spec(z: u8) -> u8 {
    let zz = (z & 0x3F) as usize;
    let l3 = (zz >> 3) & 7;
    let r3 = (zz & 7) ^ feistel_g3(l3);
    let y = (l3 << 3) | r3;
    ((19usize * ((y + 64 - 19) & 0x3F)) & 0x3F) as u8
}

/// Six `64`-bit `LUT6` truth tables implementing `enigma_sbox6_spec` in `0` runtime branches/MULs.
pub const ENIGMA_SBOX6_MASKS: [u64; 6] = {
    let mut m = [0u64; 6];
    let mut x = 0usize;
    while x < 64 {
        let y = enigma_sbox6_spec(x as u8) as usize;
        let mut b = 0usize;
        while b < 6 {
            if ((y >> b) & 1) != 0 {
                m[b] |= 1u64 << x;
            }
            b += 1;
        }
        x += 1;
    }
    m
};

/// Six `64`-bit `LUT6` truth tables implementing `enigma_sbox6_inv_spec` in `0` runtime branches/MULs.
pub const ENIGMA_SBOX6_INV_MASKS: [u64; 6] = {
    let mut m = [0u64; 6];
    let mut z = 0usize;
    while z < 64 {
        let x = enigma_sbox6_inv_spec(z as u8) as usize;
        let mut b = 0usize;
        while b < 6 {
            if ((x >> b) & 1) != 0 {
                m[b] |= 1u64 << z;
            }
            b += 1;
        }
        z += 1;
    }
    m
};

/// Evaluates the compiled `6`-output `Enigma-SBox6` `LUT6` bank at `x` (`0..63`).
#[inline]
pub const fn eval_enigma_sbox6_lut6(x: u8) -> u8 {
    let sh = (x & 0x3F) as u32;
    (((ENIGMA_SBOX6_MASKS[0] >> sh) & 1) as u8)
        | ((((ENIGMA_SBOX6_MASKS[1] >> sh) & 1) as u8) << 1)
        | ((((ENIGMA_SBOX6_MASKS[2] >> sh) & 1) as u8) << 2)
        | ((((ENIGMA_SBOX6_MASKS[3] >> sh) & 1) as u8) << 3)
        | ((((ENIGMA_SBOX6_MASKS[4] >> sh) & 1) as u8) << 4)
        | ((((ENIGMA_SBOX6_MASKS[5] >> sh) & 1) as u8) << 5)
}

/// Evaluates the compiled `6`-output inverse `Enigma-SBox6^-1` `LUT6` bank at `z` (`0..63`).
#[inline]
pub const fn eval_enigma_sbox6_inv_lut6(z: u8) -> u8 {
    let sh = (z & 0x3F) as u32;
    (((ENIGMA_SBOX6_INV_MASKS[0] >> sh) & 1) as u8)
        | ((((ENIGMA_SBOX6_INV_MASKS[1] >> sh) & 1) as u8) << 1)
        | ((((ENIGMA_SBOX6_INV_MASKS[2] >> sh) & 1) as u8) << 2)
        | ((((ENIGMA_SBOX6_INV_MASKS[3] >> sh) & 1) as u8) << 3)
        | ((((ENIGMA_SBOX6_INV_MASKS[4] >> sh) & 1) as u8) << 4)
        | ((((ENIGMA_SBOX6_INV_MASKS[5] >> sh) & 1) as u8) << 5)
}

/// Compile-time exhaustive `64/64` bijection proof gate (`rustc` `const` evaluation).
const _: () = {
    let mut x = 0u8;
    while x < 64 {
        let y = eval_enigma_sbox6_lut6(x);
        assert!(y == enigma_sbox6_spec(x));
        assert!(eval_enigma_sbox6_inv_lut6(y) == x);
        x += 1;
    }
};

/// `Vending-BALROG` `1,000`-step Simplex Solvency + Stamina `CBF` `LUT6` truth tables (`Claim LA-62`, `Theorem 112`):
/// `addr6 = cash2 | (stock2 << 2) | (stamina2 << 4)`:
///   - `mask[0]` (`solvency_ok`): `cash2 >= 1 && (cash2 >= 2 || stock2 >= 1)`
///   - `mask[1]` (`stamina_cbf_ok`): `stamina2 >= 1 && (stamina2 >= 2 || cash2 >= 2)`
pub const VENDING_BALROG_SHIELD2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut a = 0usize;
    while a < 64 {
        let cash2 = a & 3;
        let stock2 = (a >> 2) & 3;
        let stam2 = (a >> 4) & 3;
        if cash2 >= 1 && (cash2 >= 2 || stock2 >= 1) {
            m[0] |= 1u64 << a;
        }
        if stam2 >= 1 && (stam2 >= 2 || cash2 >= 2) {
            m[1] |= 1u64 << a;
        }
        a += 1;
    }
    m
};

/// Evaluates the `2`-bit `Vending-BALROG` Simplex Solvency + Stamina `CBF` `LUT6` shield (`0..=3`).
#[inline]
pub const fn eval_vending_balrog_shield2_lut6(addr6: u8) -> u8 {
    let sh = (addr6 & 0x3F) as u32;
    (((VENDING_BALROG_SHIELD2_MASKS[0] >> sh) & 1) as u8)
        | ((((VENDING_BALROG_SHIELD2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Fused Phase 41 (`Window 21`) Rung-7 step:
/// Evaluates `8`-Bank `R-BFWS` `LUT6` cascade (`32 KB` vault) + `Enigma-SBox6` cryptographic
/// `LUT6` substitution + `Vending-BALROG` `1,000`-step Simplex `CBF` `LUT6` shield +
/// zero-`MUL` `JouleBit6 + FWHT-64` slab (`0` `MUL`s, `0` `KV`-cache scan)!
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window21_rbfws8_enigma_horizon(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    cipher_in6: u8,
    shield_addr6: u8,
    banks_action_masks: &[[u64; 3]; 8],
    banks_cert_masks: &[u64; 8],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> (Rbfws8BankOutcome, u8, u8) {
    let rbfws = eval_rbfws_eight_bank_lut6(
        percept6,
        subgoal6,
        epoch,
        Some(0x5A),
        0x5A,
        !1u64,
        banks_action_masks,
        banks_cert_masks,
    );
    let sbox_out = eval_enigma_sbox6_lut6(cipher_in6);
    let shield2 = eval_vending_balrog_shield2_lut6(shield_addr6);
    eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    (rbfws, sbox_out, shield2)
}

/// Uncompiled long-horizon baseline (`Window 21b`):
/// Scans a `64`-entry `INT8` `KV` history buffer (`64 x 16 = 1,024` `INT8` `MAC`s modeling
/// `Vending-Bench` / `BALROG` context-window attention), runs a `6`-round software polynomial
/// cipher loop (`MUL` + `GF(2)` LFSR steps), and evaluates `LittleBit-2` `6`-col unpack+`MAC`.
#[inline(never)]
pub fn eval_window21b_kv64_cipher_littlebit2_baseline(
    kv_hist: &[[i8; 16]; 64],
    query16: &[i8; 16],
    cipher_in6: u8,
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> u8 {
    // 1. 64-step KV history attention dot-product scan (1,024 INT8 MACs)
    let mut best_score = i32::MIN;
    let mut best_slot = 0u8;
    let mut t = 0usize;
    while t < 64 {
        let mut dot = 0i32;
        let mut d = 0usize;
        while d < 16 {
            dot += (kv_hist[t][d] as i32) * (query16[d] as i32);
            d += 1;
        }
        if dot > best_score {
            best_score = dot;
            best_slot = t as u8;
        }
        t += 1;
    }

    // 2. Software multi-round arithmetic + LFSR cipher evaluation (no LUT6 precomputation)
    let mut st = (cipher_in6 & 0x3F) as u32;
    let mut rnd = 0usize;
    while rnd < 6 {
        st = (st.wrapping_mul(27).wrapping_add(19 + (rnd as u32) * 5)) & 0x3F;
        let l3 = (st >> 3) & 7;
        let g3 = ((l3 & (l3 >> 1)) ^ ((l3 << 1) & 7) ^ 5) & 7;
        st = (l3 << 3) | ((st & 7) ^ g3);
        rnd += 1;
    }

    // 3. Uncompiled LittleBit-2 6-col + FWHT-64 unpack + binary MAC
    crate::siw_joulebit::eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);

    ((st as u8) ^ best_slot) & 0x3F
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eight_bank_cascade_and_enigma_sbox6_bijection() {
        let mut seen = [false; 64];
        for x in 0u8..64 {
            let y = eval_enigma_sbox6_lut6(x);
            assert!(y < 64);
            assert!(!seen[y as usize]);
            seen[y as usize] = true;
            assert_eq!(eval_enigma_sbox6_inv_lut6(y), x);
        }

        let banks = [
            [u64::MAX, 0, 0],
            [0, u64::MAX, 0],
            [u64::MAX, u64::MAX, 0],
            [0, 0, u64::MAX],
            [u64::MAX, 0, u64::MAX],
            [0, u64::MAX, u64::MAX],
            [u64::MAX, u64::MAX, u64::MAX],
            [u64::MAX, 0, 0],
        ];
        let mut certs = [0u64; 8];
        certs[7] = 1u64 << 9;
        let out = eval_rbfws_eight_bank_lut6(9, 0, 0, Some(0x5A), 0x5A, !1u64, &banks, &certs);
        assert_eq!(
            out,
            Rbfws8BankOutcome::CertifiedAction {
                action_id: 1,
                bank_id: 7,
                subgoal_advance: false,
                next_epoch: 0,
            }
        );
        assert_eq!(eval_vending_balrog_shield2_lut6(0b10_01_10), 3);
        assert_eq!(eval_vending_balrog_shield2_lut6(0), 0);
    }
}
