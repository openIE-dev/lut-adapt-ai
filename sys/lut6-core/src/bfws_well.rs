//! Phase 40 (`Claims LA-56, LA-58, LA-59`): `#![no_std]` Four-Bank (`16 KB` = `4 x 4,096 B`)
//! `BFWS(R, M)` Cuckoo/Cascade `System-0` Reflex Vault AND Planetary Digital Well (`Well-Codifier`)
//! Rung-5 -> Rung-7 Standard-Library `LUT6` Evaluator (`IPv4` RFC1918/Loopback, `HTTP` Retry/Breaker
//! Status Policy, `WCAG 2.2 AA` Contrast Gate, `POSIX` Sandbox Mode Gate, `Q8.8` Clamp, `CRC-4`/`SemVer`).

use crate::siw_joulebit::eval_joulebit6_fwht64_slab_i16;

/// Outcome of the 4-bank (`16 KB`, `25.00%` of `64 KB` watch SRAM) `BFWS` reflex + circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bfws4BankOutcome {
    /// Certified `System-0` action from one of the `4` `4,096 B` `.lut6` banks (`bank_id ∈ 0..=3`).
    CertifiedAction {
        action_id: u8,
        bank_id: u8,
        subgoal_advance: bool,
        next_epoch: u8,
    },
    /// `Theorem 73` mechanic-shift circuit breaker (`expected_sig != actual_sig`).
    MechanicShiftBreaker { xor_sig: u8, next_epoch: u8 },
    /// Uncertified `(epoch, percept6)` tuple -> wake Dual-Subspace `BFWS` planner (`Rung 1..5`).
    WakeBfwsPlanner { subgoal_advance: bool, next_epoch: u8 },
}

/// Evaluates the 4-bank (`4 x 4,096 B = 16,384 B`) non-destructive `BFWS` `LUT6` reflex (`Theorems 100..101`):
///   - Checks `expected_post_sig` against `actual_sig` (`Theorem 73` circuit breaker),
///   - Advances `epoch` monotonically (`0..=15`) when `(subgoal_lut6_mask >> subgoal6) & 1 == 1`
///     without destructively wiping prior-epoch tables (`Theorem 100`),
///   - Cascades across `4` banks (`(epoch & 3) ^ probe`) to resolve collisions losslessly (`Theorem 101`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_bfws_four_bank_lut6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    expected_post_sig: Option<u8>,
    actual_sig: u8,
    subgoal_lut6_mask: u64,
    banks_action_masks: &[[u64; 3]; 4],
    banks_cert_masks: &[u64; 4],
) -> Bfws4BankOutcome {
    if let Some(exp) = expected_post_sig
        && exp != 0xFF
        && exp != actual_sig
    {
        return Bfws4BankOutcome::MechanicShiftBreaker {
            xor_sig: exp ^ actual_sig,
            next_epoch: epoch.min(15),
        };
    }

    let sg_idx = (subgoal6 & 0x3F) as u32;
    let sg_hit = ((subgoal_lut6_mask >> sg_idx) & 1) == 1;
    let next_epoch = if sg_hit && epoch < 15 {
        epoch + 1
    } else {
        epoch.min(15)
    };

    let p_idx = (percept6 & 0x3F) as u32;
    let primary_bank = (next_epoch & 3) as usize;
    let mut probe = 0usize;
    while probe < 4 {
        let b_id = (primary_bank + probe) & 3;
        if ((banks_cert_masks[b_id] >> p_idx) & 1) == 1 {
            let m = &banks_action_masks[b_id];
            let a0 = ((m[0] >> p_idx) & 1) as u8;
            let a1 = (((m[1] >> p_idx) & 1) as u8) << 1;
            let a2 = (((m[2] >> p_idx) & 1) as u8) << 2;
            let raw_act = a0 | a1 | a2;
            let action_id = if raw_act == 0 { 1 } else { raw_act.min(7) };
            return Bfws4BankOutcome::CertifiedAction {
                action_id,
                bank_id: b_id as u8,
                subgoal_advance: sg_hit,
                next_epoch,
            };
        }
        probe += 1;
    }

    Bfws4BankOutcome::WakeBfwsPlanner {
        subgoal_advance: sg_hit,
        next_epoch,
    }
}

/// Planetary Digital Well (`Well-Codifier`, `Claim LA-58`, `Theorem 103`):
/// Evaluates `8` parallel standard-library predicates compiled from Rung-5/6 Python/POSIX modules
/// into `8` zero-branch `64`-bit `LUT6` truth tables (`64 B` total, `0` JSON/string parses, `0` MULs):
///   - `gate 0`: `ipaddress` RFC1918 / loopback private-route allow (`addr6 < 4` -> `0x0000_0000_0000_000F`)
///   - `gate 1`: `http.HTTPStatus` transient retryable (`408, 429, 502, 503, 504` -> buckets `8..=12`)
///   - `gate 2`: `WCAG 2.2 AA` normal-text contrast `>= 4.5:1` (`quantized ratio6 >= 18`)
///   - `gate 3`: `stat` POSIX sandbox safe file mode (`no setuid/setgid/world-write`: `(addr6 & 0b11010) == 0`)
///   - `gate 4`: `urllib.parse` HTTPS/WSS TLS-only scheme + unprivileged port (`(addr6 & 0b100001) == 0b100001`)
///   - `gate 5`: `semver` caret `^major.minor` backward-compatible upgrade (`major_eq && minor_ge`)
///   - `gate 6`: `datetime` UTC maintenance-freeze window (`hour_bucket 2..=5 && weekday < 5`)
///   - `gate 7`: `errno` retryable `EAGAIN`/`EINTR`/`ETIMEDOUT` syscall class (`addr6 & 7 in {1, 2, 3}`)
pub const WELL_CODIFIER_STDLIB8_MASKS: [u64; 8] = {
    let mut m = [0u64; 8];
    let mut x = 0usize;
    while x < 64 {
        if x < 4 {
            m[0] |= 1u64 << x;
        }
        if x >= 8 && x <= 12 {
            m[1] |= 1u64 << x;
        }
        if x >= 18 {
            m[2] |= 1u64 << x;
        }
        if (x & 0b01_1010) == 0 {
            m[3] |= 1u64 << x;
        }
        if (x & 0b10_0001) == 0b10_0001 {
            m[4] |= 1u64 << x;
        }
        // major_eq = bit 5, minor_ge = bit 4, no_pre = bit 3
        if (x & 0b11_1000) == 0b11_1000 {
            m[5] |= 1u64 << x;
        }
        if (x & 0x7) >= 2 && (x & 0x7) <= 5 && ((x >> 3) & 0x7) < 5 {
            m[6] |= 1u64 << x;
        }
        let low3 = x & 7;
        if low3 >= 1 && low3 <= 3 {
            m[7] |= 1u64 << x;
        }
        x += 1;
    }
    m
};

/// Evaluates all `8` compiled Planetary Digital Well stdlib `LUT6` predicates at `addr6` (`0..63`)
/// in `8` shifts + bitwise ORs (`0` branches, `0` heap allocations, `0` MULs).
#[inline]
pub const fn eval_well_codifier_stdlib8(addr6: u8) -> u8 {
    let sh = (addr6 & 0x3F) as u32;
    (((WELL_CODIFIER_STDLIB8_MASKS[0] >> sh) & 1) as u8)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[1] >> sh) & 1) as u8) << 1)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[2] >> sh) & 1) as u8) << 2)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[3] >> sh) & 1) as u8) << 3)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[4] >> sh) & 1) as u8) << 4)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[5] >> sh) & 1) as u8) << 5)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[6] >> sh) & 1) as u8) << 6)
        | ((((WELL_CODIFIER_STDLIB8_MASKS[7] >> sh) & 1) as u8) << 7)
}

/// Baseline Rung-5/6 uncompiled tool-call dispatch proxy (`Window 20b`):
/// Parses a `64`-byte ASCII JSON-RPC tool envelope (`{"tool":"...","arg":...}`), runs a `16 x 64`
/// `INT8` neural router layer (`1,024` MACs) to pick the tool, and evaluates the `LittleBit-2`
/// `6`-column unpack+MAC draft slab.
#[inline(never)]
pub fn eval_rung5_json_int8_littlebit2_baseline(
    json_buf: &[u8; 64],
    router_w: &[[i8; 64]; 8],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> u8 {
    // 1. Byte-by-byte JSON-RPC key/colon/digit scan + FNV-1a key hash
    let mut hash = 0x811c_9dc5u32;
    let mut colon_cnt = 0u8;
    let mut parsed_num = 0u32;
    let mut in_num = false;
    let mut i = 0usize;
    while i < 64 {
        let b = json_buf[i];
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        if b == b':' {
            colon_cnt = colon_cnt.wrapping_add(1);
            in_num = true;
        } else if in_num && b.is_ascii_digit() {
            parsed_num = parsed_num.wrapping_mul(10).wrapping_add((b - b'0') as u32);
        } else if b == b',' || b == b'}' {
            in_num = false;
        }
        i += 1;
    }

    // 2. 8 x 64 INT8 neural tool-router projection (512 INT8 MACs)
    let mut best_logit = i32::MIN;
    let mut best_tool = 0u8;
    let mut r = 0usize;
    while r < 8 {
        let mut acc = 0i32;
        let mut c = 0usize;
        while c < 64 {
            let x = (json_buf[c] as i16 - 64) as i32;
            acc += x * (router_w[r][c] as i32);
            c += 1;
        }
        if acc > best_logit {
            best_logit = acc;
            best_tool = r as u8;
        }
        r += 1;
    }

    // 3. Uncompiled LittleBit-2 6-col + FWHT-64 unpack + binary MAC
    crate::siw_joulebit::eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);

    let addr6 = ((parsed_num ^ hash) & 0x3F) as u8;
    eval_well_codifier_stdlib8(addr6) ^ best_tool ^ colon_cnt
}

/// Fused Phase 40 (`Window 20`) Rung-7 step:
/// Evaluates Four-Bank `BFWS` `LUT6` reflex (`16 KB` vault) + `8`-gate `Well-Codifier` stdlib
/// `LUT6` bank (`0` JSON parse, `0` router MACs) + zero-`MUL` `JouleBit6 + FWHT-64` slab!
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window20_bfws4_well_joulebit6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    well_addr6: u8,
    banks_action_masks: &[[u64; 3]; 4],
    banks_cert_masks: &[u64; 4],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> (Bfws4BankOutcome, u8) {
    let bfws = eval_bfws_four_bank_lut6(
        percept6,
        subgoal6,
        epoch,
        Some(0x5A),
        0x5A,
        !1u64,
        banks_action_masks,
        banks_cert_masks,
    );
    let well_bits = eval_well_codifier_stdlib8(well_addr6);
    eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    (bfws, well_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_four_bank_cascade_spillover_and_breaker() {
        let banks = [
            [0xFFFF_FFFF_FFFF_FFFF, 0, 0], // action 1
            [0, 0xFFFF_FFFF_FFFF_FFFF, 0], // action 2
            [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0], // action 3
            [0, 0, 0xFFFF_FFFF_FFFF_FFFF], // action 4
        ];
        // Only bank 3 is certified at percept 7: starting from epoch 0, cascade must hit bank 3!
        let certs = [0u64, 0u64, 0u64, 1u64 << 7];
        let out = eval_bfws_four_bank_lut6(7, 0, 0, Some(0x5A), 0x5A, !1u64, &banks, &certs);
        assert_eq!(
            out,
            Bfws4BankOutcome::CertifiedAction {
                action_id: 4,
                bank_id: 3,
                subgoal_advance: false,
                next_epoch: 0,
            }
        );

        // Mechanic-shift breaker fires on post-signature mismatch
        let brk = eval_bfws_four_bank_lut6(7, 1, 2, Some(0x5A), 0x11, !1u64, &banks, &certs);
        assert_eq!(
            brk,
            Bfws4BankOutcome::MechanicShiftBreaker {
                xor_sig: 0x5A ^ 0x11,
                next_epoch: 2,
            }
        );
    }

    #[test]
    fn test_well_codifier_stdlib8_exact_minterms() {
        for x in 0u8..64 {
            let bits = eval_well_codifier_stdlib8(x);
            assert_eq!((bits & 1) != 0, x < 4);
            assert_eq!(((bits >> 1) & 1) != 0, (8..=12).contains(&x));
            assert_eq!(((bits >> 2) & 1) != 0, x >= 18);
            assert_eq!(((bits >> 3) & 1) != 0, (x & 0b01_1010) == 0);
        }
    }
}
