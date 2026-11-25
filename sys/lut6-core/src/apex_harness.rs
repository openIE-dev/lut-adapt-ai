//! `#![no_std]` OpenIE Ecosystem Apex Harness Gate (`Claims LA-45..LA-48`):
//! Zero-FLOP `System-0` `LUT6` policy evaluation gated simultaneously by:
//!   1. `DES-10` Digital Enzymatic Synthesis handoff attestation (`expected_handoff_tag == actual_tag`),
//!   2. `openie-cad` Level-1 Interface Physics bounds (`z0_milliohm ∈ [45_000, 55_000]`, `tj_deci_c <= 850`),
//!   3. `MathGround` Simplex `Torricelli` Forward-Reachable `CBF` (`v_mm_s^2 <= 2 * a_max_mm_s2 * d_obs_mm`),
//!   4. `RCE` operational cash/inventory reserve nonnegativity (`cash_cents > 0 && stock_units > 0`).

use crate::eval_lut6_bitsliced64;

/// Compact state envelope for a coupled `H = 256`-step physical + economic + safety step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApexStepEnvelope {
    pub percept6: u8,
    pub handoff_tag16: u16,
    pub expected_tag16: u16,
    pub z0_milliohm: u32,
    pub tj_deci_c: u16,
    pub v_mm_s: u32,
    pub a_max_mm_s2: u32,
    pub d_obs_mm: u32,
    pub cash_cents: i32,
    pub stock_units: i32,
}

/// Verified outcome of the `#![no_std]` `OpenIE-Apex-Harness` `System-0` step gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApexGateOutcome {
    /// Certified zero-FLOP `System-0` `LUT6` control action (`action_code ∈ 1..=15`) with all physical/safety barriers satisfied.
    CertifiedSystem0 { action_code: u8 },
    /// `DES-10` attestation mismatch (`expected_tag16 != handoff_tag16`): halt enzymatic drift and re-verify!
    EnzymaticAttestationTrap { xor_tag: u16 },
    /// `openie-cad` DRC/Thermal or `MathGround` Torricelli `CBF` or `RCE` reserve barrier tripped: engage Simplex baseline + Z3!
    SimplexBarrierShield { violation_mask: u8 },
    /// State valid, all invariants clean, but `LUT6` slot uncertified: wake `AlphaEvolve + Z3 + Frontier` planner.
    WakeApexSolver,
}

/// Checks all four deterministic ecosystem invariants in integer arithmetic (`0` floats, `0` heap):
///   - `bit 0`: `DES-10` enzymatic handoff mismatch (`handoff_tag16 != expected_tag16`)
///   - `bit 1`: `openie-cad` PCB microstrip impedance out of `[45.0 Ω, 55.0 Ω]` (`45_000..=55_000 mΩ`)
///   - `bit 2`: `openie-cad` lumped junction thermal limit exceeded (`tj_deci_c > 850` = `85.0 °C`)
///   - `bit 3`: `MathGround` Torricelli braking `CBF` violated (`v^2 > 2 * a_max * d_obs`)
///   - `bit 4`: `RCE` operational solvency/inventory floor violated (`cash_cents <= 0 || stock_units <= 0`)
#[inline]
pub const fn check_apex_invariants(env: &ApexStepEnvelope) -> u8 {
    let mut mask = 0u8;
    if env.handoff_tag16 != env.expected_tag16 {
        mask |= 1 << 0;
    }
    if env.z0_milliohm < 45_000 || env.z0_milliohm > 55_000 {
        mask |= 1 << 1;
    }
    if env.tj_deci_c > 850 {
        mask |= 1 << 2;
    }
    let lhs = (env.v_mm_s as u64) * (env.v_mm_s as u64);
    let rhs = 2u64 * (env.a_max_mm_s2 as u64) * (env.d_obs_mm as u64);
    if lhs > rhs {
        mask |= 1 << 3;
    }
    if env.cash_cents <= 0 || env.stock_units <= 0 {
        mask |= 1 << 4;
    }
    mask
}

/// Evaluates one `OpenIE-Apex-Harness` `System-0` `LUT6` control step across `4` `LUT6` action bitplanes
/// plus a `64`-bit certification mask, strictly gated behind `check_apex_invariants`.
#[inline]
pub const fn eval_apex_enzymatic_step_lut6(
    env: &ApexStepEnvelope,
    action_masks4: &[u64; 4],
    certified_mask: u64,
) -> ApexGateOutcome {
    let inv_mask = check_apex_invariants(env);
    if (inv_mask & 1) != 0 {
        return ApexGateOutcome::EnzymaticAttestationTrap {
            xor_tag: env.handoff_tag16 ^ env.expected_tag16,
        };
    }
    if inv_mask != 0 {
        return ApexGateOutcome::SimplexBarrierShield {
            violation_mask: inv_mask,
        };
    }
    let idx = (env.percept6 & 0x3F) as u32;
    if ((certified_mask >> idx) & 1) == 0 {
        return ApexGateOutcome::WakeApexSolver;
    }
    let x0 = if (idx & 1) != 0 { u64::MAX } else { 0 };
    let x1 = if (idx & 2) != 0 { u64::MAX } else { 0 };
    let x2 = if (idx & 4) != 0 { u64::MAX } else { 0 };
    let x3 = if (idx & 8) != 0 { u64::MAX } else { 0 };
    let x4 = if (idx & 16) != 0 { u64::MAX } else { 0 };
    let x5 = if (idx & 32) != 0 { u64::MAX } else { 0 };

    let b0 = (eval_lut6_bitsliced64(action_masks4[0], x0, x1, x2, x3, x4, x5) & 1) as u8;
    let b1 = ((eval_lut6_bitsliced64(action_masks4[1], x0, x1, x2, x3, x4, x5) & 1) as u8) << 1;
    let b2 = ((eval_lut6_bitsliced64(action_masks4[2], x0, x1, x2, x3, x4, x5) & 1) as u8) << 2;
    let b3 = ((eval_lut6_bitsliced64(action_masks4[3], x0, x1, x2, x3, x4, x5) & 1) as u8) << 3;
    let code = b0 | b1 | b2 | b3;
    if code == 0 {
        ApexGateOutcome::WakeApexSolver
    } else {
        ApexGateOutcome::CertifiedSystem0 { action_code: code }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apex_invariants_and_system0_gate() {
        let masks4 = [
            0xAAAA_AAAA_AAAA_AAAA,
            0xCCCC_CCCC_CCCC_CCCC,
            0xF0F0_F0F0_F0F0_F0F0,
            0,
        ];
        let clean = ApexStepEnvelope {
            percept6: 5,
            handoff_tag16: 0x42A1,
            expected_tag16: 0x42A1,
            z0_milliohm: 50_000,
            tj_deci_c: 620,
            v_mm_s: 1200,
            a_max_mm_s2: 3000,
            d_obs_mm: 300, // 2 * 3000 * 300 = 1,800,000 >= 1,440,000
            cash_cents: 12_500,
            stock_units: 18,
        };
        assert_eq!(check_apex_invariants(&clean), 0);
        assert_eq!(
            eval_apex_enzymatic_step_lut6(&clean, &masks4, u64::MAX),
            ApexGateOutcome::CertifiedSystem0 { action_code: 5 }
        );

        // DES-10 handoff mismatch must trap before emitting any action
        let mut bad_tag = clean;
        bad_tag.handoff_tag16 = 0x42A0;
        assert_eq!(
            eval_apex_enzymatic_step_lut6(&bad_tag, &masks4, u64::MAX),
            ApexGateOutcome::EnzymaticAttestationTrap { xor_tag: 1 }
        );

        // Torricelli CBF violation (`v^2 > 2*a*d`) + thermal violation (`> 85.0 C`) must engage Simplex shield
        let mut unsafe_phys = clean;
        unsafe_phys.v_mm_s = 2000; // 4,000,000 > 1,800,000
        unsafe_phys.tj_deci_c = 890;
        assert_eq!(
            eval_apex_enzymatic_step_lut6(&unsafe_phys, &masks4, u64::MAX),
            ApexGateOutcome::SimplexBarrierShield {
                violation_mask: (1 << 2) | (1 << 3)
            }
        );
    }
}
