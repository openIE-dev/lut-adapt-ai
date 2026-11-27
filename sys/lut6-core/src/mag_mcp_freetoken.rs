//! Phase 44 (`Claims LA-72..LA-75`): `#![no_std]` Native Rust `Magnitude 0.3.5` `MCPMark`
//! Copy-on-Write (`CoW`) CRUD & `CogLoadBench` Prefill Guard (`MCPMark-CoW-Guard LUT6`),
//! Bandwidth-Adaptive `BA-Tiered-Expert-AirMoL` CPU-GPU Co-Scheduler (`FreeToken` / `KTransformers`
//! Strict Generalization), Fourteen-Bank `56 KB` (`87.50%` Watch `SRAM`) Vault Cascade, and
//! Cross-Silicon `Window 24` vs. `Window 24b` Telemetry Kernel.

use crate::siw_joulebit::{eval_joulebit6_fwht64_slab_i16, eval_littlebit2_6col_unpack_mac_64};

/// Exact `6`-input `LUT6` specification for Stateful `MCPMark` Copy-on-Write (`CoW`) CRUD Sandbox
/// & `CogLoadBench` Linux Vulkan `<= 128`-Token Prefill Guard (`Claims LA-72 & LA-73`, `Theorems 131-133`):
///
/// `mcp_addr6 = schema_ok1 | (acid_bal_ok1 << 1) | (acl_safe1 << 2) | (idemp_new1 << 3) | (prefill_le128_1 << 4) | (crystal_hit1 << 5)`:
///   - `guard2 == 0` (`0b00`): `CoW` Shadow Rollback + Reject (`!schema_ok || !acid_bal_ok || !acl_safe` -> `0` live corruption!),
///   - `guard2 == 1` (`0b01`): Idempotent Replay Dedup (`!idemp_new` -> return sealed receipt, `0` double-debit!),
///   - `guard2 == 2` (`0b10`): `CogLoad-Filter` + `EngramSpanLUT6` `<= 120`-Token Windowing (`!prefill_le128` -> prevents `Seismic` Vulkan slab panic!),
///   - `guard2 == 3` (`0b11`): Atomic `CoW` Commit + `Rung-0 .lut6` Crystallize (`crystal_hit` fast-path or verified `Wave 1`).
#[inline(always)]
pub const fn mcpmark_cow_guard2_spec(mcp_addr6: u8) -> u8 {
    let a = (mcp_addr6 & 0x3F) as usize;
    let schema_ok = (a & 1) != 0;
    let acid_bal_ok = ((a >> 1) & 1) != 0;
    let acl_safe = ((a >> 2) & 1) != 0;
    let idemp_new = ((a >> 3) & 1) != 0;
    let prefill_le128 = ((a >> 4) & 1) != 0;
    let crystal_hit = ((a >> 5) & 1) != 0;

    if !schema_ok || !acid_bal_ok || !acl_safe {
        0
    } else if !idemp_new {
        1
    } else if crystal_hit {
        3
    } else if !prefill_le128 {
        2
    } else {
        3
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `mcpmark_cow_guard2_spec` with `0` branches.
pub const MCPMARK_COW_GUARD2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let g = mcpmark_cow_guard2_spec(x as u8) as usize;
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

/// Evaluates the compiled `2`-bit `MCPMark-CoW-Guard + CogLoad-Prefill LUT6` gate (`0..=3`).
#[inline]
pub const fn eval_mcpmark_cow_guard2_lut6(mcp_addr6: u8) -> u8 {
    let sh = (mcp_addr6 & 0x3F) as u32;
    (((MCPMARK_COW_GUARD2_MASKS[0] >> sh) & 1) as u8)
        | ((((MCPMARK_COW_GUARD2_MASKS[1] >> sh) & 1) as u8) << 1)
}

/// Exact `6`-input `LUT6` specification for Bandwidth-Adaptive `BA-Tiered-Expert-AirMoL` CPU-GPU
/// Co-Scheduling (`Claim LA-74`, `Theorems 134-135`), strictly generalizing `KTransformers` static
/// split and `FreeToken` (`Yang, Stoica, Han, Zaharia et al., Aug 2026`) reactive `K=1` global LRU:
///
/// `ba_addr6 = residency2 | (cpu_simd_faster1 << 2) | (reuse_gt_victim1 << 3) | (burst_k3_conf1 << 4) | (link_congested1 << 5)`:
///   - `residency2` (`bits [1..0]`): `0` = `VRAM` hit, `1` = `System RAM` warm, `2` = `NVMe` cold (free RAM slot), `3` = `NVMe` cold (RAM full).
///   - `sched2` (`0..=3`):
///       `3` (`0b11`): `Tier 0` `GPU VRAM` resident execution (`0` link / `NVMe` traffic),
///       `2` (`0b10`): Bandwidth-Adaptive `CPU SIMD` (`AMX`/`NEON`/`AVX-512`) Co-Execution overlapped with GPU (`link_congested && cpu_simd_faster`),
///       `1` (`0b01`): `K=3` Burst-Coalesced `NVMe -> RAM -> LinkMover VRAM` Promotion (`residency == 1`, or `residency == 2`, or `reuse_gt_victim && burst_k3_conf`),
///       `0` (`0b00`): Scan-Resistant Tail Bypass (suppresses `RAM` eviction when `residency == 3 && !(reuse_gt_victim && burst_k3_conf)`).
#[inline(always)]
pub const fn ba_airmol_coschedule2_spec(ba_addr6: u8) -> u8 {
    let a = (ba_addr6 & 0x3F) as usize;
    let residency = a & 3;
    let cpu_simd_faster = ((a >> 2) & 1) != 0;
    let reuse_gt_victim = ((a >> 3) & 1) != 0;
    let burst_k3_conf = ((a >> 4) & 1) != 0;
    let link_congested = ((a >> 5) & 1) != 0;

    if residency == 0 {
        3
    } else if link_congested && cpu_simd_faster && (residency == 1 || burst_k3_conf) {
        2
    } else if residency == 1 || residency == 2 || (reuse_gt_victim && burst_k3_conf) {
        1
    } else {
        0
    }
}

/// Two `64`-bit `LUT6` truth tables implementing `ba_airmol_coschedule2_spec` with `0` branches.
pub const BA_AIRMOL_COSCHED2_MASKS: [u64; 2] = {
    let mut m = [0u64; 2];
    let mut x = 0usize;
    while x < 64 {
        let s = ba_airmol_coschedule2_spec(x as u8) as usize;
        if (s & 1) != 0 {
            m[0] |= 1u64 << x;
        }
        if ((s >> 1) & 1) != 0 {
            m[1] |= 1u64 << x;
        }
        x += 1;
    }
    m
};

/// Evaluates the compiled `2`-bit Bandwidth-Adaptive `BA-AirMoL` co-scheduler `LUT6` (`0..=3`).
#[inline]
pub const fn eval_ba_airmol_coschedule_lut6(ba_addr6: u8) -> u8 {
    let sh = (ba_addr6 & 0x3F) as u32;
    (((BA_AIRMOL_COSCHED2_MASKS[0] >> sh) & 1) as u8)
        | ((((BA_AIRMOL_COSCHED2_MASKS[1] >> sh) & 1) as u8) << 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vault14BankOutcome {
    CertifiedAction {
        action_id: u8,
        bank_id: u8,
        next_epoch: u8,
    },
    MechanicShiftBreaker {
        xor_sig: u8,
    },
    WakeSolver {
        next_epoch: u8,
    },
}

/// Evaluates the Fourteen-Bank (`14 x 4,096 B = 57,344 B = 87.50%` of `64 KB` Watch `SRAM`,
/// leaving `4,096 B` `lut6-core` Nucleus + `4,096 B` Bare-Metal Stack = exact `65,536 B`!)
/// `System-0` `LUT6` cascade (`Theorem 136`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_fourteen_bank_vault_lut6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    expected_sig: Option<u8>,
    actual_sig: u8,
    subgoal_mask: u64,
    banks_action_masks: &[[u64; 3]; 14],
    banks_cert_masks: &[u64; 14],
) -> Vault14BankOutcome {
    let ep_c = if epoch > 13 { 13 } else { epoch };
    if let Some(exp) = expected_sig
        && exp != 0xFF
        && exp != actual_sig
    {
        return Vault14BankOutcome::MechanicShiftBreaker {
            xor_sig: exp ^ actual_sig,
        };
    }
    let sg_hit = ((subgoal_mask >> ((subgoal6 & 0x3F) as u32)) & 1) == 1;
    let next_ep = if sg_hit && ep_c < 13 { ep_c + 1 } else { ep_c };
    let p_sh = (percept6 & 0x3F) as u32;
    let primary = (next_ep as usize) % 14;
    let mut probe = 0usize;
    while probe < 14 {
        let mut b_id = primary + probe;
        if b_id >= 14 {
            b_id -= 14;
        }
        if ((banks_cert_masks[b_id] >> p_sh) & 1) == 1 {
            let m = &banks_action_masks[b_id];
            let a0 = ((m[0] >> p_sh) & 1) as u8;
            let a1 = (((m[1] >> p_sh) & 1) as u8) << 1;
            let a2 = (((m[2] >> p_sh) & 1) as u8) << 2;
            let act_id = (a0 | a1 | a2).max(1);
            return Vault14BankOutcome::CertifiedAction {
                action_id: act_id,
                bank_id: b_id as u8,
                next_epoch: next_ep,
            };
        }
        probe += 1;
    }
    Vault14BankOutcome::WakeSolver {
        next_epoch: next_ep,
    }
}

/// Compile-time exhaustive `64/64` minterm proof gate (`rustc` `const` evaluation).
const _: () = {
    let mut x = 0u8;
    while x < 64 {
        assert!(eval_mcpmark_cow_guard2_lut6(x) == mcpmark_cow_guard2_spec(x));
        assert!(eval_ba_airmol_coschedule_lut6(x) == ba_airmol_coschedule2_spec(x));
        x += 1;
    }
    // Spot-check key architectural invariants:
    // 1. Violated ACID balance (`acid_bal_ok = 0`) -> `0` (`CoW` Rollback, even if `crystal_hit = 1`!)
    assert!(eval_mcpmark_cow_guard2_lut6(0b11_1101) == 0);
    // 2. Duplicate tool call (`idemp_new = 0`, all safety bits `1`) -> `1` (Idempotent Dedup!)
    assert!(eval_mcpmark_cow_guard2_lut6(0b01_0111) == 1);
    // 3. Long polluted `CogLoadBench` prompt (`prefill_le128 = 0`, `crystal_hit = 0`, safety+idemp `1`) -> `2` (`CogLoad` Window!)
    assert!(eval_mcpmark_cow_guard2_lut6(0b00_1111) == 2);
    // 4. Verified clean CRUD commit (`0b11_1111 = 63`) -> `3` (Atomic `CoW` Commit!)
    assert!(eval_mcpmark_cow_guard2_lut6(63) == 3);
    // 5. Congested link + fast CPU SIMD on warm RAM (`residency=1, cpu_faster=1, congested=1`) -> `2` (CPU-GPU Co-Exec!)
    assert!(eval_ba_airmol_coschedule_lut6(0b10_0101) == 2);
    // 6. Full RAM + cold outlier without `Reuse > victim` (`residency=3, reuse_gt_victim=0`) -> `0` (No LRU thrash!)
    assert!(eval_ba_airmol_coschedule_lut6(0b01_0011) == 0);
};

/// Fused Phase 44 (`Window 24`) Rung-7 step:
/// Evaluates `14`-Bank (`56 KB`) `System-0` vault + `MCPMark-CoW-Guard LUT6` + `BA-AirMoL`
/// Bandwidth-Adaptive Co-Scheduler `LUT6` + `JouleBit6 + FWHT-64` slab (`0` `FLOP`s, `0` `MUL`s, `0` `DSP`s)!
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn eval_window24_mag_mcp_freetoken_joulebit6(
    percept6: u8,
    subgoal6: u8,
    epoch: u8,
    mcp_addr6: u8,
    ba_addr6: u8,
    banks_action_masks: &[[u64; 3]; 14],
    banks_cert_masks: &[u64; 14],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> (Vault14BankOutcome, u8, u8) {
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
    let mcp2 = eval_mcpmark_cow_guard2_lut6(mcp_addr6);
    let ba2 = eval_ba_airmol_coschedule_lut6(ba_addr6);
    eval_joulebit6_fwht64_slab_i16(acts64, v_codes, u_codes, out64);
    (v14, mcp2, ba2)
}

/// Uncompiled software baseline (`Window 24b`):
/// Executes:
///   1. Runtime floating-point `FreeToken` per-layer bandwidth LP split (`alpha* = T_xfer / (T_xfer + T_cpu)`)
///      across `32` expert slots + `64`-entry reactive global LRU scan/shift,
///   2. Dynamic `MCPMark` CRUD JSON/ACL/SQL byte-by-byte schema & path-traversal scanner (`64` bytes),
///   3. Uncompiled `LittleBit-2` `6`-col unpack + binary `MAC`s (`eval_littlebit2_6col_unpack_mac_64`).
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn eval_window24b_float_freetoken_lp_crud_scan_littlebit2(
    lru_queue64: &[u16; 64],
    expert_mb32: &[f32; 32],
    link_gbps: f32,
    cpu_gbps: f32,
    crud_payload64: &[u8; 64],
    acts64: &[i16; 64],
    v_codes: &[[u8; 10]; 6],
    u_codes: &[u8; 64],
    out64: &mut [i32; 64],
) -> u8 {
    // 1. Float per-expert FreeToken co-execution split + global LRU linear scan
    let mut cpu_assigned = 0u32;
    let inv_link = 1.0f32 / (link_gbps.max(0.1));
    let inv_cpu = 1.0f32 / (cpu_gbps.max(0.1));
    let mut e = 0usize;
    while e < 32 {
        let mb = expert_mb32[e];
        let t_xfer = mb * inv_link;
        let t_cpu = mb * inv_cpu;
        let alpha = t_xfer / (t_xfer + t_cpu + 1e-6);
        if alpha > 0.42 {
            cpu_assigned += 1;
        }
        e += 1;
    }
    let mut lru_hash = 0u32;
    let mut s = 0usize;
    while s < 64 {
        lru_hash = lru_hash.wrapping_mul(33) ^ (lru_queue64[s] as u32);
        s += 1;
    }

    // 2. Dynamic byte-by-byte CRUD JSON/ACL/path-traversal scanner
    let mut brace_depth = 0i32;
    let mut bad_acl = 0u32;
    let mut b_idx = 0usize;
    while b_idx < 64 {
        let b = crud_payload64[b_idx];
        if b == b'{' {
            brace_depth += 1;
        } else if b == b'}' {
            brace_depth -= 1;
        } else if b == b'.' && b_idx + 1 < 64 && crud_payload64[b_idx + 1] == b'.' {
            bad_acl |= 1;
        } else if b == b';' || b == b'\'' {
            bad_acl |= 2;
        }
        b_idx += 1;
    }

    // 3. Uncompiled LittleBit-2 6-col unpack + binary MAC
    eval_littlebit2_6col_unpack_mac_64(acts64, v_codes, u_codes, out64);

    ((cpu_assigned ^ lru_hash ^ (brace_depth as u32) ^ bad_acl) & 0x3F) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcpmark_cow_and_ba_airmol_lut6_exhaustive_parity() {
        for addr in 0..64u8 {
            assert_eq!(
                eval_mcpmark_cow_guard2_lut6(addr),
                mcpmark_cow_guard2_spec(addr)
            );
            assert_eq!(
                eval_ba_airmol_coschedule_lut6(addr),
                ba_airmol_coschedule2_spec(addr)
            );
        }
        assert_eq!(eval_mcpmark_cow_guard2_lut6(63), 3);
        assert_eq!(eval_mcpmark_cow_guard2_lut6(0b00_1111), 2);
        assert_eq!(eval_mcpmark_cow_guard2_lut6(0b01_0111), 1);
        assert_eq!(eval_mcpmark_cow_guard2_lut6(0b11_1011), 0);

        assert_eq!(eval_ba_airmol_coschedule_lut6(0), 3);
        assert_eq!(eval_ba_airmol_coschedule_lut6(0b10_0101), 2);
        assert_eq!(eval_ba_airmol_coschedule_lut6(0b01_1011), 1);
        assert_eq!(eval_ba_airmol_coschedule_lut6(0b01_0011), 0);
    }
}
