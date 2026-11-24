//! Companion `SEED_KNOWLEDGE` (`Nucleus #1`), 2-bit Wobble-Degenerate Overlay Synthesizer (`#3`),
//! and end-to-end compile-time + unit-test evolutionary ratchet verification.

use crate::bootstrap::{BIO_DOMAIN_HASH, pack_route6};
use crate::{CompactLut6Page, high_harmonic_energy, parseval_total_energy};

/// Builds a wobble-degenerate 64-bit LUT6 `INIT` mask matching the 4 structural MSBs
/// (`target_6bit >> 2`), invariant to 2-bit thermal/sensor/paraphrase jitter on `i0, i1`.
/// Spectral guarantee: Degree-5 and Degree-6 Walsh–Hadamard noise energy `== 0`.
#[inline(always)]
pub const fn wobble_codon_mask(target_6bit: u8) -> u64 {
    let base_slot = (target_6bit & 0x3C) as u32; // Clears wobble bits [0..1]
    0x0000_0000_0000_000Fu64 << base_slot        // 4 adjacent synonymous codons set!
}

/// Nucleus #1: Immutable `SEED_KNOWLEDGE` Crystal.
/// Recognizes innate baseline survival codon `0b0010_xx` (`0x08..=0x0B`) on wires `[0..5]`
/// and emits innate certified payload `0x0000_00A1` with 8 conformal parity votes (`bits 8..15`).
/// All unknown codons emit `0` votes -> cleanly tripping the Conformal Wake Gate!
pub const fn build_seed_knowledge_crystal() -> CompactLut6Page {
    let mut routes_lo = [0u32; 256];
    let mut routes_hi = [0u8; 256];
    let mut init_masks = [0u64; 256];

    let (r_lo, r_hi) = pack_route6(0, 1, 2, 3, 4, 5);
    let innate_mask = wobble_codon_mask(0x08); // Matches 0x08, 0x09, 0x0A, 0x0B

    // Payload output `0xA1` (bits 0, 5, 7 high when innate codon detected)
    routes_lo[0] = r_lo;
    routes_hi[0] = r_hi;
    init_masks[0] = innate_mask;
    routes_lo[5] = r_lo;
    routes_hi[5] = r_hi;
    init_masks[5] = innate_mask;
    routes_lo[7] = r_lo;
    routes_hi[7] = r_hi;
    init_masks[7] = innate_mask;

    // 8 Conformal Parity / Vote Gates on wires [8..15] (`conformal_mask = 0xFF00`)
    let mut v = 8usize;
    while v < 16 {
        routes_lo[v] = r_lo;
        routes_hi[v] = r_hi;
        init_masks[v] = innate_mask;
        v += 1;
    }

    let mut page = CompactLut6Page {
        magic: *b"LUT6",
        schema_ver: 1,
        flags: 0x0002,
        domain_hash: BIO_DOMAIN_HASH,
        parent_commit: [0u8; 16],
        conformal_mask: 0x0000_0000_0000_FF00,
        override_mask: 0,
        conformal_thresh: 6,
        num_layers: 1,
        gates_per_layer: 16,
        kill_max_rejects: 3,
        crc32: 0,
        _reserved: 0,
        routes_lo,
        routes_hi,
        proof_digest: [0u8; 32],
        _pad: [0u8; 672],
        init_masks,
    };
    page.crc32 = page.compute_crc32();
    page
}

pub static SEED_KNOWLEDGE: CompactLut6Page = build_seed_knowledge_crystal();

/// Nucleus #3 Compiler Helper: Synthesizes a surgical Overlay Crystal for a newly verified
/// `(frontier_codon_6bit -> verified_payload_u8)` discovery with 2-bit wobble noise immunity.
pub const fn synthesize_wobble_patch(
    frontier_state: u64,
    verified_payload: u8,
) -> CompactLut6Page {
    let mut routes_lo = [0u32; 256];
    let mut routes_hi = [0u8; 256];
    let mut init_masks = [0u64; 256];

    let (r_lo, r_hi) = pack_route6(0, 1, 2, 3, 4, 5);
    let codon_mask = wobble_codon_mask((frontier_state & 0x3F) as u8);

    // Wire payload bits [0..7] according to `verified_payload`
    let mut b = 0usize;
    while b < 8 {
        routes_lo[b] = r_lo;
        routes_hi[b] = r_hi;
        if ((verified_payload >> b) & 1) == 1 {
            init_masks[b] = codon_mask;
        }
        b += 1;
    }

    // Wire conformal vote bits [8..15] to certify ONLY this wobble family!
    let mut v = 8usize;
    while v < 16 {
        routes_lo[v] = r_lo;
        routes_hi[v] = r_hi;
        init_masks[v] = codon_mask;
        v += 1;
    }

    let mut page = CompactLut6Page {
        magic: *b"LUT6",
        schema_ver: 1,
        flags: 0x0003, // Bit 0 = Overlay Patch, Bit 1 = Active
        domain_hash: BIO_DOMAIN_HASH,
        parent_commit: [0u8; 16],
        conformal_mask: 0x0000_0000_0000_FF00,
        override_mask: 0x0000_0000_0000_FFFF, // Overrides payload [0..7] + votes [8..15]
        conformal_thresh: 6,
        num_layers: 1,
        gates_per_layer: 16,
        kill_max_rejects: 3,
        crc32: 0,
        _reserved: 0,
        routes_lo,
        routes_hi,
        proof_digest: [0u8; 32],
        _pad: [0u8; 672],
        init_masks,
    };
    page.crc32 = page.compute_crc32();
    page
}

// Compile-time spectral & Parseval proofs (`rustc` verifies all 64 codons at build time!):
const _: () = {
    assert!(SEED_KNOWLEDGE.is_valid());
    assert!(SEED_KNOWLEDGE.verify_crc32());
    assert!(parseval_total_energy(wobble_codon_mask(0x08)) == 4096);
    assert!(parseval_total_energy(1u64 << 0x34) == 4096);
    // Every wobble-encoded codon across 0..64 has EXACT ZERO degree-5/6 harmonic noise!
    let mut c = 0u8;
    while c < 64 {
        assert!(high_harmonic_energy(wobble_codon_mask(c)) == 0);
        c += 1;
    }
    // Single 1-hot point memorization leaks 7 * (-2)^2 = 28 into degree >= 5 harmonics,
    // 8-point noisy memorization leaks 448, and 6-bit parity leaks 4096:
    assert!(high_harmonic_energy(1u64 << 0x34) == 28);
    assert!(high_harmonic_energy(0x6996_9669_9669_6996) == 4096);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::SEED_AGENCY;
    use crate::{AgencyAction, Nucleus, PageLedger, transpose_64x64};

    #[test]
    fn full_biological_ratchet_lifecycle() {
        let mut nucleus = Nucleus {
            knowledge_root: &SEED_KNOWLEDGE,
            agency_root: &SEED_AGENCY,
            knowledge_overlay: None,
            agency_overlay: None,
            k_ledger: PageLedger::default(),
            a_ledger: PageLedger::default(),
            k_anchors: [
                (0x08, 0xFF, 0xA1),
                (0x0B, 0xFF, 0xA1), // Wobble-jittered innate state protected too!
            ],
            a_anchors: [
                ((1u64 << 40) | (1u64 << 48), 0x3F, 0x20), // Healthy tool-ready dispatch
                (1u64 << 63, 0x3F, 0x3C),                  // Apoptosis rollback invariant
            ],
        };

        // 1. INNATE HIT: Seed codon `0x08` resolves in 1 pass at zero FLOPs (no tools woken).
        assert_eq!(
            nucleus.step(0x08, 0b0000_0001),
            AgencyAction::EmitCertified { payload: 0xA1 }
        );
        assert_eq!(nucleus.k_ledger.hits, 1);

        // 2. NOVELTY ENCOUNTER: Unseen frontier codon `0x34` (`w48=1, w40=0`) misses Knowledge (#1)
        //    and smoothly escalates through Agency (#2) to wake the Refiner (#3).
        let novel_input = 0x34u64;
        let miss_action = nucleus.step(novel_input, 0);
        assert_eq!(
            miss_action,
            AgencyAction::WakeRefiner {
                frontier_bits: 0x34
            }
        );
        assert_eq!(nucleus.k_ledger.escalations, 1);

        // 3. HOSTILE / NOISY MUTATIONS REJECTED (`max_noise_energy = 16`):
        //    (a) 1-hot point memorization (`1u64 << 0x34`, noise=28 > 16) rejected!
        //    (b) Parity noise (`0x6996...`, noise=4096 > 16) rejected!
        //    (c) Anchor-violating overwrite of innate codon `0x08` rejected!
        let mut point_brittle = synthesize_wobble_patch(novel_input, 0x5E);
        point_brittle.init_masks[0] = 1u64 << 0x34;
        assert!(!nucleus.refine_knowledge(point_brittle, 16));

        let mut parity_brittle = synthesize_wobble_patch(novel_input, 0x5E);
        parity_brittle.init_masks[0] = 0x6996_9669_9669_6996;
        assert!(!nucleus.refine_knowledge(parity_brittle, 64));

        let conflicting_patch = synthesize_wobble_patch(0x08, 0xFF);
        assert!(!nucleus.refine_knowledge(conflicting_patch, 16));

        // 4. VERIFIED WOBBLE REFINEMENT ADMITTED (`noise == 0 <= 16` & 100% anchor retention):
        let clean_patch = synthesize_wobble_patch(novel_input, 0x5E);
        assert!(nucleus.refine_knowledge(clean_patch, 16));

        // 5. DOWNWARD COMPILATION PAYOFF:
        //    Exact repeat `0x34` AND noisy wobble variant `0x37` (+3 LSB sensor jitter!)
        //    resolve immediately in Rung-6 LUT6 logic without waking Rung-5!
        assert_eq!(
            nucleus.step(0x34, 0),
            AgencyAction::EmitCertified { payload: 0x5E }
        );
        assert_eq!(
            nucleus.step(0x37, 0),
            AgencyAction::EmitCertified { payload: 0x5E }
        );

        // 6. RETENTION PROOF: Original innate survival reflex `0x08` & `0x0B` untouched.
        assert_eq!(
            nucleus.step(0x08, 0),
            AgencyAction::EmitCertified { payload: 0xA1 }
        );
        assert_eq!(
            nucleus.step(0x0B, 0),
            AgencyAction::EmitCertified { payload: 0xA1 }
        );

        // 7. APOPTOSIS SELF-HEALING (`w63 = 1 -> 0x3C`):
        //    Online verifier reject evicts overlay cleanly while keeping SEED_KNOWLEDGE intact!
        assert_eq!(
            nucleus.step_with_riboswitch(0x34, 0, 0xFF, 0, false, true),
            AgencyAction::ApoptosisRollback
        );
        assert!(nucleus.knowledge_overlay.is_none());
        assert_eq!(
            nucleus.step(0x08, 0),
            AgencyAction::EmitCertified { payload: 0xA1 }
        );
        assert_eq!(
            nucleus.step(0x34, 0),
            AgencyAction::WakeRefiner {
                frontier_bits: 0x34
            }
        );

        // 8. METABOLIC HOMEOSTASIS (`joule_budget = 0 -> 0x00 STOP_HALT`):
        assert_eq!(
            nucleus.step_with_riboswitch(0x34, 0, 0, 0, false, false),
            AgencyAction::SleepHalt
        );
    }

    #[test]
    fn vertical_bitsliced_batch64_matches_scalar_eval_raw() {
        let mut inputs = [0u64; 64];
        for (i, slot) in inputs.iter_mut().enumerate() {
            // Mix all 64 codons + high-bit agency flags across lanes
            *slot = (i as u64)
                | (((i as u64) & 1) << 40)
                | ((((i >> 1) as u64) & 1) << 48)
                | ((((i >> 5) as u64) & 1) << 63)
                | ((0xDEAD_0000u64 + i as u64) << 14);
        }

        // Verify transpose involution
        assert_eq!(transpose_64x64(&transpose_64x64(&inputs)), inputs);

        // Verify SEED_KNOWLEDGE batch64 == 64x eval_raw
        let k_batch = SEED_KNOWLEDGE.eval_batch64(&inputs);
        for i in 0..64 {
            assert_eq!(k_batch[i], SEED_KNOWLEDGE.eval_raw(inputs[i]));
        }

        // Verify SEED_AGENCY batch64 == 64x eval_raw (tests arbitrary non-wobble gates too!)
        let a_batch = SEED_AGENCY.eval_batch64(&inputs);
        for i in 0..64 {
            assert_eq!(a_batch[i], SEED_AGENCY.eval_raw(inputs[i]));
            assert_eq!(
                SEED_AGENCY.eval_raw_unrolled4(inputs[i]),
                SEED_AGENCY.eval_raw(inputs[i])
            );
            assert_eq!(
                SEED_KNOWLEDGE.eval_raw_unrolled4(inputs[i]),
                SEED_KNOWLEDGE.eval_raw(inputs[i])
            );
        }
    }

    #[test]
    fn two_tier_tmac_tail_and_compile_time_crystals() {
        use crate::EvalOutcome;
        use crate::generated_proof_vectors::{
            HOT_TIER1_SKILL_PAYLOAD, HOT_TIER1_SKILL_WORD, VERIFIED_TMAC_OOD_VECTORS,
            VERIFIED_TMAC_SKILL_ANCHORS, mol_skills_tmac_crystal, yosys_operon_crystal,
        };
        use crate::tmac_tail::{pack_5trits, unpack_5trits, wire_state_to_bipolar_i8};

        // 1. Exhaustive 243-state EntroMCU 5-trit bijection check
        for b in 0u8..243 {
            let trits = unpack_5trits(b);
            assert_eq!(
                pack_5trits(trits[0], trits[1], trits[2], trits[3], trits[4]),
                b
            );
        }

        // 2. Verify Yosys Operon crystal & Two-Tier T-MAC crystal runtime evaluation
        let op = yosys_operon_crystal();
        assert!(op.verify_crc32());

        let skills = mol_skills_tmac_crystal();
        assert!(skills.verify_crc32());
        let mut ledger = PageLedger::default();

        // Tier-1 Rung-6 36-wire 2-layer Boolean hit (`HOT_TIER1_SKILL_WORD -> 0x11` without touching T-MAC tail)
        let f_hot = wire_state_to_bipolar_i8(HOT_TIER1_SKILL_WORD);
        assert_eq!(
            skills.evaluate_two_tier(HOT_TIER1_SKILL_WORD, &f_hot, None, &mut ledger),
            Ok(EvalOutcome::Act {
                output_bits: HOT_TIER1_SKILL_PAYLOAD,
                votes: 8
            })
        );

        // Tier-2 Zero-MUL T-MAC tail resolves all 5 harvested noisy skill paraphrases locally
        for &(word, expected_pay) in &VERIFIED_TMAC_SKILL_ANCHORS {
            let feats = wire_state_to_bipolar_i8(word);
            let Ok(res) = skills.evaluate_two_tier(word, &feats, None, &mut ledger);
            match res {
                EvalOutcome::Act { output_bits, votes } => {
                    assert_eq!(output_bits, expected_pay);
                    assert!(votes >= skills.conformal_thresh);
                }
                other => panic!("Expected Two-Tier T-MAC Act, got {other:?}"),
            }
        }

        // Tier-2 OOD hostile queries cleanly escalate (`votes < 6`)
        for &ood_word in &VERIFIED_TMAC_OOD_VECTORS {
            let feats = wire_state_to_bipolar_i8(ood_word);
            let Ok(res) = skills.evaluate_two_tier(ood_word, &feats, None, &mut ledger);
            assert!(matches!(res, EvalOutcome::Escalate { .. }));
        }
    }
}
