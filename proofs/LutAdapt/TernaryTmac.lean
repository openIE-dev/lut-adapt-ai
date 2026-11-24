/-
  LutAdapt/TernaryTmac.lean
  Section 4.30 (Claims LA-19, LA-30, LA-31):
    1. `EntroMCU` Canonical Base-3 `5-Trits/Byte` Packing (`entromcu_five_trits_fits_uint8`,
       `entromcu_five_trits_roundtrip`),
    2. `PALUTE` / `T-MAC` Zero-Multiplier 4-Trit Lookup Exactness
       (`tmac_4trit_lut_matches_ternary_dot`),
    3. Two-Tier Crystal (`Rung 6 LUT6 -> Rung 5.5 T-MAC Tail -> Rung 5 Escalate`)
       Thermodynamic Amortization (`three_tier_lut6_tmac_cascade_iff`),
    4. Three-Band Walsh–Hadamard Parseval Conservation (`parseval_three_band_conservation`),
    5. Compile-Time `const` Proof-Vector Gate Soundness (`const_proof_vector_gate_soundness`).

  Author: David Charlot, Open Interface Engineering
  License: Apache-2.0 / MIT
  Zero `sorry`, zero custom axioms, zero `opaque`, zero `inductive`, zero `ofReduceBool`.
-/

namespace LutAdapt.TernaryTmac

/--
  Packs five unsigned ternary codes `t0..t4 ∈ {0, 1, 2}` (representing `{-1, 0, +1}`)
  into a single canonical base-3 byte (`3^5 = 243 ≤ 256`, `1.585` bits/trit, `EntroMCU` layout).
-/
def pack5Trits (t0 t1 t2 t3 t4 : Nat) : Nat :=
  t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4

/--
  **`EntroMCU` 5-Trit/Byte `uint8` Fit (`Section 4.30`)**:
  Any five unsigned trits `t0..t4 < 3` pack into `pack5Trits t0 t1 t2 t3 t4 < 243 ≤ 256`,
  fitting losslessly inside one `uint8` with `13` spare escape codepoints (`243..255`).
-/
theorem entromcu_five_trits_fits_uint8
    (t0 t1 t2 t3 t4 : Nat)
    (h0 : t0 < 3) (h1 : t1 < 3) (h2 : t2 < 3) (h3 : t3 < 3) (h4 : t4 < 3) :
    pack5Trits t0 t1 t2 t3 t4 < 243 ∧ 243 ≤ 256 := by
  unfold pack5Trits
  omega

/--
  **`EntroMCU` 5-Trit/Byte Lossless Bijection (`Section 4.30`)**:
  Base-3 division and modulo recover all five packed trits `(t0, t1, t2, t3, t4)` identically.
-/
theorem entromcu_five_trits_roundtrip
    (t0 t1 t2 t3 t4 : Nat)
    (h0 : t0 < 3) (h1 : t1 < 3) (h2 : t2 < 3) (h3 : t3 < 3) (h4 : t4 < 3) :
    let b := pack5Trits t0 t1 t2 t3 t4
    b % 3 = t0 ∧
    (b / 3) % 3 = t1 ∧
    (b / 9) % 3 = t2 ∧
    (b / 27) % 3 = t3 ∧
    (b / 81) % 3 = t4 := by
  dsimp only
  unfold pack5Trits
  omega

/-- Maps unsigned trit `u ∈ {0, 1, 2}` to signed ternary weight `w ∈ {-1, 0, +1}`. -/
def tritWeight (u : Nat) : Int :=
  (u : Int) - 1

/-- Packs four unsigned trits `u0..u3 < 3` into an 81-entry `T-MAC` group index `0..80`. -/
def pack4Trits (u0 u1 u2 u3 : Nat) : Nat :=
  u0 + 3 * u1 + 9 * u2 + 27 * u3

/-- Zero-multiplier signed accumulation step (`u=0 ↦ acc - x`, `u=1 ↦ acc`, `u=2 ↦ acc + x`). -/
def addSignedByTrit (acc x : Int) (u : Nat) : Int :=
  if u = 0 then acc - x else if u = 2 then acc + x else acc

/--
  `PALUTE` / `T-MAC` 4-trit SRAM lookup entry computed purely by integer additions and
  subtractions (`0` multiplications).
-/
def tmacLutEntry (x0 x1 x2 x3 : Int) (code : Nat) : Int :=
  let a0 := addSignedByTrit 0 x0 (code % 3)
  let a1 := addSignedByTrit a0 x1 ((code / 3) % 3)
  let a2 := addSignedByTrit a1 x2 ((code / 9) % 3)
  addSignedByTrit a2 x3 ((code / 27) % 3)

/--
  **`PALUTE` / `T-MAC` Zero-Multiplier Ternary Dot-Product Exactness (`Section 4.30`)**:
  For all signed activations `x0..x3 : Int` and ternary codes `u0..u3 < 3`, the zero-multiplier
  lookup table entry `tmacLutEntry x0 x1 x2 x3 (pack4Trits u0 u1 u2 u3)` equals the exact
  algebraic dot product `∑_{k=0}^3 tritWeight(u_k) * x_k`.
-/
theorem tmac_4trit_lut_matches_ternary_dot
    (x0 x1 x2 x3 : Int) (u0 u1 u2 u3 : Nat)
    (hu0 : u0 < 3) (hu1 : u1 < 3) (hu2 : u2 < 3) (hu3 : u3 < 3) :
    tmacLutEntry x0 x1 x2 x3 (pack4Trits u0 u1 u2 u3) =
      tritWeight u0 * x0 + tritWeight u1 * x1 +
      tritWeight u2 * x2 + tritWeight u3 * x3 := by
  unfold tmacLutEntry
  dsimp only
  have d0 : pack4Trits u0 u1 u2 u3 % 3 = u0 := by unfold pack4Trits; omega
  have d1 : (pack4Trits u0 u1 u2 u3 / 3) % 3 = u1 := by unfold pack4Trits; omega
  have d2 : (pack4Trits u0 u1 u2 u3 / 9) % 3 = u2 := by unfold pack4Trits; omega
  have d3 : (pack4Trits u0 u1 u2 u3 / 27) % 3 = u3 := by unfold pack4Trits; omega
  rw [d0, d1, d2, d3]
  have s0 : addSignedByTrit 0 x0 u0 = tritWeight u0 * x0 := by
    unfold addSignedByTrit tritWeight
    rcases (by omega : u0 = 0 ∨ u0 = 1 ∨ u0 = 2) with rfl | rfl | rfl <;> simp
  have s1 : ∀ acc : Int, addSignedByTrit acc x1 u1 = acc + tritWeight u1 * x1 := by
    intro acc; unfold addSignedByTrit tritWeight
    rcases (by omega : u1 = 0 ∨ u1 = 1 ∨ u1 = 2) with rfl | rfl | rfl <;> simp <;> omega
  have s2 : ∀ acc : Int, addSignedByTrit acc x2 u2 = acc + tritWeight u2 * x2 := by
    intro acc; unfold addSignedByTrit tritWeight
    rcases (by omega : u2 = 0 ∨ u2 = 1 ∨ u2 = 2) with rfl | rfl | rfl <;> simp <;> omega
  have s3 : ∀ acc : Int, addSignedByTrit acc x3 u3 = acc + tritWeight u3 * x3 := by
    intro acc; unfold addSignedByTrit tritWeight
    rcases (by omega : u3 = 0 ∨ u3 = 1 ∨ u3 = 2) with rfl | rfl | rfl <;> simp <;> omega
  rw [s0, s1, s2, s3]

/--
  Scaled expected energy spend on a `LUT6` miss when consulting the local `672 B` `T-MAC`
  ternary tail (cost `cTmac`, resolving `qTmac` out of `scale` misses locally) before
  waking upstream `Rung 5` (`cEscalate`).
-/
def tmacTailMissSpend (scale qTmac cTmac cEscalate : Int) : Int :=
  scale * cTmac + (scale - qTmac) * cEscalate

/-- Scaled baseline spend on a `LUT6` miss when escalating directly to `Rung 5`. -/
def directEscalateMissSpend (scale cEscalate : Int) : Int :=
  scale * cEscalate

/--
  **Two-Tier `LUT6 + T-MAC` Cascade Thermodynamic Criterion (`Section 4.30`)**:
  Running the in-page `672 B` `T-MAC` ternary tail on `LUT6` misses strictly lowers expected
  energy spend relative to direct `Rung 5` escalation if and only if
    `scale * cTmac < qTmac * cEscalate` (i.e., `c_tmac < q_tmac * c_escalate`).
-/
theorem three_tier_lut6_tmac_cascade_iff (scale qTmac cTmac cEscalate : Int) :
    tmacTailMissSpend scale qTmac cTmac cEscalate <
      directEscalateMissSpend scale cEscalate ↔
      scale * cTmac < qTmac * cEscalate := by
  unfold tmacTailMissSpend directEscalateMissSpend
  have hdist : (scale - qTmac) * cEscalate = scale * cEscalate - qTmac * cEscalate :=
    Int.sub_mul scale qTmac cEscalate
  omega

/--
  **Three-Band Walsh–Hadamard Parseval Conservation (`Section 4.30`)**:
  Under Parseval's identity `eLow + eMid + eHigh = 4096`, vanishing high-harmonic parity noise
  (`eHigh = 0`) concentrates 100% of Boolean spectral energy inside the low/mid structural bands
  (`eLow + eMid = 4096`).
-/
theorem parseval_three_band_conservation
    (eLow eMid eHigh : Nat)
    (hparseval : eLow + eMid + eHigh = 4096)
    (hwobble : eHigh = 0) :
    eLow + eMid = 4096 ∧ eLow ≤ 4096 ∧ eMid ≤ 4096 := by
  omega

/--
  Compile-time `const` proof-vector assertion loop checking `hwEval k = specEval k`
  for all `k ∈ {0, ..., n - 1}`.
-/
def constVectorGate (hwEval specEval : Nat → Nat) : Nat → Bool
  | 0 => true
  | n + 1 => constVectorGate hwEval specEval n && decide (hwEval n = specEval n)

/--
  **Compile-Time `const` Proof-Vector Gate Soundness (`Section 4.30`)**:
  Whenever `constVectorGate hwEval specEval n = true` succeeds at `rustc` compile time,
  `hwEval x = specEval x` holds identically for every vector `x < n`.
-/
theorem const_proof_vector_gate_soundness
    (hwEval specEval : Nat → Nat) (n x : Nat)
    (hgate : constVectorGate hwEval specEval n = true)
    (hx : x < n) :
    hwEval x = specEval x := by
  induction n with
  | zero => omega
  | succ k ih =>
    simp [constVectorGate] at hgate
    rcases hgate with ⟨hprev, hlast⟩
    have hsplit : x < k ∨ x = k := by omega
    rcases hsplit with hlt | rfl
    · exact ih hprev hlt
    · exact hlast

end LutAdapt.TernaryTmac
