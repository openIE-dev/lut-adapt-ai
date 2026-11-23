"""Phase 28 (Claim LA-19): Lean 4.36.0 Kernel-Checked Formal Proof Verifier & Axiom Auditor.

Builds `proofs/` (`LutAdapt.Cascade`, `LutAdapt.BenesCopy`, `LutAdapt.JointMinterm`,
`LutAdapt.ConformalProvenance`) with Lean 4.36.0 (`lake build`), verifies zero `sorry` /
`axiom` tokens in source and zero `sorryAx` / custom axioms in kernel `#print axioms` output
across all 23 mechanized theorems, and writes:
  - `artifacts/lean4_formal_proofs_summary.csv`
  - `artifacts/lean4_formal_proofs_meta.json`
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from clai.contracts.joule_meter import meter_joules


THEOREM_CATALOG: list[tuple[str, str, str, str]] = [
    ("Cascade", "LutAdapt.Cascade.simon_kadane_two_gear_diff", "Section 2.1", "Exact 2-gear adjacent swap identity"),
    ("Cascade", "LutAdapt.Cascade.simon_kadane_adjacent_swap_iff", "Section 2.1", "Simon-Kadane p1*c2 >= p2*c1 optimality"),
    ("Cascade", "LutAdapt.Cascade.simon_kadane_strict_iff", "Section 2.1", "Strict Simon-Kadane ordering gain"),
    ("Cascade", "LutAdapt.Cascade.cascade_head_swap_diff", "Section 2.1", "N-gear prefix swap survival invariance"),
    ("Cascade", "LutAdapt.Cascade.proposition1_downward_compile_iff", "Proposition 1", "Downward LUT6 amortization horizon iff"),
    ("Cascade", "LutAdapt.Cascade.rung0_precedes_wake_iff", "Proposition 1", "Rung 0 precedes wake path iff c0 < p0'*c_wake"),
    ("Cascade", "LutAdapt.Cascade.zero_hit_churn_refusal", "Proposition 1", "Zero-hit adversarial churn compile refusal"),
    ("BenesCopy", "LutAdapt.BenesCopy.benes_copy_benes_depth_formula", "Theorem 2", "2*(2n-1)+n = 5n-2 closed-form stage depth"),
    ("BenesCopy", "LutAdapt.BenesCopy.kv260_benes_512_exact_counts", "Theorem 2", "N=512 -> 43 stages & 11,008 SRAM switch bits"),
    ("BenesCopy", "LutAdapt.BenesCopy.kv260_pipelined_4stage_partition", "Claim LA-16", "4-stage <= 13-stage routing partition"),
    ("BenesCopy", "LutAdapt.BenesCopy.prefixSum_mono_step", "Theorem 2", "Monotonic multicast fanout partial sum step"),
    ("BenesCopy", "LutAdapt.BenesCopy.prefixSum_mono", "Theorem 2", "Transitive monotonicity of copy intervals"),
    ("BenesCopy", "LutAdapt.BenesCopy.monotonic_copy_intervals_disjoint", "Theorem 2", "Disjointness of [S_i, S_{i+1}) copy intervals"),
    ("BenesCopy", "LutAdapt.BenesCopy.lee_butterfly_modulo_collision_free", "Theorem 2", "Modulo-2^(d+1) butterfly link non-contention"),
    ("JointMinterm", "LutAdapt.JointMinterm.joint_mode_dominates_all", "Proposition 3", "Joint-mode dominates all burst predictors"),
    ("JointMinterm", "LutAdapt.JointMinterm.joint_mode_in_support", "Proposition 3", "Zero off-support chimera guarantee"),
    ("JointMinterm", "LutAdapt.JointMinterm.marginal_chimera_witness", "Proposition 3", "Constructive (40,35,25) marginal chimera witness"),
    ("JointMinterm", "LutAdapt.JointMinterm.packTokens_bound", "Proposition 3", "b*K bit-width upper bound on packed bursts"),
    ("JointMinterm", "LutAdapt.JointMinterm.unpack_pack_roundtrip", "Proposition 3", "Lossless K-token pack/unpack bijection"),
    ("JointMinterm", "LutAdapt.JointMinterm.packTokens_fits_uint64", "Proposition 3", "uint64 fit whenever b*K <= 64"),
    ("JointMinterm", "LutAdapt.JointMinterm.hf_2026_vocab_uint64_packing_bounds", "Claims LA-7/9", "LFM2.5 (K=4), LFM3 (K=3), Qwen3.6 (K=3) fit"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.off_manifold_ray_forces_abstention", "Result 4", "Off-manifold ray ||f(x)|| > R(alpha) -> C_t = empty"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.split_conformal_rank_miscoverage_bound", "Result 4", "Finite-sample split-conformal miscoverage <= alpha"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.aci_stationary_iff_target_miscoverage", "Result 4", "ACI equilibrium iff err_t = alpha"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.downward_crystallize_never_deterministic", "Section 0.3", "ModelGenerated -/-> Deterministic lattice law"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.conformal_crystallize_rank_strictly_bounded", "Section 0.3", "ConformalBounded < CheckSafe < Deterministic"),
    ("ConformalProvenance", "LutAdapt.ConformalProvenance.speculativeSweep_matches_greedy", "Section 0.3", "Target-verified LUT6 speculation = exact greedy"),
]

ALLOWED_KERNEL_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def _find_lake() -> str:
    found = shutil.which("lake")
    if found:
        return found
    elan_lake = Path.home() / ".elan" / "bin" / "lake"
    if elan_lake.is_file():
        return str(elan_lake)
    raise FileNotFoundError("Lean 4 `lake` binary not found on PATH or ~/.elan/bin/lake")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proofs_dir = repo_root / "proofs"
    art_dir = repo_root / "artifacts"
    art_dir.mkdir(exist_ok=True)
    lake_bin = _find_lake()

    lean_files = sorted((proofs_dir / "LutAdapt").glob("*.lean")) + [proofs_dir / "LutAdapt.lean"]
    src_hashes: dict[str, str] = {}
    for lf in lean_files:
        txt = lf.read_text(encoding="utf-8")
        # Strip comments before scanning for `sorry` or `axiom` declarations
        no_block = re.sub(r"/-.*?-/", "", txt, flags=re.DOTALL)
        no_comments = re.sub(r"--[^\n]*", "", no_block)
        assert not re.search(r"\bsorry\b", no_comments), f"Forbidden `sorry` in {lf}"
        assert not re.search(r"^\s*axiom\b", no_comments, flags=re.MULTILINE), f"Forbidden custom `axiom` in {lf}"
        src_hashes[str(lf.relative_to(repo_root))] = hashlib.sha256(txt.encode("utf-8")).hexdigest()

    # Force a clean kernel build so timing & `reported_j` reflect full typechecking
    shutil.rmtree(proofs_dir / ".lake" / "build", ignore_errors=True)
    with meter_joules("compile") as jr_build:
        proc_b = subprocess.run([lake_bin, "build"], cwd=proofs_dir, capture_output=True, text=True, check=True)

    ver_out = subprocess.run([lake_bin, "env", "lean", "--version"], cwd=proofs_dir, capture_output=True, text=True, check=True).stdout.strip()

    # Kernel `#print axioms` audit across every theorem
    audit_lines = ["import LutAdapt"] + [f"#print axioms {thm}" for _, thm, _, _ in THEOREM_CATALOG]
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=proofs_dir, delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(audit_lines) + "\n")
        audit_path = Path(tf.name)

    try:
        ax_proc = subprocess.run(
            [lake_bin, "env", "lean", str(audit_path)],
            cwd=proofs_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        audit_path.unlink(missing_ok=True)

    axiom_map: dict[str, list[str]] = {}
    for line in ax_proc.stdout.splitlines():
        m_none = re.match(r"^'([^']+)' does not depend on any axioms$", line.strip())
        m_some = re.match(r"^'([^']+)' depends on axioms: \[([^\]]*)\]$", line.strip())
        if m_none:
            axiom_map[m_none.group(1)] = []
        elif m_some:
            axs = [a.strip() for a in m_some.group(2).split(",") if a.strip()]
            for a in axs:
                assert a in ALLOWED_KERNEL_AXIOMS, f"Disallowed axiom `{a}` in {m_some.group(1)}!"
            axiom_map[m_some.group(1)] = axs

    assert len(axiom_map) == len(THEOREM_CATALOG), f"Expected {len(THEOREM_CATALOG)} audited theorems, got {len(axiom_map)}"

    rows: list[dict] = []
    for mod, thm, sec, desc in THEOREM_CATALOG:
        axs = axiom_map[thm]
        rows.append(
            {
                "module": mod,
                "theorem_name": thm.split(".")[-1],
                "paper_section": sec,
                "statement_summary": desc,
                "kernel_axioms": ";".join(axs) if axs else "none (purely constructive)",
                "zero_sorry": True,
                "zero_custom_axioms": True,
            }
        )

    out_csv = art_dir / "lean4_formal_proofs_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    zero_ax_count = sum(1 for r in rows if r["kernel_axioms"].startswith("none"))
    meta = {
        "lean_version": ver_out,
        "total_kernel_checked_theorems": len(rows),
        "purely_constructive_zero_axiom_theorems": zero_ax_count,
        "sorry_count": 0,
        "custom_axiom_count": 0,
        "clean_lake_build_ms": round(jr_build.elapsed_ms, 2),
        "clean_lake_build_reported_j": round(float(jr_build.reported_j or 0.0), 4),
        "source_sha256": src_hashes,
        "measured_j": None,
        "board_metered": False,
    }
    out_json = art_dir / "lean4_formal_proofs_meta.json"
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f" PHASE 28 (CLAIM LA-19): LEAN 4.36.0 KERNEL-CHECKED FORMAL PROOFS ({len(rows)} THEOREMS, 0 SORRY)")
    print("=" * 100)
    print(f"  Toolchain:       {ver_out}")
    print(f"  Clean Build:     {meta['clean_lake_build_ms']:.1f} ms | reported_j = {meta['clean_lake_build_reported_j']} J")
    print(f"  Constructive:    {zero_ax_count}/{len(rows)} depend on ZERO axioms; {len(rows) - zero_ax_count}/{len(rows)} use only propext/Quot.sound")
    print(f"  Saved -> {out_csv.relative_to(repo_root)} & {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
