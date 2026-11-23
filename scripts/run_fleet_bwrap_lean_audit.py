"""Linux `bwrap`-Sandboxed Six-Kernel Paranoid Proof Auditor across OpenIE Repositories.

Designed to run on a Linux x86_64/aarch64 host with `/usr/bin/bwrap` (bubblewrap) and
Linux `powercap` RAPL (`intel-rapl:0`, `intel-rapl:0:0`) so `lake check --paranoid` runs
inside full kernel-namespace sandboxing (ZERO `--inadvisably-no-sandbox` bypass) across:
  1. `lut-adapt-ai/proofs` (`LutAdapt`)
  2. `openie-research-experiments/communication-avalanche/lean` (`Avalanche`)
  3. `openie-research-experiments/time-as-an-index/lean` (`TimeAsAnIndex`)
  4. `rce/spec/lean` (`RCE`)

Emits `openie_fleet_six_kernel_paranoid_audit.json` and `.csv`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


def read_rapl_uj(zone_dir: str) -> tuple[int, int]:
    p_e = Path(zone_dir) / "energy_uj"
    p_m = Path(zone_dir) / "max_energy_range_uj"
    if not p_e.is_file():
        return 0, 0
    e = int(p_e.read_text().strip())
    m = int(p_m.read_text().strip()) if p_m.is_file() else 0
    return e, m


def diff_rapl_j(start: tuple[int, int], end: tuple[int, int]) -> float | None:
    e0, m0 = start
    e1, _ = end
    if e0 == 0 and e1 == 0:
        return None
    if e1 >= e0:
        return (e1 - e0) * 1e-6
    if m0 > 0:
        return ((m0 - e0) + e1) * 1e-6
    return None


def scan_lean_sources(proof_dir: Path) -> dict:
    lean_files = sorted(
        p for p in proof_dir.rglob("*.lean")
        if ".lake" not in p.parts
    )
    sorry_cnt = 0
    custom_ax_cnt = 0
    opaque_cnt = 0
    of_reduce_cnt = 0
    skip_tc_cnt = 0
    thm_cnt = 0
    h = hashlib.sha256()
    for lf in lean_files:
        raw = lf.read_text(encoding="utf-8")
        h.update(str(lf.relative_to(proof_dir)).encode("utf-8"))
        h.update(raw.encode("utf-8"))
        no_blk = re.sub(r"/-.*?-/", "", raw, flags=re.DOTALL)
        clean = re.sub(r"--[^\n]*", "", no_blk)
        sorry_cnt += len(re.findall(r"\bsorry\b", clean))
        custom_ax_cnt += len(re.findall(r"^\s*axiom\b", clean, flags=re.MULTILINE))
        opaque_cnt += len(re.findall(r"^\s*opaque\b", clean, flags=re.MULTILINE))
        of_reduce_cnt += len(re.findall(r"\b(?:native_decide|ofReduceBool)\b", clean))
        skip_tc_cnt += len(re.findall(r"\bskipKernelTC\b", clean))
        thm_cnt += len(re.findall(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+", clean, flags=re.MULTILINE))
    return {
        "lean_files": len(lean_files),
        "theorem_lemma_decls": thm_cnt,
        "sorry_count": sorry_cnt,
        "custom_axiom_count": custom_ax_cnt,
        "opaque_count": opaque_cnt,
        "native_decide_ofReduceBool_count": of_reduce_cnt,
        "skipKernelTC_count": skip_tc_cnt,
        "tree_sha256": h.hexdigest(),
    }


def parse_paranoid_out(out: str) -> dict:
    kernels = {
        "lean_paranoid": "Lean paranoid kernel accepts the solution" in out,
        "lean4lean": "lean4lean kernel accepts the solution" in out,
        "nanoda": "nanoda kernel accepts the solution" in out,
        "con_leche_verified": "con-leche kernel accepts the solution" in out,
        "con_ron_verified": "con-ron kernel accepts the solution" in out,
        "lean_default": "Lean default kernel accepts the solution" in out,
    }
    m_l4l = re.search(r"checked\s+(\d+)\s+declarations", out)
    m_cl = re.search(r"con-leche:\s+accepted\s+(\d+)\s+declarations\s+\(--verified\)", out)
    m_cr = re.search(r"con-ron:\s+accepted\s+(\d+)\s+declarations\s+\(--verified\)", out)
    m_ax = re.search(r"Uses axioms:\s*([^\n]+)", out)
    return {
        "kernels": kernels,
        "kernels_passed": sum(1 for v in kernels.values() if v),
        "lean4lean_decls": int(m_l4l.group(1)) if m_l4l else 0,
        "con_leche_decls": int(m_cl.group(1)) if m_cl else 0,
        "con_ron_decls": int(m_cr.group(1)) if m_cr else 0,
        "uses_axioms": m_ax.group(1).strip() if m_ax else "none",
        "sandbox_warning_present": "Sandbox disabled" in out,
    }


def audit_suite(suite_id: str, repo_name: str, proof_dir: Path, lake_bin: str) -> dict:
    print(f"\n>>> Auditing [{suite_id}] at {proof_dir} ...", flush=True)
    scan = scan_lean_sources(proof_dir)
    tc_file = proof_dir / "lean-toolchain"
    orig_tc = tc_file.read_text(encoding="utf-8").strip() if tc_file.is_file() else "unknown"

    # If a suite pins an older toolchain without `lake check --paranoid` (< v4.35.0),
    # check under v4.36.0 via `ELAN_TOOLCHAIN` override while recording original toolchain!
    env = os.environ.copy()
    check_tc = orig_tc
    if "v4.36" not in orig_tc and "v4.35" not in orig_tc:
        env["ELAN_TOOLCHAIN"] = "leanprover/lean4:v4.36.0"
        check_tc = f"{orig_tc} -> audited under leanprover/lean4:v4.36.0"

    shutil.rmtree(proof_dir / ".lake" / "build", ignore_errors=True)

    pkg0 = read_rapl_uj("/sys/class/powercap/intel-rapl:0")
    cor0 = read_rapl_uj("/sys/class/powercap/intel-rapl:0:0")
    t0 = time.perf_counter()
    b_proc = subprocess.run([lake_bin, "build"], cwd=proof_dir, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    pkg1 = read_rapl_uj("/sys/class/powercap/intel-rapl:0")
    cor1 = read_rapl_uj("/sys/class/powercap/intel-rapl:0:0")
    if b_proc.returncode != 0:
        raise RuntimeError(f"`lake build` failed in {suite_id}:\n{b_proc.stdout}\n{b_proc.stderr}")

    build_ms = round((t1 - t0) * 1000.0, 2)
    build_pkg_j = diff_rapl_j(pkg0, pkg1)
    build_cor_j = diff_rapl_j(cor0, cor1)

    # Run TRUE Linux bwrap-sandboxed `lake check --paranoid` (zero `--inadvisably-no-sandbox`!)
    assert shutil.which("bwrap") is not None, "`bwrap` binary required for true sandboxed paranoid check!"
    pkg2 = read_rapl_uj("/sys/class/powercap/intel-rapl:0")
    cor2 = read_rapl_uj("/sys/class/powercap/intel-rapl:0:0")
    t2 = time.perf_counter()
    p_proc = subprocess.run([lake_bin, "check", "--paranoid"], cwd=proof_dir, env=env, capture_output=True, text=True)
    t3 = time.perf_counter()
    pkg3 = read_rapl_uj("/sys/class/powercap/intel-rapl:0")
    cor3 = read_rapl_uj("/sys/class/powercap/intel-rapl:0:0")

    combined = (p_proc.stdout or "") + "\n" + (p_proc.stderr or "")
    if p_proc.returncode != 0:
        # Check if Linux userns requires unprivileged_userns_clone; never silently weaken without recording!
        raise RuntimeError(f"`lake check --paranoid` failed in {suite_id} (rc={p_proc.returncode}):\n{combined}")

    parsed = parse_paranoid_out(combined)
    assert parsed["kernels_passed"] == 6, f"Expected 6/6 kernels in {suite_id}, got {parsed}"
    assert not parsed["sandbox_warning_present"], f"Unexpected sandbox warning in {suite_id}!"

    paranoid_ms = round((t3 - t2) * 1000.0, 2)
    paranoid_pkg_j = diff_rapl_j(pkg2, pkg3)
    paranoid_cor_j = diff_rapl_j(cor2, cor3)

    print(
        f"    PASS [{suite_id}]: {scan['theorem_lemma_decls']} thms | 6/6 bwrap-sandboxed kernels | "
        f"con-leche={parsed['con_leche_decls']} con-ron={parsed['con_ron_decls']} | "
        f"build={build_ms} ms ({build_cor_j or 0:.2f} J core) | paranoid={paranoid_ms} ms ({paranoid_cor_j or 0:.2f} J core)",
        flush=True,
    )
    return {
        "suite_id": suite_id,
        "repository": repo_name,
        "toolchain": check_tc,
        "bwrap_linux_sandbox": True,
        "lean_files": scan["lean_files"],
        "theorem_lemma_decls": scan["theorem_lemma_decls"],
        "sorry_count": scan["sorry_count"],
        "custom_axiom_count": scan["custom_axiom_count"],
        "opaque_count": scan["opaque_count"],
        "ofReduceBool_count": scan["native_decide_ofReduceBool_count"],
        "kernels_passed": f"{parsed['kernels_passed']}/6",
        "con_leche_verified_decls": parsed["con_leche_decls"],
        "con_ron_verified_decls": parsed["con_ron_decls"],
        "lean4lean_decls": parsed["lean4lean_decls"],
        "uses_axioms": parsed["uses_axioms"],
        "clean_build_ms": build_ms,
        "clean_build_rapl_core_reported_j": round(build_cor_j, 4) if build_cor_j is not None else None,
        "clean_build_rapl_pkg_reported_j": round(build_pkg_j, 4) if build_pkg_j is not None else None,
        "paranoid_6kernel_ms": paranoid_ms,
        "paranoid_6kernel_rapl_core_reported_j": round(paranoid_cor_j, 4) if paranoid_cor_j is not None else None,
        "paranoid_6kernel_rapl_pkg_reported_j": round(paranoid_pkg_j, 4) if paranoid_pkg_j is not None else None,
        "tree_sha256": scan["tree_sha256"],
    }


def main() -> None:
    root = Path(os.environ.get("AUDIT_ROOT", "/home/dcharlot/wai-build/lean6k-audit"))
    lake_bin = str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake_bin).is_file():
        found = shutil.which("lake")
        assert found, "lake not found"
        lake_bin = found

    suites = [
        ("lut-adapt", "openIE-dev/lut-adapt-ai", root / "lut-adapt-ai" / "proofs"),
        ("communication-avalanche", "openIE-dev/openie-research-experiments", root / "openie-research-experiments" / "communication-avalanche" / "lean"),
        ("time-as-an-index", "openIE-dev/openie-research-experiments", root / "openie-research-experiments" / "time-as-an-index" / "lean"),
        ("rce-spec", "openIE-dev/rce", root / "rce" / "spec" / "lean"),
    ]

    results = []
    for sid, rname, pdir in suites:
        if pdir.is_dir():
            results.append(audit_suite(sid, rname, pdir, lake_bin))

    out_csv = root / "openie_fleet_six_kernel_paranoid_audit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    meta = {
        "host": platform.node(),
        "os_kernel": platform.platform(),
        "bwrap_path": shutil.which("bwrap"),
        "total_suites_audited": len(results),
        "total_theorems_and_lemmas": sum(r["theorem_lemma_decls"] for r in results),
        "all_six_kernels_passed_sandboxed": all(r["kernels_passed"] == "6/6" and r["bwrap_linux_sandbox"] for r in results),
        "measured_j": None,
        "board_metered": False,
        "results": results,
    }
    out_json = root / "openie_fleet_six_kernel_paranoid_audit.json"
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(results)} suite audits -> {out_csv} & {out_json}", flush=True)


if __name__ == "__main__":
    main()
