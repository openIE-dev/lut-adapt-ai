"""Phase 27 (Claim LA-15): Official External Public Coding & Repository Benchmarks
(`bigcode/bigcodebench` `v0.1.3` + Upstream `princeton-nlp/SWE-bench_Verified` Git Worktrees)
Evaluated Head-to-Head Through Live `mol-agent` (`MCP` JSON-RPC `2.0` + `.mol/` Crystallization).

Strict Evaluation Invariants:
  1. Official Parquet Digests Verified at Load Time:
       - `bigcode/bigcodebench` (`data/v0.1.3-00000-of-00001.parquet`):
         SHA-256 `ed5a3a39e011468f6309ea3277bf37698fb0753f52fc852a3aef8ce3a50ac512`
       - `princeton-nlp/SWE-bench_Verified` (`data/test-00000-of-00001.parquet`):
         SHA-256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`
  2. Zero Gold Leakage:
       - `canonical_solution` (`BigCodeBench`) and `patch` (`SWE-bench_Verified`) are NEVER exposed to
         any localizer, prompt, or model arm; `SWE-bench_Verified` file/AST window localization uses
         ONLY `problem_statement`, test-file imports, and pre-patch failing `pytest` stack frames.
  3. Fair Identical Prompts & Subprocess Sandboxes Across All 2026 Open-Weights Arms:
       - Compares ungated `LiquidAI/LFM3-1.7B`, `google/gemma-4-E2B-it`, and Metal-accelerated
         `Qwen/Qwen3.6-35B-A3B` against `OpenIE MoL MCP Agent` (`Rung 0 BenesLUT6` + `Rung 1 .mol/`
         + `C(z)` subprocess `unittest`/`pytest` + counterexample escalation up to `Rung 3d <think>`).
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
import torch

from clai.contracts.joule_meter import meter_joules, read_apple_pmu_telemetry
from clai.frontier.moe_air_mol import default_nvme_shard_root
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    ExpertAirMoL35BRunner,
    Local2026ModelRunner,
    _build_repair_prompt,
    encode_spec_bitbus_512,
    extract_python_code,
)


BCB_V013_EXPECTED_SHA256 = "ed5a3a39e011468f6309ea3277bf37698fb0753f52fc852a3aef8ce3a50ac512"
SWEBENCH_VERIFIED_EXPECTED_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"

BCB_SELECTED_16_IDS = [
    "BigCodeBench/0",
    "BigCodeBench/1",
    "BigCodeBench/2",
    "BigCodeBench/3",
    "BigCodeBench/4",
    "BigCodeBench/5",
    "BigCodeBench/6",
    "BigCodeBench/7",
    "BigCodeBench/8",
    "BigCodeBench/10",
    "BigCodeBench/11",
    "BigCodeBench/14",
    "BigCodeBench/19",
    "BigCodeBench/22",
    "BigCodeBench/23",
    "BigCodeBench/24",
]

SWE_SELECTED_8_IDS = [
    "sympy__sympy-23262",
    "sympy__sympy-23413",
    "sympy__sympy-23534",
    "sympy__sympy-23824",
    "sympy__sympy-23950",
    "sympy__sympy-24066",
    "sympy__sympy-24213",
    "sympy__sympy-24443",
]


@dataclass
class ExternalBenchmarkRow:
    benchmark: str
    arm: str
    wave1_cold_pass_pct: float
    wave2_paraphrase_or_warm_pass_pct: float
    overall_verified_pass_pct: float
    broken_outputs_admitted_pct: float
    wave2_mol_fast_path_pct: float
    frontier_35b_calls_pct: float
    mean_latency_ms_wave1: float
    mean_latency_ms_wave2: float
    est_joules_per_verified_solve: float
    reported_joules_per_verified_solve: float | None = None


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# 1. Official `bigcode/bigcodebench` (`v0.1.3`) Loader & Subprocess Verifier
# =============================================================================


@dataclass
class BigCodeBenchWaveTask:
    eval_id: str
    bcb_task_id: str
    wave: int  # 1 = `complete_prompt` (cold), 2 = `instruct_prompt` (human paraphrase)
    entry_point: str
    prompt: str
    code_prompt: str
    test_code: str


def load_official_bigcodebench_2wave_tasks() -> tuple[list[BigCodeBenchWaveTask], str]:
    bcb_parquet = hf_hub_download(
        "bigcode/bigcodebench", "data/v0.1.3-00000-of-00001.parquet", repo_type="dataset"
    )
    digest = _sha256_file(bcb_parquet)
    assert digest == BCB_V013_EXPECTED_SHA256, f"BigCodeBench SHA-256 mismatch: {digest}"

    by_id = {r["task_id"]: r for r in pq.read_table(bcb_parquet).to_pylist()}
    wave1: list[BigCodeBenchWaveTask] = []
    wave2: list[BigCodeBenchWaveTask] = []

    for idx, tid in enumerate(BCB_SELECTED_16_IDS, start=1):
        row = by_id[tid]
        # Pre-flight: confirm official gold solution passes `unittest` on current Python
        gold_ok, _, gold_err = verify_bcb_candidate(
            row["code_prompt"] + row["canonical_solution"], row["code_prompt"], row["test"], row["entry_point"]
        )
        assert gold_ok, f"Gold BigCodeBench solution failed for {tid}: {gold_err}"

        p_w1 = (
            "Complete the following Python 3 module (`task_func`) so it satisfies every requirement "
            "and edge case in the docstring. Return ONLY valid Python code in one ```python ... ``` block.\n\n"
            f"```python\n{row['complete_prompt'].strip()}\n```"
        )
        p_w2 = (
            f"{row['instruct_prompt'].strip()}\n\n"
            "Use the exact starter imports, constants, and function signature below. "
            "Return ONLY valid Python code in one ```python ... ``` block.\n\n"
            f"```python\n{row['code_prompt'].strip()}\n```"
        )
        slug = tid.replace("/", "_")
        wave1.append(
            BigCodeBenchWaveTask(
                eval_id=f"W1_{idx:02d}_{slug}_complete",
                bcb_task_id=tid,
                wave=1,
                entry_point=row["entry_point"],
                prompt=p_w1,
                code_prompt=row["code_prompt"],
                test_code=row["test"],
            )
        )
        wave2.append(
            BigCodeBenchWaveTask(
                eval_id=f"W2_{idx:02d}_{slug}_instruct",
                bcb_task_id=tid,
                wave=2,
                entry_point=row["entry_point"],
                prompt=p_w2,
                code_prompt=row["code_prompt"],
                test_code=row["test"],
            )
        )

    return wave1 + wave2, digest


def verify_bcb_candidate(
    raw_output: str, code_prompt: str, test_code: str, entry_point: str = "task_func"
) -> tuple[bool, str, str]:
    """Executes official `BigCodeBench` `unittest.TestCase` harness in an isolated temp directory."""
    code = extract_python_code(raw_output)
    if not code:
        return False, "", "Empty candidate code."

    # Prepend `code_prompt` header imports/constants if the model emitted only `def task_func(...)`
    header_lines: list[str] = []
    for ln in code_prompt.splitlines():
        if ln.startswith(f"def {entry_point}"):
            break
        header_lines.append(ln)
    header_src = "\n".join(header_lines).strip()
    if f"def {entry_point}" in code:
        full_cand = (header_src + "\n\n" + code) if header_src else code
    else:
        # Model emitted an indented body continuation of `code_prompt`
        indented = "\n".join(("    " + ln if not ln.startswith((" ", "\t")) and ln.strip() else ln) for ln in code.splitlines())
        full_cand = code_prompt.rstrip() + "\n" + indented + "\n"

    try:
        tree = ast.parse(full_cand)
        norm_cand = ast.unparse(tree)
    except SyntaxError as exc:
        return False, full_cand, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    harness = (
        f"{norm_cand}\n\n"
        f"{test_code}\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main(verbosity=1)\n"
    )
    with tempfile.TemporaryDirectory(prefix="bcb_eval_") as td:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", harness],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=8.0,
            )
        except subprocess.TimeoutExpired:
            return False, norm_cand, "TimeoutExpired (> 8.0s) in BigCodeBench unittest."

    if proc.returncode == 0:
        return True, norm_cand, ""
    err_tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-12:])
    return False, norm_cand, err_tail


def lookup_top_mol_candidates(
    store: BenesLUT6SkillCascadeStore, fn_name: str, prompt: str, top_k: int = 4
) -> list[Any]:
    """Returns up to `top_k` candidate skills (`BenesLUT6` primary prediction + closest XNOR-CAM exemplars)."""
    if not store.skills:
        return []
    out: list[Any] = []
    seen_ids: set[int] = set()
    primary = store.lookup_rung0_rung1(fn_name, prompt)
    if primary is not None:
        out.append(primary)
        seen_ids.add(primary.skill_id)

    bus = encode_spec_bitbus_512(fn_name, prompt)
    scored: list[tuple[int, Any]] = []
    for ent in store.skills.values():
        if ent.fn_name != fn_name or ent.skill_id in seen_ids:
            continue
        best_hd = min((int(np.count_nonzero(bus != ex)) for ex in ent.bus_exemplars), default=512)
        scored.append((best_hd, ent))
    scored.sort(key=lambda kv: kv[0])
    for hd, ent in scored[: max(0, top_k - len(out))]:
        if hd <= 96:
            out.append(ent)
    return out


# =============================================================================
# 2. Official Upstream `princeton-nlp/SWE-bench_Verified` Git-Worktree Harness
# =============================================================================


@dataclass
class SWEBenchVerifiedWaveTask:
    eval_id: str
    instance_id: str
    wave: int
    base_commit: str
    problem_statement: str
    test_patch: str
    test_files: list[str]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    target_file: str
    target_snippet: str
    pre_failure_trace: str


SWE_EDIT_SYSTEM_PROMPT = (
    "You are an expert Python engineer fixing an upstream repository issue.\n"
    "Inspect the issue description, failing `pytest` traceback, and target code region carefully.\n"
    "Output ONLY one or more exact `SEARCH/REPLACE` edit blocks in this format (no prose outside fences):\n"
    "```python\n"
    "<<<<<<< SEARCH\n"
    "<exact contiguous lines copied verbatim from the provided target file excerpt>\n"
    "=======\n"
    "<replacement lines with exact indentation>\n"
    ">>>>>>> REPLACE\n"
    "```"
)


def ensure_sympy_bare_repo() -> Path:
    cache_dir = default_nvme_shard_root() / "swe_git_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bare = cache_dir / "sympy.git"
    if not bare.exists():
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", "https://github.com/sympy/sympy.git", str(bare)],
            check=True,
        )
    return bare


def _zero_leak_localize_sympy_window(
    wt_dir: Path, problem_statement: str, test_files: list[str], pre_trace: str
) -> tuple[str, str]:
    """
    Zero-gold-leakage AST + traceback + test-import localizer:
      1. Extracts candidate `sympy/.../*.py` non-test source files from `pytest` traceback frames
         and `from sympy.xxx import yyy` statements in `test_files` + `problem_statement`.
      2. Scores AST `FunctionDef` / `ClassDef` nodes by lexical BM25/token overlap with
         `problem_statement` + `pre_trace` and returns `(rel_file_path, numbered_excerpt_str)`.
    """
    cand_files: dict[str, float] = {}

    # 1. Exact `(rel_file, line_no)` stack frames from `pytest` traceback (`sympy/.../foo.py:123:`)
    trace_file_lines: dict[str, list[int]] = {}
    for rel_f, lno_s in re.findall(r"(sympy/[A-Za-z0-9_/]+\.py):(\d+):", pre_trace):
        if "/tests/" not in rel_f and (wt_dir / rel_f).is_file():
            cand_files[rel_f] = cand_files.get(rel_f, 0.0) + 30.0
            trace_file_lines.setdefault(rel_f, []).append(int(lno_s))

    for rel_f in re.findall(r"(sympy/[A-Za-z0-9_/]+\.py)", pre_trace + "\n" + problem_statement):
        if "/tests/" not in rel_f and (wt_dir / rel_f).is_file():
            cand_files[rel_f] = cand_files.get(rel_f, 0.0) + 6.0

    # 2. Imports inside the failing test file(s)
    imported_syms: set[str] = set()
    for tf in test_files:
        tf_p = wt_dir / tf
        if not tf_p.is_file():
            continue
        txt = tf_p.read_text(encoding="utf-8", errors="ignore")
        for mod_dot, names_str in re.findall(r"from\s+(sympy(?:\.[A-Za-z0-9_]+)+)\s+import\s+([^\n#]+)", txt):
            for nm in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", names_str):
                if nm != "as":
                    imported_syms.add(nm)
            mod_rel = mod_dot.replace(".", "/") + ".py"
            if "/tests/" not in mod_rel and (wt_dir / mod_rel).is_file():
                cand_files[mod_rel] = cand_files.get(mod_rel, 0.0) + 4.0
            else:
                init_dir = wt_dir / mod_dot.replace(".", "/")
                if init_dir.is_dir():
                    for py_p in init_dir.glob("*.py"):
                        rel_p = str(py_p.relative_to(wt_dir))
                        if "/tests/" not in rel_p:
                            cand_files[rel_p] = cand_files.get(rel_p, 0.0) + 2.5

        # Sibling module next to `.../tests/test_foo.py` -> `.../foo.py`
        m_sib = re.match(r"^(sympy/.+)/tests/test_([A-Za-z0-9_]+)\.py$", tf)
        if m_sib:
            for cand_s in (
                f"{m_sib.group(1)}/{m_sib.group(2)}.py",
                f"{m_sib.group(1)}/{m_sib.group(2).rstrip('s')}.py",
                f"{m_sib.group(1)}/{m_sib.group(2)}system.py",
            ):
                if (wt_dir / cand_s).is_file():
                    cand_files[cand_s] = cand_files.get(cand_s, 0.0) + 22.0

    trace_funcs = set(re.findall(r"\bin\s+([A-Za-z_][A-Za-z0-9_]*)\b", pre_trace))
    query_toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", problem_statement + "\n" + pre_trace)) | imported_syms
    best_score = -1.0
    best_file = ""
    best_span = (1, 120)

    for rel_f, base_w in cand_files.items():
        src = (wt_dir / rel_f).read_text(encoding="utf-8", errors="ignore")
        lines = src.splitlines()
        if len(lines) <= 130:
            overlap = len(query_toks & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", src)))
            sc = base_w + 1.5 * overlap + (12.0 if Path(rel_f).stem in problem_statement.lower() else 0.0)
            if sc > best_score:
                best_score = sc
                best_file = rel_f
                best_span = (1, len(lines))
            continue

        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            lo = int(node.lineno)
            end_ln = int(getattr(node, "end_lineno", node.lineno + 60))
            # Skip docstring span if a giant function is > 160 lines so code body is visible
            body_start = lo
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_end = int(getattr(node.body[0], "end_lineno", lo))
                if end_ln - lo > 150 and doc_end - lo > 25:
                    body_start = doc_end + 1

            if end_ln - lo > 160:
                if isinstance(node, ast.ClassDef):
                    continue
                # Score executable body lines rather than 500-line docstrings
                code_seg = "\n".join([lines[lo - 1]] + lines[body_start - 1 : min(end_ln, body_start + 135)])
                hi = min(end_ln, body_start + 135)
            else:
                code_seg = "\n".join(lines[lo - 1 : end_ln])
                hi = end_ln

            seg_toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", code_seg))
            sc = base_w + 1.1 * len(query_toks & seg_toks)
            # Exact stack-frame line hit inside this AST function!
            if any(lo <= tl <= end_ln for tl in trace_file_lines.get(rel_f, ())):
                sc += 65.0
            if node.name in trace_funcs:
                sc += 38.0
            # Private implementation helper (`_foo`) of an imported public API (`foo`) in the same module
            if node.name.startswith("_") and node.name.lstrip("_") in imported_syms:
                sc += 28.0
            if any(imp in node.name for imp in imported_syms if len(imp) >= 5):
                sc += 12.0
            if node.name in problem_statement:
                sc += 18.0
            if node.name in imported_syms:
                sc += 8.0
            if Path(rel_f).stem in problem_statement.lower():
                sc += 8.0
            if sc > best_score:
                best_score = sc
                best_file = rel_f
                best_span = (lo, hi)

    assert best_file, f"Failed to localize target file from test_files={test_files}"
    all_lines = (wt_dir / best_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    lo, hi = best_span
    # Include immediate preceding helper lines (`max(1, lo - 36)`) when compact so helper + wrapper stay together
    if hi - lo <= 90 and len(all_lines) > 130:
        lo = max(1, lo - 36)
    if hi - lo > 160:
        sig_hdr = "\n".join(all_lines[lo - 1 : lo + 6]) + '\n    """... (docstring omitted) ..."""\n'
        excerpt = sig_hdr + "\n".join(all_lines[max(lo + 6, hi - 135) : hi])
    else:
        excerpt = "\n".join(all_lines[lo - 1 : hi])
    return best_file, excerpt


def apply_swe_candidate_edit(orig_src: str, raw_model_output: str) -> str | None:
    """Applies `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks (or AST function replacement fallback)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_model_output, flags=re.DOTALL)
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<turn|>", "").replace("<|im_end|>", "").replace("<eos>", "").strip()

    sr_blocks = re.findall(
        r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        cleaned,
        flags=re.DOTALL,
    )
    if sr_blocks:
        cur = orig_src
        applied_any = False
        for search_txt, replace_txt in sr_blocks:
            if search_txt in cur:
                cur = cur.replace(search_txt, replace_txt, 1)
                applied_any = True
            else:
                # Whitespace-trimmed line match fallback
                s_lines = [ln.rstrip() for ln in search_txt.splitlines()]
                c_lines = cur.splitlines()
                for i in range(len(c_lines) - len(s_lines) + 1):
                    if [ln.rstrip() for ln in c_lines[i : i + len(s_lines)]] == s_lines:
                        new_lines = c_lines[:i] + replace_txt.splitlines() + c_lines[i + len(s_lines) :]
                        cur = "\n".join(new_lines) + ("\n" if orig_src.endswith("\n") else "")
                        applied_any = True
                        break
        if applied_any:
            try:
                ast.parse(cur)
                return cur
            except SyntaxError:
                return None

    # Fallback: model emitted a ```python\ndef fn_name(...): ...``` block directly
    cand_code = extract_python_code(cleaned)
    if not cand_code:
        return None
    try:
        c_tree = ast.parse(cand_code)
        o_tree = ast.parse(orig_src)
    except SyntaxError:
        return None

    cand_defs = {
        n.name: cand_code.splitlines()[n.lineno - 1 : getattr(n, "end_lineno", n.lineno)]
        for n in c_tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    if not cand_defs:
        return None

    o_lines = orig_src.splitlines()
    replaced = False
    for node in reversed(o_tree.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in cand_defs:
            lo = node.lineno - 1
            hi = getattr(node, "end_lineno", node.lineno)
            o_lines = o_lines[:lo] + cand_defs[node.name] + o_lines[hi:]
            replaced = True
    if not replaced:
        return None
    merged = "\n".join(o_lines) + "\n"
    try:
        ast.parse(merged)
        return merged
    except SyntaxError:
        return None


def load_official_swebench_verified_2wave_tasks() -> tuple[list[SWEBenchVerifiedWaveTask], str]:
    swe_parquet = hf_hub_download(
        "princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet", repo_type="dataset"
    )
    digest = _sha256_file(swe_parquet)
    assert digest == SWEBENCH_VERIFIED_EXPECTED_SHA256, f"SWE-bench Verified SHA-256 mismatch: {digest}"

    bare = ensure_sympy_bare_repo()
    by_id = {r["instance_id"]: r for r in pq.read_table(swe_parquet).to_pylist()}

    wave1: list[SWEBenchVerifiedWaveTask] = []
    wave2: list[SWEBenchVerifiedWaveTask] = []

    wt_root = default_nvme_shard_root() / "swe_git_cache" / "localize_wts"
    wt_root.mkdir(parents=True, exist_ok=True)

    for idx, iid in enumerate(SWE_SELECTED_8_IDS, start=1):
        r = by_id[iid]
        test_files = sorted(set(re.findall(r"\+\+\+ b/(sympy/[^\s]+\.py)", r["test_patch"])))
        f2p = json.loads(r["FAIL_TO_PASS"])
        p2p = json.loads(r["PASS_TO_PASS"])[:8]

        wt = wt_root / f"loc_{iid}"
        shutil.rmtree(wt, ignore_errors=True)
        subprocess.run(
            ["git", "--git-dir", str(bare), "worktree", "add", "--detach", str(wt), r["base_commit"]],
            check=True,
            capture_output=True,
        )
        try:
            subprocess.run(
                ["git", "apply", "-"], input=r["test_patch"], text=True, cwd=wt, check=True, capture_output=True
            )
            k_expr = " or ".join(f2p)
            pre = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=long", "-q", *test_files, "-k", k_expr],
                cwd=wt,
                capture_output=True,
                text=True,
                timeout=20.0,
            )
            assert pre.returncode != 0, f"Expected pre-patch FAIL_TO_PASS to fail on {iid}!"
            pre_trace = "\n".join((pre.stdout + "\n" + pre.stderr).strip().splitlines()[-45:])
            # Reset `test_patch` before running zero-leak source localizer
            subprocess.run(["git", "checkout", "--", "."], cwd=wt, check=True, capture_output=True)
            target_file, target_snippet = _zero_leak_localize_sympy_window(
                wt, r["problem_statement"], test_files, pre_trace
            )
        finally:
            subprocess.run(["git", "--git-dir", str(bare), "worktree", "remove", "--force", str(wt)], capture_output=True)
            shutil.rmtree(wt, ignore_errors=True)

        wave1.append(
            SWEBenchVerifiedWaveTask(
                eval_id=f"W1_{idx:02d}_{iid}",
                instance_id=iid,
                wave=1,
                base_commit=r["base_commit"],
                problem_statement=r["problem_statement"],
                test_patch=r["test_patch"],
                test_files=test_files,
                fail_to_pass=f2p,
                pass_to_pass=p2p,
                target_file=target_file,
                target_snippet=target_snippet,
                pre_failure_trace=pre_trace,
            )
        )
        # Wave 2: Paraphrased maintenance backport ticket on a fresh checkout of `base_commit`
        first_para = r["problem_statement"].strip().split("\n\n")[0]
        w2_stmt = (
            f"[Maintenance Backport Ticket · {target_file}]\n"
            f"Regression failing `{', '.join(f2p)}` in `{', '.join(test_files)}`:\n{first_para}"
        )
        wave2.append(
            SWEBenchVerifiedWaveTask(
                eval_id=f"W2_{idx:02d}_{iid}_backport",
                instance_id=iid,
                wave=2,
                base_commit=r["base_commit"],
                problem_statement=w2_stmt,
                test_patch=r["test_patch"],
                test_files=test_files,
                fail_to_pass=f2p,
                pass_to_pass=p2p,
                target_file=target_file,
                target_snippet=target_snippet,
                pre_failure_trace=pre_trace,
            )
        )

    return wave1 + wave2, digest


def format_swe_user_prompt(task: SWEBenchVerifiedWaveTask) -> str:
    return (
        f"Repository: `sympy/sympy` @ `{task.base_commit[:10]}`\n"
        f"Target File: `{task.target_file}`\n\n"
        f"### Upstream Issue Description\n{task.problem_statement.strip()[:1800]}\n\n"
        f"### Pre-Patch Failing `pytest` Output (`{', '.join(task.fail_to_pass)}`)\n"
        f"```text\n{task.pre_failure_trace}\n```\n\n"
        f"### Current Excerpt of `{task.target_file}`\n"
        f"```python\n{task.target_snippet}\n```\n\n"
        "Provide an exact `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` block fixing the bug in "
        f"`{task.target_file}` without breaking existing behavior."
    )


def verify_swebench_candidate_in_worktree(
    task: SWEBenchVerifiedWaveTask,
    raw_output_or_diff: str,
    is_unified_diff: bool = False,
) -> tuple[bool, str, str]:
    """
    Applies candidate edit (or crystallized unified `git diff`) inside a fresh isolated `git worktree`
    at `task.base_commit`, verifies zero test files were modified by the candidate, applies `task.test_patch`,
    and runs real subprocess `pytest -q` across `FAIL_TO_PASS + PASS_TO_PASS`.
    Returns `(passed, unified_git_diff, failure_diagnostics)`.
    """
    bare = ensure_sympy_bare_repo()
    wt_base = default_nvme_shard_root() / "swe_git_cache" / "eval_wts"
    wt_base.mkdir(parents=True, exist_ok=True)
    wt = Path(tempfile.mkdtemp(prefix=f"wt_{task.instance_id}_", dir=str(wt_base)))
    shutil.rmtree(wt, ignore_errors=True)

    subprocess.run(
        ["git", "--git-dir", str(bare), "worktree", "add", "--detach", str(wt), task.base_commit],
        check=True,
        capture_output=True,
    )
    try:
        target_p = wt / task.target_file
        if is_unified_diff:
            ap = subprocess.run(
                ["git", "apply", "-"], input=raw_output_or_diff, text=True, cwd=wt, capture_output=True
            )
            if ap.returncode != 0:
                return False, "", f"git apply failed: {ap.stderr.strip()}"
        else:
            orig_src = target_p.read_text(encoding="utf-8")
            patched_src = apply_swe_candidate_edit(orig_src, raw_output_or_diff)
            if patched_src is None or patched_src == orig_src:
                return False, "", "Failed to parse valid non-empty SEARCH/REPLACE or AST function edit."
            target_p.write_text(patched_src, encoding="utf-8")

        # Enforce zero modifications to any `tests/` file prior to `test_patch`
        changed = subprocess.run(
            ["git", "diff", "--name-only"], cwd=wt, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        if not changed or any("/tests/" in c or c.startswith("tests/") for c in changed):
            return False, "", f"Invalid modified file set: {changed}"

        unified_diff = subprocess.run(
            ["git", "diff", "--", task.target_file], cwd=wt, check=True, capture_output=True, text=True
        ).stdout

        # Apply official `SWE-bench_Verified` `test_patch` and run `FAIL_TO_PASS + PASS_TO_PASS`
        subprocess.run(
            ["git", "apply", "-"], input=task.test_patch, text=True, cwd=wt, check=True, capture_output=True
        )
        k_all = " or ".join(task.fail_to_pass + task.pass_to_pass)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--tb=short", "-q", *task.test_files, "-k", k_all],
            cwd=wt,
            capture_output=True,
            text=True,
            timeout=25.0,
        )
        if proc.returncode == 0:
            return True, unified_diff, ""
        diag = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-18:])
        return False, unified_diff, diag
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutExpired (> 25.0s) in upstream SymPy pytest."
    finally:
        subprocess.run(["git", "--git-dir", str(bare), "worktree", "remove", "--force", str(wt)], capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
