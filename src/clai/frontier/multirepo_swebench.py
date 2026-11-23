"""Phase 28 (Claim LA-20): Multi-Repository Upstream `princeton-nlp/SWE-bench_Verified`
Evaluation (`django/django`, `pytest-dev/pytest`, `psf/requests`, `sympy/sympy`) with a
Zero-Gold-Leakage Multi-Window AST Call-Graph Localizer and Live `.mol/` Unified-Diff Replay.

Strict Invariants:
  1. Official `princeton-nlp/SWE-bench_Verified` Parquet (`SHA-256: a45b1fe4...856808f3679c07812e9e6dcd`).
  2. Zero Gold-Patch Leakage: `row["patch"]` is NEVER inspected by any localizer, prompt, or model arm.
  3. Multi-Window AST Call-Graph Localizer:
       - Locates the primary AST function/method/class from pre-patch failing traceback frames,
         test-file imports, mirror test->source paths, and `problem_statement` identifiers.
       - Walks `ast.Call` edges (`helper(...)` / `self._helper(...)`) inside the primary AST node
         and extracts a second non-overlapping intra-file AST window so helper + caller edits are
         jointly visible inside a compact 2-window excerpt.
  4. Real Isolated `git worktree` Verification across 4 Distinct Upstream Repositories:
       - `django/django` (`tests/runtests.py`)
       - `pytest-dev/pytest` (`pytest` self-hosting test suite)
       - `psf/requests` (`pytest` HTTP client/proxy/URL suite)
       - `sympy/sympy` (`pytest` symbolic algebra/tensor suite)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from clai.frontier.external_public_benchmarks import (
    SWEBENCH_VERIFIED_EXPECTED_SHA256,
    _sha256_file,
    apply_swe_candidate_edit,
)
from clai.frontier.moe_air_mol import default_nvme_shard_root


MULTIREPO_SELECTED_12_IDS: list[str] = [
    # 1. `django/django` (4 instances)
    "django__django-17087",
    "django__django-17084",
    "django__django-17029",
    "django__django-16950",
    # 2. `pytest-dev/pytest` (3 instances)
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "pytest-dev__pytest-10356",
    # 3. `psf/requests` (2 instances)
    "psf__requests-6028",
    "psf__requests-5414",
    # 4. `sympy/sympy` (3 instances, including multi-span `23824`)
    "sympy__sympy-23413",
    "sympy__sympy-23824",
    "sympy__sympy-24213",
]

REPO_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "django/django": ("django.git", "https://github.com/django/django.git", ("django/",)),
    "pytest-dev/pytest": ("pytest.git", "https://github.com/pytest-dev/pytest.git", ("src/_pytest/",)),
    "psf/requests": ("requests.git", "https://github.com/psf/requests.git", ("requests/", "src/requests/")),
    "sympy/sympy": ("sympy.git", "https://github.com/sympy/sympy.git", ("sympy/",)),
}


@dataclass
class MultiRepoSWETask:
    eval_id: str
    repo: str
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
    num_ast_windows: int
    pre_failure_trace: str


def ensure_repo_bare_cache(repo: str) -> Path:
    bare_name, clone_url, _ = REPO_SPECS[repo]
    cache_dir = default_nvme_shard_root() / "swe_git_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bare = cache_dir / bare_name
    if not bare.exists():
        subprocess.run(
            ["git", "clone", "--bare", "--filter=blob:none", clone_url, str(bare)],
            check=True,
        )
    return bare


def _is_test_path(rel_path: str, problem_statement: str = "") -> bool:
    if (
        rel_path.startswith(("tests/", "testing/"))
        or "/tests/" in rel_path
        or Path(rel_path).name.startswith("test_")
    ):
        return True
    # Exclude framework test-runner harness files (`django/test/runner.py`, `django/test/testcases.py`, `sympy/testing/`)
    # that appear in every `unittest` assertion stack trace unless explicitly named in the issue description.
    if rel_path.startswith(("django/test/", "sympy/testing/")) and rel_path not in problem_statement:
        return True
    return False


def _run_repo_test_command(
    repo: str,
    wt: Path,
    test_files: list[str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    include_p2p: bool,
    tb_mode: str = "short",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if repo == "django/django":
        env["PYTHONPATH"] = str(wt)
        labels = [
            re.sub(r"^tests/", "", tf[:-3]).replace("/", ".")
            for tf in test_files
            if tf.startswith("tests/") and tf.endswith(".py")
        ]
        cmd = [sys.executable, "tests/runtests.py", "--verbosity=2", *labels]
        return subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=30.0)

    if repo == "pytest-dev/pytest":
        ver_file = wt / "src" / "_pytest" / "_version.py"
        if not ver_file.exists():
            ver_file.write_text("version = '7.4.0.dev0'\nversion_tuple = (7, 4, 0, 'dev0')\n", encoding="utf-8")
        env["PYTHONPATH"] = f"{wt}/src:{wt}"
        targets = list(fail_to_pass) + (list(pass_to_pass[:6]) if include_p2p else [])
        cmd = [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-m",
            "pytest",
            "-o",
            "filterwarnings=ignore",
            f"--tb={tb_mode}",
            "-q",
            *targets,
        ]
        return subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=25.0)

    if repo == "psf/requests":
        env["PYTHONPATH"] = f"{wt}/src:{wt}"
        targets = list(fail_to_pass) + (list(pass_to_pass[:6]) if include_p2p else [])
        cmd = [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-m",
            "pytest",
            f"--tb={tb_mode}",
            "-q",
            *targets,
        ]
        return subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=20.0)

    # `sympy/sympy`
    env["PYTHONPATH"] = str(wt)
    k_ids = list(fail_to_pass) + (list(pass_to_pass[:8]) if include_p2p else [])
    k_expr = " or ".join(k_ids)
    cmd = [sys.executable, "-m", "pytest", f"--tb={tb_mode}", "-q", *test_files, "-k", k_expr]
    return subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=25.0)


def _extract_called_names(node: ast.AST) -> set[str]:
    called: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                called.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                called.add(sub.func.attr)
    return called


def zero_leak_localize_multirepo_callgraph(
    repo: str,
    wt_dir: Path,
    problem_statement: str,
    test_files: list[str],
    pre_trace: str,
) -> tuple[str, str, int]:
    """
    Zero-gold-leakage Multi-Window AST Call-Graph Localizer across `django`, `pytest`, `requests`, `sympy`.
    Returns `(target_rel_file, multi_window_excerpt, num_windows)`.
    """
    _, _, src_prefixes = REPO_SPECS[repo]
    cand_files: dict[str, float] = {}
    trace_file_lines: dict[str, list[int]] = {}

    # 1. Parse stack frames from both `pytest` (`path/foo.py:123:`) and Python `unittest` (`File ".../foo.py", line 123`)
    wt_str = str(wt_dir.resolve())
    for abs_or_rel, lno_s in re.findall(r'File "([^"]+\.py)", line (\d+)', pre_trace):
        rel = abs_or_rel
        if rel.startswith(wt_str):
            rel = rel[len(wt_str) :].lstrip("/")
        elif str(wt_dir) in rel:
            rel = rel.split(str(wt_dir), 1)[1].lstrip("/")
        if any(rel.startswith(p) for p in src_prefixes) and not _is_test_path(rel, problem_statement) and (wt_dir / rel).is_file():
            cand_files[rel] = min(48.0, cand_files.get(rel, 0.0) + 34.0)
            trace_file_lines.setdefault(rel, []).append(int(lno_s))

    for rel_f, lno_s in re.findall(r"([A-Za-z0-9_./-]+\.py):(\d+):", pre_trace):
        rel_f = rel_f.lstrip("./")
        if any(rel_f.startswith(p) for p in src_prefixes) and not _is_test_path(rel_f, problem_statement) and (wt_dir / rel_f).is_file():
            cand_files[rel_f] = min(48.0, cand_files.get(rel_f, 0.0) + 34.0)
            trace_file_lines.setdefault(rel, []).append(int(lno_s)) if False else trace_file_lines.setdefault(rel_f, []).append(int(lno_s))

    # 2. Explicit file or dotted-module mentions inside `problem_statement` + `pre_trace`
    combined_txt = problem_statement + "\n" + pre_trace
    for rel_f in re.findall(r"([A-Za-z0-9_/-]+/[A-Za-z0-9_/-]+\.py)", combined_txt):
        for pfx in ("", "src/"):
            cand = (pfx + rel_f).lstrip("/")
            if any(cand.startswith(p) for p in src_prefixes) and not _is_test_path(cand) and (wt_dir / cand).is_file():
                cand_files[cand] = cand_files.get(cand, 0.0) + 18.0

    for dotted in re.findall(r"\b((?:django|_pytest|requests|sympy)(?:\.[A-Za-z0-9_]+)+)\b", combined_txt):
        for pfx in ("", "src/"):
            as_file = pfx + dotted.replace(".", "/") + ".py"
            if not _is_test_path(as_file) and (wt_dir / as_file).is_file():
                cand_files[as_file] = cand_files.get(as_file, 0.0) + 16.0
            parent_mod = pfx + "/".join(dotted.split(".")[:-1]) + ".py"
            if not _is_test_path(parent_mod) and (wt_dir / parent_mod).is_file():
                cand_files[parent_mod] = cand_files.get(parent_mod, 0.0) + 14.0

    # 3. Imports inside `test_files` + repository mirror conventions (`testing/test_capture.py` -> `src/_pytest/capture.py`)
    imported_syms: set[str] = set()
    for tf in test_files:
        tf_p = wt_dir / tf
        if tf_p.is_file():
            txt = tf_p.read_text(encoding="utf-8", errors="ignore")
            for mod_dot, names_str in re.findall(
                r"from\s+((?:django|_pytest|pytest|requests|sympy)(?:\.[A-Za-z0-9_]+)*)\s+import\s+([^\n#]+)", txt
            ):
                for nm in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", names_str):
                    if nm not in ("as", "import"):
                        imported_syms.add(nm)
                for pfx in ("", "src/"):
                    mod_rel = pfx + mod_dot.replace(".", "/") + ".py"
                    if not _is_test_path(mod_rel) and (wt_dir / mod_rel).is_file():
                        cand_files[mod_rel] = cand_files.get(mod_rel, 0.0) + 10.0
                    else:
                        init_dir = wt_dir / (pfx + mod_dot.replace(".", "/"))
                        if init_dir.is_dir():
                            for py_p in init_dir.glob("*.py"):
                                rel_p = str(py_p.relative_to(wt_dir))
                                if not _is_test_path(rel_p):
                                    cand_files[rel_p] = cand_files.get(rel_p, 0.0) + 4.5

        stem = Path(tf).stem
        clean_stem = re.sub(r"^test_", "", stem)
        clean_stem = re.sub(r"_tests?$", "", clean_stem)
        # Scan repo source tree for matching module stems (e.g., `test_writer` -> `django/db/migrations/writer.py`,
        # `test_capture` -> `src/_pytest/capture.py`, `sympy/.../tests/test_foo.py` -> `sympy/.../foo.py`)
        parent_parts = set(Path(tf).parts)
        for pfx in src_prefixes:
            root_d = wt_dir / pfx.rstrip("/")
            if not root_d.exists():
                continue
            for py_p in root_d.rglob("*.py"):
                rel_p = str(py_p.relative_to(wt_dir))
                if _is_test_path(rel_p):
                    continue
                if clean_stem and clean_stem not in ("tests", "test") and py_p.stem in (
                    clean_stem,
                    clean_stem.rstrip("s"),
                    f"{clean_stem}s",
                ):
                    bonus = 24.0 + 8.0 * len(parent_parts & set(Path(rel_p).parts))
                    cand_files[rel_p] = cand_files.get(rel_p, 0.0) + bonus

    # Fallback for small flat packages like `psf/requests` (`requests/*.py` has only ~15 files!)
    if len(cand_files) < 4:
        for pfx in src_prefixes:
            root_d = wt_dir / pfx.rstrip("/")
            if root_d.is_dir():
                py_list = [p for p in root_d.glob("*.py") if not _is_test_path(str(p.relative_to(wt_dir)))]
                if len(py_list) <= 25:
                    for py_p in py_list:
                        rel_p = str(py_p.relative_to(wt_dir))
                        cand_files[rel_p] = cand_files.get(rel_p, 0.0) + 3.0

    trace_funcs = set(re.findall(r"\bin\s+([A-Za-z_][A-Za-z0-9_]*)\b", pre_trace))
    prob_ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", problem_statement))
    query_toks = prob_ids | set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", pre_trace)) | imported_syms

    best_file = ""
    best_file_score = -1.0
    file_node_table: dict[str, list[tuple[float, int, int, str, ast.AST]]] = {}

    for rel_f, base_w in cand_files.items():
        src = (wt_dir / rel_f).read_text(encoding="utf-8", errors="ignore")
        lines = src.splitlines()
        if not lines:
            continue
        stem_bonus = 12.0 if Path(rel_f).stem in problem_statement.lower() else 0.0
        if len(lines) <= 130:
            sc = base_w + stem_bonus + 1.4 * len(query_toks & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", src)))
            file_node_table[rel_f] = [(sc, 1, len(lines), "<module>", ast.Module(body=[], type_ignores=[]))]
            if sc > best_file_score:
                best_file_score = sc
                best_file = rel_f
            continue

        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        scored_nodes: list[tuple[float, int, int, str, ast.AST]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            lo = int(node.lineno)
            end_ln = int(getattr(node, "end_lineno", lo + 50))
            if isinstance(node, ast.ClassDef) and (end_ln - lo) > 120:
                # Score individual methods inside large classes rather than the 800-line class wrapper
                continue

            body_start = lo
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc_end = int(getattr(node.body[0], "end_lineno", lo))
                if end_ln - lo > 140 and doc_end - lo > 20:
                    body_start = doc_end + 1

            hi = min(end_ln, body_start + 130) if (end_ln - lo) > 150 else end_ln
            code_seg = "\n".join([lines[lo - 1]] + lines[body_start - 1 : hi])
            seg_toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", code_seg))

            sc = base_w + stem_bonus + 1.15 * len(query_toks & seg_toks)
            if any(lo <= tl <= end_ln for tl in trace_file_lines.get(rel_f, ())):
                sc += 70.0
            if node.name in trace_funcs:
                sc += 40.0
            if node.name in prob_ids:
                sc += 28.0
            if node.name.startswith("_") and node.name.lstrip("_") in (imported_syms | prob_ids | trace_funcs):
                sc += 30.0
            if node.name in imported_syms:
                sc += 14.0
            scored_nodes.append((sc, lo, hi, node.name, node))

        if scored_nodes:
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            file_node_table[rel_f] = scored_nodes
            top_sc = scored_nodes[0][0]
            if top_sc > best_file_score:
                best_file_score = top_sc
                best_file = rel_f

    assert best_file, f"Multi-repo localizer failed on {repo} ({test_files})"
    all_lines = (wt_dir / best_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    nodes = file_node_table[best_file]
    _, p_lo, p_hi, _, p_node = nodes[0]

    # Expand primary window slightly upward (`-16` lines) to capture decorators / class header / constants
    w1_lo = max(1, p_lo - 16)
    w1_hi = min(len(all_lines), p_hi + 8)

    # Multi-Window Call-Graph Expansion: boost intra-file callees/callers of `p_node` and select Window 2!
    called_by_primary = _extract_called_names(p_node)
    secondary_candidates: list[tuple[float, int, int]] = []
    for sc, lo, hi, nm, nd in nodes[1:]:
        # Skip spans already covered (>60%) by Window 1
        overlap = max(0, min(w1_hi, hi) - max(w1_lo, lo))
        if overlap >= 0.6 * max(1, hi - lo):
            continue
        cg_bonus = 0.0
        if nm in called_by_primary:
            cg_bonus += 48.0
        if nodes[0][3] in _extract_called_names(nd):
            cg_bonus += 34.0
        secondary_candidates.append((sc + cg_bonus, max(1, lo - 6), min(len(all_lines), hi + 6)))

    windows: list[tuple[int, int]] = [(w1_lo, w1_hi)]
    if secondary_candidates:
        secondary_candidates.sort(key=lambda x: x[0], reverse=True)
        _, s_lo, s_hi = secondary_candidates[0]
        if (s_hi - s_lo) <= 110:
            windows.append((s_lo, s_hi))

    windows.sort(key=lambda sp: sp[0])
    # Merge overlapping or nearly-touching (`gap <= 10` lines) windows so context is contiguous
    merged: list[tuple[int, int]] = []
    for lo, hi in windows:
        if merged and lo <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    rendered_parts: list[str] = []
    for idx_w, (lo, hi) in enumerate(merged):
        if idx_w > 0:
            rendered_parts.append("# ... (intermediate unchanged lines omitted) ...")
        if hi - lo > 165:
            hdr = "\n".join(all_lines[lo - 1 : lo + 8]) + '\n    """... (docstring omitted) ..."""'
            rendered_parts.append(hdr + "\n" + "\n".join(all_lines[max(lo + 8, hi - 140) : hi]))
        else:
            rendered_parts.append("\n".join(all_lines[lo - 1 : hi]))

    return best_file, "\n".join(rendered_parts), len(merged)


def load_multirepo_swebench_2wave_tasks() -> tuple[list[MultiRepoSWETask], str]:
    swe_parquet = hf_hub_download(
        "princeton-nlp/SWE-bench_Verified", "data/test-00000-of-00001.parquet", repo_type="dataset"
    )
    digest = _sha256_file(swe_parquet)
    assert digest == SWEBENCH_VERIFIED_EXPECTED_SHA256, f"SWE-bench Verified SHA-256 mismatch: {digest}"

    by_id = {r["instance_id"]: r for r in pq.read_table(swe_parquet).to_pylist()}
    wt_root = default_nvme_shard_root() / "swe_git_cache" / "multirepo_loc_wts"
    wt_root.mkdir(parents=True, exist_ok=True)

    wave1: list[MultiRepoSWETask] = []
    wave2: list[MultiRepoSWETask] = []

    for idx, iid in enumerate(MULTIREPO_SELECTED_12_IDS, start=1):
        r = by_id[iid]
        repo = r["repo"]
        bare = ensure_repo_bare_cache(repo)
        test_files = sorted(set(re.findall(r"\+\+\+ b/([^\s]+\.py)", r["test_patch"])))
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
            pre = _run_repo_test_command(repo, wt, test_files, f2p, p2p, include_p2p=False, tb_mode="long")
            assert pre.returncode != 0, f"Expected pre-patch FAIL_TO_PASS to fail on {iid}!"
            pre_trace = "\n".join((pre.stdout + "\n" + pre.stderr).strip().splitlines()[-50:])

            # Reset `test_patch` & any untracked build helper before running zero-leak source localizer
            subprocess.run(["git", "checkout", "--", "."], cwd=wt, check=True, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=wt, check=True, capture_output=True)

            target_file, target_snippet, n_win = zero_leak_localize_multirepo_callgraph(
                repo, wt, r["problem_statement"], test_files, pre_trace
            )
        finally:
            subprocess.run(["git", "--git-dir", str(bare), "worktree", "remove", "--force", str(wt)], capture_output=True)
            shutil.rmtree(wt, ignore_errors=True)

        wave1.append(
            MultiRepoSWETask(
                eval_id=f"W1_{idx:02d}_{iid}",
                repo=repo,
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
                num_ast_windows=n_win,
                pre_failure_trace=pre_trace,
            )
        )
        first_para = r["problem_statement"].strip().split("\n\n")[0]
        w2_stmt = (
            f"[Maintenance Backport Ticket · {repo} · {target_file}]\n"
            f"Regression failing `{', '.join(f2p[:3])}` in `{', '.join(test_files)}`:\n{first_para}"
        )
        wave2.append(
            MultiRepoSWETask(
                eval_id=f"W2_{idx:02d}_{iid}_backport",
                repo=repo,
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
                num_ast_windows=n_win,
                pre_failure_trace=pre_trace,
            )
        )

    return wave1 + wave2, digest


def format_multirepo_swe_prompt(task: MultiRepoSWETask) -> str:
    return (
        f"Repository: `{task.repo}` @ `{task.base_commit[:10]}`\n"
        f"Target File: `{task.target_file}` ({task.num_ast_windows}-window call-graph excerpt)\n\n"
        f"### Upstream Issue Description\n{task.problem_statement.strip()[:1900]}\n\n"
        f"### Pre-Patch Failing Test Output (`{', '.join(task.fail_to_pass[:3])}`)\n"
        f"```text\n{task.pre_failure_trace}\n```\n\n"
        f"### Current Excerpt of `{task.target_file}`\n"
        f"```python\n{task.target_snippet}\n```\n\n"
        "Provide one or more exact `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks fixing the bug in "
        f"`{task.target_file}` without breaking existing behavior."
    )


def verify_multirepo_swebench_in_worktree(
    task: MultiRepoSWETask,
    raw_output_or_diff: str,
    is_unified_diff: bool = False,
) -> tuple[bool, str, str]:
    """
    Applies candidate `SEARCH/REPLACE` edit (or crystallized unified `git diff`) inside a fresh isolated
    `git worktree` of `task.repo` at `task.base_commit`, verifies zero test files were modified by the
    candidate, applies `task.test_patch`, and executes the repository's native test runner.
    """
    bare = ensure_repo_bare_cache(task.repo)
    wt_base = default_nvme_shard_root() / "swe_git_cache" / "multirepo_eval_wts"
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

        changed = subprocess.run(
            ["git", "diff", "--name-only"], cwd=wt, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        if not changed or any(_is_test_path(c) for c in changed):
            return False, "", f"Invalid modified file set: {changed}"

        unified_diff = subprocess.run(
            ["git", "diff", "--", task.target_file], cwd=wt, check=True, capture_output=True, text=True
        ).stdout

        subprocess.run(
            ["git", "apply", "-"], input=task.test_patch, text=True, cwd=wt, check=True, capture_output=True
        )
        proc = _run_repo_test_command(
            task.repo, wt, task.test_files, task.fail_to_pass, task.pass_to_pass, include_p2p=True, tb_mode="short"
        )
        if proc.returncode == 0:
            return True, unified_diff, ""
        diag = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-20:])
        return False, unified_diff, diag
    except subprocess.TimeoutExpired:
        return False, "", f"TimeoutExpired in {task.repo} test suite."
    finally:
        subprocess.run(["git", "--git-dir", str(bare), "worktree", "remove", "--force", str(wt)], capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
