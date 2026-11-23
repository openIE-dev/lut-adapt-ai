"""Phase 26 (Track 1B): Multi-File Git-Repository `pytest` Patching Cascade (`SWE-Mini` / `RepoPatchCascade`).

Evaluates real multi-file Python repository bug-fixing where:
  1. Each task materializes an isolated multi-module Python git repository (`git init`) containing
     cross-file source modules (`pkg/*.py`) and a locked `tests/test_repo.py` suite that initially FAILS.
  2. The contract verifier `C(z) in {0, 1}` applies candidate file patches inside a sandboxed worktree,
     STRICTLY forbids any modification to `tests/`, and executes real `.venv/bin/python -m pytest -q`
     in a timeout-bounded subprocess (`<= 8.0 s`), returning the exact `pytest` failure traceback as
     counterexample feedback when `C(z) = 0`.
  3. Verified patches are committed cleanly (`git diff` receipt) and crystallized downward into a
     persistent `.mol/` repository skill store indexed by a `512`-wire `BenesLUT6` issue-signature router!
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import torch

from clai.contracts.joule_meter import meter_joules
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    ExpertAirMoL35BRunner,
    Local2026ModelRunner,
)


REPO_PATCH_SYSTEM_PROMPT = (
    "You are an expert Python systems engineer fixing a multi-file repository bug.\n"
    "Read all provided repository source files and the failing `pytest` output carefully.\n"
    "Do NOT modify any file under `tests/`.\n"
    "Output ONLY the modified source file(s) using one or more fenced blocks in this exact format:\n"
    "```python path=pkg/module.py\n"
    "<complete updated contents of pkg/module.py>\n"
    "```\n"
    "Include NO prose outside the fenced file block(s)."
)


@dataclass
class MultiFileRepoBugTask:
    task_id: str
    family_id: str
    wave: int
    issue_title: str
    issue_description: str
    source_files: dict[str, str]
    test_files: dict[str, str]
    spec_mutated: bool = False


@dataclass
class RepoPatchSummaryRow:
    arm: str
    wave1_cold_pass_pct: float
    wave2_warm_pass_pct: float
    overall_verified_pass_pct: float
    broken_patches_committed_pct: float
    rung0_rung1_hit_pct_wave2: float
    mean_latency_ms_wave1: float
    mean_latency_ms_wave2: float
    est_joules_per_verified_fix: float
    reported_joules_per_verified_fix: float | None = None


def parse_file_patches(raw_output: str, valid_source_paths: list[str]) -> dict[str, str]:
    """
    Extracts `{rel_path: full_file_content}` from LLM output supporting:
      - ````python path=pkg/foo.py` or ````python pkg/foo.py`
      - `# FILE: pkg/foo.py` inside a fence
      - Single-file fallback when exactly one non-`__init__.py` module matches AST symbols
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<turn|>", "").replace("<|im_end|>", "").replace("<eos>", "").strip()

    patches: dict[str, str] = {}
    valid_set = set(valid_source_paths)

    # Pattern 1: ```python path=foo/bar.py  OR  ```python foo/bar.py
    block_pat = re.compile(r"```(?:python|py)?(?:[ \t]+(?:path=)?([^\n`]+))?\n(.*?)```", flags=re.DOTALL)
    untagged_blocks: list[str] = []
    for hdr, body in block_pat.findall(cleaned):
        hdr_clean = (hdr or "").strip().strip("'\"")
        body_str = body.strip() + "\n"
        matched_path = None
        if hdr_clean:
            if hdr_clean in valid_set:
                matched_path = hdr_clean
            else:
                for vp in valid_source_paths:
                    if vp.endswith(hdr_clean) or Path(vp).name == Path(hdr_clean).name:
                        matched_path = vp
                        break
        if matched_path is None:
            first_ln = body.lstrip().splitlines()[0] if body.strip() else ""
            m_f = re.match(r"^#\s*(?:FILE|file|path)\s*:\s*(\S+)", first_ln)
            if m_f:
                cand_p = m_f.group(1).strip()
                for vp in valid_source_paths:
                    if vp == cand_p or vp.endswith(cand_p):
                        matched_path = vp
                        break
        if matched_path is not None and not matched_path.startswith("tests/"):
            patches[matched_path] = body_str
        elif body.strip():
            untagged_blocks.append(body_str)

    # Fallback: match untagged code blocks to source files by defined class/function names
    if not patches and untagged_blocks:
        for blk in untagged_blocks:
            defs = set(re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", blk, flags=re.MULTILINE))
            if not defs:
                continue
            best_vp = None
            best_overlap = 0
            for vp in valid_source_paths:
                if vp.endswith("__init__.py"):
                    continue
                best_cand = vp
                best_overlap_chk = 1
                _ = best_cand, best_overlap_chk
            # Match against symbol names in original files passed by caller if needed
            if len([vp for vp in valid_source_paths if not vp.endswith("__init__.py")]) == 1:
                only_vp = [vp for vp in valid_source_paths if not vp.endswith("__init__.py")][0]
                patches[only_vp] = blk
                break

    return patches


def match_untagged_blocks_by_symbols(raw_output: str, orig_sources: dict[str, str]) -> dict[str, str]:
    """Resolves untagged fenced Python blocks to the exact source file sharing `def`/`class` symbols."""
    patches = parse_file_patches(raw_output, list(orig_sources.keys()))
    if patches:
        return patches
    cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<turn|>", "").replace("<|im_end|>", "").replace("<eos>", "").strip()
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", cleaned, flags=re.DOTALL)
    for blk in blocks:
        defs = set(re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", blk, flags=re.MULTILINE))
        if not defs:
            continue
        best_p = None
        best_ov = 0
        for rel_p, src in orig_sources.items():
            if rel_p.endswith("__init__.py") or rel_p.startswith("tests/"):
                continue
            src_defs = set(re.findall(r"^(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", src, flags=re.MULTILINE))
            ov = len(defs & src_defs)
            if ov > best_ov:
                best_ov = ov
                best_p = rel_p
        if best_p is not None:
            patches[best_p] = blk.strip() + "\n"
    return patches


def materialize_git_repo(root: Path, source_files: dict[str, str], test_files: dict[str, str]) -> None:
    """Creates an isolated git repository on disk with initial buggy source + locked test suite."""
    for rel_p, content in {**source_files, **test_files}.items():
        fp = root / rel_p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "mol@openie.dev"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "OpenIE-MoL"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial buggy state"], cwd=root, check=True)


def run_repo_pytest(repo_dir: Path, timeout_s: float = 8.0) -> tuple[bool, str]:
    """Runs real `pytest` in an isolated subprocess and returns `(passed, stdout_stderr_tail)`."""
    env = {**os.environ, "PYTHONPATH": str(repo_dir), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "--disable-warnings"],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired: pytest exceeded {timeout_s:.1f}s (infinite loop in candidate patch)."
    combined = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, combined[-1400:]


def verify_repo_patch_candidate(
    task: MultiFileRepoBugTask,
    raw_patch_or_json: str,
) -> tuple[bool, str, str, str]:
    """
    Deterministic Multi-File Git-Repo `pytest` Contract Verifier `C(z) in {0, 1}`:
      1. Parses candidate file edits (either JSON dict from crystallized `.mol/` skill or raw LLM blocks).
      2. Rejects empty patches, syntax errors, or any attempt to touch `tests/`.
      3. Materializes an isolated temporary git repository, applies the patch, verifies `tests/` is byte-identical,
         runs `.venv/bin/python -m pytest -q`, and captures `git diff` upon `C(z) == 1`.
    Returns `(passed, canonical_patch_json, git_diff_text, diagnostic_traceback)`.
    """
    if raw_patch_or_json.lstrip().startswith("{"):
        try:
            patches = json.loads(raw_patch_or_json)
        except json.JSONDecodeError:
            patches = match_untagged_blocks_by_symbols(raw_patch_or_json, task.source_files)
    else:
        patches = match_untagged_blocks_by_symbols(raw_patch_or_json, task.source_files)

    if not patches or not isinstance(patches, dict):
        return False, "", "", "No valid `path=...` Python file patch block found in candidate output."

    for rel_p, content in patches.items():
        if rel_p.startswith("tests/") or "test_" in Path(rel_p).name:
            return False, "", "", f"Security violation: candidate attempted to modify test file `{rel_p}`."
        if rel_p not in task.source_files:
            return False, "", "", f"Unknown target file `{rel_p}` (valid: {sorted(task.source_files.keys())})."
        try:
            compile(content, rel_p, "exec")
        except SyntaxError as exc:
            return False, "", "", f"SyntaxError in `{rel_p}` line {exc.lineno}: {exc.msg}"

    with tempfile.TemporaryDirectory(prefix="mol_swebench_") as tmp:
        repo_dir = Path(tmp)
        materialize_git_repo(repo_dir, task.source_files, task.test_files)
        for rel_p, content in patches.items():
            (repo_dir / rel_p).write_text(content, encoding="utf-8")

        # Verify locked test files were untouched
        for t_rel, t_orig in task.test_files.items():
            if (repo_dir / t_rel).read_text(encoding="utf-8") != t_orig:
                return False, "", "", f"Locked test file `{t_rel}` was altered."

        ok, pytest_out = run_repo_pytest(repo_dir)
        diff_proc = subprocess.run(["git", "diff"], cwd=repo_dir, capture_output=True, text=True, check=False)
        git_diff = diff_proc.stdout.strip()
        if ok:
            subprocess.run(["git", "commit", "-q", "-am", f"MoL verified fix for {task.task_id}"], cwd=repo_dir, check=True)
            canonical_json = json.dumps(dict(sorted(patches.items())), sort_keys=True)
            return True, canonical_json, git_diff, ""
        return False, "", git_diff, pytest_out


def format_repo_bug_prompt(task: MultiFileRepoBugTask, initial_pytest_tb: str) -> str:
    """Formats the multi-file repository context + failing `pytest` traceback for LLM synthesis."""
    parts = [
        f"ISSUE `{task.task_id}` — {task.issue_title}",
        task.issue_description,
        "\nREPOSITORY SOURCE FILES:",
    ]
    for rel_p, src in sorted(task.source_files.items()):
        parts.append(f"```python path={rel_p}\n{src.rstrip()}\n```")
    parts.append(f"\nFAILING `pytest -q` OUTPUT:\n```text\n{initial_pytest_tb}\n```")
    parts.append(
        "\nReturn ONLY the corrected source file(s) using ` ```python path=<rel_path> ` blocks. "
        "Do NOT touch `tests/`."
    )
    return "\n".join(parts)


def build_swebench_mini_repo_tasks() -> list[MultiFileRepoBugTask]:
    """
    Constructs `8` cross-module Python repository bugs x `2` waves = `16` isolated git-repo tasks
    (including `1` Wave-2 adversarial specification change on `R03_jwt_clock_skew`!).
    """
    specs = [
        # R01: Billing Cents Proration Remainder Conservation (`billing/types.py` + `billing/prorate.py`)
        (
            "R01_billing_proration",
            "Fix integer-cents proration losing remainder cents across line-item weights",
            (
                "`allocate_prorated_cents(total_cents, items)` in `billing/prorate.py` allocates `total_cents` "
                "(non-negative `int`) proportionally to `LineItem.weight > 0` defined in `billing/types.py`. "
                "Currently it floors each share (`total_cents * w // total_w`) and drops the remainder! "
                "Contract: compute floor shares AND distribute the remaining `rem = total_cents - sum(floors)` "
                "`+1` cent each to the top `rem` items ranked by largest fractional remainder `(total_cents * w) % total_w` "
                "(tie-breaking by smaller original index `i`), so `sum(result.values()) == total_cents` ALWAYS holds."
            ),
            {
                "billing/__init__.py": "",
                "billing/types.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class LineItem:\n"
                    "    sku: str\n"
                    "    weight: int\n"
                ),
                "billing/prorate.py": (
                    "from billing.types import LineItem\n\n"
                    "def allocate_prorated_cents(total_cents: int, items: list[LineItem]) -> dict[str, int]:\n"
                    "    if total_cents < 0:\n"
                    "        raise ValueError('total_cents must be >= 0')\n"
                    "    valid = [it for it in items if it.weight > 0]\n"
                    "    if not valid:\n"
                    "        return {}\n"
                    "    tot_w = sum(it.weight for it in valid)\n"
                    "    # BUG: drops remainder cents instead of largest-remainder distribution!\n"
                    "    return {it.sku: (total_cents * it.weight) // tot_w for it in valid}\n"
                ),
            },
            {
                "tests/test_billing.py": (
                    "import pytest\n"
                    "from billing.types import LineItem\n"
                    "from billing.prorate import allocate_prorated_cents\n\n"
                    "def test_exact_sum_conservation_and_tiebreak():\n"
                    "    items = [LineItem('A', 1), LineItem('B', 1), LineItem('C', 1)]\n"
                    "    res = allocate_prorated_cents(100, items)\n"
                    "    assert sum(res.values()) == 100\n"
                    "    assert res == {'A': 34, 'B': 33, 'C': 33}\n\n"
                    "def test_largest_fractional_remainder_priority():\n"
                    "    items = [LineItem('X', 1), LineItem('Y', 2), LineItem('Z', 5)]\n"
                    "    # 10 * 1/8 = 1 r2, 10 * 2/8 = 2 r4, 10 * 5/8 = 6 r2 -> rem=1 goes to Y (r4)!\n"
                    "    res = allocate_prorated_cents(10, items)\n"
                    "    assert res == {'X': 1, 'Y': 3, 'Z': 6}\n\n"
                    "def test_zero_and_negative_guard():\n"
                    "    assert allocate_prorated_cents(0, [LineItem('A', 3)]) == {'A': 0}\n"
                    "    with pytest.raises(ValueError):\n"
                    "        allocate_prorated_cents(-5, [LineItem('A', 1)])\n"
                ),
            },
        ),
        # R02: Multi-Wave DAG Scheduler State Mutation Bug (`scheduler/dag.py` + `scheduler/executor.py`)
        (
            "R02_dag_scheduler",
            "Fix `DAGExecutor.run_waves` re-emitting completed nodes and failing on cycles",
            (
                "`DAGExecutor.run_waves()` in `scheduler/executor.py` schedules `TaskSpec` nodes from `TaskDAG` "
                "(`scheduler/dag.py`) into topological waves (each wave sorted lexicographically by `task_id`). "
                "Bugs: (1) `ready` fails to exclude already-completed tasks so wave 1 repeats infinitely, and "
                "(2) if a dependency cycle prevents completion, `run_waves()` must raise `ValueError('cycle detected')`."
            ),
            {
                "scheduler/__init__.py": "",
                "scheduler/dag.py": (
                    "from dataclasses import dataclass, field\n\n"
                    "@dataclass\n"
                    "class TaskDAG:\n"
                    "    deps: dict[str, set[str]] = field(default_factory=dict)\n\n"
                    "    def add_task(self, task_id: str, depends_on: list[str] | None = None) -> None:\n"
                    "        self.deps.setdefault(task_id, set())\n"
                    "        for d in (depends_on or []):\n"
                    "            self.deps.setdefault(d, set())\n"
                    "            self.deps[task_id].add(d)\n"
                ),
                "scheduler/executor.py": (
                    "from scheduler.dag import TaskDAG\n\n"
                    "class DAGExecutor:\n"
                    "    def __init__(self, dag: TaskDAG) -> None:\n"
                    "        self.dag = dag\n\n"
                    "    def run_waves(self) -> list[list[str]]:\n"
                    "        done: set[str] = set()\n"
                    "        waves: list[list[str]] = []\n"
                    "        for _ in range(len(self.dag.deps)):\n"
                    "            ready = sorted(t for t, reqs in self.dag.deps.items() if reqs <= done)\n"
                    "            if not ready:\n"
                    "                break\n"
                    "            waves.append(ready)\n"
                    "            done.update(ready)\n"
                    "        return waves\n"
                ),
            },
            {
                "tests/test_scheduler.py": (
                    "import pytest\n"
                    "from scheduler.dag import TaskDAG\n"
                    "from scheduler.executor import DAGExecutor\n\n"
                    "def test_multi_wave_execution_order():\n"
                    "    g = TaskDAG()\n"
                    "    g.add_task('compile', ['codegen'])\n"
                    "    g.add_task('test', ['compile'])\n"
                    "    g.add_task('lint', [])\n"
                    "    g.add_task('codegen', [])\n"
                    "    assert DAGExecutor(g).run_waves() == [['codegen', 'lint'], ['compile'], ['test']]\n\n"
                    "def test_cycle_raises_value_error():\n"
                    "    g = TaskDAG()\n"
                    "    g.add_task('a', ['b'])\n"
                    "    g.add_task('b', ['a'])\n"
                    "    with pytest.raises(ValueError):\n"
                    "        DAGExecutor(g).run_waves()\n"
                ),
            },
        ),
        # R03: JWT Leeway & `nbf` Boundary Verification (`auth/token.py` + `auth/validator.py`)
        (
            "R03_jwt_clock_skew",
            "Fix inverted clock-skew leeway sign on `exp` and missing `nbf` check in `validate_claims`",
            (
                "`validate_claims(claims, now_ts, policy)` in `auth/validator.py` validates `TokenClaims` against "
                "`ValidationPolicy` (`auth/token.py`). Rules:\n"
                "1. Token is expired iff `now_ts > claims.exp + policy.leeway_sec` (at exact equality it is STILL valid).\n"
                "2. If `claims.nbf is not None`, token is not-yet-valid iff `now_ts + policy.leeway_sec < claims.nbf`.\n"
                "3. `claims.aud` must equal `policy.required_aud`."
            ),
            {
                "auth/__init__.py": "",
                "auth/token.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class TokenClaims:\n"
                    "    sub: str\n"
                    "    aud: str\n"
                    "    exp: int\n"
                    "    nbf: int | None = None\n"
                    "    iat: int | None = None\n\n"
                    "@dataclass(frozen=True)\n"
                    "class ValidationPolicy:\n"
                    "    required_aud: str\n"
                    "    leeway_sec: int = 0\n"
                ),
                "auth/validator.py": (
                    "from auth.token import TokenClaims, ValidationPolicy\n\n"
                    "def validate_claims(claims: TokenClaims, now_ts: int, policy: ValidationPolicy) -> bool:\n"
                    "    if claims.aud != policy.required_aud:\n"
                    "        return False\n"
                    "    # BUG: leeway sign is inverted and nbf is ignored!\n"
                    "    if now_ts >= claims.exp - policy.leeway_sec:\n"
                    "        return False\n"
                    "    return True\n"
                ),
            },
            {
                "tests/test_auth.py": (
                    "from auth.token import TokenClaims, ValidationPolicy\n"
                    "from auth.validator import validate_claims\n\n"
                    "def test_exp_and_nbf_with_leeway():\n"
                    "    pol = ValidationPolicy(required_aud='api', leeway_sec=5)\n"
                    "    c = TokenClaims(sub='u1', aud='api', exp=100, nbf=50)\n"
                    "    assert validate_claims(c, 105, pol) is True\n"
                    "    assert validate_claims(c, 106, pol) is False\n"
                    "    assert validate_claims(c, 45, pol) is True\n"
                    "    assert validate_claims(c, 44, pol) is False\n"
                    "    assert validate_claims(TokenClaims('u1', 'wrong', 100), 90, pol) is False\n"
                ),
            },
        ),
        # R04: Write-Ahead Log Transaction Commit Isolation (`wal/frame.py` + `wal/recovery.py`)
        (
            "R04_wal_tx_recovery",
            "Fix WAL crash recovery leaking uncommitted and aborted transactions",
            (
                "`replay_committed_kv(frames)` in `wal/recovery.py` replays `WALFrame(tx_id, op, key, val)` entries "
                "from `wal/frame.py` (`op` in `'PUT'`, `'DEL'`, `'COMMIT'`, `'ABORT'`).\n"
                "Rules: Buffer `'PUT'`/`'DEL'` operations per `tx_id` in arrival order; apply a transaction's buffered "
                "mutations to the KV store ONLY when `'COMMIT'` arrives for `tx_id` (and clear its buffer). "
                "On `'ABORT'`, discard `tx_id`'s buffer. Any transaction without `'COMMIT'` before EOF is uncommitted and MUST NOT take effect!"
            ),
            {
                "wal/__init__.py": "",
                "wal/frame.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class WALFrame:\n"
                    "    tx_id: int\n"
                    "    op: str  # 'PUT' | 'DEL' | 'COMMIT' | 'ABORT'\n"
                    "    key: str = ''\n"
                    "    val: str = ''\n"
                ),
                "wal/recovery.py": (
                    "from wal.frame import WALFrame\n\n"
                    "def replay_committed_kv(frames: list[WALFrame]) -> dict[str, str]:\n"
                    "    store: dict[str, str] = {}\n"
                    "    # BUG: applies PUT/DEL eagerly before COMMIT!\n"
                    "    for fr in frames:\n"
                    "        if fr.op == 'PUT':\n"
                    "            store[fr.key] = fr.val\n"
                    "        elif fr.op == 'DEL':\n"
                    "            store.pop(fr.key, None)\n"
                    "    return store\n"
                ),
            },
            {
                "tests/test_wal.py": (
                    "from wal.frame import WALFrame\n"
                    "from wal.recovery import replay_committed_kv\n\n"
                    "def test_interleaved_commit_abort_and_uncommitted_tail():\n"
                    "    frames = [\n"
                    "        WALFrame(1, 'PUT', 'a', '1'),\n"
                    "        WALFrame(2, 'PUT', 'a', '999'),\n"
                    "        WALFrame(1, 'PUT', 'b', '2'),\n"
                    "        WALFrame(1, 'COMMIT'),\n"
                    "        WALFrame(2, 'ABORT'),\n"
                    "        WALFrame(3, 'DEL', 'a'),  # crashes before COMMIT!\n"
                    "        WALFrame(4, 'DEL', 'b'),\n"
                    "        WALFrame(4, 'COMMIT'),\n"
                    "    ]\n"
                    "    assert replay_committed_kv(frames) == {'a': '1'}\n"
                ),
            },
        ),
        # R05: HTTP Idempotency-Aware Retry Policy (`http_client/policy.py` + `http_client/transport.py`)
        (
            "R05_http_idempotent_retry",
            "Fix `should_retry` double-sending non-idempotent POST without case-insensitive `Idempotency-Key`",
            (
                "`should_retry(req, status_code, attempt, policy)` in `http_client/transport.py` decides whether to retry "
                "`Request(method, headers)` under `RetryPolicy(max_retries, retry_statuses)` (`http_client/policy.py`).\n"
                "Rules:\n"
                "1. Return `False` if `attempt >= policy.max_retries` or `status_code not in policy.retry_statuses`.\n"
                "2. Methods `'GET'`, `'HEAD'`, `'PUT'`, `'DELETE'`, `'OPTIONS'` (case-insensitive) are inherently idempotent -> `True`.\n"
                "3. `'POST'` and `'PATCH'` are retryable ONLY if `req.headers` contains a non-empty `'Idempotency-Key'` header "
                "matched CASE-INSENSITIVELY (e.g. `'idempotency-key': 'abc'`)."
            ),
            {
                "http_client/__init__.py": "",
                "http_client/policy.py": (
                    "from dataclasses import dataclass, field\n\n"
                    "@dataclass\n"
                    "class Request:\n"
                    "    method: str\n"
                    "    headers: dict[str, str] = field(default_factory=dict)\n\n"
                    "@dataclass(frozen=True)\n"
                    "class RetryPolicy:\n"
                    "    max_retries: int = 3\n"
                    "    retry_statuses: tuple[int, ...] = (429, 502, 503, 504)\n"
                ),
                "http_client/transport.py": (
                    "from http_client.policy import Request, RetryPolicy\n\n"
                    "def should_retry(req: Request, status_code: int, attempt: int, policy: RetryPolicy) -> bool:\n"
                    "    if attempt > policy.max_retries or status_code not in policy.retry_statuses:\n"
                    "        return False\n"
                    "    # BUG: retries POST unconditionally and off-by-one on attempt >= max_retries!\n"
                    "    return True\n"
                ),
            },
            {
                "tests/test_http_retry.py": (
                    "from http_client.policy import Request, RetryPolicy\n"
                    "from http_client.transport import should_retry\n\n"
                    "def test_idempotency_and_attempt_cap():\n"
                    "    pol = RetryPolicy(max_retries=2)\n"
                    "    assert should_retry(Request('GET'), 503, 1, pol) is True\n"
                    "    assert should_retry(Request('GET'), 503, 2, pol) is False\n"
                    "    assert should_retry(Request('POST'), 503, 0, pol) is False\n"
                    "    assert should_retry(Request('POST', {'idempotency-key': 'k1'}), 503, 0, pol) is True\n"
                    "    assert should_retry(Request('POST', {'Idempotency-Key': '   '}), 503, 0, pol) is False\n"
                ),
            },
        ),
        # R06: SemVer 2.0 Prerelease Ordering Bug (`semver/version.py` + `semver/selector.py`)
        (
            "R06_semver_prerelease",
            "Fix SemVer 2.0 comparison sorting `1.0.0-alpha` HIGHER than `1.0.0` release",
            (
                "`SemVer` in `semver/version.py` and `highest_compatible(versions, major)` in `semver/selector.py` "
                "implement SemVer 2.0 precedence: `(major, minor, patch)` compared numerically first; a normal release "
                "(`prerelease == ''`) has HIGHER precedence than any prerelease (`'1.0.0-alpha' < '1.0.0-beta' < '1.0.0'`)! "
                "`highest_compatible(versions, major)` must return the highest stable (`prerelease == ''`) version string "
                "with `v.major == major`, or `None` if none exists."
            ),
            {
                "semver/__init__.py": "",
                "semver/version.py": (
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class SemVer:\n"
                    "    major: int\n"
                    "    minor: int\n"
                    "    patch: int\n"
                    "    prerelease: str = ''\n\n"
                    "    @classmethod\n"
                    "    def parse(cls, raw: str) -> 'SemVer':\n"
                    "        core, _, pre = raw.strip().partition('-')\n"
                    "        mj, mn, pt = (int(x) for x in core.split('.'))\n"
                    "        return cls(mj, mn, pt, pre)\n\n"
                    "    def sort_key(self) -> tuple:\n"
                    "        # BUG: '' < 'alpha' makes 1.0.0 sort LOWER than 1.0.0-alpha!\n"
                    "        return (self.major, self.minor, self.patch, self.prerelease)\n"
                ),
                "semver/selector.py": (
                    "from semver.version import SemVer\n\n"
                    "def highest_compatible(versions: list[str], major: int) -> str | None:\n"
                    "    parsed = [ (SemVer.parse(s), s) for s in versions ]\n"
                    "    cands = [ (v, s) for v, s in parsed if v.major == major ]\n"
                    "    if not cands:\n"
                    "        return None\n"
                    "    return max(cands, key=lambda pair: pair[0].sort_key())[1]\n"
                ),
            },
            {
                "tests/test_semver.py": (
                    "from semver.version import SemVer\n"
                    "from semver.selector import highest_compatible\n\n"
                    "def test_prerelease_precedence_and_stable_filter():\n"
                    "    assert SemVer.parse('1.0.0-alpha').sort_key() < SemVer.parse('1.0.0-beta').sort_key()\n"
                    "    assert SemVer.parse('1.0.0-rc.1').sort_key() < SemVer.parse('1.0.0').sort_key()\n"
                    "    assert highest_compatible(['1.2.0', '1.3.0-rc.1', '2.0.0'], 1) == '1.2.0'\n"
                    "    assert highest_compatible(['1.0.0-alpha'], 1) is None\n"
                ),
            },
        ),
        # R07: LRU + TTL Cache Ghost-Order Eviction Corruption (`ttlcache/entry.py` + `ttlcache/cache.py`)
        (
            "R07_lru_ttl_eviction",
            "Fix `TTLCache` leaving expired ghost keys in LRU deque and evicting live entries",
            (
                "`TTLCache` in `ttlcache/cache.py` stores `CacheEntry(val, expires_at)` (`ttlcache/entry.py`) up to `max_size`.\n"
                "Bugs: (1) When `get(key, now)` finds an expired entry (`now >= entry.expires_at`), it deletes from `self._data` "
                "but leaves `key` in `self._lru`! (2) Before `put(key, val, ttl, now)` evicts the oldest LRU key on capacity overflow, "
                "it must purge all expired entries (`now >= expires_at`) first so live entries are never evicted unnecessarily!"
            ),
            {
                "ttlcache/__init__.py": "",
                "ttlcache/entry.py": (
                    "from dataclasses import dataclass\n"
                    "from typing import Any\n\n"
                    "@dataclass\n"
                    "class CacheEntry:\n"
                    "    value: Any\n"
                    "    expires_at: float\n"
                ),
                "ttlcache/cache.py": (
                    "from collections import OrderedDict\n"
                    "from typing import Any\n"
                    "from ttlcache.entry import CacheEntry\n\n"
                    "class TTLCache:\n"
                    "    def __init__(self, max_size: int) -> None:\n"
                    "        self.max_size = max_size\n"
                    "        self._data: dict[str, CacheEntry] = {}\n"
                    "        self._lru: OrderedDict[str, None] = OrderedDict()\n\n"
                    "    def get(self, key: str, now: float) -> Any | None:\n"
                    "        ent = self._data.get(key)\n"
                    "        if ent is None:\n"
                    "            return None\n"
                    "        if now >= ent.expires_at:\n"
                    "            del self._data[key]\n"
                    "            return None\n"
                    "        self._lru.move_to_end(key)\n"
                    "        return ent.value\n\n"
                    "    def put(self, key: str, value: Any, ttl: float, now: float) -> None:\n"
                    "        if key not in self._data and len(self._lru) >= self.max_size:\n"
                    "            old_k, _ = self._lru.popitem(last=False)\n"
                    "            self._data.pop(old_k, None)\n"
                    "        self._data[key] = CacheEntry(value, now + ttl)\n"
                    "        self._lru[key] = None\n"
                    "        self._lru.move_to_end(key)\n"
                ),
            },
            {
                "tests/test_ttlcache.py": (
                    "from ttlcache.cache import TTLCache\n\n"
                    "def test_expired_purge_prevents_live_eviction():\n"
                    "    c = TTLCache(max_size=2)\n"
                    "    c.put('long', 2, ttl=100.0, now=0.0)\n"
                    "    c.put('short', 1, ttl=5.0, now=0.0)\n"
                    "    assert c.get('short', now=10.0) is None  # expired, must also leave _lru!\n"
                    "    c.put('new', 3, ttl=50.0, now=10.0)\n"
                    "    assert c.get('long', now=10.0) == 2\n"
                    "    assert c.get('new', now=10.0) == 3\n\n"
                    "def test_put_purges_unaccessed_expired_before_lru_evict():\n"
                    "    c = TTLCache(max_size=2)\n"
                    "    c.put('b', 20, ttl=50.0, now=0.0)\n"
                    "    c.put('a', 10, ttl=2.0, now=0.0)\n"
                    "    c.put('c', 30, ttl=50.0, now=5.0)  # 'a' is expired at t=5 -> must NOT evict 'b'!\n"
                    "    assert c.get('b', now=5.0) == 20\n"
                    "    assert c.get('c', now=5.0) == 30\n"
                ),
            },
        ),
        # R08: Mutable Default Schema Migration Cross-Row Aliasing (`etl/schema.py` + `etl/migrator.py`)
        (
            "R08_schema_mutable_defaults",
            "Fix `migrate_records` aliasing mutable list/dict `FIELD_DEFAULTS` across output rows",
            (
                "`migrate_records(rows)` in `etl/migrator.py` normalizes dicts using `SCHEMA_V2_DEFAULTS` from `etl/schema.py`.\n"
                "Bugs: (1) Assigning `out[k] = default_val` directly aliases mutable `list`/`dict` defaults across every row "
                "so mutating `res[0]['tags'].append('vip')` corrupts `res[1]['tags']`! Deep-copy defaults AND input mutable fields. "
                "(2) Strip whitespace on `'email'` and lowercase it."
            ),
            {
                "etl/__init__.py": "",
                "etl/schema.py": (
                    "SCHEMA_V2_DEFAULTS = {\n"
                    "    'email': '',\n"
                    "    'tags': [],\n"
                    "    'meta': {'version': 2},\n"
                    "}\n"
                ),
                "etl/migrator.py": (
                    "from etl.schema import SCHEMA_V2_DEFAULTS\n\n"
                    "def migrate_records(rows: list[dict]) -> list[dict]:\n"
                    "    out_rows = []\n"
                    "    for r in rows:\n"
                    "        rec = {}\n"
                    "        for k, dval in SCHEMA_V2_DEFAULTS.items():\n"
                    "            rec[k] = r.get(k, dval)\n"
                    "        out_rows.append(rec)\n"
                    "    return out_rows\n"
                ),
            },
            {
                "tests/test_etl.py": (
                    "from etl.migrator import migrate_records\n\n"
                    "def test_no_cross_row_aliasing_and_email_norm():\n"
                    "    res = migrate_records([{'email': '  Alice@Example.COM '}, {'email': 'bob@x.io'}])\n"
                    "    assert res[0]['email'] == 'alice@example.com'\n"
                    "    res[0]['tags'].append('admin')\n"
                    "    res[0]['meta']['region'] = 'us'\n"
                    "    assert res[1]['tags'] == []\n"
                    "    assert res[1]['meta'] == {'version': 2}\n"
                ),
            },
        ),
    ]

    tasks: list[MultiFileRepoBugTask] = []
    for idx, (fam_id, title, desc, srcs, tsts) in enumerate(specs, start=1):
        tasks.append(
            MultiFileRepoBugTask(
                task_id=f"SWE_W1_{idx:02d}_{fam_id}",
                family_id=fam_id,
                wave=1,
                issue_title=title,
                issue_description=desc,
                source_files=srcs,
                test_files=tsts,
                spec_mutated=False,
            )
        )

    # Wave 2: Recurring repository regression stream across sister services + 1 adversarial spec mutation on R03!
    for idx, (fam_id, title, desc, srcs, tsts) in enumerate(specs, start=1):
        if fam_id == "R03_jwt_clock_skew":
            mut_desc = (
                desc
                + "\n4. NEW SECURITY REQUIREMENT: If `claims.iat is not None` and `claims.iat > now_ts + policy.leeway_sec` "
                "(future-issued token), `validate_claims` MUST return `False`!"
            )
            mut_tsts = {
                "tests/test_auth.py": (
                    tsts["tests/test_auth.py"]
                    + "\ndef test_future_iat_rejected():\n"
                    "    pol = ValidationPolicy(required_aud='api', leeway_sec=5)\n"
                    "    assert validate_claims(TokenClaims('u1', 'api', 200, iat=106), 100, pol) is False\n"
                    "    assert validate_claims(TokenClaims('u1', 'api', 200, iat=105), 100, pol) is True\n"
                )
            }
            tasks.append(
                MultiFileRepoBugTask(
                    task_id=f"SWE_W2_{idx:02d}_{fam_id}_MUTATED",
                    family_id=fam_id,
                    wave=2,
                    issue_title=title + " + future-`iat` guard",
                    issue_description=mut_desc,
                    source_files=srcs,
                    test_files=mut_tsts,
                    spec_mutated=True,
                )
            )
        else:
            tasks.append(
                MultiFileRepoBugTask(
                    task_id=f"SWE_W2_{idx:02d}_{fam_id}",
                    family_id=fam_id,
                    wave=2,
                    issue_title=f"[Regression Port] {title}",
                    issue_description=desc,
                    source_files=srcs,
                    test_files=tsts,
                    spec_mutated=False,
                )
            )

    return tasks


def _generate_repo_patch_local(runner: Local2026ModelRunner, prompt: str, max_new_tokens: int = 768) -> tuple[str, float]:
    msgs = [
        {"role": "system", "content": REPO_PATCH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        chat_txt = runner.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        chat_txt = runner.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = runner.tokenizer(chat_txt, return_tensors="pt").to(runner.device)
    in_len = inputs["input_ids"].size(1)
    eos_ids = runner._collect_eos_ids()
    gen_kw = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": runner.tokenizer.pad_token_id or (eos_ids[0] if eos_ids else 0),
    }
    if eos_ids:
        gen_kw["eos_token_id"] = eos_ids
    with torch.no_grad():
        out = runner.model.generate(**inputs, **gen_kw)
    gen_ids = out.sequences[0, in_len:]
    probs = [torch.softmax(sc[0].float(), dim=-1).max().item() for sc in out.scores[: min(24, len(out.scores))]]
    conf = float(sum(probs) / len(probs)) if probs else 0.5
    txt = runner.tokenizer.decode(gen_ids, skip_special_tokens=False)
    if runner.device.type == "mps":
        torch.mps.synchronize()
    return txt, conf


def _generate_repo_patch_35b(runner: ExpertAirMoL35BRunner, prompt: str, max_new_tokens: int = 768) -> str:
    msgs = [
        {"role": "system", "content": REPO_PATCH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    chat_txt = runner.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    p_ids = runner.tokenizer.encode(chat_txt, add_special_tokens=False)
    cache = runner._DynamicCache(config=runner.tcfg)
    with torch.no_grad():
        lg, _ = runner._forward_step(torch.tensor([p_ids], device=runner.device, dtype=torch.long), cache)
        nxt = int(lg[0].argmax().item())
        emitted = [nxt]
        while len(emitted) < max_new_tokens and nxt not in runner.stop_ids:
            lg, _ = runner._forward_step(torch.tensor([[nxt]], device=runner.device, dtype=torch.long), cache)
            nxt = int(lg[0].argmax().item())
            emitted.append(nxt)
    if runner.device.type == "mps":
        torch.mps.synchronize()
    return runner.tokenizer.decode(emitted, skip_special_tokens=False)


def run_repo_patch_benchmark(device: torch.device) -> tuple[list[RepoPatchSummaryRow], dict]:
    """
    Executes the `16`-task (`8` multi-file git repos x `2` waves) `SWE-Mini` Patching Benchmark across:
      1. `A_LiquidAI_LFM3_1.7B_RepoPatch_1Shot` (Ungated)
      2. `B_Google_Gemma4_E2B_it_RepoPatch_1Shot` (Ungated)
      3. `C_LiquidAI_LFM2.5_8B_A1B_MoE_RepoPatch_1Shot` (Ungated)
      4. `D_Qwen3.6_35B_A3B_MetalAirMoL_RepoPatch_1Shot` (Ungated)
      5. `E_OpenIE_MoL_RepoPatchCascade_OURS` (`Rung 0 BenesLUT6` + `.mol/` Patch Skill Store + `pytest` `C(z)`
         + Counterexample Traceback Repair up to `35B-A3B`)
    """
    lib = Path(os.environ.get("CLAI_MODEL_LIBRARY", "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library"))
    lfm3_dir = lib / "LFM3-1.7B"
    gemma4_dir = lib / "gemma-4-E2B-it"
    lfm8b_snap = next((lib / "_hf_cache/models--LiquidAI--LFM2.5-8B-A1B/snapshots").iterdir())

    tasks = build_swebench_mini_repo_tasks()

    # Collect initial failing pytest tracebacks on every unique task family/spec
    initial_tbs: dict[str, str] = {}
    unique_tasks: list[MultiFileRepoBugTask] = []
    seen_keys: set[str] = set()
    for t in tasks:
        ukey = f"{t.family_id}|mut={t.spec_mutated}"
        if ukey not in seen_keys:
            seen_keys.add(ukey)
            unique_tasks.append(t)
            with tempfile.TemporaryDirectory(prefix="mol_init_tb_") as tmp:
                rdir = Path(tmp)
                materialize_git_repo(rdir, t.source_files, t.test_files)
                ok0, tb0 = run_repo_pytest(rdir)
                assert not ok0, f"Buggy repo `{t.task_id}` unexpectedly passed pytest before patching!"
                initial_tbs[ukey] = tb0

    # Cache 1-shot & repair candidates per model
    cache_1s: dict[str, dict[str, tuple[str, float, float, float]]] = {"lfm3": {}, "gemma4": {}, "lfm8b": {}, "qwen35b": {}}
    rep_g4: dict[str, tuple[str, float, float, float]] = {}
    rep_8b: dict[str, tuple[str, float, float, float]] = {}
    rep_35b: dict[str, tuple[str, float, float, float]] = {}

    for m_key, m_path, watts in [
        ("lfm3", lfm3_dir, 32.0),
        ("gemma4", gemma4_dir, 42.0),
        ("lfm8b", lfm8b_snap, 55.0),
    ]:
        print(f"  [SWE-Mini] Running `{m_key}` across {len(unique_tasks)} multi-file git repositories...", flush=True)
        runner = Local2026ModelRunner(m_key, m_path, device)
        for ut in unique_tasks:
            ukey = f"{ut.family_id}|mut={ut.spec_mutated}"
            prompt = format_repo_bug_prompt(ut, initial_tbs[ukey])
            with meter_joules("rung2" if watts < 50 else "rung3", device=device) as jr:
                raw_out, _ = _generate_repo_patch_local(runner, prompt, max_new_tokens=768)
            dt_ms = jr.elapsed_ms
            cache_1s[m_key][ukey] = (raw_out, dt_ms, (dt_ms / 1000.0) * watts, float(jr.reported_j or 0.0))

            if m_key == "gemma4":
                ok_l3, _, _, _ = verify_repo_patch_candidate(ut, cache_1s["lfm3"][ukey][0])
                ok_g4, _, _, diag_g4 = verify_repo_patch_candidate(ut, raw_out)
                if not (ok_l3 or ok_g4):
                    rp = f"{prompt}\n\nYOUR PREVIOUS PATCH FAILED `pytest`:\n```text\n{diag_g4}\n```\nOutput ONLY the fixed file block(s)."
                    with meter_joules("rung2", device=device) as jrr:
                        rfix, _ = _generate_repo_patch_local(runner, rp, max_new_tokens=768)
                    rep_g4[ukey] = (rfix, jrr.elapsed_ms, (jrr.elapsed_ms / 1000.0) * watts, float(jrr.reported_j or 0.0))
            elif m_key == "lfm8b":
                prev_ok = (
                    verify_repo_patch_candidate(ut, cache_1s["lfm3"][ukey][0])[0]
                    or verify_repo_patch_candidate(ut, cache_1s["gemma4"][ukey][0])[0]
                    or (ukey in rep_g4 and verify_repo_patch_candidate(ut, rep_g4[ukey][0])[0])
                )
                ok_8, _, _, diag_8 = verify_repo_patch_candidate(ut, raw_out)
                if not (prev_ok or ok_8):
                    rp = f"{prompt}\n\nYOUR PREVIOUS PATCH FAILED `pytest`:\n```text\n{diag_8}\n```\nOutput ONLY the fixed file block(s)."
                    with meter_joules("rung3", device=device) as jrr:
                        rfix, _ = _generate_repo_patch_local(runner, rp, max_new_tokens=768)
                    rep_8b[ukey] = (rfix, jrr.elapsed_ms, (jrr.elapsed_ms / 1000.0) * watts, float(jrr.reported_j or 0.0))
        runner.unload()

    print(f"  [SWE-Mini] Running Metal `Qwen3.6-35B-A3B` across {len(unique_tasks)} multi-file git repositories...", flush=True)
    r35 = ExpertAirMoL35BRunner(device=device)
    for ut in unique_tasks:
        ukey = f"{ut.family_id}|mut={ut.spec_mutated}"
        prompt = format_repo_bug_prompt(ut, initial_tbs[ukey])
        with meter_joules("rung3", device=device) as jr35:
            raw35 = _generate_repo_patch_35b(r35, prompt, max_new_tokens=768)
        dt35 = jr35.elapsed_ms
        cache_1s["qwen35b"][ukey] = (raw35, dt35, (dt35 / 1000.0) * 95.0, float(jr35.reported_j or 0.0))

        prev_ok = (
            verify_repo_patch_candidate(ut, cache_1s["lfm3"][ukey][0])[0]
            or verify_repo_patch_candidate(ut, cache_1s["gemma4"][ukey][0])[0]
            or (ukey in rep_g4 and verify_repo_patch_candidate(ut, rep_g4[ukey][0])[0])
            or verify_repo_patch_candidate(ut, cache_1s["lfm8b"][ukey][0])[0]
            or (ukey in rep_8b and verify_repo_patch_candidate(ut, rep_8b[ukey][0])[0])
        )
        if not prev_ok:
            ok35, _, _, diag35 = verify_repo_patch_candidate(ut, raw35)
            cur35, tot_dt, tot_j, tot_rep = raw35, dt35, (dt35 / 1000.0) * 95.0, float(jr35.reported_j or 0.0)
            if not ok35:
                rp35 = f"{prompt}\n\nYOUR PREVIOUS PATCH FAILED `pytest`:\n```text\n{diag35}\n```\nOutput ONLY the fixed file block(s)."
                with meter_joules("rung3", device=device) as jr35r:
                    cur35 = _generate_repo_patch_35b(r35, rp35, max_new_tokens=768)
                tot_dt += jr35r.elapsed_ms
                tot_j += (jr35r.elapsed_ms / 1000.0) * 95.0
                tot_rep += float(jr35r.reported_j or 0.0)
            rep_35b[ukey] = (cur35, tot_dt, tot_j, tot_rep)
    r35.unload()

    def _summarize(arm: str, recs: list[tuple[int, bool, bool, bool, float, float, float]]) -> RepoPatchSummaryRow:
        w1 = [x for x in recs if x[0] == 1]
        w2 = [x for x in recs if x[0] == 2]
        ok_n = sum(1 for x in recs if x[1])
        tot_j = sum(x[5] for x in recs)
        tot_rep = sum(x[6] for x in recs)
        return RepoPatchSummaryRow(
            arm=arm,
            wave1_cold_pass_pct=round(100.0 * sum(1 for x in w1 if x[1]) / len(w1), 2),
            wave2_warm_pass_pct=round(100.0 * sum(1 for x in w2 if x[1]) / len(w2), 2),
            overall_verified_pass_pct=round(100.0 * ok_n / len(recs), 2),
            broken_patches_committed_pct=round(100.0 * sum(1 for x in recs if x[2]) / len(recs), 2),
            rung0_rung1_hit_pct_wave2=round(100.0 * sum(1 for x in w2 if x[3]) / len(w2), 2),
            mean_latency_ms_wave1=round(sum(x[4] for x in w1) / len(w1), 2),
            mean_latency_ms_wave2=round(sum(x[4] for x in w2) / len(w2), 2),
            est_joules_per_verified_fix=round(tot_j / max(1, ok_n), 2),
            reported_joules_per_verified_fix=round(tot_rep / max(1, ok_n), 3) if tot_rep > 0 else None,
        )

    rows: list[RepoPatchSummaryRow] = []
    for m_key, arm_title in [
        ("lfm3", "A_LiquidAI_LFM3_1.7B_RepoPatch_1Shot"),
        ("gemma4", "B_Google_Gemma4_E2B_it_RepoPatch_1Shot"),
        ("lfm8b", "C_LiquidAI_LFM2.5_8B_A1B_MoE_RepoPatch_1Shot"),
        ("qwen35b", "D_Qwen3.6_35B_A3B_MetalAirMoL_RepoPatch_1Shot"),
    ]:
        recs = []
        for t in tasks:
            ukey = f"{t.family_id}|mut={t.spec_mutated}"
            raw_c, dt_ms, ej, rj = cache_1s[m_key][ukey]
            ok, _, _, _ = verify_repo_patch_candidate(t, raw_c)
            recs.append((t.wave, ok, not ok, False, dt_ms, ej, rj))
        rows.append(_summarize(arm_title, recs))

    # Arm E: OpenIE MoL RepoPatchCascade (`Rung 0 BenesLUT6` + `.mol/` Patch Store + `pytest` `C(z)`)
    mol_store = BenesLUT6SkillCascadeStore()
    recs_mol = []
    traces = []
    drift_caught = 0

    for t in tasks:
        ukey = f"{t.family_id}|mut={t.spec_mutated}"
        sig_prompt = f"{t.family_id}\n{t.issue_description}"
        with meter_joules("rung0") as jr0:
            hit = mol_store.lookup_rung0_rung1(t.family_id, sig_prompt)
        dt0 = jr0.elapsed_ms
        rep0 = float(jr0.reported_j or 0.0)

        if hit is not None:
            with meter_joules("rung0") as jrv:
                ok_h, _, diff_h, _ = verify_repo_patch_candidate(t, hit.normalized_code)
            if ok_h:
                hit.hits += 1
                tot_ms = dt0 + jrv.elapsed_ms
                tot_rep = rep0 + float(jrv.reported_j or 0.0)
                recs_mol.append((t.wave, True, False, True, tot_ms, (tot_ms / 1000.0) * 3.0, tot_rep))
                traces.append(
                    {
                        "task_id": t.task_id,
                        "wave": t.wave,
                        "resolved_by": f"Rung0_BenesLUT6->.mol/PatchSkill#{hit.skill_id}",
                        "latency_ms": round(tot_ms, 2),
                        "reported_j": round(tot_rep, 4),
                        "diff_lines": len(diff_h.splitlines()),
                    }
                )
                continue
            if t.spec_mutated:
                drift_caught += 1

        acc_ms = dt0
        acc_j = (dt0 / 1000.0) * 3.0
        acc_rep = rep0
        stages = [
            ("Rung2_LFM3_1.7B", *cache_1s["lfm3"][ukey]),
            ("Rung3a_Gemma4_E2B", *cache_1s["gemma4"][ukey]),
        ]
        if ukey in rep_g4:
            stages.append(("Rung3a_Gemma4_E2B_PytestRepair", *rep_g4[ukey]))
        stages.append(("Rung3b_LFM2.5_8B_A1B", *cache_1s["lfm8b"][ukey]))
        if ukey in rep_8b:
            stages.append(("Rung3b_LFM2.5_8B_A1B_PytestRepair", *rep_8b[ukey]))
        if ukey in rep_35b:
            stages.append(("Rung3c_Qwen3.6_35B_A3B_MetalAirMoL", *rep_35b[ukey]))

        solved = False
        win_json = ""
        win_src = "ABSTAIN"
        win_diff = ""
        for st_name, cand_raw, st_ms, st_j, st_rep in stages:
            acc_ms += st_ms
            acc_j += st_j
            acc_rep += st_rep
            with meter_joules("rung0") as jrv:
                ok_c, can_json, gdiff, _ = verify_repo_patch_candidate(t, cand_raw)
            acc_ms += jrv.elapsed_ms
            acc_rep += float(jrv.reported_j or 0.0)
            if ok_c:
                solved = True
                win_json = can_json
                win_src = st_name
                win_diff = gdiff
                break

        if solved:
            sid = mol_store.register_verified_skill(t.family_id, sig_prompt, win_json, win_src)
            mol_store.recompile_lut6_router()
            recs_mol.append((t.wave, True, False, False, acc_ms, acc_j, acc_rep))
            traces.append(
                {
                    "task_id": t.task_id,
                    "wave": t.wave,
                    "resolved_by": f"{win_src}->Crystallized_Patch#{sid}",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "diff_lines": len(win_diff.splitlines()),
                }
            )
        else:
            recs_mol.append((t.wave, False, False, False, acc_ms, acc_j, acc_rep))
            traces.append(
                {
                    "task_id": t.task_id,
                    "wave": t.wave,
                    "resolved_by": "VERIFIER_ABSTAIN_SAFE",
                    "latency_ms": round(acc_ms, 2),
                    "reported_j": round(acc_rep, 4),
                    "diff_lines": 0,
                }
            )

    mol_dir = Path("artifacts/.mol_repo_patches")
    mol_store.save_to_dir(mol_dir)
    rows.append(_summarize("E_OpenIE_MoL_RepoPatchCascade_OURS", recs_mol))

    return rows, {
        "meta": {
            "num_repos_per_wave": 8,
            "num_waves": 2,
            "total_repo_evaluations": len(tasks),
            "wave2_spec_mutations_caught_by_pytest_cz": drift_caught,
            "crystallized_repo_patch_skills": len(mol_store.skills),
            "persisted_mol_dir": str(mol_dir),
            "reported_j_source": "apple_silicon_l3_rusage_v6_plus_l2_ioreport_energy_model",
            "measured_j": None,
            "board_metered": False,
        },
        "traces": traces,
    }
