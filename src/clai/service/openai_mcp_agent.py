"""Phase 26 (Track 4): Drop-In OpenAI-Compatible (`/v1/models`, `/v1/chat/completions` REST + SSE)
and Model Context Protocol (`MCP` JSON-RPC `2.0` over `stdio` & `POST /mcp`) Local Coding Agent
with Persistent `.mol/` `BenesLUT6` + Canonical AST Skill Crystallization and `reported_j` Receipts.

Works out-of-the-box with:
  - OpenAI Python/TypeScript SDKs, Cursor, Continue.dev, Aider, OpenWebUI (`http://127.0.0.1:8090/v1`)
  - Claude Code / Antigravity / Zed / VS Code MCP clients (`mol-agent --mcp-stdio` or `POST /mcp`)
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import uuid

from clai.contracts.joule_meter import (
    get_apple_reported_j_reader,
    meter_joules,
    select_torch_device,
)
from clai.frontier.repo_patch_cascade import (
    MultiFileRepoBugTask,
    format_repo_bug_prompt,
    run_repo_pytest,
    verify_repo_patch_candidate,
)
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    ExpertAirMoL35BRunner,
    Local2026ModelRunner,
    FORBIDDEN_IMPORTS,
    _build_repair_prompt,
    _execution_timeout,
    extract_python_code,
)


def verify_assertion_contract(raw_output: str, fn_name: str, assertions: list[str]) -> tuple[bool, str, str]:
    """
    Deterministic Contract Verifier `C(z) in {0, 1}` for OpenAI `x_mol_contract` & MCP `assertions`:
      1. Parses AST & blocks `FORBIDDEN_IMPORTS`.
      2. Executes candidate function + all caller `assert ...` statements under a `0.35s` POSIX timeout.
    """
    code = extract_python_code(raw_output)
    if not code:
        return False, "", "Empty Python candidate."
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, code, f"SyntaxError at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return False, code, f"Forbidden import `{alias.name}` rejected by sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return False, code, f"Forbidden import `{node.module}` rejected by sandbox."

    ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "heapq": __import__("heapq"),
        "hashlib": __import__("hashlib"),
        "re": __import__("re"),
        "collections": __import__("collections"),
        "bisect": __import__("bisect"),
        "functools": __import__("functools"),
        "itertools": __import__("itertools"),
    }
    try:
        with _execution_timeout(0.35):
            exec(compile(tree, "<mol_agent_candidate>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, code, f"Candidate execution raised {type(exc).__name__}: {exc}"

    if fn_name and not callable(ns.get(fn_name)):
        return False, code, f"Expected callable `{fn_name}` was not defined."

    for idx, stmt in enumerate(assertions, start=1):
        try:
            a_tree = ast.parse(stmt)
            for node in ast.walk(a_tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False, code, "Imports are not allowed inside assertion strings."
            with _execution_timeout(0.35):
                exec(compile(a_tree, f"<assertion_{idx}>", "exec"), ns)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return False, code, f"Assertion #{idx} (`{stmt}`) failed with {type(exc).__name__}: {exc}"

    return True, ast.unparse(tree), ""


@dataclass
class MoLAgentResult:
    content: str
    verified: bool
    resolved_by: str
    skill_id: int | None
    latency_ms: float
    est_j: float
    reported_j: float | None
    reported_channels: dict[str, Any]

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "resolved_by": self.resolved_by,
            "verified": self.verified,
            "skill_id": self.skill_id,
            "latency_ms": round(self.latency_ms, 3),
            "est_j": round(self.est_j, 4),
            "reported_j": round(self.reported_j, 5) if self.reported_j is not None else None,
            "measured_j": None,
            "board_metered": False,
            "reported_channels": self.reported_channels,
        }


class MoLLocalCodingAgent:
    """
    Persistent `.mol/`-backed Verifier-Gated Cascade Engine supporting:
      - Rung 0 (`80` `LUT6`s + `512x512` Beneš Router) -> Rung 1 (`.mol/` Canonical AST Skill Store)
      - Lazy-loaded local 2026 Rung 2 (`LiquidAI/LFM3-1.7B`) -> Rung 3a (`google/gemma-4-E2B-it`)
        -> Rung 3c (`Qwen/Qwen3.6-35B-A3B` Metal `INT4` `Expert-AirMoL`), OR injectable `generator_fn`.
    """

    def __init__(
        self,
        mol_dir: Path | None = None,
        generator_fn: Callable[[str, str, int], str] | None = None,
        enable_35b_escalation: bool = False,
    ) -> None:
        self.mol_dir = mol_dir or Path(".mol")
        self.store = BenesLUT6SkillCascadeStore.load_from_dir(self.mol_dir)
        self.generator_fn = generator_fn
        self.enable_35b_escalation = enable_35b_escalation
        self._runners: dict[str, Local2026ModelRunner] = {}
        self._runner_35b: ExpertAirMoL35BRunner | None = None

    def _invoke_rung_model(self, rung_key: str, prompt: str, max_tokens: int = 640) -> str:
        if self.generator_fn is not None:
            return self.generator_fn(rung_key, prompt, max_tokens)

        dev = select_torch_device()
        lib = Path(os.environ.get("CLAI_MODEL_LIBRARY", "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library"))
        if rung_key == "qwen35b":
            if self._runner_35b is None:
                self._runner_35b = ExpertAirMoL35BRunner(device=dev)
            out, _, _, _ = self._runner_35b.generate_code(prompt, max_new_tokens=max_tokens)
            return out

        path_map = {
            "lfm3": lib / "LFM3-1.7B",
            "gemma4": lib / "gemma-4-E2B-it",
        }
        m_path = path_map[rung_key]
        if rung_key not in self._runners:
            self._runners[rung_key] = Local2026ModelRunner(rung_key, m_path, dev)
        out, _, _ = self._runners[rung_key].generate_code(prompt, max_new_tokens=max_tokens)
        return out

    def synthesize(
        self,
        prompt: str,
        fn_name: str = "",
        assertions: list[str] | None = None,
        max_tokens: int = 640,
    ) -> MoLAgentResult:
        """Executes full Rung 0/1 -> `C(z)` -> Rung 2/3 Counterexample Escalation -> `.mol/` Crystallization."""
        has_contract = bool(fn_name or assertions)
        assert_list = list(assertions or [])
        lookup_fn = fn_name or "chat_skill"

        # 1. Rung 0 (`BenesLUT6`) -> Rung 1 (`.mol/` Verified Skill Store) Fast Path
        with meter_joules("rung0") as jr0:
            cand = self.store.lookup_rung0_rung1(lookup_fn, prompt)

        if cand is not None:
            if not has_contract:
                cand.hits += 1
                self.store.save_to_dir(self.mol_dir)
                return MoLAgentResult(
                    content=f"```python\n{cand.normalized_code}\n```",
                    verified=True,
                    resolved_by=f"Rung0_BenesLUT6->Rung1_Skill#{cand.skill_id}",
                    skill_id=cand.skill_id,
                    latency_ms=jr0.elapsed_ms,
                    est_j=jr0.est_j,
                    reported_j=jr0.reported_j,
                    reported_channels=jr0.reported_channels,
                )
            with meter_joules("rung0") as jrv:
                ok_s, _, _ = verify_assertion_contract(cand.normalized_code, fn_name, assert_list)
            if ok_s:
                cand.hits += 1
                self.store.save_to_dir(self.mol_dir)
                tot_ms = jr0.elapsed_ms + jrv.elapsed_ms
                rep_j = (jr0.reported_j or 0.0) + (jrv.reported_j or 0.0) if jr0.reported_j is not None else None
                return MoLAgentResult(
                    content=f"```python\n{cand.normalized_code}\n```",
                    verified=True,
                    resolved_by=f"Rung0_BenesLUT6->Rung1_Skill#{cand.skill_id}",
                    skill_id=cand.skill_id,
                    latency_ms=tot_ms,
                    est_j=jr0.est_j + jrv.est_j,
                    reported_j=rep_j,
                    reported_channels=jrv.reported_channels or jr0.reported_channels,
                )

        # 2. Escalate through Rung 2 (`LFM3-1.7B`) -> Rung 3a (`Gemma-4-E2B-it` + Repair) -> optional `35B`
        acc_ms = jr0.elapsed_ms
        acc_est = jr0.est_j
        acc_rep = float(jr0.reported_j or 0.0)
        has_rep = jr0.reported_j is not None
        last_chans = jr0.reported_channels

        rung_plan = [("lfm3", "Rung2_LiquidAI_LFM3_1.7B"), ("gemma4", "Rung3a_Google_Gemma4_E2B")]
        if self.enable_35b_escalation:
            rung_plan.append(("qwen35b", "Rung3c_Qwen3.6_35B_A3B_MetalAirMoL"))

        last_raw = ""
        last_diag = ""
        for r_key, r_title in rung_plan:
            cur_prompt = _build_repair_prompt(prompt, last_raw, last_diag) if (last_raw and last_diag) else prompt
            rail = "rung3" if r_key == "qwen35b" else "rung2"
            with meter_joules(rail) as jrm:
                raw_out = self._invoke_rung_model(r_key, cur_prompt, max_tokens=max_tokens)
            acc_ms += jrm.elapsed_ms
            acc_est += jrm.est_j
            if jrm.reported_j is not None:
                acc_rep += jrm.reported_j
                has_rep = True
                last_chans = jrm.reported_channels

            if not has_contract:
                code_txt = extract_python_code(raw_out)
                formatted = f"```python\n{code_txt}\n```" if "def " in code_txt else raw_out.strip()
                return MoLAgentResult(
                    content=formatted,
                    verified=False,
                    resolved_by=r_title,
                    skill_id=None,
                    latency_ms=acc_ms,
                    est_j=acc_est,
                    reported_j=acc_rep if has_rep else None,
                    reported_channels=last_chans,
                )

            ok_c, norm_c, diag_c = verify_assertion_contract(raw_out, fn_name, assert_list)
            if ok_c:
                sid = self.store.register_verified_skill(lookup_fn, prompt, norm_c, r_title)
                self.store.recompile_lut6_router()
                self.store.save_to_dir(self.mol_dir)
                return MoLAgentResult(
                    content=f"```python\n{norm_c}\n```",
                    verified=True,
                    resolved_by=f"{r_title}->Crystallized_Skill#{sid}",
                    skill_id=sid,
                    latency_ms=acc_ms,
                    est_j=acc_est,
                    reported_j=acc_rep if has_rep else None,
                    reported_channels=last_chans,
                )

            # Second repair attempt on Gemma4/35B if first draft failed C(z)
            if r_key in ("gemma4", "qwen35b"):
                rep_p2 = _build_repair_prompt(prompt, raw_out, diag_c)
                with meter_joules(rail) as jrm2:
                    raw_out2 = self._invoke_rung_model(r_key, rep_p2, max_tokens=max_tokens)
                acc_ms += jrm2.elapsed_ms
                acc_est += jrm2.est_j
                if jrm2.reported_j is not None:
                    acc_rep += jrm2.reported_j
                    has_rep = True
                    last_chans = jrm2.reported_channels
                ok_c2, norm_c2, diag_c2 = verify_assertion_contract(raw_out2, fn_name, assert_list)
                if ok_c2:
                    sid = self.store.register_verified_skill(lookup_fn, prompt, norm_c2, f"{r_title}_CxRepair")
                    self.store.recompile_lut6_router()
                    self.store.save_to_dir(self.mol_dir)
                    return MoLAgentResult(
                        content=f"```python\n{norm_c2}\n```",
                        verified=True,
                        resolved_by=f"{r_title}_CxRepair->Crystallized_Skill#{sid}",
                        skill_id=sid,
                        latency_ms=acc_ms,
                        est_j=acc_est,
                        reported_j=acc_rep if has_rep else None,
                        reported_channels=last_chans,
                    )
                last_raw, last_diag = raw_out2, diag_c2
            else:
                last_raw, last_diag = raw_out, diag_c

        # Verifier-gated safe abstention (Theorem 1: 0 broken commits admitted!)
        return MoLAgentResult(
            content=f"# [MoL SAFE ABSTENTION] Candidate failed contract C(z): {last_diag}",
            verified=False,
            resolved_by="VERIFIER_ABSTAIN_SAFE",
            skill_id=None,
            latency_ms=acc_ms,
            est_j=acc_est,
            reported_j=acc_rep if has_rep else None,
            reported_channels=last_chans,
        )

    def repair_repo(self, repo_path: Path, issue_description: str) -> dict[str, Any]:
        """Repairs a multi-file Python repository in-place ONLY if `pytest -q` passes with untouched `tests/`."""
        src_files: dict[str, str] = {}
        tst_files: dict[str, str] = {}
        for py_f in sorted(repo_path.rglob("*.py")):
            rel = py_f.relative_to(repo_path).as_posix()
            if rel.startswith(".") or "__pycache__" in rel:
                continue
            if rel.startswith("tests/") or py_f.name.startswith("test_"):
                tst_files[rel] = py_f.read_text(encoding="utf-8")
            else:
                src_files[rel] = py_f.read_text(encoding="utf-8")

        ok_init, tb_init = run_repo_pytest(repo_path)
        if ok_init:
            return {"patched": False, "verified": True, "message": "Repository already passes pytest.", "git_diff": ""}

        task = MultiFileRepoBugTask(
            task_id=f"MCP_{repo_path.name}",
            family_id=repo_path.name,
            wave=1,
            issue_title=issue_description[:80],
            issue_description=issue_description,
            source_files=src_files,
            test_files=tst_files,
        )
        sig = f"{task.family_id}\n{issue_description}"
        hit = self.store.lookup_rung0_rung1(task.family_id, sig)
        if hit is not None:
            ok_h, can_json, gdiff, _ = verify_repo_patch_candidate(task, hit.normalized_code)
            if ok_h:
                for rel_p, body in json.loads(can_json).items():
                    (repo_path / rel_p).write_text(body, encoding="utf-8")
                hit.hits += 1
                self.store.save_to_dir(self.mol_dir)
                return {
                    "patched": True,
                    "verified": True,
                    "resolved_by": f"Rung0_BenesLUT6->.mol/PatchSkill#{hit.skill_id}",
                    "skill_id": hit.skill_id,
                    "git_diff": gdiff,
                }

        prompt = format_repo_bug_prompt(task, tb_init)
        with meter_joules("rung2") as jr:
            cand_raw = self._invoke_rung_model("gemma4", prompt, max_tokens=768)
        ok_c, can_json, gdiff, diag = verify_repo_patch_candidate(task, cand_raw)
        if ok_c:
            for rel_p, body in json.loads(can_json).items():
                (repo_path / rel_p).write_text(body, encoding="utf-8")
            sid = self.store.register_verified_skill(task.family_id, sig, can_json, "MCP_RepoPatch")
            self.store.recompile_lut6_router()
            self.store.save_to_dir(self.mol_dir)
            return {
                "patched": True,
                "verified": True,
                "resolved_by": f"Rung3a_Gemma4->Crystallized_Patch#{sid}",
                "skill_id": sid,
                "git_diff": gdiff,
                "reported_j": jr.reported_j,
            }
        return {
            "patched": False,
            "verified": False,
            "resolved_by": "VERIFIER_ABSTAIN_SAFE",
            "diagnostic": diag,
            "reported_j": jr.reported_j,
        }


# =============================================================================
# Model Context Protocol (`MCP` JSON-RPC 2.0) Dispatcher (`stdio` + `POST /mcp`)
# =============================================================================

MCP_TOOLS_SCHEMA = [
    {
        "name": "mol_synthesize_verified_function",
        "description": (
            "Synthesizes a Python function verified against deterministic contract assertions `C(z)` "
            "and crystallizes verified AST skills into `.mol/` (`BenesLUT6` Rung 0/1 fast path)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "fn_name": {"type": "string", "description": "Target Python function name"},
                "specification": {"type": "string", "description": "Complete behavioral specification"},
                "assertions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Python `assert ...` statements defining `C(z)`",
                },
            },
            "required": ["fn_name", "specification", "assertions"],
        },
    },
    {
        "name": "mol_repair_repo_pytest",
        "description": (
            "Repairs a multi-file Python repository bug verified by running real `pytest -q` in an "
            "isolated sandbox with locked `tests/` (`0%` broken commits admitted)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to repository root"},
                "issue_description": {"type": "string", "description": "Bug description / invariant"},
            },
            "required": ["repo_path", "issue_description"],
        },
    },
    {
        "name": "mol_lookup_crystallized_skill",
        "description": "Queries the Rung 0 `80`-`LUT6` Beneš router & `.mol/` AST skill store in `< 1 ms`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fn_name": {"type": "string"},
                "specification": {"type": "string"},
            },
            "required": ["fn_name", "specification"],
        },
    },
    {
        "name": "mol_energy_telemetry",
        "description": "Reads live Apple Silicon dual-channel `reported_j` (`L3 RUSAGE_INFO_V6` + `L2 IOReport`).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def handle_mcp_jsonrpc(agent: MoLLocalCodingAgent, req: dict[str, Any]) -> dict[str, Any] | None:
    """Processes a single MCP JSON-RPC 2.0 message and returns the JSON-RPC response (`None` for notifications)."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params") or {}

    if req_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "openie-mol-coding-agent", "version": "0.3.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}
    if method == "tools/call":
        t_name = params.get("name", "")
        t_args = params.get("arguments") or {}
        try:
            if t_name == "mol_synthesize_verified_function":
                res = agent.synthesize(
                    prompt=str(t_args["specification"]),
                    fn_name=str(t_args["fn_name"]),
                    assertions=list(t_args.get("assertions") or []),
                )
                payload = {"code": res.content, "receipt": res.receipt_dict()}
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                        "isError": not res.verified,
                    },
                }
            if t_name == "mol_repair_repo_pytest":
                rep_out = agent.repair_repo(Path(str(t_args["repo_path"])), str(t_args["issue_description"]))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(rep_out, indent=2)}],
                        "isError": not bool(rep_out.get("verified")),
                    },
                }
            if t_name == "mol_lookup_crystallized_skill":
                hit = agent.store.lookup_rung0_rung1(str(t_args["fn_name"]), str(t_args["specification"]))
                payload = (
                    {
                        "hit": True,
                        "skill_id": hit.skill_id,
                        "fn_name": hit.fn_name,
                        "code": hit.normalized_code,
                        "source_model": hit.source_model,
                        "hits": hit.hits,
                    }
                    if hit is not None
                    else {"hit": False, "total_skills": len(agent.store.skills)}
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}], "isError": False},
                }
            if t_name == "mol_energy_telemetry":
                rdr = get_apple_reported_j_reader()
                nj, inst, cyc = rdr.read_l3_proc_rusage()
                payload = {
                    "available": rdr.available,
                    "source": "apple_silicon_l3_rusage_v6_plus_l2_ioreport_energy_model" if rdr.available else "est_only",
                    "process_cumulative_cpu_j": round(nj * 1e-9, 6),
                    "process_instructions": inst,
                    "process_cycles": cyc,
                    "crystallized_skills_in_mol": len(agent.store.skills),
                    "measured_j": None,
                    "board_metered": False,
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}], "isError": False},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown MCP tool: {t_name}"},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Tool execution error: {exc}"},
            }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


# =============================================================================
# OpenAI `/v1/models` + `/v1/chat/completions` (REST + SSE) + `/mcp` HTTP Server
# =============================================================================


class MoLAgentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], agent: MoLLocalCodingAgent) -> None:
        self.agent = agent
        super().__init__(server_address, _MoLHTTPHandler)


class _MoLHTTPHandler(BaseHTTPRequestHandler):
    server: MoLAgentHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # Keep stdout clean for structured telemetry & benchmarks

    def _send_json(self, status: int, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            rdr = get_apple_reported_j_reader()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "mol_dir": str(self.server.agent.mol_dir),
                    "crystallized_skills": len(self.server.agent.store.skills),
                    "reported_j_available": rdr.available,
                },
            )
            return
        if self.path == "/v1/models":
            now_i = int(time.time())
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "openie-mol-cascade-2026", "object": "model", "created": now_i, "owned_by": "openie"},
                        {"id": "openie-mol-rung0-lut6", "object": "model", "created": now_i, "owned_by": "openie"},
                        {"id": "openie-mol-35b-airmol", "object": "model", "created": now_i, "owned_by": "openie"},
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"Unknown endpoint {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(body_raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Invalid JSON body"}})
            return

        if self.path == "/mcp":
            resp = handle_mcp_jsonrpc(self.server.agent, body)
            if resp is None:
                self.send_response(202)
                self.end_headers()
            else:
                self._send_json(200, resp)
            return

        if self.path == "/v1/chat/completions":
            msgs = body.get("messages") or []
            user_txts = [str(m.get("content", "")) for m in msgs if m.get("role") == "user"]
            prompt = "\n\n".join(user_txts).strip()
            model_name = str(body.get("model", "openie-mol-cascade-2026"))
            max_toks = int(body.get("max_tokens") or 640)
            stream = bool(body.get("stream", False))
            contract = body.get("x_mol_contract") or {}
            fn_name = str(contract.get("fn_name", ""))
            assertions = list(contract.get("assertions") or [])

            res = self.server.agent.synthesize(
                prompt=prompt, fn_name=fn_name, assertions=assertions, max_tokens=max_toks
            )
            comp_id = f"chatcmpl-mol-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if not stream:
                self._send_json(
                    200,
                    {
                        "id": comp_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": res.content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": max(1, len(prompt.split())),
                            "completion_tokens": max(1, len(res.content.split())),
                            "total_tokens": max(2, len(prompt.split()) + len(res.content.split())),
                        },
                        "x_mol_receipt": res.receipt_dict(),
                    },
                )
                return

            # OpenAI Server-Sent Events (`stream: true`)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            lines = res.content.splitlines(keepends=True) or [res.content]
            for idx_l, chunk_txt in enumerate(lines):
                delta = {"role": "assistant", "content": chunk_txt} if idx_l == 0 else {"content": chunk_txt}
                ev = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode("utf-8"))
                self.wfile.flush()

            final_ev = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "x_mol_receipt": res.receipt_dict(),
            }
            self.wfile.write(f"data: {json.dumps(final_ev)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._send_json(404, {"error": {"message": f"Unknown endpoint {self.path}"}})


def run_mcp_stdio_loop(agent: MoLLocalCodingAgent) -> None:
    """Runs MCP JSON-RPC `2.0` over newline-delimited `stdin` -> `stdout`."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue
        resp = handle_mcp_jsonrpc(agent, req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mol-agent",
        description="OpenIE Mixture of Limits (MoL) Drop-In OpenAI (`/v1`) & MCP Local Coding Agent",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for OpenAI/MCP HTTP server")
    parser.add_argument("--port", type=int, default=8090, help="Bind port for OpenAI/MCP HTTP server")
    parser.add_argument("--mol-dir", default=".mol", help="Persistent `.mol/` skill & BenesLUT6 store directory")
    parser.add_argument("--enable-35b", action="store_true", help="Enable Rung 3c Qwen3.6-35B-A3B Metal escalation")
    parser.add_argument("--mcp-stdio", action="store_true", help="Run as MCP JSON-RPC 2.0 server over stdio")
    args = parser.parse_args(argv)

    agent = MoLLocalCodingAgent(mol_dir=Path(args.mol_dir), enable_35b_escalation=args.enable_35b)
    if args.mcp_stdio:
        run_mcp_stdio_loop(agent)
        return

    srv = MoLAgentHTTPServer((args.host, args.port), agent)
    print(
        f"[OpenIE MoL Agent] Listening on http://{args.host}:{args.port} "
        f"(OpenAI `/v1/chat/completions` + MCP `/mcp`, skills={len(agent.store.skills)} in `{args.mol_dir}`)",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    cli_main()
