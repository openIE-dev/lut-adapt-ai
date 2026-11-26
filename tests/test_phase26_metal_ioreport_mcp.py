"""Phase 26 Invariant Tests:
  1. Apple Silicon Dual-Channel `reported_j` (`L3 RUSAGE_INFO_V6` + `L2 IOReport` `'GPU Energy'`, `measured_j = None`).
  2. Native Apple Metal (`torch.mps.compile_shader`) `INT4` MoE & `GatedDeltaNet` `< 1e-4` numerical parity.
  3. Persistent `.mol/` `BenesLUT6` Skill Store + Multi-File Git-Repo `pytest` Verifier (`tests/` tamper guard).
  4. Live OpenAI `/v1/models` + `/v1/chat/completions` (REST + SSE) and `MCP` JSON-RPC `2.0` (`POST /mcp`) Agent Server.
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import threading
import urllib.request

import torch
import torch.nn.functional as F

from clai.contracts.joule_meter import (
    get_apple_reported_j_reader,
    meter_joules,
    select_torch_device,
)
from clai.frontier.metal_moe_kernels import (
    fused_gated_delta_rule_metal,
    fused_q4_moe_ffn_metal,
    fused_q4_moe_ffn_metal_direct,
    get_metal_moe_library,
)
from clai.frontier.repo_patch_cascade import (
    build_swebench_mini_repo_tasks,
    materialize_git_repo,
    run_repo_pytest,
    verify_repo_patch_candidate,
)
from clai.frontier.verified_code_cascade import BenesLUT6SkillCascadeStore
from clai.service.openai_mcp_agent import MoLAgentHTTPServer, MoLLocalCodingAgent


def test_apple_silicon_dual_channel_reported_j_receipt() -> None:
    """Verifies strict 3-tier energy separation (`est_j`, `reported_j`, `measured_j is None`)."""
    dev = select_torch_device()
    if dev.type == "mps":
        _ = torch.ones((64, 64), device=dev) @ torch.ones((64, 64), device=dev)
        torch.mps.synchronize()
    with meter_joules("rung3", device=dev) as rec:
        a = torch.randn((2048, 2048), device=dev)
        b = torch.randn((2048, 2048), device=dev)
        for _ in range(48):
            a = a @ b * 0.01

    assert rec.measured_j is None, "Unmetered host must strictly report `measured_j = None`!"
    assert rec.est_j > 0.0
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        rdr = get_apple_reported_j_reader()
        assert rdr.available is True
        assert rec.reported_j is not None and rec.reported_j > 0.0
        assert rec.reported_channels["cpu_instructions"] > 0
        assert rec.reported_channels["cpu_cycles"] > 0
        if dev.type == "mps":
            assert rec.reported_channels["l2_gpu_j"] > 0.0


def test_metal_int4_moe_and_gated_deltanet_numerical_parity() -> None:
    """Verifies `< 1e-4` FP32 numerical parity of custom Metal `INT4` MoE & `GatedDeltaNet` shaders."""
    dev = select_torch_device()
    if dev.type != "mps":
        return
    assert get_metal_moe_library() is not None

    torch.manual_seed(123)
    B, T, H_k, H_v, K_d, V_d = 1, 32, 4, 8, 128, 64
    q = torch.randn((B, T, H_k, K_d), device=dev, dtype=torch.float32)
    k = torch.randn((B, T, H_k, K_d), device=dev, dtype=torch.float32)
    v = torch.randn((B, T, H_v, V_d), device=dev, dtype=torch.float32)
    g = -torch.rand((B, T, H_v), device=dev, dtype=torch.float32) * 0.2
    beta = torch.sigmoid(torch.randn((B, T, H_v), device=dev, dtype=torch.float32))
    s0 = torch.randn((B, H_v, K_d, V_d), device=dev, dtype=torch.float32) * 0.05

    # Compare MPS Metal shader against CPU fallback reference inside `fused_gated_delta_rule_metal`
    out_cpu, st_cpu = fused_gated_delta_rule_metal(
        q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), initial_state=s0.cpu(), output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    out_mtl, st_mtl = fused_gated_delta_rule_metal(
        q, k, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert st_cpu is not None and st_mtl is not None
    assert float((st_cpu - st_mtl.cpu()).abs().max().item()) < 5e-5
    assert float((out_cpu - out_mtl.cpu()).abs().max().item()) < 5e-5

    # Compare both `fused_q4_moe_ffn_metal` (compact slots) and `fused_q4_moe_ffn_metal_direct` (full expert bank)
    D, I_dim, E_tot = 256, 128, 16
    m_gu = (2 * I_dim * D) // 64
    m_dn = (D * I_dim) // 64
    m_tot = m_gu + m_dn
    packed_all = torch.randint(0, 256, (E_tot, m_tot, 32), dtype=torch.uint8, device=dev)
    mins_all = (torch.randn((E_tot, m_tot), device=dev) * 0.02).to(torch.float16)
    scales_all = (torch.rand((E_tot, m_tot), device=dev) * 0.005 + 1e-4).to(torch.float16)
    xh = torch.randn((3, D), dtype=torch.float32, device=dev)
    sel = torch.tensor([[1, 5], [5, 12], [0, 1]], dtype=torch.long, device=dev)
    rw = torch.softmax(torch.randn((3, 2), dtype=torch.float32, device=dev), dim=-1)
    uniq_e = sorted(set(sel.reshape(-1).tolist()))

    out_direct = fused_q4_moe_ffn_metal_direct(xh, sel, rw, packed_all, mins_all, scales_all, D, I_dim, m_gu, m_tot)
    out_compact = fused_q4_moe_ffn_metal(
        xh, sel, rw, uniq_e, packed_all[uniq_e], mins_all[uniq_e], scales_all[uniq_e], D, I_dim, m_gu, m_tot
    )
    assert float((out_direct - out_compact).abs().max().item()) < 1e-6

    # Unpacked PyTorch reference check
    q_u = torch.empty((E_tot, m_tot, 64), dtype=torch.float32, device=dev)
    q_u[:, :, 0::2] = (packed_all & 0x0F).float()
    q_u[:, :, 1::2] = (packed_all >> 4).float()
    deq = mins_all.float().unsqueeze(-1) + q_u * scales_all.float().unsqueeze(-1)
    ref = torch.zeros_like(xh)
    for e_id in uniq_e:
        gu_w = deq[e_id, :m_gu].reshape(2 * I_dim, D)
        dn_w = deq[e_id, m_gu:].reshape(D, I_dim)
        ti, kp = torch.where(sel == e_id)
        gp, up = F.linear(xh[ti], gu_w).chunk(2, dim=-1)
        ref.index_add_(0, ti, F.linear(F.silu(gp) * up, dn_w) * rw[ti, kp].unsqueeze(-1))
    assert float((ref - out_direct).abs().max().item()) < 5e-5


def test_persistent_mol_store_and_multifile_repo_patch_verifier(tmp_path: Path) -> None:
    """Verifies `.mol/` disk persistence and `RepoPatchCascade` `pytest` + test-tampering guards."""
    mol_dir = tmp_path / ".mol_test"
    store = BenesLUT6SkillCascadeStore()
    sid = store.register_verified_skill("add_mod", "def add_mod(a, b, m): return (a + b) % m", "def add_mod(a, b, m):\n    return (a + b) % m", "unit")
    store.save_to_dir(mol_dir)
    assert (mol_dir / "skills_v1.json").exists()
    assert (mol_dir / "benes_lut6_router.npz").exists()

    reloaded = BenesLUT6SkillCascadeStore.load_from_dir(mol_dir)
    hit = reloaded.lookup_rung0_rung1("add_mod", "def add_mod(a, b, m): return (a + b) % m")
    assert hit is not None and hit.skill_id == sid

    # Multi-file repository patch verifier checks
    tasks = build_swebench_mini_repo_tasks()
    t_r01 = next(t for t in tasks if t.family_id == "R01_billing_proration" and t.wave == 1)

    # 1. Reward-hacking attempt modifying `tests/test_billing.py` MUST be blocked!
    tamper_json = json.dumps({"tests/test_billing.py": "def test_pass(): assert True\n"})
    ok_tamper, _, _, diag_tamper = verify_repo_patch_candidate(t_r01, tamper_json)
    assert not ok_tamper and "Security violation" in diag_tamper

    # 2. Valid largest-remainder fix on `billing/prorate.py` MUST pass subprocess `pytest` and emit `git diff`!
    valid_fix = (
        "```python path=billing/prorate.py\n"
        "from billing.types import LineItem\n\n"
        "def allocate_prorated_cents(total_cents: int, items: list[LineItem]) -> dict[str, int]:\n"
        "    if total_cents < 0:\n"
        "        raise ValueError('total_cents must be >= 0')\n"
        "    valid = [it for it in items if it.weight > 0]\n"
        "    if not valid:\n"
        "        return {}\n"
        "    tot_w = sum(it.weight for it in valid)\n"
        "    floors = [(total_cents * it.weight) // tot_w for it in valid]\n"
        "    rem = total_cents - sum(floors)\n"
        "    order = sorted(range(len(valid)), key=lambda i: (-((total_cents * valid[i].weight) % tot_w), i))\n"
        "    for idx in order[:rem]:\n"
        "        floors[idx] += 1\n"
        "    return {it.sku: floors[i] for i, it in enumerate(valid)}\n"
        "```\n"
    )
    ok_fix, can_json, gdiff, diag_ok = verify_repo_patch_candidate(t_r01, valid_fix)
    assert ok_fix, f"Valid fix failed: {diag_ok}"
    assert "billing/prorate.py" in can_json and "+    floors =" in gdiff


def test_live_openai_v1_rest_sse_and_mcp_jsonrpc_agent_server(tmp_path: Path) -> None:
    """Verifies live HTTP/SSE OpenAI `/v1/chat/completions` + MCP JSON-RPC `2.0` (`POST /mcp`)."""
    mol_dir = tmp_path / ".mol_live"

    def mock_gen(rung_key: str, prompt: str, _max_toks: int) -> str:
        # Simulate Rung 2 (`lfm3`) making an off-by-one bug so `C(z)` catches it and escalates to `gemma4`!
        if "clamp_int" in prompt:
            if rung_key == "lfm3":
                return "```python\ndef clamp_int(x: int, lo: int, hi: int) -> int:\n    return max(lo, min(hi - 1, x))\n```"
            return "```python\ndef clamp_int(x: int, lo: int, hi: int) -> int:\n    return max(lo, min(hi, x))\n```"
        # Repo repair mock for `R02_dag_scheduler`
        return (
            "```python path=scheduler/executor.py\n"
            "from scheduler.dag import TaskDAG\n\n"
            "class DAGExecutor:\n"
            "    def __init__(self, dag: TaskDAG) -> None:\n"
            "        self.dag = dag\n\n"
            "    def run_waves(self) -> list[list[str]]:\n"
            "        done: set[str] = set()\n"
            "        waves: list[list[str]] = []\n"
            "        while len(done) < len(self.dag.deps):\n"
            "            ready = sorted(t for t, reqs in self.dag.deps.items() if t not in done and reqs <= done)\n"
            "            if not ready:\n"
            "                raise ValueError('cycle detected')\n"
            "            waves.append(ready)\n"
            "            done.update(ready)\n"
            "        return waves\n"
            "```\n"
        )

    agent = MoLLocalCodingAgent(mol_dir=mol_dir, generator_fn=mock_gen)
    srv = MoLAgentHTTPServer(("127.0.0.1", 0), agent)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{port}"

    def _post_json(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    try:
        # 1. GET /v1/models
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0) as r_mod:  # noqa: S310
            models_obj = json.loads(r_mod.read().decode("utf-8"))
        assert any(m["id"] == "openie-mol-cascade-2026" for m in models_obj["data"])

        # 2. POST /v1/chat/completions (Wave 1 Cold: `lfm3` fails C(z) -> `gemma4` repairs & crystallizes Skill #1!)
        spec_txt = "Implement `def clamp_int(x: int, lo: int, hi: int) -> int:` clamping `x` into `[lo, hi]`."
        c1 = _post_json(
            "/v1/chat/completions",
            {
                "model": "openie-mol-cascade-2026",
                "messages": [{"role": "user", "content": spec_txt}],
                "stream": False,
                "x_mol_contract": {
                    "fn_name": "clamp_int",
                    "assertions": ["assert clamp_int(15, 0, 10) == 10", "assert clamp_int(-3, 0, 10) == 0"],
                },
            },
        )
        rec1 = c1["x_mol_receipt"]
        assert rec1["verified"] is True
        assert rec1["skill_id"] == 1
        assert rec1["measured_j"] is None
        assert "Gemma4" in rec1["resolved_by"]

        # 3. POST /v1/chat/completions (`stream: true` SSE Wave 2 Replay -> Rung 0/1 `BenesLUT6` fast-path hit!)
        req_sse = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "openie-mol-cascade-2026",
                    "messages": [{"role": "user", "content": spec_txt}],
                    "stream": True,
                    "x_mol_contract": {
                        "fn_name": "clamp_int",
                        "assertions": ["assert clamp_int(10, 0, 10) == 10"],
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_sse, timeout=10.0) as r_sse:  # noqa: S310
            sse_txt = r_sse.read().decode("utf-8")
        assert "data: [DONE]" in sse_txt
        assert "Rung0_BenesLUT6->Rung1_Skill#1" in sse_txt

        # 4. MCP JSON-RPC 2.0 (`initialize`, `tools/list`, `mol_lookup_crystallized_skill`, `mol_repair_repo_pytest`)
        init_res = _post_json("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init_res["result"]["serverInfo"]["name"] == "openie-mol-coding-agent"

        tools_res = _post_json("/mcp", {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tool_names = {t["name"] for t in tools_res["result"]["tools"]}
        assert {
            "mol_synthesize_verified_function",
            "mol_repair_repo_pytest",
            "mol_lookup_crystallized_skill",
            "mol_energy_telemetry",
        } <= tool_names

        lk_res = _post_json(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "mol_lookup_crystallized_skill", "arguments": {"fn_name": "clamp_int", "specification": spec_txt}},
            },
        )
        lk_obj = json.loads(lk_res["result"]["content"][0]["text"])
        assert lk_obj["hit"] is True and lk_obj["skill_id"] == 1

        # Materialize buggy `R02_dag_scheduler` git repo on disk & repair it via MCP `mol_repair_repo_pytest`!
        t_r02 = next(t for t in build_swebench_mini_repo_tasks() if t.family_id == "R02_dag_scheduler" and t.wave == 1)
        repo_tmp = tmp_path / "r02_repo"
        repo_tmp.mkdir()
        materialize_git_repo(repo_tmp, t_r02.source_files, t_r02.test_files)
        assert run_repo_pytest(repo_tmp)[0] is False

        fix_res = _post_json(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "mol_repair_repo_pytest",
                    "arguments": {"repo_path": str(repo_tmp), "issue_description": t_r02.issue_description},
                },
            },
        )
        assert fix_res["result"]["isError"] is False
        assert run_repo_pytest(repo_tmp)[0] is True
    finally:
        srv.shutdown()
        srv.server_close()
