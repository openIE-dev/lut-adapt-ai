"""Phase 25 Invariant Tests: Expert-Routed Sparse-MoE `AirMoL` + Verifier-Gated `VerifiedCodeCascade`."""

from __future__ import annotations

from pathlib import Path
import shutil
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel

from clai.frontier.air_mol_streamer import NVMeIOMeter
from clai.frontier.moe_air_mol import (
    ExpertRoutedMoEAirMoLEngine,
    FrontierBenesLUT6TokenDrafter,
    default_nvme_shard_root,
)
from clai.frontier.verified_code_cascade import (
    BenesLUT6SkillCascadeStore,
    TestVector,
    build_3wave_frontier_benchmark,
    verify_python_candidate,
)


def _dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_expert_routed_moe_air_mol_cuts_nvme_io_with_exact_lossless_parity(tmp_path: Path) -> None:
    """
    Verifies Phase 25 Pillar 1 (`ExpertRoutedMoEAirMoLEngine` + `FrontierBenesLUT6TokenDrafter`):
      1. Router-sparse `F_NOCACHE` `os.pread` (`dense_sweep_all=False`) reads strictly fewer NVMe
         bytes than dense `AirLLM` (`dense_sweep_all=True`) while matching `100.0%` token-for-token!
      2. Speculative `FrontierBenesLUT6TokenDrafter` (`K=3` burst + hybrid cache rollback) matches
         standalone `T=1` output `100.0%` token-for-token with fewer NVMe sweeps!
    """
    qwen_dir = ExpertRoutedMoEAirMoLEngine.resolve_local_checkpoint_dir("Qwen/Qwen3.6-35B-A3B")
    assert qwen_dir is not None and qwen_dir.exists()

    raw_cfg = AutoConfig.from_pretrained(qwen_dir)
    raw_cfg.text_config.num_hidden_layers = 2
    raw_cfg.text_config.layer_types = ["linear_attention", "full_attention"]
    raw_cfg.text_config.num_experts = 16
    raw_cfg.text_config.num_experts_per_tok = 2
    raw_cfg.text_config.hidden_size = 256
    raw_cfg.text_config.moe_intermediate_size = 128
    raw_cfg.text_config.shared_expert_intermediate_size = 128
    raw_cfg.text_config.num_attention_heads = 4
    raw_cfg.text_config.num_key_value_heads = 2
    raw_cfg.text_config.head_dim = 64
    raw_cfg.text_config.linear_num_key_heads = 4
    raw_cfg.text_config.linear_num_value_heads = 4
    raw_cfg.text_config.linear_key_head_dim = 64
    raw_cfg.text_config.linear_value_head_dim = 64
    raw_cfg.text_config.vocab_size = 1024
    raw_cfg.text_config.bos_token_id = 1
    raw_cfg.text_config.eos_token_id = 2
    raw_cfg.text_config.pad_token_id = 0

    src_d = tmp_path / "tiny_qwen_moe_src"
    shd_d = tmp_path / "tiny_qwen_moe_shd"
    src_d.mkdir()
    raw_cfg.save_pretrained(src_d)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (qwen_dir / fn).exists():
            shutil.copy2(qwen_dir / fn, src_d / fn)

    torch.manual_seed(42)
    tm = Qwen3_5MoeTextModel(raw_cfg.text_config)
    sd = {f"model.language_model.{k}": v for k, v in tm.state_dict().items()}
    sd["lm_head.weight"] = torch.randn(1024, 256)
    save_file(sd, str(src_d / "model.safetensors"))

    ExpertRoutedMoEAirMoLEngine.shard_checkpoint("Qwen/tiny-moe-test", shd_d, source_dir=src_d)
    eng = ExpertRoutedMoEAirMoLEngine(shd_d, device=_dev(), compute_dtype=torch.float32)

    p_ids = [11, 22, 33, 44, 55]
    m_dense = NVMeIOMeter()
    t_dense, sw_d, fr_d = eng.generate_tokens(p_ids, b"def x():", 0, 9, None, m_dense, dense_sweep_all=True)

    m_sparse = NVMeIOMeter()
    t_sparse, sw_s, fr_s = eng.generate_tokens(p_ids, b"def x():", 0, 9, None, m_sparse, dense_sweep_all=False)
    assert t_sparse == t_dense
    assert fr_s < 0.45 and fr_d == 1.0
    assert m_sparse.bytes_read < 0.65 * m_dense.bytes_read

    # Partial + full burst draft rollback verification
    call_n = {"n": 0}

    def mixed_draft(ctx: list[int], _tb: bytes, _lid: int) -> list[int]:
        call_n["n"] += 1
        emitted_so_far = len(ctx) - len(p_ids)
        if call_n["n"] == 1:
            return [t_dense[emitted_so_far], t_dense[emitted_so_far + 1], (t_dense[emitted_so_far + 2] + 5) % 1024]
        return t_dense[emitted_so_far : emitted_so_far + 3]

    m_spec = NVMeIOMeter()
    t_spec, sw_spec, _ = eng.generate_tokens(p_ids, b"def x():", 0, 9, mixed_draft, m_spec, dense_sweep_all=False)
    assert t_spec == t_dense
    assert sw_spec < sw_s


def test_cz_contract_verifier_and_benes_lut6_skill_store_reject_spec_drift() -> None:
    """
    Verifies Phase 25 Pillar 2 (`verify_python_candidate` + `BenesLUT6SkillCascadeStore`):
      1. `C(z)` catches subtle RFC-6901 `~01` unescaping-order bugs and blocks forbidden imports.
      2. `BenesLUT6SkillCascadeStore` compiles `80` `LUT6`s + `512x512` Beneš routing for crystallized
         skills, and `C(z)` safely rejects stale skills under Wave-2 adversarial specification drift!
    """
    tasks = build_3wave_frontier_benchmark()
    assert len(tasks) == 48

    # Buggy RFC-6901 implementation unescaping `~0` BEFORE `~1` (classic 1-shot LLM bug)
    buggy_rfc6901 = """
def resolve_json_pointer(doc, pointer: str, default=None):
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        return default
    cur = doc
    for raw_tok in pointer[1:].split("/"):
        tok = raw_tok.replace("~0", "~").replace("~1", "/")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return default
    return cur
"""
    t_rfc = next(t for t in tasks if t.family_id == "F07_rfc6901_pointer" and t.wave == 1)
    ok_bug, _, diag = verify_python_candidate(buggy_rfc6901, t_rfc.fn_name, t_rfc.tests)
    assert not ok_bug and "/a~01b" in diag

    # Forbidden import must be blocked
    ok_imp, _, diag_imp = verify_python_candidate("import subprocess\ndef f(x): return x", "f", [TestVector((1,), 1)])
    assert not ok_imp and "Forbidden import" in diag_imp

    # Infinite loop emitted by an LLM must be bounded by the 0.35s ITIMER_REAL timeout
    ok_loop, _, diag_loop = verify_python_candidate(
        "def f(x):\n    while True:\n        x += 1\n    return x",
        "f",
        [TestVector((1,), 2)],
    )
    assert not ok_loop and "TimeoutError" in diag_loop

    # Register valid Wave-1 CRC-8/ATM skill into BenesLUT6SkillCascadeStore, then test Wave-2 mutated CRC-8/INV!
    t_crc_w1 = next(t for t in tasks if t.family_id == "F05_crc8_atm" and t.wave == 1)
    t_crc_w2_mut = next(t for t in tasks if t.family_id == "F05_crc8_atm_mut" and t.wave == 2)
    valid_crc_w1 = """
def crc8_check(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc
"""
    ok_w1, norm_w1, _ = verify_python_candidate(valid_crc_w1, t_crc_w1.fn_name, t_crc_w1.tests)
    assert ok_w1

    store = BenesLUT6SkillCascadeStore()
    sid1 = store.register_verified_skill(t_crc_w1.fn_name, t_crc_w1.prompt, norm_w1, "test")
    store.recompile_lut6_router()

    hit_w1 = store.lookup_rung0_rung1(t_crc_w1.fn_name, t_crc_w1.prompt)
    assert hit_w1 is not None and hit_w1.skill_id == sid1

    # Even if Rung 0/1 retrieves the Wave-1 skill on Wave-2 mutated prompt, C(z) MUST reject it!
    cand_mut = store.lookup_rung0_rung1(t_crc_w2_mut.fn_name, t_crc_w2_mut.prompt)
    if cand_mut is not None:
        ok_stale, _, _ = verify_python_candidate(cand_mut.normalized_code, t_crc_w2_mut.fn_name, t_crc_w2_mut.tests)
        assert not ok_stale, "Stale Wave-1 CRC-8/ATM skill must NEVER pass Wave-2 mutated C(z)!"


def test_real_sharded_moe_artifacts_and_bakeoff_csv_invariants() -> None:
    """Checks persisted Phase 25 MoE & Bakeoff CSV artifacts once generated."""
    moe_csv = Path("artifacts/hf_2026_moe_8b_35b_air_mol_baselines.csv")
    bake_csv = Path("artifacts/frontier_ai_head_to_head_bakeoff.csv")
    if moe_csv.exists():
        txt = moe_csv.read_text(encoding="utf-8")
        assert "LiquidAI/LFM2.5-8B-A1B" in txt
        assert "Qwen/Qwen3.6-35B-A3B" in txt
    if bake_csv.exists():
        txt_b = bake_csv.read_text(encoding="utf-8")
        assert "G_OpenIE_MoL_VerifiedCascade_OURS" in txt_b
        shd_root = default_nvme_shard_root()
        assert shd_root.exists()


def test_sub1bit_lut6_factor_matches_dense_binary_and_procrustes_improves_snr() -> None:
    """
    Verifies Phase 25b clean-room `Sub1BitLUT6FactorLinear`:
      1. Procrustes hypercube rotation (`LittleBit-2` principle) strictly improves reconstruction SNR
         over unrotated SVD (`LittleBit-1` principle) at matched `< 0.7 bpw`.
      2. Hardware-faithful 6-input `LUT6` partial-sum lookup (`forward_lut6`, `0` binary MACs) matches
         dense reconstructed binary matmul within floating-point tolerance (`< 1e-3`).
    """
    from clai.frontier.sub1bit_lut_factor import Sub1BitLUT6FactorLinear

    torch.manual_seed(17)
    # Construct structured low-effective-rank + heavy-tailed weight matrix resembling LLM FFN projections
    u0 = torch.randn(192, 48)
    v0 = torch.randn(48, 256)
    w = (u0 @ v0) / 6.0 + 0.15 * torch.randn(192, 256)

    m_svd = Sub1BitLUT6FactorLinear.from_pretrained_weight(
        w, target_bpw=0.65, num_stages=2, use_procrustes_rotation=False
    )
    m_itq = Sub1BitLUT6FactorLinear.from_pretrained_weight(
        w, target_bpw=0.65, num_stages=2, use_procrustes_rotation=True, procrustes_steps=25
    )
    assert m_itq.effective_bpw < 1.0 and m_itq.rank_r % 6 == 0

    err_svd = (w - m_svd.reconstruct_dense()).pow(2).sum().item()
    err_itq = (w - m_itq.reconstruct_dense()).pow(2).sum().item()
    assert err_itq < err_svd, f"Expected Procrustes hypercube rotation to reduce MSE ({err_itq} vs {err_svd})"

    x = torch.randn(8, 256)
    y_lut6 = m_itq.forward_lut6(x)
    y_dense = x @ m_itq.reconstruct_dense().t()
    assert torch.allclose(y_lut6, y_dense, atol=1e-4)
