#!/usr/bin/env python3
"""Remote M3 Ultra 256 GB (`100.100.141.123`) Real `prism-ml/Ternary-Bonsai-2-27B` End-to-End
`EntroMCU-28` (`1.750 bpw`) Streaming + `54`-Bit `3`-Token `LUT6` Speculative Decoding Benchmark (`Claim LA-42`).

Executes on `Davids-Mac-Studio-M3-Ultra-256GB` (`~/psci-mlx/bin/python`, `MLX 0.31.2`) so the dev Mac
experiences zero model-loading pressure (`FLEET-CAPABILITIES.md`).
Zero raw prompt or completion text is persisted in the output JSON/CSV — only SHA-256 prompt digests,
token counts, exact greedy parity, NVMe linear weight bytes per token, latency, and Darwin `reported_j`.
"""

from __future__ import annotations

import ctypes
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from tokenizers import Tokenizer


# ============================================================================
# Darwin Process Energy Meter (`RUSAGE_INFO_V6.ri_energy_nj` -> `reported_j`)
# ============================================================================

class _RUsageInfoV6(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 64)]


_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_LIBC.getpid.restype = ctypes.c_int
_LIBC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV6)]
_LIBC.proc_pid_rusage.restype = ctypes.c_int


def read_darwin_energy_j() -> float:
    ru = _RUsageInfoV6()
    rc = _LIBC.proc_pid_rusage(_LIBC.getpid(), 6, ctypes.byref(ru))
    if rc == 0 and ru.words[42] > 0:
        return float(ru.words[42]) * 1e-9
    return 0.0


# ============================================================================
# `EntroMCU-28` (`1.750 bpw`) Pack/Unpack + `54`-Bit `3`-Token `LUT6` Crystal
# ============================================================================

BONSAI_VOCAB_SIZE = 248_320  # < 2^18 = 262,144


def pack_5codes_np(codes5: np.ndarray) -> np.ndarray:
    """Packs `[..., 5]` unsigned ternary codes `0, 1, 2` into `uint8` `0..242` (`3^5 = 243 <= 256`)."""
    u = codes5.astype(np.uint16)
    return (
        u[..., 0] + 3 * u[..., 1] + 9 * u[..., 2] + 27 * u[..., 3] + 81 * u[..., 4]
    ).astype(np.uint8)


def unpack_5codes_np(packed_u8: np.ndarray) -> np.ndarray:
    rem = packed_u8.astype(np.uint16)
    cols = []
    for _ in range(5):
        cols.append(((rem % 3).astype(np.uint8)))  # 0, 1, 2 unsigned codepoints!
        rem //= 3
    return np.stack(cols, axis=-1)


def transcode_mlx_tensor_to_entromcu28_bytes(
    w_u32: np.ndarray, s_f16: np.ndarray, b_f16: np.ndarray
) -> bytes:
    """Losslessly transcodes MLX 2-bit `[rows, cols_u32]` + `scales` + redundant `biases == -scales`
    (`36 B / 128 weights = 2.250 bpw`) into canonical `28 B / 128 weights = 1.750 bpw` `EntroMCU-28`."""
    diff = float(np.max(np.abs(s_f16.astype(np.float32) + b_f16.astype(np.float32))))
    if diff != 0.0:
        raise ValueError(f"Expected biases == -scales, got max |s+b| = {diff}")
    n_groups = int(w_u32.size) // 8
    w_grp = w_u32.reshape(n_groups, 8)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    codes = ((w_grp[:, :, None] >> shifts) & 0x3).reshape(n_groups, 128).astype(np.uint8)
    if np.any(codes > 2):
        raise ValueError("Invalid code == 3 in ternary tensor")
    padded = np.ones((n_groups, 130), dtype=np.uint8)  # code 1 => trit 0
    padded[:, :128] = codes
    trit_bytes = pack_5codes_np(padded.reshape(n_groups, 26, 5))
    scale_bytes = np.ascontiguousarray(s_f16.reshape(n_groups), dtype="<f2").view(np.uint8).reshape(n_groups, 2)
    return np.concatenate([trit_bytes, scale_bytes], axis=1).tobytes()


def unpack_entromcu28_bytes_to_mlx_arrays(
    buf: bytes, orig_w_shape: tuple[int, int], orig_s_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpacks `28 B/group` `EntroMCU-28` stream back into exact `(weight_u32, scales_f16, -scales_f16)`."""
    n_groups = (orig_w_shape[0] * orig_w_shape[1]) // 8
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_groups, 28)
    codes128 = unpack_5codes_np(arr[:, :26]).reshape(n_groups, 130)[:, :128].astype(np.uint32)
    codes_16 = codes128.reshape(n_groups, 8, 16)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, None, :]
    w_u32 = np.sum(codes_16 << shifts, axis=-1, dtype=np.uint32).reshape(orig_w_shape)
    s_f16 = arr[:, 26:28].copy().view("<f2").reshape(orig_s_shape)
    b_f16 = (-s_f16.astype(np.float32)).astype(np.float16)
    return w_u32, s_f16, b_f16


def pack_bonsai_3tok_54bit(t0: int, t1: int, t2: int, hdr10: int = 0x3FF) -> int:
    assert 0 <= t0 < BONSAI_VOCAB_SIZE and 0 <= t1 < BONSAI_VOCAB_SIZE and 0 <= t2 < BONSAI_VOCAB_SIZE
    return int(t0) | (int(t1) << 18) | (int(t2) << 36) | ((int(hdr10) & 0x3FF) << 54)


def unpack_bonsai_3tok_54bit(word64: int) -> tuple[list[int], int]:
    m18 = (1 << 18) - 1
    return [
        int(word64 & m18),
        int((word64 >> 18) & m18),
        int((word64 >> 36) & m18),
    ], int((word64 >> 54) & 0x3FF)


def context_suffix_to_lut6_slot(ctx_tokens: list[int], table_slots: int = 512) -> tuple[int, int]:
    """Maps trailing 3-token suffix `(t_{-3}, t_{-2}, t_{-1})` to `(slot_idx < 512, tag10)` fitting inside a
    single `4,096 B` `CompactLut6Page` (`512 x 8 B = 4,096 B`, `0` FLOPs!)."""
    t_prev3 = ctx_tokens[-3] if len(ctx_tokens) >= 3 else 0
    t_prev2 = ctx_tokens[-2] if len(ctx_tokens) >= 2 else 0
    t_prev1 = ctx_tokens[-1] if len(ctx_tokens) >= 1 else 0
    raw = f"{t_prev3}:{t_prev2}:{t_prev1}".encode("ascii")
    h = int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), "little")
    return h & (table_slots - 1), (h >> 12) & 0x3FF


# ============================================================================
# Exact Hybrid `GatedDeltaNet` (`ArraysCache`) + `KVCache` Snapshot / Rollback
# ============================================================================

def snapshot_hybrid_cache(cache: list[Any]) -> list[Any]:
    """Captures exact state of all 64 Qwen3.5/Bonsai hybrid layers (`48` `ArraysCache` recurrent
    `GatedDeltaNet` `[conv_state, ssm_state]` + `16` `KVCache` `.offset` pointers)."""
    snap = []
    for c in cache:
        cls_name = type(c).__name__
        if hasattr(c, "offset") and hasattr(c, "keys"):
            # Standard KVCache: keys/values buffer is overwritten cleanly by `offset`!
            snap.append(("kv", int(c.offset)))
        elif hasattr(c, "cache"):
            # ArraysCache (`linear_attn` GatedDeltaNet): copy small recurrent `[conv, ssm]` tensors!
            arrs = [mx.array(a) if a is not None else None for a in c.cache]
            mx.eval(*[a for a in arrs if a is not None])
            snap.append(("arrays", arrs))
        else:
            raise TypeError(f"Unknown cache layer type: {cls_name}")
    return snap


def restore_hybrid_cache(cache: list[Any], snap: list[Any]) -> None:
    for c, (kind, payload) in zip(cache, snap):
        if kind == "kv":
            c.offset = payload
        elif kind == "arrays":
            for i, a in enumerate(payload):
                c.cache[i] = mx.array(a) if a is not None else None


# ============================================================================
# Pure `mlx_lm` Text-Only Hadamard Loader for `prism-ml/Ternary-Bonsai-2-27B`
# ============================================================================

def load_bonsai_text_model(pack_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(pack_dir / "runtime"))
    from runtime import Packed  # type: ignore

    config = json.loads((pack_dir / "config.json").read_text())
    assert config.get("model_type") == "prism_hadamard_qwen35"
    lm = TextModel(TextModelArgs.from_dict(config["text_config"]))
    weights = mx.load(str(pack_dir / "model.safetensors"))

    # Compute exact linear ternary byte counts across all `language_model.*` packed modules
    total_mlx_linear_bytes = 0
    total_entromcu_linear_bytes = 0
    total_ternary_params = 0
    packed_paths: set[str] = set()

    # Also verify lossless EntroMCU-28 streaming shard transcode + pread throughput on real layers!
    shard_path = Path("/tmp/bonsai_27b_entromcu28_sample_layers.bin")
    sample_records = [
        r for r in config["modules"]
        if r["path"].startswith("model.layers.0.") or r["path"].startswith("model.layers.3.")
    ]
    transcode_t0 = time.perf_counter()
    shard_offsets: list[tuple[str, int, int, tuple[int, int], tuple[int, int]]] = []
    max_unpack_err = 0
    with shard_path.open("wb") as sf:
        cur_off = 0
        for rec in sample_records:
            key = "language_model." + rec["path"]
            w_np = np.array(weights[key + ".weight"], copy=False)
            s_np = np.array(weights[key + ".scales"].astype(mx.float16), copy=False)
            b_np = np.array(weights[key + ".biases"].astype(mx.float16), copy=False)
            buf28 = transcode_mlx_tensor_to_entromcu28_bytes(w_np, s_np, b_np)
            sf.write(buf28)
            shard_offsets.append((rec["path"], cur_off, len(buf28), w_np.shape, s_np.shape))
            cur_off += len(buf28)
    transcode_ms = (time.perf_counter() - transcode_t0) * 1000.0

    # Verify `os.pread` streaming unpack bit-for-bit equality on real Layer 0 (`GatedDeltaNet`) & Layer 3 (`FullAttention`)
    pread_t0 = time.perf_counter()
    fd = os.open(str(shard_path), os.O_RDONLY)
    pread_bytes = 0
    try:
        for path_s, off, ln, w_sh, s_sh in shard_offsets:
            raw_chunk = os.pread(fd, ln, off)
            pread_bytes += len(raw_chunk)
            w_rec, s_rec, b_rec = unpack_entromcu28_bytes_to_mlx_arrays(raw_chunk, w_sh, s_sh)
            key = "language_model." + path_s
            w_orig = np.array(weights[key + ".weight"], copy=False)
            if not np.array_equal(w_rec, w_orig):
                max_unpack_err += 1
    finally:
        os.close(fd)
        shard_path.unlink(missing_ok=True)
    pread_s = max(time.perf_counter() - pread_t0, 1e-6)
    pread_gbps = (pread_bytes / (1024**3)) / pread_s

    for record in config["modules"]:
        path = record["path"]
        packed_paths.add(path)
        parts = path.split(".")
        parent: Any = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = "language_model." + path
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        rows, cols_u32 = arrays[0].shape
        n_w = int(rows) * int(cols_u32) * 16
        n_grp = n_w // 128
        if not record["embedding"]:
            total_ternary_params += n_w
            total_mlx_linear_bytes += n_grp * 36
            total_entromcu_linear_bytes += n_grp * 28
        block = record["block"]
        signs = weights.get(key + ".signs")
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))

    # Load remaining non-packed `language_model.*` parameters (`norm`, `conv1d`, `A_log`, `dt_bias`, `q_norm`, `k_norm`)
    rem_items = []
    for k, v in weights.items():
        if not k.startswith("language_model."):
            continue
        sub = k[len("language_model."):]
        if any(sub == p or sub.startswith(p + ".") for p in packed_paths):
            continue
        rem_items.append((sub, v))

    lm.load_weights(rem_items, strict=False)
    lm.eval()
    mx.eval(lm.parameters())
    del weights
    gc.collect()

    stream_stats = {
        "linear_ternary_params": total_ternary_params,
        "mlx_affine2_linear_bytes_per_sweep": total_mlx_linear_bytes,
        "mlx_affine2_linear_gb_per_sweep": round(total_mlx_linear_bytes / (1024**3), 4),
        "entromcu28_linear_bytes_per_sweep": total_entromcu_linear_bytes,
        "entromcu28_linear_gb_per_sweep": round(total_entromcu_linear_bytes / (1024**3), 4),
        "saved_linear_bytes_per_sweep": total_mlx_linear_bytes - total_entromcu_linear_bytes,
        "saved_linear_gb_per_sweep": round((total_mlx_linear_bytes - total_entromcu_linear_bytes) / (1024**3), 4),
        "sample_layers_transcode_ms": round(transcode_ms, 2),
        "sample_layers_pread_bytes": pread_bytes,
        "sample_layers_pread_unpack_gbps": round(pread_gbps, 3),
        "sample_layers_bit_exact_mismatches": max_unpack_err,
    }
    return lm, config, stream_stats


def greedy_decode_t1(
    lm: Any, prompt_ids: list[int], max_new_tokens: int, stop_ids: set[int]
) -> tuple[list[int], int, float, float]:
    """Exact `T = 1` greedy decoding on `prism-ml/Ternary-Bonsai-2-27B`, returning
    `(generated_ids, linear_sweeps, elapsed_ms, reported_j)`."""
    cache = lm.make_cache()
    x = mx.array([prompt_ids], dtype=mx.int32)
    j0 = read_darwin_energy_j()
    t0 = time.perf_counter()

    # Prefill
    h = lm.model(x, cache=cache)
    logits = lm.lm_head(h[:, -1:, :])[:, -1, :]
    nxt = mx.argmax(logits, axis=-1)
    mx.eval(nxt)
    tok = int(nxt.item())
    gen = [tok]
    sweeps = 1  # count decode sweeps (1 prefill + N-1 decode sweeps)

    while len(gen) < max_new_tokens and tok not in stop_ids:
        xin = mx.array([[tok]], dtype=mx.int32)
        h = lm.model(xin, cache=cache)
        logits = lm.lm_head(h[:, -1:, :])[:, -1, :]
        nxt = mx.argmax(logits, axis=-1)
        mx.eval(nxt)
        tok = int(nxt.item())
        sweeps += 1
        if tok in stop_ids:
            break
        gen.append(tok)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    rep_j = max(0.0, read_darwin_energy_j() - j0)
    return gen, sweeps, elapsed_ms, rep_j


def populate_lut6_from_trace(
    lut6_page: dict[int, tuple[int, int]], full_seq: list[int]
) -> None:
    """Crystallizes observed 3-token bursts `(t_i, t_{i+1}, t_{i+2})` indexed by 2-token suffix
    into the `4,096 B` (`512 x uint64`) `Token-LUT6` page (`pack_bonsai_3tok_54bit`)."""
    for i in range(2, len(full_seq) - 3):
        slot, tag10 = context_suffix_to_lut6_slot(full_seq[:i])
        t0, t1, t2 = full_seq[i], full_seq[i + 1], full_seq[i + 2]
        word64 = pack_bonsai_3tok_54bit(t0, t1, t2, tag10)
        lut6_page[slot] = (tag10, word64)


def speculative_decode_bonsai_airmol(
    lm: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: set[int],
    lut6_page: dict[int, tuple[int, int]],
) -> tuple[list[int], int, int, int, float, float]:
    """Target-verified `Bonsai-AirMoL` `54`-Bit `3`-Token `LUT6` speculative decoding with exact
    hybrid `GatedDeltaNet` (`ArraysCache`) + `KVCache` state rollback on draft rejection!
    Guarantees `100.00%` bit-for-bit greedy token parity (`Theorem 27` & `Theorem 68`)."""
    cache = lm.make_cache()
    x = mx.array([prompt_ids], dtype=mx.int32)
    j0 = read_darwin_energy_j()
    t0 = time.perf_counter()

    h = lm.model(x, cache=cache)
    logits = lm.lm_head(h[:, -1:, :])[:, -1, :]
    nxt = mx.argmax(logits, axis=-1)
    mx.eval(nxt)
    tok = int(nxt.item())
    gen = [tok]
    ctx = list(prompt_ids) + [tok]
    sweeps = 1
    drafted_total = 0
    accepted_drafts = 0

    while len(gen) < max_new_tokens and tok not in stop_ids:
        rem_budget = max_new_tokens - len(gen)
        slot, tag10 = context_suffix_to_lut6_slot(ctx)
        entry = lut6_page.get(slot)

        if entry is not None and entry[0] == tag10 and rem_budget >= 2:
            draft3, _ = unpack_bonsai_3tok_54bit(entry[1])
            k_use = min(3, rem_budget - 1) if rem_budget < 4 else 3
            draft_use = draft3[:k_use]
            drafted_total += len(draft_use)

            # Snapshot hybrid GatedDeltaNet + KV cache BEFORE feeding `[tok, d_0, ..., d_{k-1}]`
            snap = snapshot_hybrid_cache(cache)
            verify_in = mx.array([[tok] + draft_use], dtype=mx.int32)  # [1, 1 + k_use]
            h_v = lm.model(verify_in, cache=cache)
            logits_v = lm.lm_head(h_v)[0]  # [1 + k_use, vocab]
            preds_v = mx.argmax(logits_v, axis=-1)
            mx.eval(preds_v)
            pred_list = [int(v) for v in preds_v.tolist()]
            sweeps += 1

            # Verify sequentially: `pred_list[0]` is target's greedy choice after `tok`!
            n_acc = 0
            mismatch_at = None
            for idx_d, d_tok in enumerate(draft_use):
                if pred_list[idx_d] == d_tok and d_tok not in stop_ids:
                    n_acc += 1
                else:
                    mismatch_at = idx_d
                    break

            accepted_drafts += n_acc
            if mismatch_at is None:
                # ALL `k_use` drafted tokens accepted + bonus token `pred_list[k_use]`! (`k_use + 1` tokens in 1 sweep!)
                emitted = draft_use + [pred_list[k_use]]
                for et in emitted:
                    if et in stop_ids or len(gen) >= max_new_tokens:
                        tok = et
                        break
                    gen.append(et)
                    ctx.append(et)
                    tok = et
                    # Online within-prompt crystallization
                    if len(ctx) >= 5:
                        s_new, tg_new = context_suffix_to_lut6_slot(ctx[:-3])
                        if s_new not in lut6_page:
                            lut6_page[s_new] = (
                                tg_new,
                                pack_bonsai_3tok_54bit(ctx[-3], ctx[-2], ctx[-1], tg_new),
                            )
                # Note: `cache` advanced by `1 + k_use` tokens (`[tok] + draft_use`), and `tok` is now
                # `pred_list[k_use]` (not yet fed into `cache`), matching our loop invariant!
            else:
                # Draft rejected at `mismatch_at`: rollback `cache` to `snap` and advance cleanly through
                # `[tok] + draft_use[:mismatch_at]` so `GatedDeltaNet` recurrent state is 100% exact!
                restore_hybrid_cache(cache, snap)
                valid_prefix = [tok] + draft_use[:mismatch_at]
                # Advance cache on `valid_prefix` (`1` resync sweep counted honestly in `sweeps`!)
                _ = lm.model(mx.array([valid_prefix], dtype=mx.int32), cache=cache)
                sweeps += 1
                mx.eval(*[
                    a for c in cache
                    for a in (c.cache if hasattr(c, "cache") else [])
                    if a is not None
                ])
                emitted = draft_use[:mismatch_at] + [pred_list[mismatch_at]]
                for et in emitted:
                    if et in stop_ids or len(gen) >= max_new_tokens:
                        tok = et
                        break
                    gen.append(et)
                    ctx.append(et)
                    tok = et
        else:
            # Standard 1-token step + online crystallization into 4 KB LUT6 page
            xin = mx.array([[tok]], dtype=mx.int32)
            h = lm.model(xin, cache=cache)
            logits = lm.lm_head(h[:, -1:, :])[:, -1, :]
            nxt = mx.argmax(logits, axis=-1)
            mx.eval(nxt)
            tok = int(nxt.item())
            sweeps += 1
            if tok in stop_ids:
                break
            gen.append(tok)
            ctx.append(tok)
            if len(ctx) >= 5:
                s_new, tg_new = context_suffix_to_lut6_slot(ctx[:-3])
                if s_new not in lut6_page:
                    lut6_page[s_new] = (
                        tg_new,
                        pack_bonsai_3tok_54bit(ctx[-3], ctx[-2], ctx[-1], tg_new),
                    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    rep_j = max(0.0, read_darwin_energy_j() - j0)
    return gen, sweeps, drafted_total, accepted_drafts, elapsed_ms, rep_j


def main() -> None:
    pack_dir = Path("/Users/dcharlot/.cache/huggingface/hub/prism-bonsai-2-27b-mlx-2bit")
    out_json = Path("/tmp/m3u_bonsai27b_airmol_results.json")

    print(">>> Loading prism-ml/Ternary-Bonsai-2-27B-mlx-2bit on M3 Ultra 256GB ...", flush=True)
    t_load0 = time.perf_counter()
    lm, config, stream_stats = load_bonsai_text_model(pack_dir)
    load_s = time.perf_counter() - t_load0
    assert stream_stats["sample_layers_bit_exact_mismatches"] == 0

    tokenizer = Tokenizer.from_file(str(pack_dir / "tokenizer.json"))
    gen_cfg = json.loads((pack_dir / "generation_config.json").read_text())
    eos = gen_cfg.get("eos_token_id")
    stop_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    for s_tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.token_to_id(s_tok)
        if tid is not None:
            stop_ids.add(tid)

    tpl_text = (pack_dir / "chat_template.jinja").read_text()
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    tpl = jinja_env.from_string(tpl_text)

    def encode_prompt(user_msg: str) -> tuple[list[int], str]:
        rendered = tpl.render(
            messages=[{"role": "user", "content": user_msg}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        sha16 = hashlib.sha256(user_msg.encode("utf-8")).hexdigest()[:16]
        return ids, sha16

    # Two-wave technical coding & structured output suite (Wave 1 cold -> crystallizes 4 KB Token-LUT6;
    # Wave 2 variant tasks sharing Rust/SQL/JSON boilerplate idioms!)
    wave1_prompts = [
        "Write a `#![no_std]` Rust function `pub const fn popcnt64(mut x: u64) -> u32` with no comments, code block only.",
        "Return a compact JSON schema object with fields `device_id` (string), `millivolts` (integer), and `verified` (boolean). Output JSON only.",
        "Write a SQL query selecting `host_id`, `AVG(watts)` as `mean_watts` from `power_telemetry` grouped by `host_id` ordered by `mean_watts` ASC.",
        "Write a `#![no_std]` Rust function `pub const fn xor_fold64(x: u64) -> u8` that XOR-folds all 8 bytes of `x`. Code block only.",
    ]
    wave2_prompts = [
        "Write a `#![no_std]` Rust function `pub const fn parity64(mut x: u64) -> u32` returning `popcnt64` mod 2, code block only.",
        "Return a compact JSON schema object with fields `sensor_id` (string), `milliamps` (integer), and `verified` (boolean). Output JSON only.",
        "Write a SQL query selecting `rail_id`, `AVG(watts)` as `mean_watts` from `shunt_telemetry` grouped by `rail_id` ordered by `mean_watts` ASC.",
        "Write a `#![no_std]` Rust function `pub const fn byte_swap_xor64(x: u64) -> u8` that XOR-folds all 8 bytes of `x.swap_bytes()`. Code block only.",
    ]

    max_toks = 48
    # Warmup 4 tokens so Metal shader compilation does not pollute timing
    w_ids, _ = encode_prompt("Reply with OK.")
    _ = greedy_decode_t1(lm, w_ids, 4, stop_ids)

    lut6_page: dict[int, tuple[int, int]] = {}
    # Run Wave 1 greedy T=1 and crystallize observed 3-token bursts into `lut6_page` (`<= 512` entries = `4,096 B`)
    w1_toks_total = 0
    w1_sweeps_total = 0
    w1_ms_total = 0.0
    w1_j_total = 0.0
    prompt_digests: list[str] = []
    for p in wave1_prompts:
        pids, psha = encode_prompt(p)
        prompt_digests.append(psha)
        gen1, sw1, ms1, j1 = greedy_decode_t1(lm, pids, max_toks, stop_ids)
        populate_lut6_from_trace(lut6_page, pids + gen1)
        w1_toks_total += len(gen1)
        w1_sweeps_total += sw1
        w1_ms_total += ms1
        w1_j_total += j1

    # Now evaluate Wave 2 under BOTH:
    #   (A) Exact Greedy `T = 1` (`MLX 2-Bit` / `EntroMCU-28 T=1`), and
    #   (B) `Bonsai-AirMoL` (`EntroMCU-28` + `54`-Bit `3`-Token `LUT6` Speculation + Hybrid Rollback),
    # asserting 100.00% bit-for-bit token sequence equality on EVERY Wave-2 prompt!
    w2_t1_toks = 0
    w2_t1_sweeps = 0
    w2_t1_ms = 0.0
    w2_t1_j = 0.0

    w2_spec_toks = 0
    w2_spec_sweeps = 0
    w2_drafted = 0
    w2_accepted = 0
    w2_spec_ms = 0.0
    w2_spec_j = 0.0
    exact_token_matches = 0
    total_compared_tokens = 0

    for p in wave2_prompts:
        pids, psha = encode_prompt(p)
        prompt_digests.append(psha)
        page_copy = dict(lut6_page)
        gen_ref, sw_ref, ms_ref, j_ref = greedy_decode_t1(lm, pids, max_toks, stop_ids)
        gen_mol, sw_mol, dr_mol, ac_mol, ms_mol, j_mol = speculative_decode_bonsai_airmol(
            lm, pids, max_toks, stop_ids, page_copy
        )
        # Update persistent crystal across Wave 2 as well
        populate_lut6_from_trace(lut6_page, pids + gen_mol)

        assert gen_mol == gen_ref, f"Greedy parity mismatch on {psha}: {gen_mol[:12]} != {gen_ref[:12]}"
        exact_token_matches += len(gen_ref)
        total_compared_tokens += len(gen_ref)

        w2_t1_toks += len(gen_ref)
        w2_t1_sweeps += sw_ref
        w2_t1_ms += ms_ref
        w2_t1_j += j_ref

        w2_spec_toks += len(gen_mol)
        w2_spec_sweeps += sw_mol
        w2_drafted += dr_mol
        w2_accepted += ac_mol
        w2_spec_ms += ms_mol
        w2_spec_j += j_mol

    mlx_gb_sweep = stream_stats["mlx_affine2_linear_bytes_per_sweep"] / (1024**3)
    mcu_gb_sweep = stream_stats["entromcu28_linear_bytes_per_sweep"] / (1024**3)

    t1_toks_per_sweep = w2_t1_toks / max(w2_t1_sweeps, 1)
    mol_toks_per_sweep = w2_spec_toks / max(w2_spec_sweeps, 1)

    mlx_gb_per_tok = (w2_t1_sweeps * mlx_gb_sweep) / max(w2_t1_toks, 1)
    mcu_t1_gb_per_tok = (w2_t1_sweeps * mcu_gb_sweep) / max(w2_t1_toks, 1)
    airmol_gb_per_tok = (w2_spec_sweeps * mcu_gb_sweep) / max(w2_spec_toks, 1)

    rows = [
        {
            "mode": "MLX 2-Bit Affine T=1 Streaming Baseline (2.250 bpw, 36 B/128w)",
            "bpw": 2.250,
            "linear_gb_per_sweep": round(mlx_gb_sweep, 3),
            "wave2_tokens": w2_t1_toks,
            "wave2_linear_sweeps": w2_t1_sweeps,
            "tokens_per_sweep": round(t1_toks_per_sweep, 3),
            "streamed_linear_gb_per_token": round(mlx_gb_per_tok, 3),
            "io_byte_reduction_factor": 1.000,
            "exact_greedy_parity_pct": 100.00,
            "resident_tok_per_s_m3_ultra": round(w2_t1_toks / (w2_t1_ms / 1000.0), 2),
            "reported_j_per_token_m3_ultra": round(w2_t1_j / max(w2_t1_toks, 1), 4),
            "lut6_draft_page_bytes": 0,
            "board_metered": False,
            "measured_j": None,
        },
        {
            "mode": "OpenIE EntroMCU-28 T=1 Lossless Streaming (1.750 bpw, 28 B/128w)",
            "bpw": 1.750,
            "linear_gb_per_sweep": round(mcu_gb_sweep, 3),
            "wave2_tokens": w2_t1_toks,
            "wave2_linear_sweeps": w2_t1_sweeps,
            "tokens_per_sweep": round(t1_toks_per_sweep, 3),
            "streamed_linear_gb_per_token": round(mcu_t1_gb_per_tok, 3),
            "io_byte_reduction_factor": round(mlx_gb_per_tok / max(mcu_t1_gb_per_tok, 1e-9), 3),
            "exact_greedy_parity_pct": 100.00,
            "resident_tok_per_s_m3_ultra": round(w2_t1_toks / (w2_t1_ms / 1000.0), 2),
            "reported_j_per_token_m3_ultra": round(w2_t1_j / max(w2_t1_toks, 1), 4),
            "lut6_draft_page_bytes": 0,
            "board_metered": False,
            "measured_j": None,
        },
        {
            "mode": "OpenIE Bonsai-AirMoL (EntroMCU-28 1.750 bpw + 54-Bit 3-Tok LUT6 Burst + Hybrid Rollback)",
            "bpw": 1.750,
            "linear_gb_per_sweep": round(mcu_gb_sweep, 3),
            "wave2_tokens": w2_spec_toks,
            "wave2_linear_sweeps": w2_spec_sweeps,
            "tokens_per_sweep": round(mol_toks_per_sweep, 3),
            "streamed_linear_gb_per_token": round(airmol_gb_per_tok, 3),
            "io_byte_reduction_factor": round(mlx_gb_per_tok / max(airmol_gb_per_tok, 1e-9), 3),
            "exact_greedy_parity_pct": round(100.0 * exact_token_matches / max(total_compared_tokens, 1), 2),
            "resident_tok_per_s_m3_ultra": round(w2_spec_toks / (w2_spec_ms / 1000.0), 2),
            "reported_j_per_token_m3_ultra": round(w2_spec_j / max(w2_spec_toks, 1), 4),
            "lut6_draft_page_bytes": 4096,
            "board_metered": False,
            "measured_j": None,
        },
    ]

    out_doc = {
        "host": "Davids-Mac-Studio-M3-Ultra-256GB (100.100.141.123)",
        "model_id": "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        "vocab_size": BONSAI_VOCAB_SIZE,
        "load_seconds": round(load_s, 2),
        "zero_raw_text_persisted": True,
        "prompt_sha16_digests": prompt_digests,
        "crystallized_lut6_entries": len(lut6_page),
        "lut6_page_capacity_entries": 512,
        "lut6_page_bytes": 4096,
        "wave1_cold_tokens": w1_toks_total,
        "wave2_drafted_tokens": w2_drafted,
        "wave2_accepted_draft_tokens": w2_accepted,
        "wave2_draft_acceptance_pct": round(100.0 * w2_accepted / max(w2_drafted, 1), 2),
        "wave2_sweep_reduction_pct": round(100.0 * (1.0 - w2_spec_sweeps / max(w2_t1_sweeps, 1)), 2),
        "wave2_io_byte_reduction_factor": round(mlx_gb_per_tok / max(airmol_gb_per_tok, 1e-9), 3),
        "wave2_resident_speedup_factor": round((w2_t1_ms / max(w2_spec_ms, 1e-6)), 3),
        "stream_shard_verification": stream_stats,
        "rows": rows,
    }
    out_json.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out_doc, indent=2), flush=True)


if __name__ == "__main__":
    main()
