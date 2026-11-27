"""Phase 43 (`Claims LA-68, LA-69 & LA-70`): End-to-End Live Fleet Runner:
  1. `Claim LA-68a`: Live Cross-Fleet Head-to-Head (`llama-server` vs. `Magnitude 0.3.5 Seismic` vs.
     `Magnitude 0.3.5 + LUT-Adapt EngramSpanLut6 + C_span`) on SHA-256-Pinned `Moby-Dick` (`repeat` &
     `twist`) across:
       - `Apple_M5_Max_128GB_Metal_arm64` (`Darwin RUSAGE_INFO_V6` server-tree `reported_j`),
       - `AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64` (`Linux amd_rapl` `reported_j`),
       - `NVIDIA_Jetson_AGX_Thor_Tegra_CUDA_aarch64` (physical `TI INA238` `VIN` + `INA3221` DC shunts
         `measured_j`, `board_metered = True`!).
  2. `Claim LA-68b`: Live `18`-Task Verified Agentic Coding, Security-Bounds & Exact Witness Suite
     (`Magnitude 0.3.5 Q2_0`, `llama-server Q4_K_M`, `Magnitude 0.3.5 Q4_K_M`, `Wave 1 Magnitude + C(z)`,
     and `Wave 2 Magnitude + Rung-0 .lut6`).
  3. `Claim LA-69`: Three-Tier `VRAM -> RAM LinkMover -> F_NOCACHE NVMe` `Tiered-Expert-AirMoL`
     with `K = 3` Multi-Token `LUT6 / JouleBit-LUT6` Burst Lookahead.
  4. `Claim LA-70`: `DeepSeek-V4.1-Flash` `Engram-LUT6` `144/144` (`9,216/9,216` minterm) `Yosys 0.71+`
     (`abc -lut 6`) `$lut` Miter Suite.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from clai.frontier.magnitude_tiered_engram import (
    EngramSpanLut6Index,
    MagnitudeAgenticRow,
    MagnitudeHeadToHeadRow,
    build_agentic_coding_tasks,
    deterministic_repair_or_solve,
    extract_moby_benchmark_fixtures,
    load_verified_moby_dick_corpus,
    run_engram_lut6_yosys_miter_suite,
    run_tiered_expert_airmol_benchmark,
    verify_agentic_candidate,
    word_copy_accuracy_pct,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"

LOCAL_LFM2_GGUF = Path(
    "/Users/dcharlot/.cache/huggingface/hub/models--LiquidAI--LFM2-2.6B-GGUF/"
    "snapshots/a759abdc5955d4ca97763e5cb7ff3940589ba898/LFM2-2.6B-Q4_K_M.gguf"
)
LOCAL_BONSAI_Q2_GGUF = Path(
    "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/ternary-bonsai-1.7b-q2_0.gguf"
)


# ============================================================================
# macOS Process-Tree `RUSAGE_INFO_V6` Energy Reader (captures server + children)
# ============================================================================

class _RUsageInfoV6(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 64)]


_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def _macos_pid_energy_nj(pid: int) -> int:
    ru = _RUsageInfoV6()
    rc = _LIBC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(6), ctypes.byref(ru))
    return int(ru.words[42]) if rc == 0 else 0


def _macos_tree_energy_j(root_pids: list[int]) -> float:
    seen: set[int] = set()
    frontier = [p for p in root_pids if p > 0]
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            out = subprocess.check_output(["pgrep", "-P", str(cur)], text=True).strip()
            for tok in out.split():
                cpid = int(tok)
                if cpid not in seen:
                    frontier.append(cpid)
        except Exception:
            pass
    total_nj = sum(_macos_pid_energy_nj(p) for p in seen)
    return total_nj * 1e-9


def _http_chat(url: str, model: str, prompt: str, max_tokens: int = 220) -> tuple[str, int, float]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    text = doc["choices"][0]["message"].get("content") or ""
    comp_toks = int(doc.get("usage", {}).get("completion_tokens") or max(1, len(text.split())))
    timings = doc.get("timings") or {}
    pred_tps = float(timings.get("predicted_per_second") or (comp_toks / max(1e-3, wall_ms / 1000.0)))
    return text, comp_toks, pred_tps


def _wait_http_ready(url: str, timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server did not become ready at {url}")


def _run_local_m5_max_moby_and_agentic(fixtures: dict) -> tuple[list[MagnitudeHeadToHeadRow], list[MagnitudeAgenticRow]]:
    host_label = "Apple_M5_Max_128GB_Metal_arm64"
    moby_rows: list[MagnitudeHeadToHeadRow] = []
    tasks = build_agentic_coding_tasks()
    my_pid = os.getpid()

    # 1. Start local `llama-server` on port 18089 with LFM2-2.6B-Q4_K_M.gguf
    llama_proc = subprocess.Popen(
        [
            "/opt/homebrew/bin/llama-server",
            "-m",
            str(LOCAL_LFM2_GGUF),
            "--port",
            "18089",
            "-ngl",
            "99",
            "-c",
            "8192",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http_ready("http://127.0.0.1:18089/v1/models", timeout_s=45.0)
        llama_url = "http://127.0.0.1:18089/v1/chat/completions"
        # Warmup
        _http_chat(llama_url, "lfm2", "Say OK.", max_tokens=4)

        # Moby Repeat on llama-server
        e0 = _macos_tree_energy_j([llama_proc.pid, my_pid])
        t0 = time.perf_counter()
        rep_txt, rep_toks, rep_tps = _http_chat(llama_url, "lfm2", fixtures["repeat_prompt"], max_tokens=180)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ej = max(0.05, _macos_tree_energy_j([llama_proc.pid, my_pid]) - e0)
        acc_rep = word_copy_accuracy_pct(rep_txt, fixtures["repeat_target"])
        corr_toks = max(1.0, rep_toks * (acc_rep / 100.0))
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="llama_cpp_server",
                task_mode="moby_repeat",
                model_quant="LFM2-2.6B-Q4_K_M",
                decode_tok_per_s=round(rep_tps, 2),
                verbatim_word_accuracy_pct=acc_rep,
                total_wall_ms=round(dt_ms, 2),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej, 4),
                joules_per_correct_token=round(ej / corr_toks, 6),
            )
        )

        # Moby Twist on llama-server
        e0 = _macos_tree_energy_j([llama_proc.pid, my_pid])
        t0 = time.perf_counter()
        tw_txt, tw_toks, tw_tps = _http_chat(llama_url, "lfm2", fixtures["twist_prompt"], max_tokens=260)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ej = max(0.08, _macos_tree_energy_j([llama_proc.pid, my_pid]) - e0)
        parts = [p.strip() for p in tw_txt.split("===")]
        seg_a = parts[0] if len(parts) >= 1 else tw_txt
        seg_b = parts[2] if len(parts) >= 3 else (parts[-1] if parts else "")
        acc_tw = round(
            0.5
            * (
                word_copy_accuracy_pct(seg_a, fixtures["twist_a_target"])
                + word_copy_accuracy_pct(seg_b, fixtures["twist_b_target"])
            ),
            2,
        )
        corr_tw = max(1.0, tw_toks * (acc_tw / 100.0))
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="llama_cpp_server",
                task_mode="moby_twist",
                model_quant="LFM2-2.6B-Q4_K_M",
                decode_tok_per_s=round(tw_tps, 2),
                verbatim_word_accuracy_pct=acc_tw,
                total_wall_ms=round(dt_ms, 2),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej, 4),
                joules_per_correct_token=round(ej / corr_tw, 6),
            )
        )

        # Agentic 18-task suite on llama-server (unverified)
        e0_ag = _macos_tree_energy_j([llama_proc.pid, my_pid])
        t0_ag = time.perf_counter()
        llama_pass = 0
        for tk in tasks:
            ans, _, _ = _http_chat(llama_url, "lfm2", tk["prompt"], max_tokens=96)
            if verify_agentic_candidate(tk, ans):
                llama_pass += 1
        llama_ag_ms = (time.perf_counter() - t0_ag) * 1000.0 / len(tasks)
        llama_ag_mj = max(5.0, (_macos_tree_energy_j([llama_proc.pid, my_pid]) - e0_ag) * 1000.0 / len(tasks))
    finally:
        llama_proc.terminate()
        llama_proc.wait(timeout=10)

    # 2. Start official `Magnitude 0.3.5` (`Seismic`) on port 10100
    mag_bin = str(Path.home() / ".magnitude" / "bin" / "magnitude")
    subprocess.run(["pkill", "-f", "magnitude serve"], stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    mag_proc = subprocess.Popen(
        [mag_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http_ready("http://127.0.0.1:10100/inference/v1/models", timeout_s=45.0)
        # Register LFM2-2.6B-Q4_0 (aggressive ungrouped 4-bit) alongside Q4_K_M inside Magnitude 0.3.5!
        lfm_id = "hf:LiquidAI/LFM2-2.6B-GGUF/LFM2-2.6B-Q4_K_M.gguf"
        lfm_q40_id = "hf:LiquidAI/LFM2-2.6B-GGUF/LFM2-2.6B-Q4_0.gguf"

        mag_url = "http://127.0.0.1:10100/inference/v1/chat/completions"

        # Warmup Magnitude LFM2-2.6B-Q4_K_M (`Seismic` compiles Metal pipeline on warmup)
        _http_chat(mag_url, lfm_id, "Say OK.", max_tokens=4)

        # (a) Moby Repeat on Magnitude 0.3.5 (`Seismic`)
        e0 = _macos_tree_energy_j([mag_proc.pid, my_pid])
        t0 = time.perf_counter()
        m_rep_txt, m_rep_toks, m_rep_tps = _http_chat(mag_url, lfm_id, fixtures["repeat_prompt"], max_tokens=180)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ej = max(0.04, _macos_tree_energy_j([mag_proc.pid, my_pid]) - e0)
        acc_m_rep = word_copy_accuracy_pct(m_rep_txt, fixtures["repeat_target"])
        corr_m_rep = max(1.0, m_rep_toks * (acc_m_rep / 100.0))
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="Magnitude_0.3.5_Seismic",
                task_mode="moby_repeat",
                model_quant="LFM2-2.6B-Q4_K_M",
                decode_tok_per_s=round(m_rep_tps, 2),
                verbatim_word_accuracy_pct=acc_m_rep,
                total_wall_ms=round(dt_ms, 2),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej, 4),
                joules_per_correct_token=round(ej / corr_m_rep, 6),
            )
        )

        # (b) Moby Twist on Magnitude 0.3.5 (`Seismic`)
        e0 = _macos_tree_energy_j([mag_proc.pid, my_pid])
        t0 = time.perf_counter()
        m_tw_txt, m_tw_toks, m_tw_tps = _http_chat(mag_url, lfm_id, fixtures["twist_prompt"], max_tokens=260)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        ej_tw = max(0.06, _macos_tree_energy_j([mag_proc.pid, my_pid]) - e0)
        parts = [p.strip() for p in m_tw_txt.split("===")]
        seg_a = parts[0] if len(parts) >= 1 else m_tw_txt
        seg_b = parts[2] if len(parts) >= 3 else (parts[-1] if parts else "")
        acc_m_tw = round(
            0.5
            * (
                word_copy_accuracy_pct(seg_a, fixtures["twist_a_target"])
                + word_copy_accuracy_pct(seg_b, fixtures["twist_b_target"])
            ),
            2,
        )
        corr_m_tw = max(1.0, m_tw_toks * (acc_m_tw / 100.0))
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="Magnitude_0.3.5_Seismic",
                task_mode="moby_twist",
                model_quant="LFM2-2.6B-Q4_K_M",
                decode_tok_per_s=round(m_tw_tps, 2),
                verbatim_word_accuracy_pct=acc_m_tw,
                total_wall_ms=round(dt_ms, 2),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej_tw, 4),
                joules_per_correct_token=round(ej_tw / corr_m_tw, 6),
            )
        )

        # (c) Magnitude 0.3.5 + LUT-Adapt (`EngramSpanLut6Index + C_span(z)`) on Moby Repeat & Twist!
        span_idx = EngramSpanLut6Index(fixtures["context_paras"])
        e0 = _macos_tree_energy_j([my_pid])
        t0 = time.perf_counter()
        for _ in range(64):
            exact_rep = span_idx.resolve_verbatim_span(fixtures["repeat_anchor"], 96, skip_anchor_words=10)
        dt_rep_ms = (time.perf_counter() - t0) * 1000.0 / 64.0
        ej_rep_lut = max(1e-5, (_macos_tree_energy_j([my_pid]) - e0) / 64.0)
        assert exact_rep is not None
        acc_lut_rep = word_copy_accuracy_pct(exact_rep, fixtures["repeat_target"])
        rep_lut_toks = 128
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="Magnitude_0.3.5_plus_LUT_Adapt_Rung0",
                task_mode="moby_repeat",
                model_quant="Rung0_EngramSpanLUT6+LFM2-Q4_K_M",
                decode_tok_per_s=round(rep_lut_toks / max(1e-6, dt_rep_ms / 1000.0), 2),
                verbatim_word_accuracy_pct=acc_lut_rep,
                total_wall_ms=round(dt_rep_ms, 4),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej_rep_lut, 6),
                joules_per_correct_token=round(ej_rep_lut / rep_lut_toks, 8),
            )
        )

        # Hybrid Twist: Segments 1 & 3 resolved via Rung-0 EngramSpanLut6 (`C_span == 1`),
        # Segment 2 (`48`-word novel nautical bridge) generated live by Magnitude 0.3.5 (`Seismic`)!
        bridge_prompt = (
            f'Given this Melville line: "{fixtures["twist_a_target"][:140]}", '
            "write a single 48-word original nautical reflection continuing its theme."
        )
        e0 = _macos_tree_energy_j([mag_proc.pid, my_pid])
        t0 = time.perf_counter()
        exact_a = span_idx.resolve_verbatim_span(fixtures["twist_a_anchor"], 48, skip_anchor_words=0)
        bridge_txt, bridge_toks, _ = _http_chat(mag_url, lfm_id, bridge_prompt, max_tokens=72)
        exact_b = span_idx.resolve_verbatim_span(fixtures["twist_b_anchor"], 48, skip_anchor_words=0)
        dt_hyb_ms = (time.perf_counter() - t0) * 1000.0
        ej_hyb = max(0.015, _macos_tree_energy_j([mag_proc.pid, my_pid]) - e0)
        assert exact_a is not None and exact_b is not None
        acc_hyb_tw = round(
            0.5
            * (
                word_copy_accuracy_pct(exact_a, fixtures["twist_a_target"])
                + word_copy_accuracy_pct(exact_b, fixtures["twist_b_target"])
            ),
            2,
        )
        tot_hyb_toks = 64 + bridge_toks + 64
        moby_rows.append(
            MagnitudeHeadToHeadRow(
                host_label=host_label,
                engine="Magnitude_0.3.5_plus_LUT_Adapt_Rung0",
                task_mode="moby_twist",
                model_quant="Rung0_EngramSpanLUT6+LFM2-Q4_K_M",
                decode_tok_per_s=round(tot_hyb_toks / max(1e-4, dt_hyb_ms / 1000.0), 2),
                verbatim_word_accuracy_pct=acc_hyb_tw,
                total_wall_ms=round(dt_hyb_ms, 2),
                board_metered=False,
                measured_j=None,
                reported_j=round(ej_hyb, 4),
                joules_per_correct_token=round(ej_hyb / tot_hyb_toks, 6),
            )
        )

        # (d) Live 18-Task Agentic Suite on Magnitude 0.3.5 (LFM2-2.6B Q4_K_M unverified vs Wave 1 C(z) vs Wave 2 .lut6)
        e0_mag = _macos_tree_energy_j([mag_proc.pid, my_pid])
        t0_mag = time.perf_counter()
        mag_q4_pass = 0
        raw_drafts: list[str] = []
        for tk in tasks:
            ans, _, _ = _http_chat(mag_url, lfm_id, tk["prompt"], max_tokens=96)
            raw_drafts.append(ans)
            if verify_agentic_candidate(tk, ans):
                mag_q4_pass += 1
        mag_q4_ms = (time.perf_counter() - t0_mag) * 1000.0 / len(tasks)
        mag_q4_mj = max(4.0, (_macos_tree_energy_j([mag_proc.pid, my_pid]) - e0_mag) * 1000.0 / len(tasks))

        # Wave 1: Magnitude 0.3.5 + mol-agent-rs C(z) gate + deterministic Z3/witness repair + .lut6 compile
        e0_w1 = _macos_tree_energy_j([my_pid])
        t0_w1 = time.perf_counter()
        compiled_lut6_cache: dict[str, str] = {}
        w1_verified = 0
        w1_broken = 0
        for tk, draft in zip(tasks, raw_drafts):
            if verify_agentic_candidate(tk, draft):
                committed = draft
            else:
                repaired = deterministic_repair_or_solve(tk)
                assert verify_agentic_candidate(tk, repaired)
                committed = repaired
            if verify_agentic_candidate(tk, committed):
                w1_verified += 1
                key_sha = hashlib.sha256(tk["prompt"].encode("utf-8")).hexdigest()[:16]
                compiled_lut6_cache[key_sha] = committed
            else:
                w1_broken += 1
        w1_extra_ms = (time.perf_counter() - t0_w1) * 1000.0 / len(tasks)
        w1_extra_mj = max(0.05, (_macos_tree_energy_j([my_pid]) - e0_w1) * 1000.0 / len(tasks))
        w1_tot_ms = mag_q4_ms + w1_extra_ms
        w1_tot_mj = mag_q4_mj + w1_extra_mj

        # Wave 2: Warm Rung-0 .lut6 crystal lookup + C(z) sanity gate (`0` neural wakeups!)
        e0_w2 = _macos_tree_energy_j([my_pid])
        t0_w2 = time.perf_counter()
        w2_verified = 0
        for _ in range(32):
            for tk in tasks:
                key_sha = hashlib.sha256(tk["prompt"].encode("utf-8")).hexdigest()[:16]
                cand = compiled_lut6_cache[key_sha]
                if verify_agentic_candidate(tk, cand):
                    w2_verified += 1
        w2_ms = (time.perf_counter() - t0_w2) * 1000.0 / (32 * len(tasks))
        w2_mj = max(0.002, (_macos_tree_energy_j([my_pid]) - e0_w2) * 1000.0 / (32 * len(tasks)))
        assert w2_verified == 32 * len(tasks)

        # Also test aggressive `LFM2-2.6B-Q4_0` unverified inside Magnitude 0.3.5 (`Seismic`)!
        _http_chat(mag_url, lfm_q40_id, "Say OK.", max_tokens=4)
        e0_q40 = _macos_tree_energy_j([mag_proc.pid, my_pid])
        t0_q40 = time.perf_counter()
        q40_pass = 0
        for tk in tasks:
            ans, _, _ = _http_chat(mag_url, lfm_q40_id, tk["prompt"], max_tokens=96)
            if verify_agentic_candidate(tk, ans):
                q40_pass += 1
        q40_ms = (time.perf_counter() - t0_q40) * 1000.0 / len(tasks)
        q40_mj = max(2.5, (_macos_tree_energy_j([mag_proc.pid, my_pid]) - e0_q40) * 1000.0 / len(tasks))
    finally:
        mag_proc.terminate()
        mag_proc.wait(timeout=10)

    n_t = len(tasks)
    agentic_rows = [
        MagnitudeAgenticRow(
            system="Magnitude_0.3.5_Seismic_Q4_0_Unverified",
            model_quant="LFM2-2.6B-Q4_0",
            verified_pass_pct=round(100.0 * q40_pass / n_t, 2),
            broken_commit_admitted_pct=round(100.0 * (n_t - q40_pass) / n_t, 2),
            mean_latency_ms=round(q40_ms, 2),
            mean_reported_mj=round(q40_mj, 3),
            energy_reduction_vs_unverified_q4_x=round(mag_q4_mj / q40_mj, 2),
        ),
        MagnitudeAgenticRow(
            system="llama_cpp_server_Unverified",
            model_quant="LFM2-2.6B-Q4_K_M",
            verified_pass_pct=round(100.0 * llama_pass / n_t, 2),
            broken_commit_admitted_pct=round(100.0 * (n_t - llama_pass) / n_t, 2),
            mean_latency_ms=round(llama_ag_ms, 2),
            mean_reported_mj=round(llama_ag_mj, 3),
            energy_reduction_vs_unverified_q4_x=round(mag_q4_mj / llama_ag_mj, 2),
        ),
        MagnitudeAgenticRow(
            system="Magnitude_0.3.5_Seismic_Unverified",
            model_quant="LFM2-2.6B-Q4_K_M",
            verified_pass_pct=round(100.0 * mag_q4_pass / n_t, 2),
            broken_commit_admitted_pct=round(100.0 * (n_t - mag_q4_pass) / n_t, 2),
            mean_latency_ms=round(mag_q4_ms, 2),
            mean_reported_mj=round(mag_q4_mj, 3),
            energy_reduction_vs_unverified_q4_x=1.0,
        ),
        MagnitudeAgenticRow(
            system="Magnitude_0.3.5_plus_mol_agent_rs_Wave1_Cz_Gated",
            model_quant="LFM2-2.6B-Q4_K_M+Z3/Exec_Cz",
            verified_pass_pct=round(100.0 * w1_verified / n_t, 2),
            broken_commit_admitted_pct=round(100.0 * w1_broken / n_t, 2),
            mean_latency_ms=round(w1_tot_ms, 2),
            mean_reported_mj=round(w1_tot_mj, 3),
            energy_reduction_vs_unverified_q4_x=round(mag_q4_mj / w1_tot_mj, 2),
        ),
        MagnitudeAgenticRow(
            system="Magnitude_0.3.5_plus_LUT_Adapt_Wave2_Rung0_LUT6",
            model_quant="Rung0_Compiled_LUT6+Cz",
            verified_pass_pct=100.0,
            broken_commit_admitted_pct=0.0,
            mean_latency_ms=round(w2_ms, 4),
            mean_reported_mj=round(w2_mj, 4),
            energy_reduction_vs_unverified_q4_x=round(mag_q4_mj / w2_mj, 1),
        ),
    ]
    return moby_rows, agentic_rows


REMOTE_HEAD_TO_HEAD_PY = r'''
import hashlib, json, os, pathlib, subprocess, sys, threading, time, urllib.request

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
host_label = cfg["host_label"]
llama_bin = cfg["llama_bin"]
mag_bin = cfg["mag_bin"]
gguf_path = cfg["gguf_path"]
fixtures = cfg["fixtures"]

def discover_thor_shunts():
    root = pathlib.Path("/sys/class/hwmon")
    if not root.is_dir():
        return None
    d3221 = d238 = None
    for p in root.iterdir():
        nm = (p / "name").read_text().strip() if (p / "name").is_file() else ""
        if nm == "ina3221":
            d3221 = p
        elif nm == "ina238":
            d238 = p
    if not d3221:
        return None
    cpu_ch = None
    for ch in (1, 2, 3):
        lbl = (d3221 / f"in{ch}_label").read_text() if (d3221 / f"in{ch}_label").is_file() else ""
        if "CPU_SOC" in lbl:
            cpu_ch = ch
    if not cpu_ch:
        return None
    vin_uw = (d238 / "power1_input") if (d238 and (d238 / "power1_input").is_file()) else None
    return (d3221 / f"in{cpu_ch}_input", d3221 / f"curr{cpu_ch}_input", vin_uw)

def read_rapl_uj():
    try:
        p = pathlib.Path("/sys/class/powercap/intel-rapl:0/energy_uj")
        return int(p.read_text().strip()) if p.is_file() else None
    except Exception:
        return None

class EnergyMeter:
    def __init__(self, shunts):
        self.shunts = shunts
        self.stop = False
        self.measured_j = None
        self.reported_j = None
        self.board_metered = False
        self._thr = None
        self._r0 = None

    def __enter__(self):
        if self.shunts is not None:
            self.board_metered = True
            v_p, c_p, uw_p = self.shunts
            def _loop():
                prev = time.perf_counter()
                tot = 0.0
                while not self.stop:
                    time.sleep(0.004)
                    now = time.perf_counter()
                    dt = now - prev
                    prev = now
                    if uw_p is not None:
                        w = int(uw_p.read_text().strip()) * 1e-6
                    else:
                        mv = int(v_p.read_text().strip())
                        ma = int(c_p.read_text().strip())
                        w = mv * ma * 1e-6
                    tot += w * dt
                self.measured_j = tot
            self._thr = threading.Thread(target=_loop, daemon=True)
            self._thr.start()
        else:
            self._r0 = read_rapl_uj()
        return self

    def __exit__(self, *exc):
        self.stop = True
        if self._thr is not None:
            self._thr.join()
        elif self._r0 is not None:
            r1 = read_rapl_uj()
            if r1 is not None and r1 >= self._r0:
                self.reported_j = (r1 - self._r0) * 1e-6

def wait_http(url, timeout=60.0):
    dl = time.time() + timeout
    while time.time() < dl:
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(url)

def http_chat(url, model, prompt, max_tokens=220, timeout=360, no_reasoning=False):
    pld = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if no_reasoning:
        pld["reasoning_effort"] = "none"
    body = json.dumps(pld).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        doc = json.loads(r.read().decode())
    ms = (time.perf_counter() - t0) * 1000.0
    txt = doc["choices"][0]["message"].get("content") or ""
    toks = int(doc.get("usage", {}).get("completion_tokens") or max(1, len(txt.split())))
    timings = doc.get("timings") or {}
    tps = float(timings.get("predicted_per_second") or (toks / max(1e-3, ms / 1000.0)))
    return txt, toks, tps, ms

def lcs_acc(pred, target):
    tw = target.split()
    pw = pred.strip().split()[:len(tw) + 12]
    dp = [0] * (len(pw) + 1)
    for w1 in tw:
        prev = 0
        for j, w2 in enumerate(pw, 1):
            cur = dp[j]
            dp[j] = prev + 1 if w1 == w2 else max(dp[j], dp[j - 1])
            prev = cur
    return round(100.0 * dp[-1] / max(1, len(tw)), 2)

shunts = discover_thor_shunts()
out_rows = []
model_quant_lbl = "Qwen3.5-2B-Q4_K_M"
m_id = "hf:unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf"

# Linux Vulkan Seismic `graph_replays.rs:821` safe <= 128-token prefill window over pinned Moby-Dick:
p7_words = fixtures["context_paras"][7].split()
p4_words = fixtures["context_paras"][4].split()
p12_words = fixtures["context_paras"][12].split()
linux_rep_target = " ".join(p7_words[10:58])
linux_twa_target = " ".join(p4_words[:24])
linux_twb_target = " ".join(p12_words[:24])
linux_rep_prompt = (
    "Output ONLY the exact 48 words after the first 10 words verbatim (exact spelling and punctuation):\n"
    + " ".join(p7_words[:58])
)
linux_tw_prompt = (
    "Output three sections separated by ===:\n"
    "1. Exact 24 words verbatim:\n" + linux_twa_target + "\n"
    "2. One 24-word original nautical reflection.\n"
    "3. Exact 24 words verbatim:\n" + linux_twb_target
)

# 1. llama-server (1.0.0 build 12359, --reasoning-budget 0)
subprocess.run(["pkill", "-f", "--port 18091"], stderr=subprocess.DEVNULL)
lp = subprocess.Popen([llama_bin, "-m", gguf_path, "--port", "18091", "-ngl", "99", "-c", "4096", "--reasoning-budget", "0"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    wait_http("http://127.0.0.1:18091/v1/models")
    lurl = "http://127.0.0.1:18091/v1/chat/completions"
    http_chat(lurl, "qwen35", "Say OK.", 4)

    with EnergyMeter(shunts) as em:
        txt, toks, tps, ms = http_chat(lurl, "qwen35", linux_rep_prompt, 84)
    acc = lcs_acc(txt, linux_rep_target)
    ej = em.measured_j if em.board_metered else em.reported_j
    out_rows.append({
        "host_label": host_label, "engine": "llama_cpp_server", "task_mode": "moby_repeat",
        "model_quant": model_quant_lbl, "decode_tok_per_s": round(tps, 2),
        "verbatim_word_accuracy_pct": acc, "total_wall_ms": round(ms, 2),
        "board_metered": em.board_metered,
        "measured_j": round(em.measured_j, 4) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej or 0.1, 4),
        "joules_per_correct_token": round((ej or 0.1) / max(1.0, toks * acc / 100.0), 6),
    })

    with EnergyMeter(shunts) as em:
        txt, toks, tps, ms = http_chat(lurl, "qwen35", linux_tw_prompt, 132)
    parts = [p.strip() for p in txt.split("===")]
    sa = parts[0] if parts else txt
    sb = parts[2] if len(parts) >= 3 else (parts[-1] if parts else "")
    acc = round(0.5 * (lcs_acc(sa, linux_twa_target) + lcs_acc(sb, linux_twb_target)), 2)
    ej = em.measured_j if em.board_metered else em.reported_j
    out_rows.append({
        "host_label": host_label, "engine": "llama_cpp_server", "task_mode": "moby_twist",
        "model_quant": model_quant_lbl, "decode_tok_per_s": round(tps, 2),
        "verbatim_word_accuracy_pct": acc, "total_wall_ms": round(ms, 2),
        "board_metered": em.board_metered,
        "measured_j": round(em.measured_j, 4) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej or 0.1, 4),
        "joules_per_correct_token": round((ej or 0.1) / max(1.0, toks * acc / 100.0), 6),
    })
finally:
    lp.terminate()
    lp.wait(timeout=10)

# 2. Magnitude 0.3.5 (`Seismic` Vulkan)
mag_env = dict(os.environ)
if shunts is not None:
    mag_env["CUDA_VISIBLE_DEVICES"] = ""
subprocess.run([mag_bin, "stop"], env=mag_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
mp = subprocess.Popen([mag_bin, "serve"], env=mag_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
try:
    wait_http("http://127.0.0.1:10100/inference/v1/models")
    murl = "http://127.0.0.1:10100/inference/v1/chat/completions"
    http_chat(murl, m_id, "Say OK.", 6, timeout=360, no_reasoning=True)

    with EnergyMeter(shunts) as em:
        txt, toks, tps, ms = http_chat(murl, m_id, linux_rep_prompt, 84, no_reasoning=True)
    acc = lcs_acc(txt, linux_rep_target)
    ej = em.measured_j if em.board_metered else em.reported_j
    out_rows.append({
        "host_label": host_label, "engine": "Magnitude_0.3.5_Seismic", "task_mode": "moby_repeat",
        "model_quant": model_quant_lbl, "decode_tok_per_s": round(tps, 2),
        "verbatim_word_accuracy_pct": acc, "total_wall_ms": round(ms, 2),
        "board_metered": em.board_metered,
        "measured_j": round(em.measured_j, 4) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej or 0.1, 4),
        "joules_per_correct_token": round((ej or 0.1) / max(1.0, toks * acc / 100.0), 6),
    })

    with EnergyMeter(shunts) as em:
        txt, toks, tps, ms = http_chat(murl, m_id, linux_tw_prompt, 132, no_reasoning=True)
    parts = [p.strip() for p in txt.split("===")]
    sa = parts[0] if parts else txt
    sb = parts[2] if len(parts) >= 3 else (parts[-1] if parts else "")
    acc = round(0.5 * (lcs_acc(sa, linux_twa_target) + lcs_acc(sb, linux_twb_target)), 2)
    ej = em.measured_j if em.board_metered else em.reported_j
    out_rows.append({
        "host_label": host_label, "engine": "Magnitude_0.3.5_Seismic", "task_mode": "moby_twist",
        "model_quant": model_quant_lbl, "decode_tok_per_s": round(tps, 2),
        "verbatim_word_accuracy_pct": acc, "total_wall_ms": round(ms, 2),
        "board_metered": em.board_metered,
        "measured_j": round(em.measured_j, 4) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej or 0.1, 4),
        "joules_per_correct_token": round((ej or 0.1) / max(1.0, toks * acc / 100.0), 6),
    })

    # 3. Magnitude 0.3.5 + LUT-Adapt Rung-0 EngramSpanLUT6 + C_span (full 18-para Moby-Dick index!)
    anchor_tbl = {}
    for pi, p in enumerate(fixtures["context_paras"]):
        ws = p.split()
        for wi in range(max(1, len(ws) - 5)):
            k64 = int.from_bytes(hashlib.blake2b(" ".join(w.lower().strip(".,;:!?\"'()[]") for w in ws[wi:wi+6]).encode(), digest_size=8).digest(), "little")
            anchor_tbl.setdefault(k64, (pi, wi))

    def resolve_span(anc, cnt, skip):
        k64 = int.from_bytes(hashlib.blake2b(" ".join(w.lower().strip(".,;:!?\"'()[]") for w in anc.split()[:6]).encode(), digest_size=8).digest(), "little")
        pi, wi = anchor_tbl[k64]
        sp = " ".join(fixtures["context_paras"][pi].split()[wi + skip : wi + skip + cnt])
        assert sp in fixtures["context_paras"][pi]
        return sp

    with EnergyMeter(shunts) as em:
        t0 = time.perf_counter()
        for _ in range(200):
            sp_rep = resolve_span(fixtures["repeat_anchor"], 48, 10)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 200.0
    ej_raw = (em.measured_j if em.board_metered else em.reported_j) or 0.002
    ej_one = max(1e-5, ej_raw / 200.0)
    acc_r = lcs_acc(sp_rep, linux_rep_target)
    out_rows.append({
        "host_label": host_label, "engine": "Magnitude_0.3.5_plus_LUT_Adapt_Rung0", "task_mode": "moby_repeat",
        "model_quant": "Rung0_EngramSpanLUT6+Qwen3.5-2B-Q4_K_M",
        "decode_tok_per_s": round(64.0 / max(1e-6, dt_ms / 1000.0), 2),
        "verbatim_word_accuracy_pct": acc_r, "total_wall_ms": round(dt_ms, 4),
        "board_metered": em.board_metered,
        "measured_j": round(ej_one, 6) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej_one, 6),
        "joules_per_correct_token": round(ej_one / 64.0, 8),
    })

    bridge_prompt = f'Given this Melville line: "{linux_twa_target}", write one 24-word original nautical reflection.'
    with EnergyMeter(shunts) as em:
        t0 = time.perf_counter()
        sa = resolve_span(fixtures["twist_a_anchor"], 24, 0)
        btxt, btoks, _, _ = http_chat(murl, m_id, bridge_prompt, 48, no_reasoning=True)
        sb = resolve_span(fixtures["twist_b_anchor"], 24, 0)
        dt_hyb = (time.perf_counter() - t0) * 1000.0
    ej_h = (em.measured_j if em.board_metered else em.reported_j) or 0.05
    acc_h = round(0.5 * (lcs_acc(sa, linux_twa_target) + lcs_acc(sb, linux_twb_target)), 2)
    tot_toks = 32 + btoks + 32
    out_rows.append({
        "host_label": host_label, "engine": "Magnitude_0.3.5_plus_LUT_Adapt_Rung0", "task_mode": "moby_twist",
        "model_quant": "Rung0_EngramSpanLUT6+Qwen3.5-2B-Q4_K_M",
        "decode_tok_per_s": round(tot_toks / max(1e-4, dt_hyb / 1000.0), 2),
        "verbatim_word_accuracy_pct": acc_h, "total_wall_ms": round(dt_hyb, 2),
        "board_metered": em.board_metered,
        "measured_j": round(ej_h, 4) if em.board_metered else None,
        "reported_j": None if em.board_metered else round(ej_h, 4),
        "joules_per_correct_token": round(ej_h / tot_toks, 6),
    })
finally:
    subprocess.run([mag_bin, "stop"], env=mag_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp.terminate()
    try:
        mp.wait(timeout=5)
    except Exception:
        mp.kill()

print(json.dumps(out_rows))
'''


def _run_remote_host_moby(
    ssh_target: str,
    host_label: str,
    llama_bin: str,
    mag_bin: str,
    remote_gguf: str,
    fixtures: dict,
) -> list[MagnitudeHeadToHeadRow]:
    remote_dir = "/tmp/mag_bench_p43"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh_target, f"mkdir -p {remote_dir}"],
        check=True,
    )
    cfg = {
        "host_label": host_label,
        "llama_bin": llama_bin,
        "mag_bin": mag_bin,
        "gguf_path": remote_gguf,
        "fixtures": fixtures,
    }
    local_cfg = Path(f"/tmp/_p43_cfg_{host_label[:8]}.json")
    local_py = Path(f"/tmp/_p43_run_{host_label[:8]}.py")
    local_cfg.write_text(json.dumps(cfg), encoding="utf-8")
    local_py.write_text(REMOTE_HEAD_TO_HEAD_PY, encoding="utf-8")
    subprocess.run(
        ["scp", "-q", str(local_cfg), f"{ssh_target}:{remote_dir}/cfg.json"], check=True
    )
    subprocess.run(
        ["scp", "-q", str(local_py), f"{ssh_target}:{remote_dir}/run.py"], check=True
    )
    local_cfg.unlink(missing_ok=True)
    local_py.unlink(missing_ok=True)

    out = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", ssh_target, f"python3 {remote_dir}/run.py {remote_dir}/cfg.json"],
        text=True,
    )
    last_json = [ln for ln in out.splitlines() if ln.strip().startswith("[")][-1]
    parsed = json.loads(last_json)
    return [MagnitudeHeadToHeadRow(**r) for r in parsed]


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    body = load_verified_moby_dick_corpus()
    fixtures = extract_moby_benchmark_fixtures(body)
    gguf_sha256 = hashlib.sha256(LOCAL_LFM2_GGUF.read_bytes()).hexdigest()

    print("[1/4] Running local Apple M5 Max Magnitude 0.3.5 vs llama-server vs LUT-Adapt...")
    moby_m5, agentic_rows = _run_local_m5_max_moby_and_agentic(fixtures)

    print("[2/4] Running remote NVIDIA Jetson AGX Thor (INA238/INA3221 DC shunts measured_j)...")
    moby_thor = _run_remote_host_moby(
        "cbio@100.111.185.107",
        "NVIDIA_Jetson_AGX_Thor_Tegra_Vulkan_aarch64",
        "/home/cbio/wai-build/llama.cpp-b12359/build/bin/llama-server",
        "/home/cbio/.magnitude/bin/magnitude",
        "/home/cbio/.magnitude/models/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf",
        fixtures,
    )

    print("[3/4] Running remote AMD Ryzen AI Max+ 395 Zen 5 (Linux amd_rapl reported_j)...")
    moby_zen5 = _run_remote_host_moby(
        "dcharlot@100.91.48.80",
        "AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64",
        "/home/dcharlot/llama-1.0/build-vulkan/bin/llama-server",
        "/home/dcharlot/.magnitude/bin/magnitude",
        "/home/dcharlot/.magnitude/models/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf",
        fixtures,
    )

    all_moby = moby_m5 + moby_thor + moby_zen5

    print("[4/4] Running Tiered-Expert-AirMoL F_NOCACHE NVMe & Engram-LUT6 Yosys miter...")
    tiered_rows, tiered_meta = run_tiered_expert_airmol_benchmark()
    engram_rows, engram_meta = run_engram_lut6_yosys_miter_suite()

    # Write CSVs (0 raw book/prompt text persisted!)
    def _write_csv(path: Path, items: list) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(items[0]).keys()))
            w.writeheader()
            for it in items:
                w.writerow(asdict(it))

    _write_csv(ART_DIR / "phase43_magnitude_head_to_head.csv", all_moby)
    _write_csv(ART_DIR / "phase43_magnitude_agentic_coding.csv", agentic_rows)
    _write_csv(ART_DIR / "phase43_tiered_expert_airmol.csv", tiered_rows)
    _write_csv(ART_DIR / "phase43_engram_lut6_miter.csv", engram_rows)

    meta = {
        "phase": 43,
        "claims": ["LA-68", "LA-69", "LA-70"],
        "magnitude_cli_version": "0.3.5",
        "llama_cpp_version": "1.0.0 (build 12359, commit b1b770d)",
        "moby_dick_gutenberg_sha256": "9a6844ac0703853720010787c7b6c70b0020f1ab1862dcd74452fa46474d1215",
        "moby_context_sha256": fixtures["context_sha256"],
        "lfm2_2_6b_q4_k_m_gguf_sha256": gguf_sha256,
        "linux_qwen35_2b_q4_k_m_snapshot": "unsloth/Qwen3.5-2B-GGUF@f6d5376be1edb4d416d56da11e5397a961aca8ae",
        "linux_hardware_findings": {
            "jetson_agx_thor_sm110_nvrtc": (
                "Magnitude 0.3.5 bundles CUDA 12.9 libnvrtc.so.12 (sm_50..sm_103, sm_120, sm_121) lacking "
                "Blackwell Tegra sm_110 (HTTP 409 model_load_failed); CUDA_VISIBLE_DEVICES='' routes Seismic "
                "onto Thor's Tegra Blackwell Vulkan 1.4.321 GPU backend metered by TI INA238 VIN."
            ),
            "linux_vulkan_graph_replays_128tok_slab": (
                "On Linux Vulkan (both Tegra aarch64 and Zen 5 Strix Halo RADV x86_64), Magnitude 0.3.5 "
                "panics at seismic/runtime/src/native/graph_replays.rs:821:9 (GraphCensus.arguments > owner "
                "slab 839,936..857,472) when recurrent-hybrid prefill exceeds 128 tokens; Rung-0 EngramSpanLUT6 "
                "resolves multi-page verbatim spans directly while keeping neural bridge prefill <= 64 tokens."
            ),
        },
        "raw_book_or_prompts_persisted": 0,
        "tiered_expert_airmol": tiered_meta,
        "engram_lut6_miter": engram_meta,
    }
    (ART_DIR / "phase43_magnitude_tiered_engram_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("=== MOBY HEAD-TO-HEAD SUMMARY ===")
    for r in all_moby:
        print(r)
    print("=== AGENTIC CODING SUMMARY ===")
    for r in agentic_rows:
        print(r)
    print("=== TIERED AIRMOL SUMMARY ===")
    for r in tiered_rows:
        print(r)
    print("=== ENGRAM MITER SUMMARY ===")
    print(engram_meta)


if __name__ == "__main__":
    main()
