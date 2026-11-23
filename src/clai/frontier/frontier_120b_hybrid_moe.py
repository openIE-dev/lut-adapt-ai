"""Phase 28 (Claim LA-21): Measured `120B`-Scale Open-Weights Hybrid Mamba-Attention Sparse-MoE
(`nvidia/Nemotron-3-Super-120B-A12B`) & `24B` SWE Specialist (`Devstral-Small-2-24B-Instruct`)
Streaming, Speculative Lookup Parity, Dual-Channel `reported_j`, and Apex `C(z)` Escalation.

Replaces the `70B+` analytical projection caveat in Section 6 with real measured runs on Apple M5 Max
(`128 GiB` Unified Memory) via native Darwin arm64 Metal `llama-server`, reading:
  - Child-process `proc_pid_rusage(pid, RUSAGE_INFO_V6)` for exact `ri_phys_footprint` (Peak RAM MB)
    and `ri_energy_nj` (`L3` CPU `nJ`) of the `120B` / `24B` server process, plus
  - `AppleSiliconReportedJReader` (`libIOReport` `'GPU Energy'`) for `L2` Metal GPU `nJ`,
while keeping `measured_j = None` and `board_metered = False`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from typing import Any

from clai.contracts.joule_meter import get_apple_reported_j_reader


@dataclass
class SubprocessTelemetrySnapshot:
    wall_time_s: float
    child_cpu_j: float
    host_gpu_j: float
    total_reported_j: float
    child_phys_footprint_mb: float
    child_resident_rss_mb: float = 0.0


def read_darwin_pid_rusage(pid: int) -> tuple[float, float, float]:
    """
    Calls Darwin `proc_pid_rusage(pid, RUSAGE_INFO_V6, &buf)` on an external/child PID.
    Returns `(cpu_energy_joules, phys_footprint_mb, resident_rss_mb)`.
    """
    if os.uname().sysname != "Darwin":
        return 0.0, 0.0, 0.0
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buf = (ctypes.c_uint64 * 64)()
        rc = libsystem.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(6), ctypes.byref(buf))
        if rc != 0:
            return 0.0, 0.0, 0.0
        # Offsets in `struct rusage_info_v6` (sizeof=464 on macOS SDK):
        #   `ri_resident_size` is at byte offset 64 (`buf[8]`)
        #   `ri_phys_footprint` is at byte offset 72 (`buf[9]`)
        #   `ri_energy_nj` is at byte offset 336 (`buf[42]`)
        rss_bytes = int(buf[8])
        phys_bytes = int(buf[9])
        energy_nj = int(buf[42])
        return energy_nj * 1.0e-9, phys_bytes / (1024.0 * 1024.0), rss_bytes / (1024.0 * 1024.0)
    except Exception:
        return 0.0, 0.0, 0.0


class LocalMetalGGUFServer:
    """Manages an isolated local loopback `llama-server` instance on Apple Metal."""

    def __init__(
        self,
        model_path: Path,
        port: int = 18992,
        ctx_size: int = 4096,
        extra_args: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.ctx_size = ctx_size
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen[str] | None = None
        self.base_url = f"http://127.0.0.1:{port}"

    def __enter__(self) -> "LocalMetalGGUFServer":
        srv_bin = shutil.which("llama-server") or "/opt/homebrew/bin/llama-server"
        if not Path(srv_bin).is_file():
            raise FileNotFoundError(f"`llama-server` not found at {srv_bin}")
        cmd = [
            srv_bin,
            "-m",
            str(self.model_path),
            "-ngl",
            "999",
            "-c",
            str(self.ctx_size),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.perf_counter() + 180.0
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"`llama-server` exited prematurely with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "ok":
                            return self
            except Exception:
                pass
            time.sleep(0.5)
        self.close()
        raise TimeoutError(f"Timed out waiting for `llama-server` ({self.model_path.name}) on port {self.port}")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3.0)
        self.proc = None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def chat_complete_metered(
        self,
        prompt: str,
        max_tokens: int = 768,
        system_prompt: str = (
            "You are an expert Python 3 systems & algorithms engineer. "
            "Output ONLY valid self-contained Python 3 code inside one ```python ... ``` block."
        ),
    ) -> tuple[str, dict[str, Any], SubprocessTelemetrySnapshot]:
        assert self.proc is not None
        gpu_reader = get_apple_reported_j_reader()
        snap0 = gpu_reader.snapshot() if gpu_reader.available else None
        c0_j, _, _ = read_darwin_pid_rusage(self.proc.pid)
        t0 = time.perf_counter()

        payload = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": max_tokens,
                "cache_prompt": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        dt = time.perf_counter() - t0
        c1_j, phys_mb, rss_mb = read_darwin_pid_rusage(self.proc.pid)
        snap1 = gpu_reader.snapshot() if (gpu_reader.available and snap0 is not None) else None

        cpu_j = max(0.0, c1_j - c0_j)
        gpu_j = 0.0
        if snap0 is not None and snap1 is not None:
            rep = gpu_reader.diff_and_release(snap0, snap1)
            if rep is not None:
                gpu_j = rep.l2_gpu_j + rep.l2_dram_j + rep.l2_ane_j
                if rep.l2_cpu_pkg_j > 0.0:
                    cpu_j = rep.l2_cpu_pkg_j

        choice = body["choices"][0]["message"]
        txt = choice.get("content") or ""
        usage = body.get("usage", {})
        timings = body.get("timings", {})
        snap = SubprocessTelemetrySnapshot(
            wall_time_s=dt,
            child_cpu_j=round(cpu_j, 4),
            host_gpu_j=round(gpu_j, 4),
            total_reported_j=round(cpu_j + gpu_j, 4),
            child_phys_footprint_mb=round(phys_mb, 1),
            child_resident_rss_mb=round(rss_mb, 1),
        )
        return txt, {"usage": usage, "timings": timings}, snap
