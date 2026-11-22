"""Hardware energy counter instrumentation (NVML + RAPL + calibrated Apple Silicon / TDP fallback)."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _NVML_HANDLE = None
    _HAS_NVML = False

_RAPL_PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
_HAS_RAPL = os.path.exists(_RAPL_PKG) and os.access(_RAPL_PKG, os.R_OK)
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Calibrated power rails (Watts) when hardware root counters are unavailable:
# (gpu_watts, cpu_dram_watts)
_RAIL_WATTS = {
    "rung0": (0.8, 2.2) if _IS_APPLE_SILICON else (1.5, 12.0),
    "rung2": (28.0, 6.0) if _IS_APPLE_SILICON else (65.0, 15.0),
    "rung3": (95.0, 10.0) if _IS_APPLE_SILICON else (220.0, 25.0),
    "compile": (75.0, 8.0) if _IS_APPLE_SILICON else (180.0, 20.0),
}


def select_torch_device(prefer_mps: bool = True) -> torch.device:
    """Selects optimal accelerator (CLAI_DEVICE override -> CUDA -> Apple MPS -> CPU)."""
    override = os.environ.get("CLAI_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync_device(device: torch.device | None) -> None:
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class EnergyReceipt:
    joules_gpu: float
    joules_cpu_dram: float
    elapsed_ms: float

    @property
    def total_joules(self) -> float:
        return self.joules_gpu + self.joules_cpu_dram


def _read_gpu_joules() -> float:
    if _HAS_NVML:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(_NVML_HANDLE) * 1e-3
    return 0.0


def _read_rapl_joules() -> float:
    if _HAS_RAPL:
        with open(_RAPL_PKG, "r", encoding="utf-8") as f:
            return int(f.read().strip()) * 1e-6
    return 0.0


def read_apple_pmu_telemetry() -> tuple[float, int] | None:
    """
    Reads non-privileged macOS IORegistry `AppleSmartBattery.PowerTelemetryData`
    (`AccumulatedSystemEnergyConsumed` in mJ -> J, and `SystemLoadAccumulatorCount` tick).
    Updates at ~0.1 Hz OS PMU cadence; strictly separated from analytical `est_j`.
    """
    if not _IS_APPLE_SILICON:
        return None
    try:
        txt = subprocess.check_output(
            ["ioreg", "-rw0", "-c", "AppleSmartBattery"], text=True, stderr=subprocess.DEVNULL, timeout=1.5
        )
        m = re.search(r'"PowerTelemetryData"\s*=\s*\{([^}]+)\}', txt)
        if not m:
            return None
        kv = {k: int(v) for k, v in re.findall(r'"([^"]+)"=(\d+)', m.group(1))}
        if "AccumulatedSystemEnergyConsumed" in kv and "SystemLoadAccumulatorCount" in kv:
            return kv["AccumulatedSystemEnergyConsumed"] * 1e-3, kv["SystemLoadAccumulatorCount"]
    except Exception:
        pass
    return None


@contextmanager
def meter_joules(mode: str = "rung3", device: torch.device | None = None):
    """Context manager emitting an EnergyReceipt for every inference or compile pass."""
    _sync_device(device)
    t0 = time.perf_counter()
    g0, c0 = _read_gpu_joules(), _read_rapl_joules()
    receipt = EnergyReceipt(0.0, 0.0, 0.0)
    try:
        yield receipt
    finally:
        _sync_device(device)
        dt = time.perf_counter() - t0
        g1, c1 = _read_gpu_joules(), _read_rapl_joules()
        receipt.elapsed_ms = dt * 1000.0
        gpu_w, cpu_w = _RAIL_WATTS.get(mode, _RAIL_WATTS["rung3"])
        receipt.joules_gpu = (g1 - g0) if (_HAS_NVML and g1 > g0) else (dt * gpu_w)
        receipt.joules_cpu_dram = (c1 - c0) if (_HAS_RAPL and c1 > c0) else (dt * cpu_w)
