"""Hardware & on-chip energy counter instrumentation:
  - Tier 1 (`measured_j`): Physical inline rail shunt / NVML / RAPL where hardware shunts exist (`None` on unmetered boards).
  - Tier 2 (`reported_j`): Dual-channel Apple Silicon on-chip telemetry (`L3` kernel per-PID `proc_pid_rusage(RUSAGE_INFO_V6).ri_energy_nj`
    CPU nanojoules + `L2` `libIOReport.dylib` `"Energy Model"` `'GPU Energy'` nanojoules, plus whole-system `ioreg` `PMU` deltas).
  - Tier 3 (`est_j`): Calibrated analytical TDP-time / operation-count estimate (`joules_gpu + joules_cpu_dram` fallback).
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

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

# Calibrated analytical power rails (Watts) for Tier-3 `est_j` fallback:
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


# =============================================================================
# Dual-Channel Apple Silicon On-Chip `reported_j` Reader (`L3 RUSAGE_INFO_V6` + `L2 IOReport`)
# =============================================================================

_RUSAGE_INFO_V6 = 6
_RI_INSTRUCTIONS_IDX = 31  # offsetof(struct rusage_info_v6, ri_instructions) / 8 == 31 (byte 248)
_RI_CYCLES_IDX = 32        # offsetof(struct rusage_info_v6, ri_cycles) / 8 == 32 (byte 256)
_RI_ENERGY_NJ_IDX = 42     # offsetof(struct rusage_info_v6, ri_energy_nj) / 8 == 42 (byte 336)
_KCFSTRING_UTF8 = 0x08000100

_UNIT_TO_JOULES = {
    "nJ": 1e-9,
    "uJ": 1e-6,
    "µJ": 1e-6,
    "mJ": 1e-3,
    "J": 1.0,
}


class _RUsageInfoV6Buf(ctypes.Structure):
    _fields_ = [("words", ctypes.c_uint64 * 64)]  # 512 B >= sizeof(struct rusage_info_v6) (464 B)


@dataclass
class ReportedJDelta:
    """Structured OpenIE `reported_j` hardware-counter delta on Apple Silicon."""

    l3_proc_cpu_j: float
    l2_gpu_j: float
    l2_cpu_pkg_j: float
    l2_dram_j: float
    l2_ane_j: float
    cpu_instructions: int
    cpu_cycles: int
    elapsed_ms: float

    @property
    def reported_j(self) -> float:
        """Total unprivileged on-chip `reported_j` (`L3` per-PID CPU + `L2` on-chip GPU/ANE/DRAM)."""
        cpu_term = self.l2_cpu_pkg_j if self.l2_cpu_pkg_j > 0.0 else self.l3_proc_cpu_j
        return cpu_term + self.l2_gpu_j + self.l2_dram_j + self.l2_ane_j

    def to_dict(self) -> dict[str, float | int | str | None]:
        d = asdict(self)
        d["reported_j"] = round(self.reported_j, 6)
        d["measured_j"] = None
        d["source"] = "apple_silicon_l3_rusage_v6_plus_l2_ioreport_energy_model"
        return d


class AppleSiliconReportedJReader:
    """
    Zero-dependency `ctypes` reader for Apple M-series on-chip energy counters:
      1. `L3` Kernel Per-Process CPU Energy: `proc_pid_rusage(pid, RUSAGE_INFO_V6).ri_energy_nj` (`nJ`).
      2. `L2` On-Chip `libIOReport.dylib` `"Energy Model"`: `'GPU Energy'` (`nJ`), plus `'CPU Energy'`,
         `'DRAM0'`, and `'ANE0'` (`mJ`) when exposed by OS privilege level.
    """

    def __init__(self) -> None:
        self.available = False
        self._libc = None
        self._cf = None
        self._ior = None
        self._sub = None
        self._subbed = ctypes.c_void_p()
        self._k_channels = None
        if not _IS_APPLE_SILICON:
            return
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")

            cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
            cf.CFStringGetCString.restype = ctypes.c_bool
            cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
            cf.CFArrayGetCount.restype = ctypes.c_long
            cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
            cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
            cf.CFRelease.argtypes = [ctypes.c_void_p]

            ior.IOReportCopyChannelsInGroup.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
            ]
            ior.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
            ior.IOReportCreateSubscription.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            ior.IOReportCreateSubscription.restype = ctypes.c_void_p
            ior.IOReportCreateSamples.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            ior.IOReportCreateSamples.restype = ctypes.c_void_p
            ior.IOReportCreateSamplesDelta.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            ior.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
            ior.IOReportChannelGetChannelName.argtypes = [ctypes.c_void_p]
            ior.IOReportChannelGetChannelName.restype = ctypes.c_void_p
            ior.IOReportChannelGetUnitLabel.argtypes = [ctypes.c_void_p]
            ior.IOReportChannelGetUnitLabel.restype = ctypes.c_void_p
            ior.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            ior.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64

            grp = cf.CFStringCreateWithCString(None, b"Energy Model", _KCFSTRING_UTF8)
            chans = ior.IOReportCopyChannelsInGroup(grp, None, 0, 0, 0)
            cf.CFRelease(grp)
            if not chans:
                return
            subbed = ctypes.c_void_p()
            sub = ior.IOReportCreateSubscription(None, chans, ctypes.byref(subbed), 0, None)
            cf.CFRelease(chans)
            if not sub:
                return

            self._libc = libc
            self._cf = cf
            self._ior = ior
            self._sub = sub
            self._subbed = subbed
            self._k_channels = cf.CFStringCreateWithCString(None, b"IOReportChannels", _KCFSTRING_UTF8)
            self.available = True
        except Exception:
            self.available = False

    def _cfstr_to_py(self, cfs: int | None) -> str:
        if not cfs or self._cf is None:
            return ""
        buf = ctypes.create_string_buffer(128)
        if self._cf.CFStringGetCString(cfs, buf, 128, _KCFSTRING_UTF8):
            return buf.value.decode("utf-8", errors="ignore")
        return ""

    def read_l3_proc_rusage(self, pid: int | None = None) -> tuple[int, int, int]:
        """Returns `(energy_nj, instructions, cycles)` for `pid` (default `os.getpid()`)."""
        if not self.available or self._libc is None:
            return 0, 0, 0
        ru = _RUsageInfoV6Buf()
        rc = self._libc.proc_pid_rusage(pid or os.getpid(), _RUSAGE_INFO_V6, ctypes.byref(ru))
        if rc != 0:
            return 0, 0, 0
        return (
            int(ru.words[_RI_ENERGY_NJ_IDX]),
            int(ru.words[_RI_INSTRUCTIONS_IDX]),
            int(ru.words[_RI_CYCLES_IDX]),
        )

    def snapshot(self) -> tuple[int | None, tuple[int, int, int], float]:
        """Captures `(ioreport_cf_sample, (l3_nj, inst, cyc), perf_counter_s)`."""
        t = time.perf_counter()
        if not self.available or self._ior is None:
            return None, (0, 0, 0), t
        s = self._ior.IOReportCreateSamples(self._sub, self._subbed, None)
        l3 = self.read_l3_proc_rusage()
        return s, l3, t

    def diff_and_release(
        self,
        snap_start: tuple[int | None, tuple[int, int, int], float],
        snap_end: tuple[int | None, tuple[int, int, int], float],
    ) -> ReportedJDelta | None:
        """Computes `ReportedJDelta` between two snapshots and releases CoreFoundation sample dicts."""
        s1, (nj1, inst1, cyc1), t1 = snap_start
        s2, (nj2, inst2, cyc2), t2 = snap_end
        if not self.available or self._cf is None or self._ior is None or not s1 or not s2:
            if self._cf is not None:
                if s1:
                    self._cf.CFRelease(s1)
                if s2:
                    self._cf.CFRelease(s2)
            return None

        delta = self._ior.IOReportCreateSamplesDelta(s1, s2, None)
        self._cf.CFRelease(s1)
        self._cf.CFRelease(s2)
        if not delta:
            return None

        gpu_j = 0.0
        cpu_pkg_j = 0.0
        dram_j = 0.0
        ane_j = 0.0
        dt_s = max(1e-6, t2 - t1)
        try:
            arr = self._cf.CFDictionaryGetValue(delta, self._k_channels)
            if arr:
                n = self._cf.CFArrayGetCount(arr)
                for i in range(n):
                    item = self._cf.CFArrayGetValueAtIndex(arr, i)
                    nm = self._cfstr_to_py(self._ior.IOReportChannelGetChannelName(item))
                    if nm in ("GPU Energy", "CPU Energy", "DRAM0", "DRAM", "ANE0", "ANE"):
                        un = self._cfstr_to_py(self._ior.IOReportChannelGetUnitLabel(item)).strip()
                        val = int(self._ior.IOReportSimpleGetIntegerValue(item, 0))
                        if val > 0:
                            scale = _UNIT_TO_JOULES.get(un, 1e-3)
                            joules = val * scale
                            # Guard against narrow SMC/IOReport hardware-accumulator rollover (> 450 W on M-series)
                            if dt_s >= 0.02 and (joules / dt_s) > 450.0:
                                continue
                            if nm == "GPU Energy":
                                gpu_j += joules
                            elif nm == "CPU Energy":
                                cpu_pkg_j += joules
                            elif nm in ("DRAM0", "DRAM"):
                                dram_j += joules
                            elif nm in ("ANE0", "ANE"):
                                ane_j += joules
        finally:
            self._cf.CFRelease(delta)

        return ReportedJDelta(
            l3_proc_cpu_j=max(0, nj2 - nj1) * 1e-9,
            l2_gpu_j=gpu_j,
            l2_cpu_pkg_j=cpu_pkg_j,
            l2_dram_j=dram_j,
            l2_ane_j=ane_j,
            cpu_instructions=max(0, inst2 - inst1),
            cpu_cycles=max(0, cyc2 - cyc1),
            elapsed_ms=(t2 - t1) * 1000.0,
        )


_APPLE_REPORTED_READER: AppleSiliconReportedJReader | None = None


def get_apple_reported_j_reader() -> AppleSiliconReportedJReader:
    global _APPLE_REPORTED_READER
    if _APPLE_REPORTED_READER is None:
        _APPLE_REPORTED_READER = AppleSiliconReportedJReader()
    return _APPLE_REPORTED_READER


@dataclass
class EnergyReceipt:
    joules_gpu: float
    joules_cpu_dram: float
    elapsed_ms: float
    reported_j: float | None = None
    measured_j: float | None = None
    reported_channels: dict[str, float | int] = field(default_factory=dict)

    @property
    def total_joules(self) -> float:
        return self.joules_gpu + self.joules_cpu_dram

    @property
    def est_j(self) -> float:
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
    Updates at ~0.1 Hz OS PMU cadence; strictly separated from analytical `est_j` and on-chip `reported_j`.
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
    """
    Context manager emitting an `EnergyReceipt` with separated OpenIE measurement tiers:
      - `est_j` (`joules_gpu + joules_cpu_dram` analytical model)
      - `reported_j` (on-chip Apple Silicon `L3 RUSAGE_INFO_V6` + `L2 IOReport` `'GPU Energy'`, or NVML/RAPL)
      - `measured_j` (`None` unless an external inline rail shunt is attached)
    """
    _sync_device(device)
    apple_rdr = get_apple_reported_j_reader()
    snap0 = apple_rdr.snapshot() if apple_rdr.available else None
    t0 = time.perf_counter()
    g0, c0 = _read_gpu_joules(), _read_rapl_joules()
    receipt = EnergyReceipt(0.0, 0.0, 0.0)
    try:
        yield receipt
    finally:
        _sync_device(device)
        dt = time.perf_counter() - t0
        g1, c1 = _read_gpu_joules(), _read_rapl_joules()
        snap1 = apple_rdr.snapshot() if (apple_rdr.available and snap0 is not None) else None
        receipt.elapsed_ms = dt * 1000.0
        gpu_w, cpu_w = _RAIL_WATTS.get(mode, _RAIL_WATTS["rung3"])
        receipt.joules_gpu = (g1 - g0) if (_HAS_NVML and g1 > g0) else (dt * gpu_w)
        receipt.joules_cpu_dram = (c1 - c0) if (_HAS_RAPL and c1 > c0) else (dt * cpu_w)

        if snap0 is not None and snap1 is not None:
            rep = apple_rdr.diff_and_release(snap0, snap1)
            if rep is not None:
                receipt.reported_j = rep.reported_j
                receipt.reported_channels = {
                    "l3_proc_cpu_j": round(rep.l3_proc_cpu_j, 6),
                    "l2_gpu_j": round(rep.l2_gpu_j, 6),
                    "l2_dram_j": round(rep.l2_dram_j, 6),
                    "l2_ane_j": round(rep.l2_ane_j, 6),
                    "cpu_instructions": rep.cpu_instructions,
                    "cpu_cycles": rep.cpu_cycles,
                }
        elif _HAS_NVML or _HAS_RAPL:
            receipt.reported_j = (max(0.0, g1 - g0) if _HAS_NVML else 0.0) + (max(0.0, c1 - c0) if _HAS_RAPL else 0.0)
