"""Aggregate Phase 29 (Claim LA-25) Four-Vendor GPU `Burn 0.22.1 / CubeCL` Shader Parity &
Hardware-Counter (`reported_j`) Benchmark Results.

Covers four distinct GPU silicon vendors executing the exact same `#[cube]` Rust kernels
(`q4_moe_ffn_pairs_gpu` and `gated_delta_recurrent_k128_gpu`):
  1. Apple Silicon (`Apple_M5_Max_40Core_Metal` — `CubeCL_Metal_GPU` MSL)
  2. AMD RDNA 3.5 (`AMD_Radeon_8060S_StrixHalo_Vulkan` — `CubeCL_Vulkan_SPIRV`, `amdgpu` PPT + RAPL)
  3. NVIDIA Ada Lovelace (`NVIDIA_GeForce_RTX_4050_Ada_Vulkan` — `CubeCL_Vulkan_SPIRV`, `NVML` mJ + RAPL)
  4. Intel Xe-LP (`Intel_Iris_Xe_RPL_Vulkan` — `CubeCL_Vulkan_SPIRV`, `RAPL` pkg)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


VENDOR_FILES = [
    ("apple_m5_max_metal.json", "Apple", "Apple M5 Max (40-core GPU)"),
    ("amd_radeon_8060s_vulkan.json", "AMD", "AMD Radeon 8060S (gfx1151 RDNA 3.5)"),
    ("nvidia_rtx4050_vulkan.json", "NVIDIA", "NVIDIA GeForce RTX 4050 Laptop (sm_89 Ada)"),
    ("intel_iris_xe_vulkan.json", "Intel", "Intel Iris Xe Graphics (RPL-P)"),
]


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "artifacts" / "cubecl_vendors_raw"
    out_csv = root / "artifacts" / "cubecl_four_vendor_gpu_parity.csv"
    out_meta = root / "artifacts" / "cubecl_four_vendor_gpu_parity_meta.json"

    rows: list[dict[str, Any]] = []
    vendor_summaries: list[dict[str, Any]] = []

    for fname, vendor, device_model in VENDOR_FILES:
        p = raw_dir / fname
        if not p.is_file():
            raise FileNotFoundError(f"Missing vendor GPU JSON: {p}")
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["board_metered"] is False and d["measured_j"] is None

        for k_key, shape_spec in [
            ("q4_moe_gate_up_silu_down", "D=2048_I=512_K=28_P=32"),
            ("gated_delta_recurrent_k128", "B=1_Hk=16_Hv=32_T=256_K=128_V=128"),
        ]:
            kd = d[k_key]
            ge = kd["cubecl_gpu_energy"]
            ce = kd["cubecl_cpu_energy"]
            rows.append(
                {
                    "gpu_vendor": vendor,
                    "device_model": device_model,
                    "hardware_label": d["hardware_label"],
                    "os": d["os"],
                    "arch": d["arch"],
                    "gpu_backend": d["gpu_backend"],
                    "kernel_family": k_key,
                    "shape_spec": shape_spec,
                    "host_scalar_ref_ms": _fmt(kd["host_ref_ms"]),
                    "cubecl_llvm_cpu_ms": _fmt(kd["cubecl_cpu_ms"]),
                    "cubecl_llvm_cpu_speedup_x": _fmt(kd["cubecl_cpu_speedup"]),
                    "cubecl_llvm_cpu_max_abs_err": f"{kd['cubecl_cpu_max_err']:.3e}",
                    "cubecl_gpu_ms": _fmt(kd["cubecl_gpu_ms"]),
                    "cubecl_gpu_speedup_vs_ref_x": _fmt(kd["cubecl_gpu_speedup"]),
                    "cubecl_gpu_speedup_vs_cubecl_cpu_x": _fmt(kd["cubecl_cpu_ms"] / kd["cubecl_gpu_ms"]),
                    "cubecl_gpu_max_abs_err": f"{kd['cubecl_gpu_max_err']:.3e}",
                    "nvml_gpu_reported_mj_per_dispatch": _fmt(
                        ge["nvml_gpu_reported_mj_per_dispatch"] if vendor == "NVIDIA" else None
                    ),
                    "amdgpu_ppt_reported_mj_per_dispatch": _fmt(
                        ge["amdgpu_ppt_reported_mj_per_dispatch"] if vendor == "AMD" else None
                    ),
                    "rapl_pkg_reported_mj_per_dispatch": _fmt(ge["rapl_pkg_reported_mj_per_dispatch"]),
                    "darwin_l3_cpu_reported_mj_per_dispatch": _fmt(ge["darwin_l3_cpu_reported_mj_per_dispatch"]),
                    "cpu_rapl_pkg_reported_mj_per_dispatch": _fmt(ce["rapl_pkg_reported_mj_per_dispatch"]),
                    "board_metered": False,
                    "measured_j": "",
                }
            )

        vendor_summaries.append(
            {
                "gpu_vendor": vendor,
                "device_model": device_model,
                "gpu_backend": d["gpu_backend"],
                "q4_moe_gpu_ms": round(d["q4_moe_gate_up_silu_down"]["cubecl_gpu_ms"], 4),
                "q4_moe_speedup_vs_ref_x": round(d["q4_moe_gate_up_silu_down"]["cubecl_gpu_speedup"], 2),
                "q4_moe_max_abs_err": d["q4_moe_gate_up_silu_down"]["cubecl_gpu_max_err"],
                "deltanet_gpu_ms": round(d["gated_delta_recurrent_k128"]["cubecl_gpu_ms"], 4),
                "deltanet_speedup_vs_ref_x": round(d["gated_delta_recurrent_k128"]["cubecl_gpu_speedup"], 2),
                "deltanet_max_abs_err": d["gated_delta_recurrent_k128"]["cubecl_gpu_max_err"],
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "claim_id": "LA-25",
        "phase": "Phase 29: Four-Vendor Hardware GPU Shader Parity (Apple Metal + AMD Vulkan + NVIDIA Vulkan + Intel Vulkan)",
        "burn_cubecl_version": "0.22.1",
        "rustc_version": "rustc 1.100.0 (0ef1bc386 2026-11-09)",
        "distinct_gpu_vendors": len(vendor_summaries),
        "vendors": ["Apple", "AMD", "NVIDIA", "Intel"],
        "single_source_files": [
            "sys/rust/src/kernels/q4_moe.rs",
            "sys/rust/src/kernels/gated_deltanet.rs",
        ],
        "max_fp32_parity_error_across_all_4_vendors": max(
            max(v["q4_moe_max_abs_err"], v["deltanet_max_abs_err"]) for v in vendor_summaries
        ),
        "board_metered": False,
        "measured_j": None,
        "vendor_summaries": vendor_summaries,
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows -> {out_csv}")
    print(f"Wrote metadata -> {out_meta}")


if __name__ == "__main__":
    main()
