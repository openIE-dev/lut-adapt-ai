"""Phase 39 Aggregator (`Claims LA-54 & LA-55`):
  1. Exports `artifacts/phase39_bonsai27b_joulebit_stream.csv` from real `Apple M3 Ultra 256 GB`
     `prism-ml/Ternary-Bonsai-2-27B` Cold Wave-1 & Warm Wave-2 speculative streaming telemetry.
  2. Exports `artifacts/phase39_window19_cross_silicon.csv` comparing `Window 19` (`SIW(2)` Landmark
     + Fused Zero-MUL `FWHT-64 + 6-Col JouleBit-LUT6`) against `Window 19b` (`LittleBit-2` `6`-Col
     Unpack + Binary MAC) across `Jetson AGX Thor` (`TI INA3221` + `INA238` physical DC shunt
     `measured_j`, `board_metered = true`), `AMD Zen 5` (`amd_rapl` `reported_j`), and `Apple M3 Ultra`
     (`Darwin RUSAGE_INFO_V6` `reported_j`).
  3. Exports `artifacts/phase39_siw2_joulebit_meta.json` linking `ARC-AGI-3` `SIW(2)`, `JouleBit-27B`,
     `openXC7` `XC7A200T` zero-`DSP48E1` gateware, `14`-export `lut6_core.wasm` (`<= 16 KB`), and
     `98` six-kernel `Lean 4.36.0` proofs (`1,462` Linux `bwrap` fleet).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def _find_win(doc: dict, prefix: str) -> dict:
    return next(w for w in doc["windows"] if w["window"].startswith(prefix))


def main() -> None:
    m3u_27b = json.loads((ART_DIR / "phase39_bonsai27b_joulebit_m3_ultra.json").read_text(encoding="utf-8"))
    thor = json.loads((ART_DIR / "lut6_nucleus_shunt_thor.json").read_text(encoding="utf-8"))
    zen5 = json.loads((ART_DIR / "lut6_nucleus_shunt_zen5.json").read_text(encoding="utf-8"))
    m3u_shunt = json.loads((ART_DIR / "lut6_nucleus_shunt_m3_ultra.json").read_text(encoding="utf-8"))
    xc7 = json.loads((ART_DIR / "openxc7_siw_joulebit_timing.json").read_text(encoding="utf-8"))
    arc3_meta = json.loads((ART_DIR / "phase39_arc3_16env_siw2_meta.json").read_text(encoding="utf-8"))
    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))

    # 1. Write `phase39_bonsai27b_joulebit_stream.csv`
    rows_27b = m3u_27b["rows"]
    with (ART_DIR / "phase39_bonsai27b_joulebit_stream.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_27b[0].keys()))
        w.writeheader()
        w.writerows(rows_27b)

    # 2. Write `phase39_window19_cross_silicon.csv` (strictly separating `measured_j` from `reported_j`!)
    w19_rows = []
    for host_label, doc in [
        ("NVIDIA_Jetson_AGX_Thor_Tegra_aarch64", thor),
        ("AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64", zen5),
        ("Apple_M3_Ultra_256GB_macOS_arm64", m3u_shunt),
    ]:
        w19 = _find_win(doc, "Window19_")
        w19b = _find_win(doc, "Window19b_")
        bm = bool(doc["board_metered"])
        if bm:
            assert w19["measured_nj_per_op_vdd_cpu"] is not None and w19["reported_nj_per_op"] is None
            e_ratio = round(w19b["measured_nj_per_op_vdd_cpu"] / w19["measured_nj_per_op_vdd_cpu"], 3)
        else:
            assert w19["measured_nj_per_op_vdd_cpu"] is None and w19["reported_nj_per_op"] is not None
            e_ratio = round(w19b["reported_nj_per_op"] / w19["reported_nj_per_op"], 3)
        lat_speedup = round(w19b["ns_per_op"] / w19["ns_per_op"], 3)
        w19_rows.append(
            {
                "host_label": host_label,
                "board_metered": bm,
                "w19_siw2_joulebit6_ns_per_op": round(w19["ns_per_op"], 3),
                "w19b_littlebit2_mac_ns_per_op": round(w19b["ns_per_op"], 3),
                "latency_speedup_x": lat_speedup,
                "w19_measured_nj_per_op_vdd_cpu": w19["measured_nj_per_op_vdd_cpu"],
                "w19b_measured_nj_per_op_vdd_cpu": w19b["measured_nj_per_op_vdd_cpu"],
                "w19_reported_nj_per_op": w19["reported_nj_per_op"],
                "w19b_reported_nj_per_op": w19b["reported_nj_per_op"],
                "energy_reduction_x": e_ratio,
            }
        )

    with (ART_DIR / "phase39_window19_cross_silicon.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(w19_rows[0].keys()))
        w.writeheader()
        w.writerows(w19_rows)

    # 3. Verify 14-export `lut6_core.wasm` <= 16,384 B in V8 (`node`)
    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    wasm_bytes = wasm_path.stat().st_size
    assert wasm_bytes <= 16384, f"lut6_core.wasm exceeded 16 KB: {wasm_bytes} B"
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const fns = Object.keys(ex).filter(k => typeof ex[k] === 'function');
  const siwB0 = ex.wasm_eval_siw2_and_joulebit6((1 << 28) | (0x42 << 20) | 0x02, 10) >>> 0;
  const siwLm = ex.wasm_eval_siw2_and_joulebit6((1 << 28) | (0x42 << 20) | (1 << 8) | 0x01, 10) >>> 0;
  console.log(JSON.stringify({{ fnCount: fns.length, siwB0, siwLm }}));
}})();
"""
    wasm_v8 = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert wasm_v8["fnCount"] == 14

    meta_out = {
        "phase": "Phase 39 (Claims LA-53..LA-55 + 98-Theorem Claim LA-19 Upgrade)",
        "arc3_siw2_summary": arc3_meta["summary_rows"][-2:],
        "bonsai27b_joulebit_m3_ultra": {
            "real_hidden_state_top1_audit": m3u_27b["real_hidden_state_top1_audit"],
            "pread_jbit6_gbps": m3u_27b["pread_jbit6_gbps"],
            "joulebit_022_rel_frob": m3u_27b["joulebit_022_rel_frob"],
            "joulebit_045_rel_frob": m3u_27b["joulebit_045_rel_frob"],
            "ours_row": rows_27b[-2],
        },
        "window19_cross_silicon": w19_rows,
        "openxc7_siw_joulebit_xc7a200t": xc7,
        "wasm_14_export": {
            "wasm_bytes": wasm_bytes,
            "wasm_kb": round(wasm_bytes / 1024.0, 2),
            "exported_functions": wasm_v8["fnCount"],
            "fits_under_16kb": wasm_bytes <= 16384,
        },
        "lean4_six_kernel": {
            "lut_adapt_theorems": lean_meta["total_kernel_checked_theorems"],
            "fleet_bwrap_theorems": fleet_meta["total_theorems_and_lemmas"],
            "kernels_passed": lean_meta["independent_kernels_passed"],
        },
        "board_metered": False,
        "measured_j": None,
    }
    (ART_DIR / "phase39_siw2_joulebit_meta.json").write_text(
        json.dumps(meta_out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta_out, indent=2))


if __name__ == "__main__":
    main()
