"""Phase 40 (`Claims LA-58 & LA-59`): Cross-Silicon `Window 20` vs `Window 20b` Shunt & Counter Telemetry:
  1. `NVIDIA Jetson AGX Thor` (`cbio@100.111.185.107`, `TI INA3221` `VDD_CPU_SOC_MSS` physical DC shunt, `board_metered = true`, `measured_j`),
  2. `AMD Ryzen AI Max+ 395` (`dcharlot@100.91.48.80`, `Zen 5` `RAPL` `reported_j`, `board_metered = false`),
  3. `Apple M3 Ultra 256 GB` (`dcharlot@100.100.141.123`, Darwin `RUSAGE_INFO_V6` `reported_j`, `board_metered = false`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def run_remote_shunt_host(host: str, cargo_cmd: str, out_json: Path, host_tag: str) -> dict:
    tgz = Path(f"/tmp/_lut6_core_p40_{host_tag[:8]}.tgz")
    subprocess.run(
        [
            "tar",
            "--exclude=sys/lut6-core/target",
            "-czf",
            str(tgz),
            "-C",
            str(REPO_ROOT),
            "sys/lut6-core",
            "artifacts/crystals",
        ],
        check=True,
    )
    subprocess.run(["scp", "-q", str(tgz), f"{host}:/tmp/_lut6_core_p40.tgz"], check=True)
    tgz.unlink(missing_ok=True)
    remote_sh = (
        f"rm -rf /tmp/_lut6_p40 && mkdir -p /tmp/_lut6_p40 && "
        f"tar -xzf /tmp/_lut6_core_p40.tgz -C /tmp/_lut6_p40 && "
        f"cd /tmp/_lut6_p40/sys/lut6-core && {cargo_cmd} --release --features std --bin bench_lut6_nucleus_shunt -- "
        f"--host-tag {host_tag} --iters 2000000 --out-json /tmp/_lut6_p40/out.json >/dev/null && "
        f"cat /tmp/_lut6_p40/out.json && rm -rf /tmp/_lut6_p40 /tmp/_lut6_core_p40.tgz"
    )
    raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host, remote_sh], text=True)
    out_json.write_text(raw, encoding="utf-8")
    return json.loads(raw)


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    thor = run_remote_shunt_host(
        "cbio@100.111.185.107",
        "$HOME/.cargo/bin/cargo run --quiet",
        ART_DIR / "lut6_nucleus_shunt_thor.json",
        "NVIDIA_Jetson_AGX_Thor_Tegra_aarch64",
    )
    zen5 = run_remote_shunt_host(
        "dcharlot@100.91.48.80",
        "$HOME/.cargo/bin/cargo run --quiet",
        ART_DIR / "lut6_nucleus_shunt_zen5.json",
        "AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64",
    )
    m3u = run_remote_shunt_host(
        "dcharlot@100.100.141.123",
        "$HOME/.cargo/bin/cargo run --quiet",
        ART_DIR / "lut6_nucleus_shunt_m3_ultra.json",
        "Apple_M3_Ultra_256GB_macOS_arm64",
    )

    def _w(doc: dict, pfx: str) -> dict:
        return next(x for x in doc["windows"] if x["window"].startswith(pfx))

    rows = []
    for label, doc in [
        ("NVIDIA_Jetson_AGX_Thor_Tegra_aarch64", thor),
        ("AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64", zen5),
        ("Apple_M3_Ultra_256GB_macOS_arm64", m3u),
    ]:
        w20 = _w(doc, "Window20_")
        w20b = _w(doc, "Window20b_")
        bm = bool(doc["board_metered"])
        if bm:
            assert w20["measured_nj_per_op_vdd_cpu"] is not None and w20["reported_nj_per_op"] is None
            e_cut = round(w20b["measured_nj_per_op_vdd_cpu"] / w20["measured_nj_per_op_vdd_cpu"], 3)
        else:
            assert w20["measured_nj_per_op_vdd_cpu"] is None and w20["reported_nj_per_op"] is not None
            e_cut = round(w20b["reported_nj_per_op"] / w20["reported_nj_per_op"], 3)
        rows.append(
            {
                "host_label": label,
                "board_metered": bm,
                "w20_bfws4_well_joulebit6_ns_per_op": round(w20["ns_per_op"], 3),
                "w20b_rung5_json_int8_lb2_ns_per_op": round(w20b["ns_per_op"], 3),
                "latency_speedup_x": round(w20b["ns_per_op"] / w20["ns_per_op"], 3),
                "w20_measured_nj_per_op_vdd_cpu": w20["measured_nj_per_op_vdd_cpu"] if bm else None,
                "w20b_measured_nj_per_op_vdd_cpu": w20b["measured_nj_per_op_vdd_cpu"] if bm else None,
                "w20_reported_nj_per_op": None if bm else w20["reported_nj_per_op"],
                "w20b_reported_nj_per_op": None if bm else w20b["reported_nj_per_op"],
                "energy_reduction_x": e_cut,
            }
        )

    out_csv = ART_DIR / "phase40_window20_cross_silicon.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
