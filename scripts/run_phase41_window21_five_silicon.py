"""Phase 41 (`Claims LA-62 & LA-63`): Cross-Silicon `Window 21` vs `Window 21b` Shunt & Counter Telemetry:
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
    tgz = Path(f"/tmp/_lut6_core_p41_{host_tag[:8]}.tgz")
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
    subprocess.run(["scp", "-q", str(tgz), f"{host}:/tmp/_lut6_core_p41.tgz"], check=True)
    tgz.unlink(missing_ok=True)
    remote_sh = (
        "rm -rf /tmp/_lut6_p41 && mkdir -p /tmp/_lut6_p41 && "
        "tar -xzf /tmp/_lut6_core_p41.tgz -C /tmp/_lut6_p41 && "
        f"cd /tmp/_lut6_p41/sys/lut6-core && {cargo_cmd} --release --features std --bin bench_lut6_nucleus_shunt -- "
        f"--host-tag {host_tag} --iters 2000000 --out-json /tmp/_lut6_p41/out.json >/dev/null && "
        "cat /tmp/_lut6_p41/out.json && rm -rf /tmp/_lut6_p41 /tmp/_lut6_core_p41.tgz"
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
        w21 = _w(doc, "Window21_")
        w21b = _w(doc, "Window21b_")
        bm = bool(doc["board_metered"])
        if bm:
            assert w21["measured_nj_per_op_vdd_cpu"] is not None and w21["reported_nj_per_op"] is None
            e_cut = round(w21b["measured_nj_per_op_vdd_cpu"] / w21["measured_nj_per_op_vdd_cpu"], 3)
        else:
            assert w21["measured_nj_per_op_vdd_cpu"] is None and w21["reported_nj_per_op"] is not None
            e_cut = round(w21b["reported_nj_per_op"] / w21["reported_nj_per_op"], 3)
        rows.append(
            {
                "host_label": label,
                "board_metered": bm,
                "w21_rbfws8_enigma_horizon_ns_per_op": round(w21["ns_per_op"], 3),
                "w21b_kv64_polycipher_lb2_ns_per_op": round(w21b["ns_per_op"], 3),
                "latency_speedup_x": round(w21b["ns_per_op"] / w21["ns_per_op"], 3),
                "w21_measured_nj_per_op_vdd_cpu": w21["measured_nj_per_op_vdd_cpu"] if bm else None,
                "w21b_measured_nj_per_op_vdd_cpu": w21b["measured_nj_per_op_vdd_cpu"] if bm else None,
                "w21_reported_nj_per_op": None if bm else w21["reported_nj_per_op"],
                "w21b_reported_nj_per_op": None if bm else w21b["reported_nj_per_op"],
                "energy_reduction_x": e_cut,
            }
        )

    out_csv = ART_DIR / "phase41_window21_cross_silicon.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
