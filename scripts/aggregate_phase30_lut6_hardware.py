"""Phase 30 (`Claim LA-26` & `Claim LA-28`): Aggregates Cross-Silicon `lut6-core` Hardware
Telemetry (`NVIDIA Jetson AGX Thor` physical DC shunt `measured_j`, `Apple M5 Max` Darwin L3
`reported_j`, `AMD Ryzen AI Max+ 395` `amd_rapl` `reported_j`, live attached `Alif Ensemble E3`
`OpenMV-AE3` Cortex-M55 `DWT->CYCCNT` hardware cycles, and bare-metal `thumbv8m` / `thumbv6m`
release ELF section footprints).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


def _elf_sections(elf_path: Path) -> dict[str, int]:
    out = subprocess.check_output(
        ["/opt/homebrew/bin/arm-none-eabi-size", "-A", str(elf_path)], text=True
    )
    secs: dict[str, int] = {}
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            secs[parts[0]] = int(parts[1])
    return secs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    art_dir = repo_root / "artifacts"

    thor = json.loads((art_dir / "lut6_nucleus_thor_shunt.json").read_text(encoding="utf-8"))
    m5 = json.loads((art_dir / "lut6_nucleus_m5_max.json").read_text(encoding="utf-8"))
    fw = json.loads((art_dir / "lut6_nucleus_fw128b.json").read_text(encoding="utf-8"))
    ae3 = json.loads((art_dir / "alif_ae3_live_hardware_cycles.json").read_text(encoding="utf-8"))

    alif_elf = repo_root / "sys/alif-lut6-bench/target/thumbv8m.main-none-eabihf/release/alif-lut6-bench"
    pico_elf = repo_root / "sys/pico-lut6-bench/target/thumbv6m-none-eabi/release/pico-lut6-bench"
    alif_secs = _elf_sections(alif_elf)
    pico_secs = _elf_sections(pico_elf)

    rows: list[dict] = []
    for host_doc, hw_label in [
        (thor, "NVIDIA Jetson AGX Thor (INA3221 VDD_CPU_SOC_MSS Physical Shunt)"),
        (m5, "Apple M5 Max (Darwin RUSAGE_INFO_V6 Counter)"),
        (fw, "AMD Ryzen AI Max+ 395 (Linux amd_rapl Package Counter)"),
    ]:
        win_map = {w["window"]: w for w in host_doc["windows"]}
        mlp_w = win_map["Window6_Baseline_INT8_MicroMLP"]
        for w in host_doc["windows"]:
            if host_doc["board_metered"]:
                e_self = w["measured_nj_per_op_vdd_cpu"]
                e_mlp = mlp_w["measured_nj_per_op_vdd_cpu"]
            else:
                e_self = w["reported_nj_per_op"]
                e_mlp = mlp_w["reported_nj_per_op"]
            cpj_vs_mlp = round(e_mlp / e_self, 2) if (e_self and e_mlp and e_self > 0) else None
            Speedup_vs_mlp = round(mlp_w["ns_per_op"] / w["ns_per_op"], 2) if w["ns_per_op"] > 0 else None
            rows.append(
                {
                    "host_tag": host_doc["host_tag"],
                    "hardware": hw_label,
                    "window": w["window"],
                    "ns_per_op": round(w["ns_per_op"], 3),
                    "speedup_vs_int8_mlp": Speedup_vs_mlp,
                    "measured_nj_per_op_vdd_cpu": (
                        round(w["measured_nj_per_op_vdd_cpu"], 3)
                        if w["measured_nj_per_op_vdd_cpu"] is not None
                        else None
                    ),
                    "reported_nj_per_op": (
                        round(w["reported_nj_per_op"], 3)
                        if w["reported_nj_per_op"] is not None
                        else None
                    ),
                    "cpj_gain_vs_int8_mlp": cpj_vs_mlp,
                    "board_metered": host_doc["board_metered"],
                }
            )

    # Append live Alif Ensemble E3 (OpenMV-AE3) hardware cycle rows (`measured_j = None`, `reported_nj = None`)
    mlp_cyc = ae3["int8_micro_mlp_dwt_cycles_per_decision"]
    ae3_win_list = [
        ("Window1_Innate_Rung6_Hit", ae3["innate_lut6_us_per_decision"], ae3["innate_lut6_dwt_cycles_per_decision"]),
        ("Window4_PostCompile_Noisy_Wobble_Hit", ae3["wobble_overlay_us_per_decision"], ae3["wobble_overlay_dwt_cycles_per_decision"]),
        ("Window5_Vertical_BitSliced_Batch32", ae3["vertical_bitsliced_batch32_us_per_decision"], ae3["vertical_bitsliced_batch32_dwt_cycles_per_decision"]),
    ]
    if "zero_mul_tmac_tail_dwt_cycles_per_decision" in ae3:
        ae3_win_list.append(
            ("Window8_ZeroMul_Ternary_TMAC_Tail", ae3["zero_mul_tmac_tail_us_per_decision"], ae3["zero_mul_tmac_tail_dwt_cycles_per_decision"])
        )
    if "antisym40_tmac_precomp_dwt_cycles_per_decision" in ae3:
        ae3_win_list.append(
            ("Window9_Antisym40_ZeroMul_TMAC_Tail", ae3["antisym40_tmac_precomp_us_per_decision"], ae3["antisym40_tmac_precomp_dwt_cycles_per_decision"])
        )
    if "wearable_56bit_epistemic_dwt_cycles_per_decision" in ae3:
        ae3_win_list.append(
            ("Window10_Wearable_56Bit_4Tok_Epistemic_Gate", ae3["wearable_56bit_epistemic_us_per_decision"], ae3["wearable_56bit_epistemic_dwt_cycles_per_decision"])
        )
    if "ulm28m_d512_slice_dwt_cycles_per_decision" in ae3:
        ae3_win_list.append(
            ("Window11_Baseline_uLM28M_d512_Slice", ae3["ulm28m_d512_slice_us_per_decision"], ae3["ulm28m_d512_slice_dwt_cycles_per_decision"])
        )
    ae3_win_list.append(("Window6_Baseline_INT8_MicroMLP", ae3["int8_micro_mlp_us_per_decision"], mlp_cyc))
    for win_name, us_val, cyc_val in ae3_win_list:
        rows.append(
            {
                "host_tag": "openmv-ae3-alif-e3",
                "hardware": ae3["board"],
                "window": f"{win_name} ({cyc_val} DWT cyc)",
                "ns_per_op": round(us_val * 1000.0, 2),
                "speedup_vs_int8_mlp": round(mlp_cyc / cyc_val, 2),
                "measured_nj_per_op_vdd_cpu": None,
                "reported_nj_per_op": None,
                "cpj_gain_vs_int8_mlp": None,
                "board_metered": False,
            }
        )

    out_csv = art_dir / "lut6_core_hardware_matrix.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    thor_win_map = {w["window"]: w for w in thor["windows"]}
    tmac_nj = (
        round(thor_win_map["Window8_ZeroMul_Ternary_TMAC_Tail"]["measured_nj_per_op_vdd_cpu"], 2)
        if "Window8_ZeroMul_Ternary_TMAC_Tail" in thor_win_map
        else None
    )
    antisym40_nj = (
        round(thor_win_map["Window9_Antisym40_ZeroMul_TMAC_Tail"]["measured_nj_per_op_vdd_cpu"], 2)
        if "Window9_Antisym40_ZeroMul_TMAC_Tail" in thor_win_map
        else None
    )
    unrolled4_nj = (
        round(thor_win_map["Window7_Unrolled4_Scalar_LUT6"]["measured_nj_per_op_vdd_cpu"], 2)
        if "Window7_Unrolled4_Scalar_LUT6" in thor_win_map
        else None
    )
    wear_ep_nj = (
        round(thor_win_map["Window10_Wearable_56Bit_4Tok_Epistemic_Gate"]["measured_nj_per_op_vdd_cpu"], 2)
        if "Window10_Wearable_56Bit_4Tok_Epistemic_Gate" in thor_win_map
        else None
    )
    ulm_slice_nj = (
        round(thor_win_map["Window11_Baseline_uLM28M_d512_Slice"]["measured_nj_per_op_vdd_cpu"], 2)
        if "Window11_Baseline_uLM28M_d512_Slice" in thor_win_map
        else None
    )
    baremetal_meta = {
        "rust_toolchain": "rustc 1.100.0 (0ef1bc386 2026-11-09), edition = 2024",
        "cross_checked_targets": [
            "aarch64-apple-darwin",
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
            "thumbv8m.main-none-eabihf",
            "thumbv6m-none-eabi",
            "thumbv7em-none-eabihf",
            "riscv32imac-unknown-none-elf",
            "wasm32-unknown-unknown",
        ],
        "alif_e3_e4_cortex_m55_elf": {
            "target": "thumbv8m.main-none-eabihf (-C target-cpu=cortex-m55)",
            "sections_bytes": alif_secs,
        },
        "pi_pico_rp2040_cortex_m0plus_elf": {
            "target": "thumbv6m-none-eabi (-C target-cpu=cortex-m0plus)",
            "sections_bytes": pico_secs,
        },
        "live_attached_alif_ae3_dwt_telemetry": ae3,
        "thor_shunt_summary": {
            "board_metered": True,
            "innate_hit_measured_nj": round(thor["windows"][0]["measured_nj_per_op_vdd_cpu"], 2),
            "wobble_hit_measured_nj": round(thor["windows"][3]["measured_nj_per_op_vdd_cpu"], 2),
            "bitsliced_batch64_measured_nj": round(thor["windows"][4]["measured_nj_per_op_vdd_cpu"], 2),
            "unrolled4_scalar_measured_nj": unrolled4_nj,
            "zero_mul_tmac_tail_measured_nj": tmac_nj,
            "antisym40_tmac_tail_measured_nj": antisym40_nj,
            "tmac_cpj_gain_vs_int8_mlp": (
                round(thor["windows"][5]["measured_nj_per_op_vdd_cpu"] / tmac_nj, 2)
                if tmac_nj
                else None
            ),
            "antisym40_cpj_gain_vs_int8_mlp": (
                round(thor["windows"][5]["measured_nj_per_op_vdd_cpu"] / antisym40_nj, 2)
                if antisym40_nj
                else None
            ),
            "antisym40_gain_over_full81_tmac": (
                round(tmac_nj / antisym40_nj, 2)
                if (tmac_nj and antisym40_nj)
                else None
            ),
            "wearable_56bit_epistemic_measured_nj": wear_ep_nj,
            "ulm28m_d512_slice_measured_nj": ulm_slice_nj,
            "wearable_epistemic_cpj_gain_vs_ulm28m_slice": (
                round(ulm_slice_nj / wear_ep_nj, 2)
                if (wear_ep_nj and ulm_slice_nj)
                else None
            ),
            "int8_mlp_measured_nj": round(thor["windows"][5]["measured_nj_per_op_vdd_cpu"], 2),
            "bitsliced_cpj_gain_vs_int8_mlp": round(
                thor["windows"][5]["measured_nj_per_op_vdd_cpu"]
                / thor["windows"][4]["measured_nj_per_op_vdd_cpu"],
                2,
            ),
            "wobble_break_even_calls_n_crit": round(
                thor["windows"][2]["measured_nj_per_op_vdd_cpu"]
                / (
                    thor["windows"][1]["measured_nj_per_op_vdd_cpu"]
                    - thor["windows"][3]["measured_nj_per_op_vdd_cpu"]
                ),
                2,
            ),
        },
    }
    out_json = art_dir / "lut6_core_baremetal_targets.json"
    out_json.write_text(json.dumps(baremetal_meta, indent=2) + "\n", encoding="utf-8")

    print("=" * 108)
    print(" PHASE 30: LUT6-CORE CROSS-SILICON, BARE-METAL ELF, LIVE ALIF AE3, & JETSON THOR SHUNT MATRIX")
    print("=" * 108)
    print(f"  Thor Physical Shunt (`measured_j`): BitSliced={baremetal_meta['thor_shunt_summary']['bitsliced_batch64_measured_nj']} nJ/op vs INT8 MLP={baremetal_meta['thor_shunt_summary']['int8_mlp_measured_nj']} nJ/op ({baremetal_meta['thor_shunt_summary']['bitsliced_cpj_gain_vs_int8_mlp']}x CpJ!) | Break-even N_crit={baremetal_meta['thor_shunt_summary']['wobble_break_even_calls_n_crit']} calls")
    print(f"  Live Attached Alif E3 (`OpenMV-AE3`): Innate={ae3['innate_lut6_dwt_cycles_per_decision']} cyc ({ae3['speedup_innate_vs_int8_mlp']}x) | BitSliced={ae3['vertical_bitsliced_batch32_dwt_cycles_per_decision']} cyc ({ae3['speedup_bitsliced_vs_int8_mlp']}x vs INT8 MLP!)")
    print(f"  Bare-Metal ELFs: Alif M55 .text={alif_secs.get('.text')} B, .lut6_dtcm={alif_secs.get('.lut6_dtcm')} B | Pico M0+ .text={pico_secs.get('.text')} B, .data.sram_lut6={pico_secs.get('.data.sram_lut6')} B")
    print(f"  Saved -> {out_csv.relative_to(repo_root)}, {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
