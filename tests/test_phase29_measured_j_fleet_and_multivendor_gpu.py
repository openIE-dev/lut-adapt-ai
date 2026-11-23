"""Phase 29 Pytest Suite:
1. Four-Repo Linux `bwrap`-Sandboxed 6-Kernel Lean 4.36.0 Proof Audit (`Claim LA-19` Fleet Upgrade)
2. Six-Host Cross-Silicon + Physical DC Shunt (`measured_j`) Downward Compilation Benchmark (`Claim LA-24`)
3. Four-Vendor Hardware GPU `Burn 0.22.1 / CubeCL` Shader Parity & `reported_j` Benchmark (`Claim LA-25`)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def test_openie_fleet_four_repo_bwrap_sandboxed_six_kernel_audit() -> None:
    csv_p = Path("artifacts/openie_fleet_six_kernel_paranoid_audit.csv")
    json_p = Path("artifacts/openie_fleet_six_kernel_paranoid_audit.json")
    assert csv_p.is_file() and json_p.is_file()

    meta = json.loads(json_p.read_text(encoding="utf-8"))
    assert meta["total_suites_audited"] == 4
    assert meta["total_theorems_and_lemmas"] >= 1390
    assert meta["all_six_kernels_passed_sandboxed"] is True
    assert meta["measured_j"] is None and meta["board_metered"] is False
    assert meta["bwrap_path"] == "/usr/bin/bwrap"

    rows = {r["suite_id"]: r for r in csv.DictReader(csv_p.open(encoding="utf-8"))}
    assert set(rows.keys()) == {"lut-adapt", "communication-avalanche", "time-as-an-index", "rce-spec"}

    for sid, r in rows.items():
        assert r["bwrap_linux_sandbox"] == "True", f"{sid} did not run under Linux bwrap sandbox"
        assert r["kernels_passed"] == "6/6", f"{sid} failed 6-kernel check"
        assert int(r["con_leche_verified_decls"]) > 58000
        assert int(r["con_ron_verified_decls"]) > 58000
        assert int(r["lean4lean_decls"]) > 58000
        assert int(r["custom_axiom_count"]) == 0
        assert int(r["opaque_count"]) == 0
        assert int(r["ofReduceBool_count"]) == 0
        assert "sorryAx" not in r["uses_axioms"], f"{sid} environment depends on sorryAx!"
        assert float(r["paranoid_6kernel_rapl_core_reported_j"]) > 50.0


def test_six_host_measured_j_fleet_and_anti_laundering_invariants() -> None:
    csv_p = Path("artifacts/mol_six_host_measured_j_fleet.csv")
    meta_p = Path("artifacts/mol_six_host_measured_j_fleet_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["claim_id"] == "LA-24"
    assert meta["hosts_evaluated"] == 6
    assert meta["anti_laundering_verified"] is True
    assert meta["board_metered_hosts"] == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    assert len(meta["counter_reported_hosts"]) == 5

    thor_hl = meta["thor_physical_shunt_highlights"]
    assert thor_hl["quiescent_vin_drift_pct"] < 1.0
    assert thor_hl["w2_sub1bit_lut6_vs_mac_cpu_measured_j_ratio"] > 3.0
    assert thor_hl["w3_expert_airmol_vs_dense_cpu_measured_j_ratio"] > 25.0
    assert thor_hl["w5_prop1_compiled_vs_stuck_cpu_measured_j_ratio"] > 9.0
    assert thor_hl["w5_prop1_compiled_vs_stuck_vin_measured_j_ratio"] > 9.0
    assert 0.0 < thor_hl["w5_empirical_break_even_episodes_delta_t_star"] < 1.0

    rows = list(csv.DictReader(csv_p.open(encoding="utf-8")))
    hosts_seen = {r["host_label"] for r in rows}
    assert len(hosts_seen) == 6
    assert {r["os"] for r in rows} == {"linux", "macos", "windows"}
    assert {r["arch"] for r in rows} == {"aarch64", "x86_64"}

    for r in rows:
        is_thor = r["host_label"] == "NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"
        if is_thor:
            assert r["board_metered"] == "True"
            assert float(r["l2_vdd_cpu_soc_mss_measured_j"]) > 0.0
            assert float(r["l1_vin_ina238_measured_j"]) > 0.0
            assert 2.1 <= float(r["cotenant_vdd_gpu_mean_w"]) <= 2.7
            assert r["rapl_pkg_reported_j"] == ""
            assert r["darwin_l3_cpu_reported_j"] == ""
            assert r["win_nvml_gpu_reported_j"] == ""
        else:
            assert r["board_metered"] == "False"
            assert r["l2_vdd_cpu_soc_mss_measured_j"] == ""
            assert r["l1_vin_ina238_measured_j"] == ""

    # Every host must achieve 4/4 positive ABBA pairs on W2, W3, and W5, and exactly 80 zero-clock wakeups on W4b
    for hs in meta["host_summaries"]:
        assert hs["w2_sub1bit_speedup_x"] > 3.0
        assert hs["w3_airmol_speedup_x"] > 25.0
        assert hs["w4b_wakeup_reduction_x"] > 500.0
        assert hs["w5_prop1_speedup_x"] > 9.0

    w4b_ours = [
        r for r in rows
        if r["workload_id"] == "W4b_ZeroClock_Event_vs_1kHz_Poll" and r["arm_role"] == "ours"
    ]
    assert len(w4b_ours) == 6
    for r in w4b_ours:
        assert int(r["units_completed"]) == 80


def test_four_vendor_cubecl_gpu_shader_parity_and_speedup() -> None:
    csv_p = Path("artifacts/cubecl_four_vendor_gpu_parity.csv")
    meta_p = Path("artifacts/cubecl_four_vendor_gpu_parity_meta.json")
    assert csv_p.is_file() and meta_p.is_file()

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    assert meta["claim_id"] == "LA-25"
    assert meta["distinct_gpu_vendors"] == 4
    assert set(meta["vendors"]) == {"Apple", "AMD", "NVIDIA", "Intel"}
    assert meta["max_fp32_parity_error_across_all_4_vendors"] < 1e-4
    assert meta["board_metered"] is False and meta["measured_j"] is None

    rows = list(csv.DictReader(csv_p.open(encoding="utf-8")))
    assert len(rows) == 8
    for r in rows:
        assert r["board_metered"] == "False"
        assert r["measured_j"] == ""
        assert float(r["cubecl_gpu_max_abs_err"]) < 1e-4
        assert float(r["cubecl_gpu_speedup_vs_ref_x"]) > 11.0
        if r["gpu_vendor"] in {"Apple", "AMD", "NVIDIA"}:
            assert float(r["cubecl_gpu_speedup_vs_ref_x"]) > 16.0
    assert max(float(r["cubecl_gpu_speedup_vs_ref_x"]) for r in rows) > 250.0
