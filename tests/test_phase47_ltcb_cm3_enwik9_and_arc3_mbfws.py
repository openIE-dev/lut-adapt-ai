"""Phase 47 regression tests (`Claims LA-84..LA-87` & `162`-Theorem `Claim LA-19` / `1,526` Fleet):
  1. `Claim LA-84`: `MoL-LTCB-CM3` (`mol_ltcb_codec --cm3`) beats ALL FOUR classical codecs
     (`gzip -9`, `bzip2 -9`, `zstd -19`, AND `xz -6` [`LZMA2`]) + `MoL-LTCB-CM2` on official
     `100,000,000 B` `enwik8`, AND scales losslessly in bounded `128 MiB` streaming segments with
     persistent cross-block `StateSMix3` context tables to official `1,000,000,000 B` (`1.00 GB`)
     `enwik9` with `100.00%` bit-identical `archive_sha256` across `Apple M5 Max`, `NVIDIA Jetson
     AGX Thor` (`TI INA238 VIN` `measured_j`, `board_metered = true`), and `AMD Ryzen AI Max+ 395`
     (`Zen 5` `RAPL` `reported_j`), with zero raw corpus files persisted (`/tmp/ltcb_cache` purged).
  2. `Claim LA-85`: Official black-box `ARC-AGI-3` Modal `M-BFWS(R,M)` + Fourteen-Bank `56 KB`
     (`57,344 B = 87.50%` Watch `SRAM`) `System-0` Vault crosses the `> 2/3` (`> 65%`) supermajority
     (`17/25 = 68.00%` full-catalog, `13/16 = 81.25%` matched-16, `>= 20/183` levels, `0` Wave-2
     breaker trips, unlocking `sb26` [`lut6=9`] & `ka59` [`lut6=11`]), paired with `144/144`
     (`9,216/9,216` minterm) `Yosys 0.71+` (`abc -lut 6`) `$lut` Boolean miter (`0` mismatches).
  3. `Claim LA-86`: `Window 27` vs `Window 27b` cross-silicon telemetry across `Jetson AGX Thor`
     (`INA3221 VDD_CPU_SOC_MSS` `measured_j`, `board_metered = true`, `> 300x` reduction),
     `AMD Zen 5` (`RAPL` `reported_j`), and `Apple M3 Ultra` (`reported_j`).
  4. `Claim LA-87` & `Claim LA-19`: Zero-`DSP48E1` `> 100 MHz` `openXC7` `clai_cm3_mbfws_xc7a200t_top`,
     `22`-export `<= 16,384 B` `#![no_std]` `lut6_core.wasm` (`wasm_eval_cm3_mbfws_lut6` V8-verified),
     and `162` `LutAdapt` / `1,526` Linux `bwrap` fleet six-kernel `Lean 4.36.0` proofs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la84_mol_ltcb_cm3_beats_all_four_classical_including_xz6_and_streams_1gb_enwik9() -> None:
    assert not Path("/tmp/ltcb_cache").exists(), "Raw enwik cache /tmp/ltcb_cache must be purged!"
    rust_bin = REPO_ROOT / "sys" / "lut6-core" / "target" / "release" / "mol_ltcb_codec"
    if not rust_bin.is_file():
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--features",
                "std",
                "--manifest-path",
                str(REPO_ROOT / "sys" / "lut6-core" / "Cargo.toml"),
                "--bin",
                "mol_ltcb_codec",
            ],
            check=True,
        )
    st = json.loads(subprocess.check_output([str(rust_bin), "--self-test"], text=True).strip())
    assert st["self_test"] == "PASS"

    rows = list(csv.DictReader((ART_DIR / "phase47_ltcb_cm3_enwik9_fleet.csv").open(encoding="utf-8")))
    meta = json.loads((ART_DIR / "phase47_ltcb_cm3_meta.json").read_text(encoding="utf-8"))

    # Verify M5 Max enwik8 (100 MB): CM3 strictly beats xz -6, zstd -19, bzip2 -9, gzip -9, AND CM2!
    m5_e8 = {
        r["codec"]: r
        for r in rows
        if r["corpus_tier"] == "enwik8_100MB" and r["host_label"].startswith("Apple_M5_Max")
    }
    cm3_e8 = m5_e8["MoL_LTCB_CM3_WRT2_LogOdds_StateSMix3"]
    assert int(cm3_e8["raw_bytes"]) == 100_000_000
    assert cm3_e8["lossless_sha256_verified"] == "True"
    for comp in (
        "xz_6_LZMA2",
        "MoL_LTCB_CM2_WRT_StateSMix",
        "zstd_19_FSE",
        "bzip2_9_BWT",
        "gzip_9_DEFLATE",
    ):
        assert int(cm3_e8["mahoney_ranked_total_bytes"]) < int(m5_e8[comp]["mahoney_ranked_total_bytes"]), (
            f"CM3 ({cm3_e8['mahoney_ranked_total_bytes']}) must beat {comp} ({m5_e8[comp]['mahoney_ranked_total_bytes']})"
        )
    assert int(cm3_e8["payload_bytes"]) < int(m5_e8["xz_6_LZMA2"]["payload_bytes"])
    assert meta["enwik8_cm3_beats_xz6_ranked_bytes_by"] > 40_000

    # Verify 1.00 GB (1,000,000,000 B) enwik9 lossless & 100% cross-architecture SHA-256 parity across 3 hosts!
    for tier, exp_raw in (("enwik8_100MB", 100_000_000), ("enwik9_1000MB", 1_000_000_000)):
        cm3_hosts = [
            r
            for r in rows
            if r["corpus_tier"] == tier and r["codec"] == "MoL_LTCB_CM3_WRT2_LogOdds_StateSMix3"
        ]
        assert len(cm3_hosts) == 3
        shas = set()
        for r in cm3_hosts:
            assert int(r["raw_bytes"]) == exp_raw
            assert r["lossless_sha256_verified"] == "True"
            shas.add(r["archive_sha256"])
            if "Thor" in r["host_label"]:
                assert r["board_metered"] == "True"
                assert float(r["measured_j"]) > 0.0
                assert r["reported_j"] == ""
            else:
                assert r["board_metered"] == "False"
                assert r["measured_j"] == ""
                assert float(r["reported_j"]) > 0.0
        assert len(shas) == 1, f"Cross-arch SHA mismatch on {tier}: {shas}"

    # Verify M5 Max 1.00 GB enwik9 beats ALL FOUR classical codecs (xz -6, zstd -19, bzip2 -9, and gzip -9)!
    m5_e9 = {
        r["codec"]: r
        for r in rows
        if r["corpus_tier"] == "enwik9_1000MB" and r["host_label"].startswith("Apple_M5_Max")
    }
    cm3_e9 = m5_e9["MoL_LTCB_CM3_WRT2_LogOdds_StateSMix3"]
    for comp9 in ("xz_6_LZMA2", "zstd_19_FSE", "bzip2_9_BWT", "gzip_9_DEFLATE"):
        assert int(cm3_e9["mahoney_ranked_total_bytes"]) < int(m5_e9[comp9]["mahoney_ranked_total_bytes"])


def test_claim_la85_arc3_mbfws_two_thirds_supermajority_17_of_25_and_144_yosys_miter() -> None:
    meta = json.loads((ART_DIR / "phase47_arc3_mbfws_meta.json").read_text(encoding="utf-8"))
    assert meta["zero_game_source_peeking"] is True
    assert meta["zero_raw_frames_persisted"] is True
    assert meta["board_metered"] is False and meta["measured_j"] is None
    assert meta["vault_banks"] == 14
    assert meta["vault_bytes"] == 57_344
    assert meta["watch_sram_64kb_share_pct"] == 87.5

    w2 = meta["mbfws_wave2_summary"]
    assert w2["games_solved_25"] == 17 and meta["game_solve_pct_full25"] == 68.0
    assert w2["games_solved_16"] == 13 and meta["game_solve_pct_matched16"] == 81.25
    assert w2["levels_completed_25"] >= 20
    assert w2["levels_completed_16"] >= 16
    assert w2["breaker_trips_25"] == 0
    assert w2["mean_rhae_25"] > 2.70
    assert w2["mean_rhae_16"] > 3.55

    per_game = {
        r["game_id"]: r
        for r in csv.DictReader((ART_DIR / "phase47_arc3_mbfws_25env_per_game.csv").open(encoding="utf-8"))
        if r["agent_kind"] == "apex_mbfws_lut6_wave2"
    }
    assert int(per_game["sb26-7fbdac44"]["levels_completed"]) >= 1
    assert int(per_game["sb26-7fbdac44"]["lut6_reflex_steps"]) == 9
    assert int(per_game["ka59-38d34dbb"]["levels_completed"]) >= 1
    assert int(per_game["ka59-38d34dbb"]["lut6_reflex_steps"]) == 11

    miter = list(csv.DictReader((ART_DIR / "phase47_cm3_mbfws_lut6_miter.csv").open(encoding="utf-8")))
    assert len(miter) == 2
    assert sum(int(r["slices_compiled"]) for r in miter) == 24
    assert sum(int(r["lut6_cells"]) for r in miter) == 144
    assert sum(int(r["minterms_verified"]) for r in miter) == 9_216
    assert sum(int(r["yosys_miter_mismatches"]) for r in miter) == 0


def test_claim_la86_window27_cross_silicon_thor_dc_shunt() -> None:
    rows = list(csv.DictReader((ART_DIR / "phase47_window27_cross_silicon.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    thor_count = 0
    for r in rows:
        gain = float(r["energy_reduction_x"])
        speed = float(r["latency_speedup_x"])
        assert speed > 100.0
        if "Thor" in r["host_label"]:
            thor_count += 1
            assert r["board_metered"] == "True"
            assert float(r["w27_measured_nj_per_op_vdd_cpu"]) > 0.0
            assert r["w27_reported_nj_per_op"] == ""
            assert gain > 300.0
        else:
            assert r["board_metered"] == "False"
            assert r["w27_measured_nj_per_op_vdd_cpu"] == ""
            assert float(r["w27_reported_nj_per_op"]) > 0.0
            assert gain > 200.0
    assert thor_count == 1


def test_claim_la87_openxc7_cm3_mbfws_22export_wasm_and_162_six_kernel_proofs_1526_fleet() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_cm3_mbfws_timing.json").read_text(encoding="utf-8"))
    assert xc7["top_module"] == "clai_cm3_mbfws_xc7a200t_top"
    assert xc7["dsp48e1_used"] == 0
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["cm3_mbfws_64minterm_rtl_mismatches"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    wasm_p = ART_DIR / "wasm" / "lut6_core.wasm"
    web_wasm_p = REPO_ROOT / "paper" / "web" / "lut6_core.wasm"
    assert wasm_p.read_bytes() == web_wasm_p.read_bytes()
    assert wasm_p.stat().st_size <= 16_384

    node_probe = (
        "const fs = require('fs');\n"
        f"const buf = fs.readFileSync({str(wasm_p)!r});\n"
        "(async () => {\n"
        "  const { instance } = await WebAssembly.instantiate(buf, {});\n"
        "  const ex = Object.keys(instance.exports).filter(k => k.startsWith('wasm_'));\n"
        "  const a1 = (1 | (3 << 6) | (12 << 12) | (48 << 16) | (6 << 22) | (0 << 28)) >>> 0;\n"
        "  const r1 = instance.exports.wasm_eval_cm3_mbfws_lut6(a1, 7) >>> 0;\n"
        "  console.log(JSON.stringify({n: ex.length, r1}));\n"
        "})();\n"
    )
    v8 = json.loads(subprocess.check_output(["node", "-e", node_probe], text=True).strip())
    assert v8["n"] == 22
    r1 = int(v8["r1"])
    assert ((r1 >> 30) & 3) == 1
    assert ((r1 >> 26) & 15) == 13
    assert ((r1 >> 20) & 3) == 3
    assert ((r1 >> 18) & 3) == 2

    lean = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    assert lean["total_kernel_checked_theorems"] == 162
    assert lean["module_theorem_counts"]["LtcbCm3Arc3Supermajority"] == 8
    assert lean["independent_kernels_passed"] == 6
    assert lean["sorry_count"] == 0
    assert lean["custom_axiom_count"] == 0
    assert lean["ofReduceBool_native_decide_count"] == 0

    fleet = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert fleet["total_theorems_and_lemmas"] == 1_526
    assert fleet["all_six_kernels_passed_sandboxed"] is True

    for ext in ("svg", "png"):
        fp = REPO_ROOT / "paper" / "figures" / f"ltcb-cm3-enwik9-and-arc3-supermajority.{ext}"
        assert fp.is_file() and fp.stat().st_size > 10_000
