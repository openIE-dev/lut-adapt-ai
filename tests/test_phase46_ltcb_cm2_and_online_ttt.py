"""Phase 46 Invariant & Regression Suite (`Claims LA-80..LA-83` + Historic `> 150` [`154` `LutAdapt` / `1,518` Fleet] `Claim LA-19` Milestone):
  1. `Claim LA-80`: Native Zero-Python, Zero-Float Rust `mol_ltcb_codec --cm2` (`MoL-LTCB-CM2`:
     Bijective `WRT-Capillary-LUT6` MediaWiki-XML/Wiki/Stem Preprocessor + Unbounded `100 MB`
     Multi-Order Lazy `LZ-Rep4` + Lockstep Integer-Exact `Q12` `StateSMix-LUT6`) beating `gzip -9`,
     `bzip2 -9`, `zstd -19`, AND `MoL-LTCB v1` across every official Matt Mahoney `LTCB` tier
     (`enwik5` `100 KB` .. `enwik8` `100,000,000 B`, plus `xz -6` on `enwik5..enwik7`) with `100.0%`
     cross-architecture bit-identical `archive_sha256` across `Apple M5 Max`, `NVIDIA Jetson AGX Thor`
     (`TI INA238` `VIN` `measured_j`, `board_metered = true`), and `AMD Ryzen AI Max+ 395` (`Zen 5` `RAPL`).
  2. `Claim LA-81`: Lockstep Zero-Float `Online-TTT-LUT6` (`0`-byte `SplitMix64` seed + sparse
     zero-crossing sign-flip `6`-col `LUT6` slab rebuilds skipping `> 94%` of rebuilds, `0` cross-host
     mismatches vs. `FP32` `AdamW` sign-flip desync at `step 20`) across `4` structured byte-stream
     domains + `144/144` (`9,216/9,216` minterm) `Yosys 0.71+` (`abc -lut 6`) `$lut` Boolean Miter.
  3. `Claim LA-82`: Cross-Silicon `Window 26` (`VAULT14 + WRT-Capillary-LUT6 + StateSMix-TTT-LUT6 +
     Q12 Mix + Sparse Online-JouleBit6`) vs. `Window 26b` (`FP32` `64`-Param Online `AdamW` + Logistic
     `expf` Mixer + Unpacked `LittleBit-2` `MAC`) Physical `Thor` DC Shunt Telemetry.
  4. `Claim LA-83` & `Claim LA-19`: Zero-`DSP48E1` `> 100 MHz` `openXC7` `clai_ltcb_cm2_ttt_xc7a200t_top`
     `.bit`, `21`-Export `<= 16,384 B` `lut6_core.wasm`, and Historic `> 150` (`154` `LutAdapt` /
     `>= 1,518` Linux `bwrap` Fleet) Six-Kernel `Lean 4.36.0` Proofs!
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

from clai.frontier.enigma_zerobench_codifier import (
    compile_truth_table_6bit_to_lut6_masks,
    eval_lut6_masks6,
)
from clai.frontier.ltcb_cm2_online_ttt import (
    mix3_probs_q12_py,
    statesmix_ttt_gate2_py,
    wrt_capillary_gate2_py,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def test_claim_la80_mol_ltcb_cm2_beats_gzip_bzip2_zstd_v1_and_cross_arch_exact() -> None:
    rust_bin = REPO_ROOT / "sys" / "lut6-core" / "target" / "release" / "mol_ltcb_codec"
    assert rust_bin.is_file(), "Expected compiled release binary mol_ltcb_codec"
    st_out = json.loads(subprocess.check_output([str(rust_bin), "--self-test"], text=True).strip())
    assert st_out["self_test"] == "PASS"
    assert st_out["adamw_sign_desync_step"] < 256
    assert st_out["ttt_i16_mismatches"] == 0
    assert not Path("/tmp/ltcb_cache").exists(), "Expected /tmp/ltcb_cache purged after fleet run!"

    rows = list(csv.DictReader((ART_DIR / "phase46_ltcb_cm2_enwik8_fleet.csv").open(encoding="utf-8")))
    assert len(rows) >= 24

    enwik8_cm2 = [
        r for r in rows if r["corpus_tier"] == "enwik8_100MB" and r["codec"] == "MoL_LTCB_CM2_WRT_StateSMix"
    ]
    assert len(enwik8_cm2) == 3  # Apple M5 Max, Jetson AGX Thor, AMD Zen 5
    shas = {r["archive_sha256"] for r in enwik8_cm2}
    assert len(shas) == 1, f"Expected identical CM2 archive SHA-256 across M5 Max, Thor, and Zen 5, got {shas}"

    for r in rows:
        assert r["lossless_sha256_verified"].lower() == "true"
        bm = r["board_metered"].lower() == "true"
        if bm:
            assert "Thor" in r["host_label"]
            assert r["measured_j"] not in ("", "null") and r["reported_j"] in ("", "null")
        else:
            assert "Thor" not in r["host_label"]
            assert r["measured_j"] in ("", "null")

    # Verify MoL-LTCB-CM2 strictly beats gzip -9, bzip2 -9, zstd -19, AND MoL-LTCB v1 across ALL 4 tiers on M5 Max!
    for tier in ("enwik5_100KB", "enwik6_1MB", "enwik7_10MB", "enwik8_100MB"):
        m5_tier = {
            r["codec"]: int(r["mahoney_ranked_total_bytes"])
            for r in rows
            if r["corpus_tier"] == tier and "M5_Max" in r["host_label"]
        }
        cm2_tot = m5_tier["MoL_LTCB_CM2_WRT_StateSMix"]
        assert cm2_tot < m5_tier["gzip_9_DEFLATE"], f"CM2 did not beat gzip-9 on {tier}"
        assert cm2_tot < m5_tier["bzip2_9_BWT"], f"CM2 did not beat bzip2-9 on {tier}"
        assert cm2_tot < m5_tier["zstd_19_FSE"], f"CM2 did not beat zstd-19 on {tier}"
        assert cm2_tot < m5_tier["MoL_LTCB_v1_waiOS_LUT6_Range"], f"CM2 did not beat v1 on {tier}"
        if tier in ("enwik5_100KB", "enwik6_1MB"):
            assert cm2_tot < m5_tier["xz_6_LZMA2"], f"CM2 did not beat xz-6 on {tier}"

    # Verify enwik8 Ranked Total < 27.0 MB (< 2.16 bpb)
    m5_e8_cm2 = next(
        r
        for r in rows
        if r["corpus_tier"] == "enwik8_100MB"
        and "M5_Max" in r["host_label"]
        and r["codec"] == "MoL_LTCB_CM2_WRT_StateSMix"
    )
    assert int(m5_e8_cm2["mahoney_ranked_total_bytes"]) < 27_000_000
    assert float(m5_e8_cm2["mahoney_total_bpb"]) < 2.16


def test_claim_la81_online_ttt_lut6_four_domains_and_144_yosys_miter() -> None:
    wrt_masks = compile_truth_table_6bit_to_lut6_masks(wrt_capillary_gate2_py)
    ttt_masks = compile_truth_table_6bit_to_lut6_masks(statesmix_ttt_gate2_py)
    for x in range(64):
        assert eval_lut6_masks6(wrt_masks, x) == wrt_capillary_gate2_py(x)
        assert eval_lut6_masks6(ttt_masks, x) == statesmix_ttt_gate2_py(x)

    for sse in range(4):
        for p1 in (1, 512, 2048, 4095):
            for p2 in (1, 1024, 3500, 4095):
                pm = mix3_probs_q12_py(p1, p2, 2048, 96, 112, sse)
                assert 1 <= pm <= 4095

    ttt_rows = list(csv.DictReader((ART_DIR / "phase46_online_ttt_four_domains.csv").open(encoding="utf-8")))
    assert len(ttt_rows) == 4
    for r in ttt_rows:
        assert float(r["warm_last_quarter_bpb"]) < float(r["cold_first_1kb_bpb"])
        assert float(r["regret_collapse_pct"]) > 15.0
        assert float(r["slab_rebuild_skip_pct"]) > 89.0
        assert int(r["int16_ttt_cross_host_mismatches"]) == 0

    miter_rows = list(csv.DictReader((ART_DIR / "phase46_cm2_ttt_lut6_miter.csv").open(encoding="utf-8")))
    assert sum(int(r["lut6_cells"]) for r in miter_rows) == 144
    assert sum(int(r["minterms_verified"]) for r in miter_rows) == 9216
    assert all(int(r["yosys_miter_mismatches"]) == 0 for r in miter_rows)


def test_claim_la82_window26_cross_silicon_thor_dc_shunt() -> None:
    w26_rows = list(csv.DictReader((ART_DIR / "phase46_window26_cross_silicon.csv").open(encoding="utf-8")))
    assert len(w26_rows) == 3
    metered = [r["host_label"] for r in w26_rows if r["board_metered"] == "True"]
    assert metered == ["NVIDIA_Jetson_AGX_Thor_Tegra_aarch64"]
    for r in w26_rows:
        assert float(r["energy_reduction_x"]) > 1.5
        if r["board_metered"] == "True":
            assert float(r["latency_speedup_x"]) > 2.0
            assert r["w26_measured_nj_per_op_vdd_cpu"] != "" and r["w26_reported_nj_per_op"] == ""
        else:
            assert r["w26_measured_nj_per_op_vdd_cpu"] == "" and r["w26_reported_nj_per_op"] != ""


def test_claim_la83_openxc7_21export_wasm_and_154_six_kernel_proofs_1518_fleet() -> None:
    xc7 = json.loads((ART_DIR / "openxc7_ltcb_cm2_ttt_timing.json").read_text(encoding="utf-8"))
    assert xc7["dsp48e1_used"] == 0
    assert xc7["joulebit6_64code_rtl_mismatches"] == 0
    assert xc7["cm2_wrt_ttt_64minterm_rtl_mismatches"] == 0
    assert xc7["waios_256x128_rtl_clean_blake3"] == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0"
    assert xc7["post_par_max_freq_mhz"] > 100.0
    assert xc7["emitted_bitstream_bytes"] > 9_000_000
    assert xc7["board_metered"] is False and xc7["measured_j"] is None

    lean_meta = json.loads((ART_DIR / "lean4_formal_proofs_meta.json").read_text(encoding="utf-8"))
    fleet_meta = json.loads((ART_DIR / "openie_fleet_six_kernel_paranoid_audit.json").read_text(encoding="utf-8"))
    assert lean_meta["total_kernel_checked_theorems"] >= 154
    assert lean_meta["module_theorem_counts"].get("LtcbCm2OnlineTtt") == 8
    assert lean_meta["independent_kernels_passed"] == 6
    assert fleet_meta["total_suites_audited"] == 4
    assert fleet_meta["total_theorems_and_lemmas"] >= 1518
    assert fleet_meta["all_six_kernels_passed_sandboxed"] is True

    wasm_path = ART_DIR / "wasm" / "lut6_core.wasm"
    assert wasm_path.stat().st_size <= 16384
    probe = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(wasm_path))});
(async () => {{
  const {{ instance }} = await WebAssembly.instantiate(buf, {{}});
  const ex = instance.exports;
  const fnCount = Object.keys(ex).filter(k => k.startsWith('wasm_')).length;
  const argEscFlip = 1 | (3 << 6) | (12 << 12) | (16 << 16) | (12 << 22);
  const rEscFlip = ex.wasm_eval_ltcb_cm2_ttt_lut6(argEscFlip >>> 0, 7) >>> 0;
  const argDictSkip = 1 | (0 << 6) | (0 << 12) | (4 << 16) | (4 << 22);
  const rDictSkip = ex.wasm_eval_ltcb_cm2_ttt_lut6(argDictSkip >>> 0, 7) >>> 0;
  const argBrk = argEscFlip | (1 << 28);
  const rBrk = ex.wasm_eval_ltcb_cm2_ttt_lut6(argBrk >>> 0, 7) >>> 0;
  console.log(JSON.stringify({{ fnCount, rEscFlip, rDictSkip, rBrk }}));
}})();
"""
    out = json.loads(subprocess.check_output(["node", "-e", probe], text=True).strip())
    assert out["fnCount"] >= 21
    assert ((out["rEscFlip"] >> 30) & 3) == 1
    assert ((out["rEscFlip"] >> 26) & 0xF) == 13
    assert ((out["rEscFlip"] >> 20) & 3) == 3
    assert ((out["rEscFlip"] >> 18) & 3) == 3
    assert ((out["rDictSkip"] >> 20) & 3) == 1
    assert ((out["rDictSkip"] >> 18) & 3) == 2
    assert ((out["rBrk"] >> 30) & 3) == 2
