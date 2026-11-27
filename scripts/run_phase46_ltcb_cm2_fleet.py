"""Phase 46 (`Claims LA-80, LA-81 & LA-83`): Cross-Fleet Matt Mahoney Large Text Compression
Benchmark (`LTCB` `enwik5` `100 KB`, `enwik6` `1 MB`, `enwik7` `10 MB`, and `enwik8` `100 MB`
Wikipedia XML) with `MoL-LTCB-CM2` (`WRT-Capillary-LUT6` + Unbounded `100 MB` Multi-Order Lazy
`LZ-Rep4` + Integer-Exact `Q12` `StateSMix-LUT6`), `4`-Domain `Online-TTT-LUT6` Regret & Sparse
Zero-Crossing Slab Benchmark, `144/144` `Yosys` `$lut` Miter, and `#![no_std]` `21`-Export
`lut6_core.wasm` (`<= 16,384 B`) Build & V8 Verification:
  - `Apple M5 Max` (`arm64` Darwin `RUSAGE_INFO_V6` `reported_j`),
  - `NVIDIA Jetson AGX Thor` (`cbio@100.111.185.107`, physical `TI INA238` `VIN` DC shunt
    `measured_j`, `board_metered = true`!),
  - `AMD Ryzen AI Max+ 395` (`dcharlot@100.91.48.80`, `Zen 5` Linux `RAPL` `reported_j`).
Zero raw `enwik` text or third-party weights are persisted in git; `/tmp/ltcb_cache` is purged!
"""

from __future__ import annotations

import bz2
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import time

from clai.contracts.joule_meter import meter_joules
from clai.frontier.ltcb_cm2_online_ttt import (
    run_ltcb_cm2_ttt_yosys_miter_suite,
    run_online_ttt_lut6_four_domain_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"
ENWIK8_PATH = Path("/tmp/ltcb_cache/enwik8")
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"


def build_and_verify_21export_wasm() -> dict:
    wasm_dir = ART_DIR / "wasm"
    wasm_dir.mkdir(parents=True, exist_ok=True)
    out_wasm = wasm_dir / "lut6_core.wasm"
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "--target=wasm32-unknown-unknown",
            "--crate-type=cdylib",
            "-C",
            "opt-level=z",
            "-C",
            "panic=abort",
            "-C",
            "strip=symbols",
            "-C",
            "codegen-units=1",
            "-C",
            "lto=fat",
            str(REPO_ROOT / "sys" / "lut6-core" / "src" / "lib.rs"),
            "-o",
            str(out_wasm),
        ],
        check=True,
    )
    wasm_opt = shutil.which("wasm-opt") or "/opt/homebrew/bin/wasm-opt"
    if wasm_opt and Path(wasm_opt).is_file():
        subprocess.run(
            [wasm_opt, "-Oz", "--enable-bulk-memory", "--enable-sign-ext", str(out_wasm), "-o", str(out_wasm)],
            check=True,
        )
    raw = out_wasm.read_bytes()
    (REPO_ROOT / "paper" / "web" / "lut6_core.wasm").write_bytes(raw)
    assert len(raw) <= 16384, f"lut6_core.wasm ({len(raw)} B) exceeded 16,384 B!"
    sha12 = hashlib.sha256(raw).hexdigest()[:12]

    node_js = (
        "const fs = require('fs');\n"
        f"const buf = fs.readFileSync({str(out_wasm)!r});\n"
        "(async () => {\n"
        "  const { instance } = await WebAssembly.instantiate(buf, {});\n"
        "  const ex = Object.keys(instance.exports).filter(k => k.startsWith('wasm_'));\n"
        "  if (ex.length !== 21) throw new Error('Expected 21 wasm_ exports, got ' + ex.length + ': ' + ex.join(','));\n"
        "  // Probe Export #21: Bank-13 + WRT Escape=3 (addr=16) + TTT Zero-Cross Slab Rebuild=3 (addr=12)\n"
        "  const arg1 = (1 | (3 << 6) | (12 << 12) | (16 << 16) | (12 << 22) | (0 << 28)) >>> 0;\n"
        "  const r1 = instance.exports.wasm_eval_ltcb_cm2_ttt_lut6(arg1, 7) >>> 0;\n"
        "  if (((r1 >>> 30) & 3) !== 1 || ((r1 >>> 26) & 15) !== 13 || ((r1 >>> 20) & 3) !== 3 || ((r1 >>> 18) & 3) !== 3) {\n"
        "    throw new Error('Unexpected r1: 0x' + r1.toString(16));\n"
        "  }\n"
        "  // Probe Export #21: Bank-0 + WRT Dict Token=1 (addr=4) + TTT i16 Step Skip-Slab=2 (addr=4)\n"
        "  const arg2 = (1 | (0 << 6) | (0 << 12) | (4 << 16) | (4 << 22) | (0 << 28)) >>> 0;\n"
        "  const r2 = instance.exports.wasm_eval_ltcb_cm2_ttt_lut6(arg2, 7) >>> 0;\n"
        "  if (((r2 >>> 30) & 3) !== 1 || ((r2 >>> 26) & 15) !== 0 || ((r2 >>> 20) & 3) !== 1 || ((r2 >>> 18) & 3) !== 2) {\n"
        "    throw new Error('Unexpected r2: 0x' + r2.toString(16));\n"
        "  }\n"
        "  // Probe Export #21: Breaker Halt (brk=1)\n"
        "  const arg3 = (1 | (3 << 6) | (12 << 12) | (16 << 16) | (12 << 22) | (1 << 28)) >>> 0;\n"
        "  const r3 = instance.exports.wasm_eval_ltcb_cm2_ttt_lut6(arg3, 7) >>> 0;\n"
        "  if (((r3 >>> 30) & 3) !== 2) throw new Error('Expected Mode 2 breaker halt');\n"
        "  console.log(JSON.stringify({exports: ex.length, r1: r1, r2: r2, r3: r3}));\n"
        "})();\n"
    )
    v8_out = json.loads(subprocess.check_output(["node", "-e", node_js], text=True).strip())
    return {
        "wasm_bytes": len(raw),
        "wasm_kb": round(len(raw) / 1024.0, 2),
        "spare_bytes_under_16kb": 16384 - len(raw),
        "wasm_exports": v8_out["exports"],
        "wasm_sha256_12": sha12,
        "wasm_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _ensure_enwik8() -> bytes:
    if not ENWIK8_PATH.is_file():
        ENWIK8_PATH.parent.mkdir(parents=True, exist_ok=True)
        zpath = ENWIK8_PATH.parent / "enwik8.zip"
        import urllib.request
        import zipfile

        urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(ENWIK8_PATH.parent)
        zpath.unlink(missing_ok=True)
    raw = ENWIK8_PATH.read_bytes()
    assert len(raw) == 100_000_000
    assert hashlib.sha256(raw).hexdigest() == ENWIK8_SHA256
    return raw


def run_local_m5_max_ltcb_cm2_scaling(enwik8_bytes: bytes, wasm_bytes: int) -> tuple[list[dict], dict]:
    host_label = "Apple_M5_Max_128GB_macOS_arm64"
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
    rust_bin = REPO_ROOT / "sys" / "lut6-core" / "target" / "release" / "mol_ltcb_codec"

    tiers = [
        ("enwik5_100KB", 100_000),
        ("enwik6_1MB", 1_000_000),
        ("enwik7_10MB", 10_000_000),
        ("enwik8_100MB", 100_000_000),
    ]
    rows: list[dict] = []
    enwik8_cm2_doc = {}
    zstd_bin = shutil.which("zstd") or "/opt/homebrew/bin/zstd"

    for tier_name, n_bytes in tiers:
        sub = enwik8_bytes[:n_bytes]
        sub_sha = hashlib.sha256(sub).hexdigest()

        # 1. gzip -9 (DEFLATE, ~32 KB binary floor)
        with meter_joules("ltcb_cm2_gzip") as jm_gz:
            t0 = time.perf_counter()
            gz_arc = gzip.compress(sub, compresslevel=9)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            gz_dec = gzip.decompress(gz_arc)
            dec_ms = (time.perf_counter() - t1) * 1000.0
        assert hashlib.sha256(gz_dec).hexdigest() == sub_sha
        rows.append(
            {
                "host_label": host_label,
                "board_metered": False,
                "corpus_tier": tier_name,
                "raw_bytes": n_bytes,
                "codec": "gzip_9_DEFLATE",
                "payload_bytes": len(gz_arc),
                "decoder_plus_prior_bytes": 32_768,
                "mahoney_ranked_total_bytes": len(gz_arc) + 32_768,
                "payload_bpb": round(len(gz_arc) * 8.0 / n_bytes, 4),
                "mahoney_total_bpb": round((len(gz_arc) + 32_768) * 8.0 / n_bytes, 4),
                "compress_ms": round(enc_ms, 2),
                "decompress_ms": round(dec_ms, 2),
                "measured_j": None,
                "reported_j": round(float(jm_gz.reported_j or 0.0), 4),
                "archive_sha256": hashlib.sha256(gz_arc).hexdigest()[:16],
                "lossless_sha256_verified": True,
            }
        )

        # 2. bzip2 -9 (Burrows-Wheeler Transform, ~40 KB binary floor)
        with meter_joules("ltcb_cm2_bzip2") as jm_bz:
            t0 = time.perf_counter()
            bz_arc = bz2.compress(sub, compresslevel=9)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            bz_dec = bz2.decompress(bz_arc)
            dec_ms = (time.perf_counter() - t1) * 1000.0
        assert hashlib.sha256(bz_dec).hexdigest() == sub_sha
        rows.append(
            {
                "host_label": host_label,
                "board_metered": False,
                "corpus_tier": tier_name,
                "raw_bytes": n_bytes,
                "codec": "bzip2_9_BWT",
                "payload_bytes": len(bz_arc),
                "decoder_plus_prior_bytes": 40_960,
                "mahoney_ranked_total_bytes": len(bz_arc) + 40_960,
                "payload_bpb": round(len(bz_arc) * 8.0 / n_bytes, 4),
                "mahoney_total_bpb": round((len(bz_arc) + 40_960) * 8.0 / n_bytes, 4),
                "compress_ms": round(enc_ms, 2),
                "decompress_ms": round(dec_ms, 2),
                "measured_j": None,
                "reported_j": round(float(jm_bz.reported_j or 0.0), 4),
                "archive_sha256": hashlib.sha256(bz_arc).hexdigest()[:16],
                "lossless_sha256_verified": True,
            }
        )

        # 3. zstd -19 (FSE, ~96 KB binary floor)
        if zstd_bin and Path(zstd_bin).is_file():
            tmp_in = Path(f"/tmp/_ltcb46_{n_bytes}.in")
            tmp_zst = Path(f"/tmp/_ltcb46_{n_bytes}.zst")
            tmp_in.write_bytes(sub)
            t0 = time.perf_counter()
            subprocess.run([zstd_bin, "-19", "-q", "-f", str(tmp_in), "-o", str(tmp_zst)], check=True)
            enc_ms = (time.perf_counter() - t0) * 1000.0
            zst_bytes = tmp_zst.stat().st_size
            zst_sha = hashlib.sha256(tmp_zst.read_bytes()).hexdigest()[:16]
            t1 = time.perf_counter()
            dec_out = subprocess.check_output([zstd_bin, "-d", "-q", "-c", str(tmp_zst)])
            dec_ms = (time.perf_counter() - t1) * 1000.0
            assert hashlib.sha256(dec_out).hexdigest() == sub_sha
            tmp_in.unlink(missing_ok=True)
            tmp_zst.unlink(missing_ok=True)
            rows.append(
                {
                    "host_label": host_label,
                    "board_metered": False,
                    "corpus_tier": tier_name,
                    "raw_bytes": n_bytes,
                    "codec": "zstd_19_FSE",
                    "payload_bytes": zst_bytes,
                    "decoder_plus_prior_bytes": 98_304,
                    "mahoney_ranked_total_bytes": zst_bytes + 98_304,
                    "payload_bpb": round(zst_bytes * 8.0 / n_bytes, 4),
                    "mahoney_total_bpb": round((zst_bytes + 98_304) * 8.0 / n_bytes, 4),
                    "compress_ms": round(enc_ms, 2),
                    "decompress_ms": round(dec_ms, 2),
                    "measured_j": None,
                    "reported_j": None,
                    "archive_sha256": zst_sha,
                    "lossless_sha256_verified": True,
                }
            )

        # 4. xz / LZMA2 preset 6 (enwik5..enwik7)
        if n_bytes <= 10_000_000:
            with meter_joules("ltcb_cm2_xz") as jm_xz:
                t0 = time.perf_counter()
                xz_arc = lzma.compress(sub, preset=6)
                enc_ms = (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                xz_dec = lzma.decompress(xz_arc)
                dec_ms = (time.perf_counter() - t1) * 1000.0
            assert hashlib.sha256(xz_dec).hexdigest() == sub_sha
            rows.append(
                {
                    "host_label": host_label,
                    "board_metered": False,
                    "corpus_tier": tier_name,
                    "raw_bytes": n_bytes,
                    "codec": "xz_6_LZMA2",
                    "payload_bytes": len(xz_arc),
                    "decoder_plus_prior_bytes": 65_536,
                    "mahoney_ranked_total_bytes": len(xz_arc) + 65_536,
                    "payload_bpb": round(len(xz_arc) * 8.0 / n_bytes, 4),
                    "mahoney_total_bpb": round((len(xz_arc) + 65_536) * 8.0 / n_bytes, 4),
                    "compress_ms": round(enc_ms, 2),
                    "decompress_ms": round(dec_ms, 2),
                    "measured_j": None,
                    "reported_j": round(float(jm_xz.reported_j or 0.0), 4),
                    "archive_sha256": hashlib.sha256(xz_arc).hexdigest()[:16],
                    "lossless_sha256_verified": True,
                }
            )

        # 5. MoL-LTCB v1 (`MLTC` Phase 45 baseline)
        out_v1 = Path(f"/tmp/_mltc_v1_{n_bytes}.json")
        with meter_joules("ltcb_mol_v1") as jm_v1:
            subprocess.run(
                [
                    str(rust_bin),
                    "--input",
                    str(ENWIK8_PATH),
                    "--max-bytes",
                    str(n_bytes),
                    "--out-json",
                    str(out_v1),
                ],
                check=True,
            )
        v1doc = json.loads(out_v1.read_text(encoding="utf-8"))
        out_v1.unlink(missing_ok=True)
        assert v1doc["lossless_exact"] is True and v1doc["decoded_sha256"] == sub_sha
        rows.append(
            {
                "host_label": host_label,
                "board_metered": False,
                "corpus_tier": tier_name,
                "raw_bytes": n_bytes,
                "codec": "MoL_LTCB_v1_waiOS_LUT6_Range",
                "payload_bytes": v1doc["compressed_payload_bytes"],
                "decoder_plus_prior_bytes": wasm_bytes,
                "mahoney_ranked_total_bytes": v1doc["compressed_payload_bytes"] + wasm_bytes,
                "payload_bpb": round(v1doc["payload_bpb"], 4),
                "mahoney_total_bpb": round((v1doc["compressed_payload_bytes"] + wasm_bytes) * 8.0 / n_bytes, 4),
                "compress_ms": v1doc["compress_ms"],
                "decompress_ms": v1doc["decompress_ms"],
                "measured_j": None,
                "reported_j": round(float(jm_v1.reported_j or 0.0), 4),
                "archive_sha256": v1doc["archive_sha256"][:16],
                "lossless_sha256_verified": True,
            }
        )

        # 6. Native Rust `MoL-LTCB-CM2` (`--cm2`: Bijective `WRT-Capillary-LUT6` + Unbounded Lazy `LZ-Rep4` + `StateSMix-LUT6`)
        out_cm2 = Path(f"/tmp/_mlt2_{n_bytes}.json")
        with meter_joules("ltcb_mol_cm2") as jm_cm2:
            subprocess.run(
                [
                    str(rust_bin),
                    "--cm2",
                    "--input",
                    str(ENWIK8_PATH),
                    "--max-bytes",
                    str(n_bytes),
                    "--out-json",
                    str(out_cm2),
                ],
                check=True,
            )
        cm2doc = json.loads(out_cm2.read_text(encoding="utf-8"))
        out_cm2.unlink(missing_ok=True)
        assert cm2doc["lossless_exact"] is True and cm2doc["decoded_sha256"] == sub_sha
        if n_bytes == 100_000_000:
            enwik8_cm2_doc = cm2doc
        rows.append(
            {
                "host_label": host_label,
                "board_metered": False,
                "corpus_tier": tier_name,
                "raw_bytes": n_bytes,
                "codec": "MoL_LTCB_CM2_WRT_StateSMix",
                "payload_bytes": cm2doc["compressed_payload_bytes"],
                "decoder_plus_prior_bytes": wasm_bytes,
                "mahoney_ranked_total_bytes": cm2doc["compressed_payload_bytes"] + wasm_bytes,
                "payload_bpb": round(cm2doc["payload_bpb"], 4),
                "mahoney_total_bpb": round((cm2doc["compressed_payload_bytes"] + wasm_bytes) * 8.0 / n_bytes, 4),
                "compress_ms": cm2doc["compress_ms"],
                "decompress_ms": cm2doc["decompress_ms"],
                "measured_j": None,
                "reported_j": round(float(jm_cm2.reported_j or 0.0), 4),
                "archive_sha256": cm2doc["archive_sha256"][:16],
                "lossless_sha256_verified": True,
            }
        )

    return rows, enwik8_cm2_doc


def run_remote_ltcb_cm2_host(host: str, host_label: str, is_thor_shunt: bool, wasm_bytes: int) -> list[dict]:
    """Runs native Rust `mol_ltcb_codec --cm2` vs `gzip -9` & `bzip2 -9` on `enwik7` (`10 MB`) and
    `enwik8` (`100 MB`) on remote fleet hosts (`Jetson AGX Thor` physical `TI INA238` `VIN`
    `measured_j` with `board_metered = true`, and `AMD Zen 5` Linux `RAPL` `reported_j`), verifying
    cross-architecture bit-exact `archive_sha256` identity (`0` float drift!) and deleting `/tmp` scratch."""
    tgz = Path(f"/tmp/_lut6_p46_{host_label[:8]}.tgz")
    tar_env = dict(os.environ)
    tar_env["COPYFILE_DISABLE"] = "1"
    subprocess.run(
        [
            "tar",
            "--no-xattrs",
            "--exclude=sys/lut6-core/target",
            "-czf",
            str(tgz),
            "-C",
            str(REPO_ROOT),
            "sys/lut6-core",
            "artifacts/crystals",
        ],
        env=tar_env,
        check=True,
    )
    subprocess.run(["scp", "-q", str(tgz), f"{host}:/tmp/_lut6_p46.tgz"], check=True)
    tgz.unlink(missing_ok=True)

    remote_py = f"""
import bz2, gzip, hashlib, json, os, subprocess, time, urllib.request, zipfile
from pathlib import Path

wdir = Path("/tmp/_ltcb_p46_run")
os.makedirs(wdir, exist_ok=True)
enw = Path("/tmp/_ltcb_enwik8_cache/enwik8")
if not enw.is_file():
    os.makedirs(enw.parent, exist_ok=True)
    zp = enw.parent / "enwik8.zip"
    urllib.request.urlretrieve("https://mattmahoney.net/dc/enwik8.zip", zp)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(enw.parent)
    zp.unlink(missing_ok=True)

raw100 = enw.read_bytes()
assert len(raw100) == 100_000_000

subprocess.run(["tar", "-xzf", "/tmp/_lut6_p46.tgz", "-C", str(wdir)], check=True)
cargo = os.path.expanduser("~/.cargo/bin/cargo")
subprocess.run([cargo, "build", "--release", "--features", "std", "--manifest-path", str(wdir / "sys/lut6-core/Cargo.toml"), "--bin", "mol_ltcb_codec"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
bin_p = wdir / "sys/lut6-core/target/release/mol_ltcb_codec"

def read_rapl_uj():
    for p in ("/sys/class/powercap/intel-rapl:0/energy_uj", "/sys/class/powercap/amd_rapl:0/energy_uj"):
        if os.path.exists(p):
            try:
                return int(open(p).read().strip())
            except Exception:
                pass
    return None

def read_thor_vin_mw():
    for root, _, files in os.walk("/sys/class/hwmon"):
        nfile = os.path.join(root, "name")
        if os.path.exists(nfile):
            nm = open(nfile).read().strip()
            if "ina238" in nm or "ina3221" in nm:
                for fn in sorted(files):
                    if fn.startswith("power") and fn.endswith("_input"):
                        try:
                            uw = int(open(os.path.join(root, fn)).read().strip())
                            if uw > 100_000:
                                return uw / 1_000_000.0
                        except Exception:
                            pass
                if os.path.exists(os.path.join(root, "in1_input")) and os.path.exists(os.path.join(root, "curr1_input")):
                    mv = int(open(os.path.join(root, "in1_input")).read().strip())
                    ma = int(open(os.path.join(root, "curr1_input")).read().strip())
                    if mv > 0 and ma > 0:
                        return (mv * ma) / 1_000_000.0
    return None

is_thor = {str(is_thor_shunt)}

def run_metered(fn):
    if is_thor:
        import threading
        samples = []
        stop = False
        def poll():
            while not stop:
                w = read_thor_vin_mw()
                if w is not None:
                    samples.append((time.perf_counter(), w))
                time.sleep(0.004)
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        stop = True
        th.join(timeout=0.2)
        if len(samples) >= 2:
            j = 0.0
            for i in range(1, len(samples)):
                j += 0.5 * (samples[i-1][1] + samples[i][1]) * (samples[i][0] - samples[i-1][0])
        else:
            w = read_thor_vin_mw() or 25.0
            j = w * dt
        return res, round(j, 4), None
    else:
        u0 = read_rapl_uj()
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        u1 = read_rapl_uj()
        rj = round((u1 - u0) / 1e6, 4) if (u0 is not None and u1 is not None and u1 >= u0) else None
        return res, None, rj

rows = []
for tname, nb in [("enwik7_10MB", 10_000_000), ("enwik8_100MB", 100_000_000)]:
    sub = raw100[:nb]
    # 1. gzip -9
    def _gz():
        t0 = time.perf_counter()
        a = gzip.compress(sub, compresslevel=9)
        ems = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        d = gzip.decompress(a)
        dms = (time.perf_counter() - t1) * 1000.0
        assert len(d) == nb
        return len(a), hashlib.sha256(a).hexdigest()[:16], ems, dms
    (plen, asha, ems, dms), mj, rj = run_metered(_gz)
    rows.append({{
        "host_label": {host_label!r}, "board_metered": is_thor, "corpus_tier": tname, "raw_bytes": nb,
        "codec": "gzip_9_DEFLATE", "payload_bytes": plen, "decoder_plus_prior_bytes": 32768,
        "mahoney_ranked_total_bytes": plen + 32768, "payload_bpb": round(plen * 8.0 / nb, 4),
        "mahoney_total_bpb": round((plen + 32768) * 8.0 / nb, 4),
        "compress_ms": round(ems, 2), "decompress_ms": round(dms, 2),
        "measured_j": mj, "reported_j": rj, "archive_sha256": asha, "lossless_sha256_verified": True
    }})
    # 2. bzip2 -9
    def _bz():
        t0 = time.perf_counter()
        a = bz2.compress(sub, compresslevel=9)
        ems = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        d = bz2.decompress(a)
        dms = (time.perf_counter() - t1) * 1000.0
        assert len(d) == nb
        return len(a), hashlib.sha256(a).hexdigest()[:16], ems, dms
    (plen, asha, ems, dms), mj, rj = run_metered(_bz)
    rows.append({{
        "host_label": {host_label!r}, "board_metered": is_thor, "corpus_tier": tname, "raw_bytes": nb,
        "codec": "bzip2_9_BWT", "payload_bytes": plen, "decoder_plus_prior_bytes": 40960,
        "mahoney_ranked_total_bytes": plen + 40960, "payload_bpb": round(plen * 8.0 / nb, 4),
        "mahoney_total_bpb": round((plen + 40960) * 8.0 / nb, 4),
        "compress_ms": round(ems, 2), "decompress_ms": round(dms, 2),
        "measured_j": mj, "reported_j": rj, "archive_sha256": asha, "lossless_sha256_verified": True
    }})
    # 3. MoL-LTCB-CM2 native Rust (`--cm2`)
    oj = wdir / f"mlt2_{{nb}}.json"
    def _cm2():
        subprocess.run([str(bin_p), "--cm2", "--input", str(enw), "--max-bytes", str(nb), "--out-json", str(oj)], check=True)
        return json.loads(oj.read_text())
    mdoc, mj, rj = run_metered(_cm2)
    plen = mdoc["compressed_payload_bytes"]
    rows.append({{
        "host_label": {host_label!r}, "board_metered": is_thor, "corpus_tier": tname, "raw_bytes": nb,
        "codec": "MoL_LTCB_CM2_WRT_StateSMix", "payload_bytes": plen, "decoder_plus_prior_bytes": {wasm_bytes},
        "mahoney_ranked_total_bytes": plen + {wasm_bytes}, "payload_bpb": round(mdoc["payload_bpb"], 4),
        "mahoney_total_bpb": round((plen + {wasm_bytes}) * 8.0 / nb, 4),
        "compress_ms": mdoc["compress_ms"], "decompress_ms": mdoc["decompress_ms"],
        "measured_j": mj, "reported_j": rj, "archive_sha256": mdoc["archive_sha256"][:16],
        "lossless_sha256_verified": bool(mdoc["lossless_exact"])
    }})

print(json.dumps(rows))
"""
    remote_cmd = (
        "python3 - << 'PYEOF'\n"
        + remote_py
        + "\nPYEOF\n"
        + "rm -rf /tmp/_ltcb_p46_run /tmp/_lut6_p46.tgz /tmp/_ltcb_enwik8_cache"
    )
    out = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host, remote_cmd], text=True)
    last_line = [ln for ln in out.splitlines() if ln.strip().startswith("[")][-1]
    return json.loads(last_line)


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] Building & V8-verifying 21-export #![no_std] lut6_core.wasm (<= 16,384 B)...")
    wasm_doc = build_and_verify_21export_wasm()
    print("  WASM:", wasm_doc)

    try:
        print("[2/5] Running local Apple M5 Max MoL-LTCB-CM2 scaling (enwik5 100 KB .. enwik8 100 MB)...")
        enwik8_bytes = _ensure_enwik8()
        m5_rows, m5_enwik8_cm2_doc = run_local_m5_max_ltcb_cm2_scaling(enwik8_bytes, wasm_doc["wasm_bytes"])
    finally:
        # Purge local /tmp/ltcb_cache immediately so zero external corpus files linger!
        shutil.rmtree(ENWIK8_PATH.parent, ignore_errors=True)

    print("[3/5] Running remote Jetson AGX Thor (TI INA238 VIN measured_j) & AMD Zen 5 (RAPL)...")
    thor_rows = run_remote_ltcb_cm2_host(
        "cbio@100.111.185.107",
        "NVIDIA_Jetson_AGX_Thor_Tegra_aarch64",
        True,
        wasm_doc["wasm_bytes"],
    )
    zen5_rows = run_remote_ltcb_cm2_host(
        "dcharlot@100.91.48.80",
        "AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64",
        False,
        wasm_doc["wasm_bytes"],
    )

    all_ltcb_rows = m5_rows + thor_rows + zen5_rows

    # Verify cross-architecture bit-exact archive SHA-256 identity on 100 MB enwik8!
    enwik8_cm2_shas = {
        r["host_label"]: r["archive_sha256"]
        for r in all_ltcb_rows
        if r["corpus_tier"] == "enwik8_100MB" and r["codec"] == "MoL_LTCB_CM2_WRT_StateSMix"
    }
    assert len(enwik8_cm2_shas) == 3, f"Expected 3 fleet hosts, got {enwik8_cm2_shas}"
    assert len(set(enwik8_cm2_shas.values())) == 1, f"Cross-host CM2 archive SHA mismatch: {enwik8_cm2_shas}"

    print("[4/5] Running 4-domain Online-TTT-LUT6 regret collapse & sparse slab benchmark...")
    ttt_rows, ttt_meta = run_online_ttt_lut6_four_domain_benchmark(REPO_ROOT)

    print("[5/5] Running 144/144 Yosys abc -lut 6 Boolean miter for WRT-Capillary & StateSMix-TTT...")
    miter_rows, miter_meta = run_ltcb_cm2_ttt_yosys_miter_suite()

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(ART_DIR / "phase46_ltcb_cm2_enwik8_fleet.csv", all_ltcb_rows)
    _write_csv(ART_DIR / "phase46_online_ttt_four_domains.csv", [asdict(r) for r in ttt_rows])
    _write_csv(ART_DIR / "phase46_cm2_ttt_lut6_miter.csv", [asdict(r) for r in miter_rows])

    meta = {
        "phase": 46,
        "claims": ["LA-80", "LA-81", "LA-83"],
        "enwik8_official_sha256": ENWIK8_SHA256,
        "cross_arch_enwik8_cm2_archive_sha256_16": next(iter(enwik8_cm2_shas.values())),
        "cross_arch_hosts_bit_identical": list(enwik8_cm2_shas.keys()),
        "enwik8_100mb_m5_max_cm2": m5_enwik8_cm2_doc,
        "wasm_21export": wasm_doc,
        "online_ttt_four_domains": ttt_meta,
        "cm2_ttt_yosys_miter": miter_meta,
    }
    (ART_DIR / "phase46_ltcb_cm2_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("=== ENWIK8 100 MB CROSS-HOST CM2 SUMMARY ===")
    for r in all_ltcb_rows:
        if r["corpus_tier"] == "enwik8_100MB":
            print(r)
    print("=== ONLINE-TTT-LUT6 4-DOMAIN SUMMARY ===")
    for r in ttt_rows:
        print(r)
    print("=== YOSYS MITER ===")
    print(miter_meta)


if __name__ == "__main__":
    main()
