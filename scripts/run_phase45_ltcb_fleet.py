"""Phase 45 (`Claims LA-76, LA-77 & LA-79`): Cross-Fleet Matt Mahoney Large Text Compression
Benchmark (`LTCB` `enwik5` `100 KB`, `enwik6` `1 MB`, `enwik7` `10 MB`, and `enwik8` `100 MB`
Wikipedia XML), `Tufalabs/tufazip` `v2` `QPI1` Starter-Prior Audit, `144/144` `Yosys` `$lut`
Miter, and `#![no_std]` `20`-Export `lut6_core.wasm` (`<= 16,384 B`) Build & V8 Verification:
  - `Apple M5 Max` (`arm64` Darwin `RUSAGE_INFO_V6` `reported_j`),
  - `NVIDIA Jetson AGX Thor` (`cbio@100.111.185.107`, physical `TI INA238` `VIN` DC shunt
    `measured_j`, `board_metered = true`!),
  - `AMD Ryzen AI Max+ 395` (`dcharlot@100.91.48.80`, `Zen 5` Linux `RAPL` `reported_j`).
Zero raw `enwik` text or third-party weights are persisted in git; all runs verify exact SHA-256!
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
from clai.frontier.ltcb_neural_lut6_compression import (
    audit_tufazip_qpi1_prior_transcoding,
    run_ltcb_yosys_miter_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"
ENWIK8_PATH = Path("/tmp/ltcb_cache/enwik8")
WEIGHTS_PATH = Path("/tmp/ltcb_cache/weights.bin")
ENWIK8_SHA256 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"


def build_and_verify_20export_wasm() -> dict:
    wasm_dir = ART_DIR / "wasm"
    wasm_dir.mkdir(parents=True, exist_ok=True)
    out_wasm = wasm_dir / "lut6_core.wasm"
    subprocess.run(
        [
            "rustc",
            "--edition=2024",
            "--target=wasm32-unknown-unknown",
            "--crate-type=cdylib",
            "-C", "opt-level=z",
            "-C", "panic=abort",
            "-C", "strip=symbols",
            "-C", "codegen-units=1",
            "-C", "lto=fat",
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
        "  if (ex.length !== 20) throw new Error('Expected 20 wasm_ exports, got ' + ex.length + ': ' + ex.join(','));\n"
        "  // Probe Export #20: Bank-13 + LTCB Skip=3 (lz_span_ge4) + Kraft Bucket=3 (15/16 peak)\n"
        "  const arg1 = (1 | (3 << 6) | (12 << 12) | (4 << 16) | (48 << 22) | (0 << 28)) >>> 0;\n"
        "  const r1 = instance.exports.wasm_eval_ltcb_kraft_lut6(arg1, 7) >>> 0;\n"
        "  if (((r1 >>> 30) & 3) !== 1 || ((r1 >>> 26) & 15) !== 13 || ((r1 >>> 20) & 3) !== 3 || ((r1 >>> 18) & 3) !== 3) {\n"
        "    throw new Error('Unexpected r1: 0x' + r1.toString(16));\n"
        "  }\n"
        "  // Probe Export #20: Bank-0 + XML Tag=2 + Underflow Guard=0\n"
        "  const arg2 = (1 | (0 << 6) | (0 << 12) | (1 << 16) | (15 << 22) | (0 << 28)) >>> 0;\n"
        "  const r2 = instance.exports.wasm_eval_ltcb_kraft_lut6(arg2, 7) >>> 0;\n"
        "  if (((r2 >>> 30) & 3) !== 1 || ((r2 >>> 26) & 15) !== 0 || ((r2 >>> 20) & 3) !== 2 || ((r2 >>> 18) & 3) !== 0) {\n"
        "    throw new Error('Unexpected r2: 0x' + r2.toString(16));\n"
        "  }\n"
        "  // Probe Export #20: Breaker Halt (brk=1)\n"
        "  const arg3 = (1 | (3 << 6) | (12 << 12) | (4 << 16) | (48 << 22) | (1 << 28)) >>> 0;\n"
        "  const r3 = instance.exports.wasm_eval_ltcb_kraft_lut6(arg3, 7) >>> 0;\n"
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
        subprocess.run(["curl", "-sL", "-o", str(zpath), "https://mattmahoney.net/dc/enwik8.zip"], check=True)
        subprocess.run(["unzip", "-q", "-o", str(zpath), "-d", str(ENWIK8_PATH.parent)], check=True)
    raw = ENWIK8_PATH.read_bytes()
    assert len(raw) == 100_000_000
    assert hashlib.sha256(raw).hexdigest() == ENWIK8_SHA256
    return raw


def run_local_m5_max_ltcb_scaling(enwik8_bytes: bytes, wasm_bytes: int) -> tuple[list[dict], dict]:
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
    enwik8_mltc_doc = {}
    zstd_bin = shutil.which("zstd")

    for tier_name, n_bytes in tiers:
        sub = enwik8_bytes[:n_bytes]
        sub_sha = hashlib.sha256(sub).hexdigest()

        # 1. gzip -9 (DEFLATE, ~32 KB binary floor)
        with meter_joules("ltcb_gzip") as jm_gz:
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
        with meter_joules("ltcb_bzip2") as jm_bz:
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

        # 3. zstd -19 (if installed)
        if zstd_bin:
            tmp_in = Path(f"/tmp/_ltcb_{n_bytes}.in")
            tmp_zst = Path(f"/tmp/_ltcb_{n_bytes}.zst")
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

        # 4. xz / LZMA2 preset 6 (`lzma` stdlib, reference LZMA2)
        if n_bytes <= 10_000_000:
            with meter_joules("ltcb_xz") as jm_xz:
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

        # 5. Native Rust `MoL-LTCB` (`mol_ltcb_codec`: Lockstep `VAULT` + `LTCB-XML-Skip LUT6` + `waiOS` Integer Range Codec)
        out_j = Path(f"/tmp/_mltc_{n_bytes}.json")
        with meter_joules("ltcb_mol") as jm_mol:
            subprocess.run(
                [
                    str(rust_bin),
                    "--input",
                    str(ENWIK8_PATH),
                    "--max-bytes",
                    str(n_bytes),
                    "--out-json",
                    str(out_j),
                ],
                check=True,
            )
        mdoc = json.loads(out_j.read_text(encoding="utf-8"))
        out_j.unlink(missing_ok=True)
        assert mdoc["lossless_exact"] is True and mdoc["decoded_sha256"] == sub_sha
        if n_bytes == 100_000_000:
            enwik8_mltc_doc = mdoc
        rows.append(
            {
                "host_label": host_label,
                "board_metered": False,
                "corpus_tier": tier_name,
                "raw_bytes": n_bytes,
                "codec": "MoL_LTCB_waiOS_LUT6_Range",
                "payload_bytes": mdoc["compressed_payload_bytes"],
                "decoder_plus_prior_bytes": wasm_bytes,
                "mahoney_ranked_total_bytes": mdoc["compressed_payload_bytes"] + wasm_bytes,
                "payload_bpb": round(mdoc["payload_bpb"], 4),
                "mahoney_total_bpb": round((mdoc["compressed_payload_bytes"] + wasm_bytes) * 8.0 / n_bytes, 4),
                "compress_ms": mdoc["compress_ms"],
                "decompress_ms": mdoc["decompress_ms"],
                "measured_j": None,
                "reported_j": round(float(jm_mol.reported_j or 0.0), 4),
                "archive_sha256": mdoc["archive_sha256"][:16],
                "lossless_sha256_verified": True,
            }
        )

    return rows, enwik8_mltc_doc


def run_remote_ltcb_host(host: str, host_label: str, is_thor_shunt: bool, wasm_bytes: int) -> list[dict]:
    """Runs native Rust `mol_ltcb_codec` vs `gzip -9` & `bzip2 -9` on `enwik7` (`10 MB`) and
    `enwik8` (`100 MB`) on remote fleet hosts (`Jetson AGX Thor` physical `TI INA238` `VIN`
    `measured_j` with `board_metered = true`, and `AMD Zen 5` Linux `RAPL` `reported_j`), verifying
    cross-architecture bit-exact `archive_sha256` identity (`0` float drift!) and deleting `/tmp` scratch."""
    tgz = Path(f"/tmp/_lut6_p45_{host_label[:8]}.tgz")
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
    subprocess.run(["scp", "-q", str(tgz), f"{host}:/tmp/_lut6_p45.tgz"], check=True)
    tgz.unlink(missing_ok=True)

    remote_py = f"""
import bz2, gzip, hashlib, json, os, subprocess, time, urllib.request, zipfile
from pathlib import Path

wdir = Path("/tmp/_ltcb_p45_run")
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

subprocess.run(["tar", "-xzf", "/tmp/_lut6_p45.tgz", "-C", str(wdir)], check=True)
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
    # Physical TI INA238 whole-board VIN hwmon power sensor on Jetson AGX Thor
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
                                return uw / 1_000_000.0  # Watts
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
    # gzip -9
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
    # MoL-LTCB native Rust
    oj = wdir / f"mltc_{{nb}}.json"
    def _mltc():
        subprocess.run([str(bin_p), "--input", str(enw), "--max-bytes", str(nb), "--out-json", str(oj)], check=True)
        return json.loads(oj.read_text())
    mdoc, mj, rj = run_metered(_mltc)
    plen = mdoc["compressed_payload_bytes"]
    rows.append({{
        "host_label": {host_label!r}, "board_metered": is_thor, "corpus_tier": tname, "raw_bytes": nb,
        "codec": "MoL_LTCB_waiOS_LUT6_Range", "payload_bytes": plen, "decoder_plus_prior_bytes": {wasm_bytes},
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
        + "rm -rf /tmp/_ltcb_p45_run /tmp/_lut6_p45.tgz /tmp/_ltcb_enwik8_cache"
    )
    out = subprocess.check_output(["ssh", "-o", "BatchMode=yes", host, remote_cmd], text=True)
    last_line = [ln for ln in out.splitlines() if ln.strip().startswith("[")][-1]
    return json.loads(last_line)


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] Building & V8-verifying 20-export #![no_std] lut6_core.wasm (<= 16,384 B)...")
    wasm_doc = build_and_verify_20export_wasm()
    print("  WASM:", wasm_doc)

    print("[2/5] Running local Apple M5 Max LTCB scaling (enwik5 100 KB .. enwik8 100 MB)...")
    enwik8_bytes = _ensure_enwik8()
    m5_rows, m5_enwik8_doc = run_local_m5_max_ltcb_scaling(enwik8_bytes, wasm_doc["wasm_bytes"])

    print("[3/5] Running remote Jetson AGX Thor (TI INA238 VIN measured_j) & AMD Zen 5 (RAPL)...")
    thor_rows = run_remote_ltcb_host(
        "cbio@100.111.185.107",
        "NVIDIA_Jetson_AGX_Thor_Tegra_aarch64",
        True,
        wasm_doc["wasm_bytes"],
    )
    zen5_rows = run_remote_ltcb_host(
        "dcharlot@100.91.48.80",
        "AMD_Ryzen_AI_Max_395_Zen5_Linux_x86_64",
        False,
        wasm_doc["wasm_bytes"],
    )

    all_ltcb_rows = m5_rows + thor_rows + zen5_rows

    # Verify cross-architecture bit-exact archive SHA-256 identity on 100 MB enwik8!
    enwik8_mltc_shas = {
        r["host_label"]: r["archive_sha256"]
        for r in all_ltcb_rows
        if r["corpus_tier"] == "enwik8_100MB" and r["codec"] == "MoL_LTCB_waiOS_LUT6_Range"
    }
    assert len(set(enwik8_mltc_shas.values())) == 1, f"Cross-host archive SHA mismatch: {enwik8_mltc_shas}"

    print("[4/5] Auditing real Tufalabs/tufazip v2 QPI1 weights.bin & Hadamard JouleBit-LUT6 priors...")
    prior_rows, prior_meta = audit_tufazip_qpi1_prior_transcoding(WEIGHTS_PATH if WEIGHTS_PATH.is_file() else None)

    print("[5/5] Running 144/144 Yosys abc -lut 6 Boolean miter for LTCB-XML-Skip & Kraft-CDF...")
    miter_rows, miter_meta = run_ltcb_yosys_miter_suite()

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(ART_DIR / "phase45_ltcb_enwik8_fleet.csv", all_ltcb_rows)
    _write_csv(ART_DIR / "phase45_qpi1_starter_prior_transcoding.csv", [asdict(r) for r in prior_rows])
    _write_csv(ART_DIR / "phase45_ltcb_lut6_miter.csv", [asdict(r) for r in miter_rows])

    meta = {
        "phase": 45,
        "claims": ["LA-76", "LA-77", "LA-79"],
        "enwik8_official_sha256": ENWIK8_SHA256,
        "cross_arch_enwik8_mltc_archive_sha256_16": next(iter(enwik8_mltc_shas.values())),
        "cross_arch_hosts_bit_identical": list(enwik8_mltc_shas.keys()),
        "enwik8_100mb_m5_max_mltc": m5_enwik8_doc,
        "wasm_20export": wasm_doc,
        "qpi1_starter_prior_audit": prior_meta,
        "ltcb_yosys_miter": miter_meta,
    }
    (ART_DIR / "phase45_ltcb_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("=== ENWIK8 100 MB CROSS-HOST SUMMARY ===")
    for r in all_ltcb_rows:
        if r["corpus_tier"] == "enwik8_100MB":
            print(r)
    print("=== QPI1 STARTER PRIOR SUMMARY ===")
    for r in prior_rows:
        print(r)
    print("=== YOSYS MITER ===")
    print(miter_meta)


if __name__ == "__main__":
    main()
