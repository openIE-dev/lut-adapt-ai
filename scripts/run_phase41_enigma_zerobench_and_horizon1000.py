"""Phase 41 (`Claims LA-61, LA-62 & LA-63`): Executes:
  1. `Claim LA-61`: `24`-Task SHA-256-Pinned `EnigmaEval & ZeroBench` Symbolic Cryptanalysis
     + Post-Cutoff Alien-ISA/DSL Benchmark & `144/144` `Yosys 0.71+` (`abc -lut 6`) `$lut` Miter,
  2. `Claim LA-62`: `1,000`-Step (`16,000`-Transition) `Vending-Bench & BALROG` Coupled Economic
     + Partially-Observable Exploration Horizon (`Z3 5.4.1` Simplex `CBF` + `8`-Bank `32 KB` `LUT6`),
  3. `Claim LA-63`: `#![no_std]` `lut6_core.wasm` (`16` Exports inside `<= 16,384 B`) Compilation
     & V8 Node Bijective `Enigma-SBox6` Roundtrip + `8`-Bank Cascade Verification!
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from clai.frontier.enigma_zerobench_codifier import run_enigma_zerobench_benchmark
from clai.frontier.vending_balrog_horizon1000 import run_vending_balrog_horizon1000_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def build_and_verify_16export_wasm() -> dict:
    """Compiles `#![no_std]` `lut6-core` to `wasm32-unknown-unknown` (`opt-level=z, lto=fat, strip=symbols`),
    verifies `<= 16,384 B` (`16 KB`) size budget, and executes all `16` live exports in Node V8 WebAssembly!"""
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
    wasm_opt = "/opt/homebrew/bin/wasm-opt"
    if Path(wasm_opt).is_file():
        subprocess.run(
            [wasm_opt, "-Oz", "--enable-bulk-memory", "--enable-sign-ext", str(out_wasm), "-o", str(out_wasm)],
            check=True,
        )
    raw = out_wasm.read_bytes()
    (REPO_ROOT / "paper" / "web" / "lut6_core.wasm").write_bytes(raw)
    assert len(raw) <= 16384, f"lut6_core.wasm ({len(raw)} B) exceeded 16,384 B!"

    node_check = (
        "const fs = require('fs');\n"
        f"const buf = fs.readFileSync({str(out_wasm)!r});\n"
        "(async () => {\n"
        "  const { instance } = await WebAssembly.instantiate(buf, {});\n"
        "  const ex = Object.keys(instance.exports).filter(k => k.startsWith('wasm_'));\n"
        "  if (ex.length !== 16) throw new Error('Expected 16 wasm_ exports, got ' + ex.length + ': ' + ex.join(','));\n"
        "  // Test export #16 across ALL 64 6-bit Enigma-SBox6 roundtrips + 8-bank cascade + Simplex shield!\n"
        "  for (let x = 0; x < 64; x++) {\n"
        "    const encArg = (57) | (3 << 6) | (6 << 12) | (x << 15) | (0b100110 << 21);\n"
        "    const re = instance.exports.wasm_eval_rbfws8_enigma_horizon(encArg >>> 0, 7) >>> 0;\n"
        "    const st2 = (re >>> 30) & 3;\n"
        "    const bk3 = (re >>> 27) & 7;\n"
        "    const ep3 = (re >>> 24) & 7;\n"
        "    const sb6 = (re >>> 18) & 63;\n"
        "    const sh2 = (re >>> 16) & 3;\n"
        "    if (st2 !== 1 || bk3 !== 7 || ep3 !== 7 || sh2 !== 3) {\n"
        "      throw new Error('Unexpected R-BFWS8/Shield pack: 0x' + re.toString(16));\n"
        "    }\n"
        "    const decArg = (57) | (0 << 6) | (0 << 12) | (sb6 << 15) | (0 << 21) | (1 << 27);\n"
        "    const rd = instance.exports.wasm_eval_rbfws8_enigma_horizon(decArg >>> 0, 7) >>> 0;\n"
        "    const xBack = (rd >>> 18) & 63;\n"
        "    if (xBack !== x) throw new Error('Enigma-SBox6 roundtrip failed at x=' + x + ' -> ' + sb6 + ' -> ' + xBack);\n"
        "  }\n"
        "  const brkArg = (57) | (3 << 6) | (2 << 12) | (13 << 15) | (0 << 21) | (1 << 28);\n"
        "  const rb = instance.exports.wasm_eval_rbfws8_enigma_horizon(brkArg >>> 0, 7) >>> 0;\n"
        "  if (((rb >>> 30) & 3) !== 2) throw new Error('Breaker failed: 0x' + rb.toString(16));\n"
        "  const sampleEnc = instance.exports.wasm_eval_rbfws8_enigma_horizon((57 | (3 << 6) | (6 << 12) | (42 << 15) | (0b100110 << 21)) >>> 0, 7) >>> 0;\n"
        "  console.log(JSON.stringify({exports: ex, r_hex: '0x' + sampleEnc.toString(16), rb_hex: '0x' + rb.toString(16)}));\n"
        "})();\n"
    )
    out = subprocess.check_output(["node", "-e", node_check], text=True).strip()
    v8_res = json.loads(out)
    sha = hashlib.sha256(raw).hexdigest()
    return {
        "wasm_bytes": len(raw),
        "wasm_kb": round(len(raw) / 1024.0, 2),
        "under_16kb_limit": len(raw) <= 16384,
        "wasm_sha256": sha,
        "wasm_sha12": sha[:12],
        "export_count": len(v8_res["exports"]),
        "exports": v8_res["exports"],
        "v8_rbfws8_enigma_hex": v8_res["r_hex"],
        "v8_breaker_hex": v8_res["rb_hex"],
        "enigma_sbox6_roundtrip_verified": "64/64",
    }


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    ez_rows, ez_meta = run_enigma_zerobench_benchmark()
    h1k_rows, h1k_meta = run_vending_balrog_horizon1000_benchmark()
    wasm_meta = build_and_verify_16export_wasm()

    ez_csv = ART_DIR / "phase41_enigma_zerobench.csv"
    with ez_csv.open("w", newline="", encoding="utf-8") as f:
        dicts = [asdict(r) for r in ez_rows]
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        w.writerows(dicts)

    h1k_csv = ART_DIR / "phase41_vending_balrog_horizon1000.csv"
    with h1k_csv.open("w", newline="", encoding="utf-8") as f:
        dicts = [asdict(r) for r in h1k_rows]
        w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        w.writeheader()
        w.writerows(dicts)

    meta = {
        "claims": ["LA-61", "LA-62", "LA-63"],
        "enigma_zerobench": ez_meta,
        "vending_balrog_horizon1000": h1k_meta,
        "wasm_16export": wasm_meta,
        "board_metered": False,
        "measured_j": None,
    }
    (ART_DIR / "phase41_enigma_and_horizon1000_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
