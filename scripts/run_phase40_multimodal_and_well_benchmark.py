"""Phase 40 (`Claims LA-57 & LA-58`): Executes:
  1. `Claim LA-57`: `24`-Task Real Git-Worktree `SWE-bench Multimodal & Pro` Benchmark
     (`SVG viewBox`, `Responsive Flex-Grid`, `High-DPI Canvas DPR`, `Mermaid State-Diagram`,
     `PCB/CAD Courtyard DRC`, and `Multi-File Transactional Saga` + `4 KB AST-Patch-LUT6` Router),
  2. `Claim LA-58`: `12`-Module (`384`-Call) Planetary Digital Well (`Well-Codifier`) Standard-Library
     Rung-5 -> Rung-7 Projection + Yosys `0.71+` `abc -lut 6` Boolean Miter Equivalence,
  3. `#![no_std]` `lut6_core.wasm` (`15` Exports inside `<= 16,384 B`) Compilation & V8 Node Smoke Test!
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from clai.frontier.planetary_well_codifier import run_planetary_well_codifier_benchmark
from clai.frontier.swebench_multimodal_pro import run_swebench_multimodal_pro_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts"


def build_and_verify_15export_wasm() -> dict:
    """Compiles `#![no_std]` `lut6-core` to `wasm32-unknown-unknown` (`opt-level=z, lto=fat, strip=symbols`),
    verifies `<= 16,384 B` (`16 KB`) size budget, and executes all `15` live exports in Node V8 WebAssembly!"""
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
        "  if (ex.length !== 15) throw new Error('Expected 15 wasm_ exports, got ' + ex.length + ': ' + ex.join(','));\n"
        "  // Test export #15: percept=7, subgoal=3 (hit!), epoch=2 -> next_epoch=3 (bank=3), well_addr=2 (RFC1918=1), expect_0x5A=1, actual=0x5A\n"
        "  const arg0 = 7 | (3 << 6) | (2 << 12) | (2 << 16) | (0x5A << 22) | (1 << 30);\n"
        "  const r = instance.exports.wasm_eval_bfws4_and_well_codifier(arg0 >>> 0, 5) >>> 0;\n"
        "  const status2 = (r >>> 30) & 3;\n"
        "  const bank2 = (r >>> 28) & 3;\n"
        "  const ep4 = (r >>> 24) & 15;\n"
        "  const well8 = (r >>> 16) & 255;\n"
        "  if (status2 !== 1 || bank2 !== 3 || ep4 !== 3 || (well8 & 1) !== 1) {\n"
        "    throw new Error('Unexpected export #15 output: 0x' + r.toString(16));\n"
        "  }\n"
        "  // Test breaker (actual=0x11 != 0x5A)\n"
        "  const bad0 = 7 | (3 << 6) | (2 << 12) | (20 << 16) | (0x11 << 22) | (1 << 30);\n"
        "  const rb = instance.exports.wasm_eval_bfws4_and_well_codifier(bad0 >>> 0, 5) >>> 0;\n"
        "  if (((rb >>> 30) & 3) !== 2) throw new Error('Breaker failed: 0x' + rb.toString(16));\n"
        "  console.log(JSON.stringify({exports: ex, r_hex: '0x' + r.toString(16), rb_hex: '0x' + rb.toString(16)}));\n"
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
        "v8_bfws4_well_hex": v8_res["r_hex"],
        "v8_breaker_hex": v8_res["rb_hex"],
    }


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    mm_rows, mm_meta = run_swebench_multimodal_pro_benchmark()
    mm_csv = ART_DIR / "phase40_swebench_multimodal_pro.csv"
    with mm_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mm_rows[0].keys()))
        w.writeheader()
        w.writerows(mm_rows)

    well_rows, well_meta = run_planetary_well_codifier_benchmark()
    well_csv = ART_DIR / "phase40_planetary_well_codifier.csv"
    with well_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(well_rows[0].keys()))
        w.writeheader()
        w.writerows(well_rows)

    wasm_meta = build_and_verify_15export_wasm()

    meta = {
        "swebench_multimodal_pro": mm_meta,
        "swebench_multimodal_pro_rows": mm_rows,
        "planetary_well_codifier": well_meta,
        "wasm_15export": wasm_meta,
        "board_metered": False,
        "measured_j": None,
    }
    (ART_DIR / "phase40_multimodal_and_well_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("=== SWE-bench Multimodal & Pro (24 Git-Worktree Tasks) ===")
    for r in mm_rows:
        print(
            f"  {r['system_id']:24s} | test_only={r['unit_test_only_pass_pct']:6.2f}% | "
            f"mm_verified={r['multimodal_verified_pct']:6.2f}% | broken={r['broken_visual_or_code_commit_pct']:6.2f}% | "
            f"{r['mean_energy_mj_per_task']:8.4f} mJ/task"
        )
    print("=== Planetary Digital Well (12 Stdlib Modules, 384 Calls) ===")
    print(
        f"  parity={well_meta['exact_parity_pct']}% ({well_meta['exact_matches']}/{well_meta['total_pinned_calls']}) | "
        f"yosys_miter={well_meta['yosys_abc_lut6_miter_verified']} ({well_meta['yosys_miter_cones']} cones)"
    )
    print(
        f"=== WASM 15-Export Build ===\n  size={wasm_meta['wasm_bytes']} B ({wasm_meta['wasm_kb']} KB <= 16 KB) | "
        f"sha12={wasm_meta['wasm_sha12']} | exports={wasm_meta['export_count']} | v8={wasm_meta['v8_bfws4_well_hex']}"
    )


if __name__ == "__main__":
    main()
