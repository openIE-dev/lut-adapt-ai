"""Remote Zen 5 (`100.91.48.80`) Deterministic Verifier Server (`rtl`, `lean`, `z3`) for Phase 37."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def run_rtl(d: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        cand_f = tp / "cand.sv"
        ref_f = tp / "ref.sv"
        test_f = tp / "test.sv"
        cand_f.write_text(re.sub(r"\balways_comb\b", "always @(*)", d["cand"]) + "\n")
        ref_f.write_text(d["ref"] + "\n")
        clean_test = re.sub(r"^\s*\$dump(?:file|vars)\b[^\n]*$", "", d["test"], flags=re.MULTILINE)
        test_f.write_text(clean_test + "\n")
        env = dict(os.environ)
        env["PATH"] = f"/home/dcharlot/.local/bin:{env.get('PATH', '')}"

        stat_f = tp / "stat.json"
        ys_cmd = f"read_verilog -sv {cand_f}; hierarchy -check -top TopModule; proc; opt; memory; techmap; abc -lut 6; tee -q -o {stat_f} stat -json"
        try:
            p_syn = subprocess.run(["yosys", "-Q", "-q", "-p", ys_cmd], capture_output=True, text=True, env=env, timeout=15)
            synth_ok = (p_syn.returncode == 0)
        except subprocess.TimeoutExpired:
            synth_ok = False
        lut_cnt = 0
        if synth_ok and stat_f.is_file():
            try:
                st_j = json.loads(stat_f.read_text())
                for mod_d in st_j.get("modules", {}).values():
                    for k_c, v_c in mod_d.get("num_cells_by_type", {}).items():
                        if "lut" in k_c.lower():
                            lut_cnt += int(v_c)
            except Exception:
                pass

        sim_bin = tp / "sim_out"
        try:
            p_iv = subprocess.run(
                ["iverilog", "-g2012", "-o", str(sim_bin), str(test_f), str(ref_f), str(cand_f)],
                capture_output=True,
                text=True,
                env=env,
                timeout=12,
            )
        except subprocess.TimeoutExpired:
            return {"synth_ok": synth_ok, "equiv_ok": False, "lut6_gates": lut_cnt, "err": "iverilog timeout"}
        if p_iv.returncode != 0:
            return {"synth_ok": synth_ok, "equiv_ok": False, "lut6_gates": lut_cnt, "err": p_iv.stderr[-300:]}
        try:
            p_vvp = subprocess.run(["vvp", str(sim_bin)], capture_output=True, text=True, env=env, timeout=12)
        except subprocess.TimeoutExpired:
            return {"synth_ok": synth_ok, "equiv_ok": False, "lut6_gates": lut_cnt, "err": "vvp timeout"}
        out = p_vvp.stdout or ""
        m_mis = re.search(r"Mismatches:\s*(\d+)\s*in\s*(\d+)", out)
        equiv_ok = bool(m_mis and int(m_mis.group(1)) == 0 and int(m_mis.group(2)) > 0)
        return {"synth_ok": synth_ok, "equiv_ok": equiv_ok, "lut6_gates": lut_cnt, "err": out[-300:] if not equiv_ok else ""}


def run_lean(blocks: list) -> dict:
    lean_bin = "/home/dcharlot/.elan/bin/lean"
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)

        def _chk_one(item: tuple[int, tuple[str, str]]) -> tuple[str, bool]:
            idx, (tid, code) = item
            clean = re.sub(r"--[^\n]*", "", code)
            if re.search(r"\b(?:sorry|admit|axiom|native_decide|ofReduceBool)\b", clean):
                return tid, False
            fp = tp / f"p_{idx}.lean"
            fp.write_text("set_option autoImplicit false\n" + code + "\n")
            try:
                p = subprocess.run([lean_bin, str(fp)], capture_output=True, text=True, timeout=8)
                out = (p.stdout or "") + (p.stderr or "")
                return tid, (p.returncode == 0) and ("sorry" not in out)
            except subprocess.TimeoutExpired:
                return tid, False

        with ThreadPoolExecutor(max_workers=8) as ex:
            return dict(ex.map(_chk_one, enumerate(blocks)))


def run_z3(states: list) -> dict:
    z3_bin = "/home/dcharlot/.local/bin/z3"
    t0 = time.perf_counter()
    out_actions = []
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        for idx, st in enumerate(states):
            cash = st["cash"]
            stk = st["stock"]
            dem = st["demand"]
            reg = st["regime"]
            d_obs = st["d_obs"]
            tj = st["tj"]
            amb = st["amb"]
            smt = f"""
(set-logic QF_NIA)
(declare-const restock Int)
(declare-const ptier Int)
(declare-const pmode Int)
(declare-const spd Int)
(assert (and (>= restock 0) (<= restock 20)))
(assert (and (>= ptier 0) (<= ptier 2)))
(assert (and (>= pmode 0) (<= pmode 2)))
(assert (and (>= spd 350) (<= spd 1800)))
(assert (<= (* spd spd) (* 4800 {d_obs})))
(assert (>= (- (+ {stk} restock) {dem}) 2))
(assert (> (- {cash} (+ (* 180 restock) 240)) 1500))
(define-fun p_mw () Int (ite (= pmode 0) 1200 (ite (= pmode 1) 2400 4200)))
(assert (<= (+ (* 1000 {amb}) (* 650 (- {tj} {amb})) (* 95 p_mw)) 820000))
(check-sat)
(get-value (restock ptier pmode spd))
"""
            fp = tp / f"s_{idx}.smt2"
            fp.write_text(smt)
            p = subprocess.run([z3_bin, str(fp)], capture_output=True, text=True, timeout=5)
            txt = p.stdout
            vals = dict(re.findall(r"\(([a-z_]+)\s+(-?\d+)\)", txt))
            trace_w = 143 if (reg % 2 == 0) else 166
            out_actions.append(
                {
                    "restock": int(vals.get("restock", 12 if stk < 10 else 4)),
                    "ptier": int(vals.get("ptier", 1 if (reg & 1) == 0 else 0)),
                    "trace_w": trace_w,
                    "pmode": int(vals.get("pmode", 0 if tj >= 680 else 1)),
                    "spd": int(vals.get("spd", max(350, min(1600, int(math.sqrt(max(0, 4800 * d_obs))))))),
                    "z3_sat": ("sat" in txt.splitlines()[0]) if txt else False,
                }
            )
    ms = (time.perf_counter() - t0) * 1000.0
    return {"actions": out_actions, "z3_ms": ms}


if __name__ == "__main__":
    mode = sys.argv[1]
    data = json.loads(sys.stdin.read())
    if mode == "rtl":
        print(json.dumps(run_rtl(data)))
    elif mode == "lean":
        print(json.dumps(run_lean(data)))
    elif mode == "z3":
        print(json.dumps(run_z3(data)))
