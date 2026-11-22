"""Phase 16: Multi-Page BRAM/URAM 1-Cycle Hash-Indexed LUT6 Page Cache for Kria KV260.

Implements the exact bit-accurate compiler, simulator, Yosys UltraScale+ (`xcup`) synthesis
benchmark, and Verilator 5.x cycle-accurate co-simulation for `clai_benes_paged_hotswap_core`
in `hw/kv260/rtl/clai_benes_hotswap_core.sv`.

Key Architectural Innovation:
  - Decouples **Canonical Beneš-Copy-Beneš AST/CAM Wire Routing** (`480` multicast ports routed
    once in `< 1.5 ms` via `compile_multicast_bcb512`) from **Per-Context `LUT6 INIT` Truth-Table
    Switching** (`80 x 64 = 5,120` bits stored per hash slot in parallel Xilinx `RAMB36E2` /
    `URAM288` SDP banks).
  - Switches all `80` `LUT6` truth tables in **1 single clock cycle (`4.0 ns` @ `250 MHz`,
    `CPI = 1.0`)** per 6-byte source-code suffix `{lang_id[1:0], ctx6[47:0]}`, falling back in
    **0 stall cycles** on a tag miss to the Active-Bank XNOR-CAM local variable copy `LUT6` page!
  - Supports **non-blocking 1-cycle live Online Sleep-Crystallization** (`write_page`) as files
    are edited!
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from clai.frontier.benes_lut_fabric import (
    GeneralizedBenesConfig,
    compile_multicast_bcb512,
    eval_multicast_bcb512,
)
from clai.frontier.code_corpus_speculative import SourceCodeFile, encode_code_bitbus_512


def hw_hash_ctx6(lang_id: int, ctx6: bytes, page_addr_bits: int = 9) -> int:
    """
    Bit-exact replica of the zero-DSP circular-shift Galois XOR hash in
    `clai_uram_lut6_page_cache` (`hw/kv260/rtl/clai_benes_hotswap_core.sv`).
    """
    assert len(ctx6) == 6
    u48 = int.from_bytes(ctx6, byteorder="little")
    w0 = u48 & 0xFFFF
    w1 = (u48 >> 16) & 0xFFFF
    w2 = (u48 >> 32) & 0xFFFF

    f0 = (w0 ^ (((w0 & 0xFF) << 8) | ((w0 >> 8) & 0xFF))) & 0xFFFF
    f1 = (w1 ^ (((w1 & 0x1F) << 11) | ((w1 >> 5) & 0x7FF))) & 0xFFFF
    f2 = (w2 ^ (((w2 & 0x7F) << 9) | ((w2 >> 7) & 0x1FF))) & 0xFFFF

    rot_f1 = (((f1 & 0x1FFF) << 3) | ((f1 >> 13) & 0x7)) & 0xFFFF
    rot_f2 = (((f2 & 0x1FF) << 7) | ((f2 >> 9) & 0x7F)) & 0xFFFF
    mix_a = (f0 ^ rot_f1 ^ rot_f2) & 0xFFFF

    lid = lang_id & 0x3
    last_b = (u48 >> 40) & 0xFF
    mix_b = (mix_a ^ (mix_a >> 5) ^ (lid << 9) ^ ((lid << 8) | last_b)) & 0xFFFF
    return mix_b & ((1 << page_addr_bits) - 1)


def hw_pack_tag64(lang_id: int, ctx6: bytes, valid: bool = True) -> int:
    """Packs `{13'd0, valid(1b), lang_id(2b), ctx6_le(48b)}` into a 64-bit Tag BRAM word."""
    assert len(ctx6) == 6
    u48 = int.from_bytes(ctx6, byteorder="little")
    v_bit = 1 if valid else 0
    return (v_bit << 50) | ((lang_id & 0x3) << 48) | u48


class HardwareBRAMPageCache:
    """
    Bit-exact KV260 Dual-Slice (`2 x 80 = 160` LUT6s -> `32` bits = `4` bytes/cycle, with Slice 0
    `80` LUT6s -> `16` bits = `2` bytes directly mirroring `clai_benes_paged_hotswap_core`)
    Multi-Page BRAM/URAM `LUT6` Page Cache + Online Sleep-Crystallizer target.
    """

    VOTERS = 5

    def __init__(self, page_addr_bits: int = 9):
        self.page_addr_bits = page_addr_bits
        self.num_pages = 1 << page_addr_bits
        # Canonical 20 AST/Indentation/Scope wires (`[5, 4]` across the 5 voters)
        # Sensible hardware default covering indent, column, paren/brace, prev-colon, lang, char-class
        default_top20 = [
            128, 129, 132, 133,
            190, 192, 193, 160,
            166, 169, 171, 176,
            177, 178, 183, 186,
            144, 148, 149, 180,
        ]
        self.canon_ast_wires = np.array(default_top20, dtype=np.int32).reshape(5, 4)

        # Build canonical 4-byte (`32 bits x 5 voters x 6 ports`) wire routing
        self.wires_32x5x6 = self._build_canonical_wires(self.canon_ast_wires)
        # Slice 0 (`bits 0..15` = `80 LUT6s` = `480` ports) compiled through 512x512 Beneš-Copy-Beneš!
        self.slice0_desired_480 = self.wires_32x5x6[:16].reshape(-1).astype(np.int32)
        self.slice0_bcb_cfg: GeneralizedBenesConfig = compile_multicast_bcb512(self.slice0_desired_480)

        # Default Active-Bank XNOR-CAM Copy Fallback (`if b0 (wire 452) then b1 (wire 384+b) else 0`)
        self.fallback_tables_32x5 = self._build_cam_fallback_tables()

        # Simulated on-chip Tag BRAM (`[num_pages] uint64`) + Dual-Slice LUT6 INIT BRAM (`[num_pages, 32, 5] uint64`)
        self.tag_bram = np.zeros(self.num_pages, dtype=np.uint64)
        self.lut_bram = np.zeros((self.num_pages, 32, self.VOTERS), dtype=np.uint64)
        self.slot_priority = np.zeros(self.num_pages, dtype=np.int32)

        self.bram_generation = 0
        self.total_lookups = 0
        self.total_hits = 0

    @staticmethod
    def _build_canonical_wires(canon_ast_5x4: np.ndarray) -> np.ndarray:
        wires = np.zeros((32, 5, 6), dtype=np.int32)
        for b in range(32):
            for v in range(5):
                wires[b, v, 0] = 452
                wires[b, v, 1] = 384 + b
                wires[b, v, 2:6] = canon_ast_5x4[v]
        return wires

    @staticmethod
    def _build_cam_fallback_tables() -> np.ndarray:
        # Minterm m = b0 + 2*b1 + 4*s: output `b1` when `b0 == 1` (`m & 3 == 3`), else `0` -> `0xCCCCCCCCCCCCCCCC`!
        mask = np.uint64(0)
        for m in range(64):
            if (m & 1) == 1 and ((m >> 1) & 1) == 1:
                mask |= np.uint64(1) << np.uint64(m)
        return np.full((32, 5), mask, dtype=np.uint64)

    def synthesize_joint_mode_tables(self, x_mat: np.ndarray, y_bytes: np.ndarray) -> np.ndarray:
        """
        Compiles `[32, 5]` `uint64` Joint-Mode Minterm `LUT6 INIT` tables for a single 6-byte
        code context using the shared `canon_ast_wires` (so zero Beneš switch changes are needed
        when switching pages in 1 clock cycle!).
        """
        n_samp = x_mat.shape[0]
        y_u32 = (
            y_bytes[:, 0].astype(np.uint32)
            | (y_bytes[:, 1].astype(np.uint32) << np.uint32(8))
            | (y_bytes[:, 2].astype(np.uint32) << np.uint32(16))
            | (y_bytes[:, 3].astype(np.uint32) << np.uint32(24))
        )
        uniq_w, inv_w, cnt_w = np.unique(y_u32, return_inverse=True, return_counts=True)
        global_mode_u32 = int(uniq_w[int(cnt_w.argmax())])

        cam_hits = x_mat[:, 452] == 1
        if int(cam_hits.sum()) >= 2:
            cam_u32 = np.zeros(n_samp, dtype=np.uint32)
            for b in range(32):
                cam_u32 |= x_mat[:, 384 + b].astype(np.uint32) << np.uint32(b)
            trust_cam = float((cam_u32[cam_hits] == y_u32[cam_hits]).mean()) >= 0.60
        else:
            trust_cam = False

        tables = np.zeros((32, self.VOTERS), dtype=np.uint64)
        ast_weights = np.array([1, 2, 4, 8], dtype=np.int32)

        for v in range(self.VOTERS):
            ast4 = self.canon_ast_wires[v]
            sub_state = x_mat[:, ast4].astype(np.int32) @ ast_weights
            state_mode_u32 = np.full(16, global_mode_u32, dtype=np.uint32)
            state_seen = np.zeros(16, dtype=bool)
            for s in range(16):
                ms = sub_state == s
                if np.any(ms):
                    state_seen[s] = True
                    sub_counts = np.bincount(inv_w[ms], minlength=len(uniq_w))
                    state_mode_u32[s] = uniq_w[int(sub_counts.argmax())]

            for b in range(32):
                mask = np.uint64(0)
                for s in range(16):
                    const_bit = (int(state_mode_u32[s]) >> b) & 1
                    for b0 in (0, 1):
                        for b1 in (0, 1):
                            m = b0 | (b1 << 1) | (s << 2)
                            out_bit = b1 if (b0 == 1 and (trust_cam or not state_seen[s])) else const_bit
                            if out_bit:
                                mask |= np.uint64(1) << np.uint64(m)
                tables[b, v] = mask
        return tables

    def synthesize_const_tables(self, target4: bytes, allow_cam_override: bool = False) -> np.ndarray:
        """Synthesizes `32 x 5` `LUT6 INIT` tables for a fast unambiguous 4-byte continuation."""
        assert len(target4) == 4
        w32 = int.from_bytes(target4, byteorder="little")
        tables = np.zeros((32, self.VOTERS), dtype=np.uint64)
        for b in range(32):
            c_bit = (w32 >> b) & 1
            mask = np.uint64(0)
            for m in range(64):
                b0 = m & 1
                b1 = (m >> 1) & 1
                out_bit = b1 if (allow_cam_override and b0 == 1) else c_bit
                if out_bit:
                    mask |= np.uint64(1) << np.uint64(m)
            tables[b, :] = mask
        return tables

    def write_page(
        self,
        lang_id: int,
        ctx6: bytes,
        tables_32x5: np.ndarray,
        priority: int = 10,
        force_overwrite: bool = False,
    ) -> int:
        """Programs a 6-byte context page into its hardware hash bucket (`1` cycle/word, zero stall)."""
        slot = hw_hash_ctx6(lang_id, ctx6, self.page_addr_bits)
        tag = np.uint64(hw_pack_tag64(lang_id, ctx6, valid=True))
        if force_overwrite or self.tag_bram[slot] == tag or priority >= self.slot_priority[slot]:
            self.tag_bram[slot] = tag
            self.lut_bram[slot] = tables_32x5
            self.slot_priority[slot] = priority
        return slot

    def compile_from_corpus(self, train_files: list[SourceCodeFile]) -> dict:
        """
        Compiles canonical 20 AST wires + Beneš-Copy-Beneš switch bits + populates the `2^B`-page
        KV260 BRAM image with the highest-frequency 6-byte code contexts across `train_files`.
        """
        t0 = time.perf_counter()
        counts6: dict[tuple[int, bytes], dict[bytes, int]] = {}
        sample_buses: list[np.ndarray] = []
        sample_ybits: list[np.ndarray] = []

        rng = np.random.default_rng(42)
        for sf in train_files:
            rb = bytes(sf.raw_bytes.tolist())
            lid = sf.lang_id
            for pos in range(16, len(rb) - 4):
                k6 = (lid, rb[pos - 6 : pos])
                nxt = rb[pos : pos + 4]
                d = counts6.setdefault(k6, {})
                d[nxt] = d.get(nxt, 0) + 1
                if len(sample_buses) < 1024 and rng.random() < 0.012:
                    sample_buses.append(encode_code_bitbus_512(rb[max(0, pos - 256) : pos], lid))
                    shifts = np.arange(8, dtype=np.uint8)[None, :]
                    yb = ((sf.raw_bytes[pos : pos + 4, None] >> shifts) & 1).reshape(32)
                    sample_ybits.append(yb)

        if sample_buses:
            xm = np.stack(sample_buses, axis=0)[:, :384].astype(np.float32)
            ym = np.stack(sample_ybits, axis=0).astype(np.float32)
            salience = np.abs((ym - ym.mean(0, keepdims=True)).T @ (xm - xm.mean(0, keepdims=True))).sum(axis=0)
            top20 = np.argsort(-salience)[:20]
            self.canon_ast_wires = top20.astype(np.int32).reshape(5, 4)
            self.wires_32x5x6 = self._build_canonical_wires(self.canon_ast_wires)
            self.slice0_desired_480 = self.wires_32x5x6[:16].reshape(-1).astype(np.int32)
            self.slice0_bcb_cfg = compile_multicast_bcb512(self.slice0_desired_480)

        # Sort contexts by corpus frequency so hottest contexts claim hash slots first!
        ranked = sorted(counts6.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
        top_candidates = ranked[: self.num_pages * 2]
        branch_set = {k for k, v in top_candidates[:320] if len(v) >= 2 and sum(v.values()) >= 5}

        buckets: dict[tuple[int, bytes], list[tuple[np.ndarray, np.ndarray]]] = {}
        for sf in train_files:
            raw = sf.raw_bytes
            rb = bytes(raw.tolist())
            lid = sf.lang_id
            for pos in range(16, len(raw) - 4):
                k6 = (lid, rb[pos - 6 : pos])
                if k6 in branch_set and len(buckets.get(k6, ())) < 128:
                    bus = encode_code_bitbus_512(rb[max(0, pos - 256) : pos], lid)
                    buckets.setdefault(k6, []).append((bus, raw[pos : pos + 4]))

        for k6, v_dict in top_candidates:
            freq = sum(v_dict.values())
            lid, ctx6 = k6
            if k6 in buckets and len(buckets[k6]) >= 4:
                xm_k = np.stack([it[0] for it in buckets[k6]], axis=0)
                ym_k = np.stack([it[1] for it in buckets[k6]], axis=0)
                tbls = self.synthesize_joint_mode_tables(xm_k, ym_k)
            else:
                best4 = max(v_dict.items(), key=lambda kv: kv[1])[0]
                tbls = self.synthesize_const_tables(best4, allow_cam_override=False)
            self.write_page(lid, ctx6, tbls, priority=freq)

        self.bram_generation += 1
        occupied = int((self.tag_bram != 0).sum())
        return {
            "bram_capacity_pages": self.num_pages,
            "bram_occupied_pages": occupied,
            "bram_occupancy_pct": round(100.0 * occupied / self.num_pages, 1),
            "bram_compile_sec": round(time.perf_counter() - t0, 2),
        }

    def lookup_and_eval(self, bus_512: np.ndarray, lang_id: int) -> tuple[bool, int, list[int], np.ndarray]:
        """
        Bit-exact software mirror of `clai_benes_paged_hotswap_core`:
        Extracts `ctx6` from `bus_512[80:128]`, hashes in 1 cycle, checks Tag BRAM, selects either
        the cached `LUT6 INIT` page or `fallback_tables_32x5`, and returns `(hit, slot, bytes4, l0_out80)`.
        """
        ctx6_bits = bus_512[80:128].reshape(6, 8)
        weights8 = 1 << np.arange(8, dtype=np.uint8)
        ctx6 = bytes((ctx6_bits * weights8[None, :]).sum(axis=-1).astype(np.uint8).tolist())

        slot = hw_hash_ctx6(lang_id, ctx6, self.page_addr_bits)
        exp_tag = np.uint64(hw_pack_tag64(lang_id, ctx6, valid=True))
        hit = bool(self.tag_bram[slot] == exp_tag)

        self.total_lookups += 1
        if hit:
            self.total_hits += 1

        eff_tables = self.lut_bram[slot] if hit else self.fallback_tables_32x5  # [32, 5]
        p_weights = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint64)

        l0_out80 = np.zeros(80, dtype=np.uint8)
        pred_bits = np.zeros(32, dtype=np.uint8)
        for b in range(32):
            v_sum = 0
            for v in range(self.VOTERS):
                addr = int((bus_512[self.wires_32x5x6[b, v]].astype(np.uint64) * p_weights).sum())
                out_bit = int((eff_tables[b, v] >> np.uint64(addr)) & np.uint64(1))
                v_sum += out_bit
                if b < 16:
                    l0_out80[b * 5 + v] = out_bit
            pred_bits[b] = 1 if v_sum >= 3 else 0

        bytes4 = ((pred_bits.reshape(4, 8) * weights8[None, :]).sum(axis=-1)).astype(np.uint8).tolist()
        return hit, slot, [int(x) for x in bytes4], l0_out80


@dataclass
class PagedBRAMSynthesisRow:
    architecture: str
    resident_lut6_pages: int
    context_switch_cycles: int
    context_switch_ns_at_250mhz: float
    synthesized_clb_luts: int
    synthesized_ffs: int
    synthesized_ramb36_equiv: float
    dsp48_count: int


def yosys_synth_stats(sv_files: list[str], top_module: str, extra_defines: list[str] | None = None) -> dict[str, int]:
    """Runs Yosys `synth_xilinx -family xcup` and extracts exact CLB LUT, FF, BRAM, and DSP48 counts."""
    yosys = shutil.which("yosys") or str(Path.home() / ".local" / "bin" / "yosys")
    defs = " ".join(f"-D{d}" for d in (extra_defines or []))
    files_arg = " ".join(sv_files)
    cmd = [
        yosys,
        "-Q",
        "-T",
        "-p",
        f"read_verilog -sv -DSYNTHESIS {defs} {files_arg}; synth_xilinx -family xcup -top {top_module}; stat -json",
    ]
    env_vars = os.environ.copy()
    env_vars["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env_vars.get('PATH', '')}"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env_vars)
    out = proc.stdout
    j_idx = out.rfind("\n{\n")
    if j_idx == -1:
        j_idx = out.find("{")
    cells = json.loads(out[j_idx:]).get("design", {}).get("num_cells_by_type", {})

    luts = sum(int(v) for k, v in cells.items() if "LUT" in k.upper())
    ffs = sum(int(v) for k, v in cells.items() if k.upper().startswith("FD"))
    ramb36 = sum(int(v) for k, v in cells.items() if "RAMB36" in k.upper())
    ramb18 = sum(int(v) for k, v in cells.items() if "RAMB18" in k.upper())
    dsps = sum(int(v) for k, v in cells.items() if "DSP" in k.upper())
    return {
        "luts": luts,
        "ffs": ffs,
        "ramb36": ramb36,
        "ramb18": ramb18,
        "dsps": dsps,
    }


def verify_verilator_paged_bram_rtl(work_dir: Path, cache: HardwareBRAMPageCache | None = None) -> dict:
    """
    Cycle-accurate Verilator 5.x co-simulation of `clai_benes_paged_hotswap_core`:
      1. Programs canonical 512x512 Beneš-Copy-Beneš routing + Active-Bank XNOR-CAM fallback page.
      2. Programs multiple context-specific `LUT6 INIT` pages into the on-chip BRAM cache.
      3. Streams a back-to-back burst of 6-byte code contexts at **1 context per clock cycle (`CPI = 1.0`)**,
         verifying single-cycle (`4.0 ns`) `INIT` page switching, AST indentation disambiguation on the
         same 6-gram, zero-stall tag-miss fallback to XNOR-CAM copy, AND live non-blocking online
         Sleep-Crystallization (`page_prog_we`) mid-simulation!
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    verilator = shutil.which("verilator") or str(Path.home() / ".local" / "bin" / "verilator")

    if cache is None:
        cache = HardwareBRAMPageCache(page_addr_bits=9)

    # Create 3 distinct syntactic contexts (guaranteeing distinct 9-bit hashes!) + 1 unseen CAM context:
    # Context A (Python `":\n    "`): AST-branching page! Outputs `"return"` (`re` in slice0) if prev-line
    # wasn't a `def`, vs `"    "` (`  ` in slice0) if `bus[176]` (`is_def`) or higher indent is set!
    ctx_py_colon = b":\n    "
    ctx_rs_fn = b"pub fn"
    ctx_sv_alw = b"always"
    ctx_unseen = b"_x_val"
    ctx_online_edit = b"my_api"

    # Build real AST-branching page for `ctx_py_colon` (branches on `canon_ast_wires[0, 0]`)
    branch_wire = int(cache.canon_ast_wires[0, 0])
    bus_a0 = encode_code_bitbus_512(b"if ready:\n    ", lang_id=0)
    bus_a0[branch_wire] = 0
    bus_a1 = encode_code_bitbus_512(b"def foo():\n    ", lang_id=0)
    bus_a1[branch_wire] = 1
    xm_a = np.stack([bus_a0, bus_a0, bus_a1, bus_a1], axis=0)
    ym_a = np.stack(
        [
            np.frombuffer(b"retu", dtype=np.uint8),
            np.frombuffer(b"retu", dtype=np.uint8),
            np.frombuffer(b"    ", dtype=np.uint8),
            np.frombuffer(b"    ", dtype=np.uint8),
        ],
        axis=0,
    )
    tbl_a = cache.synthesize_joint_mode_tables(xm_a, ym_a)
    slot_a = cache.write_page(0, ctx_py_colon, tbl_a, priority=999_999, force_overwrite=True)

    tbl_rs = cache.synthesize_const_tables(b" new", allow_cam_override=False)
    slot_rs = cache.write_page(1, ctx_rs_fn, tbl_rs, priority=999_999, force_overwrite=True)

    tbl_sv = cache.synthesize_const_tables(b" @(p", allow_cam_override=False)
    slot_sv = cache.write_page(2, ctx_sv_alw, tbl_sv, priority=999_999, force_overwrite=True)

    assert len({slot_a, slot_rs, slot_sv}) == 3, "Test contexts collided in 9-bit hash!"
    # Ensure `ctx_unseen` misses in Tag BRAM so it tests 0-stall hardware XNOR-CAM fallback!
    slot_unseen = hw_hash_ctx6(0, ctx_unseen, cache.page_addr_bits)
    cache.tag_bram[slot_unseen] = np.uint64(0)

    # Verify software BCB-512 matches canonical slice-0 wire selection
    bus_rs = encode_code_bitbus_512(b"impl Engine {\n    pub fn", lang_id=1)
    bus_sv = encode_code_bitbus_512(b"module top;\n    always", lang_id=2)
    # Prompt with repeated `_x_val = 42` so local XNOR-CAM (`wire 452` + `wires 384..415`) fires `" = 4"`!
    bus_cam = encode_code_bitbus_512(b"let _x_val = 42;\nreturn _x_val", lang_id=0)
    assert int(bus_cam[452]) == 1, "Expected 6-gram XNOR-CAM hit on repeated local identifier!"

    stream_phase1 = [
        (bus_a0, 0, "PY_COLON_BRANCH0"),
        (bus_a1, 0, "PY_COLON_BRANCH1"),
        (bus_rs, 1, "RS_PUB_FN"),
        (bus_sv, 2, "SV_ALWAYS"),
        (bus_cam, 0, "UNSEEN_CAM_FALLBACK"),
    ]

    routed_chk = eval_multicast_bcb512(np.stack([it[0] for it in stream_phase1], axis=0), cache.slice0_bcb_cfg)
    raw_chk = np.stack([it[0] for it in stream_phase1], axis=0)[:, cache.slice0_desired_480]
    assert np.array_equal(routed_chk[:, :480], raw_chk)

    exp_phase1 = []
    for bus_v, lid_v, lbl in stream_phase1:
        hit, slot, b4, l0_80 = cache.lookup_and_eval(bus_v, lid_v)
        word16 = b4[0] | (b4[1] << 8)
        l0_u80 = sum(int(l0_80[k]) << k for k in range(80))
        exp_phase1.append((lbl, hit, slot, word16, l0_u80))

    # Phase 2: Live Online Sleep-Crystallization write for `ctx_online_edit` (`b"my_api"` -> `b"(ctx"`)
    bus_online = encode_code_bitbus_512(b"res = my_api", lang_id=0)
    slot_online = hw_hash_ctx6(0, ctx_online_edit, cache.page_addr_bits)
    cache.tag_bram[slot_online] = np.uint64(0)
    pre_hit, _, pre_b4, _ = cache.lookup_and_eval(bus_online, 0)
    assert not pre_hit

    tbl_online = cache.synthesize_const_tables(b"(ctx", allow_cam_override=False)
    cache.write_page(0, ctx_online_edit, tbl_online, priority=999_999, force_overwrite=True)
    post_hit, post_slot, post_b4, post_l0_80 = cache.lookup_and_eval(bus_online, 0)
    assert post_hit and bytes(post_b4[:2]) == b"(c"
    post_word16 = post_b4[0] | (post_b4[1] << 8)
    post_l0_u80 = sum(int(post_l0_80[k]) << k for k in range(80))

    def pack256_hex(bits256: np.ndarray) -> str:
        val = 0
        for b_i in range(256):
            if int(bits256[b_i]) == 1:
                val |= 1 << b_i
        return f"256'h{val:064X}"

    def pack512_hex(bits512: np.ndarray) -> str:
        val = 0
        for b_i in range(512):
            if int(bits512[b_i]) == 1:
                val |= 1 << b_i
        return f"512'h{val:0128X}"

    sv = [
        "`timescale 1ns/1ps",
        "module tb_clai_benes_paged;",
        "    logic        clk = 0;",
        "    logic        rst_n = 0;",
        "    always #2 clk = ~clk; // 250 MHz (4.0 ns period)",
        "    logic [511:0] bus_in = 0;",
        "    logic         bus_valid = 0;",
        "    logic [1:0]   ctx_lang_id = 0;",
        "    logic         page_cache_en = 1;",
        "    logic         cfg_we = 0;",
        "    logic [1:0]   cfg_kind = 0;",
        "    logic [6:0]   cfg_idx = 0;",
        "    logic [255:0] cfg_vec256 = 0;",
        "    logic [511:0] cfg_vec512 = 0;",
        "    logic         atomic_swap_pulse = 0;",
        "    logic         page_prog_we = 0;",
        "    logic         page_prog_is_tag = 0;",
        "    logic [8:0]   page_prog_addr = 0;",
        "    logic [6:0]   page_prog_lut_idx = 0;",
        "    logic [63:0]  page_prog_data64 = 0;",
        "",
        "    wire          active_bank;",
        "    wire          out_valid;",
        "    wire          page_hit_out;",
        "    wire  [8:0]   page_slot_out;",
        "    wire  [79:0]  l0_out_debug;",
        "    wire  [15:0]  draft_word16_out;",
        "    wire  [15:0]  draft_unanimous16_out;",
        "    wire  [31:0]  telemetry_lookups;",
        "    wire  [31:0]  telemetry_page_hits;",
        "    wire  [31:0]  telemetry_cam_fallbacks;",
        "    integer errors = 0;",
        "",
        "    clai_benes_paged_hotswap_core #(.PAGE_ADDR_BITS(9)) dut (.*);",
        "",
        "    initial begin",
        "        repeat (4) @(posedge clk); rst_n = 1; @(posedge clk);",
    ]

    # 1. Program canonical Beneš-Copy-Beneš + Active-Bank XNOR-CAM fallback into Shadow Bank -> 4ns swap
    bcb = cache.slice0_bcb_cfg
    for st in range(17):
        sv.append(
            f"        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd0; cfg_idx <= 7'd{st}; cfg_vec256 <= {pack256_hex(bcb.sw_b1[st])};"
        )
    for st in range(9):
        sv.append(
            f"        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd1; cfg_idx <= 7'd{st}; cfg_vec512 <= {pack512_hex(bcb.copy_en[st])};"
        )
    for st in range(17):
        sv.append(
            f"        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd2; cfg_idx <= 7'd{st}; cfg_vec256 <= {pack256_hex(bcb.sw_b2[st])};"
        )
    fb_flat = cache.fallback_tables_32x5[:16].reshape(80)
    for gl in range(80):
        sv.append(
            f"        @(posedge clk); cfg_we <= 1; cfg_kind <= 2'd3; cfg_idx <= 7'd{gl}; cfg_vec256 <= {{192'd0, 64'h{int(fb_flat[gl]):016X}}};"
        )
    sv.extend(
        [
            "        @(posedge clk); cfg_we <= 0; atomic_swap_pulse <= 1;",
            "        @(posedge clk); atomic_swap_pulse <= 0;",
        ]
    )

    # 2. Program the 3 pre-compiled code pages into the BRAM Page Cache (`slot_a`, `slot_rs`, `slot_sv`)
    def emit_bram_page_prog(slot: int, lid: int, ctx6: bytes, tbl_32x5: np.ndarray) -> None:
        tag_u64 = hw_pack_tag64(lid, ctx6, valid=True)
        flat80 = tbl_32x5[:16].reshape(80)
        for gl in range(80):
            sv.append(
                f"        @(posedge clk); page_prog_we <= 1; page_prog_is_tag <= 0; page_prog_addr <= 9'd{slot}; page_prog_lut_idx <= 7'd{gl}; page_prog_data64 <= 64'h{int(flat80[gl]):016X};"
            )
        sv.append(
            f"        @(posedge clk); page_prog_we <= 1; page_prog_is_tag <= 1; page_prog_addr <= 9'd{slot}; page_prog_data64 <= 64'h{tag_u64:016X};"
        )

    emit_bram_page_prog(slot_a, 0, ctx_py_colon, tbl_a)
    emit_bram_page_prog(slot_rs, 1, ctx_rs_fn, tbl_rs)
    emit_bram_page_prog(slot_sv, 2, ctx_sv_alw, tbl_sv)
    sv.append("        @(posedge clk); page_prog_we <= 0;")
    sv.append("        @(posedge clk);")

    # 3. Stream all 5 distinct contexts BACK-TO-BACK at 1 context per clock cycle (`CPI = 1.0`, zero bubbles!)
    sv.append("        fork")
    sv.append("            begin : drive_back_to_back")
    for bus_v, lid_v, _ in stream_phase1:
        sv.append(
            f"                bus_valid <= 1; ctx_lang_id <= 2'd{lid_v}; bus_in <= {pack512_hex(bus_v)}; @(posedge clk);"
        )
    sv.append("                bus_valid <= 0; @(posedge clk);")
    sv.append("            end")
    sv.append("            begin : check_one_cycle_later")
    sv.append("                @(posedge clk); // 1-cycle synchronous BRAM pipeline latency")
    for idx_s, (lbl, hit, slot, word16, l0_u80) in enumerate(exp_phase1):
        hit_b = 1 if hit else 0
        sv.extend(
            [
                "                #1;",
                f"                if (out_valid !== 1'b1 || page_hit_out !== 1'b{hit_b} || page_slot_out !== 9'd{slot} || draft_word16_out !== 16'h{word16:04X} || l0_out_debug !== 80'h{l0_u80:020X}) begin",
                f'                    $display("STREAM BEAT {idx_s} ({lbl}) MISMATCH: hit=%b(exp {hit_b}) slot=%0d(exp {slot}) w16=%04x(exp {word16:04X})", page_hit_out, page_slot_out, draft_word16_out);',
                "                    errors = errors + 1;",
                "                end",
                "                @(posedge clk);",
            ]
        )
    sv.append("            end")
    sv.append("        join")

    # 4. Live Online Sleep-Crystallization Test (`my_api` misses before prog, hits `(c` immediately after!)
    sv.append(f"        bus_valid <= 1; ctx_lang_id <= 2'd0; bus_in <= {pack512_hex(bus_online)}; @(posedge clk);")
    sv.append("        bus_valid <= 0; #1;")
    sv.append("        if (page_hit_out !== 1'b0) begin $display(\"PRE-CRYST EXPECTED MISS\"); errors = errors + 1; end")
    emit_bram_page_prog(post_slot, 0, ctx_online_edit, tbl_online)
    sv.append("        @(posedge clk); page_prog_we <= 0;")
    sv.append(f"        @(posedge clk); bus_valid <= 1; ctx_lang_id <= 2'd0; bus_in <= {pack512_hex(bus_online)};")
    sv.append("        @(posedge clk); bus_valid <= 0; #1;")
    sv.extend(
        [
            f"        if (page_hit_out !== 1'b1 || draft_word16_out !== 16'h{post_word16:04X} || l0_out_debug !== 80'h{post_l0_u80:020X}) begin",
            f'            $display("POST-CRYST MISMATCH: hit=%b w16=%04x exp={post_word16:04X}", page_hit_out, draft_word16_out);',
            "            errors = errors + 1;",
            "        end",
            "        @(posedge clk); #1;",
            "        if (telemetry_lookups !== 32'd7 || telemetry_page_hits !== 32'd5 || telemetry_cam_fallbacks !== 32'd2) begin",
            '            $display("TELEMETRY MISMATCH: lookups=%0d hits=%0d fb=%0d", telemetry_lookups, telemetry_page_hits, telemetry_cam_fallbacks);',
            "            errors = errors + 1;",
            "        end",
            '        $display("[Verilator KV260 Paged-BRAM Benes Core] CPI=1.0 Back-to-Back 4ns Page Switch + CAM Fallback + Live Sleep-Crystallization | Lookups=%0d Hits=%0d Fallbacks=%0d | Errors=%0d", telemetry_lookups, telemetry_page_hits, telemetry_cam_fallbacks, errors);',
            "        if (errors != 0) $fatal(1);",
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )

    tb_path = work_dir / "tb_clai_benes_paged.sv"
    tb_path.write_text("\n".join(sv), encoding="utf-8")
    obj_dir = work_dir / "verilator_benes_paged_obj"

    env_vars = os.environ.copy()
    env_vars["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env_vars.get('PATH', '')}"

    subprocess.run(
        [
            verilator,
            "--binary",
            "--timing",
            "-j",
            "4",
            "-Wno-UNOPTFLAT",
            "-Wno-DECLFILENAME",
            "-Wno-UNUSED",
            "--Mdir",
            str(obj_dir),
            "--top-module",
            "tb_clai_benes_paged",
            "hw/kv260/rtl/clai_benes_hotswap_core.sv",
            str(tb_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env_vars,
    )
    res = subprocess.run([str(obj_dir / "Vtb_clai_benes_paged")], check=True, capture_output=True, text=True)
    msg = res.stdout.strip()
    print(msg)
    return {
        "verilator_banner": msg,
        "cpi": 1.0,
        "lookups": 7,
        "page_hits": 5,
        "cam_fallbacks": 2,
        "errors": 0,
    }


def run_paged_bram_silicon_comparison(work_dir: Path) -> tuple[list[PagedBRAMSynthesisRow], dict]:
    """
    Compares 3 Multi-Context LUT6 Hardware Switching Mechanisms on Kria KV260 (`xck26`):
      1. `AXI4Lite_ShadowBank_Reload` (`clai_benes_hotswap_core` — 1 resident page, requires 81 AXI
         writes = `324 ns` stall per context page switch!)
      2. `Distributed_FF_MultiPage_512` (512 pages x 5,171 bits in CLB FFs + 512:1 LUT muxes —
         exceeds KV260 `234,240` FF & `117,120` LUT budgets by >11x!)
      3. `MoL_BRAM_Paged_Benes_512_OURS` (`clai_benes_paged_hotswap_core` — 512 resident `80-LUT6`
         pages in `81` `RAMB36E2` Block RAMs, **1-cycle = `4.0 ns` zero-stall page switch**, `0 DSP48`s!)
    """
    cosim = verify_verilator_paged_bram_rtl(work_dir)
    base_stats = yosys_synth_stats(["hw/kv260/rtl/clai_benes_hotswap_core.sv"], "clai_benes_hotswap_core")
    paged_stats = yosys_synth_stats(["hw/kv260/rtl/clai_benes_hotswap_core.sv"], "clai_benes_paged_hotswap_core")

    ramb36_equiv = paged_stats["ramb36"] + 0.5 * paged_stats["ramb18"]
    # Analytical exact count for 512 pages stored in CLB Flip-Flops + 512:1 read muxes (`81 * 64 = 5,184` bits/page)
    ff_multipage_ffs = base_stats["ffs"] + 512 * 5184
    ff_multipage_luts = base_stats["luts"] + 5184 * 171  # 512:1 mux takes ~171 LUT6s per bit!

    rows = [
        PagedBRAMSynthesisRow(
            architecture="1_SinglePage_ShadowBank_Reload_Baseline",
            resident_lut6_pages=1,
            context_switch_cycles=82,
            context_switch_ns_at_250mhz=328.0,
            synthesized_clb_luts=base_stats["luts"],
            synthesized_ffs=base_stats["ffs"],
            synthesized_ramb36_equiv=base_stats["ramb36"] + 0.5 * base_stats["ramb18"],
            dsp48_count=base_stats["dsps"],
        ),
        PagedBRAMSynthesisRow(
            architecture="2_Distributed_FF_Mux_512Pages_Baseline",
            resident_lut6_pages=512,
            context_switch_cycles=1,
            context_switch_ns_at_250mhz=4.0,
            synthesized_clb_luts=ff_multipage_luts,
            synthesized_ffs=ff_multipage_ffs,
            synthesized_ramb36_equiv=0.0,
            dsp48_count=0,
        ),
        PagedBRAMSynthesisRow(
            architecture="3_MoL_BRAM_HashIndexed_Benes_512Pages_OURS",
            resident_lut6_pages=512,
            context_switch_cycles=1,
            context_switch_ns_at_250mhz=4.0,
            synthesized_clb_luts=paged_stats["luts"],
            synthesized_ffs=paged_stats["ffs"],
            synthesized_ramb36_equiv=ramb36_equiv,
            dsp48_count=paged_stats["dsps"],
        ),
    ]
    return rows, {"base_yosys": base_stats, "paged_yosys": paged_stats, **cosim}
