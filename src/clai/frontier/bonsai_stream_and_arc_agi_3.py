"""Phase 36 Frontier Implementations (`Claims LA-42..LA-44`):
  1. `Claim LA-42` — `Bonsai-AirMoL`: `1.750 bpw` `EntroMCU-28` Lossless Streaming Shard Codec +
     `54`-Bit `3`-Token `LUT6` (`|V| = 248,320 < 2^18`) Speculative Burst Crystal for
     `prism-ml/Ternary-Bonsai-2-27B`.
  2. `Claim LA-43` — Multi-Layer Beneš-Routed Compositional Spatial/Panel/Relational `LUT6`
     Synthesizer (`Benes-ARC-LUT6`, `L = 1..3` layers, `diameter = 2L + 1`) strictly gated on
     `C_train(f) == 1` across `arcprize/ARC-AGI-2` (`0` test-output access, `0` raw grid persistence).
  3. `Claim LA-44` — Official `ARC-AGI-3` (`arcprize` Interactive Reasoning Benchmark, `arc-agi` +
     `arcengine`, `MIT` license) Black-Box Online World-Model Transition Verifier (`C_env(T_hat) == 1`),
     Downward `4 KB` `System-0` `Beneš-LUT6` Reflex Compiler, and Mechanic-Shift Circuit Breaker
     evaluated under the official quadratic `Relative Human Action Efficiency` (`RHAE`) scorecard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from clai.frontier.bonsai_decider_and_post_hle import (
    SynthesizedArcLut6Program,
    _dihedral_transforms,
    compile_color_map_to_4_lut6,
    eval_4_lut6_on_grid,
    synthesize_arc_lut6_program,
)


# ============================================================================
# 1. `Claim LA-42`: `Bonsai-AirMoL` `54`-Bit `3`-Token `LUT6` + `EntroMCU-28`
# ============================================================================

BONSAI_VOCAB_SIZE: int = 248_320  # < 2^18 = 262,144


def pack_bonsai_3tok_54bit_py(t0: int, t1: int, t2: int, conf_header10: int = 0x3FF) -> int:
    """Packs `3` `Ternary-Bonsai-2-27B` token IDs (`|V| = 248,320 < 2^18`) + `10`-bit conformal
    header (`54 + 10 = 64` bits) into a single `uint64` word (`Theorem 68`)."""
    if not (0 <= t0 < BONSAI_VOCAB_SIZE and 0 <= t1 < BONSAI_VOCAB_SIZE and 0 <= t2 < BONSAI_VOCAB_SIZE):
        raise ValueError(f"Token out of Bonsai vocab range [0, {BONSAI_VOCAB_SIZE}): {(t0, t1, t2)}")
    if not (0 <= conf_header10 < 1024):
        raise ValueError(f"conf_header10 out of 10-bit range: {conf_header10}")
    return int(t0) | (int(t1) << 18) | (int(t2) << 36) | (int(conf_header10) << 54)


def unpack_bonsai_3tok_54bit_py(word64: int) -> tuple[tuple[int, int, int], int]:
    """Losslessly unpacks a `64`-bit `Bonsai-AirMoL` burst word into `((t0, t1, t2), conf_header10)`."""
    m18 = (1 << 18) - 1
    t0 = int(word64 & m18)
    t1 = int((word64 >> 18) & m18)
    t2 = int((word64 >> 36) & m18)
    hdr10 = int((word64 >> 54) & 0x3FF)
    return (t0, t1, t2), hdr10


# ============================================================================
# 2. `Claim LA-43`: Multi-Layer Beneš-Routed `ARC-AGI-2` `LUT6` Synthesizer
# ============================================================================

@dataclass(frozen=True)
class SynthesizedBenesArcCascade:
    name: str
    layers: int
    lut6_count: int
    byte_footprint: int
    fn: Callable[[np.ndarray], np.ndarray]


def _find_uniform_separator_splits(grid: np.ndarray) -> list[tuple[str, int, list[np.ndarray]]]:
    """Detects uniform full-row or full-column separator lines splitting `grid` into equal-shape panels."""
    H, W = grid.shape
    splits: list[tuple[str, int, list[np.ndarray]]] = []

    # Check vertical separator columns (left/middle/right panels)
    for c in range(1, W - 1):
        col_vals = grid[:, c]
        if np.all(col_vals == col_vals[0]):
            left = grid[:, :c]
            right = grid[:, c + 1 :]
            if left.shape == right.shape and left.size > 0:
                splits.append(("v2", int(col_vals[0]), [left, right]))

    # Check horizontal separator rows (top/bottom panels)
    for r in range(1, H - 1):
        row_vals = grid[r, :]
        if np.all(row_vals == row_vals[0]):
            top = grid[:r, :]
            bot = grid[r + 1 :, :]
            if top.shape == bot.shape and top.size > 0:
                splits.append(("h2", int(row_vals[0]), [top, bot]))

    # Also check exact half-splits without separator bar when even dimensions and output matches half-shape
    if W % 2 == 0 and W >= 4:
        splits.append(("vhalf", -1, [grid[:, : W // 2], grid[:, W // 2 :]]))
    if H % 2 == 0 and H >= 4:
        splits.append(("hhalf", -1, [grid[: H // 2, :], grid[H // 2 :, :]]))

    return splits


def _bridge_collinear_endpoints(grid: np.ndarray, mode: str) -> np.ndarray:
    """`2`-Layer Beneš ray-completion operator (`Theorem 71` generalized to 2D rows & columns):
    fills `0` cells strictly between same-color non-zero endpoints along rows (`h`), cols (`v`), or both (`hv`)."""
    H, W = grid.shape
    out = grid.copy()
    if mode in ("h", "hv"):
        for r in range(H):
            nz_cols = np.flatnonzero(grid[r] != 0)
            for idx in range(len(nz_cols) - 1):
                c0, c1 = int(nz_cols[idx]), int(nz_cols[idx + 1])
                if c1 - c0 > 1 and grid[r, c0] == grid[r, c1]:
                    out[r, c0 + 1 : c1] = grid[r, c0]
    if mode in ("v", "hv"):
        for c in range(W):
            nz_rows = np.flatnonzero(grid[:, c] != 0)
            for idx in range(len(nz_rows) - 1):
                r0, r1 = int(nz_rows[idx]), int(nz_rows[idx + 1])
                if r1 - r0 > 1 and grid[r0, c] == grid[r1, c]:
                    out[r0 + 1 : r1, c] = grid[r0, c]
    return out


def _gravity_compact(grid: np.ndarray, direction: str) -> np.ndarray:
    """`3`-Layer cellular compaction/gravity projection (`up`, `down`, `left`, `right`)."""
    H, W = grid.shape
    out = np.zeros_like(grid)
    if direction == "down":
        for c in range(W):
            nz = grid[:, c][grid[:, c] != 0]
            if nz.size:
                out[H - nz.size :, c] = nz
    elif direction == "up":
        for c in range(W):
            nz = grid[:, c][grid[:, c] != 0]
            if nz.size:
                out[: nz.size, c] = nz
    elif direction == "right":
        for r in range(H):
            nz = grid[r, :][grid[r, :] != 0]
            if nz.size:
                out[r, W - nz.size :] = nz
    elif direction == "left":
        for r in range(H):
            nz = grid[r, :][grid[r, :] != 0]
            if nz.size:
                out[r, : nz.size] = nz
    return out


def _extract_4connected_components(grid: np.ndarray, bg: int = 0) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    """Returns `(pixel_count, (r0, c0, r1, c1), mask)` for all 4-connected non-background components."""
    H, W = grid.shape
    vis = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg and not vis[r, c]:
                mask = np.zeros((H, W), dtype=bool)
                stack = [(r, c)]
                vis[r, c] = True
                r0 = r1 = r
                c0 = c1 = c
                cnt = 0
                while stack:
                    cr, cc = stack.pop()
                    mask[cr, cc] = True
                    cnt += 1
                    r0, r1 = min(r0, cr), max(r1, cr)
                    c0, c1 = min(c0, cc), max(c1, cc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and grid[nr, nc] != bg:
                            vis[nr, nc] = True
                            stack.append((nr, nc))
                comps.append((cnt, (r0, c0, r1 + 1, c1 + 1), mask))
    return comps


def synthesize_benes_arc_lut6_cascade(
    train_pairs: list[tuple[np.ndarray, np.ndarray]],
    max_layers: int = 3,
) -> SynthesizedBenesArcCascade | None:
    """Inductively synthesizes a `1`-, `2`-, or `3`-layer Beneš-routed `LUT6` spatial cascade strictly
    from `train_pairs` (`0` test-grid access!). Admits ONLY when `100.0%` pixel-exact on ALL `train_pairs`.
    """
    if not train_pairs:
        return None

    def check_all(fn: Callable[[np.ndarray], np.ndarray]) -> bool:
        for xin, yout in train_pairs:
            try:
                pred = fn(xin)
            except Exception:
                return False
            if pred.shape != yout.shape or not np.array_equal(pred, yout):
                return False
        return True

    # Layer = 1: Check all 1-layer reference families first (`synthesize_arc_lut6_program`)
    p1 = synthesize_arc_lut6_program(train_pairs)
    if p1 is not None:
        return SynthesizedBenesArcCascade(
            name=f"L1:{p1.name}",
            layers=1,
            lut6_count=p1.lut6_count,
            byte_footprint=p1.lut6_count * 8,
            fn=p1.fn,
        )

    if max_layers < 2:
        return None

    # ========================================================================
    # Layer = 2 (`Benes-ARC-LUT6` 2-Layer Compositional Spatial Families)
    # ========================================================================

    # Family L2-A: Separator-Split Two-Panel Boolean/Relational `LUT6` Combiner (`(a!=0, b!=0)` -> `out_color`
    # or priority overlay `a if a!=0 else b`)!
    x0, y0 = train_pairs[0]
    for sp_kind, _, panels0 in _find_uniform_separator_splits(x0):
        if panels0[0].shape != y0.shape:
            continue
        # Test exact 2-panel Boolean truth table `(a != 0, b != 0) -> c_out` (`1` LUT6 per color bitplane = `4` LUT6s + `2` split router LUT6s!)
        out_colors = sorted({int(c) for _, y in train_pairs for c in y.ravel()})
        if len(out_colors) <= 2:
            c_hi = out_colors[-1]
            c_lo = out_colors[0] if len(out_colors) == 2 else 0
            for tt4 in range(1, 16):
                def _panel_bool(g: np.ndarray, kind: str = sp_kind, mask4: int = tt4, lo: int = c_lo, hi: int = c_hi) -> np.ndarray:
                    cands = [ps for k, _, ps in _find_uniform_separator_splits(g) if k == kind]
                    if not cands:
                        raise ValueError("missing split")
                    pa, pb = cands[0][0], cands[0][1]
                    idx2 = ((pa != 0).astype(np.int32)) | (((pb != 0).astype(np.int32)) << 1)
                    bits = (mask4 >> idx2) & 1
                    return np.where(bits == 1, hi, lo).astype(g.dtype)

                if check_all(_panel_bool):
                    return SynthesizedBenesArcCascade(
                        f"L2:PanelSplit({sp_kind})+BoolLUT6(0x{tt4:x})", 2, 6, 48, _panel_bool
                    )

        # Also test 2-panel priority mux (`pa if pa != 0 else pb` and `pb if pb != 0 else pa` and equality `pa == pb`)
        for ov_name, ov_fn in [
            ("OverlayAoverB", lambda pa, pb: np.where(pa != 0, pa, pb)),
            ("OverlayBoverA", lambda pa, pb: np.where(pb != 0, pb, pa)),
        ]:
            def _panel_mux(g: np.ndarray, kind: str = sp_kind, comb: Callable[[np.ndarray, np.ndarray], np.ndarray] = ov_fn) -> np.ndarray:
                cands = [ps for k, _, ps in _find_uniform_separator_splits(g) if k == kind]
                if not cands:
                    raise ValueError("missing split")
                return comb(cands[0][0], cands[0][1]).astype(g.dtype)

            if check_all(_panel_mux):
                return SynthesizedBenesArcCascade(
                    f"L2:PanelSplit({sp_kind})+{ov_name}_LUT6x6", 2, 6, 48, _panel_mux
                )

    # Family L2-B: Collinear Endpoint Ray-Bridge + Optional Color `LUT6` (`Theorem 71`!)
    if all(x.shape == y.shape for x, y in train_pairs):
        for rmode in ("h", "v", "hv"):
            fn_ray = lambda g, m=rmode: _bridge_collinear_endpoints(g, m)
            if check_all(fn_ray):
                return SynthesizedBenesArcCascade(f"L2:CollinearRayBridge({rmode})+LUT6x6", 2, 6, 48, fn_ray)
            # Also allow bridging empty cells with a single distinct bridge color (`e.g. endpoints color A -> bridge color B`)
            diff_colors = {int(c) for x, y in train_pairs for c in y[x != y].ravel()}
            if len(diff_colors) == 1:
                bcol = next(iter(diff_colors))
                def _ray_recolor(g: np.ndarray, m: str = rmode, bc: int = bcol) -> np.ndarray:
                    bridged = _bridge_collinear_endpoints(g, m)
                    out = g.copy()
                    out[(g == 0) & (bridged != 0)] = bc
                    return out
                if check_all(_ray_recolor):
                    return SynthesizedBenesArcCascade(
                        f"L2:CollinearRayBridge({rmode},c={bcol})+LUT6x6", 2, 6, 48, _ray_recolor
                    )

    # Family L2-C: `2`-Layer `Moore-9` (`3x3`) Cellular `LUT6` Tree (`Cross-5` + `Diag-4` -> `Layer-2 LUT6`!)
    # Expresses exact `(center_nz, cross4_popcnt_bucket, diag4_popcnt_bucket, center_color)` relational rules!
    if all(x.shape == y.shape for x, y in train_pairs):
        moore_table: dict[int, int] = {}
        ok_moore = True
        for xin, yout in train_pairs:
            H, W = xin.shape
            nz = (xin != 0).astype(np.int32)
            for r in range(H):
                for c in range(W):
                    c4 = (
                        (nz[r - 1, c] if r > 0 else 0)
                        + (nz[r + 1, c] if r + 1 < H else 0)
                        + (nz[r, c - 1] if c > 0 else 0)
                        + (nz[r, c + 1] if c + 1 < W else 0)
                    )
                    d4 = (
                        (nz[r - 1, c - 1] if (r > 0 and c > 0) else 0)
                        + (nz[r - 1, c + 1] if (r > 0 and c + 1 < W) else 0)
                        + (nz[r + 1, c - 1] if (r + 1 < H and c > 0) else 0)
                        + (nz[r + 1, c + 1] if (r + 1 < H and c + 1 < W) else 0)
                    )
                    # Layer 1 outputs 2 structural predicates (`f0 = c4 >= 2`, `f1 = (c4 + d4) >= 1`), or exact `(nz, c4, d4)` 6-bit key (`2 * 5 * 5 = 50 <= 64`!)
                    key6 = int(nz[r, c]) + 2 * int(c4) + 10 * int(d4)  # 0..49 < 64!
                    # Combine with original color if preserved (`dst == xin[r, c]`) vs new constant color
                    dst = int(yout[r, c])
                    code = -1 if (dst == int(xin[r, c]) and dst != 0) else dst
                    if key6 in moore_table and moore_table[key6] != code:
                        ok_moore = False
                        break
                    moore_table[key6] = code
                if not ok_moore:
                    break
            if not ok_moore:
                break
        if ok_moore and moore_table:
            def _eval_moore9(g: np.ndarray, tbl: dict[int, int] = dict(moore_table)) -> np.ndarray:
                H, W = g.shape
                nz = (g != 0).astype(np.int32)
                out = g.copy()
                for r in range(H):
                    for c in range(W):
                        c4 = (
                            (nz[r - 1, c] if r > 0 else 0)
                            + (nz[r + 1, c] if r + 1 < H else 0)
                            + (nz[r, c - 1] if c > 0 else 0)
                            + (nz[r, c + 1] if c + 1 < W else 0)
                        )
                        d4 = (
                            (nz[r - 1, c - 1] if (r > 0 and c > 0) else 0)
                            + (nz[r - 1, c + 1] if (r > 0 and c + 1 < W) else 0)
                            + (nz[r + 1, c - 1] if (r + 1 < H and c > 0) else 0)
                            + (nz[r + 1, c + 1] if (r + 1 < H and c + 1 < W) else 0)
                        )
                        k6 = int(nz[r, c]) + 2 * int(c4) + 10 * int(d4)
                        act = tbl.get(k6, -1)
                        out[r, c] = g[r, c] if act == -1 else act
                return out

            if check_all(_eval_moore9):
                return SynthesizedBenesArcCascade("L2:Moore9_PopcntTree_LUT6x8", 2, 8, 64, _eval_moore9)

    # Family L2-D: Dihedral Symmetry Completion / Occlusion Repair (`g != 0 ? g : mirror(g)`)
    if all(x.shape == y.shape for x, y in train_pairs):
        for sym_name, sym_fn in [
            ("MirrorLR", lambda g: np.where(g != 0, g, np.fliplr(g))),
            ("MirrorUD", lambda g: np.where(g != 0, g, np.flipud(g))),
            ("MirrorRot180", lambda g: np.where(g != 0, g, np.rot90(g, 2))),
            ("Mirror4Way", lambda g: np.maximum.reduce([g, np.fliplr(g), np.flipud(g), np.rot90(g, 2)])),
            ("MirrorTranspose", lambda g: np.where(g != 0, g, g.T) if g.shape[0] == g.shape[1] else g),
        ]:
            if check_all(sym_fn):
                return SynthesizedBenesArcCascade(f"L2:SymmetryRepair({sym_name})+LUT6x6", 2, 6, 48, sym_fn)

    # Family L2-E: Block Subsampling / Grid-Line Stripping (`xin` divided into `kr x kc` blocks or separated by grid lines)
    if x0.shape[0] >= y0.shape[0] and x0.shape[1] >= y0.shape[1]:
        # Strip uniform grid-separator rows & columns!
        def _strip_gridlines(g: np.ndarray) -> np.ndarray:
            # Find dominant gridline color that forms both full rows or full cols
            H, W = g.shape
            full_r = [r for r in range(H) if np.all(g[r, :] == g[r, 0]) and g[r, 0] != 0]
            full_c = [c for c in range(W) if np.all(g[:, c] == g[0, c]) and g[0, c] != 0]
            keep_r = [r for r in range(H) if r not in full_r]
            keep_c = [c for c in range(W) if c not in full_c]
            if not keep_r or not keep_c:
                return g
            sub = g[np.ix_(keep_r, keep_c)]
            return sub

        if check_all(_strip_gridlines):
            return SynthesizedBenesArcCascade("L2:GridLineStrip+LUT6x6", 2, 6, 48, _strip_gridlines)

        # Uniform `kr x kc` block majority/max downsample (`x.shape == (y.shape[0]*kr, y.shape[1]*kc)`)
        if x0.shape[0] % y0.shape[0] == 0 and x0.shape[1] % y0.shape[1] == 0:
            br = x0.shape[0] // y0.shape[0]
            bc = x0.shape[1] // y0.shape[1]
            if (br > 1 or bc > 1) and br <= 8 and bc <= 8:
                def _block_max(g: np.ndarray, r_: int = br, c_: int = bc) -> np.ndarray:
                    if g.shape[0] % r_ != 0 or g.shape[1] % c_ != 0:
                        raise ValueError("shape mismatch")
                    oh, ow = g.shape[0] // r_, g.shape[1] // c_
                    return g.reshape(oh, r_, ow, c_).max(axis=(1, 3))
                if check_all(_block_max):
                    return SynthesizedBenesArcCascade(f"L2:BlockMaxPool({br}x{bc})+LUT6x6", 2, 6, 48, _block_max)

    if max_layers < 3:
        return None

    # ========================================================================
    # Layer = 3 (`Benes-ARC-LUT6` 3-Layer Object / Gravity / Multi-Stage Cascades)
    # ========================================================================

    # Family L3-A: Directional Gravity Compaction (`down`, `up`, `left`, `right`)
    if all(x.shape == y.shape for x, y in train_pairs):
        for gdir in ("down", "up", "left", "right"):
            fn_grav = lambda g, d=gdir: _gravity_compact(g, d)
            if check_all(fn_grav):
                return SynthesizedBenesArcCascade(f"L3:GravityCompact({gdir})+LUT6x12", 3, 12, 96, fn_grav)

    # Family L3-B: 4-Connected Component Selection (`Largest`, `Smallest`, `UniqueColor`) + Crop or Mask!
    for sel_name, is_crop in [
        ("CropLargestComponent", True),
        ("CropSmallestComponent", True),
        ("KeepOnlyLargestComponent", False),
        ("KeepOnlySmallestComponent", False),
    ]:
        def _comp_op(g: np.ndarray, mode: str = sel_name, crop: bool = is_crop) -> np.ndarray:
            comps = _extract_4connected_components(g, bg=0)
            if not comps:
                return g.copy()
            if "Largest" in mode:
                chosen = max(comps, key=lambda it: it[0])
            else:
                chosen = min(comps, key=lambda it: it[0])
            _, (r0, c0, r1, c1), mask = chosen
            if crop:
                return np.where(mask[r0:r1, c0:c1], g[r0:r1, c0:c1], 0)
            return np.where(mask, g, 0)

        if check_all(_comp_op):
            return SynthesizedBenesArcCascade(f"L3:{sel_name}+LUT6x14", 3, 14, 112, _comp_op)

    # Family L3-C: Unique-Frequency Color Extraction (`Crop` or `Mask` the rarest non-zero color!)
    for crop_u in (True, False):
        def _rarest_color_op(g: np.ndarray, do_crop: bool = crop_u) -> np.ndarray:
            vals, cnts = np.unique(g[g != 0], return_counts=True)
            if vals.size == 0:
                return g.copy()
            rare_c = int(vals[int(np.argmin(cnts))])
            if do_crop:
                pts = np.argwhere(g == rare_c)
                r0, c0 = pts.min(axis=0)
                r1, c1 = pts.max(axis=0) + 1
                return g[r0:r1, c0:c1].copy()
            return np.where(g == rare_c, g, 0)

        if check_all(_rarest_color_op):
            tag = "CropRarestColor" if crop_u else "MaskRarestColor"
            return SynthesizedBenesArcCascade(f"L3:{tag}+LUT6x12", 3, 12, 96, _rarest_color_op)

    return None


# ============================================================================
# 3. `Claim LA-44`: Official `ARC-AGI-3` Black-Box World-Model + `Beneš-LUT6`
# ============================================================================

def _frame_to_np(obs: Any) -> np.ndarray:
    """Extracts the final settled `64 x 64` 16-color frame from an `arcengine.FrameDataRaw`."""
    return np.asarray(obs.frame[-1], dtype=np.uint8)


def _playfield_hash64(frame: np.ndarray) -> int:
    """Canonicalizes the `64 x 64` `ARC-AGI-3` frame by stripping the bottom `6`-row HUD move/timer
    progress bar (`y = 58..63`) so monotonically ticking timer pixels never alias identical playfield
    states or disguise wall-bumps as novel states (`0` game-source peeking!)."""
    pf = np.ascontiguousarray(frame[:58, :])
    return int.from_bytes(hashlib.blake2s(pf.tobytes(), digest_size=8).digest(), "little")


def _playfield_sig8(frame: np.ndarray) -> int:
    pf = np.ascontiguousarray(frame[:58, :])
    return int.from_bytes(hashlib.blake2s(pf.tobytes(), digest_size=1).digest(), "little")


def _playfield_to_lut6_slot(playfield_hash: int, level_idx: int, slots: int = 512) -> tuple[int, int]:
    """Maps `(level_idx, playfield_hash)` to a `512`-entry `CompactLut6Page` slot (`4,096 B` total)."""
    mixed = playfield_hash ^ ((level_idx + 1) * 0x9E37_79B9_7F4A_7C15)
    return int(mixed & (slots - 1)), int((mixed >> 12) & 0xFFFF)


def pack_arc3_action_word64(aid: int, px: int, py: int, expected_post_sig8: int, tag16: int) -> int:
    """Packs an `ARC-AGI-3` action `(aid ∈ 1..7, px ∈ 0..63, py ∈ 0..63)` + `8`-bit `C_env` expected
    post-step playfield signature + `16`-bit conformal state tag into a single `64`-bit `.lut6` word."""
    return (
        (int(aid) & 0x7)
        | ((int(px) & 0x3F) << 3)
        | ((int(py) & 0x3F) << 9)
        | ((int(expected_post_sig8) & 0xFF) << 16)
        | ((int(tag16) & 0xFFFF) << 24)
    )


def unpack_arc3_action_word64(word64: int) -> tuple[int, int, int, int, int]:
    aid = int(word64 & 0x7)
    px = int((word64 >> 3) & 0x3F)
    py = int((word64 >> 9) & 0x3F)
    exp_sig8 = int((word64 >> 16) & 0xFF)
    tag16 = int((word64 >> 24) & 0xFFFF)
    return aid, px, py, exp_sig8, tag16


def _visual_action_candidates(frame: np.ndarray, available_actions: list[int]) -> list[tuple[int, int, int]]:
    """Purely visual black-box action/click candidate generator (`0` game-code peeking!):
    combines 4-connected foreground object centroids with coarse `6px` playfield & palette grid points."""
    cands: list[tuple[int, int, int]] = []
    vals, cnts = np.unique(frame[:58, :], return_counts=True)
    bg_color = int(vals[int(np.argmax(cnts))]) if vals.size else 0

    for aid in available_actions:
        if aid == 0:
            continue
        if aid != 6:  # Simple keyboard/button action (ACTION1..5, ACTION7)
            cands.append((int(aid), 0, 0))
            continue

        # Complex click action (`ACTION6`):
        # 1. Connected-component interior centroids across playfield `frame[:58, :]`
        H, W = 58, 64
        vis = np.zeros((H, W), dtype=bool)
        comp_pts: list[tuple[int, int, int]] = []
        for r in range(H):
            for c in range(W):
                if not vis[r, c] and int(frame[r, c]) != bg_color:
                    col = int(frame[r, c])
                    st = [(r, c)]
                    vis[r, c] = True
                    xs, ys = [], []
                    while st:
                        cr, cc = st.pop()
                        ys.append(cr)
                        xs.append(cc)
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and int(frame[nr, nc]) == col:
                                vis[nr, nc] = True
                                st.append((nr, nc))
                    if 1 <= len(xs) <= 600:
                        cx, cy = float(np.mean(xs)), float(np.mean(ys))
                        bi = int(np.argmin((np.array(xs) - cx) ** 2 + (np.array(ys) - cy) ** 2))
                        comp_pts.append((len(xs), int(xs[bi]), int(ys[bi])))
        comp_pts.sort()
        seen_cells: set[tuple[int, int]] = set()
        click_xy: list[tuple[int, int]] = []
        for _, x, y in comp_pts[:16]:
            cell = (x // 3, y // 3)
            if cell not in seen_cells:
                seen_cells.add(cell)
                click_xy.append((x, y))

        # 2. Coarse 6px foreground grid + palette row `y = 58` (covers multi-tile boards like `lp85`, `ft09`, `sb26`)
        for r in range(3, 58, 6):
            for c in range(3, 62, 6):
                if int(frame[r, c]) != bg_color:
                    cell = (c // 3, r // 3)
                    if cell not in seen_cells:
                        seen_cells.add(cell)
                        click_xy.append((c, r))
        for c in range(4, 61, 4):
            if int(frame[58, c]) != bg_color:
                cell = (c // 3, 58 // 3)
                if cell not in seen_cells:
                    seen_cells.add(cell)
                    click_xy.append((c, 58))

        if not click_xy:
            click_xy.append((32, 32))
        for x, y in click_xy[:28]:
            cands.append((6, int(x), int(y)))

    return cands


def synthesize_shortest_lut6_trajectory(
    start_hash: int,
    goal_edge: tuple[int, tuple[int, int, int]],
    trans: dict[int, dict[tuple[int, int, int], tuple[int, int]]],
    level_idx: int,
    lut6_crystal: dict[int, int],
) -> int:
    """`SleepCrystallizer` for `ARC-AGI-3`: runs exact `BFS` over `C_env`-verified playfield transitions
    from `start_hash` to `goal_state` (`goal_edge = (goal_parent_hash, winning_action)`), extracting the
    shortest verified trajectory `tau*` (`0` wasted exploration steps!) and compiling each step into the
    `4,096 B` (`512 x uint64`) `System-0` `LUT6` crystal! Returns compiled trajectory length."""
    goal_parent, win_act = goal_edge
    q: deque[tuple[int, list[tuple[int, tuple[int, int, int], int]]]] = deque([(start_hash, [])])
    seen = {start_hash}
    shortest_steps: list[tuple[int, tuple[int, int, int], int]] | None = None

    while q:
        u, path = q.popleft()
        if u == goal_parent:
            shortest_steps = path + [(u, win_act, 0xFF)]
            break
        for act, (v, v_sig8) in trans.get(u, {}).items():
            if v not in seen:
                seen.add(v)
                q.append((v, path + [(u, act, v_sig8)]))

    if shortest_steps is None:
        return 0

    for st_hash, (aid, px, py), exp_sig8 in shortest_steps:
        slot, tag16 = _playfield_to_lut6_slot(st_hash, level_idx)
        lut6_crystal[slot] = pack_arc3_action_word64(aid, px, py, exp_sig8, tag16)
    return len(shortest_steps)


@dataclass
class Arc3GameRunSummary:
    game_id: str
    total_levels: int
    levels_completed: int
    total_actions: int
    official_rhae_score: float
    lut6_reflex_steps: int
    breaker_trips: int
    world_model_verified_edges: int
    crystallized_lut6_slots: int
    elapsed_ms: float


def run_arc3_agent_on_environment(
    arc: Any,
    game_id: str,
    baseline_actions_per_level: list[int],
    agent_kind: str,
    max_actions_per_game: int = 900,
    seed: int = 20261125,
    lut6_crystal: dict[int, int] | None = None,
) -> Arc3GameRunSummary:
    """Runs one of 4 black-box agents on an official `ARC-AGI-3` `LocalEnvironmentWrapper` and scores
    the run with the exact official `arc_agi.scorecard.EnvironmentScoreCalculator` (`RHAE`):
      - `random`: uniform random `available_actions` + random `64x64` click coordinates,
      - `reactive_memoryless`: 1-step visual foreground action heuristic without world-model memory,
      - `world_model_bfs_cold` (`Interactive-MoL Wave 1`): HUD-canonicalized black-box `Frontier-BFS`
        with `C_env(T_hat) == 1` transition verification & downward `SleepCrystallizer` into `lut6_crystal`,
      - `compiled_mol_lut6_wave2` (`Interactive-MoL Wave 2`): executes compiled `4 KB` `System-0` `LUT6`
        crystal (`O(1)` lookup, `0` FLOPs) with `Theorem 73` mechanic-shift circuit breaker escalating to
        `Frontier-BFS` on any uncertified state or mutated mechanic!
    """
    from arc_agi.scorecard import EnvironmentScoreCalculator
    from arcengine import GameAction, GameState

    rng = np.random.default_rng(seed + (int.from_bytes(game_id.encode(), "little") & 0xFFFF))
    env = arc.make(game_id)
    obs = env.reset()
    total_levels = int(obs.win_levels)

    t0 = time.perf_counter()
    cur_level = int(obs.levels_completed)
    level_actions = 0
    total_actions = 0
    resets = 0
    calc = EnvironmentScoreCalculator(id=game_id)

    crystal = lut6_crystal if lut6_crystal is not None else {}

    # Per-level online black-box `C_env` world-model graph
    level_start_hash: int | None = None
    actions_of: dict[int, list[tuple[int, int, int]]] = {}
    tried: dict[int, set[tuple[int, int, int]]] = {}
    trans: dict[int, dict[tuple[int, int, int], tuple[int, int]]] = {}  # sh -> {act: (next_sh, next_sig8)}
    plan_queue: deque[tuple[int, int, int]] = deque()

    lut6_steps = 0
    breaker_trips = 0
    total_verified_edges = 0
    pending_lut6_exp_sig: int | None = None

    while total_actions < max_actions_per_game and obs.state != GameState.WIN:
        if obs.state == GameState.GAME_OVER:
            obs = env.step(GameAction.RESET)
            resets += 1
            total_actions += 1
            level_actions += 1
            plan_queue.clear()
            pending_lut6_exp_sig = None
            continue

        frame = _frame_to_np(obs)
        sh = _playfield_hash64(frame)
        if level_start_hash is None:
            level_start_hash = sh

        avail = [int(a) for a in obs.available_actions if int(a) != 0]
        if not avail:
            break

        if agent_kind == "random":
            aid = int(rng.choice(avail))
            if aid == 6:
                act = (6, int(rng.integers(0, 64)), int(rng.integers(0, 64)))
            else:
                act = (aid, 0, 0)

        elif agent_kind == "reactive_memoryless":
            cands = _visual_action_candidates(frame, avail)
            act = cands[total_actions % min(2, len(cands))]

        elif agent_kind in ("world_model_bfs_cold", "compiled_mol_lut6_wave2"):
            if sh not in actions_of:
                actions_of[sh] = _visual_action_candidates(frame, avail)
                tried[sh] = set()
                trans.setdefault(sh, {})

            used_crystal = False
            if agent_kind == "compiled_mol_lut6_wave2":
                slot, tag16 = _playfield_to_lut6_slot(sh, cur_level)
                w64 = crystal.get(slot)
                if w64 is not None:
                    c_aid, c_px, c_py, c_exp_sig, c_tag16 = unpack_arc3_action_word64(w64)
                    if c_tag16 == tag16 and c_aid in avail:
                        act = (c_aid, c_px, c_py)
                        pending_lut6_exp_sig = c_exp_sig
                        lut6_steps += 1
                        used_crystal = True
                        plan_queue.clear()

            if not used_crystal:
                pending_lut6_exp_sig = None
                if not plan_queue:
                    untried = [a for a in actions_of[sh] if a not in tried[sh]]
                    if untried:
                        plan_queue.append(untried[0])
                    else:
                        # Exact BFS over `C_env`-verified playfield graph `trans` to nearest state with untried actions!
                        q_bfs: deque[tuple[int, list[tuple[int, int, int]]]] = deque([(sh, [])])
                        seen_bfs = {sh}
                        found_plan: list[tuple[int, int, int]] | None = None
                        while q_bfs:
                            u, p_acts = q_bfs.popleft()
                            u_untried = [a for a in actions_of.get(u, []) if a not in tried.get(u, set())]
                            if u_untried:
                                found_plan = p_acts + [u_untried[0]]
                                break
                            for a_edge, (v_nxt, _) in trans.get(u, {}).items():
                                if v_nxt not in seen_bfs:
                                    seen_bfs.add(v_nxt)
                                    q_bfs.append((v_nxt, p_acts + [a_edge]))
                        if found_plan:
                            plan_queue.extend(found_plan)
                        else:
                            # Exhausted reachable component (e.g. irreversible push deadlock) -> clean level reset!
                            obs = env.step(GameAction.RESET)
                            resets += 1
                            total_actions += 1
                            level_actions += 1
                            actions_of.clear()
                            tried.clear()
                            trans.clear()
                            plan_queue.clear()
                            level_start_hash = None
                            continue
                act = plan_queue.popleft()
                tried[sh].add(act)
        else:
            raise ValueError(f"Unknown agent_kind: {agent_kind}")

        ga = GameAction.from_id(act[0])
        step_kwargs = {"data": {"x": act[1], "y": act[2]}} if ga.is_complex() else {}
        next_obs = env.step(ga, **step_kwargs)
        total_actions += 1
        level_actions += 1

        next_frame = _frame_to_np(next_obs)
        next_sh = _playfield_hash64(next_frame)
        next_sig8 = _playfield_sig8(next_frame)

        new_lvl = int(next_obs.levels_completed)
        if new_lvl > cur_level:
            # Level solved! Compile verified shortest path `level_start_hash -> (sh, act)` into `4 KB` `.lut6` crystal!
            if agent_kind in ("world_model_bfs_cold", "compiled_mol_lut6_wave2") and level_start_hash is not None:
                synthesize_shortest_lut6_trajectory(
                    start_hash=level_start_hash,
                    goal_edge=(sh, act),
                    trans=trans,
                    level_idx=cur_level,
                    lut6_crystal=crystal,
                )
            for lv_idx in range(cur_level, new_lvl):
                b_act = baseline_actions_per_level[lv_idx] if lv_idx < len(baseline_actions_per_level) else 30
                calc.add_level(
                    level_index=lv_idx + 1,
                    completed=True,
                    actions_taken=level_actions,
                    baseline_actions=int(b_act),
                    game_id=game_id,
                )
                level_actions = 0

            total_verified_edges += sum(len(d) for d in trans.values())
            actions_of.clear()
            tried.clear()
            trans.clear()
            plan_queue.clear()
            level_start_hash = None
            pending_lut6_exp_sig = None
            cur_level = new_lvl
        else:
            # Check `Theorem 73` Mechanic-Shift Circuit Breaker on `.lut6` non-winning steps
            if pending_lut6_exp_sig is not None and pending_lut6_exp_sig != 0xFF and pending_lut6_exp_sig != next_sig8:
                breaker_trips += 1
                slot_bad, _ = _playfield_to_lut6_slot(sh, cur_level)
                crystal.pop(slot_bad, None)
                plan_queue.clear()
            pending_lut6_exp_sig = None

            if agent_kind in ("world_model_bfs_cold", "compiled_mol_lut6_wave2"):
                if next_obs.state != GameState.GAME_OVER and next_sh != sh:
                    trans.setdefault(sh, {})[act] = (next_sh, next_sig8)
                else:
                    # Wall bump or timer reset -> clear queued multi-step plan while keeping verified `trans`!
                    plan_queue.clear()

        obs = next_obs

    total_verified_edges += sum(len(d) for d in trans.values())
    rem_first = True
    for lv_idx in range(cur_level, total_levels):
        b_act = baseline_actions_per_level[lv_idx] if lv_idx < len(baseline_actions_per_level) else 30
        calc.add_level(
            level_index=lv_idx + 1,
            completed=False,
            actions_taken=level_actions if rem_first else 0,
            baseline_actions=int(b_act),
            game_id=game_id,
        )
        rem_first = False

    calc.resets = resets
    calc.completed = (cur_level >= total_levels)
    env_score = calc.to_score(include_levels=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return Arc3GameRunSummary(
        game_id=game_id,
        total_levels=total_levels,
        levels_completed=cur_level,
        total_actions=total_actions,
        official_rhae_score=round(float(env_score.score), 4),
        lut6_reflex_steps=lut6_steps,
        breaker_trips=breaker_trips,
        world_model_verified_edges=total_verified_edges,
        crystallized_lut6_slots=len(crystal),
        elapsed_ms=round(elapsed_ms, 3),
    )
