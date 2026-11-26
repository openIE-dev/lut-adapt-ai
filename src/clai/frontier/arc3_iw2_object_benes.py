"""Phase 38 (`Claim LA-49`, Shot #5): Relational Width-`2` `IW(2)` + `6`-Bit Beneš `LUT6`
Object-Pair Spatial Induction + Dynamic HUD Tick-Row Auto-Masking on Official `ARC-AGI-3`.

Strictly `100%` Black-Box (`0` `.py` game-source peeking, `0` raw `64x64` frames persisted).

Resolves the three exact algorithmic bottlenecks identified in Phase 37 `IW(1)`:
  1. **Dynamic HUD Tick-Row Auto-Masking (`62x64` + causal tick detector)**:
     Preserves interactive objects, palettes, and clickable tiles in `y = 58..61` (which
     hardcoded `frame[:58, :]` blinded on `click_only` and `keyboard_click` games) while
     automatically masking monotonic 1-2px move-counter/timer rows (`y = 62..63` plus any
     detected wall-bump tick row).
  2. **Modality-Stratified Action Expansion**:
     Prevents `28` `ACTION6(x, y)` click candidates at root/shallow states from starving
     `O(1)` keyboard trajectories (`ACTION1..5, ACTION7`) on mixed `keyboard_click` games
     (`cd82`, `sb26`, `cn04`, `ka59`, `sk48`, `m0r0`, `sc25`).
  3. **Lipovetzky & Geffner Relational Width-`2` `IW(2)` + `6`-Bit Beneš `LUT6` Predicates (`Theorem 83`)**:
     Tracks joint object-pair configurations `(obj_a, obj_b)` and `6`-bit Beneš `LUT6`
     relational spatial wires (`same_row`, `same_col`, `manhattan_le_2`, `color_eq`,
     `area_eq`, `switch_hist_parity`), unlocking post-switch / post-push corridor returns
     where width-`1` pixel novelty `IW(1) == 0`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import time
from typing import Any

import numpy as np

from clai.frontier.bonsai_stream_and_arc_agi_3 import (
    Arc3GameRunSummary,
    _frame_to_np,
    pack_arc3_action_word64,
    unpack_arc3_action_word64,
)

HUD_CANDIDATE_ROWS: tuple[int, ...] = (0, 1, 58, 59, 60, 61)
HUD_META_SLOTS: tuple[int, int] = (510, 511)


def seal_level_hud_rows(crystal: dict[int, int], level_idx: int, hud_rows: set[int]) -> None:
    """Seals the `6`-bit causal `hud_rows` mask (`+ bit 6 = valid`) for `level_idx` (`0..15`)
    into byte `level_idx % 8` of header slot `510` or `511` inside the `4,096 B` (`512 x u64`) crystal."""
    if not (0 <= level_idx < 16):
        return
    mask = 0x40  # bit 6 = sealed flag
    for bit_i, r in enumerate(HUD_CANDIDATE_ROWS):
        if r in hud_rows:
            mask |= 1 << bit_i
    slot = HUD_META_SLOTS[0] if level_idx >= 8 else HUD_META_SLOTS[1]
    shift = (level_idx % 8) * 8
    cur_w = int(crystal.get(slot, 0)) & ~(0xFF << shift)
    crystal[slot] = (cur_w | (mask << shift)) & 0xFFFFFFFFFFFFFFFF


def decode_level_hud_rows(crystal: dict[int, int], level_idx: int) -> tuple[set[int], bool]:
    """Returns `(hud_rows, is_sealed)` for `level_idx` from header slots `510..511`."""
    rows = {62, 63}
    if not (0 <= level_idx < 16):
        return rows, False
    slot = HUD_META_SLOTS[0] if level_idx >= 8 else HUD_META_SLOTS[1]
    w = crystal.get(slot)
    if w is None:
        return rows, False
    byte_v = (int(w) >> ((level_idx % 8) * 8)) & 0xFF
    if (byte_v & 0x40) == 0:
        return rows, False
    for bit_i, r in enumerate(HUD_CANDIDATE_ROWS):
        if (byte_v >> bit_i) & 1:
            rows.add(r)
    return rows, True


def _iw2_playfield_to_lut6_slot(state_hash: int, level_idx: int) -> tuple[int, int]:
    """Maps `(state_hash, level_idx)` into slots `0..509` of the `512 x uint64` (`4,096 B`)
    `System-0` `LUT6` page, reserving slots `510..511` for per-level `hud_rows` masks."""
    mixed = (int(state_hash) ^ ((int(level_idx) + 1) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    slot = int(mixed % 510)
    tag16 = int(((mixed >> 17) ^ (mixed >> 33)) & 0xFFFF)
    return slot, tag16


def _synthesize_iw2_shortest_lut6_trajectory(
    start_hash: int,
    goal_edge: tuple[int, tuple[int, int, int]],
    trans: dict[int, dict[tuple[int, int, int], tuple[int, int]]],
    level_idx: int,
    lut6_crystal: dict[int, int],
    hud_rows: set[int],
) -> int:
    """Compiles the exact `C_env`-verified shortest trajectory from `start_hash` to `goal_edge`
    into slots `0..509` and seals `level_idx`'s `hud_rows` mask in slots `510..511` (`<= 4,096 B`)."""
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

    seal_level_hud_rows(lut6_crystal, level_idx, hud_rows)
    for st_hash, (aid, px, py), exp_sig8 in shortest_steps:
        slot, tag16 = _iw2_playfield_to_lut6_slot(st_hash, level_idx)
        lut6_crystal[slot] = pack_arc3_action_word64(aid, px, py, exp_sig8, tag16)
    return len(shortest_steps)


@dataclass(frozen=True)
class SalientObject:
    color: int
    area: int
    cx: int
    cy: int
    bbox: tuple[int, int, int, int]  # (rmin, rmax, cmin, cmax)


def detect_hud_mask_rows(
    prev_frame: np.ndarray | None,
    curr_frame: np.ndarray,
    act: tuple[int, int, int],
    known_hud_rows: set[int],
) -> bool:
    """Black-box causal HUD-timer row detector (`0` source peeking):
    Always masks bottom move-counter rows `{62, 63}`; if an action leaves the `56x64` interior
    playfield `frame[2:58, :]` completely unchanged while mutating `1..64` pixels in an unclicked
    edge row `r in {0, 1, 58, 59, 60, 61}`, `r` is causally certified as a move-counter/timer row!
    Returns `True` iff `known_hud_rows` grew."""
    before = len(known_hud_rows)
    if prev_frame is not None and np.array_equal(prev_frame[2:58, :], curr_frame[2:58, :]):
        for r in HUD_CANDIDATE_ROWS:
            if act[0] == 6 and abs(act[2] - r) <= 1:
                continue
            diff_cnt = int(np.count_nonzero(prev_frame[r, :] != curr_frame[r, :]))
            if 1 <= diff_cnt <= 64:
                known_hud_rows.add(r)
    return len(known_hud_rows) > before


def canonical_playfield(frame: np.ndarray, hud_rows: set[int]) -> np.ndarray:
    """Preserves interactive `y = 58..61` palettes/buttons (`vc33`, `m0r0`, `lp85`) while zeroing
    level-start detected horizontal HUD timer bars (`{62, 63}` + flat status bars) so `level_start_hash`
    never drifts mid-level (`100%` shortest-path `LUT6` crystallization!)."""
    pf = frame[:62, :].copy()
    for r in hud_rows:
        if 0 <= r < 62:
            pf[r, :] = 0
    return np.ascontiguousarray(pf)


def playfield_hash64(frame: np.ndarray, hud_rows: set[int]) -> int:
    pf = canonical_playfield(frame, hud_rows)
    return int.from_bytes(hashlib.blake2s(pf.tobytes(), digest_size=8).digest(), "little")


def playfield_sig8(frame: np.ndarray, hud_rows: set[int]) -> int:
    pf = canonical_playfield(frame, hud_rows)
    return int.from_bytes(hashlib.blake2s(pf.tobytes(), digest_size=1).digest(), "little")


def local_patch_sig32(frame: np.ndarray, x: int, y: int) -> int:
    r0, r1 = max(0, y - 1), min(62, y + 2)
    c0, c1 = max(0, x - 1), min(64, x + 2)
    sub = np.ascontiguousarray(frame[r0:r1, c0:c1])
    return int.from_bytes(hashlib.blake2s(sub.tobytes(), digest_size=4).digest(), "little")


def extract_salient_objects(frame: np.ndarray, hud_rows: set[int]) -> tuple[int, list[SalientObject]]:
    """Extracts 4-connected non-background objects across `y = 0..61` (excluding `hud_rows`),
    sorted from rarest/smallest interactive sprites to larger boards."""
    pf = canonical_playfield(frame, hud_rows)
    vals, cnts = np.unique(pf, return_counts=True)
    bg_color = int(vals[int(np.argmax(cnts))]) if vals.size else 0
    color_freq = {int(v): int(cnt) for v, cnt in zip(vals.tolist(), cnts.tolist())}

    h_rows, w_cols = pf.shape
    vis = np.zeros((h_rows, w_cols), dtype=bool)
    for r in hud_rows:
        if 0 <= r < h_rows:
            vis[r, :] = True

    objs: list[tuple[int, int, SalientObject]] = []
    for r in range(h_rows):
        for c in range(w_cols):
            col = int(pf[r, c])
            if not vis[r, c] and col != bg_color:
                st = [(r, c)]
                vis[r, c] = True
                xs, ys = [], []
                while st:
                    cr, cc = st.pop()
                    ys.append(cr)
                    xs.append(cc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h_rows and 0 <= nc < w_cols and not vis[nr, nc] and int(pf[nr, nc]) == col:
                            vis[nr, nc] = True
                            st.append((nr, nc))
                area = len(xs)
                if 1 <= area <= 1200:
                    mx, my = float(np.mean(xs)), float(np.mean(ys))
                    bi = int(np.argmin((np.array(xs) - mx) ** 2 + (np.array(ys) - my) ** 2))
                    obj = SalientObject(
                        color=col,
                        area=area,
                        cx=int(xs[bi]),
                        cy=int(ys[bi]),
                        bbox=(min(ys), max(ys), min(xs), max(xs)),
                    )
                    objs.append((color_freq.get(col, 9999), area, obj))

    objs.sort(key=lambda t: (t[0], t[1], t[2].cy, t[2].cx))
    return bg_color, [t[2] for t in objs]


def encode_benes_lut6_rel6(a: SalientObject, b: SalientObject, hist_parity: int) -> int:
    """Computes the `6`-wire Beneš `LUT6` relational feature code (`0..63`) for object pair `(a, b)`:
      - `bit 0`: `same_row` (`|cy_a - cy_b| <= 1`)
      - `bit 1`: `same_col` (`|cx_a - cx_b| <= 1`)
      - `bit 2`: `manhattan_contact` (`|cx_a - cx_b| + |cy_a - cy_b| <= 3`)
      - `bit 3`: `color_match` (`a.color == b.color`)
      - `bit 4`: `congruent_area` (`a.area == b.area`)
      - `bit 5`: `board_switch_parity` (`hist_parity & 1`)
    Matches `lut6_core::waios_mosaic::eval_iw2_relational_lut6` wire convention."""
    dx = abs(a.cx - b.cx)
    dy = abs(a.cy - b.cy)
    b0 = 1 if dy <= 1 else 0
    b1 = 1 if dx <= 1 else 0
    b2 = 1 if (dx + dy) <= 3 else 0
    b3 = 1 if a.color == b.color else 0
    b4 = 1 if a.area == b.area else 0
    b5 = hist_parity & 1
    return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4) | (b5 << 5)


def compute_iw2_relational_novelty(
    frame: np.ndarray,
    hud_rows: set[int],
    seen_iw1_atoms: set[int],
    seen_iw2_pairs: set[int],
    seen_benes_rel: set[int],
) -> tuple[int, int, int, list[SalientObject]]:
    """Computes Lipovetzky & Geffner width-`1` (`IW(1)`) AND relational width-`2` (`IW(2)`) +
    `6`-bit Beneš `LUT6` spatial-predicate novelty (`Theorem 83`):
      - Returns `(iw1_gain, iw2_pair_gain, benes_rel_gain, salient_objects)`.
      - Even when returning through an already-visited corridor (`iw1_gain == 0`), a toggled
        remote switch, pushed block, or newly aligned object pair yields `iw2_pair_gain > 0`!"""
    pf = canonical_playfield(frame, hud_rows)
    rs, cs = np.indices(pf.shape, dtype=np.int32)
    packed1 = ((rs << 10) | (cs << 4) | (pf.astype(np.int32) & 0xF)).ravel().tolist()
    iw1_gain = 0
    for a1 in packed1:
        if a1 not in seen_iw1_atoms:
            seen_iw1_atoms.add(a1)
            iw1_gain += 1

    _, objs = extract_salient_objects(frame, hud_rows)
    # Global foreground color-count parity signature captures switch/door state changes
    vals, cnts = np.unique(pf, return_counts=True)
    hist_sig8 = 0
    for v, cnt in zip(vals.tolist(), cnts.tolist()):
        hist_sig8 ^= ((int(v) * 17 + int(cnt)) & 0xFF)

    # Top `16` interactive objects (`area <= 160`) -> at most `16 * 15 / 2 = 120` relational pairs!
    salient = [o for o in objs if o.area <= 160][:16]
    if len(salient) < 4:
        salient = objs[:12]

    iw2_gain = 0
    benes_gain = 0
    n_s = len(salient)
    # Also include `(obj_i, global_hist_sig8)` pairs so single-avatar + global switch state is width-2!
    for i in range(n_s):
        oi = salient[i]
        atom_i = ((oi.color & 0xF) << 12) | ((oi.cx & 0x3F) << 6) | (oi.cy & 0x3F)
        global_pair = (1 << 30) | (atom_i << 8) | hist_sig8
        if global_pair not in seen_iw2_pairs:
            seen_iw2_pairs.add(global_pair)
            iw2_gain += 1
        for j in range(i + 1, n_s):
            oj = salient[j]
            atom_j = ((oj.color & 0xF) << 12) | ((oj.cx & 0x3F) << 6) | (oj.cy & 0x3F)
            lo_a, hi_a = (atom_i, atom_j) if atom_i <= atom_j else (atom_j, atom_i)
            pair_key = (lo_a << 16) | hi_a
            if pair_key not in seen_iw2_pairs:
                seen_iw2_pairs.add(pair_key)
                iw2_gain += 1
            rel6 = encode_benes_lut6_rel6(oi, oj, hist_sig8 & 1)
            rel_key = ((oi.color & 0xF) << 10) | ((oj.color & 0xF) << 6) | rel6
            if rel_key not in seen_benes_rel:
                seen_benes_rel.add(rel_key)
                benes_gain += 1

    return iw1_gain, iw2_gain, benes_gain, objs


def extract_stratified_actions(
    frame: np.ndarray,
    available_actions: list[int],
    hud_rows: set[int],
    objs: list[SalientObject] | None = None,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Returns `(primary_actions, secondary_click_actions)`:
      - On mixed `keyboard_click` games, `primary_actions` holds all `O(1)` keyboard/simple
        actions (`ACTION1..5, ACTION7`) PLUS only the top `4` most salient interactive object clicks,
        while `secondary_click_actions` holds remaining object/grid clicks (expanded ONLY after
        primary branches!).
      - On `click_only` games (`6` is the sole action), `primary_actions` holds up to `28`
        deduplicated object centroids + subdivided board cells across `y = 0..61`!"""
    kb_cands: list[tuple[int, int, int]] = [
        (int(aid), 0, 0) for aid in available_actions if int(aid) not in (0, 6)
    ]
    if 6 not in available_actions:
        return kb_cands, []

    if objs is None:
        bg_color, objs = extract_salient_objects(frame, hud_rows)
    else:
        pf = canonical_playfield(frame, hud_rows)
        vals, cnts = np.unique(pf, return_counts=True)
        bg_color = int(vals[int(np.argmax(cnts))]) if vals.size else 0

    seen_cells: set[tuple[int, int]] = set()
    click_xy: list[tuple[int, int]] = []

    def _add_pt(px: int, py: int, dedup_shift: int = 1) -> None:
        if py in hud_rows or not (0 <= px < 64 and 0 <= py < 62):
            return
        key = (px >> dedup_shift, py >> dedup_shift)
        if key not in seen_cells:
            seen_cells.add(key)
            click_xy.append((px, py))

    # 1. Exact centroids of all salient connected components (`y = 0..61`, including `58..61` palettes!)
    for o in objs:
        # Use exact pixel precision (`dedup_shift = 0`) for tiny buttons/cells (`area <= 9`), 2px otherwise
        _add_pt(o.cx, o.cy, dedup_shift=0 if o.area <= 9 else 1)
        if len(click_xy) >= 32:
            break

    # 2. Subdivide larger uniform board/grid components (`area >= 36`) at 4px stride
    for o in objs:
        if o.area >= 36:
            rmin, rmax, cmin, cmax = o.bbox
            for qr in range(rmin + 2, min(rmax + 1, 62), 5):
                for qc in range(cmin + 2, min(cmax + 1, 64), 5):
                    if int(frame[qr, qc]) == o.color:
                        _add_pt(qc, qr, dedup_shift=1)

    # 3. Coarse foreground probe across `y = 2..61`
    for r in range(3, 61, 6):
        if r in hud_rows:
            continue
        for c in range(3, 62, 6):
            if int(frame[r, c]) != bg_color:
                _add_pt(c, r, dedup_shift=1)

    if not click_xy:
        click_xy.append((32, 30))

    has_4way_kb = any(int(aid) in (1, 2, 3, 4) for aid in available_actions)
    if has_4way_kb:
        # Mixed `keyboard_click` (`cd82`, `m0r0`, `sb26`, etc.): stage 0 = pure O(1) keyboard moves
        # plus the top 2 rarest salient object centroids (pruned immediately on inert click!),
        # stage 1 = remaining object/grid clicks!
        return kb_cands + [(6, x, y) for x, y in click_xy[:2]], [(6, x, y) for x, y in click_xy[2:18]]
    else:
        # Pure `click_only` or `click_action7` (`lp85`, `vc33`, `su15`, `r11l`): `ACTION7` first
        # (when present on `su15`) + top 24 object/palette centroids (`y = 0..61`), stage 1 = secondary!
        prim_clicks = kb_cands + [(6, x, y) for x, y in click_xy[:24]]
        return prim_clicks, [(6, x, y) for x, y in click_xy[24:38]]


def run_apex_iw2_arc3_agent(
    arc: Any,
    game_id: str,
    baseline_actions_per_level: list[int],
    agent_kind: str,
    max_actions_per_game: int = 1200,
    lut6_crystal: dict[int, int] | None = None,
) -> Arc3GameRunSummary:
    """Executes `OpenIE-Apex IW(2) + Beneš-LUT6 Object-Relational` agent (`apex_iw2_cold` = Wave 1
    discovery + crystallization; `apex_iw2_lut6_wave2` = Wave 2 `4 KB` `System-0` `LUT6` execution
    with `Theorem 73` mechanic-shift circuit breaker)."""
    from arc_agi.scorecard import EnvironmentScoreCalculator
    from arcengine import GameAction, GameState

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

    # Per-level causal `hud_rows`: restored from header slots `510..511` when `cur_level` was sealed in Wave 1!
    hud_rows, hud_locked = decode_level_hud_rows(crystal, cur_level)
    level_start_hash: int | None = None
    level_start_frame: np.ndarray | None = None
    level_start_avail: list[int] = []
    # In-RAM per-level transition trace (`0` disk writes!) for exact re-keying when `hud_rows` grows
    raw_edges: list[tuple[np.ndarray, tuple[int, int, int], np.ndarray, list[int]]] = []
    seen_iw1: set[int] = set()
    seen_iw2: set[int] = set()
    seen_benes: set[int] = set()
    state_frames: dict[int, np.ndarray] = {}
    prim_actions: dict[int, list[tuple[int, int, int]]] = {}
    sec_actions: dict[int, list[tuple[int, int, int]]] = {}
    tried: dict[int, set[tuple[int, int, int]]] = {}
    trans: dict[int, dict[tuple[int, int, int], tuple[int, int]]] = {}
    inert_patch_clicks: set[tuple[int, int, int]] = set()
    crystal_visited_sh: set[int] = set()
    # Priority frontier: `(pass_stage, nov_tier, depth, -iw2_score, tie_id, state_hash)`
    iw2_pq: list[tuple[int, int, int, int, int, int]] = []
    state_depth: dict[int, int] = {}
    tie_counter = 0
    plan_queue: deque[tuple[int, int, int]] = deque()

    lut6_steps = 0
    breaker_trips = 0
    total_verified_edges = 0
    pending_lut6_exp_sig: int | None = None

    def _reset_level_memory() -> None:
        nonlocal level_start_hash, level_start_frame
        level_start_hash = None
        level_start_frame = None
        level_start_avail.clear()
        raw_edges.clear()
        seen_iw1.clear()
        seen_iw2.clear()
        seen_benes.clear()
        state_frames.clear()
        prim_actions.clear()
        sec_actions.clear()
        tried.clear()
        trans.clear()
        inert_patch_clicks.clear()
        crystal_visited_sh.clear()
        iw2_pq.clear()
        state_depth.clear()
        plan_queue.clear()

    def _register_state(fr: np.ndarray, av: list[int], depth_hint: int, push_pq: bool) -> int:
        nonlocal tie_counter
        h = playfield_hash64(fr, hud_rows)
        is_new = h not in prim_actions
        if h not in state_depth or depth_hint < state_depth[h]:
            state_depth[h] = depth_hint
        if is_new:
            g1, g2, gb, objs_h = compute_iw2_relational_novelty(
                fr, hud_rows, seen_iw1, seen_iw2, seen_benes
            )
            p_a, s_a = extract_stratified_actions(fr, av, hud_rows, objs_h)
            state_frames[h] = fr.copy()
            prim_actions[h] = p_a
            sec_actions[h] = s_a
            tried.setdefault(h, set())
            trans.setdefault(h, {})
            if push_pq:
                has_nov = (g2 > 0) or (gb > 0) or (g1 > 0)
                nt = 0 if has_nov else 1
                sc = 4 * g2 + 8 * gb + g1
                tie_counter += 1
                heapq.heappush(iw2_pq, (0, nt, state_depth[h], -sc, tie_counter, h))
                if s_a:
                    tie_counter += 1
                    heapq.heappush(iw2_pq, (1, nt + 1, state_depth[h], -sc, tie_counter, h))
        return h

    def _rekey_level_graph_for_new_hud() -> None:
        nonlocal level_start_hash
        if level_start_frame is None:
            return
        seen_iw1.clear()
        seen_iw2.clear()
        seen_benes.clear()
        state_frames.clear()
        prim_actions.clear()
        sec_actions.clear()
        tried.clear()
        trans.clear()
        iw2_pq.clear()
        state_depth.clear()
        plan_queue.clear()

        level_start_hash = _register_state(level_start_frame, list(level_start_avail), 0, True)
        for u_fr, e_act, v_fr, v_av in raw_edges:
            uh = _register_state(u_fr, list(level_start_avail), state_depth.get(playfield_hash64(u_fr, hud_rows), 0), True)
            tried.setdefault(uh, set()).add(e_act)
            vh = playfield_hash64(v_fr, hud_rows)
            if vh != uh:
                d_v = state_depth.get(uh, 0) + 1
                _register_state(v_fr, v_av, d_v, True)
                trans.setdefault(uh, {})[e_act] = (vh, playfield_sig8(v_fr, hud_rows))

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
        avail = [int(a) for a in obs.available_actions if int(a) != 0]
        if not avail:
            break

        if level_start_hash is None:
            hud_rows, hud_locked = decode_level_hud_rows(crystal, cur_level)
            level_start_frame = frame.copy()
            level_start_avail.clear()
            level_start_avail.extend(avail)
            sh = _register_state(frame, avail, 0, True)
            level_start_hash = sh
        else:
            sh = playfield_hash64(frame, hud_rows)
            if sh not in prim_actions:
                _register_state(frame, avail, state_depth.get(sh, 0), False)

        used_crystal = False
        if agent_kind == "apex_iw2_lut6_wave2" and sh not in crystal_visited_sh:
            slot, tag16 = _iw2_playfield_to_lut6_slot(sh, cur_level)
            w64 = crystal.get(slot)
            if w64 is not None:
                c_aid, c_px, c_py, c_exp_sig, c_tag16 = unpack_arc3_action_word64(w64)
                if c_tag16 == tag16 and c_aid in avail:
                    crystal_visited_sh.add(sh)
                    act = (c_aid, c_px, c_py)
                    pending_lut6_exp_sig = c_exp_sig
                    lut6_steps += 1
                    used_crystal = True
                    plan_queue.clear()

        if not used_crystal:
            pending_lut6_exp_sig = None
            if not plan_queue:
                def _usable_untried(u_hash: int, stage: int) -> list[tuple[int, int, int]]:
                    u_fr = state_frames.get(u_hash)
                    pool = prim_actions.get(u_hash, []) if stage == 0 else sec_actions.get(u_hash, [])
                    out_a = []
                    for a in pool:
                        if a in tried.get(u_hash, set()):
                            continue
                        if a[0] == 6 and u_fr is not None:
                            psig = local_patch_sig32(u_fr, a[1], a[2])
                            if (a[1], a[2], psig) in inert_patch_clicks:
                                tried[u_hash].add(a)
                                continue
                        out_a.append(a)
                    return out_a

                cur_untried = _usable_untried(sh, 0) or _usable_untried(sh, 1)
                target_u: int | None = None
                target_act: tuple[int, int, int] | None = None
                while iw2_pq:
                    st_pass, _, _, _, _, cand_u = iw2_pq[0]
                    cand_untried = _usable_untried(cand_u, st_pass)
                    if not cand_untried:
                        heapq.heappop(iw2_pq)
                        continue
                    target_u = cand_u
                    target_act = cand_untried[0]
                    break

                if target_u == sh and target_act is not None:
                    plan_queue.append(target_act)
                elif target_u is not None and target_act is not None:
                    q_bfs: deque[tuple[int, list[tuple[int, int, int]]]] = deque([(sh, [])])
                    seen_bfs = {sh}
                    path_to_target: list[tuple[int, int, int]] | None = None
                    any_fallback: list[tuple[int, int, int]] | None = [cur_untried[0]] if cur_untried else None
                    while q_bfs:
                        u, p_acts = q_bfs.popleft()
                        if u == target_u:
                            path_to_target = p_acts + [target_act]
                            break
                        if any_fallback is None:
                            uu = _usable_untried(u, 0) or _usable_untried(u, 1)
                            if uu:
                                any_fallback = p_acts + [uu[0]]
                        for a_edge, (v_nxt, _) in trans.get(u, {}).items():
                            if v_nxt not in seen_bfs:
                                seen_bfs.add(v_nxt)
                                q_bfs.append((v_nxt, p_acts + [a_edge]))
                    if path_to_target is not None:
                        plan_queue.extend(path_to_target)
                    elif any_fallback is not None:
                        plan_queue.extend(any_fallback)
                    else:
                        heapq.heappop(iw2_pq)
                        continue
                elif cur_untried:
                    plan_queue.append(cur_untried[0])
                else:
                    obs = env.step(GameAction.RESET)
                    resets += 1
                    total_actions += 1
                    level_actions += 1
                    _reset_level_memory()
                    continue

            act = plan_queue.popleft()
            tried.setdefault(sh, set()).add(act)

        ga = GameAction.from_id(act[0])
        step_kwargs = {"data": {"x": act[1], "y": act[2]}} if ga.is_complex() else {}
        next_obs = env.step(ga, **step_kwargs)
        total_actions += 1
        level_actions += 1

        next_frame = _frame_to_np(next_obs)
        new_lvl = int(next_obs.levels_completed)
        if new_lvl > cur_level:
            if level_start_hash is not None:
                _synthesize_iw2_shortest_lut6_trajectory(
                    start_hash=level_start_hash,
                    goal_edge=(sh, act),
                    trans=trans,
                    level_idx=cur_level,
                    lut6_crystal=crystal,
                    hud_rows=hud_rows,
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
            _reset_level_memory()
            pending_lut6_exp_sig = None
            cur_level = new_lvl
        else:
            nxt_avail = [int(a) for a in next_obs.available_actions if int(a) != 0]
            if next_obs.state != GameState.GAME_OVER:
                raw_edges.append((frame.copy(), act, next_frame.copy(), nxt_avail))
                if not hud_locked and detect_hud_mask_rows(frame, next_frame, act, hud_rows):
                    _rekey_level_graph_for_new_hud()
                    sh = playfield_hash64(frame, hud_rows)

            next_sh = playfield_hash64(next_frame, hud_rows)
            next_sig8 = playfield_sig8(next_frame, hud_rows)

            if pending_lut6_exp_sig is not None and pending_lut6_exp_sig != 0xFF and pending_lut6_exp_sig != next_sig8:
                breaker_trips += 1
                slot_bad, _ = _iw2_playfield_to_lut6_slot(sh, cur_level)
                crystal.pop(slot_bad, None)
                plan_queue.clear()
            pending_lut6_exp_sig = None

            if next_obs.state != GameState.GAME_OVER and next_sh != sh:
                d_nxt = state_depth.get(sh, 0) + 1
                _register_state(next_frame, nxt_avail, d_nxt, True)
                trans.setdefault(sh, {})[act] = (next_sh, next_sig8)
            else:
                if act[0] == 6 and next_obs.state != GameState.GAME_OVER:
                    inert_patch_clicks.add((act[1], act[2], local_patch_sig32(frame, act[1], act[2])))
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
    traj_slots = sum(1 for k in crystal if k not in HUD_META_SLOTS)

    return Arc3GameRunSummary(
        game_id=game_id,
        total_levels=total_levels,
        levels_completed=cur_level,
        total_actions=total_actions,
        official_rhae_score=round(float(env_score.score), 4),
        lut6_reflex_steps=lut6_steps,
        breaker_trips=breaker_trips,
        world_model_verified_edges=total_verified_edges,
        crystallized_lut6_slots=traj_slots,
        elapsed_ms=round(elapsed_ms, 3),
    )
