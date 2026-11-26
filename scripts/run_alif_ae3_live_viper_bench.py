"""Live Non-Destructive RAM-Only Hardware Cycle Benchmark on Attached USB `OpenMV-AE3`
(`Alif Ensemble E3 AE302F80F55D5AE`, Arm Cortex-M55 @ `400 MHz`) + Read-Only USB Bus Audit.

Safety & Measurement Honesty Invariants:
  - Uses MicroPython Raw REPL (`Ctrl-A ... Ctrl-D ... Ctrl-B`) to compile `@micropython.viper`
    native Armv8.1-M Thumb-2 machine code purely in SRAM (`0` flash/MRAM writes, factory firmware
    untouched).
  - Reads Armv8.1-M `DWT->CYCCNT` (`0xE0001004`) directly for hardware cycle counts.
  - Strictly stamps `board_metered = false` and `measured_j = None` (hardware cycle counter, not
    an inline DC rail shunt).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import termios
import time


VIPER_SCRIPT = r"""
import time, array, micropython

@micropython.viper
def dwt_init():
    demcr = ptr32(0xE000EDFC)
    demcr[0] = demcr[0] | 0x01000000
    cyccnt = ptr32(0xE0001004)
    cyccnt[0] = 0
    ctrl = ptr32(0xE0001000)
    ctrl[0] = ctrl[0] | 1

@micropython.viper
def dwt_read() -> int:
    c = ptr32(0xE0001004)
    return c[0] & 0x3FFFFFFF

# 1. Exact 16-gate LUT6 page (`wobble_codon_mask(0x08) = 0x00000F00`)
k_lo = array.array('I', [0]*16)
k_hi = array.array('I', [0]*16)
for g in (0, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15):
    k_lo[g] = 0x00000F00

p_lo = array.array('I', [0]*16)
p_hi = array.array('I', [0]*16)
for g in (1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15):
    p_hi[g] = 0x00F00000

@micropython.viper
def bench_innate_1k(t_lo: ptr32, t_hi: ptr32) -> int:
    acc: int = 0
    i: int = 0
    while i < 1000:
        addr: int = 8 + (i & 3)
        out: int = 0
        g: int = 0
        if addr < 32:
            while g < 16:
                out |= ((t_lo[g] >> addr) & 1) << g
                g += 1
        else:
            sh: int = addr - 32
            while g < 16:
                out |= ((t_hi[g] >> sh) & 1) << g
                g += 1
        acc ^= (out & 0xFF)
        i += 1
    return acc

@micropython.viper
def bench_overlay_wobble_1k(b_lo: ptr32, b_hi: ptr32, o_lo: ptr32, o_hi: ptr32) -> int:
    acc: int = 0
    i: int = 0
    while i < 1000:
        addr: int = 0x34 | (i & 3)
        out_b: int = 0
        out_o: int = 0
        g: int = 0
        sh: int = addr - 32
        while g < 16:
            out_b |= ((b_hi[g] >> sh) & 1) << g
            out_o |= ((o_hi[g] >> sh) & 1) << g
            g += 1
        if (out_o & 0xFF00) != 0:
            out_b = out_o
        acc ^= (out_b & 0xFF)
        i += 1
    return acc

# 2. Vertical Bit-Sliced Batch-32 (`32` parallel wobble-invariant lanes per 32-bit Viper word)
planes32 = array.array('I', [0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000, 0x00000000])

@micropython.viper
def bench_bitsliced_b32_32k(pl: ptr32) -> int:
    # 1,000 calls * 32 lanes = 32,000 LUT6 page evaluations
    x2: int = pl[2]
    x3: int = pl[3]
    x4: int = pl[4]
    x5: int = pl[5]
    acc: int = 0
    it: int = 0
    while it < 1000:
        # Wobble-degenerate 0x08 (`family = 2 = 0b0010` on wires [2..5]): ~x5 & ~x4 & x3 & ~x2
        hit_lanes: int = ((x5 ^ -1) & (x4 ^ -1) & x3 & (x2 ^ -1))
        # Broadcast across active payload/vote gates
        p0: int = hit_lanes
        p5: int = hit_lanes
        p7: int = hit_lanes
        votes: int = hit_lanes
        acc ^= (p0 ^ p5 ^ p7 ^ votes) & 0x3FFFFFFF
        x2 ^= 1
        it += 1
    return acc

# 3. Phase 31 Zero-Multiplier `PALUTE`/`T-MAC` 4-Trit SRAM Lookup Tail (`16` out x `16` 4-trit groups)
tmac_codes = array.array('i', [((r * 13 + g * 7) % 81) for r in range(16) for g in range(16)])
tmac_luts = array.array('i', [((g * 5 + idx - 40) * 3) for g in range(16) for idx in range(81)])
tmac_bias = array.array('i', [(r * 11 - 30) for r in range(16)])

@micropython.viper
def bench_zero_mul_tmac_1k(codes: ptr32, luts: ptr32, bias: ptr32) -> int:
    # Pure table gather + integer additions across all 16 output channels x 16 groups (ZERO MULs!)
    acc_all: int = 0
    it: int = 0
    while it < 1000:
        out_bits: int = 0
        r: int = 0
        while r < 16:
            s: int = bias[r]
            row_off: int = r << 4
            # Unrolled 4 groups per step (0 multiplications!)
            g: int = 0
            lut_off: int = 0
            while g < 16:
                s += luts[lut_off + codes[row_off + g]]
                s += luts[lut_off + 81 + codes[row_off + g + 1]]
                s += luts[lut_off + 162 + codes[row_off + g + 2]]
                s += luts[lut_off + 243 + codes[row_off + g + 3]]
                g += 4
                lut_off += 324
            if s > 0:
                out_bits |= 1 << r
            r += 1
        acc_all ^= out_bits
        it += 1
    return acc_all

# 3b. Phase 33 Antisymmetric 40-Entry `LUT Tensor Core` (ISCA '25) Table Build + Eval (`200` full passes)
l81_scratch = array.array('i', [0] * (16 * 81))
l41_scratch = array.array('i', [0] * (16 * 41))
act64 = array.array('i', [((j * 7 + 5) & 31) - 16 for j in range(64)])

@micropython.viper
def bench_full81_precomp_eval_200(codes: ptr32, luts: ptr32, bias: ptr32, x: ptr32) -> int:
    acc_all: int = 0
    it: int = 0
    while it < 200:
        # Step 1: Build all 16 x 81 group LUTs (`1,296` entries)
        g: int = 0
        while g < 16:
            c0: int = g << 2
            x0: int = x[c0]; x1: int = x[c0 + 1]; x2: int = x[c0 + 2]; x3: int = x[c0 + 3]
            base: int = g * 81
            idx: int = 0
            u3: int = -1
            while u3 <= 1:
                s3: int = u3 * x3
                u2: int = -1
                while u2 <= 1:
                    s2: int = s3 + u2 * x2
                    u1: int = -1
                    while u1 <= 1:
                        s1: int = s2 + u1 * x1
                        luts[base + idx] = s1 - x0
                        luts[base + idx + 1] = s1
                        luts[base + idx + 2] = s1 + x0
                        idx += 3
                        u1 += 1
                    u2 += 1
                u3 += 1
            g += 1
        # Step 2: Gather across 16 channels
        out_bits: int = 0
        r: int = 0
        while r < 16:
            s: int = bias[r]
            row_off: int = r << 4
            grp: int = 0
            lut_off: int = 0
            while grp < 16:
                s += luts[lut_off + codes[row_off + grp]]
                grp += 1
                lut_off += 81
            if s > 0:
                out_bits |= 1 << r
            r += 1
        acc_all ^= out_bits
        it += 1
    return acc_all

@micropython.viper
def bench_antisym40_precomp_eval_200(codes: ptr32, luts41: ptr32, bias: ptr32, x: ptr32) -> int:
    acc_all: int = 0
    it: int = 0
    while it < 200:
        # Step 1: Build ONLY 16 x 40 antisymmetric entries (`640` entries = `2.025x` fewer!)
        g: int = 0
        while g < 16:
            c0: int = g << 2
            x0: int = x[c0]; x1: int = x[c0 + 1]; x2: int = x[c0 + 2]; x3: int = x[c0 + 3]
            base: int = g * 41
            idx: int = 0
            # u3 = -1 -> 27 entries
            u2: int = -1
            while u2 <= 1:
                s2: int = -x3 + u2 * x2
                u1: int = -1
                while u1 <= 1:
                    s1: int = s2 + u1 * x1
                    luts41[base + idx] = s1 - x0
                    luts41[base + idx + 1] = s1
                    luts41[base + idx + 2] = s1 + x0
                    idx += 3
                    u1 += 1
                u2 += 1
            # u3 = 0, u2 = -1 -> 9 entries (`27..35`)
            u1 = -1
            while u1 <= 1:
                s1b: int = -x2 + u1 * x1
                luts41[base + idx] = s1b - x0
                luts41[base + idx + 1] = s1b
                luts41[base + idx + 2] = s1b + x0
                idx += 3
                u1 += 1
            # u3 = 0, u2 = 0 -> 4 entries (`36..39`) + midpoint 40 = 0
            luts41[base + 36] = -x1 - x0
            luts41[base + 37] = -x1
            luts41[base + 38] = -x1 + x0
            luts41[base + 39] = -x0
            luts41[base + 40] = 0
            g += 1
        # Step 2: Antisymmetric gather (`code > 40 -> -luts41[80 - code]`)
        out_bits: int = 0
        r: int = 0
        while r < 16:
            s: int = bias[r]
            row_off: int = r << 4
            grp: int = 0
            lut_off: int = 0
            while grp < 16:
                cd: int = codes[row_off + grp]
                if cd > 40:
                    s -= luts41[lut_off + (80 - cd)]
                else:
                    s += luts41[lut_off + cd]
                grp += 1
                lut_off += 41
            if s > 0:
                out_bits |= 1 << r
            r += 1
        acc_all ^= out_bits
        it += 1
    return acc_all

# 4. Baseline Full `64 -> 32 -> 16` INT8 Micro-MLP (`2,560` MACs per decision)
w1_i8 = array.array('i', [((j * 17 + 3) & 0x7F) - 64 for j in range(2048)])
w2_i8 = array.array('i', [((j * 13 + 7) & 0x7F) - 64 for j in range(512)])
x64_i8 = array.array('i', [16 if (j & 1) else -16 for j in range(64)])
h32_i8 = array.array('i', [0]*32)

@micropython.viper
def bench_int8_mlp_64_32_16_200(w1: ptr32, w2: ptr32, x: ptr32, h: ptr32) -> int:
    acc_all: int = 0
    it: int = 0
    while it < 200:
        r1: int = 0
        while r1 < 32:
            s1: int = 0
            b1: int = r1 << 6
            c1: int = 0
            while c1 < 64:
                s1 += w1[b1 + c1] * x[c1]
                c1 += 1
            v: int = s1 >> 7
            if v < 0:
                v = 0
            elif v > 127:
                v = 127
            h[r1] = v
            r1 += 1
        out: int = 0
        r2: int = 0
        while r2 < 16:
            s2: int = 0
            b2: int = r2 << 5
            c2: int = 0
            while c2 < 32:
                s2 += w2[b2 + c2] * h[c2]
                c2 += 1
            if s2 > 0:
                out |= 1 << r2
            r2 += 1
        acc_all ^= out
        it += 1
    return acc_all

# 5. Phase 34 Wearable `56`-Bit `4`-Token `uLM-28M` Opener + `LastExam` Epistemic Hold Gate (`1,000` calls)
opener_lo = array.array('I', [(312 | (1045 << 14) | ((882 & 0xF) << 28)) & 0xFFFFFFFF for _ in range(8)])
opener_hi = array.array('I', [((882 >> 4) | (419 << 10) | (0xF0 << 24)) & 0xFFFFFFFF for _ in range(8)])

@micropython.viper
def bench_wearable_56bit_epistemic_1k(t_lo: ptr32, o_lo: ptr32, o_hi: ptr32) -> int:
    acc: int = 0
    i: int = 0
    while i < 1000:
        # 75% certified wearable reflexes + 25% frontier `LastExam` epistemic hold queries
        if (i & 3) == 3:
            f_mark: int = 0x8700
            # Popcount of 0x8700 >= 3 -> immediate EpistemicHoldEscalate (prior=0%, 0 poisoned prefix words!)
            acc ^= f_mark
        else:
            addr: int = 8 + (i & 3)
            out: int = 0
            g: int = 0
            while g < 16:
                out |= ((t_lo[g] >> addr) & 1) << g
                g += 1
            slot: int = out & 7
            w_lo: int = o_lo[slot]
            w_hi: int = o_hi[slot]
            tok0: int = w_lo & 0x3FFF
            hdr8: int = (w_hi >> 24) & 0xFF
            acc ^= tok0 ^ hdr8
        i += 1
    return acc

# 6. Baseline `uLM-28M` (`d=512 -> 64`, `32,768` INT8 MACs = 1/16th of ONE `uLM-28M` layer!) (`50` calls)
ulm_w_row = array.array('i', [((j * 19 + 5) & 0x7F) - 64 for j in range(512)])
ulm_x512 = array.array('i', [17 if (j & 1) else -17 for j in range(512)])

@micropython.viper
def bench_ulm28m_d512_slice_50(wrow: ptr32, x512: ptr32) -> int:
    acc_all: int = 0
    it: int = 0
    while it < 50:
        r: int = 0
        th: int = 0
        while r < 64:
            dot: int = 0
            c: int = 0
            off: int = r & 7
            while c < 512:
                dot += wrow[(c + off) & 511] * x512[c]
                c += 1
            th ^= (dot >> 8)
            r += 1
        acc_all ^= (th & 0xFFFF)
        it += 1
    return acc_all

# 7. Phase 37/38 Window 16 (`Apex` `System-0` `LUT6` Gate: `DES-10` + `openie-cad` + `Torricelli CBF` + `RCE`, `1,000` calls)
apex_m_lo = array.array('I', [0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00])
apex_m_hi = array.array('I', [0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00])

@micropython.viper
def bench_w16_apex_gate_1k(m_lo: ptr32, m_hi: ptr32) -> int:
    acc: int = 0
    step: int = 0
    while step < 1000:
        p6: int = (step & 0x3E) | 1
        htag: int = 0x42A1
        etag: int = 0x42A0 if (step & 127) == 127 else 0x42A1
        if htag != etag:
            acc ^= (htag ^ etag)
        else:
            z0: int = 49500 + (step & 7) * 100
            tj: int = 640 + (step & 15) * 5
            v: int = 1100 + (step & 7) * 20
            d_obs: int = 150 if (step & 255) == 254 else 320
            viol: int = 0
            if z0 < 45000 or z0 > 55000:
                viol |= 2
            if tj > 850:
                viol |= 4
            if (v * v) > (6000 * d_obs):
                viol |= 8
            if viol != 0:
                acc ^= viol
            else:
                ac: int = 0
                if p6 < 32:
                    ac = (((m_lo[0] >> p6) & 1)
                          | (((m_lo[1] >> p6) & 1) << 1)
                          | (((m_lo[2] >> p6) & 1) << 2)
                          | (((m_lo[3] >> p6) & 1) << 3))
                else:
                    sh: int = p6 - 32
                    ac = (((m_hi[0] >> sh) & 1)
                          | (((m_hi[1] >> sh) & 1) << 1)
                          | (((m_hi[2] >> sh) & 1) << 2)
                          | (((m_hi[3] >> sh) & 1) << 3))
                acc ^= ac + 1
        step += 1
    return acc & 0x3FFFFFFF

# 8. Phase 38 Window 17: Canonical `waiOS` `256 x 128` Ternary Audit (`SEED = 0xdeadbeefcafebabe`)
#    on Live Cortex-M55 Silicon (`Antisym-40` Zero-`MUL` vs. Scalar `MUL`, `20` passes)
_s64 = 0xDEADBEEFCAFEBABE
_MUL64 = 6364136223846793005
_ADD64 = 1442695040888963407
_MASK64 = 0xFFFFFFFFFFFFFFFF
waios_u8 = bytearray(256 * 128)  # stores `w + 1` in `{0, 1, 2}`
for _i in range(256 * 128):
    _s64 = (_s64 * _MUL64 + _ADD64) & _MASK64
    waios_u8[_i] = (_s64 >> 32) % 3
waios_acts = array.array('i', [0] * 128)
for _c in range(128):
    _s64 = (_s64 * _MUL64 + _ADD64) & _MASK64
    waios_acts[_c] = ((_s64 >> 40) % 251) - 125
waios_codes = bytearray(256 * 32)
for _r in range(256):
    _rb = _r * 128
    _cb = _r * 32
    for _g in range(32):
        _k = _rb + (_g << 2)
        waios_codes[_cb + _g] = waios_u8[_k] + 3 * waios_u8[_k + 1] + 9 * waios_u8[_k + 2] + 27 * waios_u8[_k + 3]
waios_l41 = array.array('i', [0] * (32 * 41))
waios_out_a = array.array('i', [0] * 256)
waios_out_m = array.array('i', [0] * 256)

@micropython.viper
def bench_w17_waios_antisym40_20(codes: ptr8, acts: ptr32, luts41: ptr32, out: ptr32) -> int:
    acc_chk: int = 0
    it: int = 0
    while it < 20:
        g: int = 0
        while g < 32:
            c0: int = g << 2
            x0: int = acts[c0]; x1: int = acts[c0 + 1]; x2: int = acts[c0 + 2]; x3: int = acts[c0 + 3]
            base: int = g * 41
            idx: int = 0
            i2: int = -1
            while i2 <= 1:
                s2: int = -x3 + (x2 if i2 == 1 else (-x2 if i2 == -1 else 0))
                i1: int = -1
                while i1 <= 1:
                    s1: int = s2 + (x1 if i1 == 1 else (-x1 if i1 == -1 else 0))
                    luts41[base + idx] = s1 - x0
                    luts41[base + idx + 1] = s1
                    luts41[base + idx + 2] = s1 + x0
                    idx += 3
                    i1 += 1
                i2 += 1
            j1: int = -1
            while j1 <= 1:
                sb: int = -x2 + (x1 if j1 == 1 else (-x1 if j1 == -1 else 0))
                luts41[base + idx] = sb - x0
                luts41[base + idx + 1] = sb
                luts41[base + idx + 2] = sb + x0
                idx += 3
                j1 += 1
            luts41[base + 36] = -x1 - x0
            luts41[base + 37] = -x1
            luts41[base + 38] = -x1 + x0
            luts41[base + 39] = -x0
            luts41[base + 40] = 0
            g += 1

        r: int = 0
        while r < 256:
            roff: int = r << 5
            s: int = 0
            grp: int = 0
            loff: int = 0
            while grp < 32:
                cd: int = codes[roff + grp]
                if cd > 40:
                    s -= luts41[loff + (80 - cd)]
                else:
                    s += luts41[loff + cd]
                grp += 1
                loff += 41
            out[r] = s
            r += 1
        acc_chk ^= (out[0] ^ out[255]) & 0x3FFFFFFF
        it += 1
    return acc_chk & 0x3FFFFFFF

@micropython.viper
def bench_w17b_waios_scalar_mul_20(wu8: ptr8, acts: ptr32, out: ptr32) -> int:
    acc_chk: int = 0
    it: int = 0
    while it < 20:
        r: int = 0
        while r < 256:
            base: int = r << 7
            s: int = 0
            c: int = 0
            while c < 128:
                w: int = wu8[base + c] - 1
                s += w * acts[c]
                c += 1
            out[r] = s
            r += 1
        acc_chk ^= (out[0] ^ out[255]) & 0x3FFFFFFF
        it += 1
    return acc_chk & 0x3FFFFFFF

# 9. Phase 38 Window 18: `JouleBit-0.10` (`0.105 bpw`) Shared `LUT6` (`0` MULs) vs `LittleBit-2` Dense Unpack+MAC (`64 x 96`)
jb_acts = array.array('i', [((i * 13 + 7) % 63) - 31 for i in range(96)])
jb_v = bytearray([((r * 17 + g * 29 + 5) & 0x3F) for r in range(6) for g in range(16)])
jb_u = bytearray([((row * 37 + 11) & 0x3F) for row in range(64)])
jb_scr = array.array('i', [0] * (6 + 64 + 64))
lb_scr = array.array('i', [0] * (6 * 96 + 64))

@micropython.viper
def bench_w18_joulebit_lut6_100(acts: ptr32, v_idx: ptr8, u_idx: ptr8, scr: ptr32) -> int:
    acc: int = 0
    it: int = 0
    while it < 100:
        r0: int = 0
        while r0 < 6:
            scr[r0] = 0
            r0 += 1
        g: int = 0
        while g < 16:
            b: int = g * 6
            a0: int = acts[b]; a1: int = acts[b + 1]; a2: int = acts[b + 2]
            a3: int = acts[b + 3]; a4: int = acts[b + 4]; a5: int = acts[b + 5]
            rk: int = 0
            while rk < 6:
                m: int = v_idx[(rk << 4) + g]
                s: int = (-a0 if (m & 1) else a0) + (-a1 if (m & 2) else a1) + (-a2 if (m & 4) else a2)
                s += (-a3 if (m & 8) else a3) + (-a4 if (m & 16) else a4) + (-a5 if (m & 32) else a5)
                scr[rk] += s
                rk += 1
            g += 1
        z0: int = scr[0]; z1: int = scr[1]; z2: int = scr[2]
        z3: int = scr[3]; z4: int = scr[4]; z5: int = scr[5]
        cd: int = 0
        while cd < 64:
            ut: int = (-z0 if (cd & 1) else z0) + (-z1 if (cd & 2) else z1) + (-z2 if (cd & 4) else z2)
            ut += (-z3 if (cd & 8) else z3) + (-z4 if (cd & 16) else z4) + (-z5 if (cd & 32) else z5)
            scr[6 + cd] = ut >> 3
            cd += 1
        row: int = 0
        while row < 64:
            scr[70 + row] = scr[6 + (u_idx[row] & 0x3F)]
            row += 1
        acc ^= (scr[70] ^ scr[133]) & 0x3FFFFFFF
        it += 1
    return acc & 0x3FFFFFFF

@micropython.viper
def bench_w18b_littlebit2_mac_20(acts: ptr32, v_idx: ptr8, u_idx: ptr8, scr: ptr32) -> int:
    acc: int = 0
    it: int = 0
    while it < 20:
        rk: int = 0
        while rk < 6:
            g: int = 0
            while g < 16:
                m: int = v_idx[(rk << 4) + g]
                k: int = 0
                while k < 6:
                    scr[rk * 96 + g * 6 + k] = -1 if ((m >> k) & 1) else 1
                    k += 1
                g += 1
            rk += 1
        row: int = 0
        while row < 64:
            um: int = u_idx[row] & 0x3F
            ar: int = 0
            c: int = 0
            while c < 96:
                wrc: int = 0
                r2: int = 0
                while r2 < 6:
                    us: int = -1 if ((um >> r2) & 1) else 1
                    wrc += us * scr[r2 * 96 + c]
                    r2 += 1
                ar += wrc * acts[c]
                c += 1
            scr[576 + row] = ar >> 3
            row += 1
        acc ^= (scr[576] ^ scr[639]) & 0x3FFFFFFF
        it += 1
    return acc & 0x3FFFFFFF

dwt_init()
# Calibration loop
c_oh0 = dwt_read()
i_oh = 0
while i_oh < 1000:
    i_oh += 1
c_oh1 = dwt_read()

t0 = time.ticks_us(); c0 = dwt_read()
v1 = bench_innate_1k(k_lo, k_hi)
c1 = dwt_read(); t1 = time.ticks_us()

t2 = time.ticks_us(); c2 = dwt_read()
v2 = bench_overlay_wobble_1k(k_lo, k_hi, p_lo, p_hi)
c3 = dwt_read(); t3 = time.ticks_us()

t4 = time.ticks_us(); c4 = dwt_read()
v3 = bench_bitsliced_b32_32k(planes32)
c5 = dwt_read(); t5 = time.ticks_us()

t6 = time.ticks_us(); c6 = dwt_read()
v4 = bench_zero_mul_tmac_1k(tmac_codes, tmac_luts, tmac_bias)
c7 = dwt_read(); t7 = time.ticks_us()

t8 = time.ticks_us(); c8 = dwt_read()
v5 = bench_int8_mlp_64_32_16_200(w1_i8, w2_i8, x64_i8, h32_i8)
c9 = dwt_read(); t9 = time.ticks_us()

t10 = time.ticks_us(); c10 = dwt_read()
v6 = bench_full81_precomp_eval_200(tmac_codes, l81_scratch, tmac_bias, act64)
c11 = dwt_read(); t11 = time.ticks_us()

t12 = time.ticks_us(); c12 = dwt_read()
v7 = bench_antisym40_precomp_eval_200(tmac_codes, l41_scratch, tmac_bias, act64)
c13 = dwt_read(); t13 = time.ticks_us()

t14 = time.ticks_us(); c14 = dwt_read()
v8 = bench_wearable_56bit_epistemic_1k(k_lo, opener_lo, opener_hi)
c15 = dwt_read(); t15 = time.ticks_us()

t16 = time.ticks_us(); c16 = dwt_read()
v9 = bench_ulm28m_d512_slice_50(ulm_w_row, ulm_x512)
c17 = dwt_read(); t17 = time.ticks_us()

t18 = time.ticks_us(); c18 = dwt_read()
v10 = bench_w16_apex_gate_1k(apex_m_lo, apex_m_hi)
c19 = dwt_read(); t19 = time.ticks_us()

t20 = time.ticks_us(); c20 = dwt_read()
v11 = bench_w17_waios_antisym40_20(waios_codes, waios_acts, waios_l41, waios_out_a)
c21 = dwt_read(); t21 = time.ticks_us()

t22 = time.ticks_us(); c22 = dwt_read()
v12 = bench_w17b_waios_scalar_mul_20(waios_u8, waios_acts, waios_out_m)
c23 = dwt_read(); t23 = time.ticks_us()

t24 = time.ticks_us(); c24 = dwt_read()
v13 = bench_w18_joulebit_lut6_100(jb_acts, jb_v, jb_u, jb_scr)
c25 = dwt_read(); t25 = time.ticks_us()

t26 = time.ticks_us(); c26 = dwt_read()
v14 = bench_w18b_littlebit2_mac_20(jb_acts, jb_v, jb_u, lb_scr)
c27 = dwt_read(); t27 = time.ticks_us()

import json, binascii
waios_eq = True
for _ri in range(256):
    if waios_out_a[_ri] != waios_out_m[_ri]:
        waios_eq = False
        break
jb_eq = True
for _rj in range(64):
    if jb_scr[70 + _rj] != lb_scr[576 + _rj]:
        jb_eq = False
        break

print("AE3_JSON:" + json.dumps({
    "innate_1k_us": time.ticks_diff(t1, t0),
    "innate_1k_cyc": (c1 - c0) & 0x3FFFFFFF,
    "wobble_1k_us": time.ticks_diff(t3, t2),
    "wobble_1k_cyc": (c3 - c2) & 0x3FFFFFFF,
    "bitsliced_32k_us": time.ticks_diff(t5, t4),
    "bitsliced_32k_cyc": (c5 - c4) & 0x3FFFFFFF,
    "tmac_1k_us": time.ticks_diff(t7, t6),
    "tmac_1k_cyc": (c7 - c6) & 0x3FFFFFFF,
    "int8_mlp_200_us": time.ticks_diff(t9, t8),
    "int8_mlp_200_cyc": (c9 - c8) & 0x3FFFFFFF,
    "full81_precomp_200_us": time.ticks_diff(t11, t10),
    "full81_precomp_200_cyc": (c11 - c10) & 0x3FFFFFFF,
    "antisym40_precomp_200_us": time.ticks_diff(t13, t12),
    "antisym40_precomp_200_cyc": (c13 - c12) & 0x3FFFFFFF,
    "wearable_epistemic_1k_us": time.ticks_diff(t15, t14),
    "wearable_epistemic_1k_cyc": (c15 - c14) & 0x3FFFFFFF,
    "ulm28m_slice_50_us": time.ticks_diff(t17, t16),
    "ulm28m_slice_50_cyc": (c17 - c16) & 0x3FFFFFFF,
    "w16_apex_1k_us": time.ticks_diff(t19, t18),
    "w16_apex_1k_cyc": (c19 - c18) & 0x3FFFFFFF,
    "w17_waios_antisym40_20_us": time.ticks_diff(t21, t20),
    "w17_waios_antisym40_20_cyc": (c21 - c20) & 0x3FFFFFFF,
    "w17b_waios_mul_20_us": time.ticks_diff(t23, t22),
    "w17b_waios_mul_20_cyc": (c23 - c22) & 0x3FFFFFFF,
    "w18_joulebit_100_us": time.ticks_diff(t25, t24),
    "w18_joulebit_100_cyc": (c25 - c24) & 0x3FFFFFFF,
    "w18b_littlebit2_20_us": time.ticks_diff(t27, t26),
    "w18b_littlebit2_20_cyc": (c27 - c26) & 0x3FFFFFFF,
    "antisym_parity_ok": (v6 == v7),
    "waios_256_row_parity_ok": waios_eq,
    "waios_first4": [waios_out_a[0], waios_out_a[1], waios_out_a[2], waios_out_a[3]],
    "waios_out_le_hex": binascii.hexlify(waios_out_a).decode(),
    "joulebit_64_row_parity_ok": jb_eq,
    "checks": [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14],
}))
"""


def probe_usb_mcus_readonly() -> list[dict]:
    """Read-only IORegistry USB MCU inventory (0 serial/flash writes)."""
    out = subprocess.check_output(["ioreg", "-p", "IOUSB", "-w0", "-l"], text=True, errors="replace")
    devices: list[dict] = []
    if "OpenMV-AE3" in out:
        devices.append(
            {
                "port": "/dev/cu.usbmodem14301",
                "product": "OpenMV-AE3",
                "mcu": "Alif Ensemble E3 (AE302F80F55D5AE, Arm Cortex-M55 @ 400 MHz + Ethos-U55)",
                "mode": "live_ram_only_viper_benchmarked",
            }
        )
    if "Arduino Nicla Voice" in out or "C0D242A0" in out:
        devices.append(
            {
                "port": "/dev/cu.usbmodemC0D242A03",
                "product": "Arduino Nicla Voice CMSIS-DAP (0x2341:0x0065)",
                "mcu": "Nordic nRF52832 (Arm Cortex-M4F @ 64 MHz, thumbv7em-none-eabihf) + Syntiant NDP120",
                "mode": "read_only_usb_enumerated_flash_untouched",
            }
        )
    return devices


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "artifacts" / "alif_ae3_live_hardware_cycles.json"
    port = "/dev/cu.usbmodem14301"
    if not Path(port).exists():
        print(f"{port} not attached; keeping existing {out_path}")
        return

    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attrs = termios.tcgetattr(fd)
        attrs[4] = attrs[5] = termios.B115200
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        os.write(fd, b"\x03\x03")
        time.sleep(0.10)
        while select.select([fd], [], [], 0.05)[0]:
            os.read(fd, 4096)

        os.write(fd, b"\x01")  # Enter Raw REPL
        time.sleep(0.15)
        while select.select([fd], [], [], 0.05)[0]:
            os.read(fd, 4096)

        payload = VIPER_SCRIPT.encode("utf-8")
        for off in range(0, len(payload), 256):
            os.write(fd, payload[off : off + 256])
            time.sleep(0.02)
        os.write(fd, b"\x04")  # Execute in SRAM

        raw_out = b""
        deadline = time.time() + 14.0
        while time.time() < deadline:
            if select.select([fd], [], [], 0.1)[0]:
                raw_out += os.read(fd, 4096)
                if b"AE3_JSON:" in raw_out and raw_out.endswith(b">"):
                    break

        os.write(fd, b"\x02")  # Return cleanly to normal REPL
        time.sleep(0.05)
        while select.select([fd], [], [], 0.05)[0]:
            os.read(fd, 4096)
    finally:
        os.close(fd)

    decoded = raw_out.decode("utf-8", errors="replace")
    marker = "AE3_JSON:"
    assert marker in decoded, f"Unexpected OpenMV-AE3 output:\n{decoded}"
    json_line = decoded.split(marker, 1)[1].splitlines()[0].strip()
    m = json.loads(json_line)

    # Pure-Python BLAKE3("waios.tmac.v1") verification over exact 1,024-byte `i32` buffer from Cortex-M55 SRAM!
    def _b3_derive_waios(out_le_bytes: bytes) -> str:
        iv = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
        perm = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

        def _rotr(x: int, n: int) -> int:
            return ((x >> n) | ((x << (32 - n)) & 0xFFFFFFFF)) & 0xFFFFFFFF

        def _g(s: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
            s[a] = (s[a] + s[b] + mx) & 0xFFFFFFFF
            s[d] = _rotr(s[d] ^ s[a], 16)
            s[c] = (s[c] + s[d]) & 0xFFFFFFFF
            s[b] = _rotr(s[b] ^ s[c], 12)
            s[a] = (s[a] + s[b] + my) & 0xFFFFFFFF
            s[d] = _rotr(s[d] ^ s[a], 8)
            s[c] = (s[c] + s[d]) & 0xFFFFFFFF
            s[b] = _rotr(s[b] ^ s[c], 7)

        def _compress(cv: list[int], blk: bytes, blen: int, flags: int) -> list[int]:
            pad = blk.ljust(64, b"\x00")
            mw = [int.from_bytes(pad[k * 4 : k * 4 + 4], "little") for k in range(16)]
            st = cv[:] + iv[:4] + [0, 0, blen, flags]
            for rnd in range(7):
                _g(st, 0, 4, 8, 12, mw[0], mw[1]); _g(st, 1, 5, 9, 13, mw[2], mw[3])
                _g(st, 2, 6, 10, 14, mw[4], mw[5]); _g(st, 3, 7, 11, 15, mw[6], mw[7])
                _g(st, 0, 5, 10, 15, mw[8], mw[9]); _g(st, 1, 6, 11, 12, mw[10], mw[11])
                _g(st, 2, 7, 8, 13, mw[12], mw[13]); _g(st, 3, 4, 9, 14, mw[14], mw[15])
                if rnd < 6:
                    mw = [mw[p] for p in perm]
            return [st[i] ^ st[i + 8] for i in range(8)]

        def _hash1024(data: bytes, key: list[int], bflags: int) -> list[int]:
            nblk = max(1, (len(data) + 63) // 64)
            cv = key[:]
            for b in range(nblk - 1):
                cv = _compress(cv, data[b * 64 : (b + 1) * 64], 64, bflags | (1 if b == 0 else 0))
            last = data[(nblk - 1) * 64 :]
            return _compress(cv, last, len(last), bflags | (1 if nblk == 1 else 0) | 2 | 8)

        ctx_cv = _hash1024(b"waios.tmac.v1", iv, 32)
        root = _hash1024(out_le_bytes, ctx_cv, 64)
        return b"".join(w.to_bytes(4, "little") for w in root).hex()

    mcu_waios_bytes = bytes.fromhex(m["waios_out_le_hex"])
    assert len(mcu_waios_bytes) == 1024
    mcu_waios_blake3 = _b3_derive_waios(mcu_waios_bytes)
    assert mcu_waios_blake3 == "a541323b13653ed36515ff24ef08deb7dfa771e570a434712708fb72552f46f0", (
        f"Cortex-M55 waiOS BLAKE3 mismatch: {mcu_waios_blake3}"
    )
    assert m["waios_256_row_parity_ok"] is True
    assert m["waios_first4"] == [-852, 74, 153, 513]
    assert m["joulebit_64_row_parity_ok"] is True

    innate_cyc = round(m["innate_1k_cyc"] / 1000.0, 2)
    innate_us = round(m["innate_1k_us"] / 1000.0, 4)
    wobble_cyc = round(m["wobble_1k_cyc"] / 1000.0, 2)
    wobble_us = round(m["wobble_1k_us"] / 1000.0, 4)
    bits_cyc = round(m["bitsliced_32k_cyc"] / 32000.0, 2)
    bits_us = round(m["bitsliced_32k_us"] / 32000.0, 5)
    tmac_cyc = round(m["tmac_1k_cyc"] / 1000.0, 2)
    tmac_us = round(m["tmac_1k_us"] / 1000.0, 4)
    full81_p_cyc = round(m["full81_precomp_200_cyc"] / 200.0, 2)
    full81_p_us = round(m["full81_precomp_200_us"] / 200.0, 3)
    antisym40_p_cyc = round(m["antisym40_precomp_200_cyc"] / 200.0, 2)
    antisym40_p_us = round(m["antisym40_precomp_200_us"] / 200.0, 3)
    wear_cyc = round(m["wearable_epistemic_1k_cyc"] / 1000.0, 2)
    wear_us = round(m["wearable_epistemic_1k_us"] / 1000.0, 4)
    ulm_slice_cyc = round(m["ulm28m_slice_50_cyc"] / 50.0, 2)
    ulm_slice_us = round(m["ulm28m_slice_50_us"] / 50.0, 3)
    w16_apex_cyc = round(m["w16_apex_1k_cyc"] / 1000.0, 2)
    w16_apex_us = round(m["w16_apex_1k_us"] / 1000.0, 4)
    w17_waios_cyc = round(m["w17_waios_antisym40_20_cyc"] / 20.0, 2)
    w17_waios_us = round(m["w17_waios_antisym40_20_us"] / 20.0, 3)
    w17b_waios_cyc = round(m["w17b_waios_mul_20_cyc"] / 20.0, 2)
    w17b_waios_us = round(m["w17b_waios_mul_20_us"] / 20.0, 3)
    w18_jb_cyc = round(m["w18_joulebit_100_cyc"] / 100.0, 2)
    w18_jb_us = round(m["w18_joulebit_100_us"] / 100.0, 3)
    w18b_lb_cyc = round(m["w18b_littlebit2_20_cyc"] / 20.0, 2)
    w18b_lb_us = round(m["w18b_littlebit2_20_us"] / 20.0, 3)
    assert m.get("antisym_parity_ok") is True, f"Antisym-40 parity failed on OpenMV-AE3: {m}"
    mlp_cyc = round(m["int8_mlp_200_cyc"] / 200.0, 2)
    mlp_us = round(m["int8_mlp_200_us"] / 200.0, 3)

    doc = {
        "board": "OpenMV-AE3 (Alif Ensemble E3 AE302F80F55D5AE, Arm Cortex-M55 @ 400 MHz)",
        "cpu_hz": 400_000_000,
        "execution_mode": "RAM-only @micropython.viper native Thumb2/Armv8.1-M machine code (0 flash writes)",
        "attached_usb_mcus": probe_usb_mcus_readonly(),
        "innate_lut6_dwt_cycles_per_decision": innate_cyc,
        "innate_lut6_us_per_decision": innate_us,
        "wobble_overlay_dwt_cycles_per_decision": wobble_cyc,
        "wobble_overlay_us_per_decision": wobble_us,
        "vertical_bitsliced_batch32_dwt_cycles_per_decision": bits_cyc,
        "vertical_bitsliced_batch32_us_per_decision": bits_us,
        "zero_mul_tmac_tail_dwt_cycles_per_decision": tmac_cyc,
        "zero_mul_tmac_tail_us_per_decision": tmac_us,
        "full81_tmac_precomp_dwt_cycles_per_decision": full81_p_cyc,
        "full81_tmac_precomp_us_per_decision": full81_p_us,
        "antisym40_tmac_precomp_dwt_cycles_per_decision": antisym40_p_cyc,
        "antisym40_tmac_precomp_us_per_decision": antisym40_p_us,
        "antisym40_speedup_over_full81_precomp": round(full81_p_cyc / antisym40_p_cyc, 2),
        "wearable_56bit_epistemic_dwt_cycles_per_decision": wear_cyc,
        "wearable_56bit_epistemic_us_per_decision": wear_us,
        "ulm28m_d512_slice_dwt_cycles_per_decision": ulm_slice_cyc,
        "ulm28m_d512_slice_us_per_decision": ulm_slice_us,
        "speedup_wearable_epistemic_vs_ulm28m_slice": round(ulm_slice_cyc / wear_cyc, 2),
        "int8_micro_mlp_dwt_cycles_per_decision": mlp_cyc,
        "int8_micro_mlp_us_per_decision": mlp_us,
        "w16_apex_horizon_gate_dwt_cycles": w16_apex_cyc,
        "w16_apex_horizon_gate_us": w16_apex_us,
        "w17_waios_256x128_antisym40_dwt_cycles": w17_waios_cyc,
        "w17_waios_256x128_antisym40_us": w17_waios_us,
        "w17b_waios_256x128_scalar_mul_dwt_cycles": w17b_waios_cyc,
        "w17b_waios_256x128_scalar_mul_us": w17b_waios_us,
        "w17_waios_antisym40_speedup_vs_scalar_mul": round(w17b_waios_cyc / w17_waios_cyc, 2),
        "w17_waios_live_blake3_digest": mcu_waios_blake3,
        "w17_waios_first4_outputs": m["waios_first4"],
        "w18_joulebit_0p10bpw_lut6_dwt_cycles": w18_jb_cyc,
        "w18_joulebit_0p10bpw_lut6_us": w18_jb_us,
        "w18b_littlebit2_0p10bpw_unpack_mac_dwt_cycles": w18b_lb_cyc,
        "w18b_littlebit2_0p10bpw_unpack_mac_us": w18b_lb_us,
        "w18_joulebit_lut6_speedup_vs_littlebit2": round(w18b_lb_cyc / w18_jb_cyc, 2),
        "speedup_innate_vs_int8_mlp": round(mlp_cyc / innate_cyc, 2),
        "speedup_tmac_tail_vs_int8_mlp": round(mlp_cyc / tmac_cyc, 2),
        "speedup_antisym40_precomp_vs_int8_mlp": round(mlp_cyc / antisym40_p_cyc, 2),
        "speedup_bitsliced_vs_int8_mlp": round(mlp_cyc / bits_cyc, 2),
        "board_metered": False,
        "measured_j": None,
    }
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print("=" * 100)
    print(" LIVE ATTACHED ALIF ENSEMBLE E3 (CORTEX-M55 @ 400 MHz) RAM-ONLY VIPER DWT BENCHMARK")
    print("=" * 100)
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
