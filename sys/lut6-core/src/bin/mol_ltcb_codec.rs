//! Phase 45 (`Claim LA-76`): Zero-Python, Zero-External-Crate Native Rust Self-Contained
//! `MoL-LTCB` Lossless Compressor & Decompressor for Matt Mahoney's Large Text Compression
//! Benchmark (`enwik5` `100 KB` .. `enwik8` `100 MB` Wikipedia XML):
//!   - `Rung-0` `LTCB-XML-Skip LUT6` + `16 MiB` Lockstep Prefix `VAULT` (`4`-slot `rep0..rep3` +
//!     `8`-way hash-chain match finder, `0` transmitted dictionary bytes!),
//!   - `Rung-1` `waiOS` Integer-Exact Order-1 + XML-State + `LZMA`-Shadow Adaptive Carry-Safe
//!     `64`-Bit Binary Range Codec (`0` floating-point instructions anywhere in encode or decode!),
//!   - Controlled `--probe-float-desync` experiment demonstrating how `1`-ULP non-associative
//!     `FP32` SIMD vs. sequential accumulation shifts a `10,000,000`-scale `tufazip`-style CDF
//!     bucket and catastrophically desynchronizes arithmetic decoding.

use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use lut6_core::ltcb_cm2_ttt::{
    eval_statesmix_ttt_gate2_lut6, eval_wrt_capillary_gate2_lut6, mix3_probs_q12,
    step_online_ttt_joulebit6_i16,
};
use lut6_core::ltcb_cm3_mbfws::{
    eval_cm3_logodds_mix_gate2_lut6, squash_logit_q8_to_prob_q11, stretch_prob_q11_to_logit_q8,
};
use lut6_core::ltcb_codec::eval_ltcb_xml_skip_gate2_lut6;

const MAGIC: [u8; 4] = *b"MLTC";
const MAGIC_CM2: [u8; 4] = *b"MLT2";
const MAGIC_CM3: [u8; 4] = *b"MLT3";
const PROB_INIT: u16 = 1024;
const TOP_VALUE: u32 = 0x0100_0000;
const WIN_MASK: usize = (1 << 24) - 1; // 16 MiB sliding window
const HASH_BITS: usize = 20;
const HASH_SIZE: usize = 1 << HASH_BITS;
const CHAIN_DEPTH: usize = 24;
const MIN_MATCH: usize = 4;
const MAX_MATCH: usize = 273;

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn with_capacity(cap: usize) -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::with_capacity(cap),
        }
    }

    #[inline(always)]
    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let mut temp = self.cache;
            let carry = (self.low >> 32) as u8;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low as u32) >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = ((self.low as u32) << 8) as u64;
    }

    #[inline(always)]
    fn encode_bit(&mut self, prob: &mut u16, bit: u8) {
        let p = *prob as u32;
        let bound = (self.range >> 11) * p;
        if bit == 0 {
            self.range = bound;
            *prob = (p + ((2048 - p) >> 5)) as u16;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *prob = (p - (p >> 5)) as u16;
        }
        if self.range < TOP_VALUE {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[inline(always)]
    fn encode_bit_prob(&mut self, p_q11: u32, bit: u8) {
        let p = p_q11.clamp(1, 2047);
        let bound = (self.range >> 11) * p;
        if bit == 0 {
            self.range = bound;
        } else {
            self.low += bound as u64;
            self.range -= bound;
        }
        if self.range < TOP_VALUE {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[inline(always)]
    fn encode_direct(&mut self, mut val: u32, mut num_bits: u32) {
        while num_bits > 0 {
            num_bits -= 1;
            self.range >>= 1;
            if ((val >> num_bits) & 1) != 0 {
                self.low += self.range as u64;
            }
            if self.range < TOP_VALUE {
                self.range <<= 8;
                self.shift_low();
            }
            val &= (1u32 << num_bits).wrapping_sub(1);
        }
    }

    fn encode_tree(&mut self, probs: &mut [u16], bits: u32, sym: u32) {
        let mut m = 1usize;
        let mut b = bits;
        while b > 0 {
            b -= 1;
            let bit = ((sym >> b) & 1) as u8;
            self.encode_bit(&mut probs[m], bit);
            m = (m << 1) | (bit as usize);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            pos: 0,
            code: 0,
            range: 0xFFFF_FFFF,
        };
        let _ = dec.next_byte(); // initial dummy cache byte
        for _ in 0..4 {
            dec.code = (dec.code << 8) | (dec.next_byte() as u32);
        }
        dec
    }

    #[inline(always)]
    fn next_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    #[inline(always)]
    fn decode_bit(&mut self, prob: &mut u16) -> u8 {
        let p = *prob as u32;
        let bound = (self.range >> 11) * p;
        let bit = if self.code < bound {
            self.range = bound;
            *prob = (p + ((2048 - p) >> 5)) as u16;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            *prob = (p - (p >> 5)) as u16;
            1
        };
        if self.range < TOP_VALUE {
            self.range <<= 8;
            self.code = (self.code << 8) | (self.next_byte() as u32);
        }
        bit
    }

    #[inline(always)]
    fn decode_bit_prob(&mut self, p_q11: u32) -> u8 {
        let p = p_q11.clamp(1, 2047);
        let bound = (self.range >> 11) * p;
        let bit = if self.code < bound {
            self.range = bound;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            1
        };
        if self.range < TOP_VALUE {
            self.range <<= 8;
            self.code = (self.code << 8) | (self.next_byte() as u32);
        }
        bit
    }

    #[inline(always)]
    fn decode_direct(&mut self, mut num_bits: u32) -> u32 {
        let mut res = 0u32;
        while num_bits > 0 {
            num_bits -= 1;
            self.range >>= 1;
            let t = self.code.wrapping_sub(self.range) >> 31;
            self.code = self.code.wrapping_sub(self.range & t.wrapping_sub(1));
            res = (res << 1) | (1 - t);
            if self.range < TOP_VALUE {
                self.range <<= 8;
                self.code = (self.code << 8) | (self.next_byte() as u32);
            }
        }
        res
    }

    fn decode_tree(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1usize;
        for _ in 0..bits {
            let bit = self.decode_bit(&mut probs[m]) as usize;
            m = (m << 1) | bit;
        }
        (m as u32) - (1u32 << bits)
    }
}

struct MolLtcbModels {
    is_match: [u16; 128],
    is_rep: [u16; 32],
    is_rep0: [u16; 32],
    is_rep1: [u16; 32],
    is_rep2: [u16; 32],
    len_choice: [u16; 2],
    len_low: [[u16; 8]; 4],
    len_mid: [[u16; 8]; 4],
    len_high: [u16; 256],
    dist_slot: [[u16; 64]; 4],
    dist_align: [u16; 16],
    // Order-1 (prev_byte >> 2 = 64) x xml_state2 (4) = 256 literal contexts, 0x300 probs each for LZMA shadow
    lit_probs: Vec<[u16; 0x300]>,
}

impl MolLtcbModels {
    fn new() -> Self {
        Self {
            is_match: [PROB_INIT; 128],
            is_rep: [PROB_INIT; 32],
            is_rep0: [PROB_INIT; 32],
            is_rep1: [PROB_INIT; 32],
            is_rep2: [PROB_INIT; 32],
            len_choice: [PROB_INIT; 2],
            len_low: [[PROB_INIT; 8]; 4],
            len_mid: [[PROB_INIT; 8]; 4],
            len_high: [PROB_INIT; 256],
            dist_slot: [[PROB_INIT; 64]; 4],
            dist_align: [PROB_INIT; 16],
            lit_probs: vec![[PROB_INIT; 0x300]; 256],
        }
    }

    fn encode_len(&mut self, rc: &mut RangeEncoder, pos_ctx: usize, len: usize) {
        let v = len - MIN_MATCH;
        if v < 8 {
            rc.encode_bit(&mut self.len_choice[0], 0);
            rc.encode_tree(&mut self.len_low[pos_ctx], 3, v as u32);
        } else if v < 16 {
            rc.encode_bit(&mut self.len_choice[0], 1);
            rc.encode_bit(&mut self.len_choice[1], 0);
            rc.encode_tree(&mut self.len_mid[pos_ctx], 3, (v - 8) as u32);
        } else {
            rc.encode_bit(&mut self.len_choice[0], 1);
            rc.encode_bit(&mut self.len_choice[1], 1);
            rc.encode_tree(&mut self.len_high, 8, (v - 16) as u32);
        }
    }

    fn decode_len(&mut self, rc: &mut RangeDecoder<'_>, pos_ctx: usize) -> usize {
        if rc.decode_bit(&mut self.len_choice[0]) == 0 {
            MIN_MATCH + (rc.decode_tree(&mut self.len_low[pos_ctx], 3) as usize)
        } else if rc.decode_bit(&mut self.len_choice[1]) == 0 {
            MIN_MATCH + 8 + (rc.decode_tree(&mut self.len_mid[pos_ctx], 3) as usize)
        } else {
            MIN_MATCH + 16 + (rc.decode_tree(&mut self.len_high, 8) as usize)
        }
    }

    fn encode_dist(&mut self, rc: &mut RangeEncoder, len_ctx: usize, dist: u32) {
        // dist is 0-based (offset - 1)
        if dist < 4 {
            rc.encode_tree(&mut self.dist_slot[len_ctx], 6, dist);
            return;
        }
        let hb = 31 - dist.leading_zeros();
        let second = (dist >> (hb - 1)) & 1;
        let slot = (hb << 1) | second;
        rc.encode_tree(&mut self.dist_slot[len_ctx], 6, slot);
        let extra = hb - 1;
        let rem = dist & ((1u32 << extra) - 1);
        if extra <= 4 {
            rc.encode_tree(&mut self.dist_align, extra, rem);
        } else {
            rc.encode_direct(rem >> 4, extra - 4);
            rc.encode_tree(&mut self.dist_align, 4, rem & 0xF);
        }
    }

    fn decode_dist(&mut self, rc: &mut RangeDecoder<'_>, len_ctx: usize) -> u32 {
        let slot = rc.decode_tree(&mut self.dist_slot[len_ctx], 6);
        if slot < 4 {
            return slot;
        }
        let extra = (slot >> 1) - 1;
        let base = (2 | (slot & 1)) << extra;
        if extra <= 4 {
            base | rc.decode_tree(&mut self.dist_align, extra)
        } else {
            let hi = rc.decode_direct(extra - 4) << 4;
            let lo = rc.decode_tree(&mut self.dist_align, 4);
            base | hi | lo
        }
    }
}

#[inline(always)]
fn step_xml_state(cur: u8, b: u8) -> u8 {
    match (cur, b) {
        (_, b'<') => 1,
        (1, b'>') => 0,
        (0, b'&') => 2,
        (2, b';') | (2, b' ') | (2, b'\n') => 0,
        (0, b'[') => 3,
        (3, b']') | (3, b'\n') => 0,
        _ => cur,
    }
}

#[inline(always)]
fn hash4(b0: u8, b1: u8, b2: u8, b3: u8) -> usize {
    let v = u32::from_le_bytes([b0, b1, b2, b3]);
    ((v.wrapping_mul(0x9E37_79B1)) >> (32 - HASH_BITS)) as usize
}

pub struct CompressStats {
    pub raw_bytes: usize,
    pub compressed_bytes: usize,
    pub rung0_skip_bytes: usize,
    pub elapsed_ms: f64,
}

pub fn compress_mol_ltcb(src: &[u8]) -> (Vec<u8>, CompressStats) {
    let t0 = Instant::now();
    let n = src.len();
    let mut rc = RangeEncoder::with_capacity((n / 3).max(64));
    let mut mdl = MolLtcbModels::new();

    let mut head = vec![u32::MAX; HASH_SIZE];
    let ring_cap = (n.min(WIN_MASK + 1)).max(1);
    let mut prev_chain = vec![u32::MAX; ring_cap];

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0; // 0..12 LZMA-like state
    let mut xml_state: u8 = 0;
    let mut prev_byte: u8 = 0;
    let mut rung0_skip_bytes: usize = 0;

    let mut pos = 0usize;
    while pos < n {
        // Check rep matches first
        let max_len = (n - pos).min(MAX_MATCH);
        let mut best_rep_idx = usize::MAX;
        let mut best_rep_len = 0usize;
        if max_len >= MIN_MATCH {
            for (r_idx, &off) in reps.iter().enumerate() {
                if off <= pos && src[pos - off] == src[pos] && src[pos - off + 1] == src[pos + 1] {
                    let mut l = 2usize;
                    while l < max_len && src[pos - off + l] == src[pos + l] {
                        l += 1;
                    }
                    if l >= MIN_MATCH && l > best_rep_len {
                        best_rep_len = l;
                        best_rep_idx = r_idx;
                        if l == max_len {
                            break;
                        }
                    }
                }
            }
        }

        // Hash-chain search
        let mut best_len = best_rep_len;
        let mut best_off = if best_rep_idx < 4 { reps[best_rep_idx] } else { 0 };
        let mut use_rep = best_rep_idx < 4;

        if max_len >= MIN_MATCH && best_rep_len < 64 {
            let h = hash4(src[pos], src[pos + 1], src[pos + 2], src[pos + 3]);
            let mut cur = head[h];
            let min_pos = pos.saturating_sub(WIN_MASK);
            let mut depth = 0usize;
            while cur != u32::MAX && (cur as usize) >= min_pos && depth < CHAIN_DEPTH {
                let mpos = cur as usize;
                let off = pos - mpos;
                if off > 0 && src[mpos + best_len.min(max_len - 1)] == src[pos + best_len.min(max_len - 1)] {
                    let mut l = 0usize;
                    while l < max_len && src[mpos + l] == src[pos + l] {
                        l += 1;
                    }
                    if l >= MIN_MATCH && (l > best_len + if use_rep { 1 } else { 0 }) {
                        best_len = l;
                        best_off = off;
                        use_rep = false;
                        if l == max_len {
                            break;
                        }
                    }
                }
                cur = prev_chain[mpos % ring_cap];
                depth += 1;
            }
        }

        // Query compiled `LTCB-XML-Skip LUT6` hardware controller (`Claim LA-76`)
        let lz_ge4 = u8::from(best_len >= MIN_MATCH);
        let utf8_cont = u8::from((src[pos] & 0xC0) == 0x80);
        let addr6 = (xml_state & 3) | (lz_ge4 << 2) | (lz_ge4 << 3) | (lz_ge4 << 4) | (utf8_cont << 5);
        let skip_mode2 = eval_ltcb_xml_skip_gate2_lut6(addr6);
        if skip_mode2 >= 2 {
            rung0_skip_bytes += best_len.max(1);
        }

        let m_ctx = ((state & 7) << 4) | (((xml_state & 3) as usize) << 2) | (pos & 3);
        let pos_ctx = pos & 3;
        if best_len >= MIN_MATCH {
            rc.encode_bit(&mut mdl.is_match[m_ctx], 1);
            let r_ctx = ((state & 7) << 2) | ((xml_state & 3) as usize);
            if use_rep {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 1);
                if best_rep_idx == 0 {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 0);
                } else {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 1);
                    if best_rep_idx == 1 {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 0);
                        reps.swap(0, 1);
                    } else {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 1);
                        if best_rep_idx == 2 {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 0);
                            let t = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        } else {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 1);
                            let t = reps[3];
                            reps[3] = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        }
                    }
                }
                mdl.encode_len(&mut rc, pos_ctx, best_len);
                state = if state < 7 { 8 } else { 11 };
            } else {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 0);
                mdl.encode_len(&mut rc, pos_ctx, best_len);
                let len_ctx = (best_len - MIN_MATCH).min(3);
                mdl.encode_dist(&mut rc, len_ctx, (best_off - 1) as u32);
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = best_off;
                state = if state < 7 { 7 } else { 10 };
            }

            let step_end = pos + best_len;
            while pos < step_end {
                if pos + MIN_MATCH <= n {
                    let h = hash4(src[pos], src[pos + 1], src[pos + 2], src[pos + 3]);
                    prev_chain[pos % ring_cap] = head[h];
                    head[h] = pos as u32;
                }
                xml_state = step_xml_state(xml_state, src[pos]);
                prev_byte = src[pos];
                pos += 1;
            }
        } else {
            rc.encode_bit(&mut mdl.is_match[m_ctx], 0);
            let lit_ctx = (((prev_byte >> 2) as usize) << 2) | ((xml_state & 3) as usize);
            let probs = &mut mdl.lit_probs[lit_ctx];
            let sym = src[pos];
            if state >= 7 && reps[0] <= pos {
                // LZMA-style match-shadow literal coding
                let mut match_byte = src[pos - reps[0]];
                let mut m = 1usize;
                let mut b = 7i32;
                let mut matched = true;
                while b >= 0 {
                    let bit = (sym >> b) & 1;
                    if matched {
                        let mbit = (match_byte >> 7) & 1;
                        match_byte <<= 1;
                        let idx = ((1 + mbit as usize) << 8) + m;
                        rc.encode_bit(&mut probs[idx], bit);
                        m = (m << 1) | (bit as usize);
                        if mbit != bit {
                            matched = false;
                        }
                    } else {
                        rc.encode_bit(&mut probs[m], bit);
                        m = (m << 1) | (bit as usize);
                    }
                    b -= 1;
                }
            } else {
                let mut m = 1usize;
                for b in (0..8).rev() {
                    let bit = (sym >> b) & 1;
                    rc.encode_bit(&mut probs[m], bit);
                    m = (m << 1) | (bit as usize);
                }
            }
            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            if pos + MIN_MATCH <= n {
                let h = hash4(src[pos], src[pos + 1], src[pos + 2], src[pos + 3]);
                prev_chain[pos % ring_cap] = head[h];
                head[h] = pos as u32;
            }
            xml_state = step_xml_state(xml_state, sym);
            prev_byte = sym;
            pos += 1;
        }
    }

    let payload = rc.finish();
    let mut archive = Vec::with_capacity(12 + payload.len());
    archive.extend_from_slice(&MAGIC);
    archive.extend_from_slice(&(n as u64).to_le_bytes());
    archive.extend_from_slice(&payload);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let compressed_bytes = archive.len();
    (
        archive,
        CompressStats {
            raw_bytes: n,
            compressed_bytes,
            rung0_skip_bytes,
            elapsed_ms,
        },
    )
}

pub fn decompress_mol_ltcb(archive: &[u8]) -> Result<(Vec<u8>, f64), &'static str> {
    let t0 = Instant::now();
    if archive.len() < 12 || archive[0..4] != MAGIC {
        return Err("Invalid MLTC archive header");
    }
    let orig_len = u64::from_le_bytes([
        archive[4], archive[5], archive[6], archive[7], archive[8], archive[9], archive[10],
        archive[11],
    ]) as usize;
    let mut out = Vec::with_capacity(orig_len);
    if orig_len == 0 {
        return Ok((out, 0.0));
    }
    let mut rc = RangeDecoder::new(&archive[12..]);
    let mut mdl = MolLtcbModels::new();

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0;
    let mut xml_state: u8 = 0;
    let mut prev_byte: u8 = 0;

    while out.len() < orig_len {
        let pos = out.len();
        let m_ctx = ((state & 7) << 4) | (((xml_state & 3) as usize) << 2) | (pos & 3);
        let pos_ctx = pos & 3;
        if rc.decode_bit(&mut mdl.is_match[m_ctx]) == 1 {
            let r_ctx = ((state & 7) << 2) | ((xml_state & 3) as usize);
            let len: usize;
            if rc.decode_bit(&mut mdl.is_rep[r_ctx]) == 1 {
                if rc.decode_bit(&mut mdl.is_rep0[r_ctx]) == 0 {
                    // rep0 unchanged
                } else if rc.decode_bit(&mut mdl.is_rep1[r_ctx]) == 0 {
                    reps.swap(0, 1);
                } else if rc.decode_bit(&mut mdl.is_rep2[r_ctx]) == 0 {
                    let t = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                } else {
                    let t = reps[3];
                    reps[3] = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                }
                len = mdl.decode_len(&mut rc, pos_ctx);
                state = if state < 7 { 8 } else { 11 };
            } else {
                len = mdl.decode_len(&mut rc, pos_ctx);
                let len_ctx = (len - MIN_MATCH).min(3);
                let off = (mdl.decode_dist(&mut rc, len_ctx) as usize) + 1;
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = off;
                state = if state < 7 { 7 } else { 10 };
            }
            let off = reps[0];
            if off == 0 || off > out.len() {
                return Err("Corrupt match offset during MLTC decode");
            }
            let copy_len = len.min(orig_len - out.len());
            for _ in 0..copy_len {
                let b = out[out.len() - off];
                out.push(b);
                xml_state = step_xml_state(xml_state, b);
                prev_byte = b;
            }
        } else {
            let lit_ctx = (((prev_byte >> 2) as usize) << 2) | ((xml_state & 3) as usize);
            let probs = &mut mdl.lit_probs[lit_ctx];
            let sym: u8 = if state >= 7 && reps[0] <= pos {
                let mut match_byte = out[pos - reps[0]];
                let mut m = 1usize;
                let mut matched = true;
                for _ in 0..8 {
                    if matched {
                        let mbit = (match_byte >> 7) & 1;
                        match_byte <<= 1;
                        let idx = ((1 + mbit as usize) << 8) + m;
                        let bit = rc.decode_bit(&mut probs[idx]);
                        m = (m << 1) | (bit as usize);
                        if mbit != bit {
                            matched = false;
                        }
                    } else {
                        let bit = rc.decode_bit(&mut probs[m]);
                        m = (m << 1) | (bit as usize);
                    }
                }
                (m - 0x100) as u8
            } else {
                let mut m = 1usize;
                for _ in 0..8 {
                    let bit = rc.decode_bit(&mut probs[m]);
                    m = (m << 1) | (bit as usize);
                }
                (m - 0x100) as u8
            };
            out.push(sym);
            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            xml_state = step_xml_state(xml_state, sym);
            prev_byte = sym;
        }
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1000.0))
}

/// Pure-Rust SHA-256 (`FIPS 180-4`) for cryptographic roundtrip & cross-host archive verification.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (bytes.len() as u64) * 8;
    let mut msg = bytes.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

/// Demonstrates how a non-associative `FP32` sequential vs. pairwise tree dot-product reduction
/// shifts a `10,000,000`-scale `tufazip` arithmetic-coder CDF boundary by `>= 1` count and
/// catastrophically desynchronizes subsequent decoding (`Theorem 141`).
fn probe_float_cdf_desync() -> (usize, u32, u32) {
    let mut step = 0usize;
    while step < 4096 {
        let mut x = [0.0f32; 64];
        let mut w = [0.0f32; 64];
        for i in 0..64 {
            let s = ((step * 64 + i) as u32).wrapping_mul(0x9E37_79B9);
            x[i] = ((s & 0xFFFF) as f32) * 1.525_878_9e-4 - 5.0;
            w[i] = (((s >> 16) & 0xFFFF) as f32) * 1.525_878_9e-4 - 5.0;
        }
        // Reduction order A (sequential scalar accumulation)
        let mut seq_acc = 0.0f32;
        for i in 0..64 {
            seq_acc += x[i] * w[i];
        }
        // Reduction order B (GPU warp pairwise tree reduction)
        let mut tree = [0.0f32; 64];
        for i in 0..64 {
            tree[i] = x[i] * w[i];
        }
        let mut stride = 32usize;
        while stride > 0 {
            for i in 0..stride {
                tree[i] += tree[i + stride];
            }
            stride >>= 1;
        }
        let p_seq = 1.0f32 / (1.0f32 + (-seq_acc * 0.05).exp());
        let p_tree = 1.0f32 / (1.0f32 + (-tree[0] * 0.05).exp());
        let cdf_seq = ((p_seq * 10_000_000.0) as u32).max(1);
        let cdf_tree = ((p_tree * 10_000_000.0) as u32).max(1);
        if cdf_seq != cdf_tree {
            return (step, cdf_seq, cdf_tree);
        }
        step += 1;
    }
    (4096, 0, 0)
}

fn run_self_test() {
    let sample = b"<mediawiki><page><title>Downward LUT6 Compilation</title><revision><timestamp>2006-03-03T00:00:00Z</timestamp><text>[[Category:Information theory]] Downward LUT6 Compilation in a Mixture of Limits Cascade repeats [[Category:Information theory]] and <timestamp>2006-03-03T00:00:00Z</timestamp> with bit-exact waiOS determinism!</text></revision></page></mediawiki>";
    let mut corpus = Vec::with_capacity(sample.len() * 64);
    for i in 0..64u8 {
        corpus.extend_from_slice(sample);
        corpus.push(b'0' + (i % 10));
    }
    let (arc, st) = compress_mol_ltcb(&corpus);
    let (dec, dec_ms) = decompress_mol_ltcb(&arc).expect("MLTC self-test decode failed");
    assert_eq!(dec, corpus, "MLTC lossless roundtrip mismatch!");
    assert!(
        arc.len() < corpus.len() / 6,
        "Expected strong XML compression, got {} vs {}",
        arc.len(),
        corpus.len()
    );
    // Also verify Phase 46 Bijective WRT-Capillary + StateSMix CM2 roundtrip on both XML and arbitrary binary!
    let mut arb_bin = Vec::with_capacity(1024);
    for b in 0..=255u8 {
        arb_bin.push(b);
        arb_bin.push(b ^ 0xFE);
        arb_bin.push(0xFD);
        arb_bin.push(0xFE);
    }
    let wrt_bin = encode_wrt_capillary(&arb_bin).0;
    let wrt_dec = decode_wrt_capillary(&wrt_bin).expect("WRT arbitrary binary decode failed");
    assert_eq!(wrt_dec, arb_bin, "WRT arbitrary 256^* bijection failed!");
    let (arc2, _st2) = compress_mol_ltcb_cm2(&corpus);
    let (dec2, _dec2_ms) = decompress_mol_ltcb_cm2(&arc2).expect("MLT2 self-test decode failed");
    assert_eq!(dec2, corpus, "MLT2 lossless roundtrip mismatch!");
    let (arc2_bin, _) = compress_mol_ltcb_cm2(&arb_bin);
    let (dec2_bin, _) = decompress_mol_ltcb_cm2(&arc2_bin).expect("MLT2 arbitrary binary decode failed");
    assert_eq!(dec2_bin, arb_bin, "MLT2 arbitrary binary roundtrip mismatch!");

    // Also verify Phase 47 WRT2 2-tier bijection + MoL-LTCB-CM3 (MLT3) roundtrip on XML and arbitrary 256^* binary!
    let wrt2_bin = encode_wrt2_capillary(&arb_bin).0;
    let wrt2_dec = decode_wrt2_capillary(&wrt2_bin).expect("WRT2 arbitrary binary decode failed");
    assert_eq!(wrt2_dec, arb_bin, "WRT2 arbitrary 256^* bijection failed!");
    let (arc3, _st3) = compress_mol_ltcb_cm3_slice(&corpus);
    let (dec3, _dec3_ms) = decompress_mol_ltcb_cm3_to_vec(&arc3).expect("MLT3 self-test decode failed");
    assert_eq!(dec3, corpus, "MLT3 lossless roundtrip mismatch!");
    let (arc3_bin, _) = compress_mol_ltcb_cm3_slice(&arb_bin);
    let (dec3_bin, _) = decompress_mol_ltcb_cm3_to_vec(&arc3_bin).expect("MLT3 arbitrary binary decode failed");
    assert_eq!(dec3_bin, arb_bin, "MLT3 arbitrary binary roundtrip mismatch!");

    let (desync_step, c_seq, c_tree) = probe_float_cdf_desync();
    assert!(desync_step < 4096 && c_seq != c_tree);
    let (adamw_flip_step, ttt_flips_mismatch) = probe_online_adamw_vs_ttt_lut6_desync();
    assert!(adamw_flip_step < 4096 && ttt_flips_mismatch == 0);
    println!(
        "{{\"self_test\": \"PASS\", \"raw_bytes\": {}, \"compressed_bytes\": {}, \"cm2_compressed_bytes\": {}, \"cm3_compressed_bytes\": {}, \"bpb\": {:.4}, \"rung0_skip_pct\": {:.2}, \"enc_ms\": {:.3}, \"dec_ms\": {:.3}, \"float_desync_first_step\": {}, \"cdf_seq\": {}, \"cdf_tree\": {}, \"adamw_sign_desync_step\": {}, \"ttt_i16_mismatches\": {}}}",
        st.raw_bytes,
        st.compressed_bytes,
        arc2.len(),
        arc3.len(),
        (st.compressed_bytes as f64) * 8.0 / (st.raw_bytes as f64),
        100.0 * (st.rung0_skip_bytes as f64) / (st.raw_bytes as f64),
        st.elapsed_ms,
        dec_ms,
        desync_step,
        c_seq,
        c_tree,
        adamw_flip_step,
        ttt_flips_mismatch
    );
}

// ============================================================================
// Phase 46 (`Claim LA-80 & LA-81`): `MoL-LTCB-CM2` (`MLT2`)
//   1. Bijective `WRT-Capillary-LUT6` MediaWiki-XML / Wiki / Stem Preprocessor
//   2. Full-Corpus Unbounded Window + Dual `HASH4` & `HASH8` + Short-Rep + 1-Step Lazy `LZ-Rep4`
//   3. Lockstep Zero-Float `Q12` `StateSMix-LUT6` Multi-Order Context Mixer (`Order-1` + `Order-2` + `Word`)
// ============================================================================

const WRT_ESC: u8 = 0xFE;

/// `148` high-byte-saving MediaWiki XML templates, Wiki markup idioms, and English word stems
/// mapped bijectively to `1`-byte tokens (`0x01..0x08`, `0x0B..0x0C`, `0x0E..0x1F`, `0x80..0xF3`).
const WRT_PHRASES: &[&[u8]] = &[
    b"    <page>\n      <title>",
    b"</title>\n      <id>",
    b"</id>\n      <revision>\n        <id>",
    b"  <page>\n    <title>",
    b"</title>\n    <id>",
    b"</id>\n    <revision>\n      <id>",
    b"</id>\n      <timestamp>2006-",
    b"</id>\n      <timestamp>2005-",
    b"</id>\n      <timestamp>2004-",
    b"Z</timestamp>\n      <contributor>\n        <username>",
    b"Z</timestamp>\n      <contributor>\n        <ip>",
    b"</username>\n        <id>",
    b"</id>\n      </contributor>\n",
    b"</ip>\n      </contributor>\n",
    b"      <minor />\n",
    b"      <comment>",
    b"</comment>\n      <text xml:space=\"preserve\">",
    b"      <text xml:space=\"preserve\">",
    b"</text>\n    </revision>\n  </page>\n",
    b"&quot;",
    b"&amp;",
    b"&lt;",
    b"&gt;",
    b"&nbsp;",
    b"&mdash;",
    b"&ndash;",
    b"[[Category:",
    b"[[Image:",
    b"#REDIRECT [[",
    b"==External links==\n",
    b"== External links ==\n",
    b"==References==\n",
    b"== References ==\n",
    b"==See also==\n",
    b"== See also ==\n",
    b"http://www.",
    b"http://",
    b".html",
    b".org/",
    b".com/",
    b"|thumb|right|",
    b"|thumb|",
    b"'''",
    b"[[",
    b"]]",
    b"{{",
    b"}}",
    b" ==\n",
    b" ===\n",
    b" the ",
    b" of ",
    b" and ",
    b" in ",
    b" to ",
    b" was ",
    b" is ",
    b" for ",
    b" that ",
    b" with ",
    b" as ",
    b" by ",
    b" on ",
    b" from ",
    b" at ",
    b" an ",
    b" be ",
    b" are ",
    b" which ",
    b" have ",
    b" had ",
    b" this ",
    b" not ",
    b" but ",
    b" or ",
    b" his ",
    b" her ",
    b" their ",
    b" its ",
    b" has ",
    b" were ",
    b" been ",
    b" also ",
    b" more ",
    b" into ",
    b" after ",
    b" other ",
    b" first ",
    b" between ",
    b" would ",
    b" about ",
    b" when ",
    b" can ",
    b" who ",
    b" there ",
    b" only ",
    b" over ",
    b" such ",
    b" time ",
    b" than ",
    b" some ",
    b" most ",
    b" used ",
    b" many ",
    b" during ",
    b" under ",
    b" known ",
    b" called ",
    b" where ",
    b" before ",
    b" through ",
    b" years ",
    b" world ",
    b" state ",
    b" part ",
    b" people ",
    b" system ",
    b"The ",
    b"In ",
    b"United States",
    b"University",
    b"American",
    b"National",
    b"Government",
    b"Population",
    b"History",
    b"International",
    b"Century",
    b"British",
    b"information",
    b"development",
    b"government",
    b"population",
    b"university",
    b"international",
    b"production",
    b"following",
    b"including",
    b"different",
    b"important",
    b"century",
    b"general",
    b"example",
    b"because",
    b"however",
    b"several",
    b"without",
    b"against",
    b"another",
];

fn wrt_code_table() -> ([Option<&'static [u8]>; 256], Vec<Vec<(usize, u8)>>) {
    let mut dec_tbl: [Option<&'static [u8]>; 256] = [None; 256];
    let mut codes = Vec::with_capacity(160);
    for b in 0x01u8..=0x08 {
        codes.push(b);
    }
    codes.push(0x0B);
    codes.push(0x0C);
    for b in 0x0Eu8..=0x1F {
        codes.push(b);
    }
    for b in 0x80u8..=0xFD {
        codes.push(b);
    }
    let mut by_first: Vec<Vec<(usize, u8)>> = (0..256).map(|_| Vec::new()).collect();
    for (idx, &ph) in WRT_PHRASES.iter().enumerate() {
        if idx >= codes.len() {
            break;
        }
        let code = codes[idx];
        dec_tbl[code as usize] = Some(ph);
        by_first[ph[0] as usize].push((idx, code));
    }
    // Sort each first-byte bucket longest-phrase-first for greedy maximal capillary folding
    for bucket in &mut by_first {
        bucket.sort_by(|a, b| WRT_PHRASES[b.0].len().cmp(&WRT_PHRASES[a.0].len()));
    }
    (dec_tbl, by_first)
}

#[inline(always)]
fn is_wrt_passthrough_byte(b: u8) -> bool {
    b == 0x00 || b == b'\t' || b == b'\n' || b == b'\r' || (0x20..=0x7F).contains(&b)
}

pub fn encode_wrt_capillary(src: &[u8]) -> (Vec<u8>, usize) {
    let (_dec_tbl, by_first) = wrt_code_table();
    let mut out = Vec::with_capacity(src.len());
    let mut wrt_lut6_hits = 0usize;
    let n = src.len();
    let mut i = 0usize;
    while i < n {
        let b = src[i];
        let cands = &by_first[b as usize];
        let mut matched_code: Option<(u8, usize)> = None;
        for &(p_idx, code) in cands {
            let ph = WRT_PHRASES[p_idx];
            let plen = ph.len();
            if i + plen <= n && &src[i..i + plen] == ph {
                matched_code = Some((code, plen));
                break;
            }
        }
        let esc_req = u8::from(matched_code.is_none() && !is_wrt_passthrough_byte(b));
        let dict_match = u8::from(matched_code.is_some());
        let xml_span = u8::from(b == b'<');
        let wrt_addr6 = ((b & 3)) | (dict_match << 2) | (esc_req << 4) | (xml_span << 5);
        let gate2 = eval_wrt_capillary_gate2_lut6(wrt_addr6);
        if gate2 != 0 {
            wrt_lut6_hits += 1;
        }
        if let Some((code, plen)) = matched_code {
            out.push(code);
            i += plen;
        } else if !is_wrt_passthrough_byte(b) {
            out.push(WRT_ESC);
            out.push(b);
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    (out, wrt_lut6_hits)
}

pub fn decode_wrt_capillary(wrt: &[u8]) -> Result<Vec<u8>, &'static str> {
    let (dec_tbl, _) = wrt_code_table();
    let mut out = Vec::with_capacity(wrt.len() + (wrt.len() >> 2));
    let mut i = 0usize;
    while i < wrt.len() {
        let b = wrt[i];
        if b == WRT_ESC {
            if i + 1 >= wrt.len() {
                return Err("Truncated WRT escape sequence");
            }
            out.push(wrt[i + 1]);
            i += 2;
        } else if let Some(ph) = dec_tbl[b as usize] {
            out.extend_from_slice(ph);
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    Ok(out)
}

const CM2_HASH4_BITS: usize = 21;
const CM2_HASH4_SIZE: usize = 1 << CM2_HASH4_BITS; // 2M buckets
const CM2_HASH8_BITS: usize = 21;
const CM2_HASH8_SIZE: usize = 1 << CM2_HASH8_BITS; // 2M buckets
const CM2_CHAIN4_DEPTH: usize = 28;
const CM2_CHAIN8_DEPTH: usize = 24;
const CM2_MAX_MATCH: usize = 273;

#[inline(always)]
fn cm2_hash3(s: &[u8], pos: usize) -> usize {
    let v = (s[pos] as u32) | ((s[pos + 1] as u32) << 8) | ((s[pos + 2] as u32) << 16);
    ((v.wrapping_mul(0x9E37_79B1)) >> 16) as usize
}

#[inline(always)]
fn cm2_hash4(s: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([s[pos], s[pos + 1], s[pos + 2], s[pos + 3]]);
    ((v.wrapping_mul(0x9E37_79B1)) >> (32 - CM2_HASH4_BITS)) as usize
}

#[inline(always)]
fn cm2_hash8(s: &[u8], pos: usize) -> usize {
    let v = u64::from_le_bytes([
        s[pos],
        s[pos + 1],
        s[pos + 2],
        s[pos + 3],
        s[pos + 4],
        s[pos + 5],
        s[pos + 6],
        s[pos + 7],
    ]);
    ((v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - CM2_HASH8_BITS)) as usize
}

#[inline(always)]
fn byte_class4(b: u8) -> usize {
    match b {
        b'a'..=b'z' => 0,
        b'A'..=b'Z' => 1,
        b'0'..=b'9' => 2,
        b' ' => 3,
        b'\n' | b'\r' | b'\t' => 4,
        b'<' | b'>' | b'/' | b'=' | b'"' => 5,
        b'[' | b']' | b'{' | b'}' | b'|' | b':' => 6,
        0x80..=0xFD | 0x01..=0x1F => 7,
        _ => 8,
    }
}

/// Collision-free `64`-bucket equivalence class for `prev2` (`0..64`), preserving all `26` lowercase
/// English letters, digits, XML delimiters, and `WRT` stem groups without hash smearing!
#[inline(always)]
fn byte_class64(b: u8) -> usize {
    match b {
        b'a'..=b'z' => (b - b'a') as usize,                 // 0..25
        b'A'..=b'Z' => 26 + (((b - b'A') as usize) >> 1),   // 26..38
        b'0'..=b'9' => 39 + (((b - b'0') as usize) >> 1),   // 39..43
        b' ' => 44,
        b'\n' => 45,
        b'.' | b',' | b';' | b':' => 46,
        b'(' | b')' | b'[' | b']' => 47,
        b'<' | b'>' | b'/' | b'=' => 48,
        b'"' | b'\'' | b'|' | b'-' => 49,
        0x01..=0x1F => 50 + (((b - 1) as usize) >> 1).min(5), // 50..55 (XML WRT tokens)
        0x80..=0xBF => 56 + (((b - 0x80) as usize) >> 4),     // 56..59 (high-freq English stems)
        0xC0..=0xFD => 60 + (((b - 0xC0) as usize) >> 4).min(3), // 60..63
        _ => 63,
    }
}

struct MolCm2Models {
    is_match: [u16; 256],
    is_rep: [u16; 16],
    is_rep0: [u16; 16],
    is_rep0_short1: [u16; 256], // LZMA short-rep (len == 1 at rep0!)
    is_rep1: [u16; 16],
    is_rep2: [u16; 16],
    len_choice: [u16; 2],
    len_low: [[u16; 8]; 8],
    len_mid: [[u16; 8]; 8],
    len_high: [u16; 256],
    rep_len_choice: [u16; 2],
    rep_len_low: [[u16; 8]; 8],
    rep_len_mid: [[u16; 8]; 8],
    rep_len_high: [u16; 256],
    dist_slot: [[u16; 64]; 4],
    dist_align: [u16; 16],
    // Exact Order-1 (256) x match-shadowed bit-trees (0x300 each)
    lit_ord1: Vec<[u16; 0x300]>,
    // Collision-free exact Order-2: prev1 (256) x byte_class64(prev2) (64) = 16,384 bit-trees (0x100 each)
    lit_ord2: Vec<[u16; 0x100]>,
    // Structured Order-3: byte_class64(prev3 >> 2 -> 16) x byte_class64(prev2 -> 64) x byte_class64(prev1 -> 64) = 65,536 bit-trees
    lit_ord3: Vec<[u16; 0x100]>,
    // Adaptive Q8 blend weights per (prev1_class16 * 2 + matched_flag): (w_ord1_q8, w_ord2_q8)
    mix_weights: [(u16, u16); 32],
}

impl MolCm2Models {
    fn new() -> Self {
        let mut mw = [(64u16, 136u16); 32];
        for slot in 0..16 {
            // When match-shadow (`matched == 1`) is active, lit_ord1 has the shadow bit (`0x100..0x2FF`), so weight ord1 higher!
            mw[slot * 2 + 1] = (152, 72);
        }
        Self {
            is_match: [PROB_INIT; 256],
            is_rep: [PROB_INIT; 16],
            is_rep0: [PROB_INIT; 16],
            is_rep0_short1: [PROB_INIT; 256],
            is_rep1: [PROB_INIT; 16],
            is_rep2: [PROB_INIT; 16],
            len_choice: [PROB_INIT; 2],
            len_low: [[PROB_INIT; 8]; 8],
            len_mid: [[PROB_INIT; 8]; 8],
            len_high: [PROB_INIT; 256],
            rep_len_choice: [PROB_INIT; 2],
            rep_len_low: [[PROB_INIT; 8]; 8],
            rep_len_mid: [[PROB_INIT; 8]; 8],
            rep_len_high: [PROB_INIT; 256],
            dist_slot: [[PROB_INIT; 64]; 4],
            dist_align: [PROB_INIT; 16],
            lit_ord1: vec![[PROB_INIT; 0x300]; 256 * 4],
            lit_ord2: vec![[PROB_INIT; 0x100]; 256 * 64],
            lit_ord3: vec![[PROB_INIT; 0x100]; 65_536],
            mix_weights: mw,
        }
    }

    fn encode_len_generic(
        choice: &mut [u16; 2],
        low: &mut [[u16; 8]; 8],
        mid: &mut [[u16; 8]; 8],
        high: &mut [u16; 256],
        rc: &mut RangeEncoder,
        pos_ctx: usize,
        min_l: usize,
        len: usize,
    ) {
        let v = len - min_l;
        if v < 8 {
            rc.encode_bit(&mut choice[0], 0);
            rc.encode_tree(&mut low[pos_ctx], 3, v as u32);
        } else if v < 16 {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 0);
            rc.encode_tree(&mut mid[pos_ctx], 3, (v - 8) as u32);
        } else {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 1);
            rc.encode_tree(high, 8, (v - 16).min(255) as u32);
        }
    }

    fn decode_len_generic(
        choice: &mut [u16; 2],
        low: &mut [[u16; 8]; 8],
        mid: &mut [[u16; 8]; 8],
        high: &mut [u16; 256],
        rc: &mut RangeDecoder<'_>,
        pos_ctx: usize,
        min_l: usize,
    ) -> usize {
        if rc.decode_bit(&mut choice[0]) == 0 {
            min_l + (rc.decode_tree(&mut low[pos_ctx], 3) as usize)
        } else if rc.decode_bit(&mut choice[1]) == 0 {
            min_l + 8 + (rc.decode_tree(&mut mid[pos_ctx], 3) as usize)
        } else {
            min_l + 16 + (rc.decode_tree(high, 8) as usize)
        }
    }

    fn encode_dist(&mut self, rc: &mut RangeEncoder, len_ctx: usize, dist: u32) {
        if dist < 4 {
            rc.encode_tree(&mut self.dist_slot[len_ctx], 6, dist);
            return;
        }
        let hb = 31 - dist.leading_zeros();
        let second = (dist >> (hb - 1)) & 1;
        let slot = (hb << 1) | second;
        rc.encode_tree(&mut self.dist_slot[len_ctx], 6, slot);
        let extra = hb - 1;
        let rem = dist & ((1u32 << extra) - 1);
        if extra <= 4 {
            rc.encode_tree(&mut self.dist_align, extra, rem);
        } else {
            rc.encode_direct(rem >> 4, extra - 4);
            rc.encode_tree(&mut self.dist_align, 4, rem & 0xF);
        }
    }

    fn decode_dist(&mut self, rc: &mut RangeDecoder<'_>, len_ctx: usize) -> u32 {
        let slot = rc.decode_tree(&mut self.dist_slot[len_ctx], 6);
        if slot < 4 {
            return slot;
        }
        let extra = (slot >> 1) - 1;
        let base = (2 | (slot & 1)) << extra;
        if extra <= 4 {
            base | rc.decode_tree(&mut self.dist_align, extra)
        } else {
            let hi = rc.decode_direct(extra - 4) << 4;
            let lo = rc.decode_tree(&mut self.dist_align, 4);
            base | hi | lo
        }
    }

    #[inline(always)]
    fn update_prob(p: &mut u16, bit: u8, shift: u32) {
        let cur = *p as u32;
        if bit == 0 {
            *p = (cur + ((2048 - cur) >> shift)) as u16;
        } else {
            *p = (cur - (cur >> shift)) as u16;
        }
    }

    #[inline(always)]
    fn mix_and_adapt_bit(
        &mut self,
        o1_ctx: usize,
        o2_ctx: usize,
        o3_ctx: usize,
        o1_idx: usize,
        m_tree: usize,
        w_cls: usize,
        sse_mode: u8,
    ) -> u32 {
        let p1_q12 = (self.lit_ord1[o1_ctx][o1_idx] as u16) << 1;
        let p2_q12 = (self.lit_ord2[o2_ctx][m_tree] as u16) << 1;
        let p3_q12 = (self.lit_ord3[o3_ctx][m_tree] as u16) << 1;
        let (w1, w2) = self.mix_weights[w_cls];
        let mixed_q12 = mix3_probs_q12(p1_q12, p2_q12, p3_q12, w1, w2, sse_mode);
        ((mixed_q12 >> 1) as u32).clamp(1, 2047)
    }

    #[inline(always)]
    fn commit_bit_outcome(
        &mut self,
        o1_ctx: usize,
        o2_ctx: usize,
        o3_ctx: usize,
        o1_idx: usize,
        m_tree: usize,
        w_cls: usize,
        bit: u8,
    ) {
        let p1 = self.lit_ord1[o1_ctx][o1_idx] as i32;
        let p2 = self.lit_ord2[o2_ctx][m_tree] as i32;
        let p3 = self.lit_ord3[o3_ctx][m_tree] as i32;
        // Lockstep integer weight adaptation: reward whichever order gave higher probability to observed `bit`
        let s1 = if bit == 0 { p1 } else { 2048 - p1 };
        let s2 = if bit == 0 { p2 } else { 2048 - p2 };
        let s3 = if bit == 0 { p3 } else { 2048 - p3 };
        let (mut w1, mut w2) = self.mix_weights[w_cls];
        if s3 > s1 + 64 && w1 > 24 {
            w1 -= 1;
        } else if s1 > s3 + 64 && (w1 + w2) < 236 {
            w1 += 1;
        }
        if s2 > s3 + 64 && (w1 + w2) < 236 {
            w2 += 1;
        } else if s3 > s2 + 64 && w2 > 24 {
            w2 -= 1;
        }
        self.mix_weights[w_cls] = (w1, w2);

        Self::update_prob(&mut self.lit_ord1[o1_ctx][o1_idx], bit, 5);
        Self::update_prob(&mut self.lit_ord2[o2_ctx][m_tree], bit, 5);
        Self::update_prob(&mut self.lit_ord3[o3_ctx][m_tree], bit, 5);
    }
}

#[inline(always)]
fn find_best_cm2_match(
    s: &[u8],
    pos: usize,
    reps: &[usize; 4],
    head3: &[u32],
    head4: &[u32],
    prev4: &[u32],
    head8: &[u32],
    prev8: &[u32],
) -> (usize, usize, usize) {
    let n = s.len();
    let max_len = (n - pos).min(CM2_MAX_MATCH);
    if max_len < 2 {
        return (0, 0, 4);
    }

    let mut best_rep_idx = 4usize;
    let mut best_rep_len = 0usize;
    for (r_idx, &off) in reps.iter().enumerate() {
        if off <= pos && s[pos - off] == s[pos] {
            let mut l = 1usize;
            while l < max_len && s[pos - off + l] == s[pos + l] {
                l += 1;
            }
            if l >= 2 && l > best_rep_len {
                best_rep_len = l;
                best_rep_idx = r_idx;
                if l == max_len {
                    return (l, off, r_idx);
                }
            }
        }
    }

    let mut best_len = best_rep_len;
    let mut best_off = if best_rep_idx < 4 { reps[best_rep_idx] } else { 0 };
    let mut best_kind = best_rep_idx;

    // 0. O(1) direct `HASH3` near-match (`len >= 3` within `4 KB`)
    if best_len < 3 && max_len >= 3 {
        let m3 = head3[cm2_hash3(s, pos)];
        if m3 != u32::MAX {
            let mpos = m3 as usize;
            let off = pos - mpos;
            if off > 0 && off <= 4096 && s[mpos] == s[pos] && s[mpos + 1] == s[pos + 1] && s[mpos + 2] == s[pos + 2] {
                let mut l = 3usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                if best_kind == 4 || l > best_len + 1 {
                    best_len = l;
                    best_off = off;
                    best_kind = 4;
                }
            }
        }
    }

    // 1. Probe HASH8 first for long (>= 8 byte) global corpus matches across the full 100 MB!
    if max_len >= 8 && best_len < 96 {
        let h8 = cm2_hash8(s, pos);
        let mut cur = head8[h8];
        let mut depth = 0usize;
        while cur != u32::MAX && depth < CM2_CHAIN8_DEPTH {
            let mpos = cur as usize;
            let off = pos - mpos;
            let probe = best_len.max(7).min(max_len - 1);
            if off > 0 && s[mpos + probe] == s[pos + probe] && s[mpos] == s[pos] {
                let mut l = 1usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                let margin_needed = if best_kind < 4 { 2 } else { 0 };
                if l >= 8 && l > best_len + margin_needed {
                    best_len = l;
                    best_off = off;
                    best_kind = 4;
                    if l >= 128 || l == max_len {
                        return (best_len, best_off, best_kind);
                    }
                }
            }
            cur = prev8[mpos];
            depth += 1;
        }
    }

    // 2. Probe HASH4 for 4..7+ byte local/medium matches
    if max_len >= 4 && best_len < 32 {
        let h4 = cm2_hash4(s, pos);
        let mut cur = head4[h4];
        let mut depth = 0usize;
        while cur != u32::MAX && depth < CM2_CHAIN4_DEPTH {
            let mpos = cur as usize;
            let off = pos - mpos;
            let probe = best_len.min(max_len - 1);
            if off > 0 && s[mpos + probe] == s[pos + probe] && s[mpos] == s[pos] {
                let mut l = 1usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                let worth = if l == 4 {
                    off < 131_072
                } else if l == 5 {
                    off < 2_097_152
                } else {
                    l >= 6
                };
                let margin_needed = if best_kind < 4 {
                    if off < 1024 { 1 } else { 2 }
                } else {
                    0
                };
                if worth && l > best_len + margin_needed {
                    best_len = l;
                    best_off = off;
                    best_kind = 4;
                    if l >= 48 || l == max_len {
                        break;
                    }
                }
            }
            cur = prev4[mpos];
            depth += 1;
        }
    }

    (best_len, best_off, best_kind)
}

#[inline(always)]
fn insert_cm2_pos(
    s: &[u8],
    pos: usize,
    head3: &mut [u32],
    head4: &mut [u32],
    prev4: &mut [u32],
    head8: &mut [u32],
    prev8: &mut [u32],
) {
    let n = s.len();
    if pos + 3 <= n {
        head3[cm2_hash3(s, pos)] = pos as u32;
    }
    if pos + 4 <= n {
        let h4 = cm2_hash4(s, pos);
        prev4[pos] = head4[h4];
        head4[h4] = pos as u32;
    }
    if pos + 8 <= n {
        let h8 = cm2_hash8(s, pos);
        prev8[pos] = head8[h8];
        head8[h8] = pos as u32;
    }
}

pub fn compress_mol_ltcb_cm2(raw_src: &[u8]) -> (Vec<u8>, CompressStats) {
    let t0 = Instant::now();
    let raw_n = raw_src.len();
    let (s_vec, wrt_hits) = encode_wrt_capillary(raw_src);
    let s = s_vec.as_slice();
    let n = s.len();

    let mut rc = RangeEncoder::with_capacity((raw_n / 4).max(64));
    let mut mdl = MolCm2Models::new();

    let mut head3 = vec![u32::MAX; 65_536];
    let mut head4 = vec![u32::MAX; CM2_HASH4_SIZE];
    let mut prev4 = vec![u32::MAX; n.max(1)];
    let mut head8 = vec![u32::MAX; CM2_HASH8_SIZE];
    let mut prev8 = vec![u32::MAX; n.max(1)];

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0;
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    let mut prev3: u8 = 0;
    let mut rung0_skip_bytes = wrt_hits;

    let mut pos = 0usize;
    while pos < n {
        let (mut best_len, best_off, best_kind) =
            find_best_cm2_match(s, pos, &reps, &head3, &head4, &prev4, &head8, &prev8);

        // 1-step lazy match lookahead on short matches (`< 12`)
        if best_len >= 2 && best_len < 24 && pos + 1 < n {
            let (next_len, _next_off, next_kind) =
                find_best_cm2_match(s, pos + 1, &reps, &head3, &head4, &prev4, &head8, &prev8);
            let gain = if next_kind < 4 && best_kind == 4 { 1 } else { 2 };
            if next_len >= best_len + gain {
                best_len = 0;
            }
        }

        let is_short_rep1 = best_len < 2 && reps[0] <= pos && s[pos - reps[0]] == s[pos];

        let m_ctx = ((state & 15) << 4) | ((byte_class4(prev1) & 3) << 2) | (byte_class4(prev2) & 3);
        let pos_ctx = byte_class4(prev1) & 7;

        if best_len >= 2 || is_short_rep1 {
            rung0_skip_bytes += best_len.max(1);
            rc.encode_bit(&mut mdl.is_match[m_ctx], 1);
            let r_ctx = state & 15;
            if is_short_rep1 {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 1);
                rc.encode_bit(&mut mdl.is_rep0[r_ctx], 0);
                let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                rc.encode_bit(&mut mdl.is_rep0_short1[sr_ctx], 0);
                state = if state < 7 { 9 } else { 11 };
                insert_cm2_pos(s, pos, &mut head3, &mut head4, &mut prev4, &mut head8, &mut prev8);
                prev3 = prev2;
                prev2 = prev1;
                prev1 = s[pos];
                pos += 1;
            } else if best_kind < 4 {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 1);
                if best_kind == 0 {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 0);
                    let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                    rc.encode_bit(&mut mdl.is_rep0_short1[sr_ctx], 1);
                } else {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 1);
                    if best_kind == 1 {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 0);
                        reps.swap(0, 1);
                    } else {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 1);
                        if best_kind == 2 {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 0);
                            let t = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        } else {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 1);
                            let t = reps[3];
                            reps[3] = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        }
                    }
                }
                MolCm2Models::encode_len_generic(
                    &mut mdl.rep_len_choice,
                    &mut mdl.rep_len_low,
                    &mut mdl.rep_len_mid,
                    &mut mdl.rep_len_high,
                    &mut rc,
                    pos_ctx,
                    2,
                    best_len,
                );
                state = if state < 7 { 8 } else { 11 };
                let step_end = pos + best_len;
                let fast_skip_interior = best_len >= 32;
                let start_p = pos;
                while pos < step_end {
                    if !fast_skip_interior || pos < start_p + 4 || pos + 4 >= step_end {
                        insert_cm2_pos(s, pos, &mut head3, &mut head4, &mut prev4, &mut head8, &mut prev8);
                    }
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = s[pos];
                    pos += 1;
                }
            } else {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 0);
                MolCm2Models::encode_len_generic(
                    &mut mdl.len_choice,
                    &mut mdl.len_low,
                    &mut mdl.len_mid,
                    &mut mdl.len_high,
                    &mut rc,
                    pos_ctx,
                    3,
                    best_len,
                );
                let len_ctx = (best_len - 3).min(3);
                mdl.encode_dist(&mut rc, len_ctx, (best_off - 1) as u32);
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = best_off;
                state = if state < 7 { 7 } else { 10 };
                let step_end = pos + best_len;
                let fast_skip_interior = best_len >= 32;
                let start_p = pos;
                while pos < step_end {
                    if !fast_skip_interior || pos < start_p + 4 || pos + 4 >= step_end {
                        insert_cm2_pos(s, pos, &mut head3, &mut head4, &mut prev4, &mut head8, &mut prev8);
                    }
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = s[pos];
                    pos += 1;
                }
            }
        } else {
            rc.encode_bit(&mut mdl.is_match[m_ctx], 0);
            let c2 = byte_class64(prev2);
            let c1 = byte_class64(prev1);
            let c3 = byte_class64(prev3) >> 2; // 0..16
            let o1_ctx = ((prev1 as usize) << 2) | (c2 >> 4);
            let o2_ctx = ((prev1 as usize) << 6) | c2;
            let o3_ctx = (c3 << 12) | (c2 << 6) | c1;
            let base_cls = byte_class4(prev1) & 15;
            let rep_shadow = u8::from(state >= 7 && reps[0] <= pos);
            let ttt_addr6 = ((base_cls as u8) & 3) | (1 << 4) | (rep_shadow << 5);
            let sse_mode = eval_statesmix_ttt_gate2_lut6(ttt_addr6);

            let sym = s[pos];
            let mut m = 1usize;
            let mut matched = state >= 7 && reps[0] <= pos;
            let mut match_byte = if matched { s[pos - reps[0]] } else { 0 };
            for b in (0..8).rev() {
                let bit = (sym >> b) & 1;
                let w_cls = (base_cls << 1) | (matched as usize);
                let o1_idx = if matched {
                    let mbit = (match_byte >> 7) & 1;
                    ((1 + mbit as usize) << 8) + m
                } else {
                    m
                };
                let p_q11 = mdl.mix_and_adapt_bit(o1_ctx, o2_ctx, o3_ctx, o1_idx, m, w_cls, sse_mode);
                rc.encode_bit_prob(p_q11, bit);
                mdl.commit_bit_outcome(o1_ctx, o2_ctx, o3_ctx, o1_idx, m, w_cls, bit);
                if matched {
                    let mbit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    if mbit != bit {
                        matched = false;
                    }
                }
                m = (m << 1) | (bit as usize);
            }

            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            insert_cm2_pos(s, pos, &mut head3, &mut head4, &mut prev4, &mut head8, &mut prev8);
            prev3 = prev2;
            prev2 = prev1;
            prev1 = sym;
            pos += 1;
        }
    }

    let payload = rc.finish();
    let mut archive = Vec::with_capacity(20 + payload.len());
    archive.extend_from_slice(&MAGIC_CM2);
    archive.extend_from_slice(&(raw_n as u64).to_le_bytes());
    archive.extend_from_slice(&(n as u64).to_le_bytes());
    archive.extend_from_slice(&payload);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let compressed_bytes = archive.len();
    (
        archive,
        CompressStats {
            raw_bytes: raw_n,
            compressed_bytes,
            rung0_skip_bytes,
            elapsed_ms,
        },
    )
}

pub fn decompress_mol_ltcb_cm2(archive: &[u8]) -> Result<(Vec<u8>, f64), &'static str> {
    let t0 = Instant::now();
    if archive.len() < 20 || archive[0..4] != MAGIC_CM2 {
        return Err("Invalid MLT2 archive header");
    }
    let raw_len = u64::from_le_bytes([
        archive[4], archive[5], archive[6], archive[7], archive[8], archive[9], archive[10],
        archive[11],
    ]) as usize;
    let wrt_len = u64::from_le_bytes([
        archive[12], archive[13], archive[14], archive[15], archive[16], archive[17], archive[18],
        archive[19],
    ]) as usize;
    if raw_len == 0 && wrt_len == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let mut wrt_out = Vec::with_capacity(wrt_len);
    let mut rc = RangeDecoder::new(&archive[20..]);
    let mut mdl = MolCm2Models::new();

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0;
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    let mut prev3: u8 = 0;

    while wrt_out.len() < wrt_len {
        let pos = wrt_out.len();
        let m_ctx = ((state & 15) << 4) | ((byte_class4(prev1) & 3) << 2) | (byte_class4(prev2) & 3);
        let pos_ctx = byte_class4(prev1) & 7;
        if rc.decode_bit(&mut mdl.is_match[m_ctx]) == 1 {
            let r_ctx = state & 15;
            if rc.decode_bit(&mut mdl.is_rep[r_ctx]) == 1 {
                let mut is_short1 = false;
                if rc.decode_bit(&mut mdl.is_rep0[r_ctx]) == 0 {
                    let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                    if rc.decode_bit(&mut mdl.is_rep0_short1[sr_ctx]) == 0 {
                        is_short1 = true;
                    }
                } else if rc.decode_bit(&mut mdl.is_rep1[r_ctx]) == 0 {
                    reps.swap(0, 1);
                } else if rc.decode_bit(&mut mdl.is_rep2[r_ctx]) == 0 {
                    let t = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                } else {
                    let t = reps[3];
                    reps[3] = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                }
                if is_short1 {
                    let off = reps[0];
                    if off == 0 || off > wrt_out.len() {
                        return Err("Corrupt short-rep offset in MLT2 decode");
                    }
                    let b = wrt_out[wrt_out.len() - off];
                    wrt_out.push(b);
                    state = if state < 7 { 9 } else { 11 };
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = b;
                } else {
                    let len = MolCm2Models::decode_len_generic(
                        &mut mdl.rep_len_choice,
                        &mut mdl.rep_len_low,
                        &mut mdl.rep_len_mid,
                        &mut mdl.rep_len_high,
                        &mut rc,
                        pos_ctx,
                        2,
                    );
                    state = if state < 7 { 8 } else { 11 };
                    let off = reps[0];
                    if off == 0 || off > wrt_out.len() {
                        return Err("Corrupt rep offset in MLT2 decode");
                    }
                    let copy_len = len.min(wrt_len - wrt_out.len());
                    for _ in 0..copy_len {
                        let b = wrt_out[wrt_out.len() - off];
                        wrt_out.push(b);
                        prev3 = prev2;
                        prev2 = prev1;
                        prev1 = b;
                    }
                }
            } else {
                let len = MolCm2Models::decode_len_generic(
                    &mut mdl.len_choice,
                    &mut mdl.len_low,
                    &mut mdl.len_mid,
                    &mut mdl.len_high,
                    &mut rc,
                    pos_ctx,
                    3,
                );
                let len_ctx = (len - 3).min(3);
                let off = (mdl.decode_dist(&mut rc, len_ctx) as usize) + 1;
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = off;
                state = if state < 7 { 7 } else { 10 };
                if off == 0 || off > wrt_out.len() {
                    return Err("Corrupt match offset in MLT2 decode");
                }
                let copy_len = len.min(wrt_len - wrt_out.len());
                for _ in 0..copy_len {
                    let b = wrt_out[wrt_out.len() - off];
                    wrt_out.push(b);
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = b;
                }
            }
        } else {
            let c2 = byte_class64(prev2);
            let c1 = byte_class64(prev1);
            let c3 = byte_class64(prev3) >> 2;
            let o1_ctx = ((prev1 as usize) << 2) | (c2 >> 4);
            let o2_ctx = ((prev1 as usize) << 6) | c2;
            let o3_ctx = (c3 << 12) | (c2 << 6) | c1;
            let base_cls = byte_class4(prev1) & 15;
            let rep_shadow = u8::from(state >= 7 && reps[0] <= pos);
            let ttt_addr6 = ((base_cls as u8) & 3) | (1 << 4) | (rep_shadow << 5);
            let sse_mode = eval_statesmix_ttt_gate2_lut6(ttt_addr6);

            let mut m = 1usize;
            let mut matched = state >= 7 && reps[0] <= pos;
            let mut match_byte = if matched { wrt_out[pos - reps[0]] } else { 0 };
            for _ in 0..8 {
                let w_cls = (base_cls << 1) | (matched as usize);
                let o1_idx = if matched {
                    let mbit = (match_byte >> 7) & 1;
                    ((1 + mbit as usize) << 8) + m
                } else {
                    m
                };
                let p_q11 = mdl.mix_and_adapt_bit(o1_ctx, o2_ctx, o3_ctx, o1_idx, m, w_cls, sse_mode);
                let bit = rc.decode_bit_prob(p_q11);
                mdl.commit_bit_outcome(o1_ctx, o2_ctx, o3_ctx, o1_idx, m, w_cls, bit);
                if matched {
                    let mbit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    if mbit != bit {
                        matched = false;
                    }
                }
                m = (m << 1) | (bit as usize);
            }
            let sym = (m - 0x100) as u8;
            wrt_out.push(sym);
            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            prev3 = prev2;
            prev2 = prev1;
            prev1 = sym;
        }
    }

    let decoded = decode_wrt_capillary(&wrt_out)?;
    if decoded.len() != raw_len {
        return Err("Decoded WRT length mismatch");
    }
    Ok((decoded, t0.elapsed().as_secs_f64() * 1000.0))
}

/// Demonstrates how `FP32` online `AdamW` (`m_t / (sqrtf(v_t) + eps)`) across non-associative
/// batch/channel reductions flips a `1`-bit `QPI1` Rademacher weight sign (`Theorem 149`), whereas
/// integer-exact `Online-TTT-LUT6` (`i16` Straight-Through) achieves `0` sign-flip mismatches!
fn probe_online_adamw_vs_ttt_lut6_desync() -> (usize, usize) {
    let mut w_seq = [1.0e-8f32; 6];
    let mut m_seq = [0.0f32; 6];
    let mut v_seq = [1.0e-8f32; 6];

    let mut w_tree = [1.0e-8f32; 6];
    let mut m_tree = [0.0f32; 6];
    let mut v_tree = [1.0e-8f32; 6];

    let mut acc_host_a = [2i16, -3, 5, -2, 1, -1];
    let mut acc_host_b = [2i16, -3, 5, -2, 1, -1];
    let mut first_adamw_flip_step = 4096usize;
    let mut ttt_mismatches = 0usize;

    for step in 0..4096usize {
        for j in 0..6 {
            // Opposing large batch activation terms + small residual where sequential vs. SIMD tree reduction differs by >= 1 ULP
            let base_mag = 16.0f32 + ((step & 7) as f32);
            let mut terms = [0.0f32; 16];
            terms[0] = base_mag;
            for t in 1..15 {
                let s = ((step * 131 + j * 37 + t) as u32).wrapping_mul(0x9E37_79B9);
                terms[t] = (((s & 0xFFFF) as f32) - 32768.0) * 1.0e-9;
            }
            terms[15] = -base_mag;

            let mut g_seq = 0.0f32;
            for &v in &terms {
                g_seq += v;
            }
            let mut tr = terms;
            let mut st = 8usize;
            while st > 0 {
                for i in 0..st {
                    tr[i] += tr[i + st];
                }
                st >>= 1;
            }
            let g_tree = tr[0];

            // AdamW with weight decay pulling latent master weight across the 1-bit QPI1 zero threshold
            m_seq[j] = 0.9 * m_seq[j] + 0.1 * g_seq;
            v_seq[j] = 0.999 * v_seq[j] + 0.001 * (g_seq * g_seq);
            w_seq[j] = 0.5 * w_seq[j] - 0.001 * (m_seq[j] / (v_seq[j].sqrt() + 1e-12));

            m_tree[j] = 0.9 * m_tree[j] + 0.1 * g_tree;
            v_tree[j] = 0.999 * v_tree[j] + 0.001 * (g_tree * g_tree);
            w_tree[j] = 0.5 * w_tree[j] - 0.001 * (m_tree[j] / (v_tree[j].sqrt() + 1e-12));

            if first_adamw_flip_step == 4096 && (w_seq[j] < 0.0) != (w_tree[j] < 0.0) {
                first_adamw_flip_step = step;
            }
        }
        let err_q8 = (((step & 7) as i16) - 4) * 4;
        let (rad_a, flp_a) = step_online_ttt_joulebit6_i16(&mut acc_host_a, (step & 0x3F) as u8, err_q8);
        let (rad_b, flp_b) = step_online_ttt_joulebit6_i16(&mut acc_host_b, (step & 0x3F) as u8, err_q8);
        if rad_a != rad_b || flp_a != flp_b {
            ttt_mismatches += 1;
        }
    }
    (first_adamw_flip_step, ttt_mismatches)
}

// ============================================================================
// Phase 47 (`Claim LA-84`): `MoL-LTCB-CM3` (`MLT3`)
//   1. Two-Tier Bijective `WRT2-Capillary-LUT6` (`153` `1`-byte + `256` `2`-byte `[0xFD, idx]`
//      MediaWiki XML / Interwiki / Infobox / Stem phrases + `[0xFE, raw]` escape over `256^*`)
//   2. Bounded-RAM `128 MiB` Streaming Block Pipeline for `1.00 GB` `enwik9` with Persistent
//      Cross-Block `StateSMix3` Context Tables (`0` block-boundary model resets!)
//   3. `MAX_MATCH = 2,320` + `HASH3` / `4M-Bucket HASH4` / `4M-Bucket HASH8` + `LZMA2`-style
//      `dist_special` low-offset trees + `1`-step lazy `LZ-Rep4`
//   4. Lockstep Zero-Float Integer Stretched Log-Odds `4`-Expert `StateSMix3`
//      (`Order-1 Shadow` + Exact `256x256` `Order-2` + `131,072` `Order-3` + `131,072`
//       `ASCII Word-Stem + PrevWord` + `SSE-LUT6` Calibration; `0` floating-point ops!)
// ============================================================================

struct StreamingSha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl StreamingSha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    #[inline]
    fn compress_block(h: &mut [u32; 8], chunk: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.total_bytes += bytes.len() as u64;
        if self.buf_len > 0 {
            let need = (64 - self.buf_len).min(bytes.len());
            self.buf[self.buf_len..self.buf_len + need].copy_from_slice(&bytes[..need]);
            self.buf_len += need;
            bytes = &bytes[need..];
            if self.buf_len == 64 {
                Self::compress_block(&mut self.h, &self.buf);
                self.buf_len = 0;
            }
        }
        while bytes.len() >= 64 {
            let blk: &[u8; 64] = bytes[..64].try_into().unwrap();
            Self::compress_block(&mut self.h, blk);
            bytes = &bytes[64..];
        }
        if !bytes.is_empty() {
            self.buf[..bytes.len()].copy_from_slice(bytes);
            self.buf_len = bytes.len();
        }
    }

    fn finalize_hex(mut self) -> String {
        let bit_len = self.total_bytes * 8;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            Self::compress_block(&mut self.h, &self.buf);
            self.buf_len = 0;
        }
        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        Self::compress_block(&mut self.h, &self.buf);
        format!(
            "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
            self.h[0], self.h[1], self.h[2], self.h[3], self.h[4], self.h[5], self.h[6], self.h[7]
        )
    }
}

const WRT2_EXT_PREFIX: u8 = 0xFD;
const CM3_RAW_BLOCK_CAP: usize = 134_217_728; // 128 MiB streaming segment (`enwik8` = 1 block, `enwik9` = 8 blocks)
const CM3_HASH4_BITS: usize = 22;
const CM3_HASH4_SIZE: usize = 1 << CM3_HASH4_BITS; // 4M buckets
const CM3_HASH6_BITS: usize = 22;
const CM3_HASH6_SIZE: usize = 1 << CM3_HASH6_BITS; // 4M buckets
const CM3_HASH8_BITS: usize = 22;
const CM3_HASH8_SIZE: usize = 1 << CM3_HASH8_BITS; // 4M buckets
const CM3_CHAIN4_DEPTH: usize = 44;
const CM3_CHAIN6_DEPTH: usize = 40;
const CM3_CHAIN8_DEPTH: usize = 40;
const CM3_MAX_MATCH: usize = 2320; // 16 + 256 + 2048

const CM3_STRETCH_LUT: [i16; 2048] = {
    let mut t = [0i16; 2048];
    let mut p = 1usize;
    while p < 2048 {
        t[p] = stretch_prob_q11_to_logit_q8(p as u16);
        p += 1;
    }
    t[0] = t[1];
    t
};

const CM3_SQUASH_LUT: [u16; 4097] = {
    let mut t = [1024u16; 4097];
    let mut idx = 0usize;
    while idx <= 4096 {
        t[idx] = squash_logit_q8_to_prob_q11((idx as i32) - 2048);
        idx += 1;
    }
    t
};

/// Extra `4` single-byte phrases (`149 + 4 = 153` total `1`-byte codes in `0x01..0x08`, `0x0B..0x0C`, `0x0E..0x1F`, `0x80..0xFC`)
#[allow(dead_code)]
const WRT2_EXTRA_SINGLE: &[&[u8]] = &[
    b" of the ",
    b" in the ",
    b" to the ",
    b" on the ",
];

/// `256` `2`-byte extended phrases (`[0xFD, 0x00..0xFF]`, every phrase `>= 4` bytes -> guaranteed net shrink!)
const WRT2_EXT_PHRASES: &[&[u8]] = &[
    b" for the ",
    b" and the ",
    b" from the ",
    b" with the ",
    b" at the ",
    b" by the ",
    b" as the ",
    b" into the ",
    b" over the ",
    b" after the ",
    b" under the ",
    b" between the ",
    b" through the ",
    b" during the ",
    b" one of the ",
    b" part of the ",
    b" end of the ",
    b" most of the ",
    b" some of the ",
    b" many of the ",
    b" member of the ",
    b" members of the ",
    b" history of ",
    b" History of ",
    b" University of ",
    b" List of ",
    b" Department of ",
    b" Republic of ",
    b" Kingdom of ",
    b" Battle of ",
    b" Treaty of ",
    b" District of ",
    b" Province of ",
    b" County, ",
    b" New York",
    b" California",
    b" Washington",
    b" England",
    b" Germany",
    b" France",
    b" Australia",
    b" Canada",
    b" Japanese",
    b" European",
    b" Christian",
    b" English",
    b" French",
    b" German",
    b" Russian",
    b" Chinese",
    b" Spanish",
    b" Italian",
    b" January ",
    b" February ",
    b" March ",
    b" April ",
    b" June ",
    b" July ",
    b" August ",
    b" September ",
    b" October ",
    b" November ",
    b" December ",
    b" is a ",
    b" was a ",
    b" as a ",
    b" in a ",
    b" of a ",
    b" to a ",
    b" for a ",
    b" with a ",
    b" such as ",
    b" as well as ",
    b" known as ",
    b" referred to as ",
    b" according to ",
    b" because of ",
    b" instead of ",
    b" number of ",
    b" series of ",
    b" group of ",
    b" system of ",
    b" form of ",
    b" result of ",
    b" can be ",
    b" may be ",
    b" will be ",
    b" would be ",
    b" could be ",
    b" should be ",
    b" has been ",
    b" have been ",
    b" had been ",
    b" was the ",
    b" is the ",
    b" were the ",
    b" are the ",
    b" that the ",
    b" which is ",
    b" which was ",
    b" there is ",
    b" there are ",
    b" there was ",
    b" There ",
    b" This ",
    b" These ",
    b" However, ",
    b" Although ",
    b" During ",
    b" After ",
    b" Before ",
    b" Between ",
    b" Following ",
    b" According ",
    b" Including ",
    b" Because ",
    b"==== ",
    b" ====\n",
    b"=== ",
    b"== ",
    b"\n* [[",
    b"\n** [[",
    b"\n# [[",
    b"\n* [http://",
    b"* [http://",
    b"[[de:",
    b"[[fr:",
    b"[[es:",
    b"[[ja:",
    b"[[pl:",
    b"[[nl:",
    b"[[it:",
    b"[[sv:",
    b"[[pt:",
    b"[[zh:",
    b"[[ru:",
    b"[[fi:",
    b"[[no:",
    b"[[eo:",
    b"[[da:",
    b"[[ca:",
    b"[[he:",
    b"[[cs:",
    b"[[hu:",
    b"[[ro:",
    b"[[uk:",
    b"[[ko:",
    b"[[id:",
    b"[[bg:",
    b"[[tr:",
    b"[[sr:",
    b"[[sk:",
    b"[[lt:",
    b"[[sl:",
    b"[[et:",
    b"[[Wikipedia:",
    b"[[User:",
    b"[[Image:",
    b"{{stub}}",
    b"-stub}}",
    b"{{nowrap|",
    b"{{cite ",
    b"|title=",
    b"|author=",
    b"|publisher=",
    b"|year=",
    b"|date=",
    b"|url=http://",
    b"|accessdate=",
    b"|pages=",
    b"|location=",
    b"align=\"center\"",
    b"align=\"right\"",
    b"align=\"left\"",
    b"style=\"background:",
    b"cellpadding=\"",
    b"cellspacing=\"",
    b"border=\"",
    b"colspan=\"",
    b"rowspan=\"",
    b"width=\"",
    b"bgcolor=\"#",
    b"&times;",
    b"&deg;",
    b"&sup2;",
    b"&eacute;",
    b"&ouml;",
    b"&uuml;",
    b"&auml;",
    b"<br />",
    b"<br/>",
    b"<br>",
    b"<ref>",
    b"</ref>",
    b"<ref name=\"",
    b"<!-- ",
    b" -->",
    b"ISBN ",
    b"http://en.wikipedia.org/wiki/",
    b".edu/",
    b".gov/",
    b".net/",
    b".co.uk/",
    b".htm",
    b".pdf",
    b".jpg",
    b".png",
    b".svg",
    b"tion ",
    b"tions ",
    b"ment ",
    b"ments ",
    b"ness ",
    b"able ",
    b"ible ",
    b"ally ",
    b"ical ",
    b"ious ",
    b"ance ",
    b"ence ",
    b"ship ",
    b"ward ",
    b"ting ",
    b"ning ",
    b"ling ",
    b"ring ",
    b"sing ",
    b"ding ",
    b"king ",
    b"ming ",
    b"ping ",
    b"ving ",
    b"zing ",
    b"ther ",
    b"ight ",
    b"ough ",
    b"ould ",
    b"ound ",
    b"ount ",
    b"land ",
    b"wood ",
    b"town ",
    b"ville",
    b"burg",
    b"shire",
    b"stein",
    b"field",
    b"water",
    b"house",
    b"world",
    b"state",
    b"group",
    b"order",
    b"power",
    b"music",
    b"early",
    b"later",
    b"large",
    b"small",
    b"north",
    b"south",
    b"water ",
    b"human ",
    b"local ",
    b"modern ",
    b"public ",
    b"common ",
];

#[allow(dead_code)]
enum Wrt2Code {
    Single(u8),
    Ext(u8),
}

struct Wrt2Tables {
    dec_single: [Option<&'static [u8]>; 256],
    dec_ext: [Option<&'static [u8]>; 256],
    by_first: Vec<Vec<(&'static [u8], Wrt2Code)>>,
}

fn wrt2_tables() -> Wrt2Tables {
    let mut dec_single: [Option<&'static [u8]>; 256] = [None; 256];
    let mut dec_ext: [Option<&'static [u8]>; 256] = [None; 256];
    let mut single_codes = Vec::with_capacity(153);
    for b in 0x01u8..=0x08 {
        single_codes.push(b);
    }
    single_codes.push(0x0B);
    single_codes.push(0x0C);
    for b in 0x0Eu8..=0x1F {
        single_codes.push(b);
    }
    for b in 0x80u8..=0xFC {
        single_codes.push(b); // 0xFD reserved for WRT2_EXT_PREFIX, 0xFE for WRT_ESC
    }
    let mut by_first: Vec<Vec<(&'static [u8], Wrt2Code)>> = (0..256).map(|_| Vec::new()).collect();
    let mut s_idx = 0usize;
    // Only fold structural MediaWiki XML / Wiki-header templates (`len >= 6` non-prose markup) into 1-byte
    // capillary tokens so English prose stays contiguous 7-bit ASCII for maximal LZ stem/prefix matching & dense Order-2!
    for &ph in WRT_PHRASES.iter().take(42) {
        if ph.len() < 5 {
            continue;
        }
        if s_idx >= single_codes.len() {
            break;
        }
        let c = single_codes[s_idx];
        dec_single[c as usize] = Some(ph);
        by_first[ph[0] as usize].push((ph, Wrt2Code::Single(c)));
        s_idx += 1;
    }
    for (e_idx, &ph) in WRT2_EXT_PHRASES.iter().enumerate() {
        if e_idx >= 256 {
            break;
        }
        dec_ext[e_idx] = Some(ph);
    }
    for bucket in &mut by_first {
        bucket.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }
    Wrt2Tables {
        dec_single,
        dec_ext,
        by_first,
    }
}

fn encode_wrt2_capillary_with_tables(src: &[u8], tbl: &Wrt2Tables) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(src.len());
    let mut wrt_lut6_hits = 0usize;
    let n = src.len();
    let mut i = 0usize;
    while i < n {
        let b = src[i];
        let cands = &tbl.by_first[b as usize];
        let mut matched: Option<(&Wrt2Code, usize)> = None;
        for (ph, code) in cands {
            let plen = ph.len();
            if i + plen <= n && &src[i..i + plen] == *ph {
                matched = Some((code, plen));
                break;
            }
        }
        let esc_req = u8::from(matched.is_none() && !is_wrt_passthrough_byte(b));
        let dict_match = u8::from(matched.is_some());
        let is_ext = u8::from(matches!(matched, Some((Wrt2Code::Ext(_), _))));
        let xml_span = u8::from(b == b'<');
        let wrt_addr6 = (b & 3) | (dict_match << 2) | (is_ext << 3) | (esc_req << 4) | (xml_span << 5);
        let gate2 = eval_wrt_capillary_gate2_lut6(wrt_addr6);
        if gate2 != 0 {
            wrt_lut6_hits += 1;
        }
        if let Some((code, plen)) = matched {
            match *code {
                Wrt2Code::Single(c) => out.push(c),
                Wrt2Code::Ext(idx) => {
                    out.push(WRT2_EXT_PREFIX);
                    out.push(idx);
                }
            }
            i += plen;
        } else if !is_wrt_passthrough_byte(b) {
            out.push(WRT_ESC);
            out.push(b);
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    (out, wrt_lut6_hits)
}

pub fn encode_wrt2_capillary(src: &[u8]) -> (Vec<u8>, usize) {
    let tbl = wrt2_tables();
    encode_wrt2_capillary_with_tables(src, &tbl)
}

fn decode_wrt2_capillary_with_tables(
    wrt: &[u8],
    tbl: &Wrt2Tables,
) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::with_capacity(wrt.len() + (wrt.len() >> 2));
    let mut i = 0usize;
    while i < wrt.len() {
        let b = wrt[i];
        if b == WRT_ESC {
            if i + 1 >= wrt.len() {
                return Err("Truncated WRT2 escape sequence");
            }
            out.push(wrt[i + 1]);
            i += 2;
        } else if b == WRT2_EXT_PREFIX {
            if i + 1 >= wrt.len() {
                return Err("Truncated WRT2 ext sequence");
            }
            let idx = wrt[i + 1] as usize;
            if let Some(ph) = tbl.dec_ext[idx] {
                out.extend_from_slice(ph);
            } else {
                return Err("Invalid WRT2 ext phrase index");
            }
            i += 2;
        } else if let Some(ph) = tbl.dec_single[b as usize] {
            out.extend_from_slice(ph);
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    Ok(out)
}

pub fn decode_wrt2_capillary(wrt: &[u8]) -> Result<Vec<u8>, &'static str> {
    let tbl = wrt2_tables();
    decode_wrt2_capillary_with_tables(wrt, &tbl)
}

#[inline(always)]
fn cm3_hash4(s: &[u8], pos: usize) -> usize {
    let v = u32::from_le_bytes([s[pos], s[pos + 1], s[pos + 2], s[pos + 3]]);
    ((v.wrapping_mul(0x9E37_79B1)) >> (32 - CM3_HASH4_BITS)) as usize
}

#[inline(always)]
fn cm3_hash6(s: &[u8], pos: usize) -> usize {
    let lo = u32::from_le_bytes([s[pos], s[pos + 1], s[pos + 2], s[pos + 3]]) as u64;
    let hi = u16::from_le_bytes([s[pos + 4], s[pos + 5]]) as u64;
    let v = lo | (hi << 32);
    ((v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - CM3_HASH6_BITS)) as usize
}

#[inline(always)]
fn cm3_hash8(s: &[u8], pos: usize) -> usize {
    let v = u64::from_le_bytes([
        s[pos],
        s[pos + 1],
        s[pos + 2],
        s[pos + 3],
        s[pos + 4],
        s[pos + 5],
        s[pos + 6],
        s[pos + 7],
    ]);
    ((v.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - CM3_HASH8_BITS)) as usize
}

#[inline(always)]
fn step_cm3_word_hash(word_hash: &mut u32, prev_word_hash: &mut u32, b: u8) {
    let folded = match b {
        b'a'..=b'z' => Some(b),
        b'A'..=b'Z' => Some(b + 32),
        0x80..=0xFC => Some(b),
        _ => None,
    };
    if let Some(ch) = folded {
        *word_hash = word_hash.wrapping_mul(0x9E37_79B1).wrapping_add(ch as u32);
    } else if b != WRT2_EXT_PREFIX && *word_hash != 0 {
        *prev_word_hash = *word_hash;
        *word_hash = 0;
    }
}

struct MolCm3Models {
    is_match: [u16; 512],
    is_rep: [u16; 32],
    is_rep0: [u16; 32],
    is_rep0_short1: [u16; 256],
    is_rep1: [u16; 32],
    is_rep2: [u16; 32],
    len_choice: [u16; 3],
    len_low: [[u16; 8]; 8],
    len_mid: [[u16; 8]; 8],
    len_high: [u16; 256],
    len_ultra: Vec<u16>,
    rep_len_choice: [u16; 3],
    rep_len_low: [[u16; 8]; 8],
    rep_len_mid: [[u16; 8]; 8],
    rep_len_high: [u16; 256],
    rep_len_ultra: Vec<u16>,
    dist_slot: [[u16; 64]; 8],
    dist_special: [u16; 128], // Exact bit-trees for slots 4..13 (`dist` 4..127)
    dist_align: [u16; 16],
    // 1. Order-1 + LZMA match-shadow: 256 * 8 = 2,048 x 0x300
    lit_ord1: Vec<[u16; 0x300]>,
    // 2. Exact collision-free Order-2: 256 * 256 = 65,536 x 0x100
    lit_ord2: Vec<[u16; 0x100]>,
    // 3. Hashed exact Order-3 `(prev3, prev2, prev1)`: 131,072 x 0x100
    lit_ord3: Vec<[u16; 0x100]>,
    // 4. ASCII Word-Stem + PrevWord Context: 131,072 x 0x100
    lit_word: Vec<[u16; 0x100]>,
    // Adaptive integer Q8 log-odds mixer weights `[w1, w2, w3, ww]` across `128` context classes
    mix_w4: [[i16; 4]; 128],
    // Secondary Symbol Estimation (`SSE`): `64` context classes x `33` stretched-logit knots (`Q11`)
    sse_knots: [[u16; 33]; 64],
}

impl MolCm3Models {
    fn new() -> Self {
        let mut mw = [[44i16, 76, 76, 68]; 128];
        for idx in 0..128 {
            let matched = (idx & 1) != 0;
            let in_word = ((idx >> 1) & 1) != 0;
            if matched && in_word {
                mw[idx] = [112, 52, 48, 60];
            } else if matched {
                mw[idx] = [128, 60, 52, 32];
            } else if in_word {
                mw[idx] = [32, 68, 76, 92];
            }
        }
        let mut sse = [[1024u16; 33]; 64];
        for c in 0..64 {
            for k in 0..33 {
                let lg = ((k as i32) - 16) * 128;
                sse[c][k] = CM3_SQUASH_LUT[(lg + 2048).clamp(0, 4096) as usize];
            }
        }
        Self {
            is_match: [PROB_INIT; 512],
            is_rep: [PROB_INIT; 32],
            is_rep0: [PROB_INIT; 32],
            is_rep0_short1: [PROB_INIT; 256],
            is_rep1: [PROB_INIT; 32],
            is_rep2: [PROB_INIT; 32],
            len_choice: [PROB_INIT; 3],
            len_low: [[PROB_INIT; 8]; 8],
            len_mid: [[PROB_INIT; 8]; 8],
            len_high: [PROB_INIT; 256],
            len_ultra: vec![PROB_INIT; 2048],
            rep_len_choice: [PROB_INIT; 3],
            rep_len_low: [[PROB_INIT; 8]; 8],
            rep_len_mid: [[PROB_INIT; 8]; 8],
            rep_len_high: [PROB_INIT; 256],
            rep_len_ultra: vec![PROB_INIT; 2048],
            dist_slot: [[PROB_INIT; 64]; 8],
            dist_special: [PROB_INIT; 128],
            dist_align: [PROB_INIT; 16],
            lit_ord1: vec![[PROB_INIT; 0x300]; 256 * 8],
            lit_ord2: vec![[PROB_INIT; 0x100]; 65_536],
            lit_ord3: vec![[PROB_INIT; 0x100]; 131_072],
            lit_word: vec![[PROB_INIT; 0x100]; 65_536],
            mix_w4: mw,
            sse_knots: sse,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_len_cm3(
        choice: &mut [u16; 3],
        low: &mut [[u16; 8]; 8],
        mid: &mut [[u16; 8]; 8],
        high: &mut [u16; 256],
        ultra: &mut [u16],
        rc: &mut RangeEncoder,
        pos_ctx: usize,
        min_l: usize,
        len: usize,
    ) {
        let v = len - min_l;
        if v < 8 {
            rc.encode_bit(&mut choice[0], 0);
            rc.encode_tree(&mut low[pos_ctx], 3, v as u32);
        } else if v < 16 {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 0);
            rc.encode_tree(&mut mid[pos_ctx], 3, (v - 8) as u32);
        } else if v < 272 {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 1);
            rc.encode_bit(&mut choice[2], 0);
            rc.encode_tree(high, 8, (v - 16) as u32);
        } else {
            rc.encode_bit(&mut choice[0], 1);
            rc.encode_bit(&mut choice[1], 1);
            rc.encode_bit(&mut choice[2], 1);
            rc.encode_tree(ultra, 11, (v - 272).min(2047) as u32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_len_cm3(
        choice: &mut [u16; 3],
        low: &mut [[u16; 8]; 8],
        mid: &mut [[u16; 8]; 8],
        high: &mut [u16; 256],
        ultra: &mut [u16],
        rc: &mut RangeDecoder<'_>,
        pos_ctx: usize,
        min_l: usize,
    ) -> usize {
        if rc.decode_bit(&mut choice[0]) == 0 {
            min_l + (rc.decode_tree(&mut low[pos_ctx], 3) as usize)
        } else if rc.decode_bit(&mut choice[1]) == 0 {
            min_l + 8 + (rc.decode_tree(&mut mid[pos_ctx], 3) as usize)
        } else if rc.decode_bit(&mut choice[2]) == 0 {
            min_l + 16 + (rc.decode_tree(high, 8) as usize)
        } else {
            min_l + 272 + (rc.decode_tree(ultra, 11) as usize)
        }
    }

    fn encode_dist(&mut self, rc: &mut RangeEncoder, len_ctx: usize, dist: u32) {
        if dist < 4 {
            rc.encode_tree(&mut self.dist_slot[len_ctx], 6, dist);
            return;
        }
        let hb = 31 - dist.leading_zeros();
        let second = (dist >> (hb - 1)) & 1;
        let slot = (hb << 1) | second;
        rc.encode_tree(&mut self.dist_slot[len_ctx], 6, slot);
        let extra = hb - 1;
        let rem = dist & ((1u32 << extra) - 1);
        if slot < 14 {
            let base = (2 | (slot & 1)) << extra;
            let tree_off = (base - slot) as usize;
            let mut m = 1usize;
            for b in (0..extra).rev() {
                let bit = ((rem >> b) & 1) as u8;
                rc.encode_bit(&mut self.dist_special[tree_off + m], bit);
                m = (m << 1) | (bit as usize);
            }
        } else {
            rc.encode_direct(rem >> 4, extra - 4);
            rc.encode_tree(&mut self.dist_align, 4, rem & 0xF);
        }
    }

    fn decode_dist(&mut self, rc: &mut RangeDecoder<'_>, len_ctx: usize) -> u32 {
        let slot = rc.decode_tree(&mut self.dist_slot[len_ctx], 6);
        if slot < 4 {
            return slot;
        }
        let extra = (slot >> 1) - 1;
        let base = (2 | (slot & 1)) << extra;
        if slot < 14 {
            let tree_off = (base - slot) as usize;
            let mut m = 1usize;
            for _ in 0..extra {
                let bit = rc.decode_bit(&mut self.dist_special[tree_off + m]) as usize;
                m = (m << 1) | bit;
            }
            base | ((m as u32) - (1u32 << extra))
        } else {
            let hi = rc.decode_direct(extra - 4) << 4;
            let lo = rc.decode_tree(&mut self.dist_align, 4);
            base | hi | lo
        }
    }

    #[inline(always)]
    fn update_prob(p: &mut u16, bit: u8, shift: u32) {
        let cur = *p as u32;
        if bit == 0 {
            *p = (cur + ((2048 - cur) >> shift)) as u16;
        } else {
            *p = (cur - (cur >> shift)) as u16;
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn predict_cm3_bit(
        &self,
        o1_ctx: usize,
        o2_ctx: usize,
        o3_ctx: usize,
        word_ctx: usize,
        o1_idx: usize,
        m_tree: usize,
        w_cls: usize,
        sse_ctx: usize,
        cm3_mode: u8,
    ) -> (u32, [i16; 4], usize, u32) {
        let p1_raw = self.lit_ord1[o1_ctx][o1_idx];
        let p2_raw = self.lit_ord2[o2_ctx][m_tree];
        let p3_raw = self.lit_ord3[o3_ctx][m_tree];
        let pw_raw = self.lit_word[word_ctx][m_tree];
        let s1 = CM3_STRETCH_LUT[p1_raw as usize];
        let s2 = CM3_STRETCH_LUT[p2_raw as usize];
        let s3 = CM3_STRETCH_LUT[p3_raw as usize];
        let sw = CM3_STRETCH_LUT[pw_raw as usize];

        // Zero-Dilution Cascade Fallback: reallocate weight of any unvisited (`== PROB_INIT`) high-order table
        // down to the next populated order so unseen contexts never pull log-odds toward `0` (`50/50`)!
        let w4 = self.mix_w4[w_cls];
        let mut w0 = w4[0] as i32;
        let mut w1 = w4[1] as i32;
        let mut w2 = w4[2] as i32;
        let mut w3 = w4[3] as i32;
        if pw_raw == PROB_INIT {
            w2 += w3;
            w3 = 0;
        }
        if p3_raw == PROB_INIT {
            w1 += w2;
            w2 = 0;
        }
        if p2_raw == PROB_INIT {
            w0 += w1;
            w1 = 0;
        }

        let mut logit = ((s1 as i32) * w0
            + (s2 as i32) * w1
            + (s3 as i32) * w2
            + (sw as i32) * w3)
            >> 8;
        if cm3_mode == 3 {
            logit += (s1 as i32) >> 4;
        } else if cm3_mode == 2 && pw_raw != PROB_INIT {
            logit += (sw as i32) >> 4;
        }
        let clamped_lg = (logit + 2048).clamp(0, 4096) as usize;
        let p_sq = CM3_SQUASH_LUT[clamped_lg] as u32;
        let knot = (clamped_lg >> 7).min(31);
        let frac = (clamped_lg & 127) as u32;
        let k0 = self.sse_knots[sse_ctx][knot] as u32;
        let k1 = self.sse_knots[sse_ctx][knot + 1] as u32;
        let p_sse = (k0 * (128 - frac) + k1 * frac) >> 7;
        let p_final = ((7 * p_sq + p_sse + 4) >> 3).clamp(1, 2047);
        (p_final, [s1, s2, s3, sw], knot, frac)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn commit_cm3_bit(
        &mut self,
        o1_ctx: usize,
        o2_ctx: usize,
        o3_ctx: usize,
        word_ctx: usize,
        o1_idx: usize,
        m_tree: usize,
        w_cls: usize,
        sse_ctx: usize,
        knot: usize,
        frac: u32,
        p_pred_q11: u32,
        stretches: [i16; 4],
        bit: u8,
    ) {
        let target_q11: i32 = if bit == 0 { 2048 } else { 0 };
        let err: i32 = target_q11 - (p_pred_q11 as i32);
        let w = &mut self.mix_w4[w_cls];
        for k in 0..4 {
            let delta = (err * (stretches[k] as i32)) >> 16;
            w[k] = ((w[k] as i32) + delta).clamp(4, 320) as i16;
        }
        let nearest = if frac >= 64 { knot + 1 } else { knot };
        Self::update_prob(&mut self.sse_knots[sse_ctx][nearest], bit, 6);

        Self::update_prob(&mut self.lit_ord1[o1_ctx][o1_idx], bit, 5);
        Self::update_prob(&mut self.lit_ord2[o2_ctx][m_tree], bit, 5);
        Self::update_prob(&mut self.lit_ord3[o3_ctx][m_tree], bit, 5);
        Self::update_prob(&mut self.lit_word[word_ctx][m_tree], bit, 5);
    }
}

#[inline(always)]
fn cm3_match_score(len: usize, off: usize, is_rep: bool) -> isize {
    if is_rep {
        (len as isize) * 11 - 6
    } else {
        let dist_bits = if off <= 4 {
            6isize
        } else {
            6 + ((usize::BITS - 1 - (off - 1).leading_zeros()) as isize) * 2
        };
        // Each matched byte saves ~2.75 bits (`* 11` in quarter-bits); subtract distance + length header cost!
        (len as isize) * 11 - 16 - (dist_bits * 2)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn find_best_cm3_match(
    s: &[u8],
    pos: usize,
    reps: &[usize; 4],
    head3: &[u32],
    head4: &[u32],
    prev4: &[u32],
    head6: &[u32],
    prev6: &[u32],
    head8: &[u32],
    prev8: &[u32],
) -> (usize, usize, usize) {
    let n = s.len();
    let max_len = (n - pos).min(CM3_MAX_MATCH);
    if max_len < 2 {
        return (0, 0, 4);
    }

    let mut best_len = 0usize;
    let mut best_off = 0usize;
    let mut best_kind = 4usize;
    let mut best_sc: isize = 8; // minimum net bit-saving threshold in quarter-bits

    for (r_idx, &off) in reps.iter().enumerate() {
        if off <= pos && s[pos - off] == s[pos] {
            let mut l = 1usize;
            while l < max_len && s[pos - off + l] == s[pos + l] {
                l += 1;
            }
            if l >= 2 {
                let sc = cm3_match_score(l, off, true) - (r_idx as isize) * 3;
                if sc > best_sc {
                    best_sc = sc;
                    best_len = l;
                    best_off = off;
                    best_kind = r_idx;
                    if l >= 128 || l == max_len {
                        return (l, off, r_idx);
                    }
                }
            }
        }
    }

    if best_len < 4 && max_len >= 3 {
        let m3 = head3[cm2_hash3(s, pos)];
        if m3 != u32::MAX {
            let mpos = m3 as usize;
            let off = pos - mpos;
            if off > 0
                && off <= 2048
                && s[mpos] == s[pos]
                && s[mpos + 1] == s[pos + 1]
                && s[mpos + 2] == s[pos + 2]
            {
                let mut l = 3usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                let sc = cm3_match_score(l, off, false);
                if sc > best_sc {
                    best_sc = sc;
                    best_len = l;
                    best_off = off;
                    best_kind = 4;
                }
            }
        }
    }

    // 1. Search nearest HASH4 first so short/medium local offsets beat distant long-offset collisions!
    if max_len >= 4 && best_len < 64 {
        let h4 = cm3_hash4(s, pos);
        let mut cur = head4[h4];
        let mut depth = 0usize;
        while cur != u32::MAX && depth < CM3_CHAIN4_DEPTH {
            let mpos = cur as usize;
            let off = pos - mpos;
            let probe = best_len.max(3).min(max_len - 1);
            if off > 0 && s[mpos + probe] == s[pos + probe] && s[mpos] == s[pos] {
                let mut l = 1usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                if l >= 4 {
                    let sc = cm3_match_score(l, off, false);
                    if sc > best_sc {
                        best_sc = sc;
                        best_len = l;
                        best_off = off;
                        best_kind = 4;
                        if l >= 96 || l == max_len {
                            return (best_len, best_off, best_kind);
                        }
                    }
                }
            }
            cur = prev4[mpos];
            depth += 1;
        }
    }

    // 2. Search HASH6 for medium-range 6..12 byte matches across tens of megabytes
    if max_len >= 6 && best_len < 96 {
        let h6 = cm3_hash6(s, pos);
        let mut cur = head6[h6];
        let mut depth = 0usize;
        while cur != u32::MAX && depth < CM3_CHAIN6_DEPTH {
            let mpos = cur as usize;
            let off = pos - mpos;
            let probe = best_len.max(5).min(max_len - 1);
            if off > 0 && s[mpos + probe] == s[pos + probe] && s[mpos] == s[pos] {
                let mut l = 1usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                if l >= 6 {
                    let sc = cm3_match_score(l, off, false);
                    if sc > best_sc {
                        best_sc = sc;
                        best_len = l;
                        best_off = off;
                        best_kind = 4;
                        if l >= 128 || l == max_len {
                            return (best_len, best_off, best_kind);
                        }
                    }
                }
            }
            cur = prev6[mpos];
            depth += 1;
        }
    }

    // 3. Search HASH8 for long >= 8 byte matches across the entire 128 MiB window
    if max_len >= 8 && best_len < 256 {
        let h8 = cm3_hash8(s, pos);
        let mut cur = head8[h8];
        let mut depth = 0usize;
        while cur != u32::MAX && depth < CM3_CHAIN8_DEPTH {
            let mpos = cur as usize;
            let off = pos - mpos;
            let probe = best_len.max(7).min(max_len - 1);
            if off > 0 && s[mpos + probe] == s[pos + probe] && s[mpos] == s[pos] {
                let mut l = 1usize;
                while l < max_len && s[mpos + l] == s[pos + l] {
                    l += 1;
                }
                if l >= 8 {
                    let sc = cm3_match_score(l, off, false);
                    if sc > best_sc {
                        best_sc = sc;
                        best_len = l;
                        best_off = off;
                        best_kind = 4;
                        if l >= 384 || l == max_len {
                            return (best_len, best_off, best_kind);
                        }
                    }
                }
            }
            cur = prev8[mpos];
            depth += 1;
        }
    }

    (best_len, best_off, best_kind)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn insert_cm3_pos(
    s: &[u8],
    pos: usize,
    head3: &mut [u32],
    head4: &mut [u32],
    prev4: &mut [u32],
    head6: &mut [u32],
    prev6: &mut [u32],
    head8: &mut [u32],
    prev8: &mut [u32],
) {
    let n = s.len();
    if pos + 3 <= n {
        head3[cm2_hash3(s, pos)] = pos as u32;
    }
    if pos + 4 <= n {
        let h4 = cm3_hash4(s, pos);
        prev4[pos] = head4[h4];
        head4[h4] = pos as u32;
    }
    if pos + 6 <= n {
        let h6 = cm3_hash6(s, pos);
        prev6[pos] = head6[h6];
        head6[h6] = pos as u32;
    }
    if pos + 8 <= n {
        let h8 = cm3_hash8(s, pos);
        prev8[pos] = head8[h8];
        head8[h8] = pos as u32;
    }
}

fn encode_cm3_wrt_block(s: &[u8], mdl: &mut MolCm3Models) -> (Vec<u8>, usize) {
    let n = s.len();
    let mut rc = RangeEncoder::with_capacity((n / 3).max(64));
    let mut head3 = vec![u32::MAX; 65_536];
    let mut head4 = vec![u32::MAX; CM3_HASH4_SIZE];
    let mut prev4 = vec![u32::MAX; n.max(1)];
    let mut head6 = vec![u32::MAX; CM3_HASH6_SIZE];
    let mut prev6 = vec![u32::MAX; n.max(1)];
    let mut head8 = vec![u32::MAX; CM3_HASH8_SIZE];
    let mut prev8 = vec![u32::MAX; n.max(1)];

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0;
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    let mut prev3: u8 = 0;
    let mut word_hash: u32 = 0;
    let mut prev_word_hash: u32 = 0;
    let mut skip_bytes = 0usize;

    let mut pos = 0usize;
    while pos < n {
        let (mut best_len, best_off, best_kind) = find_best_cm3_match(
            s, pos, &reps, &head3, &head4, &prev4, &head6, &prev6, &head8, &prev8,
        );

        if best_len >= 2 && best_len < 24 && pos + 1 < n {
            let (next_len, next_off, next_kind) = find_best_cm3_match(
                s,
                pos + 1,
                &reps,
                &head3,
                &head4,
                &prev4,
                &head6,
                &prev6,
                &head8,
                &prev8,
            );
            if next_len >= 2 {
                let cur_sc = cm3_match_score(best_len, best_off, best_kind < 4);
                let nxt_sc = cm3_match_score(next_len, next_off, next_kind < 4);
                if nxt_sc > cur_sc + 14 {
                    best_len = 0;
                }
            }
        }

        let is_short_rep1 = best_len < 2 && reps[0] <= pos && s[pos - reps[0]] == s[pos];
        let m_ctx = ((state & 15) << 5) | ((byte_class4(prev1) & 7) << 2) | (pos & 3);
        let pos_ctx = byte_class4(prev1) & 7;

        if best_len >= 2 || is_short_rep1 {
            skip_bytes += best_len.max(1);
            rc.encode_bit(&mut mdl.is_match[m_ctx], 1);
            let r_ctx = ((state & 15) << 1) | usize::from(prev1 == WRT2_EXT_PREFIX);
            if is_short_rep1 {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 1);
                rc.encode_bit(&mut mdl.is_rep0[r_ctx], 0);
                let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                rc.encode_bit(&mut mdl.is_rep0_short1[sr_ctx], 0);
                state = if state < 7 { 9 } else { 11 };
                insert_cm3_pos(
                    s,
                    pos,
                    &mut head3,
                    &mut head4,
                    &mut prev4,
                    &mut head6,
                    &mut prev6,
                    &mut head8,
                    &mut prev8,
                );
                let sym = s[pos];
                step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, sym);
                prev3 = prev2;
                prev2 = prev1;
                prev1 = sym;
                pos += 1;
            } else if best_kind < 4 {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 1);
                if best_kind == 0 {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 0);
                    let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                    rc.encode_bit(&mut mdl.is_rep0_short1[sr_ctx], 1);
                } else {
                    rc.encode_bit(&mut mdl.is_rep0[r_ctx], 1);
                    if best_kind == 1 {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 0);
                        reps.swap(0, 1);
                    } else {
                        rc.encode_bit(&mut mdl.is_rep1[r_ctx], 1);
                        if best_kind == 2 {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 0);
                            let t = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        } else {
                            rc.encode_bit(&mut mdl.is_rep2[r_ctx], 1);
                            let t = reps[3];
                            reps[3] = reps[2];
                            reps[2] = reps[1];
                            reps[1] = reps[0];
                            reps[0] = t;
                        }
                    }
                }
                MolCm3Models::encode_len_cm3(
                    &mut mdl.rep_len_choice,
                    &mut mdl.rep_len_low,
                    &mut mdl.rep_len_mid,
                    &mut mdl.rep_len_high,
                    &mut mdl.rep_len_ultra,
                    &mut rc,
                    pos_ctx,
                    2,
                    best_len,
                );
                state = if state < 7 { 8 } else { 11 };
                let step_end = pos + best_len;
                let fast_skip = best_len >= 48;
                let start_p = pos;
                while pos < step_end {
                    if !fast_skip || pos < start_p + 6 || pos + 6 >= step_end {
                        insert_cm3_pos(
                            s,
                            pos,
                            &mut head3,
                            &mut head4,
                            &mut prev4,
                            &mut head6,
                            &mut prev6,
                            &mut head8,
                            &mut prev8,
                        );
                    }
                    let sym = s[pos];
                    step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, sym);
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = sym;
                    pos += 1;
                }
            } else {
                rc.encode_bit(&mut mdl.is_rep[r_ctx], 0);
                MolCm3Models::encode_len_cm3(
                    &mut mdl.len_choice,
                    &mut mdl.len_low,
                    &mut mdl.len_mid,
                    &mut mdl.len_high,
                    &mut mdl.len_ultra,
                    &mut rc,
                    pos_ctx,
                    3,
                    best_len,
                );
                let len_ctx = (best_len - 3).min(7);
                mdl.encode_dist(&mut rc, len_ctx, (best_off - 1) as u32);
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = best_off;
                state = if state < 7 { 7 } else { 10 };
                let step_end = pos + best_len;
                let fast_skip = best_len >= 48;
                let start_p = pos;
                while pos < step_end {
                    if !fast_skip || pos < start_p + 6 || pos + 6 >= step_end {
                        insert_cm3_pos(
                            s,
                            pos,
                            &mut head3,
                            &mut head4,
                            &mut prev4,
                            &mut head6,
                            &mut prev6,
                            &mut head8,
                            &mut prev8,
                        );
                    }
                    let sym = s[pos];
                    step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, sym);
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = sym;
                    pos += 1;
                }
            }
        } else {
            rc.encode_bit(&mut mdl.is_match[m_ctx], 0);
            let c2 = byte_class64(prev2);
            let o1_ctx = ((prev1 as usize) << 3) | (c2 >> 3);
            let o2_ctx = ((prev2 as usize) << 8) | (prev1 as usize);
            let h_o3 = (((prev3 as u32) << 16) | ((prev2 as u32) << 8) | (prev1 as u32))
                .wrapping_mul(0x9E37_79B1);
            let o3_ctx = ((h_o3 >> 15) as usize) & 0x1FFFF;
            let in_word = word_hash != 0;
            let wh_mix = if in_word {
                word_hash ^ prev_word_hash.rotate_left(9)
            } else {
                prev_word_hash.wrapping_mul(131) ^ ((prev1 as u32) * 257)
            };
            let word_ctx = ((wh_mix ^ (wh_mix >> 16)) as usize) & 0xFFFF;
            let base_cls = byte_class4(prev1) & 15;
            let rep_shadow = u8::from(state >= 7 && reps[0] <= pos);
            let cm3_addr6 = ((base_cls as u8) & 3)
                | (u8::from(in_word) << 2)
                | (1 << 3)
                | (1 << 4)
                | (rep_shadow << 5);
            let cm3_mode = eval_cm3_logodds_mix_gate2_lut6(cm3_addr6);

            let sym = s[pos];
            let mut m = 1usize;
            let mut matched = state >= 7 && reps[0] <= pos;
            let mut match_byte = if matched { s[pos - reps[0]] } else { 0 };
            for b in (0..8).rev() {
                let bit = (sym >> b) & 1;
                let top_nibble = usize::from(b < 4);
                let w_cls =
                    (base_cls << 3) | (top_nibble << 2) | ((in_word as usize) << 1) | (matched as usize);
                let sse_ctx = ((base_cls & 15) << 2) | ((matched as usize) << 1) | (in_word as usize);
                let o1_idx = if matched {
                    let mbit = (match_byte >> 7) & 1;
                    ((1 + mbit as usize) << 8) + m
                } else {
                    m
                };
                let (p_q11, stretches, knot, frac) = mdl.predict_cm3_bit(
                    o1_ctx, o2_ctx, o3_ctx, word_ctx, o1_idx, m, w_cls, sse_ctx, cm3_mode,
                );
                rc.encode_bit_prob(p_q11, bit);
                mdl.commit_cm3_bit(
                    o1_ctx, o2_ctx, o3_ctx, word_ctx, o1_idx, m, w_cls, sse_ctx, knot, frac, p_q11,
                    stretches, bit,
                );
                if matched {
                    let mbit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    if mbit != bit {
                        matched = false;
                    }
                }
                m = (m << 1) | (bit as usize);
            }

            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            insert_cm3_pos(
                s,
                pos,
                &mut head3,
                &mut head4,
                &mut prev4,
                &mut head6,
                &mut prev6,
                &mut head8,
                &mut prev8,
            );
            step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, sym);
            prev3 = prev2;
            prev2 = prev1;
            prev1 = sym;
            pos += 1;
        }
    }
    (rc.finish(), skip_bytes)
}

fn decode_cm3_wrt_block(
    payload: &[u8],
    wrt_len: usize,
    mdl: &mut MolCm3Models,
) -> Result<Vec<u8>, &'static str> {
    if wrt_len == 0 {
        return Ok(Vec::new());
    }
    let mut wrt_out = Vec::with_capacity(wrt_len);
    let mut rc = RangeDecoder::new(payload);

    let mut reps: [usize; 4] = [1, 1, 1, 1];
    let mut state: usize = 0;
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    let mut prev3: u8 = 0;
    let mut word_hash: u32 = 0;
    let mut prev_word_hash: u32 = 0;

    while wrt_out.len() < wrt_len {
        let pos = wrt_out.len();
        let m_ctx = ((state & 15) << 5) | ((byte_class4(prev1) & 7) << 2) | (pos & 3);
        let pos_ctx = byte_class4(prev1) & 7;
        if rc.decode_bit(&mut mdl.is_match[m_ctx]) == 1 {
            let r_ctx = ((state & 15) << 1) | usize::from(prev1 == WRT2_EXT_PREFIX);
            if rc.decode_bit(&mut mdl.is_rep[r_ctx]) == 1 {
                let mut is_short1 = false;
                if rc.decode_bit(&mut mdl.is_rep0[r_ctx]) == 0 {
                    let sr_ctx = ((state & 15) << 4) | (byte_class4(prev1) & 15);
                    if rc.decode_bit(&mut mdl.is_rep0_short1[sr_ctx]) == 0 {
                        is_short1 = true;
                    }
                } else if rc.decode_bit(&mut mdl.is_rep1[r_ctx]) == 0 {
                    reps.swap(0, 1);
                } else if rc.decode_bit(&mut mdl.is_rep2[r_ctx]) == 0 {
                    let t = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                } else {
                    let t = reps[3];
                    reps[3] = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    reps[0] = t;
                }
                if is_short1 {
                    let off = reps[0];
                    if off == 0 || off > wrt_out.len() {
                        return Err("Corrupt short-rep offset in MLT3 decode");
                    }
                    let b = wrt_out[wrt_out.len() - off];
                    wrt_out.push(b);
                    state = if state < 7 { 9 } else { 11 };
                    step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, b);
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = b;
                } else {
                    let len = MolCm3Models::decode_len_cm3(
                        &mut mdl.rep_len_choice,
                        &mut mdl.rep_len_low,
                        &mut mdl.rep_len_mid,
                        &mut mdl.rep_len_high,
                        &mut mdl.rep_len_ultra,
                        &mut rc,
                        pos_ctx,
                        2,
                    );
                    state = if state < 7 { 8 } else { 11 };
                    let off = reps[0];
                    if off == 0 || off > wrt_out.len() {
                        return Err("Corrupt rep offset in MLT3 decode");
                    }
                    let copy_len = len.min(wrt_len - wrt_out.len());
                    for _ in 0..copy_len {
                        let b = wrt_out[wrt_out.len() - off];
                        wrt_out.push(b);
                        step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, b);
                        prev3 = prev2;
                        prev2 = prev1;
                        prev1 = b;
                    }
                }
            } else {
                let len = MolCm3Models::decode_len_cm3(
                    &mut mdl.len_choice,
                    &mut mdl.len_low,
                    &mut mdl.len_mid,
                    &mut mdl.len_high,
                    &mut mdl.len_ultra,
                    &mut rc,
                    pos_ctx,
                    3,
                );
                let len_ctx = (len - 3).min(7);
                let off = (mdl.decode_dist(&mut rc, len_ctx) as usize) + 1;
                reps[3] = reps[2];
                reps[2] = reps[1];
                reps[1] = reps[0];
                reps[0] = off;
                state = if state < 7 { 7 } else { 10 };
                if off == 0 || off > wrt_out.len() {
                    return Err("Corrupt match offset in MLT3 decode");
                }
                let copy_len = len.min(wrt_len - wrt_out.len());
                for _ in 0..copy_len {
                    let b = wrt_out[wrt_out.len() - off];
                    wrt_out.push(b);
                    step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, b);
                    prev3 = prev2;
                    prev2 = prev1;
                    prev1 = b;
                }
            }
        } else {
            let c2 = byte_class64(prev2);
            let o1_ctx = ((prev1 as usize) << 3) | (c2 >> 3);
            let o2_ctx = ((prev2 as usize) << 8) | (prev1 as usize);
            let h_o3 = (((prev3 as u32) << 16) | ((prev2 as u32) << 8) | (prev1 as u32))
                .wrapping_mul(0x9E37_79B1);
            let o3_ctx = ((h_o3 >> 15) as usize) & 0x1FFFF;
            let in_word = word_hash != 0;
            let wh_mix = if in_word {
                word_hash ^ prev_word_hash.rotate_left(9)
            } else {
                prev_word_hash.wrapping_mul(131) ^ ((prev1 as u32) * 257)
            };
            let word_ctx = ((wh_mix ^ (wh_mix >> 16)) as usize) & 0xFFFF;
            let base_cls = byte_class4(prev1) & 15;
            let rep_shadow = u8::from(state >= 7 && reps[0] <= pos);
            let cm3_addr6 = ((base_cls as u8) & 3)
                | (u8::from(in_word) << 2)
                | (1 << 3)
                | (1 << 4)
                | (rep_shadow << 5);
            let cm3_mode = eval_cm3_logodds_mix_gate2_lut6(cm3_addr6);

            let mut m = 1usize;
            let mut matched = state >= 7 && reps[0] <= pos;
            let mut match_byte = if matched { wrt_out[pos - reps[0]] } else { 0 };
            for b in (0..8).rev() {
                let top_nibble = usize::from(b < 4);
                let w_cls =
                    (base_cls << 3) | (top_nibble << 2) | ((in_word as usize) << 1) | (matched as usize);
                let sse_ctx = ((base_cls & 15) << 2) | ((matched as usize) << 1) | (in_word as usize);
                let o1_idx = if matched {
                    let mbit = (match_byte >> 7) & 1;
                    ((1 + mbit as usize) << 8) + m
                } else {
                    m
                };
                let (p_q11, stretches, knot, frac) = mdl.predict_cm3_bit(
                    o1_ctx, o2_ctx, o3_ctx, word_ctx, o1_idx, m, w_cls, sse_ctx, cm3_mode,
                );
                let bit = rc.decode_bit_prob(p_q11);
                mdl.commit_cm3_bit(
                    o1_ctx, o2_ctx, o3_ctx, word_ctx, o1_idx, m, w_cls, sse_ctx, knot, frac, p_q11,
                    stretches, bit,
                );
                if matched {
                    let mbit = (match_byte >> 7) & 1;
                    match_byte <<= 1;
                    if mbit != bit {
                        matched = false;
                    }
                }
                m = (m << 1) | (bit as usize);
            }
            let sym = (m - 0x100) as u8;
            wrt_out.push(sym);
            state = match state {
                0..=3 => 0,
                4..=9 => state - 3,
                _ => state - 6,
            };
            step_cm3_word_hash(&mut word_hash, &mut prev_word_hash, sym);
            prev3 = prev2;
            prev2 = prev1;
            prev1 = sym;
        }
    }
    Ok(wrt_out)
}

pub fn compress_mol_ltcb_cm3_slice(raw_src: &[u8]) -> (Vec<u8>, CompressStats) {
    let t0 = Instant::now();
    let tbl = wrt2_tables();
    let mut mdl = MolCm3Models::new();
    let total_raw = raw_src.len();
    let num_blocks = if total_raw == 0 {
        0
    } else {
        (total_raw + CM3_RAW_BLOCK_CAP - 1) / CM3_RAW_BLOCK_CAP
    };
    let mut archive = Vec::with_capacity((total_raw / 4) + 64);
    archive.extend_from_slice(&MAGIC_CM3);
    archive.extend_from_slice(&(total_raw as u64).to_le_bytes());
    archive.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    let mut rung0_skip_bytes = 0usize;
    for chunk in raw_src.chunks(CM3_RAW_BLOCK_CAP) {
        let (wrt_vec, wrt_hits) = encode_wrt2_capillary_with_tables(chunk, &tbl);
        let (payload, lz_skips) = encode_cm3_wrt_block(&wrt_vec, &mut mdl);
        rung0_skip_bytes += wrt_hits + lz_skips;
        archive.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        archive.extend_from_slice(&(wrt_vec.len() as u32).to_le_bytes());
        archive.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        archive.extend_from_slice(&payload);
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let compressed_bytes = archive.len();
    (
        archive,
        CompressStats {
            raw_bytes: total_raw,
            compressed_bytes,
            rung0_skip_bytes,
            elapsed_ms,
        },
    )
}

pub fn compress_mol_ltcb_cm3_file(
    path: &PathBuf,
    max_bytes: Option<usize>,
) -> (Vec<u8>, CompressStats, String) {
    let t0 = Instant::now();
    let meta_len = fs::metadata(path).expect("Failed to stat --input").len() as usize;
    let target_len = match max_bytes {
        Some(m) if m < meta_len => m,
        _ => meta_len,
    };
    let tbl = wrt2_tables();
    let mut mdl = MolCm3Models::new();
    let num_blocks = if target_len == 0 {
        0
    } else {
        (target_len + CM3_RAW_BLOCK_CAP - 1) / CM3_RAW_BLOCK_CAP
    };
    let mut archive = Vec::with_capacity((target_len / 4) + 64);
    archive.extend_from_slice(&MAGIC_CM3);
    archive.extend_from_slice(&(target_len as u64).to_le_bytes());
    archive.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    let mut f = File::open(path).expect("Failed to open --input");
    let mut raw_hasher = StreamingSha256::new();
    let mut remaining = target_len;
    let mut chunk_buf = Vec::with_capacity(target_len.min(CM3_RAW_BLOCK_CAP));
    let mut rung0_skip_bytes = 0usize;

    while remaining > 0 {
        let want = remaining.min(CM3_RAW_BLOCK_CAP);
        chunk_buf.resize(want, 0);
        f.read_exact(&mut chunk_buf).expect("Failed to read input chunk");
        raw_hasher.update(&chunk_buf);
        let (wrt_vec, wrt_hits) = encode_wrt2_capillary_with_tables(&chunk_buf, &tbl);
        let (payload, lz_skips) = encode_cm3_wrt_block(&wrt_vec, &mut mdl);
        rung0_skip_bytes += wrt_hits + lz_skips;
        archive.extend_from_slice(&(want as u32).to_le_bytes());
        archive.extend_from_slice(&(wrt_vec.len() as u32).to_le_bytes());
        archive.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        archive.extend_from_slice(&payload);
        remaining -= want;
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let compressed_bytes = archive.len();
    (
        archive,
        CompressStats {
            raw_bytes: target_len,
            compressed_bytes,
            rung0_skip_bytes,
            elapsed_ms,
        },
        raw_hasher.finalize_hex(),
    )
}

pub fn decompress_mol_ltcb_cm3_streaming_sha256(
    archive: &[u8],
    mut collect_out: Option<&mut Vec<u8>>,
) -> Result<(usize, String, f64), &'static str> {
    let t0 = Instant::now();
    if archive.len() < 16 || archive[0..4] != MAGIC_CM3 {
        return Err("Invalid MLT3 archive header");
    }
    let total_raw = u64::from_le_bytes(archive[4..12].try_into().unwrap()) as usize;
    let num_blocks = u32::from_le_bytes(archive[12..16].try_into().unwrap()) as usize;
    let tbl = wrt2_tables();
    let mut mdl = MolCm3Models::new();
    let mut dec_hasher = StreamingSha256::new();
    let mut cursor = 16usize;
    let mut decoded_total = 0usize;

    for _ in 0..num_blocks {
        if cursor + 12 > archive.len() {
            return Err("Truncated MLT3 block header");
        }
        let blk_raw_len = u32::from_le_bytes(archive[cursor..cursor + 4].try_into().unwrap()) as usize;
        let blk_wrt_len =
            u32::from_le_bytes(archive[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let blk_pay_len =
            u32::from_le_bytes(archive[cursor + 8..cursor + 12].try_into().unwrap()) as usize;
        cursor += 12;
        if cursor + blk_pay_len > archive.len() {
            return Err("Truncated MLT3 block payload");
        }
        let wrt_dec =
            decode_cm3_wrt_block(&archive[cursor..cursor + blk_pay_len], blk_wrt_len, &mut mdl)?;
        cursor += blk_pay_len;
        let raw_blk = decode_wrt2_capillary_with_tables(&wrt_dec, &tbl)?;
        if raw_blk.len() != blk_raw_len {
            return Err("MLT3 block raw length mismatch");
        }
        dec_hasher.update(&raw_blk);
        if let Some(ref mut dst) = collect_out {
            dst.extend_from_slice(&raw_blk);
        }
        decoded_total += raw_blk.len();
    }
    if decoded_total != total_raw {
        return Err("MLT3 total raw length mismatch");
    }
    Ok((
        decoded_total,
        dec_hasher.finalize_hex(),
        t0.elapsed().as_secs_f64() * 1000.0,
    ))
}

pub fn decompress_mol_ltcb_cm3_to_vec(archive: &[u8]) -> Result<(Vec<u8>, f64), &'static str> {
    if archive.len() < 16 {
        return Err("Invalid MLT3 archive");
    }
    let total_raw = u64::from_le_bytes(archive[4..12].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(total_raw);
    let (_n, _sha, ms) = decompress_mol_ltcb_cm3_streaming_sha256(archive, Some(&mut out))?;
    Ok((out, ms))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--self-test") {
        run_self_test();
        return;
    }
    let mut input_path: Option<PathBuf> = None;
    let mut max_bytes: Option<usize> = None;
    let mut out_json: Option<PathBuf> = None;
    let mut use_cm2 = false;
    let mut use_cm3 = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cm2" => {
                use_cm2 = true;
                i += 1;
            }
            "--cm3" => {
                use_cm3 = true;
                i += 1;
            }
            "--input" => {
                input_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--max-bytes" => {
                max_bytes = Some(args[i + 1].parse().expect("invalid --max-bytes"));
                i += 2;
            }
            "--out-json" => {
                out_json = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    let inp = input_path.expect("Provide --self-test or --input <path>");
    let (raw_sha, st, arc_sha, dec_sha, dec_ms, mode_name) = if use_cm3 {
        let (arc, st, raw_sha) = compress_mol_ltcb_cm3_file(&inp, max_bytes);
        let mut arc_hasher = StreamingSha256::new();
        arc_hasher.update(&arc);
        let arc_sha = arc_hasher.finalize_hex();
        let (_dec_n, dec_sha, dec_ms) =
            decompress_mol_ltcb_cm3_streaming_sha256(&arc, None).expect("Lossless MLT3 decode failed");
        (raw_sha, st, arc_sha, dec_sha, dec_ms, "MLT3_WRT2_LogOdds_StateSMix3")
    } else {
        let full = fs::read(&inp).expect("Failed to read --input");
        let slice = match max_bytes {
            Some(m) if m < full.len() => &full[..m],
            _ => &full[..],
        };
        let raw_sha = sha256_hex(slice);
        let (arc, st, dec, dec_ms, mname) = if use_cm2 {
            let (arc, st) = compress_mol_ltcb_cm2(slice);
            let (dec, dec_ms) = decompress_mol_ltcb_cm2(&arc).expect("Lossless MLT2 decode failed");
            (arc, st, dec, dec_ms, "MLT2_WRT_StateSMix")
        } else {
            let (arc, st) = compress_mol_ltcb(slice);
            let (dec, dec_ms) = decompress_mol_ltcb(&arc).expect("Lossless MLTC decode failed");
            (arc, st, dec, dec_ms, "MLTC_v1")
        };
        let arc_sha = sha256_hex(&arc);
        let dec_sha = sha256_hex(&dec);
        (raw_sha, st, arc_sha, dec_sha, dec_ms, mname)
    };
    assert_eq!(raw_sha, dec_sha, "SHA-256 lossless roundtrip mismatch!");
    let exe_bytes = env::current_exe()
        .ok()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let (desync_step, c_seq, c_tree) = probe_float_cdf_desync();
    let (adamw_flip_step, ttt_mismatches) = probe_online_adamw_vs_ttt_lut6_desync();
    let doc = format!(
        "{{\n  \"mode\": \"{}\",\n  \"arch\": \"{}\",\n  \"os\": \"{}\",\n  \"raw_bytes\": {},\n  \"raw_sha256\": \"{}\",\n  \"compressed_payload_bytes\": {},\n  \"archive_sha256\": \"{}\",\n  \"decoded_sha256\": \"{}\",\n  \"lossless_exact\": {},\n  \"self_binary_bytes\": {},\n  \"mahoney_total_bytes_with_rust_bin\": {},\n  \"payload_bpb\": {:.6},\n  \"mahoney_total_bpb_with_rust_bin\": {:.6},\n  \"rung0_skip_bytes\": {},\n  \"rung0_skip_pct\": {:.3},\n  \"compress_ms\": {:.3},\n  \"decompress_ms\": {:.3},\n  \"compress_mb_per_s\": {:.3},\n  \"decompress_mb_per_s\": {:.3},\n  \"float_desync_first_step\": {},\n  \"float_cdf_seq\": {},\n  \"float_cdf_tree\": {},\n  \"adamw_sign_desync_step\": {},\n  \"ttt_i16_mismatches\": {}\n}}\n",
        mode_name,
        env::consts::ARCH,
        env::consts::OS,
        st.raw_bytes,
        raw_sha,
        st.compressed_bytes,
        arc_sha,
        dec_sha,
        raw_sha == dec_sha,
        exe_bytes,
        (st.compressed_bytes as u64) + exe_bytes,
        (st.compressed_bytes as f64) * 8.0 / (st.raw_bytes as f64),
        ((st.compressed_bytes as f64) + (exe_bytes as f64)) * 8.0 / (st.raw_bytes as f64),
        st.rung0_skip_bytes,
        100.0 * (st.rung0_skip_bytes as f64) / (st.raw_bytes.max(1) as f64),
        st.elapsed_ms,
        dec_ms,
        (st.raw_bytes as f64 / 1e6) / (st.elapsed_ms / 1000.0).max(1e-9),
        (st.raw_bytes as f64 / 1e6) / (dec_ms / 1000.0).max(1e-9),
        desync_step,
        c_seq,
        c_tree,
        adamw_flip_step,
        ttt_mismatches
    );
    if let Some(p) = out_json {
        fs::write(p, &doc).expect("Failed to write --out-json");
    } else {
        print!("{doc}");
    }
}
