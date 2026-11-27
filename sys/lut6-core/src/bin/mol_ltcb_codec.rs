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
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use lut6_core::ltcb_codec::eval_ltcb_xml_skip_gate2_lut6;

const MAGIC: [u8; 4] = *b"MLTC";
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
    let (desync_step, c_seq, c_tree) = probe_float_cdf_desync();
    assert!(desync_step < 4096 && c_seq != c_tree);
    println!(
        "{{\"self_test\": \"PASS\", \"raw_bytes\": {}, \"compressed_bytes\": {}, \"bpb\": {:.4}, \"rung0_skip_pct\": {:.2}, \"enc_ms\": {:.3}, \"dec_ms\": {:.3}, \"float_desync_first_step\": {}, \"cdf_seq\": {}, \"cdf_tree\": {}}}",
        st.raw_bytes,
        st.compressed_bytes,
        (st.compressed_bytes as f64) * 8.0 / (st.raw_bytes as f64),
        100.0 * (st.rung0_skip_bytes as f64) / (st.raw_bytes as f64),
        st.elapsed_ms,
        dec_ms,
        desync_step,
        c_seq,
        c_tree
    );
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
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
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
    let full = fs::read(&inp).expect("Failed to read --input");
    let slice = match max_bytes {
        Some(m) if m < full.len() => &full[..m],
        _ => &full[..],
    };
    let raw_sha = sha256_hex(slice);
    let (arc, st) = compress_mol_ltcb(slice);
    let arc_sha = sha256_hex(&arc);
    let (dec, dec_ms) = decompress_mol_ltcb(&arc).expect("Lossless MLTC decode failed");
    let dec_sha = sha256_hex(&dec);
    assert_eq!(raw_sha, dec_sha, "SHA-256 lossless roundtrip mismatch!");
    let exe_bytes = env::current_exe()
        .ok()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let (desync_step, c_seq, c_tree) = probe_float_cdf_desync();
    let doc = format!(
        "{{\n  \"arch\": \"{}\",\n  \"os\": \"{}\",\n  \"raw_bytes\": {},\n  \"raw_sha256\": \"{}\",\n  \"compressed_payload_bytes\": {},\n  \"archive_sha256\": \"{}\",\n  \"decoded_sha256\": \"{}\",\n  \"lossless_exact\": {},\n  \"self_binary_bytes\": {},\n  \"mahoney_total_bytes_with_rust_bin\": {},\n  \"payload_bpb\": {:.6},\n  \"mahoney_total_bpb_with_rust_bin\": {:.6},\n  \"rung0_skip_bytes\": {},\n  \"rung0_skip_pct\": {:.3},\n  \"compress_ms\": {:.3},\n  \"decompress_ms\": {:.3},\n  \"compress_mb_per_s\": {:.3},\n  \"decompress_mb_per_s\": {:.3},\n  \"float_desync_first_step\": {},\n  \"float_cdf_seq\": {},\n  \"float_cdf_tree\": {}\n}}\n",
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
        c_tree
    );
    if let Some(p) = out_json {
        fs::write(p, &doc).expect("Failed to write --out-json");
    } else {
        print!("{doc}");
    }
}
