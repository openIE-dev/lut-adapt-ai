//! Pure-Rust Rayon Host Cascade: Cache-Line Aligned 64-Episode u64 Bit-Planes + Mamba-3 + HC-SOINN.

#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use std::fs;
use std::time::Instant;

pub const NUM_HEADS: usize = 8;
pub const BUCKETS: usize = 65_536;
pub const BUS_WIRES: usize = 512;
pub const LUT_WIDTH: usize = 512;
pub const LUT_LAYERS: usize = 3;
pub const NUM_CLASSES: usize = 8;
pub const MAMBA_D: usize = 128;
pub const MAMBA_N: usize = 64;
pub const MAMBA_R: usize = 4;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct BitPlaneChunk512 {
    pub wires: [u64; BUS_WIRES],
}

pub struct JouleMeter {
    rapl_available: bool,
}

impl Default for JouleMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl JouleMeter {
    pub fn new() -> Self {
        let rapl_available =
            fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj").is_ok();
        Self { rapl_available }
    }

    #[inline]
    fn read_uj(&self) -> u64 {
        if self.rapl_available
            && let Ok(s) = fs::read_to_string("/sys/class/powercap/intel-rapl:0/energy_uj")
        {
            return s.trim().parse::<u64>().unwrap_or(0);
        }
        0
    }

    pub fn measure<F, R>(&self, power_watts_est: f64, f: F) -> (R, f64, f64)
    where
        F: FnOnce() -> R,
    {
        let u0 = self.read_uj();
        let t0 = Instant::now();
        let out = f();
        let dt = t0.elapsed().as_secs_f64();
        let u1 = self.read_uj();
        let joules = if self.rapl_available && u1 >= u0 {
            (u1 - u0) as f64 * 1e-6
        } else {
            dt * power_watts_est
        };
        (out, joules, dt * 1e6)
    }
}

pub struct EngramBitBus {
    table: Vec<[u64; BUCKETS]>,
    salts: [u64; NUM_HEADS],
}

impl EngramBitBus {
    pub fn new(seed: u64) -> Self {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut table = Vec::with_capacity(NUM_HEADS);
        for _ in 0..NUM_HEADS {
            let mut bucket_arr = [0u64; BUCKETS];
            for item in bucket_arr.iter_mut() {
                *item = next_u64();
            }
            table.push(bucket_arr);
        }
        let mut salts = [0u64; NUM_HEADS];
        for h in 0..NUM_HEADS {
            salts[h] = next_u64() | 1;
        }
        Self { table, salts }
    }

    pub fn fetch_transposed_64(&self, batch: &[Vec<u8>; 64]) -> BitPlaneChunk512 {
        let mut ep_words = [[0u64; NUM_HEADS]; 64];

        for ep in 0..64 {
            let stream = &batch[ep];
            let quads = stream.len() / 4;
            let mut ngram_mix = 0u64;
            let mut xor_lo = 0u64;
            let mut xor_hi = 0u64;
            let mut slot_sum = [0u32; 16];
            for (idx, &b) in stream.iter().enumerate() {
                slot_sum[idx & 15] += b as u32;
            }
            let n_chunks = stream.len().div_ceil(16).max(1) as u32;

            for q in 0..quads {
                let base = q * 4;
                let word32 = (stream[base] as u64)
                    | ((stream[base + 1] as u64) << 8)
                    | ((stream[base + 2] as u64) << 16)
                    | ((stream[base + 3] as u64) << 24);
                ngram_mix =
                    ngram_mix.wrapping_add(word32.wrapping_mul((q as u64 + 1) * 0x9E37_79B1));
                if q & 1 == 0 {
                    xor_lo ^= word32 << ((q % 4) * 16);
                } else {
                    xor_hi ^= word32 << ((q % 4) * 16);
                }
            }

            ep_words[ep][0] = xor_lo;
            ep_words[ep][1] = xor_hi;

            for grp in 0..4 {
                let mut pack64 = 0u64;
                for s_idx in 0..4 {
                    let norm = slot_sum[grp * 4 + s_idx] / n_chunks;
                    let mut t16 = 0u64;
                    for lvl in 0..16 {
                        if norm >= ((lvl as u32 + 1) * 15) {
                            t16 |= 1u64 << lvl;
                        }
                    }
                    pack64 |= t16 << (s_idx * 16);
                }
                ep_words[ep][2 + grp] = pack64;
            }

            for h in 6..NUM_HEADS {
                let bucket = (((ngram_mix.wrapping_mul(self.salts[h])) ^ (ngram_mix >> 13))
                    as usize)
                    & (BUCKETS - 1);
                ep_words[ep][h] = self.table[h][bucket];
            }
        }

        let mut planes = [0u64; BUS_WIRES];
        for h in 0..NUM_HEADS {
            let wire_base = h * 64;
            for w in 0..64 {
                let mut col = 0u64;
                for ep in 0..64 {
                    col |= ((ep_words[ep][h] >> w) & 1) << ep;
                }
                planes[wire_base + w] = col;
            }
        }
        BitPlaneChunk512 { wires: planes }
    }
}

pub struct CompiledLut6Layer {
    pub routing: Vec<[u16; 6]>,
    pub init_masks: Vec<u64>,
}

impl CompiledLut6Layer {
    pub fn new_random(seed: u64) -> Self {
        let mut s = seed ^ 0xBF58_476D_1CE4_E5B9;
        let mut next_u64 = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut routing = Vec::with_capacity(LUT_WIDTH);
        let mut init_masks = Vec::with_capacity(LUT_WIDTH);
        for _ in 0..LUT_WIDTH {
            routing.push([
                (next_u64() as usize % BUS_WIRES) as u16,
                (next_u64() as usize % BUS_WIRES) as u16,
                (next_u64() as usize % BUS_WIRES) as u16,
                (next_u64() as usize % BUS_WIRES) as u16,
                (next_u64() as usize % BUS_WIRES) as u16,
                (next_u64() as usize % BUS_WIRES) as u16,
            ]);
            init_masks.push(next_u64());
        }
        Self {
            routing,
            init_masks,
        }
    }

    #[inline(always)]
    pub fn eval_64(&self, input: &BitPlaneChunk512) -> BitPlaneChunk512 {
        let mut out = [0u64; LUT_WIDTH];
        out.par_iter_mut().enumerate().for_each(|(n, dst)| {
            let idx = self.routing[n];
            let w0 = input.wires[idx[0] as usize];
            let w1 = input.wires[idx[1] as usize];
            let w2 = input.wires[idx[2] as usize];
            let w3 = input.wires[idx[3] as usize];
            let w4 = input.wires[idx[4] as usize];
            let w5 = input.wires[idx[5] as usize];

            let (nw0, nw1, nw2, nw3, nw4, nw5) = (!w0, !w1, !w2, !w3, !w4, !w5);
            let t = self.init_masks[n];
            let mut acc = 0u64;

            for m in 0..64u64 {
                let t_mask = 0u64.wrapping_sub((t >> m) & 1);
                let b0 = if (m & 1) != 0 { w0 } else { nw0 };
                let b1 = if (m & 2) != 0 { w1 } else { nw1 };
                let b2 = if (m & 4) != 0 { w2 } else { nw2 };
                let b3 = if (m & 8) != 0 { w3 } else { nw3 };
                let b4 = if (m & 16) != 0 { w4 } else { nw4 };
                let b5 = if (m & 32) != 0 { w5 } else { nw5 };
                acc |= t_mask & b0 & b1 & b2 & b3 & b4 & b5;
            }
            *dst = acc;
        });
        BitPlaneChunk512 { wires: out }
    }
}

pub struct Mamba3ComplexMimo {
    a_decay: f32,
}

impl Default for Mamba3ComplexMimo {
    fn default() -> Self {
        Self::new()
    }
}

impl Mamba3ComplexMimo {
    pub fn new() -> Self {
        Self { a_decay: -0.15 }
    }

    pub fn forward_stream(&self, bytes: &[u8]) -> ([f32; MAMBA_D], [f32; NUM_CLASSES], f32) {
        let mut h = [[0.0f32; MAMBA_D]; MAMBA_N];
        let mut prev_bx = [[0.0f32; MAMBA_D]; MAMBA_N];
        let mut phase = [0.0f32; MAMBA_N / 2];

        for &byte in bytes {
            let x_val = (byte as f32 - 127.5) * (1.0 / 128.0);
            let dt = 0.05 + 0.02 * x_val.abs();
            let lam = 0.6f32;
            let alpha = (dt * self.a_decay).exp();
            let gamma = lam * dt;
            let beta = (1.0 - lam) * dt * alpha;

            for k in 0..(MAMBA_N / 2) {
                phase[k] += dt * (x_val * (k as f32 + 1.0) * 0.1);
                let (sin_p, cos_p) = phase[k].sin_cos();
                let b0 = cos_p - sin_p;
                let b1 = sin_p + cos_p;

                let n0 = 2 * k;
                let n1 = 2 * k + 1;
                for p in 0..MAMBA_D {
                    let x_p = x_val * (((p + 1) & 7) as f32 * 0.125);
                    let bx0 = b0 * x_p * (MAMBA_R as f32);
                    let bx1 = b1 * x_p * (MAMBA_R as f32);

                    h[n0][p] = alpha * h[n0][p] + beta * prev_bx[n0][p] + gamma * bx0;
                    h[n1][p] = alpha * h[n1][p] + beta * prev_bx[n1][p] + gamma * bx1;
                    prev_bx[n0][p] = bx0;
                    prev_bx[n1][p] = bx1;
                }
            }
        }

        let mut summary = [0.0f32; MAMBA_D];
        let mut norm_sq = 1e-8f32;
        for p in 0..MAMBA_D {
            let mut acc = 0.0f32;
            for n in 0..MAMBA_N {
                acc += h[n][p];
            }
            summary[p] = acc;
            norm_sq += acc * acc;
        }
        let inv_norm = norm_sq.sqrt().recip();
        for item in summary.iter_mut() {
            *item *= inv_norm;
        }

        let mut logits = [0.0f32; NUM_CLASSES];
        for c in 0..NUM_CLASSES {
            logits[c] = summary[c * (MAMBA_D / NUM_CLASSES)];
        }
        let mut sorted = logits;
        sorted.sort_by(|a, b| b.total_cmp(a));
        let margin = sorted[0] - sorted[1];
        (summary, logits, margin)
    }
}

pub struct HcSoinnGraph {
    pub prototypes: Vec<[f32; MAMBA_D]>,
    pub labels: Vec<usize>,
    pub counts: Vec<f32>,
    pub taus: Vec<f32>,
}

impl Default for HcSoinnGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl HcSoinnGraph {
    pub fn new() -> Self {
        Self {
            prototypes: Vec::with_capacity(512),
            labels: Vec::with_capacity(512),
            counts: Vec::with_capacity(512),
            taus: Vec::with_capacity(512),
        }
    }

    pub fn query(&self, z: &[f32; MAMBA_D], margin_thresh: f32) -> Option<usize> {
        if self.prototypes.len() < 2 {
            return None;
        }
        let mut best_d = f32::MAX;
        let mut best_idx = 0;
        let mut rival_d = f32::MAX;

        for (i, proto) in self.prototypes.iter().enumerate() {
            let dot: f32 = proto.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let d = (1.0 - dot).max(0.0);
            if d < best_d {
                best_d = d;
                best_idx = i;
            }
        }
        let win_label = self.labels[best_idx];
        for (i, proto) in self.prototypes.iter().enumerate() {
            if self.labels[i] != win_label {
                let dot: f32 = proto.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
                rival_d = rival_d.min((1.0 - dot).max(0.0));
            }
        }
        if best_d <= self.taus[best_idx] && (rival_d - best_d) >= margin_thresh {
            Some(win_label)
        } else {
            None
        }
    }

    pub fn assimilate(&mut self, z: &[f32; MAMBA_D], y: usize) {
        let mut best_d = f32::MAX;
        let mut best_idx = None;
        for (i, proto) in self.prototypes.iter().enumerate() {
            let dot: f32 = proto.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let d = (1.0 - dot).max(0.0);
            if d < best_d {
                best_d = d;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) if self.labels[idx] == y && best_d <= self.taus[idx] => {
                self.counts[idx] += 1.0;
                let eta = 1.0 / self.counts[idx];
                let mut norm_sq = 1e-8f32;
                for p in 0..MAMBA_D {
                    self.prototypes[idx][p] += eta * (z[p] - self.prototypes[idx][p]);
                    norm_sq += self.prototypes[idx][p] * self.prototypes[idx][p];
                }
                let inv = norm_sq.sqrt().recip();
                for p in 0..MAMBA_D {
                    self.prototypes[idx][p] *= inv;
                }
            }
            Some(idx) if self.labels[idx] != y && best_d <= self.taus[idx] => {
                self.taus[idx] = (best_d * 0.75).max(0.03);
                self.prototypes.push(*z);
                self.labels.push(y);
                self.counts.push(1.0);
                self.taus.push(0.25);
            }
            _ => {
                self.prototypes.push(*z);
                self.labels.push(y);
                self.counts.push(1.0);
                self.taus.push(0.25);
            }
        }
    }
}

pub fn run_rayon_host_demo() {
    let meter = JouleMeter::new();
    let engram = EngramBitBus::new(42);
    let layers: [CompiledLut6Layer; LUT_LAYERS] = [
        CompiledLut6Layer::new_random(1),
        CompiledLut6Layer::new_random(2),
        CompiledLut6Layer::new_random(3),
    ];
    let mamba3 = Mamba3ComplexMimo::new();
    let mut soinn = HcSoinnGraph::new();

    let batch: [Vec<u8>; 64] = std::array::from_fn(|ep| {
        let len = 16 + (ep % 48);
        (0..len).map(|i| ((ep * 31 + i * 17) & 0xFF) as u8).collect()
    });

    let (votes, j_r0, us_r0) = meter.measure(3.0, || {
        let mut planes = engram.fetch_transposed_64(&batch);
        for layer in &layers {
            planes = layer.eval_64(&planes);
        }
        let mut class_votes = [[0u32; NUM_CLASSES]; 64];
        let luts_per_cls = LUT_WIDTH / NUM_CLASSES;
        for c in 0..NUM_CLASSES {
            for l in 0..luts_per_cls {
                let word = planes.wires[c * luts_per_cls + l];
                for ep in 0..64 {
                    class_votes[ep][c] += ((word >> ep) & 1) as u32;
                }
            }
        }
        class_votes
    });

    let mut r0_satiated = 0usize;
    let mut r25_settled = 0usize;
    let (_, j_upper, _) = meter.measure(28.0, || {
        for ep in 0..64 {
            let mut v = votes[ep];
            v.sort_unstable_by(|a, b| b.cmp(a));
            if v[0] - v[1] >= 4 {
                r0_satiated += 1;
                continue;
            }
            let (z_sphere, _, margin) = mamba3.forward_stream(&batch[ep]);
            let gt_class = if ep >= 56 { 99 } else { ep % NUM_CLASSES };
            if margin < 0.55 {
                if soinn.query(&z_sphere, 0.08).is_some() {
                    r25_settled += 1;
                } else {
                    soinn.assimilate(&z_sphere, gt_class);
                }
            }
        }
    });

    println!(
        "[Rust 1.100 | Rayon Host] 64-Batch -> R0 LUT6: {:.1} us ({}/64 satiated, {:.3e} J) | R2.5 SOINN Nodes: {} (Settled: {}, Total: {:.3e} J)",
        us_r0,
        r0_satiated,
        j_r0,
        soinn.prototypes.len(),
        r25_settled,
        j_r0 + j_upper
    );
}
