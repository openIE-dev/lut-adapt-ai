//! Full Native Rust + Burn 0.22.1 / CubeCL 4-Rung Collapse Runtime & Sleep Crystallizer.
//! Executes the complete 5,120-episode Mixture of Limits loop on Apple Metal / CUDA / CPU
//! without Python:
//!   - Rung 0:   3-Layer x 512-LUT6 (1,536 LUT6s) #[cube] GPU Chain (`rung0_lut6_gpu_3l`)
//!   - Rung 2:   Trainable Burn Module wrapping `mamba3_chunk_ssd_gpu` #[cube] Fwd+Bwd
//!   - Rung 2.5: Hyperspherical HC-SOINN Continual Prototype Graph (EMA `node_radii` floor)
//!   - Rung 3:   Trainable Burn Latent JEPA-EBRM Iterative Basin Settler + Concept Anchors
//!   - Compiler: 3rd-Order Walsh-Hadamard Cumulant Pre-Router + Straight-Through Burn DiffLUT
//!               Trainer + Synthesizable Xilinx KV260 LUT6 Verilog Exporter.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use burn::module::{Module, Param};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig,
};
use burn::optim::{AdamWConfig, GradientsParams, ModuleOptimizer};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};
use rayon::prelude::*;

use crate::cpu_cascade::JouleMeter;
use crate::ext::{mamba3_chunk_ssd_gpu, rung0_lut6_gpu_3l};

pub const BUS_BITS: usize = 512;
pub const LUT_WIDTH: usize = 512;
pub const BASE_CLASSES: usize = 8;
pub const ENGRAM_BUCKETS: usize = 65_536;
pub const EMBED_DIM: usize = 128;

#[derive(Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: (seed ^ 0x9E37_79B9_7F4A_7C15).max(1),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u64 << 24) as f32))
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

/// Deterministic variable-length streaming constraint environment with live schema mutation
/// and open-world novel class expansion (`spawn_novel_class`).
pub struct RustModularConstraintEnv {
    pub base_n_classes: usize,
    pub n_classes: usize,
    pub regime_id: usize,
    pub proj: Vec<Vec<i32>>, // [16][n_classes]
    pub parity_mask: [u8; 16],
    rng: XorShift64,
}

impl RustModularConstraintEnv {
    pub fn new(n_classes: usize, seed: u64) -> Self {
        let mut env = Self {
            base_n_classes: n_classes,
            n_classes,
            regime_id: 0,
            proj: vec![vec![0; n_classes]; 16],
            parity_mask: [0u8; 16],
            rng: XorShift64::new(seed),
        };
        env.build_regime_rules();
        env
    }

    fn build_regime_rules(&mut self) {
        let mut r = XorShift64::new(1000 + (self.regime_id as u64) * 97);
        self.proj = vec![vec![0i32; self.n_classes]; 16];

        let mut slots: Vec<usize> = (0..16).collect();
        r.shuffle(&mut slots);
        let clause_slots = &slots[0..6];

        let mut raw = vec![vec![0.0f64; self.n_classes]; 6];
        for i in 0..6 {
            let mut row_mean = 0.0;
            for c in 0..self.n_classes {
                let v = (r.next_usize(9) as i32 - 4) as f64;
                raw[i][c] = v;
                row_mean += v;
            }
            row_mean /= self.n_classes as f64;
            for c in 0..self.n_classes {
                raw[i][c] -= row_mean;
            }
        }
        for c in 0..self.n_classes {
            let mut col_mean = 0.0;
            for i in 0..6 {
                col_mean += raw[i][c];
            }
            col_mean /= 6.0;
            for i in 0..6 {
                raw[i][c] -= col_mean;
            }
        }
        for (i, &s) in clause_slots.iter().enumerate() {
            for c in 0..self.n_classes {
                self.proj[s][c] = (raw[i][c] * 3.0).round() as i32;
            }
        }
        for c in 0..self.n_classes {
            if (0..16).all(|s| self.proj[s][c] == 0) {
                self.proj[clause_slots[c % 6]][c] = 4;
            }
        }

        self.parity_mask = [0u8; 16];
        let mut pslots: Vec<usize> = (0..16).collect();
        r.shuffle(&mut pslots);
        self.parity_mask[pslots[0]] = 1;
        self.parity_mask[pslots[1]] = 1;
    }

    pub fn mutate_regime(&mut self) {
        self.regime_id += 1;
        self.build_regime_rules();
    }

    pub fn spawn_novel_class(&mut self) -> usize {
        let mut r =
            XorShift64::new(5000 + (self.n_classes as u64) * 31 + (self.regime_id as u64));
        let mut col = [0i32; 16];
        let mut sum = 0i32;
        for s in 0..16 {
            col[s] = (r.next_usize(13) as i32) - 6;
            sum += col[s];
        }
        let mean = sum / 16;
        for s in 0..16 {
            col[s] -= mean;
        }
        let mut bslots: Vec<usize> = (0..16).collect();
        r.shuffle(&mut bslots);
        for &s in &bslots[0..3] {
            col[s] = 9;
        }
        for s in 0..16 {
            self.proj[s].push(col[s]);
        }
        self.n_classes += 1;
        self.n_classes - 1
    }

    pub fn sample_episode(&mut self) -> (Vec<u8>, usize) {
        let len = 16 + self.rng.next_usize(49); // 16..=64
        let mut x = Vec::with_capacity(len);
        for _ in 0..len {
            x.push((self.rng.next_u64() & 0xFF) as u8);
        }
        let gt = self.ground_truth(&x);
        (x, gt)
    }

    pub fn sample_microbatch(&mut self, batch_size: usize) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut eps = Vec::with_capacity(batch_size);
        let mut gts = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let (ep, gt) = self.sample_episode();
            eps.push(ep);
            gts.push(gt);
        }
        (eps, gts)
    }

    pub fn ground_truth(&self, x: &[u8]) -> usize {
        let mut folded_sum = [0i32; 16];
        let mut folded_xor = [0u8; 16];
        for (i, &b) in x.iter().enumerate() {
            folded_sum[i & 15] += b as i32;
            folded_xor[i & 15] ^= b;
        }
        let mut scores = vec![0i32; self.n_classes];
        for c in 0..self.n_classes {
            let mut acc = 0i32;
            for s in 0..16 {
                acc += folded_sum[s] * self.proj[s][c];
            }
            scores[c] = acc;
        }
        let mut p_xor = 0u8;
        for s in 0..16 {
            p_xor ^= folded_xor[s] & self.parity_mask[s];
        }
        if (p_xor & 1) == 1 {
            scores.reverse();
        }
        let mut best_c = 0usize;
        let mut best_v = i32::MIN;
        for (c, &v) in scores.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_c = c;
            }
        }
        best_c
    }

    #[inline]
    pub fn verify(&self, x: &[u8], proposed: usize) -> bool {
        proposed == self.ground_truth(x)
    }
}

/// Exact 512-Wire Deterministic + Associative Engram Bus matching `engram_store.py`.
/// - Words 0..1 (Wires 0..127):   Unscrambled 16-byte little-endian streaming XOR fold
/// - Words 2..5 (Wires 128..383): 16-slot x 16-level chunk-normalized monotonic thermometer bars
/// - Words 6..7 (Wires 384..511): 2-head x 64-bit associative N-gram DRAM table (`write_association_u64`)
pub struct RustEngramBus {
    table6: Vec<u64>,
    table7: Vec<u64>,
    salt6: u64,
    salt7: u64,
    class_sigs: [[u64; 2]; 64],
}

impl RustEngramBus {
    pub fn new(seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let mut table6 = vec![0u64; ENGRAM_BUCKETS];
        let mut table7 = vec![0u64; ENGRAM_BUCKETS];
        for i in 0..ENGRAM_BUCKETS {
            table6[i] = rng.next_u64();
            table7[i] = rng.next_u64();
        }
        let salt6 = rng.next_u64() | 1;
        let salt7 = rng.next_u64() | 1;
        let mut class_sigs = [[0u64; 2]; 64];
        for c in 0..64 {
            class_sigs[c] = [rng.next_u64(), rng.next_u64()];
        }
        Self {
            table6,
            table7,
            salt6,
            salt7,
            class_sigs,
        }
    }

    #[inline]
    fn hash_buckets(&self, x: &[u8]) -> (usize, usize) {
        let n_quads = x.len() / 4;
        let mut mix = 0u64;
        for q in 0..n_quads {
            let b = q * 4;
            let quad = (x[b] as u64)
                | ((x[b + 1] as u64) << 8)
                | ((x[b + 2] as u64) << 16)
                | ((x[b + 3] as u64) << 24);
            let w = (q as u64 + 1).wrapping_mul(0x9E37_79B1);
            mix = mix.wrapping_add(quad.wrapping_mul(w));
        }
        let h6 = (((mix.wrapping_mul(self.salt6)) ^ (mix >> 13)) as usize) & (ENGRAM_BUCKETS - 1);
        let h7 = (((mix.wrapping_mul(self.salt7)) ^ (mix >> 13)) as usize) & (ENGRAM_BUCKETS - 1);
        (h6, h7)
    }

    pub fn encode_episode_u64(&self, x: &[u8]) -> [u64; 8] {
        let mut xor_bytes = [0u8; 16];
        let mut slot_sum = [0i32; 16];
        for (i, &b) in x.iter().enumerate() {
            xor_bytes[i & 15] ^= b;
            slot_sum[i & 15] += b as i32;
        }
        let word0 = u64::from_le_bytes(xor_bytes[0..8].try_into().unwrap());
        let word1 = u64::from_le_bytes(xor_bytes[8..16].try_into().unwrap());

        let n_chunks = x.len().div_ceil(16).max(1) as i32;
        let mut words = [0u64; 8];
        words[0] = word0;
        words[1] = word1;

        for grp in 0..4 {
            let mut pack = 0u64;
            for sub in 0..4 {
                let norm = slot_sum[grp * 4 + sub] / n_chunks;
                let mut t16 = 0u64;
                for lvl in 0..16 {
                    let thresh = ((lvl + 1) * 15) as i32;
                    if norm >= thresh {
                        t16 |= 1u64 << lvl;
                    }
                }
                pack |= t16 << (sub * 16);
            }
            words[2 + grp] = pack;
        }

        let (h6, h7) = self.hash_buckets(x);
        words[6] = self.table6[h6];
        words[7] = self.table7[h7];
        words
    }

    pub fn encode_batch_u64(&self, episodes: &[Vec<u8>]) -> Vec<[u64; 8]> {
        episodes.iter().map(|ep| self.encode_episode_u64(ep)).collect()
    }

    pub fn write_association_u64(&mut self, x: &[u8], verified_class: usize) {
        let (h6, h7) = self.hash_buckets(x);
        let sig = self.class_sigs[verified_class & 63];
        self.table6[h6] = sig[0];
        self.table7[h7] = sig[1];
    }

    pub fn words_to_cubecl_tensor(words: &[[u64; 8]], device: &Device) -> Tensor<2, Int> {
        let b = words.len();
        let mut flat = Vec::with_capacity(b * 16);
        for w8 in words {
            for &w in w8 {
                flat.push((w as u32) as i32);
                flat.push(((w >> 32) as u32) as i32);
            }
        }
        Tensor::<2, Int>::from_data(TensorData::new(flat, [b, 16]), device)
    }

    pub fn unpack_words_float(words: &[[u64; 8]]) -> Vec<[f32; BUS_BITS]> {
        words
            .par_iter()
            .map(|w8| {
                let mut out = [0.0f32; BUS_BITS];
                for h in 0..8 {
                    let word = w8[h];
                    for b in 0..64 {
                        out[h * 64 + b] = ((word >> b) & 1) as f32;
                    }
                }
                out
            })
            .collect()
    }
}

/// Snapped 3-Layer x 512-LUT6 (1,536 LUT6s) Rung 0 GPU State + KV260 Verilog Emitter.
pub struct CompiledMetalRung0 {
    pub num_classes: usize,
    pub host_wires: [[[i32; 6]; LUT_WIDTH]; 3],
    pub host_tables: [[u64; LUT_WIDTH]; 3],
    pub gpu_wires: [Tensor<2, Int>; 3],
    pub gpu_tables: [Tensor<2, Int>; 3],
}

impl CompiledMetalRung0 {
    pub fn from_snapped(
        num_classes: usize,
        host_wires: [[[i32; 6]; LUT_WIDTH]; 3],
        host_tables: [[u64; LUT_WIDTH]; 3],
        device: &Device,
    ) -> Self {
        let make_gpu = |layer_idx: usize| -> (Tensor<2, Int>, Tensor<2, Int>) {
            let mut w_flat = Vec::with_capacity(LUT_WIDTH * 6);
            let mut t_flat = Vec::with_capacity(LUT_WIDTH * 2);
            for n in 0..LUT_WIDTH {
                w_flat.extend_from_slice(&host_wires[layer_idx][n]);
                let mask = host_tables[layer_idx][n];
                t_flat.push((mask as u32) as i32);
                t_flat.push(((mask >> 32) as u32) as i32);
            }
            (
                Tensor::<2, Int>::from_data(TensorData::new(w_flat, [LUT_WIDTH, 6]), device),
                Tensor::<2, Int>::from_data(TensorData::new(t_flat, [LUT_WIDTH, 2]), device),
            )
        };
        let (w0, t0) = make_gpu(0);
        let (w1, t1) = make_gpu(1);
        let (w2, t2) = make_gpu(2);
        Self {
            num_classes,
            host_wires,
            host_tables,
            gpu_wires: [w0, w1, w2],
            gpu_tables: [t0, t1, t2],
        }
    }

    /// Executes 3-layer 64-lane LUT6 bit-planes directly on Metal/CUDA via `rung0_lut6_gpu_3l`.
    pub fn infer_batch(
        &self,
        words_u64: &[[u64; 8]],
        margin_threshold: i32,
        device: &Device,
    ) -> Vec<(usize, i32, bool)> {
        let words_t = RustEngramBus::words_to_cubecl_tensor(words_u64, device);
        let votes_t = rung0_lut6_gpu_3l(
            words_t,
            self.gpu_wires[0].clone(),
            self.gpu_tables[0].clone(),
            self.gpu_wires[1].clone(),
            self.gpu_tables[1].clone(),
            self.gpu_wires[2].clone(),
            self.gpu_tables[2].clone(),
            self.num_classes,
        );
        let data = votes_t.into_data();
        let slice = data.as_slice::<i32>().unwrap();
        let mut out = Vec::with_capacity(words_u64.len());
        for ep in 0..words_u64.len() {
            let row = &slice[ep * self.num_classes..(ep + 1) * self.num_classes];
            let mut best_c = 0usize;
            let mut best_v = i32::MIN;
            let mut second_v = i32::MIN;
            for (c, &v) in row.iter().enumerate() {
                if v > best_v {
                    second_v = best_v;
                    best_v = v;
                    best_c = c;
                } else if v > second_v {
                    second_v = v;
                }
            }
            let margin = best_v - second_v;
            out.push((best_c, margin, margin >= margin_threshold));
        }
        out
    }

    pub fn export_kv260_verilog(&self, module_name: &str) -> String {
        let mut s = String::with_capacity(280_000);
        s.push_str("`timescale 1ns/1ps\n\n");
        s.push_str("`ifndef SYNTHESIS\n`ifndef XILINX_LUT6_SIM_CELL\n`define XILINX_LUT6_SIM_CELL\n");
        s.push_str("module LUT6 #(parameter [63:0] INIT = 64'h0000000000000000) (\n");
        s.push_str("    output wire O,\n    input  wire I0,\n    input  wire I1,\n    input  wire I2,\n    input  wire I3,\n    input  wire I4,\n    input  wire I5\n);\n");
        s.push_str("    assign O = INIT[{I5, I4, I3, I2, I1, I0}];\nendmodule\n`endif\n`endif\n\n");
        s.push_str(&format!(
            "module {module_name} (\n    input  wire [511:0] bus_in,\n    output wire [511:0] lut_votes_out\n);\n"
        ));

        let mut prev = "bus_in".to_string();
        for l in 0..3 {
            let cur = format!("stage_{l}");
            s.push_str(&format!("    wire [511:0] {cur};\n"));
            for n in 0..LUT_WIDTH {
                let init = self.host_tables[l][n];
                let w = self.host_wires[l][n];
                s.push_str(&format!(
                    "    LUT6 #(.INIT(64'h{init:016X})) u_l{l}_n{n} (.O({cur}[{n}]), .I0({prev}[{}]), .I1({prev}[{}]), .I2({prev}[{}]), .I3({prev}[{}]), .I4({prev}[{}]), .I5({prev}[{}]));\n",
                    w[0], w[1], w[2], w[3], w[4], w[5]
                ));
            }
            prev = cur;
        }
        s.push_str(&format!("    assign lut_votes_out = {prev};\nendmodule\n"));
        s
    }
}

/// Differentiable Burn 0.22.1 Straight-Through LUT6 Layer for `SleepCrystallizer`.
#[derive(Module, Debug)]
pub struct BurnDiffLutLayer {
    pub table_logits: Param<Tensor<2>>, // [512, 64]
    pub flat_wires: Tensor<1, Int>,     // [512 * 6] discrete pre-routed indices
    pub basis_p: [Tensor<3>; 6],        // 6 x [1, 1, 64] minterm bit masks
}

impl BurnDiffLutLayer {
    pub fn new(
        wires: &[[i32; 6]; LUT_WIDTH],
        init_table_logits: &[[f32; 64]; LUT_WIDTH],
        device: &Device,
    ) -> Self {
        let mut flat_w = Vec::with_capacity(LUT_WIDTH * 6);
        let mut flat_t = Vec::with_capacity(LUT_WIDTH * 64);
        for n in 0..LUT_WIDTH {
            flat_w.extend_from_slice(&wires[n]);
            flat_t.extend_from_slice(&init_table_logits[n]);
        }
        let basis_p = std::array::from_fn(|p| {
            let mut b = [0.0f32; 64];
            for m in 0..64 {
                b[m] = ((m >> p) & 1) as f32;
            }
            Tensor::<3>::from_data(TensorData::new(b.to_vec(), [1, 1, 64]), device)
        });
        Self {
            table_logits: Param::from_tensor(Tensor::<2>::from_data(
                TensorData::new(flat_t, [LUT_WIDTH, 64]),
                device,
            )),
            flat_wires: Tensor::<1, Int>::from_data(
                TensorData::new(flat_w, [LUT_WIDTH * 6]),
                device,
            ),
            basis_p,
        }
    }

    pub fn forward(&self, x: Tensor<2>, tau: f32) -> Tensor<2> {
        let b = x.dims()[0];
        // Inter-layer Straight-Through binarization: exact {0, 1} forward, continuous backward!
        let x_hard = x.clone().greater_scalar(0.5).float();
        let x_st = x_hard - x.clone().detach() + x;

        // Gather [B, 512 * 6] -> [B, 512, 6]
        let u = x_st
            .select(1, self.flat_wires.clone())
            .reshape([b, LUT_WIDTH, 6]);

        let mut corner_probs: Option<Tensor<3>> = None;
        for p in 0..6 {
            let up = u.clone().slice([0..b, 0..LUT_WIDTH, p..(p + 1)]); // [B, 512, 1]
            let bp = self.basis_p[p].clone();
            let one_minus_up = up.clone().neg().add_scalar(1.0);
            let one_minus_bp = bp.clone().neg().add_scalar(1.0);
            let factor = up * bp + one_minus_up * one_minus_bp; // [B, 512, 64]
            corner_probs = Some(match corner_probs {
                None => factor,
                Some(acc) => acc * factor,
            });
        }

        let logits = self.table_logits.val();
        let t_soft = activation::sigmoid(logits.clone().div_scalar(tau.max(0.2)));
        let t_hard = logits.greater_scalar(0.0).float();
        let t_st = (t_hard - t_soft.clone()).detach() + t_soft; // [512, 64]

        (corner_probs.unwrap() * t_st.unsqueeze_dim::<3>(0))
            .sum_dim(2)
            .squeeze_dim::<2>(2)
    }

    pub fn snap_tables_u64(&self) -> [u64; LUT_WIDTH] {
        let data = self.table_logits.val().into_data();
        let slice = data.as_slice::<f32>().unwrap();
        let mut out = [0u64; LUT_WIDTH];
        for n in 0..LUT_WIDTH {
            let mut mask = 0u64;
            for m in 0..64 {
                if slice[n * 64 + m] > 0.0 {
                    mask |= 1u64 << m;
                }
            }
            out[n] = mask;
        }
        out
    }
}

#[derive(Module, Debug)]
pub struct BurnDiffLutNet {
    pub l0: BurnDiffLutLayer,
    pub l1: BurnDiffLutLayer,
    pub l2: BurnDiffLutLayer,
}

impl BurnDiffLutNet {
    pub fn forward(&self, x: Tensor<2>, tau: f32, num_classes: usize) -> Tensor<2> {
        let b = x.dims()[0];
        let h0 = self.l0.forward(x, tau);
        let h1 = self.l1.forward(h0, tau);
        let h2 = self.l2.forward(h1, tau);
        let luts_per_cls = LUT_WIDTH / num_classes;
        let active = num_classes * luts_per_cls;
        h2.slice([0..b, 0..active])
            .reshape([b, num_classes, luts_per_cls])
            .sum_dim(2)
            .squeeze_dim::<2>(2)
    }
}

/// Native Rust Ring-Buffer Sleep Crystallizer with 3rd-order Walsh-Hadamard Cumulant Pre-Router.
pub struct RustSleepCrystallizer {
    pub capacity: usize,
    pub word_ring: Vec<[u64; 8]>,
    pub label_ring: Vec<usize>,
    pub write_ptr: usize,
    pub count: usize,
    pub epochs: usize,
    pub lr: f64,
    pub min_agreement: f32,
    rng: XorShift64,
}

impl RustSleepCrystallizer {
    pub fn new(capacity: usize, epochs: usize, lr: f64, min_agreement: f32) -> Self {
        Self {
            capacity,
            word_ring: vec![[0u64; 8]; capacity],
            label_ring: vec![0usize; capacity],
            write_ptr: 0,
            count: 0,
            epochs,
            lr,
            min_agreement,
            rng: XorShift64::new(777),
        }
    }

    pub fn record(
        &mut self,
        engram: &mut RustEngramBus,
        raw_ep: &[u8],
        words: [u64; 8],
        verified_class: usize,
    ) {
        engram.write_association_u64(raw_ep, verified_class);
        self.word_ring[self.write_ptr] = words;
        self.label_ring[self.write_ptr] = verified_class;
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
        self.count = (self.count + 1).min(self.capacity);
    }

    pub fn clear(&mut self) {
        self.write_ptr = 0;
        self.count = 0;
    }

    /// 3rd-order Walsh-Hadamard cumulant pre-router (`E[z_thermo * s_w1 * s_w2 | y=c]`) in Rayon.
    fn seed_spectral_priors(
        x_train: &[[f32; BUS_BITS]],
        y_train: &[usize],
        num_classes: usize,
    ) -> (
        [[[i32; 6]; LUT_WIDTH]; 3],
        [[[f32; 64]; LUT_WIDTH]; 3],
    ) {
        let n_samples = x_train.len();
        let mut mean_th = [0.0f32; 256];
        for row in x_train {
            for j in 0..256 {
                mean_th[j] += 2.0 * row[128 + j] - 1.0;
            }
        }
        for j in 0..256 {
            mean_th[j] /= n_samples.max(1) as f32;
        }

        let s_xor: Vec<[f32; 128]> = x_train
            .iter()
            .map(|r| std::array::from_fn(|i| 2.0 * r[i] - 1.0))
            .collect();
        let s_th: Vec<[f32; 256]> = x_train
            .iter()
            .map(|r| std::array::from_fn(|j| (2.0 * r[128 + j] - 1.0) - mean_th[j]))
            .collect();

        // Active class-salient subspace profiles
        let mut profiles: Vec<[f32; 256]> = Vec::new();
        for c in 0..num_classes {
            let mut acc = [0.0f32; 256];
            let mut cnt = 0usize;
            for i in 0..n_samples {
                if y_train[i] == c {
                    cnt += 1;
                    for j in 0..256 {
                        acc[j] += s_th[i][j];
                    }
                }
            }
            if cnt > 4 {
                let mut norm_sq = 1e-8f32;
                for j in 0..256 {
                    acc[j] /= cnt as f32;
                    norm_sq += acc[j] * acc[j];
                }
                let inv = norm_sq.sqrt().recip();
                for j in 0..256 {
                    acc[j] *= inv;
                }
                profiles.push(acc);
            }
        }

        let z_proj: Vec<Vec<f32>> = s_th
            .iter()
            .map(|row| {
                profiles
                    .iter()
                    .map(|p| row.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f32>())
                    .collect()
            })
            .collect();

        // Parallel 128x128 3rd-order Walsh-Hadamard cumulant scan
        let scores_128: Vec<[f32; 128]> = (0..128)
            .into_par_iter()
            .map(|w1| {
                let mut row_max = [0.0f32; 128];
                for c in 0..num_classes {
                    let idx_c: Vec<usize> =
                        (0..n_samples).filter(|&i| y_train[i] == c).collect();
                    if idx_c.len() < 8 {
                        continue;
                    }
                    let inv_n = 1.0 / (idx_c.len() as f32);
                    for r in 0..profiles.len() {
                        let z_mean = idx_c.iter().map(|&i| z_proj[i][r]).sum::<f32>() * inv_n;
                        for w2 in (w1 + 1)..128 {
                            let mut acc = 0.0f32;
                            for &i in &idx_c {
                                acc += (z_proj[i][r] - z_mean) * s_xor[i][w1] * s_xor[i][w2];
                            }
                            let val = (acc * inv_n).abs();
                            if val > row_max[w2] {
                                row_max[w2] = val;
                            }
                        }
                    }
                }
                row_max
            })
            .collect();

        let mut pw1 = 0usize;
        let mut pw2 = 8usize;
        let mut best_s = -1.0f32;
        for w1 in 0..128 {
            for w2 in (w1 + 1)..128 {
                if scores_128[w1][w2] > best_s {
                    best_s = scores_128[w1][w2];
                    pw1 = w1;
                    pw2 = w2;
                }
            }
        }

        let p_hat: Vec<f32> = s_xor.iter().map(|sx| sx[pw1] * sx[pw2]).collect();

        let mut wires = [[[0i32; 6]; LUT_WIDTH]; 3];
        let mut tables = [[[0.0f32; 64]; LUT_WIDTH]; 3];
        let luts_per_cls = LUT_WIDTH / num_classes;

        for c in 0..num_classes {
            let mut sal_pos = [[0.0f32; 16]; 16];
            let mut sal_neg = [[0.0f32; 16]; 16];
            let mut cnt_p = 0usize;
            let mut cnt_n = 0usize;
            for i in 0..n_samples {
                if y_train[i] == c {
                    if p_hat[i] > 0.0 {
                        cnt_p += 1;
                        for s in 0..16 {
                            for th in 0..16 {
                                sal_pos[s][th] += s_th[i][s * 16 + th];
                            }
                        }
                    } else {
                        cnt_n += 1;
                        for s in 0..16 {
                            for th in 0..16 {
                                sal_neg[s][th] += s_th[i][s * 16 + th];
                            }
                        }
                    }
                }
            }
            if cnt_p > 0 {
                for s in 0..16 {
                    for th in 0..16 {
                        sal_pos[s][th] /= cnt_p as f32;
                    }
                }
            }
            if cnt_n > 0 {
                for s in 0..16 {
                    for th in 0..16 {
                        sal_neg[s][th] /= cnt_n as f32;
                    }
                }
            }

            let select_diverse = |sal: &[[f32; 16]; 16]| -> Vec<(i32, bool)> {
                let mut slot_order: Vec<(usize, f32)> = (0..16)
                    .map(|s| {
                        let m = sal[s].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        (s, m)
                    })
                    .collect();
                slot_order.sort_by(|a, b| b.1.total_cmp(&a.1));
                let mut picked = Vec::with_capacity(24);
                for &(s_id, _) in &slot_order[0..6] {
                    let mut ths: Vec<(usize, f32)> = (0..16)
                        .map(|th| (th, sal[s_id][th].abs()))
                        .collect();
                    ths.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for &(th, _) in &ths[0..4] {
                        picked.push(((128 + s_id * 16 + th) as i32, sal[s_id][th] >= 0.0));
                    }
                }
                picked
            };

            let wp = select_diverse(&sal_pos);
            let wn = select_diverse(&sal_neg);

            for k in 0..luts_per_cls {
                let n = c * luts_per_cls + k;
                let (ip0, ip1) = ((k * 2) % wp.len(), (k * 2 + 5) % wp.len());
                let (in0, in1) = ((k * 2) % wn.len(), (k * 2 + 5) % wn.len());
                wires[0][n] = [
                    wp[ip0].0,
                    wp[ip1].0,
                    wn[in0].0,
                    wn[in1].0,
                    pw1 as i32,
                    pw2 as i32,
                ];
                for m in 0..64 {
                    let b0 = (m & 1) as f32;
                    let b1 = ((m >> 1) & 1) as f32;
                    let b2 = ((m >> 2) & 1) as f32;
                    let b3 = ((m >> 3) & 1) as f32;
                    let p_xor = (((m >> 4) & 1) ^ ((m >> 5) & 1)) as f32;
                    let vp0 = if wp[ip0].1 { b0 } else { 1.0 - b0 };
                    let vp1 = if wp[ip1].1 { b1 } else { 1.0 - b1 };
                    let vn0 = if wn[in0].1 { b2 } else { 1.0 - b2 };
                    let vn1 = if wn[in1].1 { b3 } else { 1.0 - b3 };
                    let mux = (1.0 - p_xor) * 0.5 * (vp0 + vp1) + p_xor * 0.5 * (vn0 + vn1);
                    tables[0][n][m] = 1.8 * (2.0 * mux - 1.0);
                }
            }
        }

        for l in 1..3 {
            for c in 0..num_classes {
                for k in 0..luts_per_cls {
                    let n = c * luts_per_cls + k;
                    for p in 0..6 {
                        wires[l][n][p] =
                            (c * luts_per_cls + ((k + p * 11) % luts_per_cls)) as i32;
                    }
                    for m in 0..64 {
                        let pop = (m as u32).count_ones();
                        let maj = if pop >= 3 { 1.0 } else { 0.0 };
                        tables[l][n][m] = 1.8 * (2.0 * maj - 1.0);
                    }
                }
            }
        }

        (wires, tables)
    }

    pub fn compile_downward(
        &mut self,
        num_classes: usize,
        device: &Device,
    ) -> (Option<CompiledMetalRung0>, f32) {
        if self.count < 256 {
            return (None, 0.0);
        }

        let mut indices: Vec<usize> = (0..self.count).collect();
        self.rng.shuffle(&mut indices);

        let val_size = (((self.count as f32 * 0.15) as usize) / 64 * 64).max(64);
        let split = self.count - val_size;

        let train_words: Vec<[u64; 8]> =
            indices[0..split].iter().map(|&i| self.word_ring[i]).collect();
        let train_y: Vec<usize> =
            indices[0..split].iter().map(|&i| self.label_ring[i]).collect();
        let val_words: Vec<[u64; 8]> =
            indices[split..].iter().map(|&i| self.word_ring[i]).collect();
        let val_y: Vec<usize> =
            indices[split..].iter().map(|&i| self.label_ring[i]).collect();

        let x_train_f = RustEngramBus::unpack_words_float(&train_words);
        let (seeded_wires, seeded_tables) =
            Self::seed_spectral_priors(&x_train_f, &train_y, num_classes);

        let ad_dev = device.clone().autodiff();
        let mut net = BurnDiffLutNet {
            l0: BurnDiffLutLayer::new(&seeded_wires[0], &seeded_tables[0], &ad_dev),
            l1: BurnDiffLutLayer::new(&seeded_wires[1], &seeded_tables[1], &ad_dev),
            l2: BurnDiffLutLayer::new(&seeded_wires[2], &seeded_tables[2], &ad_dev),
        };
        let ce = CrossEntropyLossConfig::new().init(&ad_dev);
        let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();

        let batch_sz = 128usize;
        let mut batch_order: Vec<usize> = (0..split).collect();

        for ep in 0..self.epochs {
            let progress = ep as f32 / (self.epochs.saturating_sub(1).max(1) as f32);
            let tau = (0.65 - 0.47 * progress).max(0.18);
            self.rng.shuffle(&mut batch_order);

            for chunk in batch_order.chunks(batch_sz) {
                let b_cur = chunk.len();
                let mut x_flat = Vec::with_capacity(b_cur * BUS_BITS);
                let mut y_flat = Vec::with_capacity(b_cur);
                for &idx in chunk {
                    x_flat.extend_from_slice(&x_train_f[idx]);
                    y_flat.push(train_y[idx] as i32);
                }
                let xb =
                    Tensor::<2>::from_data(TensorData::new(x_flat, [b_cur, BUS_BITS]), &ad_dev);
                let yb = Tensor::<1, Int>::from_data(TensorData::new(y_flat, [b_cur]), &ad_dev);

                let votes = net.forward(xb, tau, num_classes).mul_scalar(0.5);
                let loss = ce.forward(votes, yb);
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = opt.step(self.lr * 1.5, net, grads);
            }
        }

        let val_net = net.valid();
        let snapped_tables = [
            val_net.l0.snap_tables_u64(),
            val_net.l1.snap_tables_u64(),
            val_net.l2.snap_tables_u64(),
        ];

        let compiled =
            CompiledMetalRung0::from_snapped(num_classes, seeded_wires, snapped_tables, device);
        let preds = compiled.infer_batch(&val_words, 0, device);
        let correct = preds
            .iter()
            .zip(val_y.iter())
            .filter(|&((p, _, _), &gt)| *p == gt)
            .count();
        let agreement = correct as f32 / (val_y.len() as f32);

        if agreement >= self.min_agreement {
            (Some(compiled), agreement)
        } else {
            (None, agreement)
        }
    }
}

/// Rung 2 + Rung 3 Trainable Burn 0.22.1 Module invoking `mamba3_chunk_ssd_gpu` #[cube] Fwd+Bwd
/// + Non-Collapsing Latent JEPA-EBRM Concept Anchors on Metal/CUDA.
#[derive(Module, Debug)]
pub struct UpperCascadeNet {
    pub byte_emb: Embedding,
    pub ssd_x_proj: Linear,
    pub ssd_b_proj: Linear,
    pub ssd_c_proj: Linear,
    pub stream_fusion: Linear,
    pub r2_head: Linear,
    pub jepa_proposer: Linear,
    pub ebrm_energy_fc1: Linear,
    pub ebrm_energy_fc2: Linear,
    pub r3_head: Linear,
    pub class_anchors: Param<Tensor<2>>, // [16, EMBED_DIM] non-collapsing concept basins
    pub h0_init: Param<Tensor<3>>,       // [1, 64, 64]
}

impl UpperCascadeNet {
    pub fn new(max_classes: usize, device: &Device) -> Self {
        // Orthogonal-ish separated class concept anchors z*(y) != 0
        let mut rng = XorShift64::new(2026);
        let mut anchors = Vec::with_capacity(max_classes * EMBED_DIM);
        for _ in 0..(max_classes * EMBED_DIM) {
            anchors.push((rng.next_f32() * 2.0 - 1.0) * 0.6);
        }
        Self {
            byte_emb: EmbeddingConfig::new(256, 64).init(device),
            ssd_x_proj: LinearConfig::new(64, 256).init(device), // T=64 x 256 == 2 x 128 x 64
            ssd_b_proj: LinearConfig::new(64, 256).init(device),
            ssd_c_proj: LinearConfig::new(64, 128).init(device), // T=64 x 128 == 2 x 64 x 64
            stream_fusion: LinearConfig::new(64 + 32, EMBED_DIM).init(device),
            r2_head: LinearConfig::new(EMBED_DIM, max_classes).init(device),
            jepa_proposer: LinearConfig::new(EMBED_DIM, EMBED_DIM).init(device),
            ebrm_energy_fc1: LinearConfig::new(EMBED_DIM * 2, EMBED_DIM).init(device),
            ebrm_energy_fc2: LinearConfig::new(EMBED_DIM, 1).init(device),
            r3_head: LinearConfig::new(EMBED_DIM * 2, max_classes).init(device),
            class_anchors: Param::from_tensor(Tensor::<2>::from_data(
                TensorData::new(anchors, [max_classes, EMBED_DIM]),
                device,
            )),
            h0_init: Param::from_tensor(Tensor::<3>::ones([1, 64, 64], device).mul_scalar(0.02)),
        }
    }

    /// Runs custom `mamba3_chunk_ssd_gpu` #[cube] kernel + length-invariant slot summary -> (s_x, r2_logits).
    pub fn encode_rung2(&self, episodes: &[Vec<u8>], active_classes: usize, dev: &Device) -> (Tensor<2>, Tensor<2>) {
        let b = episodes.len();
        let mut tok_flat = vec![0i32; b * 64];
        let mut feat32_flat = vec![0.0f32; b * 32];

        for (ep_i, ep) in episodes.iter().enumerate() {
            let mut slot_sum = [0.0f32; 16];
            let mut slot_xor = [0u8; 16];
            for (t, &byte) in ep.iter().take(64).enumerate() {
                tok_flat[ep_i * 64 + t] = byte as i32;
                slot_sum[t & 15] += byte as f32;
                slot_xor[t & 15] ^= byte;
            }
            let n_chunks = ep.len().div_ceil(16).max(1) as f32;
            for s in 0..16 {
                feat32_flat[ep_i * 32 + s] = (slot_sum[s] / n_chunks - 127.5) / 64.0;
                feat32_flat[ep_i * 32 + 16 + s] = if (slot_xor[s] & 1) == 1 { 1.0 } else { -1.0 };
            }
        }

        let toks = Tensor::<2, Int>::from_data(TensorData::new(tok_flat, [b, 64]), dev);
        let feat32 = Tensor::<2>::from_data(TensorData::new(feat32_flat, [b, 32]), dev);

        let emb = self.byte_emb.forward(toks); // [B, 64, 64]
        // 2 chunks of Q=32, rank R=2 => QR=64, Q2R=128
        let x_hat = self
            .ssd_x_proj
            .forward(emb.clone())
            .reshape([b, 2, 128, 64]);
        let b_hat = self
            .ssd_b_proj
            .forward(emb.clone())
            .reshape([b, 2, 128, 64])
            .mul_scalar(0.15);
        let c_rot = self
            .ssd_c_proj
            .forward(emb)
            .reshape([b, 2, 64, 64])
            .mul_scalar(0.15);

        let cum_a = Tensor::<3>::ones([b, 2, 32], dev).mul_scalar(-0.05);
        let end_dec = Tensor::<3>::ones([b, 2, 128], dev).mul_scalar(0.85);
        let chunk_dec = Tensor::<2>::ones([b, 2], dev).mul_scalar(0.30);
        let h0 = self.h0_init.val().repeat_dim(0, b);

        // Custom #[cube] GPU Mamba-3 Chunkwise SSD Forward (and custom Mamba3SsdBackward in train!)
        let y_ssd = mamba3_chunk_ssd_gpu(x_hat, b_hat, c_rot, cum_a, end_dec, chunk_dec, h0);
        let ssd_pool = y_ssd
            .reshape([b, 128, 64])
            .mean_dim(1)
            .squeeze_dim::<2>(1); // [B, 64]

        let fused = activation::silu(
            self.stream_fusion
                .forward(Tensor::cat(vec![ssd_pool, feat32], 1)),
        );
        // Unit hypersphere projection for HC-SOINN compatibility
        let norm = fused
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .add_scalar(1e-6)
            .sqrt();
        let s_x = fused / norm;
        let logits = self
            .r2_head
            .forward(s_x.clone())
            .slice([0..b, 0..active_classes]);
        (s_x, logits)
    }

    pub fn ebrm_energy_and_logits(
        &self,
        s_x: Tensor<2>,
        z: Tensor<2>,
        active_classes: usize,
    ) -> (Tensor<1>, Tensor<2>) {
        let b = s_x.dims()[0];
        let joint = Tensor::cat(vec![s_x, z], 1);
        let h = activation::silu(self.ebrm_energy_fc1.forward(joint.clone()));
        let energy = activation::softplus(self.ebrm_energy_fc2.forward(h), 1.0)
            .squeeze_dim::<1>(1);
        let r3_logits = self.r3_head.forward(joint).slice([0..b, 0..active_classes]);
        (energy, r3_logits)
    }
}

/// Hyperspherical HC-SOINN Continual Prototype Graph with EMA `node_radii` floor.
pub struct RustHcSoinn {
    pub prototypes: Vec<[f32; EMBED_DIM]>,
    pub labels: Vec<usize>,
    pub counts: Vec<f32>,
    pub radii: Vec<f32>,
}

impl Default for RustHcSoinn {
    fn default() -> Self {
        Self::new()
    }
}

impl RustHcSoinn {
    pub fn new() -> Self {
        Self {
            prototypes: Vec::with_capacity(512),
            labels: Vec::with_capacity(512),
            counts: Vec::with_capacity(512),
            radii: Vec::with_capacity(512),
        }
    }

    pub fn clear(&mut self) {
        self.prototypes.clear();
        self.labels.clear();
        self.counts.clear();
        self.radii.clear();
    }

    pub fn query(&self, z: &[f32; EMBED_DIM], margin_thresh: f32) -> Option<usize> {
        if self.prototypes.len() < 2 {
            return None;
        }
        let mut best_d = f32::MAX;
        let mut best_i = 0usize;
        for (i, p) in self.prototypes.iter().enumerate() {
            let dot: f32 = p.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let d = (1.0 - dot).max(0.0);
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        let win_cls = self.labels[best_i];
        let mut rival_d = f32::MAX;
        for (i, p) in self.prototypes.iter().enumerate() {
            if self.labels[i] != win_cls {
                let dot: f32 = p.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
                rival_d = rival_d.min((1.0 - dot).max(0.0));
            }
        }
        if best_d <= self.radii[best_i] && (rival_d - best_d) >= margin_thresh {
            Some(win_cls)
        } else {
            None
        }
    }

    pub fn assimilate(&mut self, z: &[f32; EMBED_DIM], y: usize) {
        let mut best_d = f32::MAX;
        let mut best_i = None;
        for (i, p) in self.prototypes.iter().enumerate() {
            let dot: f32 = p.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let d = (1.0 - dot).max(0.0);
            if d < best_d {
                best_d = d;
                best_i = Some(i);
            }
        }
        match best_i {
            Some(idx) if self.labels[idx] == y && best_d <= self.radii[idx] => {
                self.counts[idx] += 1.0;
                let eta = 1.0 / self.counts[idx];
                let mut norm_sq = 1e-8f32;
                for d in 0..EMBED_DIM {
                    self.prototypes[idx][d] += eta * (z[d] - self.prototypes[idx][d]);
                    norm_sq += self.prototypes[idx][d] * self.prototypes[idx][d];
                }
                let inv = norm_sq.sqrt().recip();
                for d in 0..EMBED_DIM {
                    self.prototypes[idx][d] *= inv;
                }
                // EMA within-cluster radius floor prevents shrinkage collapse!
                self.radii[idx] = (0.92 * self.radii[idx] + 0.08 * (best_d * 1.8)).max(0.14);
            }
            Some(idx) if self.labels[idx] != y && best_d <= self.radii[idx] => {
                // Schema-mutation / boundary conflict fission
                self.radii[idx] = (best_d * 0.70).max(0.05);
                if self.prototypes.len() < 1024 {
                    self.prototypes.push(*z);
                    self.labels.push(y);
                    self.counts.push(1.0);
                    self.radii.push(0.24);
                }
            }
            _ => {
                if self.prototypes.len() < 1024 {
                    self.prototypes.push(*z);
                    self.labels.push(y);
                    self.counts.push(1.0);
                    self.radii.push(0.24);
                }
            }
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct RustBatchReceipt {
    pub batch_size: usize,
    pub verified: usize,
    pub r0: usize,
    pub r2: usize,
    pub soinn: usize,
    pub r3: usize,
    pub joules: f64,
}

/// Complete Native Rust + Metal GPU Mixture of Limits Runtime.
pub struct RustMixtureOfLimits {
    pub device: Device,
    pub ad_device: Device,
    pub meter: JouleMeter,
    pub engram: RustEngramBus,
    pub rung0: Option<CompiledMetalRung0>,
    pub upper: UpperCascadeNet,
    pub upper_opt: ModuleOptimizer,
    pub soinn: RustHcSoinn,
    pub crystallizer: RustSleepCrystallizer,
    pub rejection_accum: usize,
    pub max_parametric_classes: usize,
}

impl RustMixtureOfLimits {
    pub fn new(device: Device) -> Self {
        let ad_device = device.clone().autodiff();
        let max_parametric_classes = 8;
        Self {
            meter: JouleMeter::new(),
            engram: RustEngramBus::new(42),
            rung0: None,
            upper: UpperCascadeNet::new(max_parametric_classes, &ad_device),
            upper_opt: AdamWConfig::new().with_weight_decay(1e-4).init(),
            soinn: RustHcSoinn::new(),
            crystallizer: RustSleepCrystallizer::new(4096, 28, 0.04, 0.56),
            rejection_accum: 0,
            max_parametric_classes,
            device,
            ad_device,
        }
    }

    pub fn step_microbatch(
        &mut self,
        env: &RustModularConstraintEnv,
        episodes: &[Vec<u8>],
    ) -> RustBatchReceipt {
        let b = episodes.len();
        let mut rec = RustBatchReceipt {
            batch_size: b,
            ..Default::default()
        };

        // 1. RUNG 0: 64-Lane Metal #[cube] LUT6 Chain (3.0 W Apple Silicon LUT/SRAM rail profile)
        let (words_u64, r0_out, j_r0) = {
            let (res, j, _) = self.meter.measure(3.0, || {
                let w = self.engram.encode_batch_u64(episodes);
                let out = self
                    .rung0
                    .as_ref()
                    .map(|r0| r0.infer_batch(&w, 4, &self.device));
                (w, out)
            });
            (res.0, res.1, j)
        };
        rec.joules += j_r0;

        let mut escalate_r2 = vec![true; b];
        if let Some(preds) = r0_out {
            let mut satiated = 0usize;
            let mut rejected = 0usize;
            for i in 0..b {
                let (pred_c, _margin, ok) = preds[i];
                if ok {
                    if env.verify(&episodes[i], pred_c) {
                        satiated += 1;
                        escalate_r2[i] = false;
                    } else {
                        rejected += 1;
                    }
                }
            }
            rec.r0 = satiated;
            rec.verified += satiated;

            if rejected > 2 * satiated && rejected >= 3 {
                self.rejection_accum += rejected;
                if self.rejection_accum >= 5 {
                    self.rung0 = None;
                    self.crystallizer.clear();
                    self.soinn.clear();
                    self.rejection_accum = 0;
                }
            } else if satiated >= rejected {
                self.rejection_accum = 0;
            }
        }

        let r2_indices: Vec<usize> = (0..b).filter(|&i| escalate_r2[i]).collect();
        if r2_indices.is_empty() {
            return rec;
        }

        // 2. RUNG 2 (Mamba-3 SSD Metal GPU) + RUNG 2.5 (Hyperspherical HC-SOINN)
        let r2_eps: Vec<Vec<u8>> = r2_indices.iter().map(|&i| episodes[i].clone()).collect();
        let ((s_x_vecs, r2_probs), j_r2, _) = self.meter.measure(18.0, || {
            let val_upper = self.upper.valid();
            let (s_x, logits) =
                val_upper.encode_rung2(&r2_eps, self.max_parametric_classes, &self.device);
            let probs = activation::softmax(logits, 1);
            let sx_d = s_x.into_data();
            let pr_d = probs.into_data();
            let sx_s = sx_d.as_slice::<f32>().unwrap();
            let pr_s = pr_d.as_slice::<f32>().unwrap();

            let mut sx_out = Vec::with_capacity(r2_eps.len());
            let mut pr_out = Vec::with_capacity(r2_eps.len());
            for i in 0..r2_eps.len() {
                let arr: [f32; EMBED_DIM] = sx_s[i * EMBED_DIM..(i + 1) * EMBED_DIM]
                    .try_into()
                    .unwrap();
                sx_out.push(arr);
                pr_out.push(
                    pr_s[i * self.max_parametric_classes..(i + 1) * self.max_parametric_classes]
                        .to_vec(),
                );
            }
            (sx_out, pr_out)
        });
        rec.joules += j_r2;

        let mut escalate_r3_local = Vec::new();
        for (loc_i, &glb_i) in r2_indices.iter().enumerate() {
            let pr = &r2_probs[loc_i];
            let mut best_c = 0usize;
            let mut best_p = -1.0f32;
            let mut sec_p = -1.0f32;
            for (c, &p) in pr.iter().enumerate() {
                if p > best_p {
                    sec_p = best_p;
                    best_p = p;
                    best_c = c;
                } else if p > sec_p {
                    sec_p = p;
                }
            }
            if (best_p - sec_p) >= 0.42 && env.verify(&episodes[glb_i], best_c) {
                rec.r2 += 1;
                rec.verified += 1;
                self.soinn.assimilate(&s_x_vecs[loc_i], best_c);
                if best_c < BASE_CLASSES {
                    self.crystallizer.record(
                        &mut self.engram,
                        &episodes[glb_i],
                        words_u64[glb_i],
                        best_c,
                    );
                }
                continue;
            }

            if let Some(soinn_c) = self.soinn.query(&s_x_vecs[loc_i], 0.035)
                && env.verify(&episodes[glb_i], soinn_c)
            {
                rec.soinn += 1;
                rec.verified += 1;
                self.soinn.assimilate(&s_x_vecs[loc_i], soinn_c);
                if soinn_c < BASE_CLASSES {
                    self.crystallizer.record(
                        &mut self.engram,
                        &episodes[glb_i],
                        words_u64[glb_i],
                        soinn_c,
                    );
                }
                continue;
            }

            escalate_r3_local.push(loc_i);
        }

        if escalate_r3_local.is_empty() {
            return rec;
        }

        // 3. RUNG 3: Iterative JEPA-EBRM Settling + Contrastive Autodiff Adaptation on Metal GPU
        let (_, j_r3, _) = self.meter.measure(68.0, || {
            let b3 = escalate_r3_local.len();
            let val_upper = self.upper.valid();
            let mut sx_flat = Vec::with_capacity(b3 * EMBED_DIM);
            for &loc_i in &escalate_r3_local {
                sx_flat.extend_from_slice(&s_x_vecs[loc_i]);
            }
            let sx_t =
                Tensor::<2>::from_data(TensorData::new(sx_flat, [b3, EMBED_DIM]), &self.device);
            let mut z = val_upper.jepa_proposer.forward(sx_t.clone());
            let anchors = val_upper.class_anchors.val(); // [K, D]

            // Multi-step iterative relaxation toward nearest concept valley
            for _step in 0..6 {
                // Soft attention over concept anchors + energy-guided attraction
                let sim = z.clone().matmul(anchors.clone().transpose()).mul_scalar(3.0);
                let attn = activation::softmax(sim, 1);
                let valley = attn.matmul(anchors.clone());
                z = z.mul_scalar(0.65) + valley.mul_scalar(0.35);
            }
            let (energy, r3_logits) =
                val_upper.ebrm_energy_and_logits(sx_t, z, self.max_parametric_classes);
            let e_data = energy.into_data();
            let l_data = r3_logits.into_data();
            let e_s = e_data.as_slice::<f32>().unwrap();
            let l_s = l_data.as_slice::<f32>().unwrap();

            let mut train_eps = Vec::new();
            let mut train_gts = Vec::new();

            for (k, &loc_i) in escalate_r3_local.iter().enumerate() {
                let glb_i = r2_indices[loc_i];
                let row =
                    &l_s[k * self.max_parametric_classes..(k + 1) * self.max_parametric_classes];
                let mut pred_c = 0usize;
                let mut best_v = f32::MIN;
                for (c, &v) in row.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        pred_c = c;
                    }
                }
                if e_s[k] <= 1.25 && env.verify(&episodes[glb_i], pred_c) {
                    rec.r3 += 1;
                    rec.verified += 1;
                }

                let gt = env.ground_truth(&episodes[glb_i]);
                self.soinn.assimilate(&s_x_vecs[loc_i], gt);
                if gt < BASE_CLASSES {
                    self.crystallizer.record(
                        &mut self.engram,
                        &episodes[glb_i],
                        words_u64[glb_i],
                        gt,
                    );
                    train_eps.push(episodes[glb_i].clone());
                    train_gts.push(gt as i32);
                }
            }

            if !train_eps.is_empty() {
                let bt = train_eps.len();
                let (sx_ad, r2_log_ad) = self.upper.encode_rung2(
                    &train_eps,
                    self.max_parametric_classes,
                    &self.ad_device,
                );
                let gt_t = Tensor::<1, Int>::from_data(
                    TensorData::new(train_gts.clone(), [bt]),
                    &self.ad_device,
                );
                let rival_vec: Vec<i32> = train_gts
                    .iter()
                    .map(|&y| (y + 3) % (self.max_parametric_classes as i32))
                    .collect();
                let rival_t = Tensor::<1, Int>::from_data(
                    TensorData::new(rival_vec, [bt]),
                    &self.ad_device,
                );

                let z_target = self.upper.class_anchors.val().select(0, gt_t.clone());
                let z_rival = self.upper.class_anchors.val().select(0, rival_t);
                let z_init = self.upper.jepa_proposer.forward(sx_ad.clone());

                let (e_pos, r3_log_ad) = self.upper.ebrm_energy_and_logits(
                    sx_ad.clone(),
                    z_target.clone(),
                    self.max_parametric_classes,
                );
                let (e_riv, _) = self.upper.ebrm_energy_and_logits(
                    sx_ad,
                    z_rival,
                    self.max_parametric_classes,
                );

                let ce = CrossEntropyLossConfig::new().init(&self.ad_device);
                let loss = ce.forward(r2_log_ad, gt_t.clone())
                    + ce.forward(r3_log_ad, gt_t)
                    + (z_init - z_target.detach()).powf_scalar(2.0).mean().mul_scalar(0.5)
                    + e_pos.powf_scalar(2.0).mean()
                    + activation::relu(e_riv.neg().add_scalar(0.85))
                        .powf_scalar(2.0)
                        .mean()
                        .mul_scalar(0.5);

                let grads = GradientsParams::from_grads(loss.backward(), &self.upper);
                self.upper = self.upper_opt.step(2.5e-3, self.upper.clone(), grads);
            }
        });
        rec.joules += j_r3;

        rec
    }
}
