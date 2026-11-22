//! Rung 0 Custom #[cube] GPU Kernels: 64-Lane Bit-Plane Transpose, LUT6 Shannon/Minterm Eval, & Voter.
//! Uses dual-word (lo_32, hi_32) 32-bit integer planes so Apple Metal (MSL), NVIDIA CUDA (PTX),
//! AMD ROCm (HIP), Vulkan (SPIR-V), and LLVM CPU all hit 100% native ALU speed with zero 64-bit
//! shader emulation traps.

use burn_backend::{Shape, TensorMetadata, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    cubecl::{self, CubeCount, CubeDim, ir::ElemType, prelude::*},
    kernel::into_contiguous,
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// Kernel 1: Corner-turns [B, 16] wire-packed Engram words (8 heads x 2x32-bit halves per episode)
/// directly into [C, 512, 2] episode-packed 64-lane bit-planes inside Shared SRAM!
/// Grid: (C_chunks, 8_heads, 1) | CubeDim: (64, 1, 1)
#[cube(launch)]
pub fn transpose_engram_64_kernel<I: Int>(
    words_in: &Tensor<I>,         // [B, 16]
    planes_out: &mut Tensor<I>,   // [C, 512, 2] (lo_32_episodes, hi_32_episodes)
    #[comptime] batch_b: u32,
    #[define(I)] _dtype: ElemType,
) {
    let chunk_id = CUBE_POS_X;
    let head_id = CUBE_POS_Y;
    let lane = UNIT_POS_X; // 0..64

    let mut smem_lo = Shared::new_slice(64usize);
    let mut smem_hi = Shared::new_slice(64usize);

    let ep_idx = chunk_id * 64u32 + lane;
    if ep_idx < batch_b {
        let base = (ep_idx * 16u32 + head_id * 2u32) as usize;
        smem_lo[lane as usize] = u32::cast_from(words_in[base]);
        smem_hi[lane as usize] = u32::cast_from(words_in[base + 1usize]);
    } else {
        smem_lo[lane as usize] = 0u32;
        smem_hi[lane as usize] = 0u32;
    }

    sync_cube();

    let w = lane;
    let is_upper_wire = w >= 32u32;
    let bit_shift = if is_upper_wire { w - 32u32 } else { w };

    let mut col_lo = 0u32;
    let mut col_hi = 0u32;

    #[unroll]
    for ep in 0u32..32u32 {
        let src0 = if is_upper_wire {
            smem_hi[ep as usize]
        } else {
            smem_lo[ep as usize]
        };
        let src1 = if is_upper_wire {
            smem_hi[(ep + 32u32) as usize]
        } else {
            smem_lo[(ep + 32u32) as usize]
        };
        col_lo |= ((src0 >> bit_shift) & 1u32) << ep;
        col_hi |= ((src1 >> bit_shift) & 1u32) << ep;
    }

    let wire_global = head_id * 64u32 + w;
    let out_base = ((chunk_id * 512u32 + wire_global) * 2u32) as usize;
    planes_out[out_base] = I::cast_from(col_lo);
    planes_out[out_base + 1usize] = I::cast_from(col_hi);
}

/// Kernel 2: Evaluates 64 episodes simultaneously per GPU thread across arbitrary LUT6 truth tables.
/// Grid: (C_chunks, ceil(num_luts / 128), 1) | CubeDim: (128, 1, 1)
#[cube(launch)]
pub fn lut6_shannon_64lane_kernel<I: Int>(
    in_planes: &Tensor<I>,      // [C, in_wires, 2]
    wires: &Tensor<I>,          // [num_luts, 6]
    tables: &Tensor<I>,         // [num_luts, 2] (lo_32_minterms, hi_32_minterms)
    out_planes: &mut Tensor<I>, // [C, num_luts, 2]
    #[comptime] in_wires: u32,
    #[comptime] num_luts: u32,
    #[define(I)] _dtype: ElemType,
) {
    let chunk_id = CUBE_POS_X;
    let lut_id = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;

    if lut_id < num_luts {
        let w_base = (lut_id * 6u32) as usize;
        let idx0 = u32::cast_from(wires[w_base]);
        let idx1 = u32::cast_from(wires[w_base + 1usize]);
        let idx2 = u32::cast_from(wires[w_base + 2usize]);
        let idx3 = u32::cast_from(wires[w_base + 3usize]);
        let idx4 = u32::cast_from(wires[w_base + 4usize]);
        let idx5 = u32::cast_from(wires[w_base + 5usize]);

        let c_off = chunk_id * in_wires * 2u32;
        let w0_lo = u32::cast_from(in_planes[(c_off + idx0 * 2u32) as usize]);
        let w0_hi = u32::cast_from(in_planes[(c_off + idx0 * 2u32 + 1u32) as usize]);
        let w1_lo = u32::cast_from(in_planes[(c_off + idx1 * 2u32) as usize]);
        let w1_hi = u32::cast_from(in_planes[(c_off + idx1 * 2u32 + 1u32) as usize]);
        let w2_lo = u32::cast_from(in_planes[(c_off + idx2 * 2u32) as usize]);
        let w2_hi = u32::cast_from(in_planes[(c_off + idx2 * 2u32 + 1u32) as usize]);
        let w3_lo = u32::cast_from(in_planes[(c_off + idx3 * 2u32) as usize]);
        let w3_hi = u32::cast_from(in_planes[(c_off + idx3 * 2u32 + 1u32) as usize]);
        let w4_lo = u32::cast_from(in_planes[(c_off + idx4 * 2u32) as usize]);
        let w4_hi = u32::cast_from(in_planes[(c_off + idx4 * 2u32 + 1u32) as usize]);
        let w5_lo = u32::cast_from(in_planes[(c_off + idx5 * 2u32) as usize]);
        let w5_hi = u32::cast_from(in_planes[(c_off + idx5 * 2u32 + 1u32) as usize]);

        let nw0_lo = !w0_lo;
        let nw0_hi = !w0_hi;
        let nw1_lo = !w1_lo;
        let nw1_hi = !w1_hi;
        let nw2_lo = !w2_lo;
        let nw2_hi = !w2_hi;
        let nw3_lo = !w3_lo;
        let nw3_hi = !w3_hi;
        let nw4_lo = !w4_lo;
        let nw4_hi = !w4_hi;
        let nw5_lo = !w5_lo;
        let nw5_hi = !w5_hi;

        let t_lo = u32::cast_from(tables[(lut_id * 2u32) as usize]);
        let t_hi = u32::cast_from(tables[(lut_id * 2u32 + 1u32) as usize]);

        let mut acc_lo = 0u32;
        let mut acc_hi = 0u32;

        #[unroll]
        for m in 0u32..64u32 {
            let t_bit = if m < 32u32 {
                (t_lo >> m) & 1u32
            } else {
                (t_hi >> (m - 32u32)) & 1u32
            };
            let t_mask = if t_bit != 0u32 { 0xFFFF_FFFFu32 } else { 0u32 };

            let b0_lo = if (m & 1u32) != 0u32 { w0_lo } else { nw0_lo };
            let b0_hi = if (m & 1u32) != 0u32 { w0_hi } else { nw0_hi };
            let b1_lo = if (m & 2u32) != 0u32 { w1_lo } else { nw1_lo };
            let b1_hi = if (m & 2u32) != 0u32 { w1_hi } else { nw1_hi };
            let b2_lo = if (m & 4u32) != 0u32 { w2_lo } else { nw2_lo };
            let b2_hi = if (m & 4u32) != 0u32 { w2_hi } else { nw2_hi };
            let b3_lo = if (m & 8u32) != 0u32 { w3_lo } else { nw3_lo };
            let b3_hi = if (m & 8u32) != 0u32 { w3_hi } else { nw3_hi };
            let b4_lo = if (m & 16u32) != 0u32 { w4_lo } else { nw4_lo };
            let b4_hi = if (m & 16u32) != 0u32 { w4_hi } else { nw4_hi };
            let b5_lo = if (m & 32u32) != 0u32 { w5_lo } else { nw5_lo };
            let b5_hi = if (m & 32u32) != 0u32 { w5_hi } else { nw5_hi };

            acc_lo |= t_mask & b0_lo & b1_lo & b2_lo & b3_lo & b4_lo & b5_lo;
            acc_hi |= t_mask & b0_hi & b1_hi & b2_hi & b3_hi & b4_hi & b5_hi;
        }

        let o_base = ((chunk_id * num_luts + lut_id) * 2u32) as usize;
        out_planes[o_base] = I::cast_from(acc_lo);
        out_planes[o_base + 1usize] = I::cast_from(acc_hi);
    }
}

/// Kernel 3: Grouped bit-plane voter summing class popcounts directly from [C, num_luts, 2] bit-planes.
/// Grid: (C_chunks, num_classes, 1) | CubeDim: (64, 1, 1)
#[cube(launch)]
pub fn lut6_bitplane_voter_kernel<I: Int>(
    final_planes: &Tensor<I>, // [C, num_luts, 2]
    votes_out: &mut Tensor<I>,// [B, num_classes]
    #[comptime] batch_b: u32,
    #[comptime] num_luts: u32,
    #[comptime] num_classes: u32,
    #[comptime] luts_per_class: u32,
    #[define(I)] _dtype: ElemType,
) {
    let chunk_id = CUBE_POS_X;
    let cls_id = CUBE_POS_Y;
    let ep_lane = UNIT_POS_X; // 0..64

    let ep_global = chunk_id * 64u32 + ep_lane;
    if ep_global < batch_b {
        let is_hi = ep_lane >= 32u32;
        let bit_pos = if is_hi { ep_lane - 32u32 } else { ep_lane };
        let word_off = if is_hi { 1u32 } else { 0u32 };

        let lut_start = cls_id * luts_per_class;
        let mut vote_sum = 0u32;

        for k in 0u32..luts_per_class {
            let idx = ((chunk_id * num_luts + lut_start + k) * 2u32 + word_off) as usize;
            let word = u32::cast_from(final_planes[idx]);
            vote_sum += (word >> bit_pos) & 1u32;
        }

        votes_out[(ep_global * num_classes + cls_id) as usize] = I::cast_from(vote_sum);
    }
}

/// Host Launcher for CubeBackend across Metal, CUDA, ROCm, Vulkan, & CPU.
pub fn launch_rung0_lut6_chain(
    words: CubeTensor,
    layer_wires: &[CubeTensor],
    layer_tables: &[CubeTensor],
    num_classes: usize,
) -> CubeTensor {
    let b = words.shape().dims::<2>()[0];
    let c_chunks = b.div_ceil(64);
    let client = words.client.clone();
    let device = words.device.clone();
    let i_dtype = words.dtype;
    let storage_ty = dtype_to_storage_type(i_dtype);

    let words_contig = into_contiguous(words);
    let mut cur_planes = empty_device_dtype(
        client.clone(),
        device.clone(),
        Shape::new([c_chunks, 512, 2]),
        i_dtype,
    );

    transpose_engram_64_kernel::launch(
        &client,
        CubeCount::Static(c_chunks as u32, 8, 1),
        CubeDim::new_1d(64),
        words_contig.into_tensor_arg(),
        cur_planes.clone().into_tensor_arg(),
        b as u32,
        storage_ty,
    );

    let mut in_wires = 512usize;
    for (w_t, t_t) in layer_wires.iter().zip(layer_tables.iter()) {
        let w_contig = into_contiguous(w_t.clone());
        let t_contig = into_contiguous(t_t.clone());
        let num_luts = w_contig.shape().dims::<2>()[0];
        let next_planes = empty_device_dtype(
            client.clone(),
            device.clone(),
            Shape::new([c_chunks, num_luts, 2]),
            i_dtype,
        );
        let groups_y = num_luts.div_ceil(128) as u32;

        lut6_shannon_64lane_kernel::launch(
            &client,
            CubeCount::Static(c_chunks as u32, groups_y, 1),
            CubeDim::new_1d(128),
            cur_planes.into_tensor_arg(),
            w_contig.into_tensor_arg(),
            t_contig.into_tensor_arg(),
            next_planes.clone().into_tensor_arg(),
            in_wires as u32,
            num_luts as u32,
            storage_ty,
        );
        cur_planes = next_planes;
        in_wires = num_luts;
    }

    let final_width = in_wires;
    let luts_per_class = final_width / num_classes;
    let votes_out = empty_device_dtype(
        client.clone(),
        device,
        Shape::new([b, num_classes]),
        i_dtype,
    );

    lut6_bitplane_voter_kernel::launch(
        &client,
        CubeCount::Static(c_chunks as u32, num_classes as u32, 1),
        CubeDim::new_1d(64),
        cur_planes.into_tensor_arg(),
        votes_out.clone().into_tensor_arg(),
        b as u32,
        final_width as u32,
        num_classes as u32,
        luts_per_class as u32,
        storage_ty,
    );

    votes_out
}
