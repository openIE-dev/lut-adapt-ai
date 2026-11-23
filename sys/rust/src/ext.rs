//! Burn 0.22.1 `#[backend_extension(Cube, Fusion, Autodiff)]` Dispatch + Custom Autodiff Backward.
//! Fixes the spec's detached `Autodiff::from_inner` placeholder by registering a genuine
//! `Backward<B, 4>` graph node invoking `mamba3_ssd_backward_cube_kernel` on Metal/CUDA/ROCm/CPU.

use burn::backend::{
    Autodiff, Backend, Dispatch, backend_extension,
    autodiff::{
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    fusion::custom::TensorSpec,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{DType, Int, Shape, Tensor};
use burn_backend::{AutodiffBackend, TensorMetadata};
use burn_cubecl::CubeBackend;

use crate::kernels::{
    lut6_cube::launch_rung0_lut6_chain,
    mamba3_ssd_cube::{launch_mamba3_ssd_backward, launch_mamba3_ssd_forward},
    q4_moe_deltanet_cube::{launch_gated_delta_recurrent_k128, launch_q4_moe_ffn_pairs},
};

fn lut6_votes_meta(
    words: &TensorSpec,
    _l0_w: &TensorSpec,
    _l0_t: &TensorSpec,
    _l1_w: &TensorSpec,
    _l1_t: &TensorSpec,
    num_classes: &usize,
) -> TensorSpec {
    TensorSpec {
        shape: Shape::new([words.shape.dims::<2>()[0], *num_classes]),
        dtype: words.dtype,
    }
}

#[allow(clippy::too_many_arguments)]
fn lut6_votes_3l_meta(
    words: &TensorSpec,
    _l0_w: &TensorSpec,
    _l0_t: &TensorSpec,
    _l1_w: &TensorSpec,
    _l1_t: &TensorSpec,
    _l2_w: &TensorSpec,
    _l2_t: &TensorSpec,
    num_classes: &usize,
) -> TensorSpec {
    TensorSpec {
        shape: Shape::new([words.shape.dims::<2>()[0], *num_classes]),
        dtype: words.dtype,
    }
}

fn ssd_fwd_meta(
    x_hat: &TensorSpec,
    _b_hat: &TensorSpec,
    c_rot: &TensorSpec,
    _cum_a: &TensorSpec,
    _end_dec: &TensorSpec,
    _chunk_dec: &TensorSpec,
    _h0: &TensorSpec,
) -> TensorSpec {
    let [bh, c, _, _] = x_hat.shape.dims::<4>();
    let qr = c_rot.shape.dims::<4>()[2];
    TensorSpec {
        shape: Shape::new([bh, c, qr, 64]),
        dtype: x_hat.dtype,
    }
}

#[allow(clippy::too_many_arguments)]
fn ssd_bwd_meta(
    x_hat: &TensorSpec,
    b_hat: &TensorSpec,
    c_rot: &TensorSpec,
    cum_a: &TensorSpec,
    _end_dec: &TensorSpec,
    _chunk_dec: &TensorSpec,
    h0: &TensorSpec,
    _dy: &TensorSpec,
) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
    (
        x_hat.clone(),
        b_hat.clone(),
        c_rot.clone(),
        cum_a.clone(),
        h0.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
fn q4_moe_meta(
    _packed_u32: &TensorSpec,
    _mins_all: &TensorSpec,
    _scales_all: &TensorSpec,
    flat_h: &TensorSpec,
    pair_tok: &TensorSpec,
    _pair_slot: &TensorSpec,
    _pair_rw: &TensorSpec,
    _i_dim: &usize,
) -> TensorSpec {
    let [_t, d_model] = flat_h.shape.dims::<2>();
    let [p_pairs] = pair_tok.shape.dims::<1>();
    TensorSpec {
        shape: Shape::new([p_pairs, d_model]),
        dtype: flat_h.dtype,
    }
}

fn deltanet_meta(
    _q_all: &TensorSpec,
    _k_all: &TensorSpec,
    v_all: &TensorSpec,
    _g_all: &TensorSpec,
    _beta_all: &TensorSpec,
    state_in: &TensorSpec,
) -> (TensorSpec, TensorSpec) {
    (v_all.clone(), state_in.clone())
}

#[backend_extension(Cube, Fusion, Autodiff)]
pub trait CentralLimitOps: Backend {
    /// Rung 0: 64-lane zero-float LUT6 hardware evaluation across two layers + voter.
    #[fusion(meta = lut6_votes_meta)]
    fn rung0_lut6_votes(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self>;

    /// Rung 0: 64-lane zero-float LUT6 hardware evaluation across three layers (1,536 LUT6s) + voter.
    #[fusion(meta = lut6_votes_3l_meta)]
    #[allow(clippy::too_many_arguments)]
    fn rung0_lut6_votes_3l(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        l2_wires: IntTensor<Self>,
        l2_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self>;

    /// Rung 2: Fused Mamba-3 Chunkwise SSD Forward (differentiable through custom CubeCL adjoints).
    #[fusion(meta = ssd_fwd_meta)]
    #[allow(clippy::too_many_arguments)]
    fn mamba3_chunk_ssd_y(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Primitive backward adjoint launcher invoked by `Autodiff<B, C>`.
    #[fusion(meta = ssd_bwd_meta)]
    #[allow(clippy::too_many_arguments)]
    fn mamba3_chunk_ssd_bwd_raw(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );

    /// Rung 3: Cross-platform fused INT4 Block-64 Sparse-MoE Gate+Up+SiLU + Weighted Down-Projection.
    #[fusion(meta = q4_moe_meta)]
    #[allow(clippy::too_many_arguments)]
    fn q4_moe_ffn_pairs(
        packed_u32: IntTensor<Self>,
        mins_all: FloatTensor<Self>,
        scales_all: FloatTensor<Self>,
        flat_h: FloatTensor<Self>,
        pair_tok: IntTensor<Self>,
        pair_slot: IntTensor<Self>,
        pair_rw: FloatTensor<Self>,
        i_dim: usize,
    ) -> FloatTensor<Self>;

    /// Rung 3: Cross-platform fused Multi-Step Gated-DeltaNet Linear-Attention Recurrence (`K_dim=128`).
    #[fusion(meta = deltanet_meta)]
    fn gated_delta_recurrent_k128(
        q_all: FloatTensor<Self>,
        k_all: FloatTensor<Self>,
        v_all: FloatTensor<Self>,
        g_all: FloatTensor<Self>,
        beta_all: FloatTensor<Self>,
        state_in: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
}

impl CentralLimitOps for CubeBackend {
    fn rung0_lut6_votes(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self> {
        launch_rung0_lut6_chain(
            words_u32,
            &[l0_wires, l1_wires],
            &[l0_tables, l1_tables],
            num_classes,
        )
    }

    fn rung0_lut6_votes_3l(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        l2_wires: IntTensor<Self>,
        l2_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self> {
        launch_rung0_lut6_chain(
            words_u32,
            &[l0_wires, l1_wires, l2_wires],
            &[l0_tables, l1_tables, l2_tables],
            num_classes,
        )
    }

    fn mamba3_chunk_ssd_y(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let (y_out, _h_final, _prev_h) =
            launch_mamba3_ssd_forward(x_hat, b_hat, c_rot, cum_a, end_decay, chunk_dec, h0);
        y_out
    }

    fn mamba3_chunk_ssd_bwd_raw(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (_y_out, h_final, prev_h) = launch_mamba3_ssd_forward(
            x_hat.clone(),
            b_hat.clone(),
            c_rot.clone(),
            cum_a.clone(),
            end_decay,
            chunk_dec.clone(),
            h0.clone(),
        );
        let dh_zero = burn_cubecl::ops::numeric::zeros(
            h0.device.clone(),
            h0.shape(),
            DType::F32,
        );
        launch_mamba3_ssd_backward(
            x_hat, b_hat, c_rot, cum_a, chunk_dec, prev_h, h_final, dy, dh_zero,
        )
    }

    fn q4_moe_ffn_pairs(
        packed_u32: IntTensor<Self>,
        mins_all: FloatTensor<Self>,
        scales_all: FloatTensor<Self>,
        flat_h: FloatTensor<Self>,
        pair_tok: IntTensor<Self>,
        pair_slot: IntTensor<Self>,
        pair_rw: FloatTensor<Self>,
        i_dim: usize,
    ) -> FloatTensor<Self> {
        launch_q4_moe_ffn_pairs(
            packed_u32, mins_all, scales_all, flat_h, pair_tok, pair_slot, pair_rw, i_dim,
        )
    }

    fn gated_delta_recurrent_k128(
        q_all: FloatTensor<Self>,
        k_all: FloatTensor<Self>,
        v_all: FloatTensor<Self>,
        g_all: FloatTensor<Self>,
        beta_all: FloatTensor<Self>,
        state_in: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        launch_gated_delta_recurrent_k128(q_all, k_all, v_all, g_all, beta_all, state_in)
    }
}

impl<B: Backend + CentralLimitOps, C: CheckpointStrategy> CentralLimitOps for Autodiff<B, C> {
    fn rung0_lut6_votes(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self> {
        B::rung0_lut6_votes(words_u32, l0_wires, l0_tables, l1_wires, l1_tables, num_classes)
    }

    fn rung0_lut6_votes_3l(
        words_u32: IntTensor<Self>,
        l0_wires: IntTensor<Self>,
        l0_tables: IntTensor<Self>,
        l1_wires: IntTensor<Self>,
        l1_tables: IntTensor<Self>,
        l2_wires: IntTensor<Self>,
        l2_tables: IntTensor<Self>,
        num_classes: usize,
    ) -> IntTensor<Self> {
        B::rung0_lut6_votes_3l(
            words_u32,
            l0_wires,
            l0_tables,
            l1_wires,
            l1_tables,
            l2_wires,
            l2_tables,
            num_classes,
        )
    }

    fn mamba3_chunk_ssd_y(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Mamba3SsdBackward;

        impl<InnerB: Backend + CentralLimitOps> Backward<InnerB, 5> for Mamba3SsdBackward {
            type State = (
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
                FloatTensor<InnerB>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_b, node_c, node_ca, node_h0] = ops.parents;
                let dy = grads.consume::<InnerB>(&ops.node);
                let (x_p, b_p, c_p, ca_p, ed_p, cd_p, h0_p) = ops.state;

                let (dx, db, dc, dca, dh0) =
                    InnerB::mamba3_chunk_ssd_bwd_raw(x_p, b_p, c_p, ca_p, ed_p, cd_p, h0_p, dy);

                if let Some(n) = node_x {
                    grads.register::<InnerB>(n.id, dx);
                }
                if let Some(n) = node_b {
                    grads.register::<InnerB>(n.id, db);
                }
                if let Some(n) = node_c {
                    grads.register::<InnerB>(n.id, dc);
                }
                if let Some(n) = node_ca {
                    grads.register::<InnerB>(n.id, dca);
                }
                if let Some(n) = node_h0 {
                    grads.register::<InnerB>(n.id, dh0);
                }
            }
        }

        match Mamba3SsdBackward
            .prepare::<C>([
                x_hat.node(),
                b_hat.node(),
                c_rot.node(),
                cum_a.node(),
                h0.node(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let x_p = x_hat.primitive().clone();
                let b_p = b_hat.primitive().clone();
                let c_p = c_rot.primitive().clone();
                let ca_p = cum_a.primitive().clone();
                let ed_p = end_decay.primitive().clone();
                let cd_p = chunk_dec.primitive().clone();
                let h0_p = h0.primitive().clone();

                let y = B::mamba3_chunk_ssd_y(
                    x_p.clone(),
                    b_p.clone(),
                    c_p.clone(),
                    ca_p.clone(),
                    ed_p.clone(),
                    cd_p.clone(),
                    h0_p.clone(),
                );
                prep.finish((x_p, b_p, c_p, ca_p, ed_p, cd_p, h0_p), y)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::mamba3_chunk_ssd_y(
                x_hat.into_primitive(),
                b_hat.into_primitive(),
                c_rot.into_primitive(),
                cum_a.into_primitive(),
                end_decay.into_primitive(),
                chunk_dec.into_primitive(),
                h0.into_primitive(),
            )),
        }
    }

    fn mamba3_chunk_ssd_bwd_raw(
        x_hat: FloatTensor<Self>,
        b_hat: FloatTensor<Self>,
        c_rot: FloatTensor<Self>,
        cum_a: FloatTensor<Self>,
        end_decay: FloatTensor<Self>,
        chunk_dec: FloatTensor<Self>,
        h0: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let _ = (x_hat, b_hat, c_rot, cum_a, end_decay, chunk_dec, h0, dy);
        unreachable!("mamba3_chunk_ssd_bwd_raw is an inner primitive called inside Backward")
    }

    fn q4_moe_ffn_pairs(
        packed_u32: IntTensor<Self>,
        mins_all: FloatTensor<Self>,
        scales_all: FloatTensor<Self>,
        flat_h: FloatTensor<Self>,
        pair_tok: IntTensor<Self>,
        pair_slot: IntTensor<Self>,
        pair_rw: FloatTensor<Self>,
        i_dim: usize,
    ) -> FloatTensor<Self> {
        Autodiff::<B, C>::from_inner(B::q4_moe_ffn_pairs(
            packed_u32,
            mins_all.into_primitive(),
            scales_all.into_primitive(),
            flat_h.into_primitive(),
            pair_tok,
            pair_slot,
            pair_rw.into_primitive(),
            i_dim,
        ))
    }

    fn gated_delta_recurrent_k128(
        q_all: FloatTensor<Self>,
        k_all: FloatTensor<Self>,
        v_all: FloatTensor<Self>,
        g_all: FloatTensor<Self>,
        beta_all: FloatTensor<Self>,
        state_in: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (out_p, st_p) = B::gated_delta_recurrent_k128(
            q_all.into_primitive(),
            k_all.into_primitive(),
            v_all.into_primitive(),
            g_all.into_primitive(),
            beta_all.into_primitive(),
            state_in.into_primitive(),
        );
        (
            Autodiff::<B, C>::from_inner(out_p),
            Autodiff::<B, C>::from_inner(st_p),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn q4_moe_ffn_pairs_gpu(
    packed_u32: Tensor<3, Int>,
    mins_all: Tensor<2>,
    scales_all: Tensor<2>,
    flat_h: Tensor<2>,
    pair_tok: Tensor<1, Int>,
    pair_slot: Tensor<1, Int>,
    pair_rw: Tensor<1>,
    i_dim: usize,
) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CentralLimitOps>::q4_moe_ffn_pairs(
        packed_u32.into_dispatch(),
        mins_all.into_dispatch(),
        scales_all.into_dispatch(),
        flat_h.into_dispatch(),
        pair_tok.into_dispatch(),
        pair_slot.into_dispatch(),
        pair_rw.into_dispatch(),
        i_dim,
    ))
}

pub fn gated_delta_recurrent_k128_gpu(
    q_all: Tensor<4>,
    k_all: Tensor<4>,
    v_all: Tensor<4>,
    g_all: Tensor<3>,
    beta_all: Tensor<3>,
    state_in: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    let (out_d, st_d) = <Dispatch as CentralLimitOps>::gated_delta_recurrent_k128(
        q_all.into_dispatch(),
        k_all.into_dispatch(),
        v_all.into_dispatch(),
        g_all.into_dispatch(),
        beta_all.into_dispatch(),
        state_in.into_dispatch(),
    );
    (Tensor::from_dispatch(out_d), Tensor::from_dispatch(st_d))
}

/// Ergonomic generic-free Burn 0.22.1 Tensor<D> wrappers dispatching through `Dispatch`.
pub fn rung0_lut6_gpu(
    words_u32: Tensor<2, Int>,
    l0_wires: Tensor<2, Int>,
    l0_tables: Tensor<2, Int>,
    l1_wires: Tensor<2, Int>,
    l1_tables: Tensor<2, Int>,
    num_classes: usize,
) -> Tensor<2, Int> {
    Tensor::from_dispatch(<Dispatch as CentralLimitOps>::rung0_lut6_votes(
        words_u32.into_dispatch(),
        l0_wires.into_dispatch(),
        l0_tables.into_dispatch(),
        l1_wires.into_dispatch(),
        l1_tables.into_dispatch(),
        num_classes,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn rung0_lut6_gpu_3l(
    words_u32: Tensor<2, Int>,
    l0_wires: Tensor<2, Int>,
    l0_tables: Tensor<2, Int>,
    l1_wires: Tensor<2, Int>,
    l1_tables: Tensor<2, Int>,
    l2_wires: Tensor<2, Int>,
    l2_tables: Tensor<2, Int>,
    num_classes: usize,
) -> Tensor<2, Int> {
    Tensor::from_dispatch(<Dispatch as CentralLimitOps>::rung0_lut6_votes_3l(
        words_u32.into_dispatch(),
        l0_wires.into_dispatch(),
        l0_tables.into_dispatch(),
        l1_wires.into_dispatch(),
        l1_tables.into_dispatch(),
        l2_wires.into_dispatch(),
        l2_tables.into_dispatch(),
        num_classes,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn mamba3_chunk_ssd_gpu(
    x_hat: Tensor<4>,
    b_hat: Tensor<4>,
    c_rot: Tensor<4>,
    cum_a: Tensor<3>,
    end_decay: Tensor<3>,
    chunk_dec: Tensor<2>,
    h0: Tensor<3>,
) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as CentralLimitOps>::mamba3_chunk_ssd_y(
        x_hat.into_dispatch(),
        b_hat.into_dispatch(),
        c_rot.into_dispatch(),
        cum_a.into_dispatch(),
        end_decay.into_dispatch(),
        chunk_dec.into_dispatch(),
        h0.into_dispatch(),
    ))
}
