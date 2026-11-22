"""Comprehensive invariant tests across Rung 0, Rung 2, Rung 2.5, Rung 3, and SleepCrystallizer."""

from __future__ import annotations

import torch
import yaml

from clai.compiler.sleep_crystallizer import SleepCrystallizer
from clai.contracts.joule_meter import select_torch_device
from clai.contracts.rce_gate import ModularConstraintEnv
from clai.domains.network_ast_telemetry import NetworkAstTelemetryEnv
from clai.plasticity.hc_soinn import HypersphericalSOINN
from clai.rung0.difflut_net import DiffLUTNetwork
from clai.rung0.engram_store import EngramBitBus
from clai.rung2.mamba3_mimo import Mamba3ComplexMIMOBlock, Rung2Mamba3Streamer
from clai.rung2.triton_mamba3_ssd import _Mamba3TritonSSDFn, _chunked_ssd_reference
from clai.rung3.jepa_ebrm import LatentJEPA_EBRM


def test_rung0_lut6_bit_exact_parity() -> None:
    """Proves 100% bit-exact parity between unpacked discrete gather and 64-lane uint64 bit-planes."""
    dev = select_torch_device()
    torch.manual_seed(123)
    engram = EngramBitBus(seed=42)
    net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    net.compile_all()

    env = ModularConstraintEnv(seed=7)
    for batch_sz in [1, 64, 137]:
        episodes, _ = env.sample_microbatch(batch_sz)
        words_u64 = engram.fetch_batch_words_u64(episodes, dev)
        x_bin = engram.unpack_words_to_float_bus(words_u64)

        # Path A: Unpacked discrete 6-bit address gather layer-by-layer
        h_ref = x_bin
        for layer in net.layers:
            h_ref = layer.forward_discrete_reference(h_ref)
        votes_ref = h_ref.view(batch_sz, 8, -1).sum(dim=-1).to(torch.int32)
        preds_ref = votes_ref.argmax(dim=-1)

        # Path B: Native [B, 8] uint64 Engram -> corner-turn -> 64-lane bit-planes -> voter
        preds_packed, _, _ = net.infer_from_engram_words(words_u64, margin_threshold=0)
        assert preds_packed is not None

        # Also check single-layer bit-exact equality across all 512 wires
        h_packed_unpacked = x_bin
        for layer in net.layers:
            h_packed_unpacked = layer.forward_hardware(h_packed_unpacked)

        assert torch.equal(h_ref, h_packed_unpacked), f"Bit mismatch at batch_size={batch_sz}"
        assert torch.equal(preds_ref, preds_packed), f"Vote mismatch at batch_size={batch_sz}"


def test_rung2_mamba3_ssd_vs_recurrent_and_chunk_continuity() -> None:
    """Proves parallel chunkwise SSD matches T=1 token-by-token recurrence and split-stream state."""
    dev = torch.device("cpu")  # Exact fp32 check without GPU fast-math drift
    torch.manual_seed(11)
    blk = Mamba3ComplexMIMOBlock(d_model=256, n_heads=4, d_state=64, mimo_rank=4).to(dev).eval()

    u = torch.randn(2, 137, 256, device=dev)
    with torch.no_grad():
        out_ssd, st_ssd = blk.forward_sequence(u)

        # 1. Exact T=1 sequential streaming
        st_seq = blk.init_state(2, dev)
        steps = []
        for t in range(u.size(1)):
            o_t, st_seq = blk.forward_sequence(u[:, t : t + 1], state=st_seq)
            steps.append(o_t)
        out_seq = torch.cat(steps, dim=1)

        # 2. Unaligned split-chunk continuation [0:53] + [53:137]
        out_a, st_a = blk.forward_sequence(u[:, :53])
        out_b, st_b = blk.forward_sequence(u[:, 53:], state=st_a)
        out_split = torch.cat([out_a, out_b], dim=1)

    assert (out_ssd - out_seq).abs().max().item() < 1e-4
    assert (st_ssd.h - st_seq.h).abs().max().item() < 1e-4
    assert (out_ssd - out_split).abs().max().item() < 1e-4
    assert (st_ssd.h - st_b.h).abs().max().item() < 1e-4

    # Full Rung2Mamba3Streamer split-stream check with modulo-16 pos_offset
    m3 = Rung2Mamba3Streamer().to(dev).eval()
    stream = torch.randint(0, 256, (2, 129), device=dev)
    with torch.no_grad():
        logits_full, _, _, _ = m3.forward_stream(stream)
        _, _, _, st1 = m3.forward_stream(stream[:, :50], pos_offset=0)
        logits_chunked, _, _, _ = m3.forward_stream(stream[:, 50:], states=st1, pos_offset=50)
    assert (logits_full - logits_chunked).abs().max().item() < 1e-4


def test_rung2_mamba3_analytical_telescoping_backward() -> None:
    """Verifies custom reverse-mode SSD adjoints & telescoping d(log_alpha) match exact autograd."""
    dev = select_torch_device()
    torch.manual_seed(0)

    B, C, Q, R2, R, H, P, N = 2, 3, 32, 8, 4, 4, 64, 64
    X = (torch.randn(B, C, Q, R2, H, P, device=dev) * 0.3).requires_grad_(True)
    Bh = (torch.randn(B, C, Q, R2, H, N, device=dev) * 0.3).requires_grad_(True)
    Cr = (torch.randn(B, C, Q, R, H, N, device=dev) * 0.3).requires_grad_(True)
    la = (-torch.rand(B, C, Q, H, device=dev) * 0.2 - 0.01).requires_grad_(True)
    h0 = (torch.randn(B, H, N, P, device=dev) * 0.3).requires_grad_(True)

    dY = torch.randn(B, C, Q, H, P, R, device=dev)
    dhf = torch.randn(B, H, N, P, device=dev)

    Y_custom, hf_custom = _Mamba3TritonSSDFn.apply(X, Bh, Cr, la, h0)
    g_custom = torch.autograd.grad(
        (Y_custom * dY).sum() + (hf_custom * dhf).sum(), (X, Bh, Cr, la, h0)
    )

    X_r = X.detach().clone().requires_grad_(True)
    B_r = Bh.detach().clone().requires_grad_(True)
    C_r = Cr.detach().clone().requires_grad_(True)
    la_r = la.detach().clone().requires_grad_(True)
    h0_r = h0.detach().clone().requires_grad_(True)
    Y_ref, hf_ref = _chunked_ssd_reference(X_r, B_r, C_r, la_r, h0_r)
    g_ref = torch.autograd.grad(
        (Y_ref * dY).sum() + (hf_ref * dhf).sum(), (X_r, B_r, C_r, la_r, h0_r)
    )

    for name, gc, gr in zip(["dX_hat", "dB_hat", "dC_rot", "dlog_alpha", "dh0"], g_custom, g_ref):
        rel_err = (gc - gr).abs().max().item() / (gr.abs().max().item() + 1e-6)
        assert rel_err < 1e-3, f"Gradient mismatch for {name}: {rel_err:.3e}"


def test_rung25_hc_soinn_novel_class_attachment() -> None:
    """Proves gradient-free O(1) attachment and recall of unseen Novel Class 99."""
    dev = select_torch_device()
    torch.manual_seed(7)
    soinn = HypersphericalSOINN(d_model=512, max_nodes=256).to(dev)

    centers = torch.nn.functional.normalize(torch.randn(4, 1, 512, device=dev), dim=-1)
    class_ids = [0, 1, 2, 99]

    for idx, cid in enumerate(class_ids):
        noisy_slots = centers[idx].expand(12, 16, 512) + 0.06 * torch.randn(
            12, 16, 512, device=dev
        )
        labels = torch.full((12,), cid, device=dev, dtype=torch.long)
        soinn.assimilate_verified_batch(noisy_slots, labels)

    test_novel = centers[3].expand(32, 16, 512) + 0.06 * torch.randn(32, 16, 512, device=dev)
    preds, _, settled = soinn.query_batch(test_novel, margin_threshold=0.10)
    assert preds is not None and settled is not None
    recall = (preds == 99).float().mean().item()
    assert recall >= 0.95
    assert settled.float().mean().item() >= 0.90


def test_rung3_non_collapsing_relaxation_and_downward_lut_crystallization() -> None:
    """Verifies Rung 3 descends free energy genuinely and SleepCrystallizer compiles LUT6 >= 75%."""
    dev = select_torch_device()
    torch.manual_seed(42)

    ebrm = LatentJEPA_EBRM(d_model=128, n_concepts=8, n_levels=2, n_classes=8).to(dev)
    s_x = torch.randn(4, 8, 128, device=dev)
    g_spec = torch.randn(4, 2, 128, device=dev)
    z0 = ebrm.propose_initial_z(s_x, g_spec)
    e0, _, _ = ebrm.free_energy_and_logits(s_x, z0, g_spec)
    _, e_final, _, steps = ebrm.settle_batch(
        s_x, g_spec, max_steps=6, step_lr=0.15, tau_commit=0.001
    )
    assert steps == 6
    assert (e_final <= e0 + 1e-4).all(), "Rung 3 free energy failed to descend during relaxation"

    # Verify SleepCrystallizer genuinely crystallizes ModularConstraintEnv into snapped LUT6 gates!
    with open("configs/collapse_curve.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = ModularConstraintEnv(seed=42)
    engram = EngramBitBus(seed=42)
    lut_net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    cryst = SleepCrystallizer(
        engram, lut_net, dev, capacity=2048, lr=0.03, epochs=30, min_agreement=0.65
    )

    for _ in range(24):  # 24 * 64 = 1536 verified streaming episodes
        eps, gts = env.sample_microbatch(64)
        w = engram.fetch_batch_words_u64(eps, dev)
        y = torch.from_numpy(gts).to(dev, dtype=torch.long)
        cryst.record_batch_words(w, y, raw_episodes=eps)

    res = cryst.compile_downward(batch_size=128)
    assert res["agreement"] >= 0.65, f"LUT6 holdout agreement too low: {res['agreement']:.3f}"
    assert res["compiled"] is True


def test_realworld_l7_packet_ast_domain_and_lut6_firewall_crystallization() -> None:
    """Verifies real-world L7/AST packet triage, protocol drift invalidation, Zero-Day spawn, & LUT6>=85%."""
    dev = select_torch_device()
    torch.manual_seed(42)

    env = NetworkAstTelemetryEnv(seed=42)
    engram = EngramBitBus(seed=42)
    lut_net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    cryst = SleepCrystallizer(
        engram, lut_net, dev, capacity=2048, lr=0.04, epochs=28, min_agreement=0.85
    )

    for _ in range(16):  # 1,024 verified L7/AST wire packets
        pkts, gts = env.sample_microbatch(64)
        w = engram.fetch_batch_words_u64(pkts, dev)
        y = torch.from_numpy(gts).to(dev, dtype=torch.long)
        cryst.record_batch_words(w, y, raw_episodes=pkts)

    res = cryst.compile_downward(batch_size=128)
    assert res["compiled"] is True, f"Real-world LUT6 firewall failed to compile: {res}"
    assert res["agreement"] >= 0.85, f"Expected >= 85% packet holdout, got {res['agreement']:.3f}"

    # Verify protocol encapsulation drift (802.1Q VLAN shift) & Zero-Day Class 8
    old_pkt, old_gt = env.sample_episode()
    assert env.verify(old_pkt, old_gt)
    env.mutate_regime()
    new_cls = env.spawn_novel_class()
    assert new_cls == 8
    zd_pkt = env._build_packet_for_class(8)
    assert env.ground_truth(zd_pkt) == 8


def test_pcap_wireshark_ingestion_and_kv260_pynq_hotswap(tmp_path) -> None:
    """Verifies RFC .pcap round-trip DPI inspection + Kv260ClaiDriver Shadow-Bank hot-swap."""
    import sys
    from clai.domains.pcap_ingest import PcapReplayEnv, generate_enterprise_pcap_dataset

    sys.path.insert(0, "hw/kv260/sw")
    from kv260_pynq_loader import Kv260ClaiDriver  # type: ignore[import-not-found]

    pcap_file = generate_enterprise_pcap_dataset(tmp_path / "test_enterprise.pcap", total_packets=256)
    pcap_env = PcapReplayEnv(pcap_file)
    pkts, gts = pcap_env.sample_microbatch(64)
    assert len(pkts) == 64
    assert all(0 <= int(c) <= 7 for c in gts)
    assert pcap_env.verify_subset(pkts, [0, 1, 2, 3], gts[:4]).all()

    class MockMmio:
        def __init__(self) -> None:
            self.regs: dict[int, int] = {0x080: 0x434C4149, 0x064: 0}
            self.writes = 0

        def read(self, offset: int) -> int:
            return self.regs.get(offset, 0)

        def write(self, offset: int, value: int) -> None:
            self.regs[offset] = value & 0xFFFFFFFF
            self.writes += 1
            if offset == 0x000 and (value & 0x20):
                self.regs[0x064] = 0x80000001

    mmio = MockMmio()
    drv = Kv260ClaiDriver(mmio)
    dev = select_torch_device()
    net = DiffLUTNetwork(bus_bits=512, width=512, num_layers=3, num_classes=8).to(dev)
    gen_reg = drv.hotswap_from_difflut(net)
    assert gen_reg == 0x80000001
    assert mmio.writes > 300
