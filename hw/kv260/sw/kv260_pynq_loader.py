"""PYNQ / Linux MMIO Overlay Loader & Zero-Downtime LUT6 Hot-Swap Bridge for Kria KV260.

Runs directly on AMD/Xilinx Kria KV260 Ubuntu 22.04/24.04 (`pynq.Overlay`) OR against any
MMIO-compatible register array for hardware-in-the-loop testing.
"""

from __future__ import annotations

from typing import Protocol
import numpy as np

from clai.rung0.difflut_net import DiffLUTNetwork


class MmioRegSpace(Protocol):
    def read(self, offset: int) -> int: ...
    def write(self, offset: int, value: int) -> None: ...


class Kv260ClaiDriver:
    """High-level Zynq PS controller for `clai_kv260_axi_top` (`0xA000_0000`)."""

    MAGIC = 0x434C4149

    def __init__(self, mmio: MmioRegSpace):
        self.mmio = mmio
        magic = self.mmio.read(0x080)
        if magic != self.MAGIC:
            raise RuntimeError(f"Invalid CLAI KV260 magic: 0x{magic:08X}")

    @classmethod
    def from_pynq_overlay(cls, bitfile: str = "artifacts/clai_kv260.bit") -> tuple["Kv260ClaiDriver", object]:
        from pynq import Overlay  # type: ignore[import-not-found]

        ol = Overlay(bitfile)
        return cls(ol.clai_rung0_subsys), ol.axi_pkt_dma

    def read_latched_engram_words(self) -> np.ndarray:
        """Reads the hardware-computed 512-bit EngramBitBus (`[1, 8]` uint64) after escalation."""
        words = np.zeros((1, 8), dtype=np.uint64)
        for h in range(8):
            lo = self.mmio.read(0x100 + h * 8) & 0xFFFFFFFF
            hi = self.mmio.read(0x100 + h * 8 + 4) & 0xFFFFFFFF
            words[0, h] = (np.uint64(hi) << np.uint64(32)) | np.uint64(lo)
        return words

    def write_engram_bram(self, bank: int, bucket_addr: int, sig_u64: int) -> None:
        self.mmio.write(0x044, sig_u64 & 0xFFFFFFFF)
        self.mmio.write(0x048, (sig_u64 >> 32) & 0xFFFFFFFF)
        self.mmio.write(0x040, 0x80000000 | ((bank & 1) << 16) | (bucket_addr & 0x3FF))

    def hotswap_from_difflut(self, net: DiffLUTNetwork) -> int:
        """
        Streams the top 8 LUT6s per class (64 LUT6s in L0 + 64 LUT6s in L1) from a compiled
        `DiffLUTNetwork` into the KV260 Shadow Bank and executes a 4.0 ns atomic bank flip!
        """
        if not net.layers[0].is_compiled:
            net.compile_all()

        luts_per_cls_src = net.width // 8
        for layer_sel in (0, 1):
            src_layer = net.layers[layer_sel]
            wires = src_layer.compiled_wires.cpu().numpy()
            packed_t = src_layer.packed_tables_i64.cpu().numpy().view(np.uint64)

            for c in range(8):
                for k in range(4):
                    dst_idx = c * 4 + k
                    src_idx = c * luts_per_cls_src + k
                    t_u64 = int(packed_t[src_idx])
                    w6 = [int(x) for x in wires[src_idx]]
                    if layer_sel == 1:
                        # Map source class group [c*64..c*64+63] -> compact overlay [c*4..c*4+3]
                        w6 = [c * 4 + (w % 4) for w in w6]

                    w_lo = (w6[0] & 0x1FF) | ((w6[1] & 0x1FF) << 9) | ((w6[2] & 0x1FF) << 18)
                    w_hi = (w6[3] & 0x1FF) | ((w6[4] & 0x1FF) << 9) | ((w6[5] & 0x1FF) << 18)
                    self.mmio.write(0x054, t_u64 & 0xFFFFFFFF)
                    self.mmio.write(0x058, (t_u64 >> 32) & 0xFFFFFFFF)
                    self.mmio.write(0x05C, w_lo)
                    self.mmio.write(0x060, w_hi)
                    self.mmio.write(0x050, 0x80000000 | (layer_sel << 8) | dst_idx)

        # CSR_CTRL = IRQ_ENABLE (0x4) | OVERLAY_ENABLE (0x10) | ATOMIC_BANK_SWAP (0x20) = 0x34
        self.mmio.write(0x000, 0x34)
        return self.mmio.read(0x064)
