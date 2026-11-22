"""Native uint64 512-bit Bit-Bus Bridge supporting ragged variable-length 64-episode micro-batches.

Wire allocation across [B, 8] uint64 words (512 binary wires, zero floating-point MACs):
  - Words 0..1 (Wires   0..127): Unscrambled 128-bit streaming XOR fold across 16-byte frames
                                 (preserves exact bit-level parity & raw bit sensitivity).
  - Words 2..5 (Wires 128..383): 16-slot x 16-level monotonic Thermometer bit-planes of the
                                 chunk-normalized 16-slot byte accumulators (eliminates Hamming
                                 cliffs so LUT6 fabrics learn linear thresholds with zero MACs).
  - Words 6..7 (Wires 384..511): 2-head x 64-bit Host-DRAM Associative Engram N-gram table
                                 (O(1) conditional lookup + downward sleep writeback).
"""

from __future__ import annotations

import numpy as np
import torch


def pad_ragged_episodes(
    episodes: list[np.ndarray], multiple_of: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Packs a variable-length list of uint8 1D arrays into [B, T_pad] uint8 + exact lengths [B]."""
    B = len(episodes)
    lengths = np.array([len(ep) for ep in episodes], dtype=np.int64)
    max_len = int(lengths.max()) if B > 0 else multiple_of
    rem = max_len % multiple_of
    t_pad = max_len + ((multiple_of - rem) % multiple_of)

    padded = np.zeros((B, t_pad), dtype=np.uint8)
    for i, ep in enumerate(episodes):
        padded[i, : len(ep)] = ep
    return padded, lengths


class EngramBitBus:
    """
    Native uint64 Host-RAM Conditional Memory + Zero-MAC Bit-Plane Encoder.
    - Table footprint: 8 heads x 65536 buckets x 8 bytes = 4.0 MB (fits inside CPU L3 cache).
    - Output per episode: [B, 8] int64 words (64 bytes/episode = 512 binary wires).
    """

    def __init__(self, num_buckets: int = 65536, num_heads: int = 8, seed: int = 42):
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.assoc_heads = 2  # Words 6..7 carry associative DRAM Engram signatures
        rng = np.random.default_rng(seed)

        # Native packed 64-bit signatures in DRAM: [H, Buckets] uint64
        self.table = rng.integers(
            0, 0xFFFFFFFFFFFFFFFF, size=(num_heads, num_buckets), dtype=np.uint64
        )
        self.salts = rng.integers(1, 0x7FFFFFFF, size=(num_heads,), dtype=np.uint64) | np.uint64(1)
        # Precomputed class signature codebook for O(1) Engram association writeback
        self.class_signatures = rng.integers(
            0, 0xFFFFFFFFFFFFFFFF, size=(64, self.assoc_heads), dtype=np.uint64
        )
        # 16 evenly spaced thermometer thresholds in [15, 240]
        self._thermo_thresholds = np.linspace(15, 240, 16, dtype=np.int32)  # [16]
        self._bit16_shifts = np.arange(16, dtype=np.uint64)

    def _hash_ngrams_ragged(self, x_padded: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Length-exact position-sensitive rolling 4-gram hash -> [B, H] bucket indices."""
        B, T_pad = x_padded.shape
        n_quads = T_pad // 4
        b4 = x_padded[:, : n_quads * 4].astype(np.uint64)

        quads = (
            b4[:, 0::4]
            | (b4[:, 1::4] << np.uint64(8))
            | (b4[:, 2::4] << np.uint64(16))
            | (b4[:, 3::4] << np.uint64(24))
        )  # [B, n_quads]

        quad_idx = np.arange(n_quads, dtype=np.int64)[None, :]
        valid_mask = (quad_idx < (lengths[:, None] // 4)).astype(np.uint64)
        # Position-weighted Golden-ratio mix so permutations do not collide
        pos_weights = (np.arange(1, n_quads + 1, dtype=np.uint64) * np.uint64(0x9E3779B1))[None, :]
        ngram_mix = (quads * pos_weights * valid_mask).sum(axis=1, keepdims=True)  # [B, 1]

        return ((ngram_mix * self.salts[None, :]) ^ (ngram_mix >> np.uint64(13))) % np.uint64(
            self.num_buckets
        )

    def _encode_thermometer_words(self, x_padded: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Encodes chunk-normalized 16-slot byte sums into 4 uint64 words (16 slots x 16 thermometer bits).
        Dividing all 16 slots by n_chunks = ceil(L / 16) preserves exact argmax(folded_sum @ proj)
        invariance across arbitrary stream lengths while bounding every slot to [0, 255].
        """
        B, T_pad = x_padded.shape
        blocks = x_padded.reshape(B, T_pad // 16, 16).astype(np.int32)
        folded_sum = blocks.sum(axis=1)  # [B, 16]
        n_chunks = np.maximum(1, (lengths + 15) // 16)[:, None].astype(np.int32)  # [B, 1]
        norm_slots = folded_sum // n_chunks  # [B, 16] in 0..255 (pure integer normalize)

        # [B, 16, 16] binary thermometer comparisons -> pack into [B, 16] uint16 -> [B, 4] uint64
        thermo_bits = (norm_slots[:, :, None] >= self._thermo_thresholds[None, None, :]).astype(
            np.uint64
        )
        slot_u16 = (thermo_bits << self._bit16_shifts[None, None, :]).sum(axis=-1)  # [B, 16]
        s4 = slot_u16.reshape(B, 4, 4)
        words_4 = (
            s4[:, :, 0]
            | (s4[:, :, 1] << np.uint64(16))
            | (s4[:, :, 2] << np.uint64(32))
            | (s4[:, :, 3] << np.uint64(48))
        )  # [B, 4] uint64
        return words_4

    def fetch_batch_words_u64(
        self, episodes: list[np.ndarray], device: torch.device
    ) -> torch.Tensor:
        """
        Folds B ragged variable-length episodes directly into [B, 8] int64 wire words.
        At B=64, transfers 4,096 bytes and fills 100% of one 64-lane uint64 bit-plane chunk.
        """
        x_padded, lengths = pad_ragged_episodes(episodes, multiple_of=16)
        B = x_padded.shape[0]
        words = np.empty((B, self.num_heads), dtype=np.uint64)

        # 1. Words 0..1 (128 wires): Clean unscrambled 16-byte streaming XOR fold
        chunks_u64 = x_padded.view(np.uint64).reshape(B, -1, 2)
        words[:, :2] = np.bitwise_xor.reduce(chunks_u64, axis=1)

        # 2. Words 2..5 (256 wires): 16-slot x 16-level monotonic thermometer bit-planes
        words[:, 2:6] = self._encode_thermometer_words(x_padded, lengths)

        # 3. Words 6..7 (128 wires): Host-DRAM Associative Engram N-gram lookup
        hashes = self._hash_ngrams_ragged(x_padded, lengths)  # [B, 8]
        words[:, 6] = self.table[6, hashes[:, 6]]
        words[:, 7] = self.table[7, hashes[:, 7]]

        return torch.from_numpy(words.view(np.int64)).to(device=device)

    def fetch_wire_words_u64(
        self, x_bytes: np.ndarray | list[np.ndarray], device: torch.device
    ) -> torch.Tensor:
        """Single-episode or pre-batched compatibility wrapper."""
        if isinstance(x_bytes, list):
            return self.fetch_batch_words_u64(x_bytes, device)
        if x_bytes.ndim == 1:
            return self.fetch_batch_words_u64([x_bytes], device)
        return self.fetch_batch_words_u64(
            [x_bytes[i] for i in range(x_bytes.shape[0])], device
        )

    def unpack_words_to_float_bus(self, words_i64: torch.Tensor) -> torch.Tensor:
        """Unpacks cached [N, 8] int64 wire words -> [N, 512] float32 on device for SleepCrystallizer."""
        shifts = torch.arange(64, device=words_i64.device, dtype=torch.int64).view(1, 1, 64)
        bits = (words_i64.unsqueeze(-1) >> shifts) & 1
        return bits.reshape(words_i64.size(0), 512).to(torch.float32)

    def encode_bus_float(
        self, x_bytes: np.ndarray | list[np.ndarray], device: torch.device
    ) -> torch.Tensor:
        """Offline float32 bus unpacker for SleepCrystallizer training."""
        words_i64 = self.fetch_wire_words_u64(x_bytes, device)
        return self.unpack_words_to_float_bus(words_i64)

    def write_association_u64(self, x_bytes: np.ndarray, verified_class: int) -> None:
        """Instant O(1) DRAM Engram writeback binding an episode's N-gram hash to its verified class."""
        x_padded, lengths = pad_ragged_episodes([x_bytes], multiple_of=16)
        hashes = self._hash_ngrams_ragged(x_padded, lengths)[0]
        sig = self.class_signatures[int(verified_class) % len(self.class_signatures)]
        self.table[6, hashes[6]] = sig[0]
        self.table[7, hashes[7]] = sig[1]
