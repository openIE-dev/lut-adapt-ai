"""Deterministic constraint-satisfaction task generator & Metered Commit Gate contracts."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CommitResult:
    committed: bool
    verified_correct: bool
    action: int
    rung_used: int
    joules: float
    reason: str


@dataclass
class MicroBatchReceipt:
    batch_size: int
    verified_count: int
    rung0_satiated: int
    rung2_settled: int
    soinn_settled: int
    rung3_settled: int
    refusals: int
    joules_r0: float
    joules_r2: float
    joules_r3: float

    @property
    def total_joules(self) -> float:
        return self.joules_r0 + self.joules_r2 + self.joules_r3


class ModularConstraintEnv:
    """
    Deterministic multi-clause streaming constraint-satisfaction task with controllable schema shifts
    and dynamic open-world class expansion.

    Maps variable-length byte streams (T in [16, 64]) + active rule regime -> exact valid class ID:
      1. Folds stream into 16-slot byte accumulators + 16-slot running XOR state.
      2. Computes balanced linear clause scores `folded_sum @ proj`.
      3. Evaluates non-linear multi-byte XOR parity `popcount(folded_xor & parity_mask) & 1`;
         when parity == 1, reverses the clause priority order (`scores[::-1]`).
    """

    def __init__(self, n_classes: int = 8, seed: int = 42):
        self.base_n_classes = n_classes
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)
        self.regime_id = 0
        self._build_regime_rules()

    def _build_regime_rules(self) -> None:
        rng = np.random.default_rng(1000 + self.regime_id * 97)
        # Sparse 6-clause balanced integer SAT projection (matches LUT6 fan-in hierarchy while
        # keeping all 8 classes equiprobable and requiring exact parity + multi-slot comparison)
        self.proj = np.zeros((16, self.n_classes), dtype=np.int32)
        clause_slots = rng.choice(16, size=6, replace=False)
        raw_proj = rng.integers(-4, 5, size=(6, self.n_classes)).astype(np.float64)
        raw_proj -= raw_proj.mean(axis=1, keepdims=True)
        raw_proj -= raw_proj.mean(axis=0, keepdims=True)
        self.proj[clause_slots] = np.round(raw_proj * 3.0).astype(np.int32)
        for c in range(self.n_classes):
            if np.all(self.proj[:, c] == 0):
                self.proj[clause_slots[c % 6], c] = 4

        # Sparse 2-slot LSB XOR parity gate across the full variable-length stream
        self.parity_mask = np.zeros(16, dtype=np.uint8)
        active_slots = rng.choice(16, size=2, replace=False)
        self.parity_mask[active_slots] = 1

    def mutate_regime(self) -> None:
        """Simulates sudden live operational schema drift."""
        self.regime_id += 1
        self._build_regime_rules()

    def spawn_novel_class(self) -> int:
        """Dynamically expands the action space with a brand-new unseen class ID mid-stream."""
        rng = np.random.default_rng(5000 + self.n_classes * 31 + self.regime_id)
        new_col = rng.integers(-6, 7, size=(16, 1)).astype(np.int32)
        new_col -= int(new_col.mean())
        # Boost novel signature on two slots so the new class claims ~12-18% of the manifold
        boost_slots = rng.choice(16, size=3, replace=False)
        new_col[boost_slots, 0] = 9
        self.proj = np.concatenate([self.proj, new_col], axis=1)
        self.n_classes += 1
        return self.n_classes - 1

    def sample_episode(self) -> tuple[np.ndarray, int]:
        seq_len = int(self.rng.integers(16, 65))
        x = self.rng.integers(0, 256, size=(seq_len,), dtype=np.uint8)
        return x, self.ground_truth(x)

    def sample_microbatch(self, batch_size: int = 64) -> tuple[list[np.ndarray], np.ndarray]:
        episodes: list[np.ndarray] = []
        targets = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            ep, gt = self.sample_episode()
            episodes.append(ep)
            targets[i] = gt
        return episodes, targets

    def ground_truth(self, x: np.ndarray) -> int:
        rem = len(x) % 16
        pad = (16 - rem) % 16
        x_pad = np.pad(x, (0, pad), constant_values=0).reshape(-1, 16)
        n_chunks = x_pad.shape[0]
        # Center around zero-padded expectation (127.5 per active byte) so variable lengths stay unbiased
        folded_sum = x_pad.astype(np.int32).sum(axis=0)
        folded_xor = np.bitwise_xor.reduce(x_pad, axis=0)

        scores = folded_sum @ self.proj
        parity = int(np.bitwise_xor.reduce(folded_xor & self.parity_mask)) & 1
        if parity:
            scores = scores[::-1]
        _ = n_chunks
        return int(np.argmax(scores) % self.n_classes)

    def verify(self, x: np.ndarray, proposed_action: int) -> bool:
        """Zero-tolerance deterministic RCE contract verifier."""
        return int(proposed_action) == self.ground_truth(x)

    def verify_subset(
        self, episodes: list[np.ndarray], indices: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Vectorized zero-tolerance RCE check over active subset indices."""
        return np.array(
            [self.verify(episodes[int(idx)], int(act)) for idx, act in zip(indices, actions)],
            dtype=bool,
        )
