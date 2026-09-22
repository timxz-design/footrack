"""Windowing, augmentation and splitting for training.

Turns sequences into (history, horizon) pairs — the supervision for the
forecasting pretraining task. Labels are free: the horizon is simply what
happened next (MODEL.md §1.1).

Three things here are load-bearing and easy to get wrong:

- **Splits are by whole match.** Adjacent frames are near-duplicates, so a
  random frame split leaks the answer across it and produces a validation score
  that means nothing.
- **Augmentation happens per sample, not once.** Mirror and perspective swap are
  applied to the window at access time, so each epoch sees different variants.
- **Uniform sampling is a trap.** Most frames are low-information — dead ball,
  a goalkeeper holding it, players walking back. Weighted sampling is available
  and recommended (MODEL.md §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence as TypingSequence

import numpy as np

from footrack.features import POS_SCALE, ModelInput, PlayerProfile, build
from footrack.state import N_ENTITIES, Phase, Sequence

# Default sampling weights by phase of the window's final history frame.
# Dead-ball frames teach almost nothing; turnovers are where positioning
# decisions are densest.
DEFAULT_PHASE_WEIGHTS: dict[int, float] = {
    Phase.UNKNOWN: 0.2,
    Phase.IN_POSSESSION: 1.0,
    Phase.OUT_OF_POSSESSION: 1.0,
    Phase.TRANSITION: 2.0,
    Phase.SET_PIECE: 0.6,
}


@dataclass
class TrainingSample:
    """One (history, horizon) pair."""

    inputs: ModelInput  # history frames
    target_positions: np.ndarray  # (F, 23, 2) model scale
    target_mask: np.ndarray  # (F, 23) bool
    anchor: np.ndarray  # (23, 2) last observed position, model scale
    anchor_valid: np.ndarray  # (23,) bool

    def target_deltas(self) -> np.ndarray:
        """(F, 23, 2) displacement from the anchor.

        Predicting displacement rather than absolute position is better
        conditioned and easier to learn (MODEL.md §L4). Undefined where the
        anchor is missing, so always read alongside ``anchor_valid``.
        """
        return self.target_positions - self.anchor[None, :, :]


def split_by_match(
    sequences: TypingSequence[Sequence],
    val_matches: TypingSequence[str] | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Sequence], list[Sequence]]:
    """Partition sequences into train and validation by **whole match**.

    Never split on frames. Adjacent frames are near-duplicates, so a random
    frame split puts almost-identical samples on both sides and inflates the
    validation score beyond any use.

    Prefer naming ``val_matches`` explicitly: evaluation wants full-optical
    sources such as Metrica and DFL, where every player is present on every
    frame, rather than broadcast data with coverage gaps (DATA.md §2).
    """
    matches = sorted({s.match_id for s in sequences})
    if val_matches is None:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(matches) * val_fraction)))
        val_matches = set(rng.permutation(matches)[:n_val].tolist())
    else:
        val_matches = set(val_matches)
        unknown = val_matches - set(matches)
        if unknown:
            raise ValueError(f"val_matches not present in data: {sorted(unknown)}")

    train = [s for s in sequences if s.match_id not in val_matches]
    val = [s for s in sequences if s.match_id in val_matches]
    return train, val


class WindowDataset:
    """Fixed-length windows sampled from a collection of sequences."""

    def __init__(
        self,
        sequences: TypingSequence[Sequence],
        history: int = 25,
        horizon: int = 20,
        stride: int = 5,
        target_rate: float = 10.0,
        min_coverage: float = 0.6,
        augment: bool = False,
        profiles: dict[str, PlayerProfile] | None = None,
        seed: int = 0,
    ):
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")

        self.history = history
        self.horizon = horizon
        self.augment = augment
        self.profiles = profiles
        self.rng = np.random.default_rng(seed)

        span = history + horizon
        self.sequences: list[Sequence] = []
        index: list[tuple[int, int]] = []

        for seq in sequences:
            factor = max(1, int(round(seq.frame_rate / target_rate)))
            ds = seq.downsample(factor)
            if len(ds) < span:
                continue

            si = len(self.sequences)
            self.sequences.append(ds)

            starts = np.arange(0, len(ds) - span + 1, stride)
            if min_coverage > 0 and len(starts):
                # Reject windows where too much of the pitch is unobserved. A
                # window that is mostly missing players teaches the model about
                # the tracking system, not about football.
                cov = np.array([ds.mask[s : s + span].mean() for s in starts])
                starts = starts[cov >= min_coverage]

            index.extend((si, int(s)) for s in starts)

        self.index: np.ndarray = np.array(index, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> TrainingSample:
        si, start = self.index[i]
        window = self.sequences[si].window(int(start), self.history + self.horizon)

        if self.augment:
            if self.rng.random() < 0.5:
                window = window.mirror()
            if self.rng.random() < 0.5:
                window = window.swap_perspective()

        hist = window.window(0, self.history)
        future = window.window(self.history, self.horizon)

        anchor_valid = hist.mask[-1].copy()
        anchor = np.where(
            anchor_valid[:, None], hist.positions[-1] / POS_SCALE, 0.0
        ).astype(np.float32)

        return TrainingSample(
            inputs=build(hist, profiles=self.profiles),
            target_positions=(future.positions / POS_SCALE).astype(np.float32),
            target_mask=future.mask.copy(),
            anchor=anchor,
            anchor_valid=anchor_valid,
        )

    # -- sampling -------------------------------------------------------------

    def phase_of_windows(self) -> np.ndarray:
        """(N,) phase at the last history frame of each window."""
        out = np.full(len(self), Phase.UNKNOWN, dtype=np.int8)
        for n, (si, start) in enumerate(self.index):
            phase = self.sequences[si].phase
            if phase is not None:
                out[n] = phase[start + self.history - 1]
        return out

    def sampling_weights(
        self, weights: dict[int, float] | None = None
    ) -> np.ndarray:
        """(N,) normalised sampling weights, by phase.

        Uniform sampling is dominated by low-information moments, where
        predicting "everyone drifts toward the centroid" scores well and means
        nothing (MODEL.md §9).
        """
        table = DEFAULT_PHASE_WEIGHTS if weights is None else weights
        w = np.array([table.get(int(p), 1.0) for p in self.phase_of_windows()])
        total = w.sum()
        if total <= 0:
            raise ValueError("sampling weights sum to zero")
        return w / total

    def batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        weighted: bool = False,
        drop_last: bool = True,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield collated batches for one epoch."""
        n = len(self)
        if weighted:
            order = self.rng.choice(n, size=n, replace=True, p=self.sampling_weights())
        elif shuffle:
            order = self.rng.permutation(n)
        else:
            order = np.arange(n)

        stop = n - (n % batch_size) if drop_last else n
        for lo in range(0, stop, batch_size):
            yield collate([self[int(j)] for j in order[lo : lo + batch_size]])


def collate(samples: list[TrainingSample]) -> dict[str, np.ndarray]:
    """Stack samples into batched arrays, ready to hand to a model."""
    return {
        "entity": np.stack([s.inputs.entity for s in samples]),
        "context": np.stack([s.inputs.context for s in samples]),
        "role_ids": np.stack([s.inputs.role_ids for s in samples]),
        "player_ids": np.stack([s.inputs.player_ids for s in samples]),
        "input_mask": np.stack([s.inputs.mask for s in samples]),
        "target_positions": np.stack([s.target_positions for s in samples]),
        "target_mask": np.stack([s.target_mask for s in samples]),
        "target_deltas": np.stack([s.target_deltas() for s in samples]),
        "anchor": np.stack([s.anchor for s in samples]),
        "anchor_valid": np.stack([s.anchor_valid for s in samples]),
    }


def constant_velocity_baseline(sample: TrainingSample, frame_rate: float) -> np.ndarray:
    """(F, 23, 2) positions predicted by extrapolating the last velocity.

    The baseline the world model must beat before it is worth anything
    (MODEL.md §9, stage S1 gate). Roughly right over ~400ms and increasingly
    wrong beyond that, which is precisely the gap a learned model should close.
    """
    from footrack.features import SPEED_SCALE

    last_vel = sample.inputs.entity[-1, :, 2:4] * SPEED_SCALE  # m/s
    steps = np.arange(1, sample.target_positions.shape[0] + 1, dtype=np.float32)
    drift = steps[:, None, None] * last_vel[None, :, :] / frame_rate
    return sample.anchor[None, :, :] + drift / POS_SCALE
