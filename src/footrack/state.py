"""Canonical state representation.

A :class:`Sequence` is a contiguous run of frames from one match period with a
**stable slot assignment**: entity ``i`` at time ``t`` is the same player as
entity ``i`` at time ``t-1``. Sequences are cut at period boundaries and at
substitutions, which is what makes velocities and temporal attention meaningful.

Slot layout is fixed at 23 entities::

    0        ball
    1..11    team A  (the advised team; always attacking +x)
    12..22   team B  (the opponent)

Positions are metres, centre-origin (see :mod:`footrack.pitch`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Sequence as TypingSequence

import numpy as np

N_ENTITIES = 23
BALL = 0
TEAM_A = slice(1, 12)
TEAM_B = slice(12, 23)


class Phase(IntEnum):
    """Phase of play, from the advised team's perspective.

    Part of the canonical state rather than a derived feature: the architecture
    puts phase classification in L0 (MODEL.md §L0), because every layer above it
    conditions on knowing which phase it is in.
    """

    UNKNOWN = 0
    IN_POSSESSION = 1
    OUT_OF_POSSESSION = 2
    TRANSITION = 3
    SET_PIECE = 4


N_PHASES = len(Phase)


@dataclass(frozen=True)
class SlotMap:
    """Which real player occupies each slot.

    Needed to join per-player attributes (physical capability, tendency
    residuals) onto the tensor, and to resolve which player a coach dragged when
    overriding a mark.
    """

    player_ids: tuple[str | None, ...]
    jersey_numbers: tuple[int | None, ...]
    team_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        for field in (self.player_ids, self.jersey_numbers, self.team_ids):
            if len(field) != N_ENTITIES:
                raise ValueError(f"SlotMap fields must have {N_ENTITIES} entries")

    def slot_of(self, player_id: str) -> int | None:
        try:
            return self.player_ids.index(player_id)
        except ValueError:
            return None

    def swapped(self) -> SlotMap:
        """Team A and team B slot blocks exchanged."""

        def swap(seq: tuple) -> tuple:
            return (seq[0], *seq[12:23], *seq[1:12])

        return SlotMap(
            player_ids=swap(self.player_ids),
            jersey_numbers=swap(self.jersey_numbers),
            team_ids=swap(self.team_ids),
        )


@dataclass(frozen=True)
class Sequence:
    """A contiguous run of frames with stable slots."""

    positions: np.ndarray  # (T, 23, 2) float32, metres, centre-origin
    velocities: np.ndarray  # (T, 23, 2) float32, m/s
    velocity_valid: np.ndarray  # (T, 23) bool — velocity present AND trustworthy
    mask: np.ndarray  # (T, 23) bool — entity present and valid
    frame_ids: np.ndarray  # (T,) int64
    frame_rate: float
    slots: SlotMap
    match_id: str
    period_id: int
    ball_owning_slot_block: np.ndarray | None = None  # (T,) int8: 0=A, 1=B, -1=unknown
    phase: np.ndarray | None = None  # (T,) int8 of Phase; None when undetermined

    def __post_init__(self) -> None:
        t = self.positions.shape[0]
        expected = {
            "positions": (t, N_ENTITIES, 2),
            "velocities": (t, N_ENTITIES, 2),
            "velocity_valid": (t, N_ENTITIES),
            "mask": (t, N_ENTITIES),
            "frame_ids": (t,),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")

    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def duration_s(self) -> float:
        return len(self) / self.frame_rate

    # -- transforms -----------------------------------------------------------

    def mirror(self) -> Sequence:
        """Reflect about the long axis (y -> -y).

        Free 2x augmentation: football is left/right symmetric, so a mirrored
        sequence is an equally valid sample.
        """
        pos = self.positions.copy()
        vel = self.velocities.copy()
        pos[..., 1] *= -1
        vel[..., 1] *= -1
        return replace(self, positions=pos, velocities=vel)

    def swap_perspective(self) -> Sequence:
        """View the same frames from the opponent's side.

        Rotates the pitch 180 degrees so team B now attacks +x, and exchanges the
        two team slot blocks. Another 2x of augmentation, and the mechanism by
        which one model advises either team.
        """
        pos = -self.positions
        vel = -self.velocities

        def swap_entities(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            out[:, 1:12] = arr[:, 12:23]
            out[:, 12:23] = arr[:, 1:12]
            return out

        owning = self.ball_owning_slot_block
        if owning is not None:
            owning = np.where(owning < 0, owning, 1 - owning).astype(np.int8)

        # Phase is defined from the advised team's perspective, so in- and
        # out-of-possession exchange. Transition, set piece and unknown are
        # symmetric and stay put.
        phase = self.phase
        if phase is not None:
            phase = np.select(
                [phase == Phase.IN_POSSESSION, phase == Phase.OUT_OF_POSSESSION],
                [Phase.OUT_OF_POSSESSION, Phase.IN_POSSESSION],
                default=phase,
            ).astype(np.int8)

        return replace(
            self,
            positions=swap_entities(pos),
            velocities=swap_entities(vel),
            velocity_valid=swap_entities(self.velocity_valid),
            mask=swap_entities(self.mask),
            slots=self.slots.swapped(),
            ball_owning_slot_block=owning,
            phase=phase,
        )

    def window(self, start: int, length: int) -> Sequence:
        """Extract a contiguous sub-sequence."""
        if start < 0 or start + length > len(self):
            raise IndexError(f"window [{start}, {start + length}) outside sequence of {len(self)}")
        stop = start + length
        owning = self.ball_owning_slot_block
        return replace(
            self,
            positions=self.positions[start:stop],
            velocities=self.velocities[start:stop],
            velocity_valid=self.velocity_valid[start:stop],
            mask=self.mask[start:stop],
            frame_ids=self.frame_ids[start:stop],
            ball_owning_slot_block=None if owning is None else owning[start:stop],
            phase=None if self.phase is None else self.phase[start:stop],
        )

    def downsample(self, factor: int) -> Sequence:
        """Keep every ``factor``-th frame.

        Velocities are computed at native rate before downsampling, so they stay
        accurate — differencing already-sparse frames would not.
        """
        if factor < 1:
            raise ValueError("factor must be >= 1")
        owning = self.ball_owning_slot_block
        return replace(
            self,
            positions=self.positions[::factor],
            velocities=self.velocities[::factor],
            velocity_valid=self.velocity_valid[::factor],
            mask=self.mask[::factor],
            frame_ids=self.frame_ids[::factor],
            frame_rate=self.frame_rate / factor,
            ball_owning_slot_block=None if owning is None else owning[::factor],
            phase=None if self.phase is None else self.phase[::factor],
        )

    # -- convenience ----------------------------------------------------------

    @property
    def ball(self) -> np.ndarray:
        """(T, 2) ball positions."""
        return self.positions[:, BALL]

    @property
    def team_a(self) -> np.ndarray:
        """(T, 11, 2) advised-team positions."""
        return self.positions[:, TEAM_A]

    @property
    def team_b(self) -> np.ndarray:
        """(T, 11, 2) opponent positions."""
        return self.positions[:, TEAM_B]

    def coverage(self) -> float:
        """Fraction of entity-frames that are present.

        1.0 for full optical tracking; expect 0.7-0.85 for broadcast-derived
        sources, where off-camera players are missing. Those gaps are the
        training signal for masked agent modelling, not a defect.
        """
        return float(self.mask.mean())

    def velocity_coverage(self) -> float:
        """Fraction of entity-frames with a trustworthy velocity.

        Lower than :meth:`coverage` by the amount lost to teleports in the
        source data — restarts, set pieces, identity swaps.
        """
        return float(self.velocity_valid.mean())

    def occupied_slots(self) -> int:
        """Number of slots with a real player assigned (ball excluded)."""
        return sum(pid is not None for pid in self.slots.player_ids[1:])

    def speeds(self) -> np.ndarray:
        """(T, 23) speed in m/s."""
        return np.linalg.norm(self.velocities, axis=-1)


def concat_stats(sequences: TypingSequence[Sequence]) -> dict[str, float]:
    """Summary over a collection of sequences, for sanity-checking a load."""
    if not sequences:
        return {}
    frames = sum(len(s) for s in sequences)
    speeds = np.concatenate([s.speeds()[s.velocity_valid] for s in sequences])
    return {
        "sequences": len(sequences),
        "frames": frames,
        "minutes": frames / sequences[0].frame_rate / 60,
        "mean_coverage": float(np.mean([s.coverage() for s in sequences])),
        "mean_velocity_coverage": float(np.mean([s.velocity_coverage() for s in sequences])),
        "median_len": float(np.median([len(s) for s in sequences])),
        "speed_p50": float(np.percentile(speeds, 50)),
        "speed_p99": float(np.percentile(speeds, 99)),
        "speed_max": float(speeds.max()),
    }
