"""Sequence -> model input tensors (MODEL.md §4).

Features are computed on demand rather than stored. Raw positions for the whole
corpus are under a gigabyte; the derived feature set is an order of magnitude
larger, so it belongs in the dataloader, not on disk.

Two things are deliberately *not* here:

- **Role and tendency embeddings.** Those are learned lookups. This module emits
  integer indices; the model owns the embedding tables.
- **Data-derived normalisation.** Every scale below is a fixed constant. Deriving
  them from the training set leaks statistics and silently mismatches at
  inference, when a single live frame has no distribution to normalise against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footrack import pitch
from footrack.state import BALL, N_ENTITIES, N_PHASES, TEAM_A, TEAM_B, Phase, Sequence

# Fixed normalisation constants. Never derive these from data.
POS_SCALE = pitch.HALF_LENGTH  # 52.5 m -> 1.0, isotropic
SPEED_SCALE = 10.0  # m/s; ~p99.99 of player speed
DIST_SCALE = pitch.HALF_LENGTH


@dataclass(frozen=True)
class PlayerProfile:
    """Physical capability for one player (MODEL.md §5.2).

    Enters the model twice: as an input to prediction, since a quick full-back
    changes which futures are reachable, and as a constraint on output, so the
    system does not demand a run the player cannot make.
    """

    max_speed: float = 8.0  # m/s
    max_accel: float = 4.0  # m/s^2
    stamina: float = 0.5  # 0-1 index

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.max_speed / SPEED_SCALE, self.max_accel / 8.0, self.stamina],
            dtype=np.float32,
        )


DEFAULT_PROFILE = PlayerProfile()
N_PROFILE_DIMS = 3


ENTITY_FEATURES: tuple[str, ...] = (
    "pos_x",
    "pos_y",
    "vel_x",
    "vel_y",
    "speed",
    "vel_dir_cos",
    "vel_dir_sin",
    "dist_ball",
    "bear_ball_cos",
    "bear_ball_sin",
    "dist_own_goal",
    "bear_own_goal_cos",
    "bear_own_goal_sin",
    "dist_opp_goal",
    "bear_opp_goal_cos",
    "bear_opp_goal_sin",
    "orient_cos",
    "orient_sin",
    "orient_valid",
    "is_ball",
    "is_own_team",
    "is_opponent",
    "present",
    "velocity_valid",
    "prof_max_speed",
    "prof_max_accel",
    "prof_stamina",
)
N_ENTITY_FEATURES = len(ENTITY_FEATURES)

CONTEXT_FEATURES: tuple[str, ...] = (
    "phase_unknown",
    "phase_in_poss",
    "phase_out_poss",
    "phase_transition",
    "phase_set_piece",
    "ball_own_unknown",
    "ball_own_us",
    "ball_own_them",
    "period_norm",
    "progress",
)
N_CONTEXT_FEATURES = len(CONTEXT_FEATURES)


@dataclass
class ModelInput:
    """What the encoder consumes."""

    entity: np.ndarray  # (T, 23, N_ENTITY_FEATURES) float32
    context: np.ndarray  # (T, N_CONTEXT_FEATURES) float32
    role_ids: np.ndarray  # (23,) int32 — index into the role table
    player_ids: np.ndarray  # (23,) int32 — index into the tendency table, -1 unknown
    mask: np.ndarray  # (T, 23) bool
    feature_names: tuple[str, ...] = field(default=ENTITY_FEATURES, repr=False)

    def __len__(self) -> int:
        return self.entity.shape[0]

    def column(self, name: str) -> np.ndarray:
        """(T, 23) view of one named feature. For debugging and plotting."""
        return self.entity[..., self.feature_names.index(name)]


def _unit(delta: np.ndarray) -> np.ndarray:
    """Normalise displacement vectors to unit length, zero where degenerate.

    Directions are emitted as (cos, sin) pairs rather than angles throughout:
    raw radians have a discontinuity at +/-pi that a model cannot represent
    smoothly.
    """
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-6)


def build_context(seq: Sequence) -> np.ndarray:
    """(T, N_CONTEXT_FEATURES) global features.

    Phase is read from the sequence rather than recomputed: classifying it is an
    L0 responsibility (see :func:`footrack.sources.derive_phase`), and doing it
    here would mean the feature builder and the state layer could disagree.
    """
    t = len(seq)
    phase = seq.phase if seq.phase is not None else np.full(t, Phase.UNKNOWN, np.int8)

    ctx = np.zeros((t, N_CONTEXT_FEATURES), dtype=np.float32)
    ctx[np.arange(t), phase] = 1.0  # phase one-hot occupies columns 0..N_PHASES-1

    owning = seq.ball_owning_slot_block
    base = N_PHASES
    if owning is None:
        ctx[:, base] = 1.0
    else:
        ctx[:, base] = owning < 0
        ctx[:, base + 1] = owning == 0
        ctx[:, base + 2] = owning == 1

    ctx[:, base + 3] = (seq.period_id - 1) / 2.0
    ctx[:, base + 4] = np.linspace(0.0, 1.0, t, dtype=np.float32)
    return ctx


def build_entity_features(
    seq: Sequence,
    profiles: dict[str, PlayerProfile] | None = None,
) -> np.ndarray:
    """(T, 23, N_ENTITY_FEATURES) per-entity features."""
    t = len(seq)
    out = np.zeros((t, N_ENTITIES, N_ENTITY_FEATURES), dtype=np.float32)

    pos = seq.positions
    vel = seq.velocities
    mask = seq.mask
    vvalid = seq.velocity_valid

    out[..., 0:2] = pos / POS_SCALE
    out[..., 2:4] = vel / SPEED_SCALE

    speed = np.linalg.norm(vel, axis=-1)
    out[..., 4] = speed / SPEED_SCALE
    out[..., 5:7] = _unit(vel)

    ball = pos[:, BALL : BALL + 1, :]
    to_ball = ball - pos
    out[..., 7] = np.linalg.norm(to_ball, axis=-1) / DIST_SCALE
    out[..., 8:10] = _unit(to_ball)

    to_own = pitch.OWN_GOAL[None, None, :] - pos
    out[..., 10] = np.linalg.norm(to_own, axis=-1) / DIST_SCALE
    out[..., 11:13] = _unit(to_own)

    to_opp = pitch.OPP_GOAL[None, None, :] - pos
    out[..., 13] = np.linalg.norm(to_opp, axis=-1) / DIST_SCALE
    out[..., 14:16] = _unit(to_opp)

    # Body orientation: reserved slot, zero-filled (MODEL.md §7). No public
    # tracking dataset carries it. Velocity direction above is the ~80% proxy
    # while a player is moving; the residual matters most when they are not.
    # Columns 16..18 stay zero until a pose source exists.

    # Zero the observation-derived block (0..18) where the entity is absent, and
    # the velocity-derived block (2..6) where its velocity is untrustworthy.
    # Done before the structural features below, which must survive both.
    out[..., 0:19][~mask] = 0.0
    out[..., 2:7][~vvalid] = 0.0

    # Structural features: properties of the slot, not of the frame, so they are
    # set unconditionally. A masked entity must still be identifiable as an
    # opponent with a given physical profile — otherwise masked agent modelling
    # has nothing to condition on (MODEL.md task 2), and the same applies to
    # manual override, which is that head run in reverse.
    out[:, BALL, 19] = 1.0
    out[:, TEAM_A, 20] = 1.0
    out[:, TEAM_B, 21] = 1.0
    out[..., 22] = mask
    out[..., 23] = vvalid

    prof = np.tile(DEFAULT_PROFILE.as_array(), (N_ENTITIES, 1))
    if profiles:
        for slot, pid in enumerate(seq.slots.player_ids):
            if pid is not None and pid in profiles:
                prof[slot] = profiles[pid].as_array()
    prof[BALL] = 0.0
    out[..., 24:27] = prof[None, :, :]

    return out


def build(
    seq: Sequence,
    profiles: dict[str, PlayerProfile] | None = None,
    role_ids: np.ndarray | None = None,
    player_index: dict[str, int] | None = None,
) -> ModelInput:
    """Assemble the full model input for a sequence.

    ``role_ids`` and ``player_index`` are lookups into learned embedding tables.
    Both default to "unknown", which is the correct cold-start behaviour: an
    unseen player falls back to their role, and an unassigned role falls back to
    a shared default (MODEL.md §5.2).
    """
    if role_ids is None:
        role_ids = np.zeros(N_ENTITIES, dtype=np.int32)

    pid_idx = np.full(N_ENTITIES, -1, dtype=np.int32)
    if player_index:
        for slot, pid in enumerate(seq.slots.player_ids):
            if pid is not None and pid in player_index:
                pid_idx[slot] = player_index[pid]

    return ModelInput(
        entity=build_entity_features(seq, profiles),
        context=build_context(seq),
        role_ids=role_ids.astype(np.int32),
        player_ids=pid_idx,
        mask=seq.mask.copy(),
    )


def estimate_profiles(
    sequences: list[Sequence], speed_pct: float = 99.5, accel_pct: float = 99.0
) -> dict[str, PlayerProfile]:
    """Physical profiles from observed tracking.

    Percentiles rather than maxima: the single fastest frame is usually a
    tracking artefact, not an athletic ceiling.

    **Do not estimate these from the same windows you train on.** Derive them
    from separate matches, or from GPS-vest exports, which is what the club's
    existing hardware is for — and needs no live access (PLAN.md §3.3).
    """
    speeds: dict[str, list[np.ndarray]] = {}
    accels: dict[str, list[np.ndarray]] = {}

    for seq in sequences:
        sp = seq.speeds()
        acc = np.abs(np.gradient(sp, 1.0 / seq.frame_rate, axis=0))
        for slot, pid in enumerate(seq.slots.player_ids):
            if pid is None or slot == BALL:
                continue
            valid = seq.velocity_valid[:, slot]
            if not valid.any():
                continue
            speeds.setdefault(pid, []).append(sp[valid, slot])
            accels.setdefault(pid, []).append(acc[valid, slot])

    profiles: dict[str, PlayerProfile] = {}
    for pid, chunks in speeds.items():
        s = np.concatenate(chunks)
        a = np.concatenate(accels[pid])
        if s.size < 100:
            continue
        top = float(np.percentile(s, speed_pct))
        profiles[pid] = PlayerProfile(
            max_speed=top,
            max_accel=float(np.percentile(a, accel_pct)),
            # Crude stand-in: fraction of time above 70% of top speed. A real
            # stamina index needs load across a session, which is exactly what
            # GPS vests already measure.
            stamina=float((s > 0.7 * top).mean()),
        )
    return profiles
