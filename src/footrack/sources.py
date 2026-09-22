"""Tracking sources.

Every source resolves to the same :class:`~footrack.state.Sequence` objects, so
replay files, live LPS, camera CV and any future provider are interchangeable
behind one interface (PLAN.md §L0). Today only kloppy is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter

from footrack import pitch
from footrack.state import BALL, N_ENTITIES, Phase, Sequence, SlotMap

# Frames either side of a possession change that count as transition.
TRANSITION_WINDOW_S = 3.0

# How far ahead to look when deciding a segment's slot assignment. Long enough
# to see past broadcast gaps, short enough not to swallow a substitution.
_LOOKAHEAD_FRAMES = 250

# Savitzky-Golay parameters for velocity. Raw differencing of tracking data is
# dominated by measurement noise; SG smooths and differentiates in one pass and
# is the standard treatment in the tracking literature.
_SG_WINDOW_S = 0.5
_SG_POLYORDER = 2

# Physical plausibility ceilings, m/s. Steps implying more than these are
# teleports, not motion: tracking data jumps the ball at restarts and set
# pieces, and occasionally swaps player identities. The positions either side
# are real observations — it is the velocity *across* the jump that is
# meaningless, so it gets zeroed and flagged rather than smoothed over.
MAX_PLAYER_SPEED = 12.0  # world-record sprint is ~12.4 m/s
MAX_BALL_SPEED = 50.0  # hardest recorded shots are ~38 m/s


class TrackingSource(ABC):
    """Anything that can produce canonical sequences."""

    @abstractmethod
    def load(self) -> list[Sequence]:
        """Return all sequences available from this source."""


def derive_phase(
    owning: np.ndarray | None,
    set_piece: np.ndarray | None,
    frame_rate: float,
    length: int,
) -> np.ndarray:
    """(T,) int8 phase from possession, with hysteresis at turnovers.

    Naive frame-wise classification flickers badly at turnovers (MODEL.md §L0),
    so a window either side of each possession change is labelled TRANSITION
    rather than flipping instantly between possession states. Set pieces take
    precedence, since a restart is a restart regardless of who holds the ball.
    """
    if owning is None:
        phase = np.full(length, Phase.UNKNOWN, dtype=np.int8)
    else:
        phase = np.select(
            [owning == 0, owning == 1],
            [Phase.IN_POSSESSION, Phase.OUT_OF_POSSESSION],
            default=Phase.UNKNOWN,
        ).astype(np.int8)

        known = owning >= 0
        changed = np.zeros(length, dtype=bool)
        changed[1:] = known[1:] & known[:-1] & (owning[1:] != owning[:-1])

        if changed.any():
            half = max(1, int(round(TRANSITION_WINDOW_S * frame_rate / 2)))
            spread = binary_dilation(changed, structure=np.ones(2 * half + 1, dtype=bool))
            phase[spread & known] = Phase.TRANSITION

    if set_piece is not None:
        phase[set_piece] = Phase.SET_PIECE

    return phase


def _velocities(
    positions: np.ndarray, mask: np.ndarray, frame_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed velocities in m/s and a validity mask, both shape (T, 23, ...).

    Gaps are linearly interpolated before differentiating so a missing stretch
    does not produce spurious spikes at its edges. Teleports — steps implying
    speeds above the physical ceilings — contaminate the whole smoothing window
    around them, so velocity is zeroed there and marked invalid. The returned
    mask is ``True`` only where velocity is both present and trustworthy.
    """
    t = positions.shape[0]
    filled = positions.copy()

    # Linear interpolation across gaps, per entity per axis.
    frames = np.arange(t)
    for entity in range(N_ENTITIES):
        valid = mask[:, entity]
        if valid.sum() < 2:
            continue
        if not valid.all():
            for axis in range(2):
                filled[:, entity, axis] = np.interp(
                    frames, frames[valid], positions[valid, entity, axis]
                )

    window = int(round(_SG_WINDOW_S * frame_rate))
    window = max(5, window | 1)  # odd, at least 5

    if t <= window or t <= _SG_POLYORDER + 1:
        # Too short to smooth; fall back to a plain gradient.
        vel = np.gradient(filled, 1.0 / frame_rate, axis=0)
    else:
        # mode='nearest' rather than the default 'interp': at sequence edges
        # 'interp' fits a polynomial to the boundary window and extrapolates its
        # derivative, which diverges wildly — it produced 164 m/s readings on
        # near-stationary players. Edge padding biases the boundary derivative
        # toward zero instead, which is conservative and stable.
        vel = savgol_filter(
            filled,
            window_length=window,
            polyorder=_SG_POLYORDER,
            deriv=1,
            delta=1.0 / frame_rate,
            axis=0,
            mode="nearest",
        )

    # Flag teleports and everything their smoothing window touches.
    ceilings = np.full(N_ENTITIES, MAX_PLAYER_SPEED, dtype=np.float32)
    ceilings[BALL] = MAX_BALL_SPEED
    max_step = ceilings / frame_rate

    step = np.linalg.norm(np.diff(filled, axis=0), axis=-1)  # (T-1, 23)
    jump = step > max_step[None, :]

    # A jump between t and t+1 invalidates both endpoints, then spreads over
    # the smoothing window.
    contaminated = np.zeros((t, N_ENTITIES), dtype=bool)
    contaminated[:-1] |= jump
    contaminated[1:] |= jump
    if contaminated.any():
        contaminated = binary_dilation(
            contaminated, structure=np.ones((window, 1), dtype=bool)
        )

    # The smoothing window is only partly populated at the sequence edges, so
    # those derivatives are unreliable regardless of padding mode. Cheap to drop
    # — a handful of frames out of tens of thousands.
    half = window // 2
    contaminated[:half] = True
    contaminated[-half:] = True

    velocity_valid = mask & ~contaminated
    vel[~velocity_valid] = 0.0
    return vel.astype(np.float32), velocity_valid


class KloppySource(TrackingSource):
    """Adapter over any kloppy tracking dataset.

    The dataset is re-oriented to ``STATIC_HOME_AWAY`` so the home team attacks
    +x in both halves, removing the half-time flip. Team A slots therefore always
    hold the home side; call :meth:`Sequence.swap_perspective` to advise the away
    team instead.
    """

    def __init__(
        self,
        dataset,
        match_id: str = "unknown",
        min_frames: int = 50,
        events=None,
    ):
        from kloppy.domain import Orientation

        if dataset.metadata.orientation != Orientation.STATIC_HOME_AWAY:
            dataset = dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)

        self.dataset = dataset
        self.match_id = match_id
        self.min_frames = min_frames
        self.events = events

        grounds = {t.team_id: str(t.ground) for t in dataset.metadata.teams}
        self._home_ids = {tid for tid, g in grounds.items() if g == "home"}

    def _is_home(self, team_id: str) -> bool:
        return team_id in self._home_ids

    def _assign_slots(self, frames) -> SlotMap:
        """Pick the 11 most-present players per team across a lookahead window."""
        counts: dict[str, Counter] = {"home": Counter(), "away": Counter()}
        meta: dict[str, tuple[int | None, str]] = {}

        for frame in frames:
            for player in frame.players_data:
                side = "home" if self._is_home(player.team.team_id) else "away"
                counts[side][player.player_id] += 1
                meta[player.player_id] = (player.jersey_no, player.team.team_id)

        player_ids: list[str | None] = [None] * N_ENTITIES
        jerseys: list[int | None] = [None] * N_ENTITIES
        team_ids: list[str | None] = [None] * N_ENTITIES

        for side, offset in (("home", 1), ("away", 12)):
            for i, (pid, _) in enumerate(counts[side].most_common(11)):
                jersey, team_id = meta[pid]
                player_ids[offset + i] = pid
                jerseys[offset + i] = jersey
                team_ids[offset + i] = team_id

        return SlotMap(tuple(player_ids), tuple(jerseys), tuple(team_ids))

    def load(self) -> list[Sequence]:
        by_period: dict[int, list] = {}
        for frame in self.dataset.records:
            by_period.setdefault(frame.period.id, []).append(frame)

        sequences: list[Sequence] = []
        for period_id, frames in sorted(by_period.items()):
            sequences.extend(self._load_period(period_id, frames))
        return sequences

    def _load_period(self, period_id: int, frames: list) -> list[Sequence]:
        sequences: list[Sequence] = []
        cursor = 0

        while cursor < len(frames):
            slots = self._assign_slots(frames[cursor : cursor + _LOOKAHEAD_FRAMES])
            slotted = {pid for pid in slots.player_ids if pid is not None}
            if not slotted:
                cursor += 1
                continue

            # Extend while every player on the pitch has a slot. A frame
            # introducing an unslotted player is a substitution: cut there, so
            # slot identity stays stable within a sequence.
            end = cursor
            while end < len(frames):
                present = {p.player_id for p in frames[end].players_data}
                if not present.issubset(slotted):
                    break
                end += 1

            segment = frames[cursor:end]
            if len(segment) >= self.min_frames:
                sequences.append(self._build(period_id, segment, slots))

            cursor = end if end > cursor else cursor + 1

        return sequences

    def _build(self, period_id: int, frames: list, slots: SlotMap) -> Sequence:
        t = len(frames)
        positions = np.zeros((t, N_ENTITIES, 2), dtype=np.float32)
        mask = np.zeros((t, N_ENTITIES), dtype=bool)
        frame_ids = np.zeros(t, dtype=np.int64)
        owning = np.full(t, -1, dtype=np.int8)

        slot_of = {pid: i for i, pid in enumerate(slots.player_ids) if pid is not None}

        for i, frame in enumerate(frames):
            frame_ids[i] = frame.frame_id

            if frame.ball_coordinates is not None:
                raw = np.array(
                    [frame.ball_coordinates.x, frame.ball_coordinates.y], dtype=np.float32
                )
                if np.isfinite(raw).all():
                    positions[i, 0] = pitch.from_kloppy(raw)
                    mask[i, 0] = True

            for player, data in frame.players_data.items():
                slot = slot_of.get(player.player_id)
                if slot is None or data.coordinates is None:
                    continue
                raw = np.array([data.coordinates.x, data.coordinates.y], dtype=np.float32)
                if not np.isfinite(raw).all():
                    continue
                positions[i, slot] = pitch.from_kloppy(raw)
                mask[i, slot] = True

            team = frame.ball_owning_team
            if team is not None:
                owning[i] = 0 if self._is_home(team.team_id) else 1

        frame_rate = float(self.dataset.metadata.frame_rate)
        velocities, velocity_valid = _velocities(positions, mask, frame_rate)

        # An event feed, where available, supplies possession that tracking
        # alone cannot. Team A is the home side by construction here.
        set_piece = None
        if self.events is not None:
            from footrack.events import possession_series, set_piece_series

            owning = possession_series(self.events, frame_ids, home_is_team_a=True)
            set_piece = set_piece_series(self.events, frame_ids, frame_rate)

        phase = derive_phase(
            owning if (owning >= 0).any() else None, set_piece, frame_rate, t
        )

        return Sequence(
            positions=positions,
            velocities=velocities,
            velocity_valid=velocity_valid,
            mask=mask,
            frame_ids=frame_ids,
            frame_rate=frame_rate,
            slots=slots,
            match_id=self.match_id,
            period_id=period_id,
            ball_owning_slot_block=owning if (owning >= 0).any() else None,
            phase=phase,
        )


def load_metrica(
    match_id: int = 1, limit: int | None = None, with_events: bool = True
) -> list[Sequence]:
    """Convenience loader for Metrica open data — the cleanest starting point.

    ``with_events`` fetches the matching event feed to derive possession and
    phase. Without it every frame is phase UNKNOWN, because kloppy's Metrica
    tracking loader carries no ball ownership.
    """
    from kloppy import metrica

    kwargs = {"match_id": match_id}
    if limit is not None:
        kwargs["limit"] = limit
    dataset = metrica.load_open_data(**kwargs)

    events = None
    if with_events:
        from footrack.events import load_metrica_events

        try:
            events = load_metrica_events(match_id)
        except Exception as exc:  # network or unavailable match
            print(f"warning: event feed unavailable ({exc}); phase will be UNKNOWN")

    return KloppySource(dataset, match_id=f"metrica_{match_id}", events=events).load()
