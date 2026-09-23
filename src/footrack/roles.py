"""Role assignment (ROLES.md §3).

The taxonomy defines eight base roles; this assigns players to them. Without it
``role_ids`` stays at zero for every player, so the model is asked to place a
hidden player knowing only their team and physical profile — which is why its
reconstructions collapsed onto a single lateral line. Depth correlates with
max speed (keepers are slow), channel does not.

Assignment here is **inferred from average position**, because public datasets
carry no role labels. At the club, a coach assigns roles directly and this is
not used; treat it as a stand-in for data where nobody has told us who plays
where.

Only the eight base roles are inferred. Modifiers (ball-playing vs stopper,
inverted vs touchline) express intent rather than average position and cannot be
recovered this way — they come from the coach.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from footrack.state import BALL, Sequence, TEAM_A


class Role(IntEnum):
    """The eight base roles of ROLES.md §3, plus a fallback."""

    UNKNOWN = 0
    GK = 1
    CB = 2
    FB = 3
    DM = 4
    CM = 5
    AM = 6
    W = 7
    ST = 8


N_BASE_ROLES = len(Role)

# Thresholds in metres from the player's own goal line, so they apply to both
# teams. Calibrated from **observed match averages**, not from the reference
# positions in ROLES.md §6. Those describe where a role sits in possession; a
# season average is far more compressed, because a full-back who both defends
# and attacks averages out near mid-pitch. Measured over Metrica, regular
# outfield players span only 37-67m, against the 26-78m the reference positions
# would imply — using the latter classified every outfielder as a midfielder.
DEF_LINE = 46.0  # roughly the 40th percentile of outfield depth
MID_LINE = 56.0  # roughly the 75th
DM_LINE = 50.0  # within midfield, deeper half
ST_LINE = 61.0  # within the forward band, the most advanced

GK_MAX_DEPTH = 25.0
GK_MAX_SPEED = 7.0

# Width is judged on the 85th percentile of |y|, not the mean. Mean |y| barely
# separates anyone — p25 to p75 is 9.2m to 15.8m — because wide players drift
# infield constantly. The 85th percentile captures how wide they actually get,
# and spreads 17m to 26m over the same players.
WIDE_P85 = 22.0
WIDTH_PERCENTILE = 85


def _own_half_depth(positions: np.ndarray, is_team_a: bool) -> np.ndarray:
    """Distance up the pitch from the player's OWN goal, in metres.

    Sequences are stored with team A attacking +x, so team B's perspective is
    mirrored. Without this flip every opponent would be classified as if they
    were attacking the wrong way and defenders would read as forwards.
    """
    x = positions[..., 0] if is_team_a else -positions[..., 0]
    return x + 52.5


def _classify(depth: float, width: float, max_speed: float) -> Role:
    """Base role from mean depth, 85th-percentile width, and top speed."""
    if depth < GK_MAX_DEPTH and max_speed < GK_MAX_SPEED:
        return Role.GK

    wide = width >= WIDE_P85
    if depth < DEF_LINE:
        return Role.FB if wide else Role.CB
    if depth < MID_LINE:
        if wide:
            return Role.W
        return Role.DM if depth < DM_LINE else Role.CM
    if wide:
        return Role.W
    return Role.ST if depth >= ST_LINE else Role.AM


def infer_roles(sequences: list[Sequence]) -> dict[str, int]:
    """Map each player id to a base role index, from their average position."""
    depth: dict[str, list[np.ndarray]] = {}
    lateral: dict[str, list[np.ndarray]] = {}
    speed: dict[str, list[np.ndarray]] = {}

    for seq in sequences:
        speeds = seq.speeds()
        for slot, pid in enumerate(seq.slots.player_ids):
            if pid is None or slot == BALL:
                continue
            valid = seq.mask[:, slot]
            if not valid.any():
                continue
            is_a = slot < TEAM_A.stop
            depth.setdefault(pid, []).append(
                _own_half_depth(seq.positions[valid, slot], is_a)
            )
            lateral.setdefault(pid, []).append(np.abs(seq.positions[valid, slot, 1]))
            vvalid = seq.velocity_valid[:, slot]
            if vvalid.any():
                speed.setdefault(pid, []).append(speeds[vvalid, slot])

    roles: dict[str, int] = {}
    for pid, chunks in depth.items():
        d = float(np.mean(np.concatenate(chunks)))
        width = float(np.percentile(np.concatenate(lateral[pid]), WIDTH_PERCENTILE))
        s = speed.get(pid)
        top = float(np.percentile(np.concatenate(s), 99.5)) if s else 8.0
        roles[pid] = int(_classify(d, width, top))
    return roles


def role_id_array(slots, roles: dict[str, int]) -> np.ndarray:
    """(23,) role index per slot, UNKNOWN where unassigned."""
    out = np.full(23, Role.UNKNOWN, dtype=np.int32)
    for slot, pid in enumerate(slots.player_ids):
        if pid is not None and pid in roles:
            out[slot] = roles[pid]
    return out


def build_player_index(sequences: list[Sequence]) -> dict[str, int]:
    """Stable integer id per player, for the tendency embedding table.

    Sorted so the mapping is reproducible across runs — a shifting index would
    silently attach one player's learned tendency to another.
    """
    ids = sorted({
        pid for seq in sequences for pid in seq.slots.player_ids if pid is not None
    })
    return {pid: i for i, pid in enumerate(ids)}


def summarise(sequences: list[Sequence], roles: dict[str, int]) -> dict[str, int]:
    """Count players per role, for sanity-checking an inference."""
    counts: dict[str, int] = {}
    for pid, role in roles.items():
        counts[Role(role).name] = counts.get(Role(role).name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
