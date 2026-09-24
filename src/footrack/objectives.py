"""L3 — tactical objectives and local refinement (ROLES.md §4.1).

Scores a candidate position for a player against what their role is actually
trying to achieve, then hill-climbs L2's output a short distance to improve it.

The weights are a coaching philosophy expressed as numbers, which makes them a
*feature*: different coaches get different weight profiles, and the same engine
produces a high-pressing shape or a low-block one depending on them.

Refinement is deliberately local. This layer improves L1+L2's proposal rather
than searching the pitch, because the constraint layer encodes principles the
optimiser has no way to rediscover and every metre it wanders is a metre a coach
has to argue with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footrack import pitch
from footrack.roles import Role
from footrack.state import Phase

# Ordinal 0-3 as in ROLES.md §4.1: 0 ignore, 1 minor, 2 important, 3 primary.
# Collapsed from role-plus-modifier to base roles, since modifiers come from the
# coach and are not inferable from data.
IN_POSSESSION: dict[Role, dict[str, float]] = {
    Role.GK: {"width": 0, "depth": 0, "progression": 1, "space": 0},
    Role.CB: {"width": 0, "depth": 0, "progression": 2, "space": 1},
    Role.FB: {"width": 2.5, "depth": 1.5, "progression": 1.5, "space": 1.5},
    Role.DM: {"width": 0, "depth": 0, "progression": 2.5, "space": 2},
    Role.CM: {"width": 1, "depth": 1.5, "progression": 2, "space": 2},
    Role.AM: {"width": 0.5, "depth": 2, "progression": 3, "space": 3},
    Role.W: {"width": 2, "depth": 2.5, "progression": 1.5, "space": 2},
    Role.ST: {"width": 0, "depth": 2.5, "progression": 1.5, "space": 1},
    Role.UNKNOWN: {"width": 1, "depth": 1, "progression": 1.5, "space": 1.5},
}

OUT_OF_POSSESSION: dict[Role, dict[str, float]] = {
    Role.GK: {"pressure": 0, "lanes": 0, "goal": 3, "compact": 0},
    Role.CB: {"pressure": 2, "lanes": 2, "goal": 3, "compact": 2},
    Role.FB: {"pressure": 2, "lanes": 2.5, "goal": 1.5, "compact": 2.5},
    Role.DM: {"pressure": 1, "lanes": 2.5, "goal": 2.5, "compact": 3},
    Role.CM: {"pressure": 3, "lanes": 2.5, "goal": 2, "compact": 2.5},
    Role.AM: {"pressure": 1.5, "lanes": 2, "goal": 1, "compact": 1.5},
    Role.W: {"pressure": 1, "lanes": 2, "goal": 0, "compact": 1},
    Role.ST: {"pressure": 2, "lanes": 1.5, "goal": 0, "compact": 0.5},
    Role.UNKNOWN: {"pressure": 1.5, "lanes": 2, "goal": 1.5, "compact": 1.5},
}


@dataclass
class Scene:
    """The state a candidate position is scored against."""

    ball_xy: np.ndarray | None
    teammates: np.ndarray  # (11, 2) metres — current proposal for the whole team
    opponents: np.ndarray | None  # (N, 2) metres
    phase: Phase


def _nearest(xy: np.ndarray, others: np.ndarray | None, skip: int | None = None) -> float:
    if others is None or len(others) == 0:
        return pitch.PITCH_LENGTH
    d = np.linalg.norm(others - xy, axis=-1)
    if skip is not None and 0 <= skip < len(d):
        d = np.delete(d, skip)
    return float(d.min()) if len(d) else pitch.PITCH_LENGTH


def _point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.clip(float((p - a) @ ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def score(xy: np.ndarray, slot: int, role: Role, scene: Scene) -> float:
    """Weighted objective value for placing ``slot`` at ``xy``. Higher is better.

    Each term is normalised to roughly [0, 1] so the ordinal weights compose
    predictably; without that, whichever term happened to have the largest raw
    range would dominate regardless of its weight.
    """
    total = 0.0

    if scene.phase is Phase.OUT_OF_POSSESSION:
        w = OUT_OF_POSSESSION.get(role, OUT_OF_POSSESSION[Role.UNKNOWN])

        if scene.ball_xy is not None and w["pressure"]:
            d = float(np.linalg.norm(xy - scene.ball_xy))
            total += w["pressure"] * np.exp(-d / 18.0)

        if scene.ball_xy is not None and w["goal"]:
            # Between the ball and our own goal, close to that line.
            dist = _point_to_segment(xy, scene.ball_xy, pitch.OWN_GOAL)
            total += w["goal"] * np.exp(-dist / 14.0)

        if scene.opponents is not None and scene.ball_xy is not None and w["lanes"]:
            # Block passing lanes from the ball to opponents ahead of it.
            blocked = 0.0
            for opp in scene.opponents:
                if opp[0] <= scene.ball_xy[0]:
                    continue
                blocked += np.exp(-_point_to_segment(xy, scene.ball_xy, opp) / 6.0)
            total += w["lanes"] * min(blocked / 3.0, 1.0)

        if w["compact"]:
            centroid = scene.teammates.mean(axis=0)
            total += w["compact"] * np.exp(-float(np.linalg.norm(xy - centroid)) / 30.0)

    else:
        w = IN_POSSESSION.get(role, IN_POSSESSION[Role.UNKNOWN])

        if w["width"]:
            total += w["width"] * (abs(float(xy[1])) / pitch.HALF_WIDTH)

        if w["depth"]:
            # Reward getting beyond the opposition's deepest outfield defender.
            if scene.opponents is not None and len(scene.opponents):
                line = float(np.sort(scene.opponents[:, 0])[min(1, len(scene.opponents) - 1)])
            else:
                line = 0.0
            total += w["depth"] * float(np.clip((xy[0] - line) / 30.0 + 0.5, 0.0, 1.0))

        if scene.ball_xy is not None and w["progression"]:
            # Available for a forward pass: ahead of the ball, within range.
            ahead = float(xy[0]) > float(scene.ball_xy[0])
            d = float(np.linalg.norm(xy - scene.ball_xy))
            total += w["progression"] * (np.exp(-abs(d - 22.0) / 14.0) if ahead else 0.0)

        if w["space"]:
            free = min(
                _nearest(xy, scene.teammates, skip=slot),
                _nearest(xy, scene.opponents),
            )
            total += w["space"] * float(np.clip(free / 15.0, 0.0, 1.0))

    return total


def refine(
    targets: np.ndarray,
    roles: tuple[Role, ...],
    scene: Scene,
    radius_m: float = 6.0,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Hill-climb each target a short distance. Returns positions and gain.

    Candidates are sampled on two rings around the current proposal rather than
    over the pitch: L1 and L2 encode shape and principles that this layer cannot
    rediscover, so its job is to improve their answer locally, not replace it.
    """
    out = targets.astype(np.float32).copy()
    gain = np.zeros(len(out), dtype=np.float32)

    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    offsets = np.array(
        [[0.0, 0.0]]
        + [[r * np.cos(a), r * np.sin(a)] for r in (radius_m / 2, radius_m) for a in angles],
        dtype=np.float32,
    )

    for _ in range(iterations):
        scene.teammates = out
        for i, role in enumerate(roles):
            if role is Role.GK:
                continue
            candidates = out[i] + offsets
            candidates[:, 0] = np.clip(candidates[:, 0], -pitch.HALF_LENGTH, pitch.HALF_LENGTH)
            candidates[:, 1] = np.clip(candidates[:, 1], -pitch.HALF_WIDTH, pitch.HALF_WIDTH)
            values = np.array([score(c, i, role, scene) for c in candidates])
            best = int(values.argmax())
            if best != 0:
                gain[i] += float(values[best] - values[0])
                out[i] = candidates[best]

    return out, gain
