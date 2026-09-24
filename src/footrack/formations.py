"""L1 — formation templates and ball-oriented deformation (ROLES.md §4.2, §6).

The floor of the stack. Pure geometry, no learning, no data: given a formation,
the ball's position and the phase of play, produce a target position for every
outfield slot.

This layer is never *wrong*, only unsubtle, and it is what the system falls back
to when everything above it is uncertain or unavailable (PLAN.md §L1). It is
also what gives the labelling tool something to propose — correcting a proposal
is far faster than placing players from a blank pitch, and disagreement is a
stronger signal than absolute placement.

Reference positions are stored in normalised [-1, 1] coordinates so one template
serves any pitch size; they convert to metres on use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footrack import pitch
from footrack.roles import Role
from footrack.state import Phase


@dataclass(frozen=True)
class BallResponse:
    """How a role's reference position moves with the ball (ROLES.md §4.2).

    ``bx``/``by`` are gains on the ball's normalised position: the whole block
    shifts with the ball rather than chasing it. ``asym`` is the extra pull
    toward the centre when the ball is on the far side — often the defining
    trait of a role, since an inverted winger is defined by what it does when
    the ball is away from it.
    """

    bx: float
    by: float
    asym: float


# ROLES.md §4.2, collapsed to base roles. Modifiers refine these; the coach
# supplies those, so the base values stand in until then.
BALL_RESPONSE: dict[Role, BallResponse] = {
    Role.GK: BallResponse(0.25, 0.12, 0.05),
    Role.CB: BallResponse(0.72, 0.32, 0.15),
    Role.FB: BallResponse(0.78, 0.50, 0.55),
    Role.DM: BallResponse(0.52, 0.55, 0.25),
    Role.CM: BallResponse(0.70, 0.68, 0.35),
    Role.AM: BallResponse(0.72, 0.58, 0.35),
    Role.W: BallResponse(0.78, 0.30, 0.70),
    Role.ST: BallResponse(0.82, 0.42, 0.25),
    Role.UNKNOWN: BallResponse(0.60, 0.45, 0.25),
}


@dataclass(frozen=True)
class Slot:
    """One position in a formation."""

    label: str
    role: Role
    ref_x: float  # normalised, -1 own goal line .. +1 opponent goal line
    ref_y: float  # normalised, -1 left touchline .. +1 right touchline


@dataclass(frozen=True)
class Formation:
    name: str
    slots: tuple[Slot, ...]

    def __post_init__(self) -> None:
        if len(self.slots) != 11:
            raise ValueError(f"{self.name} has {len(self.slots)} slots, expected 11")

    @property
    def roles(self) -> tuple[Role, ...]:
        return tuple(s.role for s in self.slots)

    def reference_metres(self) -> np.ndarray:
        """(11, 2) reference positions in metres, centre-origin."""
        return np.array(
            [[s.ref_x * pitch.HALF_LENGTH, s.ref_y * pitch.HALF_WIDTH] for s in self.slots],
            dtype=np.float32,
        )


# ROLES.md §6. Neutral state: own possession, ball at the centre circle.
FOUR_THREE_THREE = Formation(
    "4-3-3",
    (
        Slot("GK", Role.GK, -0.92, 0.00),
        Slot("LCB", Role.CB, -0.60, -0.18),
        Slot("RCB", Role.CB, -0.60, +0.18),
        Slot("LB", Role.FB, -0.50, -0.68),
        Slot("RB", Role.FB, -0.50, +0.68),
        Slot("DM", Role.DM, -0.32, 0.00),
        Slot("LCM", Role.CM, -0.10, -0.30),
        Slot("RCM", Role.CM, -0.10, +0.30),
        Slot("LW", Role.W, +0.30, -0.80),
        Slot("RW", Role.W, +0.30, +0.80),
        Slot("ST", Role.ST, +0.40, 0.00),
    ),
)

FOUR_FOUR_TWO = Formation(
    "4-4-2",
    (
        Slot("GK", Role.GK, -0.92, 0.00),
        Slot("LCB", Role.CB, -0.62, -0.18),
        Slot("RCB", Role.CB, -0.62, +0.18),
        Slot("LB", Role.FB, -0.52, -0.66),
        Slot("RB", Role.FB, -0.52, +0.66),
        Slot("LM", Role.W, -0.08, -0.72),
        Slot("RM", Role.W, -0.08, +0.72),
        Slot("LCM", Role.CM, -0.18, -0.22),
        Slot("RCM", Role.CM, -0.18, +0.22),
        Slot("LST", Role.ST, +0.34, -0.14),
        Slot("RST", Role.ST, +0.34, +0.14),
    ),
)

FOUR_TWO_THREE_ONE = Formation(
    "4-2-3-1",
    (
        Slot("GK", Role.GK, -0.92, 0.00),
        Slot("LCB", Role.CB, -0.62, -0.18),
        Slot("RCB", Role.CB, -0.62, +0.18),
        Slot("LB", Role.FB, -0.52, -0.68),
        Slot("RB", Role.FB, -0.52, +0.68),
        Slot("LDM", Role.DM, -0.34, -0.16),
        Slot("RDM", Role.DM, -0.34, +0.16),
        Slot("AM", Role.AM, -0.02, 0.00),
        Slot("LW", Role.W, +0.10, -0.76),
        Slot("RW", Role.W, +0.10, +0.76),
        Slot("ST", Role.ST, +0.42, 0.00),
    ),
)

FORMATIONS: dict[str, Formation] = {
    f.name: f for f in (FOUR_THREE_THREE, FOUR_FOUR_TWO, FOUR_TWO_THREE_ONE)
}

# A declared formation is more spread out than a team's actual average shape.
# The reference positions in ROLES.md §6 describe the ideal in-possession
# picture a coach draws; measured over a DFL match, the real instantaneous
# spread is 53m deep and 40m wide against the template's 69m and 54m.
#
# Calibrated by sweep against observed positions: these factors take mean
# deviation from 17.3m to 14.0m, matching the 14.1m ceiling measured for a
# *perfect* team-specific ball-conditioned shape.
#
# **Apply this only when predicting where players will be, never when telling
# them where to stand.** For the practice tool the declared shape is the
# instruction, not an estimate of it, and shrinking it would teach players a
# compressed version of what the coach drew. Same numbers, opposite purpose —
# which is the distinction the whole product direction turns on (PLAN.md §1.0).
SHAPE_CALIBRATION = (0.65, 0.80)

# Phase adjustments, applied after ball shifting.
#   drop:     how far back the block sits, in normalised x
#   compress: how much lateral spread is pulled toward the centre, 0-1
PHASE_ADJUST: dict[Phase, tuple[float, float]] = {
    Phase.IN_POSSESSION: (0.00, 0.00),
    Phase.OUT_OF_POSSESSION: (-0.18, 0.22),
    Phase.TRANSITION: (-0.09, 0.11),
    Phase.SET_PIECE: (0.00, 0.00),
    Phase.UNKNOWN: (-0.05, 0.06),
}


def deform(
    formation: Formation,
    ball_xy: np.ndarray | None,
    phase: Phase = Phase.UNKNOWN,
    line_height: float = 0.0,
    calibrated: bool = False,
) -> np.ndarray:
    """(11, 2) target positions in metres for the advised team.

    ``ball_xy`` is in metres, centre-origin, with the advised team attacking +x.
    ``line_height`` shifts the whole block up or down the pitch in normalised
    units — the coach's dial for how high to play.

    ``calibrated`` shrinks the shape toward observed match averages
    (SHAPE_CALIBRATION). Use it when **predicting** where players will be; leave
    it off when **instructing** them where to stand, which is the practice
    tool's case — there the declared shape is the target, not an estimate.

    The block *shifts with* the ball rather than chasing it: each slot moves by
    a gain on the ball's position, so shape is preserved while the team slides
    across and up. Chasing would collapse everyone onto the ball.
    """
    ref = np.array([[s.ref_x, s.ref_y] for s in formation.slots], dtype=np.float32)
    out = ref.copy()

    if ball_xy is not None:
        bx = float(ball_xy[0]) / pitch.HALF_LENGTH
        by = float(ball_xy[1]) / pitch.HALF_WIDTH

        gains = np.array(
            [[BALL_RESPONSE[s.role].bx, BALL_RESPONSE[s.role].by] for s in formation.slots],
            dtype=np.float32,
        )
        out[:, 0] += gains[:, 0] * bx
        out[:, 1] += gains[:, 1] * by

        # Ball-far behaviour: roles with high asymmetry tuck toward the centre
        # when the ball is on the opposite flank. This is what distinguishes an
        # inverted winger from a touchline one, and a full-back that becomes a
        # third centre-back from one that holds width.
        asym = np.array([BALL_RESPONSE[s.role].asym for s in formation.slots], dtype=np.float32)
        far_side = np.sign(ref[:, 1]) * np.sign(by) < 0
        pull = asym * abs(by) * far_side
        out[:, 1] -= pull * ref[:, 1]

    drop, compress = PHASE_ADJUST.get(phase, (0.0, 0.0))
    out[:, 0] += drop + line_height
    out[:, 1] *= 1.0 - compress

    # The keeper does not shift with the block; it tracks the ball's side only.
    keeper = [i for i, s in enumerate(formation.slots) if s.role is Role.GK]
    for i in keeper:
        out[i, 0] = formation.slots[i].ref_x + 0.06 * max(0.0, out[i, 0] - formation.slots[i].ref_x)

    out[:, 0] = np.clip(out[:, 0], -0.99, 0.99)
    out[:, 1] = np.clip(out[:, 1], -0.99, 0.99)

    sx, sy = SHAPE_CALIBRATION if calibrated else (1.0, 1.0)
    metres = np.empty_like(out)
    metres[:, 0] = out[:, 0] * pitch.HALF_LENGTH * sx
    metres[:, 1] = out[:, 1] * pitch.HALF_WIDTH * sy
    return metres


def assign_to_slots(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Match actual players to formation slots, minimising total travel.

    Slot order in a formation is arbitrary with respect to who is actually on
    the pitch, so comparing slot *i* to player *i* would produce nonsense
    corrections. A linear assignment gives each player the slot they are
    genuinely closest to occupying.

    Returns an index array such that ``targets[result[j]]`` is the target for
    player ``j``.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=-1)
    _, col = linear_sum_assignment(cost)
    return col
