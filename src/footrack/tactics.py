"""The rules-based proposal: L1 -> L2 -> L3, with a reason for every mark.

This is the floor of PLAN.md's stack, working end to end without any model. It
is what the labelling tool proposes and what the system falls back to whenever
the learned layers are uncertain.

Every mark carries an attribution, because a mark without a reason is
unteachable. The attribution names whichever layer actually decided the
position: the formation, a specific constraint, or a dominant objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footrack import constraints, objectives
from footrack.formations import FORMATIONS, Formation, assign_to_slots, deform
from footrack.roles import Role
from footrack.state import Phase

# Below this, a player is close enough that drawing a mark is noise. The
# deviation-only display in PLAN.md §1.1 depends on it: no mark means you are
# right, which is far less cluttered than eleven permanent dots.
#
# **Calibrated, and much larger than intuition suggests.** A ball-conditioned
# positional template cannot get below ~14m mean deviation — measured against a
# DFL match, even a *perfect* team-specific shape sits at 14.1m, because
# football positions vary that much around any static reference. Flagging at 4m
# would flag 89% of players and say nothing. At 20m it flags 21%, roughly two
# or three per frame, which is a usable interaction.
#
# The honest consequence: this layer catches **gross** positional errors, not
# fine ones. It cannot tell a coach someone is 3m too narrow — it does not have
# that resolution, and neither does any rules-based approach.
DEVIATION_THRESHOLD_M = 20.0


@dataclass
class Mark:
    """One proposed position, with why."""

    slot: int
    role: Role
    label: str
    position: np.ndarray  # metres, centre-origin
    actual: np.ndarray
    deviation_m: float
    source: str  # "formation" | "constraint" | "objective"
    reason: str

    @property
    def significant(self) -> bool:
        return self.deviation_m >= DEVIATION_THRESHOLD_M


def propose(
    actual: np.ndarray,
    roles: tuple[Role, ...],
    ball_xy: np.ndarray | None,
    phase: Phase = Phase.UNKNOWN,
    formation: Formation | str = "4-3-3",
    opponents: np.ndarray | None = None,
    line_height: float = 0.0,
    refine: bool = False,
) -> list[Mark]:
    """Propose a position for each of the advised team's 11 players.

    ``actual`` is (11, 2) in metres with the advised team attacking +x. ``roles``
    is what the coach assigned, used only to pick objective weights — the
    formation supplies the shape.

    **``refine`` defaults to off.** Measured against a DFL match, the layers
    contribute: L1 alone 14.0m mean deviation, plus constraints 14.4m, plus
    objective refinement 16.0m. L3's weights were written by hand with nothing
    to calibrate against, and the only available measurement says they push
    positions away from where professionals actually stand. Turn it on once
    coach corrections exist to tune the weights against — that is the target
    that was missing.

    L2 costs 0.4m and is kept: matching elite positions is not actually the
    objective, enforcing the coach's principles is, and 0.4m is a fair price
    for constraints a coach would insist on regardless.
    """
    if isinstance(formation, str):
        formation = FORMATIONS[formation]

    # L1 — shape, shifted with the ball and adjusted for phase.
    targets = deform(formation, ball_xy, phase, line_height)

    # Match players to slots by who is closest to occupying them. Slot order is
    # arbitrary with respect to who is on the pitch, so pairing index-to-index
    # would produce nonsense corrections.
    order = assign_to_slots(actual, targets)
    targets = targets[order]
    slot_roles = tuple(formation.slots[j].role for j in order)
    labels = tuple(formation.slots[j].label for j in order)

    before_constraints = targets.copy()

    # L2 — coaching principles.
    targets, reports = constraints.apply(
        targets, slot_roles, ball_xy, phase, opponents, refining=refine
    )
    after_constraints = targets.copy()

    # L3 — local objective refinement.
    gain = np.zeros(len(targets), dtype=np.float32)
    if refine:
        scene = objectives.Scene(
            ball_xy=ball_xy, teammates=targets, opponents=opponents, phase=phase
        )
        targets, gain = objectives.refine(targets, slot_roles, scene)

    # Attribute each mark to whichever layer actually moved it.
    by_slot: dict[int, list[constraints.Adjustment]] = {}
    for report in reports:
        for s in report.slots:
            by_slot.setdefault(s, []).append(report)

    marks: list[Mark] = []
    for i, role in enumerate(slot_roles):
        constraint_move = float(np.linalg.norm(after_constraints[i] - before_constraints[i]))
        objective_move = float(np.linalg.norm(targets[i] - after_constraints[i]))

        if constraint_move >= max(objective_move, 1.0) and i in by_slot:
            top = max(by_slot[i], key=lambda r: r.moved_m)
            source, reason = "constraint", top.detail or top.name
        elif objective_move >= 1.0 and gain[i] > 0:
            source, reason = "objective", _dominant_objective(role, phase)
        else:
            source, reason = "formation", f"{formation.name} {labels[i]} position"

        marks.append(
            Mark(
                slot=i,
                role=role,
                label=labels[i],
                position=targets[i],
                actual=actual[i],
                deviation_m=float(np.linalg.norm(targets[i] - actual[i])),
                source=source,
                reason=reason,
            )
        )
    return marks


def _dominant_objective(role: Role, phase: Phase) -> str:
    table = (
        objectives.OUT_OF_POSSESSION if phase is Phase.OUT_OF_POSSESSION
        else objectives.IN_POSSESSION
    )
    weights = table.get(role, table[Role.UNKNOWN])
    name = max(weights, key=weights.get)
    phrasing = {
        "width": "hold width",
        "depth": "get in behind",
        "progression": "offer a forward pass",
        "space": "occupy space",
        "pressure": "press the ball",
        "lanes": "cover the passing lane",
        "goal": "protect the goal",
        "compact": "stay compact",
    }
    return phrasing.get(name, name)


def significant(marks: list[Mark]) -> list[Mark]:
    """Only the marks worth drawing (PLAN.md §1.1 deviation-only display)."""
    return [m for m in marks if m.significant]


def summarise(marks: list[Mark]) -> dict:
    """Aggregate view of a proposal, for logging and sanity checks."""
    devs = np.array([m.deviation_m for m in marks])
    sources: dict[str, int] = {}
    for m in marks:
        sources[m.source] = sources.get(m.source, 0) + 1
    return {
        "players": len(marks),
        "flagged": int((devs >= DEVIATION_THRESHOLD_M).sum()),
        "mean_deviation_m": float(devs.mean()),
        "max_deviation_m": float(devs.max()),
        "by_source": sources,
    }
