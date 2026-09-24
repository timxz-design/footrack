"""L2 — coaching principles as constraints (ROLES.md §4.3, §5).

Applied to L1's targets as sequential adjustments. Hard constraints are
inviolable; soft ones pull with a weight.

Most coaching lives at **team** level rather than role level, which is why this
is a separate layer rather than more fields on a role. The clean illustration is
width: whether the full-back or the winger holds the touchline depends on what
the other is doing. Encode "hold width" in both roles and you get two players on
the same line; encode it once here and the layer assigns it dynamically.

Every adjustment returns a report, because a mark without a reason is
unteachable (PLAN.md §L5) and these reasons are the ones a coach recognises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footrack import pitch
from footrack.roles import Role
from footrack.state import Phase

# ROLES.md §5. Defaults; a coach overrides them per team.
MAX_VERTICAL_SPREAD = 28.0  # metres, out of possession
BACK_LINE_SPACING = (8.0, 14.0)  # metres between adjacent defenders
MIN_GOAL_SIDE = 2  # players between ball and own goal
WING_CHANNEL = 0.6  # |y| / half-width beyond which a slot holds width
MAX_PER_CELL = 3  # players in one channel x third cell

PENALTY_AREA_X = -pitch.HALF_LENGTH + 16.5
PENALTY_AREA_Y = 20.16


@dataclass
class Adjustment:
    """One constraint firing, with enough detail to explain a mark."""

    name: str
    kind: str  # "hard" | "soft"
    slots: tuple[int, ...]
    moved_m: float
    detail: str = ""


@dataclass
class Context:
    """Everything the constraints need to know about the current frame."""

    ball_xy: np.ndarray | None
    phase: Phase
    roles: tuple[Role, ...]
    opponents: np.ndarray | None = None  # (N, 2) metres
    refining: bool = False  # whether L3 will run after this
    reports: list[Adjustment] = field(default_factory=list)


def _record(ctx: Context, name: str, kind: str, before: np.ndarray,
            after: np.ndarray, slots: tuple[int, ...], detail: str = "") -> None:
    if not slots:
        return
    moved = float(np.linalg.norm(after[list(slots)] - before[list(slots)], axis=-1).max())
    if moved > 0.05:
        ctx.reports.append(Adjustment(name, kind, slots, moved, detail))


def keeper_in_box(t: np.ndarray, ctx: Context) -> np.ndarray:
    """HARD — the goalkeeper stays inside their penalty area."""
    out = t.copy()
    slots = tuple(i for i, r in enumerate(ctx.roles) if r is Role.GK)
    for i in slots:
        out[i, 0] = np.clip(out[i, 0], -pitch.HALF_LENGTH + 0.5, PENALTY_AREA_X)
        out[i, 1] = np.clip(out[i, 1], -PENALTY_AREA_Y, PENALTY_AREA_Y)
    _record(ctx, "keeper in box", "hard", t, out, slots)
    return out


def vertical_compactness(t: np.ndarray, ctx: Context) -> np.ndarray:
    """HARD out of possession — the block stays within a set vertical spread.

    The single most-drilled principle in defensive coaching, and the one a
    rules layer can enforce exactly where a model only approximates it.
    """
    if ctx.phase not in (Phase.OUT_OF_POSSESSION, Phase.TRANSITION):
        return t

    out = t.copy()
    outfield = [i for i, r in enumerate(ctx.roles) if r is not Role.GK]
    xs = out[outfield, 0]
    spread = xs.max() - xs.min()
    if spread <= MAX_VERTICAL_SPREAD:
        return out

    centre = xs.mean()
    scale = MAX_VERTICAL_SPREAD / spread
    out[outfield, 0] = centre + (xs - centre) * scale
    _record(ctx, "vertical compactness", "hard", t, out, tuple(outfield),
            f"block was {spread:.0f}m, compressed to {MAX_VERTICAL_SPREAD:.0f}m")
    return out


def goal_side_cover(t: np.ndarray, ctx: Context) -> np.ndarray:
    """HARD — at least two outfield players stay between the ball and own goal.

    Counter-attack protection. Without it the optimiser will happily push every
    player forward when the objectives favour it.
    """
    if ctx.ball_xy is None:
        return t

    out = t.copy()
    outfield = [i for i, r in enumerate(ctx.roles) if r is not Role.GK]
    ball_x = float(ctx.ball_xy[0])
    behind = [i for i in outfield if out[i, 0] < ball_x]
    if len(behind) >= MIN_GOAL_SIDE:
        return out

    # Recall the deepest players that are not already behind the ball.
    ahead = sorted((i for i in outfield if i not in behind), key=lambda i: out[i, 0])
    need = MIN_GOAL_SIDE - len(behind)
    pulled = tuple(ahead[:need])
    for i in pulled:
        out[i, 0] = ball_x - 4.0
    _record(ctx, "goal-side cover", "hard", t, out, pulled,
            f"only {len(behind)} behind the ball, need {MIN_GOAL_SIDE}")
    return out


def back_line_spacing(t: np.ndarray, ctx: Context) -> np.ndarray:
    """SOFT — adjacent defenders hold 8-14m apart.

    Too tight and a switch of play beats the line; too wide and a ball between
    them does.
    """
    out = t.copy()
    line = sorted((i for i, r in enumerate(ctx.roles) if r in (Role.CB, Role.FB)),
                  key=lambda i: out[i, 1])
    if len(line) < 2:
        return out

    lo, hi = BACK_LINE_SPACING
    for a, b in zip(line, line[1:]):
        gap = out[b, 1] - out[a, 1]
        if gap < lo:
            push = (lo - gap) / 2
        elif gap > hi:
            push = -(gap - hi) / 2
        else:
            continue
        out[a, 1] -= push
        out[b, 1] += push
    _record(ctx, "back line spacing", "soft", t, out, tuple(line),
            f"held {lo:.0f}-{hi:.0f}m between adjacent defenders")
    return out


def width_occupancy(t: np.ndarray, ctx: Context) -> np.ndarray:
    """SOFT — exactly one player holds each flank in possession.

    The constraint that cannot be reduced to roles: whether the full-back or the
    winger provides width depends on the other. With both told to hold width you
    get two players on one touchline; assigned here, whichever is already wider
    takes it and the other tucks in.
    """
    if ctx.phase is not Phase.IN_POSSESSION:
        return t

    out = t.copy()
    touched: list[int] = []
    for side in (-1, 1):
        candidates = [i for i, r in enumerate(ctx.roles)
                      if r in (Role.W, Role.FB) and np.sign(out[i, 1]) == side]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda i: -abs(out[i, 1]))
        holder, *others = candidates
        out[holder, 1] = side * WING_CHANNEL * pitch.HALF_WIDTH * 1.25
        for i in others:
            out[i, 1] *= 0.55
        touched.extend(candidates)
    _record(ctx, "width occupancy", "soft", t, out, tuple(touched),
            "one player holds each flank, the rest tuck in")
    return out


def anti_clustering(t: np.ndarray, ctx: Context) -> np.ndarray:
    """SOFT — at most three players share a channel-by-third cell.

    Exists to stop an objective that rewards proximity to the ball collapsing
    the shape into a clump. It is therefore only useful alongside L3, and is
    skipped otherwise: measured on a DFL match it fired on 190 of 1217 frames
    and moved players a mean 9.1m, which is a large intervention to make against
    a formation that was not clumping in the first place.

    The limit is three rather than two because with 11 players across 15 cells,
    three in a cell is ordinary in a compact block, not a defect.
    """
    if not ctx.refining:
        return t

    out = t.copy()
    cells: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(ctx.roles):
        if r is Role.GK:
            continue
        key = (int(pitch.third_of(out[i, 0])), int(pitch.channel_of(out[i, 1])))
        cells.setdefault(key, []).append(i)

    moved: list[int] = []
    for (third, channel), members in cells.items():
        if len(members) <= MAX_PER_CELL:
            continue
        members.sort(key=lambda i: abs(out[i, 1]))
        for n, i in enumerate(members[MAX_PER_CELL:], start=1):
            out[i, 1] += np.sign(out[i, 1] or 1.0) * 6.0 * n
            moved.append(i)
    _record(ctx, "anti-clustering", "soft", t, out, tuple(moved),
            f"no more than {MAX_PER_CELL} players per channel-third cell")
    return out


PIPELINE = (
    keeper_in_box,
    width_occupancy,
    back_line_spacing,
    anti_clustering,
    vertical_compactness,
    goal_side_cover,
)


def apply(
    targets: np.ndarray,
    roles: tuple[Role, ...],
    ball_xy: np.ndarray | None,
    phase: Phase = Phase.UNKNOWN,
    opponents: np.ndarray | None = None,
    refining: bool = False,
) -> tuple[np.ndarray, list[Adjustment]]:
    """Run every constraint in order. Returns adjusted targets and what fired.

    Hard constraints run **last** so that soft adjustments cannot violate them —
    ordering matters here, and running compactness before width occupancy would
    let the latter stretch the block back open.
    """
    ctx = Context(ball_xy=ball_xy, phase=phase, roles=roles, opponents=opponents,
                  refining=refining)
    out = targets.astype(np.float32).copy()
    for step in PIPELINE:
        out = step(out, ctx)

    out[:, 0] = np.clip(out[:, 0], -pitch.HALF_LENGTH, pitch.HALF_LENGTH)
    out[:, 1] = np.clip(out[:, 1], -pitch.HALF_WIDTH, pitch.HALF_WIDTH)
    return out, ctx.reports
