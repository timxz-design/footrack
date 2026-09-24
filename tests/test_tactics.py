"""L1/L2/L3 â€” formations, constraints and the rules-based proposal.

The recurring failure in this layer has been calibrating against *designed*
numbers rather than measured ones: reference positions drawn wider than teams
actually play, a 4m deviation threshold against a 14m floor, objective weights
invented with nothing to tune them on. Several tests here pin measured values
precisely so a future edit has to confront the measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from footrack import constraints, objectives, pitch, tactics
from footrack.formations import (
    FORMATIONS,
    FOUR_THREE_THREE,
    SHAPE_CALIBRATION,
    assign_to_slots,
    deform,
)
from footrack.roles import Role
from footrack.state import Phase

CENTRE = np.zeros(2, dtype=np.float32)


class TestFormations:
    def test_all_have_eleven_slots(self):
        for name, f in FORMATIONS.items():
            assert len(f.slots) == 11, name

    def test_wrong_slot_count_rejected(self):
        from footrack.formations import Formation, Slot

        with pytest.raises(ValueError, match="expected 11"):
            Formation("bad", (Slot("X", Role.CB, 0.0, 0.0),))

    def test_every_formation_has_exactly_one_keeper(self):
        for name, f in FORMATIONS.items():
            assert sum(r is Role.GK for r in f.roles) == 1, name

    def test_deform_shape_and_units(self):
        out = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        assert out.shape == (11, 2)
        assert np.abs(out[:, 0]).max() <= pitch.HALF_LENGTH
        assert np.abs(out[:, 1]).max() <= pitch.HALF_WIDTH

    def test_no_ball_still_produces_a_shape(self):
        out = deform(FOUR_THREE_THREE, None, Phase.UNKNOWN)
        assert out.shape == (11, 2)
        assert np.isfinite(out).all()

    def test_block_shifts_toward_the_ball_not_onto_it(self):
        """Gains move the whole block; chasing would collapse everyone in."""
        left = deform(FOUR_THREE_THREE, np.array([0.0, -25.0], np.float32), Phase.IN_POSSESSION)
        right = deform(FOUR_THREE_THREE, np.array([0.0, +25.0], np.float32), Phase.IN_POSSESSION)
        assert left[:, 1].mean() < right[:, 1].mean()
        spread_left = left[:, 1].max() - left[:, 1].min()
        assert spread_left > 20.0, "block collapsed toward the ball"

    def test_advancing_ball_pushes_the_line_up(self):
        back = deform(FOUR_THREE_THREE, np.array([-40.0, 0.0], np.float32), Phase.IN_POSSESSION)
        fwd = deform(FOUR_THREE_THREE, np.array([+40.0, 0.0], np.float32), Phase.IN_POSSESSION)
        assert fwd[:, 0].mean() > back[:, 0].mean()

    def test_out_of_possession_sits_deeper_and_narrower(self):
        on = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        off = deform(FOUR_THREE_THREE, CENTRE, Phase.OUT_OF_POSSESSION)
        assert off[:, 0].mean() < on[:, 0].mean()
        assert np.abs(off[:, 1]).mean() < np.abs(on[:, 1]).mean()

    def test_line_height_raises_the_whole_block(self):
        low = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION, line_height=-0.2)
        high = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION, line_height=+0.2)
        assert high[:, 0].mean() > low[:, 0].mean()

    def test_keeper_barely_moves_with_the_block(self):
        gk = [i for i, s in enumerate(FOUR_THREE_THREE.slots) if s.role is Role.GK][0]
        back = deform(FOUR_THREE_THREE, np.array([-40.0, 0.0], np.float32), Phase.IN_POSSESSION)
        fwd = deform(FOUR_THREE_THREE, np.array([+40.0, 0.0], np.float32), Phase.IN_POSSESSION)
        assert abs(fwd[gk, 0] - back[gk, 0]) < 8.0

    def test_calibration_is_applied(self):
        """A declared formation is drawn wider than teams actually play; the
        measured correction takes mean deviation 17.3m -> 14.0m."""
        assert SHAPE_CALIBRATION == (0.65, 0.80)
        out = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        raw_span = np.ptp(FOUR_THREE_THREE.reference_metres()[:, 0])
        assert np.ptp(out[:, 0]) < raw_span


class TestSlotAssignment:
    def test_returns_a_permutation(self):
        rng = np.random.default_rng(0)
        pos = rng.uniform(-40, 40, size=(11, 2)).astype(np.float32)
        targets = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        order = assign_to_slots(pos, targets)
        assert sorted(order.tolist()) == list(range(11))

    def test_identical_layout_maps_to_itself(self):
        targets = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        order = assign_to_slots(targets, targets)
        np.testing.assert_array_equal(order, np.arange(11))

    def test_beats_naive_pairing(self):
        """Slot order is arbitrary with respect to who is on the pitch, so
        index-to-index pairing produces nonsense corrections."""
        targets = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        shuffled = targets[::-1].copy()
        order = assign_to_slots(shuffled, targets)
        matched = np.linalg.norm(shuffled - targets[order], axis=-1).mean()
        naive = np.linalg.norm(shuffled - targets, axis=-1).mean()
        assert matched < naive


def ctx_roles():
    return tuple(FOUR_THREE_THREE.slots[i].role for i in range(11))


class TestConstraints:
    def test_keeper_confined_to_the_box(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[0] = np.array([20.0, 0.0], dtype=np.float32)  # keeper at halfway
        out, reports = constraints.apply(t, ctx_roles(), CENTRE, Phase.IN_POSSESSION)
        assert out[0, 0] <= constraints.PENALTY_AREA_X + 1e-3
        assert any(r.name == "keeper in box" for r in reports)

    def test_vertical_compactness_enforced_out_of_possession(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[:, 0] = np.linspace(-50, 50, 11)  # 100m block
        out, reports = constraints.apply(t, ctx_roles(), CENTRE, Phase.OUT_OF_POSSESSION)
        outfield = [i for i, r in enumerate(ctx_roles()) if r is not Role.GK]
        spread = np.ptp(out[outfield, 0])
        assert spread <= constraints.MAX_VERTICAL_SPREAD + 1e-3
        assert any(r.name == "vertical compactness" for r in reports)

    def test_compactness_not_enforced_in_possession(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[:, 0] = np.linspace(-50, 50, 11)
        out, _ = constraints.apply(t, ctx_roles(), CENTRE, Phase.IN_POSSESSION)
        outfield = [i for i, r in enumerate(ctx_roles()) if r is not Role.GK]
        assert np.ptp(out[outfield, 0]) > constraints.MAX_VERTICAL_SPREAD

    def test_goal_side_cover_pulls_players_back(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[:, 0] = 40.0  # everyone ahead of the ball
        ball = np.array([0.0, 0.0], dtype=np.float32)
        out, reports = constraints.apply(t, ctx_roles(), ball, Phase.IN_POSSESSION)
        outfield = [i for i, r in enumerate(ctx_roles()) if r is not Role.GK]
        behind = sum(1 for i in outfield if out[i, 0] < ball[0])
        assert behind >= constraints.MIN_GOAL_SIDE
        assert any(r.name == "goal-side cover" for r in reports)

    def test_anti_clustering_only_runs_when_refining(self):
        """It exists to stop L3 clumping the shape; without L3 it is a large
        intervention against a formation that was not clumping."""
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[1:, :] = np.array([0.0, 0.0], dtype=np.float32)  # everyone stacked
        off, r_off = constraints.apply(t, ctx_roles(), CENTRE, Phase.IN_POSSESSION, refining=False)
        on, r_on = constraints.apply(t, ctx_roles(), CENTRE, Phase.IN_POSSESSION, refining=True)
        assert not any(r.name == "anti-clustering" for r in r_off)
        assert any(r.name == "anti-clustering" for r in r_on)

    def test_everything_stays_on_the_pitch(self):
        rng = np.random.default_rng(1)
        t = rng.uniform(-80, 80, size=(11, 2)).astype(np.float32)
        out, _ = constraints.apply(t, ctx_roles(), CENTRE, Phase.OUT_OF_POSSESSION)
        assert np.abs(out[:, 0]).max() <= pitch.HALF_LENGTH + 1e-3
        assert np.abs(out[:, 1]).max() <= pitch.HALF_WIDTH + 1e-3

    def test_reports_carry_a_magnitude(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        t[0] = np.array([30.0, 0.0], dtype=np.float32)
        _, reports = constraints.apply(t, ctx_roles(), CENTRE, Phase.IN_POSSESSION)
        assert reports and all(r.moved_m > 0 for r in reports)


class TestObjectives:
    def test_score_is_finite_everywhere(self):
        scene = objectives.Scene(
            ball_xy=CENTRE,
            teammates=deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION),
            opponents=np.zeros((11, 2), dtype=np.float32),
            phase=Phase.IN_POSSESSION,
        )
        for role in Role:
            v = objectives.score(np.array([10.0, 5.0], np.float32), 3, role, scene)
            assert np.isfinite(v)

    def test_width_role_prefers_the_touchline(self):
        scene = objectives.Scene(ball_xy=CENTRE, teammates=np.zeros((11, 2), np.float32),
                                 opponents=None, phase=Phase.IN_POSSESSION)
        wide = objectives.score(np.array([0.0, 30.0], np.float32), 3, Role.FB, scene)
        narrow = objectives.score(np.array([0.0, 2.0], np.float32), 3, Role.FB, scene)
        assert wide > narrow

    def test_pressing_role_prefers_being_near_the_ball(self):
        ball = np.array([20.0, 0.0], dtype=np.float32)
        scene = objectives.Scene(ball_xy=ball, teammates=np.zeros((11, 2), np.float32),
                                 opponents=None, phase=Phase.OUT_OF_POSSESSION)
        near = objectives.score(np.array([22.0, 0.0], np.float32), 5, Role.CM, scene)
        far = objectives.score(np.array([-30.0, 0.0], np.float32), 5, Role.CM, scene)
        assert near > far

    def test_refine_keeps_the_keeper_put(self):
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        scene = objectives.Scene(ball_xy=CENTRE, teammates=t, opponents=None,
                                 phase=Phase.IN_POSSESSION)
        out, _ = objectives.refine(t, ctx_roles(), scene)
        np.testing.assert_allclose(out[0], t[0])

    def test_refine_moves_nobody_far(self):
        """Local by design: L1 and L2 encode shape and principles this layer
        cannot rediscover."""
        t = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        scene = objectives.Scene(ball_xy=CENTRE, teammates=t, opponents=None,
                                 phase=Phase.IN_POSSESSION)
        out, _ = objectives.refine(t, ctx_roles(), scene, radius_m=6.0, iterations=2)
        assert np.linalg.norm(out - t, axis=-1).max() <= 6.0 * 2 + 1e-3


class TestPropose:
    def setup_scene(self):
        rng = np.random.default_rng(3)
        actual = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        actual = actual + rng.normal(0, 6, size=actual.shape).astype(np.float32)
        opponents = -actual
        return actual, opponents

    def test_one_mark_per_player(self):
        actual, opp = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION,
                                opponents=opp)
        assert len(marks) == 11

    def test_every_mark_has_a_reason(self):
        actual, opp = self.setup_scene()
        for m in tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION, opponents=opp):
            assert m.reason
            assert m.source in {"formation", "constraint", "objective"}

    def test_refinement_is_off_by_default(self):
        """Measured on a DFL match, L3 raises mean deviation 14.2m -> 15.9m."""
        actual, opp = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION, opponents=opp)
        assert all(m.source != "objective" for m in marks)

    def test_refinement_can_be_enabled(self):
        actual, opp = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION,
                                opponents=opp, refine=True)
        assert len(marks) == 11

    def test_threshold_matches_the_measured_floor(self):
        """A ball-conditioned template cannot beat ~14m mean deviation, so a 4m
        threshold would flag 89% of players and say nothing."""
        assert tactics.DEVIATION_THRESHOLD_M == 20.0

    def test_significant_filters_by_threshold(self):
        actual, opp = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION, opponents=opp)
        flagged = tactics.significant(marks)
        assert all(m.deviation_m >= tactics.DEVIATION_THRESHOLD_M for m in flagged)
        assert len(flagged) <= len(marks)

    def test_well_placed_team_gets_few_marks(self):
        exact = deform(FOUR_THREE_THREE, CENTRE, Phase.IN_POSSESSION)
        marks = tactics.propose(exact, ctx_roles(), CENTRE, Phase.IN_POSSESSION)
        assert len(tactics.significant(marks)) == 0

    def test_accepts_a_formation_by_name(self):
        actual, opp = self.setup_scene()
        for name in FORMATIONS:
            marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION,
                                    formation=name, opponents=opp)
            assert len(marks) == 11

    def test_works_without_ball_or_opponents(self):
        actual, _ = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), None, Phase.UNKNOWN)
        assert len(marks) == 11
        assert all(np.isfinite(m.position).all() for m in marks)

    def test_summarise_reports_consistently(self):
        actual, opp = self.setup_scene()
        marks = tactics.propose(actual, ctx_roles(), CENTRE, Phase.IN_POSSESSION, opponents=opp)
        info = tactics.summarise(marks)
        assert info["players"] == 11
        assert info["flagged"] == len(tactics.significant(marks))
        assert info["max_deviation_m"] >= info["mean_deviation_m"]
