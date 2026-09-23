"""Role inference and identity wiring.

Both were silently absent for several training runs: ``role_ids`` sat at zero
and ``player_ids`` at -1 for every player, so the model was asked to place a
hidden player knowing only their team and physical profile. These tests exist so
that cannot recur unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from footrack import N_ENTITIES, Role, build_player_index, infer_roles, role_id_array
from footrack.roles import DEF_LINE, MID_LINE, WIDE_P85, _classify, _own_half_depth
from footrack.state import SlotMap

from test_state import make_sequence


def seq_with(players, t=200, match_id="m1"):
    """A sequence whose slots carry the given player ids."""
    s = make_sequence(t=t)
    ids = [None] + list(players) + [None] * (N_ENTITIES - 1 - len(players))
    slots = SlotMap(
        player_ids=tuple(ids[:N_ENTITIES]),
        jersey_numbers=tuple([None] * N_ENTITIES),
        team_ids=tuple([None] + ["home"] * 11 + ["away"] * 11),
    )
    return type(s)(**{**s.__dict__, "slots": slots, "match_id": match_id})


class TestClassify:
    def test_keeper_is_deep_and_slow(self):
        assert _classify(depth=12.0, width=8.0, max_speed=5.5) == Role.GK

    def test_a_fast_deep_player_is_not_a_keeper(self):
        """Sweeper-keepers aside, depth alone must not imply GK — a deep
        defender would otherwise be misread."""
        assert _classify(depth=20.0, width=8.0, max_speed=9.0) != Role.GK

    def test_deep_and_narrow_is_a_centre_back(self):
        assert _classify(DEF_LINE - 5, width=10.0, max_speed=8.5) == Role.CB

    def test_deep_and_wide_is_a_full_back(self):
        assert _classify(DEF_LINE - 5, width=WIDE_P85 + 5, max_speed=8.5) == Role.FB

    def test_midfield_splits_on_depth(self):
        assert _classify(47.0, width=10.0, max_speed=8.5) == Role.DM
        assert _classify(54.0, width=10.0, max_speed=8.5) == Role.CM

    def test_advanced_and_narrow_is_a_forward(self):
        assert _classify(65.0, width=8.0, max_speed=8.5) == Role.ST

    def test_wide_stays_wide_at_every_depth(self):
        wide = WIDE_P85 + 5
        assert _classify(MID_LINE - 2, width=wide, max_speed=8.5) == Role.W
        assert _classify(MID_LINE + 6, width=wide, max_speed=8.5) == Role.W

    def test_every_outcome_is_a_real_role(self):
        for depth in (10, 30, 45, 52, 60, 70):
            for width in (5, 30):
                for speed in (5.0, 9.0):
                    assert isinstance(_classify(depth, width, speed), Role)


class TestOwnHalfDepth:
    def test_team_a_measures_from_the_negative_end(self):
        pos = np.array([[-52.5, 0.0], [0.0, 0.0], [52.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(_own_half_depth(pos, True), [0.0, 52.5, 105.0])

    def test_team_b_is_mirrored(self):
        """Sequences store team A attacking +x. Without the flip, every opponent
        defender would read as a forward."""
        pos = np.array([[52.5, 0.0], [0.0, 0.0], [-52.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(_own_half_depth(pos, False), [0.0, 52.5, 105.0])

    def test_the_two_teams_disagree_about_the_same_point(self):
        pos = np.array([[30.0, 0.0]], dtype=np.float32)
        a = _own_half_depth(pos, True)[0]
        b = _own_half_depth(pos, False)[0]
        assert a + b == pytest.approx(105.0)


class TestInferRoles:
    def test_assigns_every_named_player(self):
        seqs = [seq_with([f"p{i}" for i in range(22)])]
        roles = infer_roles(seqs)
        assert len(roles) == 22

    def test_returns_valid_role_indices(self):
        roles = infer_roles([seq_with([f"p{i}" for i in range(22)])])
        assert all(r in {int(x) for x in Role} for r in roles.values())

    def test_ignores_unoccupied_slots(self):
        roles = infer_roles([seq_with(["p0", "p1"])])
        assert set(roles) == {"p0", "p1"}

    def test_combines_evidence_across_sequences(self):
        seqs = [seq_with(["p0"], match_id="m1"), seq_with(["p0"], match_id="m2")]
        assert len(infer_roles(seqs)) == 1


class TestRoleIdArray:
    def test_shape_and_dtype(self):
        seq = seq_with(["p0", "p1"])
        arr = role_id_array(seq.slots, {"p0": int(Role.CB), "p1": int(Role.ST)})
        assert arr.shape == (N_ENTITIES,)
        assert arr.dtype == np.int32

    def test_maps_players_to_their_slots(self):
        seq = seq_with(["p0", "p1"])
        arr = role_id_array(seq.slots, {"p0": int(Role.CB), "p1": int(Role.ST)})
        assert arr[1] == Role.CB
        assert arr[2] == Role.ST

    def test_ball_and_empty_slots_are_unknown(self):
        seq = seq_with(["p0"])
        arr = role_id_array(seq.slots, {"p0": int(Role.CB)})
        assert arr[0] == Role.UNKNOWN
        assert arr[5] == Role.UNKNOWN

    def test_unlisted_player_falls_back_to_unknown(self):
        """Cold start: a player with no assigned role must not crash or borrow
        someone else's."""
        seq = seq_with(["p0", "p1"])
        arr = role_id_array(seq.slots, {"p0": int(Role.CB)})
        assert arr[2] == Role.UNKNOWN


class TestPlayerIndex:
    def test_is_stable_across_calls(self):
        seqs = [seq_with([f"p{i}" for i in range(22)])]
        assert build_player_index(seqs) == build_player_index(seqs)

    def test_is_independent_of_sequence_order(self):
        """A shifting index would attach one player's learned tendency to
        another."""
        a = seq_with(["x", "y"], match_id="m1")
        b = seq_with(["y", "z"], match_id="m2")
        assert build_player_index([a, b]) == build_player_index([b, a])

    def test_indices_are_contiguous_from_zero(self):
        idx = build_player_index([seq_with([f"p{i}" for i in range(10)])])
        assert sorted(idx.values()) == list(range(len(idx)))

    def test_excludes_empty_slots(self):
        idx = build_player_index([seq_with(["p0", "p1"])])
        assert None not in idx
        assert len(idx) == 2
