"""Feature builder invariants.

The recurring failure mode here is silent: a feature that leaks through a mask,
a normalisation that is subtly anisotropic, or a phase that does not flip when
perspective does. None of these raise — they just make training quietly worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from footrack import (
    BALL,
    CONTEXT_FEATURES,
    ENTITY_FEATURES,
    N_CONTEXT_FEATURES,
    N_ENTITIES,
    N_ENTITY_FEATURES,
    N_PHASES,
    Phase,
    PlayerProfile,
    build,
    derive_phase,
)
from footrack.features import SPEED_SCALE, build_context, build_entity_features

from test_state import make_sequence


OBSERVATION_FEATURES = set(ENTITY_FEATURES[:19])
STRUCTURAL_FEATURES = set(ENTITY_FEATURES[19:])


def with_phase(seq, phase_values):
    return type(seq)(**{**seq.__dict__, "phase": np.asarray(phase_values, dtype=np.int8)})


class TestShapes:
    def test_entity_tensor_shape(self):
        s = make_sequence(t=30)
        f = build_entity_features(s)
        assert f.shape == (30, N_ENTITIES, N_ENTITY_FEATURES)
        assert f.dtype == np.float32

    def test_context_shape(self):
        s = make_sequence(t=30)
        assert build_context(s).shape == (30, N_CONTEXT_FEATURES)

    def test_feature_name_lists_match_widths(self):
        assert len(ENTITY_FEATURES) == N_ENTITY_FEATURES
        assert len(CONTEXT_FEATURES) == N_CONTEXT_FEATURES

    def test_all_finite(self):
        mi = build(make_sequence(t=30))
        assert np.isfinite(mi.entity).all()
        assert np.isfinite(mi.context).all()

    def test_column_accessor(self):
        mi = build(make_sequence(t=30))
        np.testing.assert_array_equal(mi.column("speed"), mi.entity[..., 4])


class TestMasking:
    def test_observation_features_zeroed_when_absent(self):
        s = make_sequence(t=60)
        f = build_entity_features(s)
        absent = ~s.mask
        assert absent.any(), "fixture should contain absent entities"
        for i, name in enumerate(ENTITY_FEATURES):
            if name in OBSERVATION_FEATURES:
                assert (f[..., i][absent] == 0).all(), f"{name} leaks through the mask"

    def test_structural_features_survive_absence(self):
        """A masked entity must remain identifiable, or masked agent modelling
        has nothing to condition on."""
        s = make_sequence(t=60)
        f = build_entity_features(s)
        absent = ~s.mask
        team_flags = f[..., 19:22].sum(axis=-1)
        assert (team_flags[absent] == 1).all()

    def test_velocity_zeroed_when_untrustworthy(self):
        s = make_sequence(t=60)
        f = build_entity_features(s)
        bad = ~s.velocity_valid
        for i in range(2, 7):  # vel_x, vel_y, speed, vel_dir_cos, vel_dir_sin
            assert (f[..., i][bad] == 0).all()

    def test_entity_type_is_one_hot(self):
        f = build_entity_features(make_sequence(t=30))
        np.testing.assert_allclose(f[..., 19:22].sum(axis=-1), 1.0)

    def test_ball_is_slot_zero_only(self):
        f = build_entity_features(make_sequence(t=30))
        assert (f[:, BALL, 19] == 1).all()
        assert (f[:, 1:, 19] == 0).all()


class TestNormalisation:
    def test_position_scaling_is_isotropic(self):
        """10m along x and 10m along y must produce equal feature magnitudes."""
        s = make_sequence(t=4)
        pos = s.positions.copy()
        pos[:] = 0.0
        pos[:, 1] = [10.0, 0.0]
        pos[:, 2] = [0.0, 10.0]
        s = type(s)(**{**s.__dict__, "positions": pos})
        f = build_entity_features(s)
        assert abs(f[0, 1, 0]) == pytest.approx(abs(f[0, 2, 1]))

    def test_direction_features_are_unit_length(self):
        s = make_sequence(t=40)
        f = build_entity_features(s)
        for lo in (8, 11, 14):  # bearings to ball, own goal, opp goal
            norm = np.linalg.norm(f[..., lo : lo + 2], axis=-1)
            present = s.mask
            assert np.all((norm[present] < 1e-6) | (np.abs(norm[present] - 1) < 1e-4))

    def test_orientation_slot_reserved_and_empty(self):
        """Reserved for pose data that no public dataset carries yet."""
        f = build_entity_features(make_sequence(t=30))
        assert (f[..., 16:19] == 0).all()

    def test_profile_feeds_through(self):
        s = make_sequence(t=10)
        fast = {s.slots.player_ids[1]: PlayerProfile(max_speed=9.5)}
        f = build_entity_features(s, profiles=fast)
        assert f[0, 1, 24] == pytest.approx(9.5 / SPEED_SCALE)

    def test_ball_has_no_physical_profile(self):
        f = build_entity_features(make_sequence(t=10))
        assert (f[:, BALL, 24:27] == 0).all()


class TestPhase:
    def test_context_phase_is_one_hot(self):
        s = with_phase(make_sequence(t=5), [0, 1, 2, 3, 4])
        ctx = build_context(s)
        np.testing.assert_allclose(ctx[:, :N_PHASES].sum(axis=1), 1.0)

    def test_phase_column_matches_value(self):
        s = with_phase(make_sequence(t=5), [0, 1, 2, 3, 4])
        ctx = build_context(s)
        np.testing.assert_array_equal(np.argmax(ctx[:, :N_PHASES], axis=1), [0, 1, 2, 3, 4])

    def test_missing_phase_defaults_to_unknown(self):
        s = type(make_sequence(t=5))(**{**make_sequence(t=5).__dict__, "phase": None})
        ctx = build_context(s)
        assert (ctx[:, Phase.UNKNOWN] == 1).all()

    def test_swap_perspective_flips_possession_phase(self):
        s = with_phase(make_sequence(t=5), [0, 1, 2, 3, 4])
        swapped = s.swap_perspective()
        np.testing.assert_array_equal(
            swapped.phase,
            [Phase.UNKNOWN, Phase.OUT_OF_POSSESSION, Phase.IN_POSSESSION,
             Phase.TRANSITION, Phase.SET_PIECE],
        )

    def test_swap_perspective_phase_is_an_involution(self):
        s = with_phase(make_sequence(t=5), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(s.swap_perspective().swap_perspective().phase, s.phase)


class TestDerivePhase:
    def test_unknown_without_possession(self):
        p = derive_phase(None, None, 25.0, 10)
        assert (p == Phase.UNKNOWN).all()

    def test_maps_possession_to_phases(self):
        """Sequence must be long enough that the ends sit outside the +/-1.5s
        transition window, or every frame is legitimately TRANSITION."""
        owning = np.zeros(200, dtype=np.int8)
        owning[100:] = 1
        p = derive_phase(owning, None, 25.0, 200)
        assert p[0] == Phase.IN_POSSESSION
        assert p[-1] == Phase.OUT_OF_POSSESSION

    def test_short_sequence_around_a_turnover_is_all_transition(self):
        owning = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        p = derive_phase(owning, None, 25.0, 6)
        assert (p == Phase.TRANSITION).all()

    def test_turnover_produces_transition(self):
        """Without hysteresis, frame-wise labels flicker at every turnover."""
        owning = np.zeros(200, dtype=np.int8)
        owning[100:] = 1
        p = derive_phase(owning, None, 25.0, 200)
        assert p[99] == Phase.TRANSITION
        assert p[100] == Phase.TRANSITION
        assert p[0] == Phase.IN_POSSESSION
        assert p[-1] == Phase.OUT_OF_POSSESSION

    def test_set_piece_takes_precedence(self):
        owning = np.zeros(20, dtype=np.int8)
        sp = np.zeros(20, dtype=bool)
        sp[5:10] = True
        p = derive_phase(owning, sp, 25.0, 20)
        assert (p[5:10] == Phase.SET_PIECE).all()
        assert p[0] == Phase.IN_POSSESSION

    def test_unknown_possession_stays_unknown(self):
        owning = np.full(20, -1, dtype=np.int8)
        p = derive_phase(owning, None, 25.0, 20)
        assert (p == Phase.UNKNOWN).all()
