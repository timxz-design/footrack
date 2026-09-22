"""Invariants of the canonical state representation.

The transforms here are augmentation primitives: if `mirror` or
`swap_perspective` is subtly wrong, training silently learns on corrupted data
and nothing downstream complains. These tests are the guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from footrack import BALL, N_ENTITIES, TEAM_A, TEAM_B, Sequence, SlotMap
from footrack import pitch


def make_sequence(t: int = 40, seed: int = 0) -> Sequence:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-50, 50, size=(t, N_ENTITIES, 2)).astype(np.float32)
    velocities = rng.uniform(-8, 8, size=(t, N_ENTITIES, 2)).astype(np.float32)
    mask = rng.random((t, N_ENTITIES)) > 0.05
    velocity_valid = mask & (rng.random((t, N_ENTITIES)) > 0.05)
    slots = SlotMap(
        player_ids=(None,) + tuple(f"p{i}" for i in range(22)),
        jersey_numbers=(None,) + tuple(range(1, 23)),
        team_ids=(None,) + ("home",) * 11 + ("away",) * 11,
    )
    return Sequence(
        positions=positions,
        velocities=velocities,
        velocity_valid=velocity_valid,
        mask=mask,
        frame_ids=np.arange(t, dtype=np.int64),
        frame_rate=25.0,
        slots=slots,
        match_id="test",
        period_id=1,
        ball_owning_slot_block=np.zeros(t, dtype=np.int8),
    )


class TestMirror:
    def test_is_an_involution(self):
        s = make_sequence()
        np.testing.assert_allclose(s.mirror().mirror().positions, s.positions)
        np.testing.assert_allclose(s.mirror().mirror().velocities, s.velocities)

    def test_flips_y_only(self):
        s = make_sequence()
        m = s.mirror()
        np.testing.assert_allclose(m.positions[..., 0], s.positions[..., 0])
        np.testing.assert_allclose(m.positions[..., 1], -s.positions[..., 1])

    def test_preserves_distances(self):
        """The whole point: a mirrored sequence is geometrically identical."""
        s = make_sequence()
        d = np.linalg.norm(s.positions[:, 1] - s.positions[:, 2], axis=-1)
        dm = np.linalg.norm(s.mirror().positions[:, 1] - s.mirror().positions[:, 2], axis=-1)
        np.testing.assert_allclose(d, dm, rtol=1e-5)

    def test_does_not_change_team_assignment(self):
        s = make_sequence()
        assert s.mirror().slots.team_ids == s.slots.team_ids


class TestSwapPerspective:
    def test_is_an_involution(self):
        s = make_sequence()
        back = s.swap_perspective().swap_perspective()
        np.testing.assert_allclose(back.positions, s.positions)
        np.testing.assert_allclose(back.velocities, s.velocities)
        np.testing.assert_array_equal(back.mask, s.mask)
        assert back.slots.player_ids == s.slots.player_ids

    def test_exchanges_team_blocks(self):
        s = make_sequence()
        w = s.swap_perspective()
        np.testing.assert_allclose(w.positions[:, TEAM_A], -s.positions[:, TEAM_B])
        np.testing.assert_allclose(w.positions[:, TEAM_B], -s.positions[:, TEAM_A])

    def test_ball_stays_in_slot_zero(self):
        s = make_sequence()
        np.testing.assert_allclose(s.swap_perspective().positions[:, BALL], -s.positions[:, BALL])

    def test_slotmap_follows_the_data(self):
        s = make_sequence()
        w = s.swap_perspective()
        assert w.slots.player_ids[1:12] == s.slots.player_ids[12:23]
        assert w.slots.player_ids[12:23] == s.slots.player_ids[1:12]

    def test_ball_ownership_flips(self):
        s = make_sequence()
        assert np.all(s.swap_perspective().ball_owning_slot_block == 1)

    def test_unknown_ownership_stays_unknown(self):
        s = make_sequence()
        s = type(s)(**{**s.__dict__, "ball_owning_slot_block": np.full(len(s), -1, np.int8)})
        assert np.all(s.swap_perspective().ball_owning_slot_block == -1)

    def test_preserves_distances(self):
        s = make_sequence()
        d = np.linalg.norm(s.positions[:, 1] - s.positions[:, 13], axis=-1)
        w = s.swap_perspective()
        dw = np.linalg.norm(w.positions[:, 12] - w.positions[:, 2], axis=-1)
        np.testing.assert_allclose(d, dw, rtol=1e-5)


class TestWindowing:
    def test_window_shape_and_content(self):
        s = make_sequence(t=40)
        w = s.window(10, 15)
        assert len(w) == 15
        np.testing.assert_allclose(w.positions, s.positions[10:25])
        np.testing.assert_array_equal(w.frame_ids, s.frame_ids[10:25])

    def test_window_out_of_range_raises(self):
        s = make_sequence(t=40)
        with pytest.raises(IndexError):
            s.window(30, 20)

    def test_downsample_adjusts_frame_rate(self):
        s = make_sequence(t=40)
        d = s.downsample(5)
        assert len(d) == 8
        assert d.frame_rate == pytest.approx(5.0)

    def test_downsample_identity(self):
        s = make_sequence()
        np.testing.assert_allclose(s.downsample(1).positions, s.positions)


class TestValidation:
    def test_mismatched_shapes_rejected(self):
        s = make_sequence()
        with pytest.raises(ValueError, match="velocity_valid"):
            Sequence(**{**s.__dict__, "velocity_valid": np.zeros((len(s), 5), dtype=bool)})

    def test_slotmap_length_enforced(self):
        with pytest.raises(ValueError):
            SlotMap(("a",), (1,), ("home",))


class TestPitchGeometry:
    def test_kloppy_roundtrip(self):
        xy = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
        np.testing.assert_allclose(pitch.to_kloppy(pitch.from_kloppy(xy)), xy, atol=1e-5)

    def test_centre_maps_to_origin(self):
        np.testing.assert_allclose(
            pitch.from_kloppy(np.array([0.5, 0.5], dtype=np.float32)), [0.0, 0.0], atol=1e-5
        )

    def test_model_scale_is_isotropic(self):
        """A 10m displacement must have the same magnitude whichever axis it is on."""
        dx = pitch.to_model_scale(np.array([10.0, 0.0], dtype=np.float32))
        dy = pitch.to_model_scale(np.array([0.0, 10.0], dtype=np.float32))
        assert np.linalg.norm(dx) == pytest.approx(np.linalg.norm(dy))

    def test_axis_norm_reaches_unit_at_touchlines(self):
        corner = np.array([pitch.HALF_LENGTH, pitch.HALF_WIDTH], dtype=np.float32)
        np.testing.assert_allclose(pitch.to_axis_norm(corner), [1.0, 1.0], atol=1e-5)

    def test_channel_boundaries(self):
        w = pitch.HALF_WIDTH
        ys = np.array([-w, -0.7 * w, -0.4 * w, 0.0, 0.4 * w, 0.7 * w, w], dtype=np.float32)
        np.testing.assert_array_equal(pitch.channel_of(ys), [0, 0, 1, 2, 3, 4, 4])

    def test_thirds(self):
        half = pitch.HALF_LENGTH
        xs = np.array([-half, -0.5 * half, 0.0, 0.5 * half, half], dtype=np.float32)
        np.testing.assert_array_equal(pitch.third_of(xs), [0, 0, 1, 2, 2])

    def test_bearing_is_unit_length(self):
        xy = np.array([[0.0, 0.0], [20.0, -15.0]], dtype=np.float32)
        b = pitch.bearing_to(xy, pitch.OPP_GOAL)
        np.testing.assert_allclose(np.linalg.norm(b, axis=-1), 1.0, atol=1e-5)

    def test_bearing_at_target_is_zero_not_nan(self):
        b = pitch.bearing_to(pitch.OPP_GOAL.copy(), pitch.OPP_GOAL)
        assert np.isfinite(b).all()
        np.testing.assert_allclose(b, [0.0, 0.0])
