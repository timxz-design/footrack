"""Cache round-tripping, windowing, splitting and sampling.

The split test matters most. Splitting on frames rather than matches produces a
validation score that looks excellent and means nothing, because adjacent frames
are near-duplicates sitting on both sides of the boundary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from footrack import (
    N_ENTITIES,
    Phase,
    WindowDataset,
    collate,
    constant_velocity_baseline,
    load_sequences,
    save_sequences,
    split_by_match,
)
from footrack.dataset import DEFAULT_PHASE_WEIGHTS
from footrack.features import POS_SCALE

from test_state import make_sequence


def seq_for(match_id: str, t: int = 400, seed: int = 0):
    s = make_sequence(t=t, seed=seed)
    phase = np.full(t, Phase.IN_POSSESSION, dtype=np.int8)
    phase[t // 2 :] = Phase.TRANSITION
    return type(s)(**{**s.__dict__, "match_id": match_id, "phase": phase})


class TestCache:
    def test_round_trip_preserves_arrays(self, tmp_path):
        original = [seq_for("m1", t=120, seed=1), seq_for("m1", t=90, seed=2)]
        save_sequences(original, tmp_path, "m1")
        loaded = load_sequences(tmp_path, "m1")

        assert len(loaded) == len(original)
        for a, b in zip(original, loaded):
            np.testing.assert_allclose(a.positions, b.positions)
            np.testing.assert_allclose(a.velocities, b.velocities)
            np.testing.assert_array_equal(a.mask, b.mask)
            np.testing.assert_array_equal(a.velocity_valid, b.velocity_valid)
            np.testing.assert_array_equal(a.frame_ids, b.frame_ids)
            np.testing.assert_array_equal(a.phase, b.phase)

    def test_round_trip_preserves_metadata(self, tmp_path):
        original = [seq_for("m1", t=80)]
        save_sequences(original, tmp_path, "m1")
        b = load_sequences(tmp_path, "m1")[0]
        a = original[0]
        assert (b.match_id, b.period_id, b.frame_rate) == (a.match_id, a.period_id, a.frame_rate)
        assert b.slots.player_ids == a.slots.player_ids
        assert b.slots.jersey_numbers == a.slots.jersey_numbers

    def test_optional_fields_survive_being_absent(self, tmp_path):
        s = seq_for("m1", t=60)
        s = type(s)(**{**s.__dict__, "phase": None, "ball_owning_slot_block": None})
        save_sequences([s], tmp_path, "m1")
        loaded = load_sequences(tmp_path, "m1")[0]
        assert loaded.phase is None
        assert loaded.ball_owning_slot_block is None

    def test_missing_cache_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sequences(tmp_path, "nope")

    def test_version_mismatch_raises(self, tmp_path):
        save_sequences([seq_for("m1", t=60)], tmp_path, "m1")
        path = next((tmp_path / "m1").glob("seq*.npz"))
        with np.load(path, allow_pickle=False) as z:
            arrays = dict(z)
        meta = json.loads(str(arrays["meta"]))
        meta["version"] = 999
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)

        with pytest.raises(ValueError, match="cache version"):
            load_sequences(tmp_path, "m1")

    def test_resave_removes_stale_sequences(self, tmp_path):
        save_sequences([seq_for("m1"), seq_for("m1"), seq_for("m1")], tmp_path, "m1")
        save_sequences([seq_for("m1")], tmp_path, "m1")
        assert len(load_sequences(tmp_path, "m1")) == 1


class TestSplit:
    def test_splits_on_match_not_frames(self):
        seqs = [seq_for("a"), seq_for("a"), seq_for("b"), seq_for("c")]
        train, val = split_by_match(seqs, val_matches=["b"])
        assert {s.match_id for s in train} == {"a", "c"}
        assert {s.match_id for s in val} == {"b"}

    def test_no_match_appears_on_both_sides(self):
        seqs = [seq_for(m) for m in ("a", "b", "c", "d", "e")]
        train, val = split_by_match(seqs, val_fraction=0.4, seed=3)
        assert not ({s.match_id for s in train} & {s.match_id for s in val})

    def test_every_sequence_is_placed(self):
        seqs = [seq_for(m) for m in ("a", "b", "c", "d")]
        train, val = split_by_match(seqs, val_fraction=0.5, seed=1)
        assert len(train) + len(val) == len(seqs)

    def test_unknown_val_match_rejected(self):
        with pytest.raises(ValueError, match="not present"):
            split_by_match([seq_for("a")], val_matches=["zzz"])


class TestWindowDataset:
    def build(self, **kw):
        defaults = dict(history=10, horizon=5, stride=5, target_rate=25.0, min_coverage=0.0)
        return WindowDataset([seq_for("m1", t=200)], **{**defaults, **kw})

    def test_sample_shapes(self):
        ds = self.build()
        s = ds[0]
        assert s.inputs.entity.shape[:2] == (10, N_ENTITIES)
        assert s.target_positions.shape == (5, N_ENTITIES, 2)
        assert s.anchor.shape == (N_ENTITIES, 2)

    def test_windows_do_not_cross_sequence_boundaries(self):
        ds = WindowDataset(
            [seq_for("m1", t=60), seq_for("m1", t=60)],
            history=10, horizon=5, stride=5, target_rate=25.0, min_coverage=0.0,
        )
        for si, start in ds.index:
            assert start + 15 <= len(ds.sequences[si])

    def test_too_short_sequences_are_dropped(self):
        ds = WindowDataset(
            [seq_for("m1", t=5)], history=10, horizon=5, target_rate=25.0, min_coverage=0.0
        )
        assert len(ds) == 0

    def test_target_deltas_are_relative_to_anchor(self):
        s = self.build()[0]
        np.testing.assert_allclose(
            s.target_deltas(), s.target_positions - s.anchor[None, :, :], atol=1e-6
        )

    def test_anchor_matches_last_history_position(self):
        ds = self.build()
        si, start = ds.index[0]
        seq = ds.sequences[si]
        expected = seq.positions[start + ds.history - 1] / POS_SCALE
        s = ds[0]
        np.testing.assert_allclose(s.anchor[s.anchor_valid], expected[s.anchor_valid], atol=1e-5)

    def test_min_coverage_filters_windows(self):
        loose = self.build(min_coverage=0.0)
        strict = self.build(min_coverage=1.0)
        assert len(strict) < len(loose)

    def test_augmentation_is_off_by_default(self):
        ds = self.build()
        np.testing.assert_allclose(ds[0].inputs.entity, ds[0].inputs.entity)

    def test_augmentation_changes_samples(self):
        ds = self.build(augment=True, seed=7)
        variants = {ds[0].inputs.entity.tobytes() for _ in range(12)}
        assert len(variants) > 1, "augmented dataset returned identical samples"

    def test_augmentation_preserves_shapes(self):
        ds = self.build(augment=True, seed=7)
        for _ in range(5):
            s = ds[0]
            assert s.inputs.entity.shape[:2] == (10, N_ENTITIES)
            assert s.target_positions.shape == (5, N_ENTITIES, 2)


class TestSampling:
    def test_weights_are_normalised(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        assert ds.sampling_weights().sum() == pytest.approx(1.0)

    def test_transition_is_upweighted(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        phases = ds.phase_of_windows()
        w = ds.sampling_weights()
        trans = w[phases == Phase.TRANSITION].sum() / max((phases == Phase.TRANSITION).sum(), 1)
        poss = w[phases == Phase.IN_POSSESSION].sum() / max((phases == Phase.IN_POSSESSION).sum(), 1)
        assert trans > poss

    def test_custom_weights_respected(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        flat = ds.sampling_weights({p: 1.0 for p in Phase})
        np.testing.assert_allclose(flat, np.full(len(ds), 1 / len(ds)))

    def test_default_table_covers_every_phase(self):
        assert set(DEFAULT_PHASE_WEIGHTS) == set(Phase)


class TestCollate:
    def test_batch_shapes(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        batch = collate([ds[0], ds[1], ds[2]])
        assert batch["entity"].shape[0] == 3
        assert batch["target_positions"].shape == (3, 5, N_ENTITIES, 2)
        assert batch["anchor_valid"].shape == (3, N_ENTITIES)

    def test_batches_iterator_yields_full_batches(self):
        ds = WindowDataset(
            [seq_for("m1", t=400)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0, seed=0,
        )
        sizes = [b["entity"].shape[0] for b in ds.batches(batch_size=8)]
        assert sizes and all(s == 8 for s in sizes)


class TestBaseline:
    def test_zero_velocity_predicts_no_movement(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        s = ds[0]
        s.inputs.entity[-1, :, 2:4] = 0.0
        pred = constant_velocity_baseline(s, 25.0)
        for step in range(pred.shape[0]):
            np.testing.assert_allclose(pred[step], s.anchor, atol=1e-6)

    def test_prediction_shape(self):
        ds = WindowDataset(
            [seq_for("m1", t=200)], history=10, horizon=5, stride=5,
            target_rate=25.0, min_coverage=0.0,
        )
        s = ds[0]
        assert constant_velocity_baseline(s, 25.0).shape == s.target_positions.shape
