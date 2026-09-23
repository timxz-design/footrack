"""Label store invariants.

This corpus is the project's only irreplaceable asset — architecture and
datasets are public, coach judgement is not. The split test matters most: if a
frame can drift between train and eval, the coach-agreement metric is quietly
contaminated and there is no way to notice after the fact.
"""

from __future__ import annotations

import pytest

from footrack.labeling import Correction, Label, LabelStore, assign_split


def make_label(key="m1/1/100", verdict="agree", **kw) -> Label:
    return Label(
        frame_key=key,
        match_id=kw.pop("match_id", "m1"),
        sequence_index=kw.pop("sequence_index", 0),
        frame_index=kw.pop("frame_index", 100),
        verdict=verdict,
        **kw,
    )


class TestSplitAssignment:
    def test_is_deterministic(self):
        assert assign_split("m1/1/100") == assign_split("m1/1/100")

    def test_differs_across_frames(self):
        splits = {assign_split(f"m/1/{i}") for i in range(200)}
        assert splits == {"train", "eval"}

    def test_roughly_matches_requested_fraction(self):
        keys = [f"m/1/{i}" for i in range(4000)]
        share = sum(assign_split(k) == "eval" for k in keys) / len(keys)
        assert 0.16 < share < 0.24, f"eval share {share:.3f} far from 0.2"

    def test_fraction_is_configurable(self):
        keys = [f"m/1/{i}" for i in range(2000)]
        share = sum(assign_split(k, 0.5) == "eval" for k in keys) / len(keys)
        assert 0.45 < share < 0.55

    def test_relabelling_keeps_the_same_side(self):
        """A frame drifting between splits would silently contaminate eval."""
        first = make_label(verdict="agree")
        second = make_label(verdict="correct")
        assert first.split == second.split


class TestLabel:
    def test_rejects_unknown_verdict(self):
        with pytest.raises(ValueError, match="unknown verdict"):
            make_label(verdict="maybe")

    def test_accepts_the_three_verdicts(self):
        for v in ("agree", "correct", "skip"):
            assert make_label(verdict=v).verdict == v

    def test_timestamps_itself(self):
        assert make_label().created_at.startswith("20")

    def test_preserves_explicit_timestamp(self):
        stamp = "2020-01-01T00:00:00+00:00"
        assert make_label(created_at=stamp).created_at == stamp


class TestLabelStore:
    def test_round_trip(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(
            verdict="correct",
            corrections=[Correction(5, (1.0, 2.0), (3.0, 4.0))],
            note="too narrow",
        ))
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].note == "too narrow"
        assert loaded[0].corrections[0].slot == 5
        assert loaded[0].corrections[0].to_xy == (3.0, 4.0)

    def test_coordinates_stay_tuples_across_a_round_trip(self, tmp_path):
        """JSON has no tuples; without coercion a loaded correction would differ
        in type from one built in memory."""
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(
            verdict="correct", corrections=[Correction(5, (1.0, 2.0), (3.0, 4.0))]
        ))
        c = store.load()[0].corrections[0]
        assert isinstance(c.from_xy, tuple) and isinstance(c.to_xy, tuple)

    def test_correction_distance(self):
        assert Correction(1, (0.0, 0.0), (3.0, 4.0)).distance() == pytest.approx(5.0)

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert LabelStore(tmp_path / "nope.jsonl").load() == []

    def test_appends_rather_than_overwrites(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        for i in range(3):
            store.append(make_label(key=f"m1/1/{i}"))
        assert len(store.load()) == 3

    def test_later_judgement_supersedes(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(key="m1/1/7", verdict="agree"))
        store.append(make_label(key="m1/1/7", verdict="correct"))
        latest = store.latest_by_frame()
        assert len(latest) == 1
        assert latest["m1/1/7"].verdict == "correct"

    def test_history_is_retained_after_supersede(self, tmp_path):
        """Append-only: the earlier judgement stays on disk for audit."""
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(key="m1/1/7", verdict="agree"))
        store.append(make_label(key="m1/1/7", verdict="correct"))
        assert len(store.load()) == 2


class TestStats:
    def test_empty_store(self, tmp_path):
        s = LabelStore(tmp_path / "c.jsonl").stats()
        assert s["frames_judged"] == 0
        assert s["agreement_rate"] is None

    def test_agreement_rate_excludes_skips(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(key="a", verdict="agree"))
        store.append(make_label(key="b", verdict="agree"))
        store.append(make_label(key="c", verdict="correct"))
        store.append(make_label(key="d", verdict="skip"))
        s = store.stats()
        assert s["frames_judged"] == 3
        assert s["skipped"] == 1
        assert s["agreement_rate"] == pytest.approx(2 / 3)

    def test_counts_player_moves(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        store.append(make_label(
            key="a", verdict="correct",
            corrections=[Correction(1, (0, 0), (1, 1)), Correction(2, (0, 0), (2, 2))],
        ))
        assert store.stats()["player_moves"] == 2

    def test_split_counts_add_up(self, tmp_path):
        store = LabelStore(tmp_path / "c.jsonl")
        for i in range(50):
            store.append(make_label(key=f"m/1/{i}", verdict="agree"))
        s = store.stats()
        assert s["train"] + s["eval"] == s["frames_judged"]
