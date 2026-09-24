"""Session plans — the coach's authored content.

Like the label corpus, this is the user's own work rather than something
regenerable, so the store is plain readable JSON and the tests are mostly about
not losing or corrupting it.
"""

from __future__ import annotations

import json

import pytest

from footrack.sessions import Drill, Session, SessionStore, Step, example_session
from footrack.state import Phase


class TestStep:
    def test_defaults_to_the_centre_in_possession(self):
        s = Step()
        assert s.ball == (0.0, 0.0)
        assert s.phase == Phase.IN_POSSESSION.name

    def test_coerces_ball_to_floats(self):
        s = Step(ball=(1, 2))
        assert s.ball == (1.0, 2.0)
        assert all(isinstance(v, float) for v in s.ball)

    def test_rejects_unknown_phase(self):
        with pytest.raises(ValueError, match="unknown phase"):
            Step(phase="SOMETHING_ELSE")

    def test_accepts_every_real_phase(self):
        for p in Phase:
            assert Step(phase=p.name).phase == p.name

    def test_pinned_keys_normalise_to_strings(self):
        """JSON object keys are always strings, so a round trip would otherwise
        change the type of every key."""
        s = Step(pinned={3: (1.0, 2.0)})
        assert set(s.pinned) == {"3"}
        assert s.pinned["3"] == (1.0, 2.0)


class TestDrill:
    def test_requires_a_name(self):
        with pytest.raises(ValueError, match="needs a name"):
            Drill(name="   ")

    def test_rehydrates_steps_from_dicts(self):
        d = Drill(name="x", steps=[{"ball": (1.0, 2.0), "note": "hi"}])
        assert isinstance(d.steps[0], Step)
        assert d.steps[0].note == "hi"

    def test_accepts_step_objects(self):
        d = Drill(name="x", steps=[Step(note="already built")])
        assert d.steps[0].note == "already built"


class TestSession:
    def test_slug_is_filesystem_safe(self):
        assert Session(name="Tuesday: pressing / shape!").slug == "tuesday-pressing-shape"

    def test_slug_never_empty(self):
        assert Session(name="!!!").slug == "session"

    def test_dates_itself(self):
        assert Session(name="x").created.startswith("20")

    def test_rehydrates_drills_from_dicts(self):
        s = Session(name="x", drills=[{"name": "d", "steps": [{"note": "n"}]}])
        assert isinstance(s.drills[0], Drill)
        assert s.drills[0].steps[0].note == "n"


class TestStore:
    def test_round_trip(self, tmp_path):
        store = SessionStore(tmp_path)
        original = example_session()
        store.save(original)
        loaded = store.load(original.slug)

        assert loaded.name == original.name
        assert len(loaded.drills) == len(original.drills)
        assert loaded.drills[1].steps[2].note == original.drills[1].steps[2].note
        assert loaded.drills[1].steps[0].phase == Phase.OUT_OF_POSSESSION.name

    def test_written_as_readable_json(self, tmp_path):
        """A session plan is content a coach should be able to mail or hand-edit."""
        store = SessionStore(tmp_path)
        store.save(example_session())
        raw = (tmp_path / "example-session.json").read_text(encoding="utf-8")
        assert "\n" in raw
        assert json.loads(raw)["drills"][0]["name"]

    def test_missing_session_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SessionStore(tmp_path).load("nope")

    def test_list_summarises(self, tmp_path):
        store = SessionStore(tmp_path)
        store.save(example_session())
        rows = store.list()
        assert len(rows) == 1
        assert rows[0]["drills"] == 2
        assert rows[0]["steps"] == 8

    def test_list_skips_corrupt_files(self, tmp_path):
        store = SessionStore(tmp_path)
        store.save(example_session())
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert len(store.list()) == 1

    def test_saving_again_overwrites_in_place(self, tmp_path):
        store = SessionStore(tmp_path)
        s = example_session()
        store.save(s)
        s.drills = s.drills[:1]
        store.save(s)
        assert len(store.load(s.slug).drills) == 1
        assert len(store.list()) == 1

    def test_delete(self, tmp_path):
        store = SessionStore(tmp_path)
        store.save(example_session())
        assert store.delete("example-session") is True
        assert store.list() == []

    def test_delete_missing_is_false_not_an_error(self, tmp_path):
        assert SessionStore(tmp_path).delete("nope") is False

    def test_creates_its_directory(self, tmp_path):
        nested = tmp_path / "a" / "b"
        SessionStore(nested)
        assert nested.exists()


class TestExampleSession:
    def test_is_valid(self):
        """Ships so the tool is never empty on first run — an empty tool asks
        for setup before showing whether it is any good."""
        s = example_session()
        assert len(s.drills) >= 2
        assert all(d.steps for d in s.drills)

    def test_covers_both_phases(self):
        phases = {st.phase for d in example_session().drills for st in d.steps}
        assert Phase.IN_POSSESSION.name in phases
        assert Phase.OUT_OF_POSSESSION.name in phases

    def test_every_step_has_a_note(self):
        for d in example_session().drills:
            for st in d.steps:
                assert st.note, f"{d.name} has a step with no coaching point"
