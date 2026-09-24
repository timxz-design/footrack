"""Session plans — the unit a coach actually works in.

A coach does not think "show me a 4-3-3". They think "we're doing the pressing
pattern, then build-up from the back". So the primary object is a **session**:
an ordered list of drills, each a sequence of situations.

This exists because authoring and operating happen at different times under
completely different constraints. Authoring is a desk activity the night before,
with time to think. Operating is done standing on a pitch with twenty players
waiting, where the coach has roughly three seconds of attention to spare before
they give up on the device and just shout.

Putting the work in authoring is what lets the pitch-side interaction collapse
to a single tap. **The scarce resource is the coach's attention, not the tool's
expressiveness.**
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from footrack.state import Phase

DEFAULT_ROOT = Path("data/sessions")


@dataclass
class Step:
    """One situation within a drill: where the ball is and what shape to hold."""

    ball: tuple[float, float] = (0.0, 0.0)  # metres, centre-origin
    phase: str = Phase.IN_POSSESSION.name
    note: str = ""
    # Slot -> position. A pinned player is held exactly there and the rest
    # re-solve around them, which is how a coach expresses "he stays here".
    pinned: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ball = (float(self.ball[0]), float(self.ball[1]))
        if self.phase not in Phase.__members__:
            raise ValueError(f"unknown phase {self.phase!r}")
        self.pinned = {
            str(k): (float(v[0]), float(v[1])) for k, v in (self.pinned or {}).items()
        }


@dataclass
class Drill:
    """A shape to practise, and the situations to walk through."""

    name: str
    formation: str = "4-3-3"
    line_height: float = 0.0
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.steps = [s if isinstance(s, Step) else Step(**s) for s in self.steps]
        if not self.name.strip():
            raise ValueError("a drill needs a name")


@dataclass
class Session:
    name: str
    created: str = ""
    drills: list[Drill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.drills = [d if isinstance(d, Drill) else Drill(**d) for d in self.drills]
        if not self.created:
            self.created = date.today().isoformat()

    @property
    def slug(self) -> str:
        """Filesystem-safe name. Runs of separators collapse to one.

        A single ``replace("--", "-")`` is not enough: "a / b" produces three
        separators, and one pass leaves two behind.
        """
        import re

        keep = "".join(c.lower() if c.isalnum() else "-" for c in self.name.strip())
        return re.sub(r"-+", "-", keep).strip("-") or "session"


class SessionStore:
    """One JSON file per session, under ``data/sessions``.

    Plain readable JSON rather than a database: a coach's session plan is
    content they should be able to copy, mail or edit by hand, and there will
    never be enough of them to need indexing.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, slug: str) -> Path:
        return self.root / f"{slug}.json"

    def save(self, session: Session) -> Path:
        path = self.path_for(session.slug)
        path.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")
        return path

    def load(self, slug: str) -> Session:
        path = self.path_for(slug)
        if not path.exists():
            raise FileNotFoundError(f"no session {slug!r} in {self.root}")
        return Session(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            out.append({
                "slug": path.stem,
                "name": raw.get("name", path.stem),
                "created": raw.get("created", ""),
                "drills": len(raw.get("drills", [])),
                "steps": sum(len(d.get("steps", [])) for d in raw.get("drills", [])),
            })
        return out

    def delete(self, slug: str) -> bool:
        path = self.path_for(slug)
        if path.exists():
            path.unlink()
            return True
        return False


def example_session() -> Session:
    """A plausible session, so the tool is never empty on first run.

    An empty tool asks the coach to do setup work before they can see whether
    it is any good, which is the fastest way to lose them.
    """
    return Session(
        name="Example session",
        drills=[
            Drill(
                name="Build-up from the back",
                formation="4-3-3",
                line_height=-0.1,
                steps=[
                    Step(ball=(-45.0, 0.0), note="Keeper has it — split the centre-backs"),
                    Step(ball=(-38.0, -18.0), note="Out to the left-back"),
                    Step(ball=(-20.0, -22.0), note="Full-back progresses, winger holds width"),
                    Step(ball=(0.0, -8.0), note="Inside to the eight"),
                ],
            ),
            Drill(
                name="Mid block, shift and press",
                formation="4-4-2",
                line_height=0.0,
                steps=[
                    Step(ball=(10.0, 0.0), phase=Phase.OUT_OF_POSSESSION.name,
                         note="Set the block — 28m, stay compact"),
                    Step(ball=(5.0, 24.0), phase=Phase.OUT_OF_POSSESSION.name,
                         note="Ball goes wide right — whole block shifts"),
                    Step(ball=(-5.0, 28.0), phase=Phase.OUT_OF_POSSESSION.name,
                         note="Press trigger: back-pass to their full-back"),
                    Step(ball=(-15.0, 10.0), phase=Phase.TRANSITION.name,
                         note="Won it — first thought forward"),
                ],
            ),
        ],
    )
