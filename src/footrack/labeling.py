"""Storage for coach corrections.

This corpus is the project's actual moat (PLAN.md §1.2). The architecture and
the datasets are public; coach judgement labelled against positioning is not.
So it is stored append-only, in a plain readable format, with enough provenance
to reconstruct any label months later.

**Corrections, not placements.** A label records which players were moved and
where — untouched players are implicitly acceptable. Recognising an error is a
far more reliable judgement than pinpointing an optimum, so the signal worth
capturing is the disagreement, not an absolute answer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STORE = Path("data/labels/corrections.jsonl")

# Held out from training to serve as the coach-agreement metric (MODEL.md §3.2).
# Assignment is by hash of the frame key, so it is stable across sessions and a
# frame can never drift between train and eval.
EVAL_FRACTION = 0.2


@dataclass
class Correction:
    """One player moved from where they were to where they should have been."""

    slot: int
    from_xy: tuple[float, float]  # metres, centre-origin
    to_xy: tuple[float, float]

    def __post_init__(self) -> None:
        # JSON has no tuple type, so a round-tripped correction would otherwise
        # come back as lists and differ in type from one built in memory.
        self.from_xy = (float(self.from_xy[0]), float(self.from_xy[1]))
        self.to_xy = (float(self.to_xy[0]), float(self.to_xy[1]))

    def distance(self) -> float:
        """How far the player was moved, in metres."""
        return float(
            ((self.to_xy[0] - self.from_xy[0]) ** 2
             + (self.to_xy[1] - self.from_xy[1]) ** 2) ** 0.5
        )


@dataclass
class Label:
    """One judged frame."""

    frame_key: str  # match/sequence/frame — stable, reconstructible
    match_id: str
    sequence_index: int
    frame_index: int
    verdict: str  # "agree" | "correct" | "skip"
    corrections: list[Correction] = field(default_factory=list)
    note: str = ""
    labeler: str = "unknown"
    created_at: str = ""
    split: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.split:
            self.split = assign_split(self.frame_key)
        if self.verdict not in {"agree", "correct", "skip"}:
            raise ValueError(f"unknown verdict {self.verdict!r}")


def assign_split(frame_key: str, eval_fraction: float = EVAL_FRACTION) -> str:
    """Deterministic train/eval assignment from the frame key.

    Hash-based rather than random so that re-labelling a frame, or rebuilding
    the corpus, always puts it on the same side. A frame that drifts between
    splits silently contaminates the only metric that matters.
    """
    import hashlib

    digest = hashlib.sha1(frame_key.encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return "eval" if bucket < eval_fraction else "train"


class LabelStore:
    """Append-only JSONL store."""

    def __init__(self, path: Path | str = DEFAULT_STORE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, label: Label) -> None:
        payload = asdict(label)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def load(self) -> list[Label]:
        if not self.path.exists():
            return []
        out: list[Label] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            raw["corrections"] = [Correction(**c) for c in raw.get("corrections", [])]
            out.append(Label(**raw))
        return out

    def latest_by_frame(self) -> dict[str, Label]:
        """Most recent label per frame — later judgements supersede earlier ones."""
        latest: dict[str, Label] = {}
        for label in self.load():
            latest[label.frame_key] = label
        return latest

    def stats(self) -> dict:
        labels = list(self.latest_by_frame().values())
        judged = [x for x in labels if x.verdict != "skip"]
        corrected = [x for x in judged if x.verdict == "correct"]
        moves = sum(len(x.corrections) for x in corrected)
        return {
            "frames_seen": len(labels),
            "frames_judged": len(judged),
            "agreed": len(judged) - len(corrected),
            "corrected": len(corrected),
            "skipped": len(labels) - len(judged),
            "player_moves": moves,
            "train": sum(1 for x in judged if x.split == "train"),
            "eval": sum(1 for x in judged if x.split == "eval"),
            "agreement_rate": (len(judged) - len(corrected)) / len(judged) if judged else None,
        }
