"""Event data, used to derive possession and phase of play.

Tracking alone cannot tell you who has the ball. kloppy's Metrica open-data
loader returns tracking only, so ``ball_owning_team`` is empty and every frame
would otherwise be labelled UNKNOWN — leaving a required model input dead.

This module reads the matching event feed and derives a per-frame possession and
set-piece series from it. Metrica is the first implementation; PFF and SkillCorner
ship their own event feeds, and SkillCorner's open data even includes phase-of-play
labels directly (DATA.md §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICA_EVENTS_URL = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
    "data/Sample_Game_{match_id}/Sample_Game_{match_id}_RawEventsData.csv"
)

# Events that establish which team is on the ball.
ON_BALL_TYPES = frozenset({"PASS", "SHOT", "RECOVERY", "SET PIECE", "CHALLENGE"})

# Events after which nobody is in possession.
DEAD_BALL_TYPES = frozenset({"BALL OUT"})

SET_PIECE_WINDOW_S = 2.0


def load_metrica_events(match_id: int = 1) -> pd.DataFrame:
    """Fetch the Metrica event feed for a sample match (games 1 and 2 only)."""
    return pd.read_csv(METRICA_EVENTS_URL.format(match_id=match_id))


def possession_series(
    events: pd.DataFrame,
    frame_ids: np.ndarray,
    home_is_team_a: bool = True,
) -> np.ndarray:
    """(T,) int8 possession per frame: 0 = team A, 1 = team B, -1 = unknown.

    Possession is carried forward from each on-ball event until the next one,
    and cleared when the ball goes out. ``BALL LOST`` is treated as on-ball for
    the losing team up to that moment, then superseded by whatever follows —
    usually the opponent's RECOVERY.
    """
    usable = events[events["Type"].isin(ON_BALL_TYPES | DEAD_BALL_TYPES)]
    usable = usable.sort_values("Start Frame")
    if usable.empty:
        return np.full(len(frame_ids), -1, dtype=np.int8)

    starts = usable["Start Frame"].to_numpy(dtype=np.int64)
    is_home = (usable["Team"] == "Home").to_numpy()
    team_a = is_home if home_is_team_a else ~is_home

    values = np.where(team_a, 0, 1).astype(np.int8)
    values[usable["Type"].isin(DEAD_BALL_TYPES).to_numpy()] = -1

    # Each frame takes the value of the most recent preceding event.
    idx = np.searchsorted(starts, frame_ids, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], -1)
    return out.astype(np.int8)


def set_piece_series(
    events: pd.DataFrame, frame_ids: np.ndarray, frame_rate: float
) -> np.ndarray:
    """(T,) bool — frames within the restart window after a set piece."""
    sp = events[events["Type"] == "SET PIECE"]
    out = np.zeros(len(frame_ids), dtype=bool)
    if sp.empty:
        return out

    window = int(round(SET_PIECE_WINDOW_S * frame_rate))
    for start in sp["Start Frame"].to_numpy(dtype=np.int64):
        lo = np.searchsorted(frame_ids, start, side="left")
        hi = np.searchsorted(frame_ids, start + window, side="right")
        out[lo:hi] = True
    return out
