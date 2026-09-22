"""On-disk cache for parsed sequences.

Parsing a match through kloppy and re-orienting it takes ~87 seconds. At that
cost, loading would dominate every training run and every experiment, so parsed
sequences are written once to a compact binary format and read back in
milliseconds.

Sequences are cached at their **native frame rate**. Downsampling is a dataset
concern, so changing the model's operating rate does not invalidate the cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from footrack.state import Sequence, SlotMap

CACHE_VERSION = 1


def _meta(seq: Sequence) -> str:
    return json.dumps(
        {
            "version": CACHE_VERSION,
            "frame_rate": seq.frame_rate,
            "match_id": seq.match_id,
            "period_id": seq.period_id,
            "player_ids": list(seq.slots.player_ids),
            "jersey_numbers": list(seq.slots.jersey_numbers),
            "team_ids": list(seq.slots.team_ids),
        }
    )


def save_sequences(sequences: list[Sequence], root: Path | str, match_id: str) -> Path:
    """Write a match's sequences to ``root/match_id/``."""
    out = Path(root) / match_id
    out.mkdir(parents=True, exist_ok=True)

    for stale in out.glob("seq*.npz"):
        stale.unlink()

    for i, seq in enumerate(sequences):
        arrays = {
            "positions": seq.positions,
            "velocities": seq.velocities,
            "velocity_valid": seq.velocity_valid,
            "mask": seq.mask,
            "frame_ids": seq.frame_ids,
            "meta": np.array(_meta(seq)),
        }
        if seq.ball_owning_slot_block is not None:
            arrays["ball_owning_slot_block"] = seq.ball_owning_slot_block
        if seq.phase is not None:
            arrays["phase"] = seq.phase
        np.savez_compressed(out / f"seq{i:03d}.npz", **arrays)

    return out


def load_sequences(root: Path | str, match_id: str) -> list[Sequence]:
    """Read back a cached match. Raises FileNotFoundError if absent."""
    path = Path(root) / match_id
    files = sorted(path.glob("seq*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached sequences in {path}")

    sequences = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("version") != CACHE_VERSION:
                raise ValueError(
                    f"{f} was written by cache version {meta.get('version')}, "
                    f"expected {CACHE_VERSION} — delete the cache and rebuild"
                )
            sequences.append(
                Sequence(
                    positions=z["positions"],
                    velocities=z["velocities"],
                    velocity_valid=z["velocity_valid"],
                    mask=z["mask"],
                    frame_ids=z["frame_ids"],
                    frame_rate=meta["frame_rate"],
                    slots=SlotMap(
                        player_ids=tuple(meta["player_ids"]),
                        jersey_numbers=tuple(meta["jersey_numbers"]),
                        team_ids=tuple(meta["team_ids"]),
                    ),
                    match_id=meta["match_id"],
                    period_id=meta["period_id"],
                    ball_owning_slot_block=(
                        z["ball_owning_slot_block"] if "ball_owning_slot_block" in z else None
                    ),
                    phase=z["phase"] if "phase" in z else None,
                )
            )
    return sequences


def cached(
    match_id: str,
    build: Callable[[], list[Sequence]],
    root: Path | str = "data/processed",
    refresh: bool = False,
) -> list[Sequence]:
    """Return cached sequences for a match, building and caching on a miss."""
    if not refresh:
        try:
            return load_sequences(root, match_id)
        except (FileNotFoundError, ValueError):
            pass

    sequences = build()
    save_sequences(sequences, root, match_id)
    return sequences


def cached_metrica(
    match_id: int = 1, root: Path | str = "data/processed", refresh: bool = False
) -> list[Sequence]:
    """Metrica open data, cached."""
    from footrack.sources import load_metrica

    return cached(f"metrica_{match_id}", lambda: load_metrica(match_id), root, refresh)
