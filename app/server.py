"""Local labelling tool — browse frames, drag players, record corrections.

Run:  .venv\\Scripts\\python.exe -m uvicorn app.server:app --port 8000
Then open http://127.0.0.1:8000

Frames are drawn from the validation matches and sampled with the phase
weighting from the dataset layer, so transitions — where positioning decisions
are densest — come up more often than dead ball.

**The model's own suggestion is hidden by default, deliberately.** Showing a
weak proposal before you judge anchors you to it, and anchored labels would
bake the model's current errors into the only ground truth the project will
ever have. Toggle it on to compare *after* forming a view.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from footrack import WindowDataset, cached_metrica, estimate_profiles, split_by_match
from footrack.features import POS_SCALE
from footrack.labeling import Correction, Label, LabelStore
from footrack.model import Footrack, ModelConfig, apply_agent_masking
from footrack.state import Phase

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
HISTORY, HORIZON = 25, 20

app = FastAPI(title="Footrack labelling")
_state: dict = {}


def _load() -> None:
    sequences = []
    for m in (1, 2):
        sequences.extend(cached_metrica(m, root=ROOT / "data" / "processed"))
    train_seqs, val_seqs = split_by_match(sequences, val_matches=["metrica_2"])

    dataset = WindowDataset(
        val_seqs, history=HISTORY, horizon=HORIZON, stride=10, target_rate=10.0,
        augment=False, profiles=estimate_profiles(train_seqs), seed=0,
    )

    # Order frames by the phase weighting so transitions surface early.
    weights = dataset.sampling_weights()
    rng = np.random.default_rng(0)
    order = rng.choice(len(dataset), size=len(dataset), replace=False,
                       p=weights / weights.sum())

    _state["dataset"] = dataset
    _state["order"] = order
    _state["store"] = LabelStore(ROOT / "data" / "labels" / "corrections.jsonl")
    _state["labeler"] = os.environ.get("FOOTRACK_LABELER", "unknown")

    ckpt = ROOT / "checkpoints" / "s1_latest.pt"
    if ckpt.exists():
        blob = torch.load(ckpt, weights_only=False)
        model = Footrack(ModelConfig(**blob["cfg"])).eval()
        model.load_state_dict(blob["state"])
        _state["model"] = model
    else:
        _state["model"] = None


@app.on_event("startup")
def startup() -> None:
    _load()


def _frame_key(dataset: WindowDataset, index: int) -> tuple[str, str, int, int]:
    si, start = dataset.index[index]
    seq = dataset.sequences[si]
    frame_id = int(seq.frame_ids[start + HISTORY - 1])
    return f"{seq.match_id}/{seq.period_id}/{frame_id}", seq.match_id, int(si), frame_id


@app.get("/api/frame/{position}")
def get_frame(position: int) -> dict:
    dataset: WindowDataset = _state["dataset"]
    order = _state["order"]
    if not 0 <= position < len(order):
        raise HTTPException(404, "position out of range")

    index = int(order[position])
    sample = dataset[index]
    key, match_id, si, frame_id = _frame_key(dataset, index)

    xy = sample.anchor * POS_SCALE  # metres, centre-origin
    vel = sample.inputs.entity[-1, :, 2:4] * 10.0  # m/s
    slots = dataset.sequences[dataset.index[index][0]].slots
    phase_col = sample.inputs.context[-1, : len(Phase)]
    phase = Phase(int(np.argmax(phase_col))).name

    players = []
    for slot in range(23):
        players.append({
            "slot": slot,
            "present": bool(sample.anchor_valid[slot]),
            "team": "ball" if slot == 0 else ("a" if slot < 12 else "b"),
            "jersey": slots.jersey_numbers[slot],
            "x": float(xy[slot, 0]),
            "y": float(xy[slot, 1]),
            "vx": float(vel[slot, 0]),
            "vy": float(vel[slot, 1]),
        })

    existing = _state["store"].latest_by_frame().get(key)
    return {
        "position": position,
        "total": len(order),
        "frame_key": key,
        "match_id": match_id,
        "sequence_index": si,
        "frame_index": frame_id,
        "phase": phase,
        "players": players,
        "already_labelled": existing.verdict if existing else None,
    }


@app.get("/api/suggestion/{position}")
def get_suggestion(position: int) -> dict:
    """The model's own placement for each player, for comparison after judging."""
    model = _state["model"]
    if model is None:
        raise HTTPException(404, "no checkpoint available")

    dataset: WindowDataset = _state["dataset"]
    index = int(_state["order"][position])
    sample = dataset[index]

    entity = torch.from_numpy(sample.inputs.entity)[None]
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # Hide everyone in turn is too slow; hide a quarter and report those.
        masked, hidden = apply_agent_masking(entity, p=1.0, generator=gen)
        out = model(masked,
                    torch.from_numpy(sample.inputs.context)[None],
                    torch.from_numpy(sample.inputs.role_ids)[None],
                    torch.from_numpy(sample.inputs.player_ids)[None])
        recon = out["recon"][0, -1].numpy() * POS_SCALE

    return {
        "position": position,
        "players": [
            {"slot": s, "x": float(recon[s, 0]), "y": float(recon[s, 1])}
            for s in range(23)
            if bool(sample.anchor_valid[s])
        ],
        "caveat": "reconstruction head, ~15m mean error — weaker than it looks",
    }


class CorrectionIn(BaseModel):
    slot: int
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]


class LabelIn(BaseModel):
    position: int
    verdict: str
    corrections: list[CorrectionIn] = []
    note: str = ""


@app.post("/api/label")
def post_label(body: LabelIn) -> dict:
    dataset: WindowDataset = _state["dataset"]
    index = int(_state["order"][body.position])
    key, match_id, si, frame_id = _frame_key(dataset, index)

    label = Label(
        frame_key=key,
        match_id=match_id,
        sequence_index=si,
        frame_index=frame_id,
        verdict=body.verdict,
        corrections=[Correction(c.slot, tuple(c.from_xy), tuple(c.to_xy))
                     for c in body.corrections],
        note=body.note,
        labeler=_state["labeler"],
    )
    _state["store"].append(label)
    return {"ok": True, "split": label.split, "stats": _state["store"].stats()}


@app.get("/api/stats")
def get_stats() -> dict:
    return _state["store"].stats()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
