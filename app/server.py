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

from footrack.dataset import WindowDataset, standard_datasets
from footrack.features import POS_SCALE
from footrack.labeling import Correction, Label, LabelStore
from footrack.model import OBSERVATION_SLICE, PRESENT_COL, Footrack, ModelConfig
from footrack.state import BALL, Phase

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
HISTORY, HORIZON = 25, 20

app = FastAPI(title="Footrack labelling")
_state: dict = {}


def _load() -> None:
    _, dataset, _ = standard_datasets(
        ROOT / "data" / "processed", history=HISTORY, horizon=HORIZON, val_stride=10
    )

    # Only serve frames where the ball is tracked. It is absent in ~40% of
    # Metrica frames, and a position cannot be judged without knowing where the
    # ball is — the whole shape is organised around it.
    weights = dataset.sampling_weights()
    has_ball = np.array([
        bool(dataset.sequences[si].mask[start + HISTORY - 1, BALL])
        for si, start in dataset.index
    ])
    weights = np.where(has_ball, weights, 0.0)
    if weights.sum() == 0:
        raise RuntimeError("no frames with a tracked ball")
    weights = weights / weights.sum()

    # Order by the phase weighting so transitions surface before dead ball.
    rng = np.random.default_rng(0)
    candidates = np.flatnonzero(weights > 0)
    order = rng.choice(candidates, size=len(candidates), replace=False,
                       p=weights[candidates] / weights[candidates].sum())

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
    """Where the model would place each player, given every other player.

    One player is hidden at a time, batched into a single forward pass. Hiding
    everyone at once — as an earlier version did — leaves the model nothing to
    condition on, so it returns the mean position of everything and the overlay
    collapses into a meaningless cluster at the centre circle.
    """
    model = _state["model"]
    if model is None:
        raise HTTPException(404, "no checkpoint available")

    dataset: WindowDataset = _state["dataset"]
    index = int(_state["order"][position])
    sample = dataset[index]

    slots = [s for s in range(1, 23) if bool(sample.anchor_valid[s])]
    if not slots:
        return {"position": position, "players": [], "caveat": "no players present"}

    base = torch.from_numpy(sample.inputs.entity)
    batch = base.unsqueeze(0).repeat(len(slots), 1, 1, 1)
    for i, slot in enumerate(slots):
        batch[i, :, slot, OBSERVATION_SLICE] = 0.0
        batch[i, :, slot, PRESENT_COL] = 0.0

    def stack(arr):
        return torch.from_numpy(arr).unsqueeze(0).repeat(
            len(slots), *([1] * arr.ndim)
        )

    with torch.no_grad():
        out = model(batch, stack(sample.inputs.context),
                    stack(sample.inputs.role_ids), stack(sample.inputs.player_ids))
        recon = out["recon"][:, -1].numpy() * POS_SCALE

    truth = sample.inputs.entity[-1, :, 0:2] * POS_SCALE
    players, errors = [], []
    for i, slot in enumerate(slots):
        err = float(np.linalg.norm(recon[i, slot] - truth[slot]))
        errors.append(err)
        players.append({
            "slot": slot,
            "x": float(recon[i, slot, 0]),
            "y": float(recon[i, slot, 1]),
            "error_m": round(err, 1),
        })

    return {
        "position": position,
        "players": players,
        "caveat": f"each player hidden in turn — mean error {np.mean(errors):.1f}m here",
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
