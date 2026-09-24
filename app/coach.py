"""The practice tool — a live shape board.

Move the ball, the shape responds. Switch formation, phase, line height. Pin a
player and the other ten re-solve around them.

Deliberately has **no dependency on data, tracking or the model** (PLAN.md §1.0).
It runs on formation templates and constraints alone, which is the whole point:
in a drill the coach declares the shape, so the template is the instruction
rather than a prediction of it.

Run:  .venv\\Scripts\\python.exe -m uvicorn app.coach:app --port 8100
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from footrack import constraints, pitch
from footrack.formations import FORMATIONS, deform
from footrack.roles import Role
from footrack.sessions import Session, SessionStore, example_session
from footrack.state import Phase

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Footrack practice tool")


class ShapeRequest(BaseModel):
    formation: str = "4-3-3"
    ball_x: float = 0.0  # metres, centre-origin
    ball_y: float = 0.0
    phase: str = "IN_POSSESSION"
    line_height: float = 0.0  # normalised, -0.3 deep .. +0.3 high
    # Slot index -> [x, y] in metres. A pinned player is held exactly there and
    # the rest re-solve around them — the coach's manual override, with no model
    # involved.
    pinned: dict[int, tuple[float, float]] = {}


@app.get("/api/formations")
def list_formations() -> dict:
    return {
        "formations": [
            {
                "name": name,
                "slots": [
                    {"label": s.label, "role": s.role.name} for s in f.slots
                ],
            }
            for name, f in FORMATIONS.items()
        ],
        "phases": [p.name for p in Phase if p is not Phase.UNKNOWN],
        "pitch": {"length": pitch.PITCH_LENGTH, "width": pitch.PITCH_WIDTH},
    }


@app.post("/api/shape")
def shape(req: ShapeRequest) -> dict:
    if req.formation not in FORMATIONS:
        raise HTTPException(404, f"unknown formation {req.formation}")
    try:
        phase = Phase[req.phase]
    except KeyError:
        raise HTTPException(400, f"unknown phase {req.phase}")

    formation = FORMATIONS[req.formation]
    ball = np.array([req.ball_x, req.ball_y], dtype=np.float32)

    targets = deform(formation, ball, phase, req.line_height)

    # Re-solve around any pinned players: hold them exactly, let the constraint
    # layer adjust the rest. The pinned positions are re-applied after, since a
    # constraint must not move what the coach explicitly placed.
    for slot, xy in req.pinned.items():
        if 0 <= slot < len(targets):
            targets[slot] = np.array(xy, dtype=np.float32)

    roles = tuple(s.role for s in formation.slots)
    targets, reports = constraints.apply(targets, roles, ball, phase)

    for slot, xy in req.pinned.items():
        if 0 <= slot < len(targets):
            targets[slot] = np.array(xy, dtype=np.float32)

    outfield = [i for i, r in enumerate(roles) if r is not Role.GK]
    xs = targets[outfield, 0]

    return {
        "formation": req.formation,
        "phase": phase.name,
        "marks": [
            {
                "slot": i,
                "label": formation.slots[i].label,
                "role": formation.slots[i].role.name,
                "x": float(targets[i, 0]),
                "y": float(targets[i, 1]),
                "pinned": i in req.pinned,
            }
            for i in range(len(targets))
        ],
        # Live readouts of the principles a coach actually talks about.
        "shape": {
            "vertical_spread_m": round(float(xs.max() - xs.min()), 1),
            "width_m": round(float(targets[:, 1].max() - targets[:, 1].min()), 1),
            "line_height_m": round(float(xs.min() + pitch.HALF_LENGTH), 1),
            "compact_limit_m": constraints.MAX_VERTICAL_SPREAD,
        },
        "applied": [
            {"name": r.name, "kind": r.kind, "detail": r.detail, "moved_m": round(r.moved_m, 1)}
            for r in reports
        ],
    }


# -- sessions -----------------------------------------------------------------

STORE = SessionStore(Path(__file__).resolve().parents[1] / "data" / "sessions")
if not STORE.list():
    STORE.save(example_session())


@app.get("/api/sessions")
def list_sessions() -> dict:
    return {"sessions": STORE.list()}


@app.get("/api/sessions/{slug}")
def get_session(slug: str) -> dict:
    try:
        return asdict(STORE.load(slug))
    except FileNotFoundError:
        raise HTTPException(404, f"no session {slug}")


class SessionIn(BaseModel):
    name: str
    created: str = ""
    drills: list[dict] = []


@app.put("/api/sessions")
def save_session(body: SessionIn) -> dict:
    try:
        session = Session(**body.model_dump())
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    STORE.save(session)
    return {"ok": True, "slug": session.slug, "sessions": STORE.list()}


@app.delete("/api/sessions/{slug}")
def delete_session(slug: str) -> dict:
    if not STORE.delete(slug):
        raise HTTPException(404, f"no session {slug}")
    return {"ok": True, "sessions": STORE.list()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "coach.html")


@app.get("/run")
def run() -> FileResponse:
    """Pitch-side view: pick a drill, tap to advance, nothing else."""
    return FileResponse(STATIC / "run.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
