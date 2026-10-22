# Footrack

Tactical positioning for football training. A coach plans a session at a desk,
then runs it on the pitch — formation shapes respond to where the ball is, and
eventually get projected onto the grass itself.

## The product

**The practice tool.** Rules-based shape generation, authored off-pitch and
operated with a single tap on it.

```powershell
.venv\Scripts\python.exe -m uvicorn app.coach:app --port 8100 --reload
```

| Page | For |
|---|---|
| `/` | **Planner** — desk. Drag the ball, switch formation and phase, set line height, pin players. |
| `/run` | **Pitch-side** — pick a drill, tap anywhere to advance. Nothing else on screen. |

Authoring and operating are split deliberately. On a pitch a coach has roughly
three seconds of attention before they abandon the device and just shout, so all
the expressiveness lives in the planner and the pitch gets a tap target.

**No dependency on data, tracking, a GPU or any model.**

## How it works

| Layer | Does |
|---|---|
| **L1** `formations.py` | Formation templates, ball-oriented deformation, phase adjustment |
| **L2** `constraints.py` | Coaching principles — compactness, width occupancy, goal-side cover, back-line spacing |
| **L3** `objectives.py` | Objective scoring and local refinement. **Off by default** — see below |
| | `tactics.py` | Runs L1→L2→L3 and attaches a reason to every mark |

Roles, reference positions and objective weights come from
[`ROLES.md`](ROLES.md), confirmed by coaching review.

## Measured things worth knowing

**A ball-conditioned positional template cannot beat ~14m mean deviation.** On a
DFL match: a static team shape gives 22.1m, conditioning on ball position 14.1m,
and a team's own positions 0.8s earlier 1.5m. Football positions vary that much
around any static reference.

That makes the rules layer unsuitable for *analysis* — it catches gross errors,
not fine ones — but it is exactly right for *instruction*, where the coach
declares the shape and the template is the instruction rather than a prediction
of it. Same code, opposite epistemic status. `SHAPE_CALIBRATION` must only be
applied in the predicting direction.

**L3 is off by default** because it makes proposals worse: L1 alone 14.0m, plus
constraints 14.2m, plus objective refinement 15.9m. Its weights were written by
hand with nothing to calibrate against. It turns on once coach corrections exist
to tune them on.

## Shelved: the analysis product

A world model — spatio-temporal encoder with forecast and reconstruction heads —
trained on 29 open matches. It beats constant velocity at 1s and 2s on every
split, and reconstructs a hidden player to 11.8m against an 18.4m trivial
baseline. It is **not** needed by the practice tool and is kept for if the
analysis product is revived.

`model.py` · `dataset.py` · `features.py` · `cache.py` · `sources.py` ·
`events.py` · `labeling.py` · `app/server.py`

See [`MODEL.md`](MODEL.md), which records three distinct mode-collapse failures
and the fact that the last one resolved with data rather than code.

## Docs

| | |
|---|---|
| [PLAN.md](PLAN.md) | Scope, product direction, architecture, roadmap |
| [ROLES.md](ROLES.md) | Role taxonomy, formations, team constraints |
| [MODEL.md](MODEL.md) | The learned layer, and what went wrong in it |
| [DATA.md](DATA.md) | Open tracking sources, licensing, measured coverage |
| [FEATURES.md](FEATURES.md) | Candidate features with scenarios and effort/value |
| [SETUP.md](SETUP.md) | Environment, scripts, hardware notes |

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pytest tests -q
```

210 tests. Python 3.12 — the sports-analytics stack has patchier wheel coverage
on 3.13.
