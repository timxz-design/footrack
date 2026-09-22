# Footrack — Environment

## Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Or skip activation entirely and call the interpreter directly:
`.venv\Scripts\python.exe script.py`

## Verify

```powershell
.venv\Scripts\python.exe scripts\verify_setup.py
```

Loads Metrica match 1 through kloppy and writes `outputs/verify_frame.png`.

## What is installed

Python **3.12.1** (not 3.13 — the sports-analytics stack has patchier wheel coverage there).

| Package | Version | Purpose |
|---|---|---|
| kloppy | 3.19.0 | Unified tracking/event loading across 9 providers |
| mplsoccer | 1.8.1 | Pitch rendering |
| statsbombpy | 1.22.0 | StatsBomb open data |
| pandas | 3.0.6 | |
| numpy | 2.5.3 | |
| scipy | 1.18.1 | |
| matplotlib | 3.11.2 | |
| jupyterlab | 4.6.4 | |

Full lock in `requirements.txt`. Rebuild with:
`.venv\Scripts\python.exe -m pip install -r requirements.txt`

## Deliberately NOT installed: socceraction

`socceraction` 1.5.3 pins `numpy<2` and `pandas 2.x`, which forces the whole
environment back to a 2023-era stack and breaks matplotlib's contourpy dependency.

It is only needed for VAEP, which belongs to the value head in Phase 3 — months away.
When that time comes, either:

- install it in a **separate venv** used only for computing VAEP labels offline, or
- implement xT directly. xT is a grid plus a Markov chain, roughly 50 lines, and
  implementing it yourself is arguably better for understanding the value signal that
  the whole policy layer depends on.

Do not install it into this environment without re-verifying the numpy/pandas pins.

## PyTorch

Not installed yet — not needed until Phase 3a (~week 14). When you get there, install
the CUDA build matching your GPU from pytorch.org rather than the default CPU wheel.

## Coordinate systems

kloppy's default coordinate system is **normalized [0, 1] on both axes**.

**Canonical internal representation is metres, centre-origin** — see
`src/footrack/pitch.py`. Not the per-axis `[-1, +1]` that ROLES.md §1 describes:
normalizing each axis independently destroys the aspect ratio, because one x-unit would
be 52.5m and one y-unit 34m. Every distance, speed and pitch-control value would be
anisotropic. Metres keeps distances real and velocities in m/s, directly comparable to
the max-speed figures the capability model reads from GPS vests.

Two normalized views exist for the cases where normalization is actually correct:

| Function | Purpose |
|---|---|
| `to_model_scale` | Both axes / HALF_LENGTH. **Isotropic** — this is what the model consumes |
| `to_axis_norm` | Per-axis [-1, 1]. Channel and third lookups only — never distances |

Conversion happens once at ingest in `sources.py`, alongside attack-direction
normalization (kloppy's `STATIC_HOME_AWAY` orientation), not scattered through the
codebase.

## Layout

```
footrack/
  .venv/                virtual environment
  data/raw/             downloaded datasets (gitignored)
  data/processed/       preprocessed binary format (gitignored)
  notebooks/            exploration
  scripts/              runnable entry points
  src/footrack/         the package
  outputs/              generated images, renders (gitignored)
```
