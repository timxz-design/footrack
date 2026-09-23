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

## Scripts

| Script | Purpose |
|---|---|
| `verify_setup.py` | Environment check — loads a match, renders one frame |
| `replay.py` | Animated replay with velocity vectors |
| `build_cache.py` | Parse and cache matches, report dataset and baseline stats |
| `inspect_features.py` | Per-feature ranges; flags anything leaking through the mask |
| `train.py` | Stage S1 pretraining |
| `visualize_predictions.py` | Render forecasts and reconstructions — run after any model change |
| `diagnose_modes.py` | Mode usage, diversity and effective count |

## Labelling tool

```powershell
$env:FOOTRACK_LABELER = "your-name"
.venv\Scripts\python.exe -m uvicorn app.server:app --port 8000
```

Then open http://127.0.0.1:8000.

Drag any player to where they should have been. Keys: <kbd>A</kbd> looks right,
<kbd>S</kbd> save corrections, <kbd>D</kbd> skip, <kbd>R</kbd> reset,
<kbd>M</kbd> show the model's guess, arrows to navigate.

Frames come from the validation match, ordered by the dataset's phase weighting so
transitions surface first. Labels append to `data/labels/corrections.jsonl`.

**The model's suggestion is hidden until you press M, deliberately.** Seeing a weak
proposal before forming a view anchors you to it, and anchored labels would bake the
model's current errors into the only ground truth this project will ever have.

Set `FOOTRACK_LABELER` before starting so multi-rater agreement can be measured later.

**Run long scripts unbuffered** (`python -u`) if you want to watch progress — Python
block-buffers stdout when it is not attached to a terminal, so a training run redirected
to a file shows nothing until it finishes.

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

Installed: **2.6.0+cu124**, chosen because CUDA 12.4 wheels still ship Pascal kernels.

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### This machine's GPU

| | |
|---|---|
| Device | GeForce GTX 1050, 4 GB |
| Capability | sm_61 (Pascal) — present in this wheel's arch list |
| Measured | **~0.56 TFLOPS fp32** |
| Training | ~200 s/epoch on 14.4k windows, batch 32, 1.15M params |

**Do not enable mixed precision here.** `torch.cuda.is_bf16_supported()` returns True,
but Pascal has no bf16 hardware — that is emulation, and it runs *slower* than fp32.
Pascal also has no Tensor Cores, so fp16 buys nothing either. Autocast is a real
speedup on Ampere and later; on this card it is a pessimisation.

**What this card can and cannot do.** It is enough to validate the architecture, debug
the training loop, and run small experiments on a couple of matches. It is roughly
50-100x slower than a current training GPU, so the full 84-match pretraining run
belongs on rented hardware or the machine in PLAN.md §5. Budget accordingly rather than
discovering it mid-run.

Checkpoints land in `checkpoints/` (gitignored).

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
