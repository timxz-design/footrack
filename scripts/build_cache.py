"""Parse matches once and cache them, then report dataset statistics.

Run:  .venv\\Scripts\\python.exe scripts\\build_cache.py

The constant-velocity numbers at the end are the bar the world model has to
clear (MODEL.md stage S1). Record them now — they are the only thing that makes
a first training run interpretable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from footrack import (
    Phase,
    WindowDataset,
    cached_metrica,
    constant_velocity_baseline,
    estimate_profiles,
    split_by_match,
)
from footrack.features import POS_SCALE

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed"
MATCHES = (1, 2)
TARGET_RATE = 10.0


def main() -> None:
    sequences = []
    for m in MATCHES:
        t0 = time.time()
        seqs = cached_metrica(m, root=CACHE)
        frames = sum(len(s) for s in seqs)
        print(f"metrica_{m}: {len(seqs)} sequences, {frames} frames, {time.time() - t0:.1f}s")
        sequences.extend(seqs)

    train_seqs, val_seqs = split_by_match(sequences, val_matches=["metrica_2"])
    print(f"\nsplit by match: {len(train_seqs)} train / {len(val_seqs)} val sequences")

    profiles = estimate_profiles(train_seqs)
    print(f"profiles from training matches only: {len(profiles)} players")

    train = WindowDataset(
        train_seqs, history=25, horizon=20, stride=5,
        target_rate=TARGET_RATE, augment=True, profiles=profiles, seed=0,
    )
    val = WindowDataset(
        val_seqs, history=25, horizon=20, stride=25,
        target_rate=TARGET_RATE, augment=False, profiles=profiles, seed=0,
    )
    print(f"windows: {len(train)} train / {len(val)} val")

    counts = np.bincount(train.phase_of_windows(), minlength=len(Phase))
    print("\ntraining windows by phase:")
    for p in Phase:
        print(f"   {p.name:18s} {counts[p]:6d}  {counts[p] / len(train):6.1%}")

    w = train.sampling_weights()
    print(f"\nweighted sampling shifts effective share of TRANSITION windows "
          f"from {counts[Phase.TRANSITION] / len(train):.1%} "
          f"to {w[train.phase_of_windows() == Phase.TRANSITION].sum():.1%}")

    t0 = time.time()
    batch = next(train.batches(batch_size=32))
    print(f"\nfirst batch in {time.time() - t0:.2f}s")
    for k, v in batch.items():
        print(f"   {k:18s} {str(v.shape):22s} {v.dtype}")

    baseline_report(val)


def baseline_report(val: WindowDataset) -> None:
    """Constant-velocity error, broken down.

    Reporting one pooled average is misleading: most entity-frames are players
    walking, whose motion extrapolates almost perfectly, so the pooled number
    flatters the baseline and sets a bar the model would clear while learning
    nothing. The splits below are what actually has to improve.
    """
    steps = (4, 9, 19)  # 0.5s, 1.0s, 2.0s at 10Hz
    buckets: dict[tuple[str, int], list[float]] = {}

    def add(key: str, step: int, values: np.ndarray) -> None:
        if values.size:
            buckets.setdefault((key, step), []).append(float(values.mean()))

    phases = val.phase_of_windows()
    stride = max(1, len(val) // 500)

    for i in range(0, len(val), stride):
        s = val[i]
        pred = constant_velocity_baseline(s, TARGET_RATE)
        err = np.linalg.norm(pred - s.target_positions, axis=-1) * POS_SCALE

        # Only score entities whose velocity was trustworthy: elsewhere the
        # baseline predicts "stay still" for free and looks better than it is.
        vvalid = s.inputs.entity[-1, :, 23].astype(bool)
        valid = s.target_mask & (s.anchor_valid & vvalid)[None, :]

        speed = np.linalg.norm(s.inputs.entity[-1, :, 2:4], axis=-1) * 10.0
        fast = speed > 4.0

        for step in steps:
            v = valid[step]
            add("all entities", step, err[step][v])
            add("ball", step, err[step][:1][v[:1]])
            add("players", step, err[step][1:][v[1:]])
            add("players >4 m/s", step, err[step][1:][(v & fast)[1:]])
            add(Phase(phases[i]).name.lower(), step, err[step][1:][v[1:]])

    keys = ["all entities", "ball", "players", "players >4 m/s"]
    keys += [p.name.lower() for p in Phase if ("players", 4) and (p.name.lower(), 4) in buckets]

    print("\nconstant-velocity baseline — mean displacement error (m)")
    header = "".join(f"{f't+{(s + 1) / TARGET_RATE:.1f}s':>10s}" for s in steps)
    print(f"{'':22s}{header}")
    print("-" * (22 + 10 * len(steps)))
    for key in keys:
        row = "".join(
            f"{np.mean(buckets[(key, s)]):10.2f}" if (key, s) in buckets else f"{'-':>10s}"
            for s in steps
        )
        print(f"{key:22s}{row}")

    print("\nThe pooled figure is flattered by stationary players. 'players >4 m/s'")
    print("and 'transition' are the numbers MODEL.md stage S1 has to beat.")


if __name__ == "__main__":
    main()
