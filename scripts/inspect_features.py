"""Print feature ranges and invariants for a built ModelInput.

A feature silently out of range, or leaking through a mask, is the kind of bug
that degrades training without ever raising. Run this after changing features.py.

Run:  .venv\\Scripts\\python.exe scripts\\inspect_features.py
"""

from __future__ import annotations

import numpy as np

from footrack import (
    CONTEXT_FEATURES,
    ENTITY_FEATURES,
    N_PHASES,
    Phase,
    build,
    estimate_profiles,
    load_metrica,
)


def main() -> None:
    seqs = load_metrica(1)
    profiles = estimate_profiles(seqs)
    print(f"profiles estimated for {len(profiles)} players")
    for pid, p in sorted(profiles.items(), key=lambda kv: -kv[1].max_speed)[:3]:
        print(f"   {pid:10s} top {p.max_speed:5.2f} m/s   accel {p.max_accel:5.2f}   stamina {p.stamina:.3f}")

    mi = build(seqs[0].window(1000, 3000).downsample(5), profiles=profiles)
    print(f"\nentity  {mi.entity.shape}  context {mi.context.shape}")
    print(f"finite: entity={np.isfinite(mi.entity).all()}  context={np.isfinite(mi.context).all()}")

    absent = ~mi.mask
    structural = {"is_ball", "is_own_team", "is_opponent", "present", "velocity_valid",
                  "prof_max_speed", "prof_max_accel", "prof_stamina"}

    print(f"\n{'feature':22s} {'min':>8s} {'max':>8s} {'mean':>8s}   {'when absent':>12s}")
    print("-" * 66)
    for i, name in enumerate(ENTITY_FEATURES):
        col = mi.entity[..., i]
        if absent.any():
            zeroed = bool((col[absent] == 0).all())
            note = "structural" if name in structural else ("zeroed" if zeroed else "*** LEAK ***")
        else:
            note = "n/a"
        print(f"{name:22s} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f}   {note:>12s}")

    print(f"\n{'context':22s} {'min':>8s} {'max':>8s} {'mean':>8s}")
    print("-" * 52)
    for i, name in enumerate(CONTEXT_FEATURES):
        col = mi.context[..., i]
        print(f"{name:22s} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f}")

    onehot = mi.context[:, 0:N_PHASES].sum(axis=1)
    print(f"\nphase one-hot sums to 1 on every frame: {np.allclose(onehot, 1.0)}")
    types = mi.entity[..., 19:22].sum(axis=-1)
    print(f"entity type one-hot sums to 1 everywhere: {np.allclose(types, 1.0)}")

    # Phase balance across the whole match, not just the inspected window.
    print("\nphase distribution over all sequences:")
    counts = np.zeros(N_PHASES, dtype=np.int64)
    total = 0
    for s in seqs:
        if s.phase is None:
            continue
        counts += np.bincount(s.phase, minlength=N_PHASES)
        total += len(s)
    for p in Phase:
        print(f"   {p.name:18s} {counts[p] / total:6.1%}")


if __name__ == "__main__":
    main()
