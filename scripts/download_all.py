"""Fetch and cache every open match: Metrica, DFL/IDSSE and SkillCorner.

Run:  .venv\\Scripts\\python.exe -u scripts\\download_all.py

Parsing is slow — roughly 5 minutes per Sportec match — but it happens once.
Cached loads are ~0.4s thereafter. Safe to re-run: matches already cached are
skipped.
"""

from __future__ import annotations

import time
from pathlib import Path

from footrack.cache import cached_metrica, cached_skillcorner, cached_sportec
from footrack.sources import SKILLCORNER_MATCHES, SPORTEC_MATCHES

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed"


def main() -> None:
    jobs: list[tuple[str, str, callable]] = []
    # Metrica game 3 is skipped: it ships in a different format from games 1
    # and 2, has no event feed at the URL they use, and parses to zero usable
    # sequences after nine minutes of work.
    for m in (1, 2):
        jobs.append((f"metrica_{m}", "metrica", lambda m=m: cached_metrica(m, root=CACHE)))
    for m in SPORTEC_MATCHES:
        jobs.append((f"sportec_{m}", "sportec", lambda m=m: cached_sportec(m, root=CACHE)))
    for m in SKILLCORNER_MATCHES:
        jobs.append((f"skillcorner_{m}", "skillcorner", lambda m=m: cached_skillcorner(m, root=CACHE)))

    print(f"{len(jobs)} matches to ensure cached\n")
    total_frames = 0
    ok, failed = 0, []

    for i, (name, kind, fn) in enumerate(jobs, 1):
        # Check for actual sequence files, not just the directory: a match that
        # parses to nothing still leaves an empty folder behind, which would
        # otherwise read as "cached" forever while every load of it raises.
        if any((CACHE / name).glob("seq*.npz")):
            print(f"[{i:2d}/{len(jobs)}] {name:22s} cached")
            ok += 1
            continue
        t0 = time.time()
        try:
            seqs = fn()
        except Exception as exc:
            print(f"[{i:2d}/{len(jobs)}] {name:22s} FAILED {type(exc).__name__}: {exc}")
            failed.append(name)
            continue
        frames = sum(len(s) for s in seqs)
        total_frames += frames
        ok += 1
        print(f"[{i:2d}/{len(jobs)}] {name:22s} {len(seqs):2d} seq  {frames:7d} frames  "
              f"{time.time() - t0:5.0f}s")

    print(f"\ncached {ok}/{len(jobs)} matches, {total_frames} new frames")
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()
