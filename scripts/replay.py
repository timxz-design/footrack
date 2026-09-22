"""Render a stretch of a sequence as an animated replay.

Run:  .venv\\Scripts\\python.exe scripts\\replay.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mplsoccer import Pitch

from footrack import BALL, TEAM_A, TEAM_B, load_metrica
from footrack.pitch import to_kloppy

OUT = Path(__file__).resolve().parents[1] / "outputs"

TEAM_A_COLOR = "#4ea3ff"
TEAM_B_COLOR = "#ff5a5a"
GRASS = "#1b3a1b"


def render(seq, start_s: float, duration_s: float, fps: int, path: Path) -> None:
    factor = max(1, int(round(seq.frame_rate / fps)))
    start = int(start_s * seq.frame_rate)
    length = int(duration_s * seq.frame_rate)
    clip = seq.window(start, length).downsample(factor)

    pitch = Pitch(
        pitch_type="metricasports",
        pitch_length=105,
        pitch_width=68,
        pitch_color=GRASS,
        line_color="#c8d6c8",
    )
    fig, ax = pitch.draw(figsize=(13, 8.5))
    fig.set_facecolor(GRASS)

    xy = to_kloppy(clip.positions)

    scat_a = ax.scatter([], [], s=300, c=TEAM_A_COLOR, edgecolors="white", linewidth=1.5, zorder=3)
    scat_b = ax.scatter([], [], s=300, c=TEAM_B_COLOR, edgecolors="white", linewidth=1.5, zorder=3)
    scat_ball = ax.scatter([], [], s=120, c="white", edgecolors="black", linewidth=1.2, zorder=6)
    quiver = ax.quiver(
        np.zeros(22), np.zeros(22), np.zeros(22), np.zeros(22),
        color="white", alpha=0.65, width=0.0022, scale=12, zorder=4,
    )
    clock = ax.text(
        0.5, 1.02, "", transform=ax.transAxes, ha="center",
        color="white", fontsize=11, family="monospace",
    )

    # Velocity arrows are drawn in normalised pitch units, so scale m/s by the
    # same factor the positions use. 1 second of travel at current velocity.
    vel_scale = np.array([1 / 105.0, -1 / 68.0], dtype=np.float32)

    def update(t: int):
        for scat, block in ((scat_a, TEAM_A), (scat_b, TEAM_B)):
            vis = clip.mask[t, block]
            scat.set_offsets(xy[t, block][vis] if vis.any() else np.empty((0, 2)))

        if clip.mask[t, BALL]:
            scat_ball.set_offsets(xy[t, BALL][None, :])
        else:
            scat_ball.set_offsets(np.empty((0, 2)))

        players = np.r_[1:23]
        shown = clip.velocity_valid[t, players]
        vel = clip.velocities[t, players] * vel_scale
        quiver.set_offsets(xy[t, players])
        quiver.set_UVC(np.where(shown, vel[:, 0], 0), np.where(shown, vel[:, 1], 0))

        secs = start_s + t / clip.frame_rate
        clock.set_text(f"period {clip.period_id}   {int(secs // 60):02d}:{secs % 60:05.2f}")
        return scat_a, scat_b, scat_ball, quiver, clock

    anim = FuncAnimation(fig, update, frames=len(clip), interval=1000 / fps, blit=False)
    OUT.mkdir(exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": GRASS})
    plt.close(fig)
    print(f"wrote {path}  ({len(clip)} frames @ {fps}fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=1)
    ap.add_argument("--sequence", type=int, default=0)
    ap.add_argument("--start", type=float, default=60.0, help="seconds into the sequence")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    seqs = load_metrica(args.match)
    seq = seqs[args.sequence]
    print(f"sequence {args.sequence}: period {seq.period_id}, {seq.duration_s / 60:.1f} min")
    render(seq, args.start, args.duration, args.fps, OUT / "replay.gif")


if __name__ == "__main__":
    main()
