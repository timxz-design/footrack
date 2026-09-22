"""Verify the environment end to end: load Metrica open data, render a frame.

Run:  .venv\\Scripts\\python.exe scripts\\verify_setup.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kloppy import metrica
from mplsoccer import Pitch

OUT = Path(__file__).resolve().parents[1] / "outputs"
FRAME_INDEX = 400


def main() -> None:
    dataset = metrica.load_open_data(match_id=1, limit=1000)
    meta = dataset.metadata
    print(f"loaded {len(dataset.records)} frames @ {meta.frame_rate}Hz from {meta.provider}")

    frame = dataset.records[FRAME_INDEX]

    # kloppy's default coordinate system is normalized [0, 1] on both axes.
    # mplsoccer's 'metricasports' pitch type expects exactly that.
    pitch = Pitch(
        pitch_type="metricasports",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#1b3a1b",
        line_color="#c8d6c8",
    )
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor("#1b3a1b")

    teams = {t.team_id: t for t in meta.teams}
    colors = {list(teams)[0]: "#4ea3ff", list(teams)[1]: "#ff5a5a"}

    for player, data in frame.players_data.items():
        if data.coordinates is None:
            continue
        pitch.scatter(
            data.coordinates.x,
            data.coordinates.y,
            s=260,
            color=colors[player.team.team_id],
            edgecolors="white",
            linewidth=1.5,
            zorder=3,
            ax=ax,
        )
        ax.text(
            data.coordinates.x,
            data.coordinates.y,
            str(player.jersey_no),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            zorder=4,
        )

    if frame.ball_coordinates is not None:
        pitch.scatter(
            frame.ball_coordinates.x,
            frame.ball_coordinates.y,
            s=110,
            color="white",
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
            ax=ax,
        )

    OUT.mkdir(exist_ok=True)
    path = OUT / "verify_frame.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
