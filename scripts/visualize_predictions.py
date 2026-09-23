"""Look at what the model actually predicts.

Two panels, two different questions:

**Forecast** — where does the model think everyone goes over the next 2s? The K
modes are drawn with opacity proportional to their probability, so you can see
whether the model is confident or hedging.

**Reconstruction** — hide some players entirely and ask the model to place them
from their team-mates alone. This is the mechanism behind manual override:
pinning a dragged player and re-solving the rest is the same head run the other
way round. It is also the closest thing to a positional opinion the model
currently has, since on elite training data "where would this player be" is a
usable proxy for "where should they be".

Run:  .venv\\Scripts\\python.exe scripts\\visualize_predictions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mplsoccer import Pitch

from footrack import WindowDataset, cached_metrica, estimate_profiles, split_by_match
from footrack.features import POS_SCALE
from footrack.model import Footrack, ModelConfig, apply_agent_masking
from footrack.pitch import to_kloppy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
GRASS = "#1b3a1b"
TEAM_A_COLOR = "#4ea3ff"
TEAM_B_COLOR = "#ff5a5a"
MODE_COLORS = ["#ffd166", "#06d6a0", "#ef476f", "#c77dff", "#f78c6b", "#4cc9f0"]


def new_pitch(ax_title: str):
    pitch = Pitch(pitch_type="metricasports", pitch_length=105, pitch_width=68,
                  pitch_color=GRASS, line_color="#c8d6c8")
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.set_facecolor(GRASS)
    ax.set_title(ax_title, color="white", fontsize=12, pad=12)
    return pitch, fig, ax


def draw_players(pitch, ax, xy, mask, alpha=1.0, size=240):
    for lo, hi, colour in ((1, 12, TEAM_A_COLOR), (12, 23, TEAM_B_COLOR)):
        sel = mask[lo:hi]
        if sel.any():
            pitch.scatter(xy[lo:hi][sel, 0], xy[lo:hi][sel, 1], s=size, color=colour,
                          edgecolors="white", linewidth=1.4, alpha=alpha, zorder=4, ax=ax)
    if mask[0]:
        pitch.scatter(xy[0, 0], xy[0, 1], s=110, color="white", edgecolors="black",
                      linewidth=1.2, alpha=alpha, zorder=6, ax=ax)


def plot_forecast(model, sample, device, path: Path) -> None:
    with torch.no_grad():
        batch = {k: torch.from_numpy(v)[None].to(device) for k, v in (
            ("entity", sample.inputs.entity), ("context", sample.inputs.context),
            ("role_ids", sample.inputs.role_ids), ("player_ids", sample.inputs.player_ids))}
        out = model(batch["entity"], batch["context"], batch["role_ids"], batch["player_ids"])
        probs = torch.softmax(out["mode_logits"][0], dim=0).cpu().numpy()
        deltas = out["deltas"][0].cpu().numpy()  # (K, F, N, 2)

    anchor = sample.anchor
    present = sample.anchor_valid

    pitch, fig, ax = new_pitch("forecast — white = what happened, colours = predicted modes")
    draw_players(pitch, ax, to_kloppy(anchor * POS_SCALE), present)

    truth = to_kloppy(sample.target_positions * POS_SCALE)
    for n in range(23):
        if not present[n]:
            continue
        ax.plot(truth[:, n, 0], truth[:, n, 1], color="white", lw=2.0, alpha=0.9, zorder=5)

    order = np.argsort(-probs)
    for rank, k in enumerate(order):
        pred = to_kloppy((anchor[None] + deltas[k]) * POS_SCALE)
        colour = MODE_COLORS[k % len(MODE_COLORS)]
        alpha = float(np.clip(probs[k], 0.08, 1.0))
        for n in range(23):
            if not present[n]:
                continue
            ax.plot(pred[:, n, 0], pred[:, n, 1], color=colour, lw=1.6,
                    alpha=alpha, zorder=3)
        ax.plot([], [], color=colour, lw=2.5, alpha=max(alpha, 0.5),
                label=f"mode {k}  p={probs[k]:.2f}")

    leg = ax.legend(loc="upper left", framealpha=0.25, facecolor="black", fontsize=8)
    for text in leg.get_texts():
        text.set_color("white")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=GRASS)
    plt.close(fig)
    print(f"wrote {path}")


def plot_reconstruction(model, sample, device, path: Path, seed: int = 3) -> dict:
    entity = torch.from_numpy(sample.inputs.entity)[None].to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    masked, hidden = apply_agent_masking(entity, p=0.25, generator=gen)

    with torch.no_grad():
        out = model(masked,
                    torch.from_numpy(sample.inputs.context)[None].to(device),
                    torch.from_numpy(sample.inputs.role_ids)[None].to(device),
                    torch.from_numpy(sample.inputs.player_ids)[None].to(device))
        recon = out["recon"][0, -1].cpu().numpy()  # (N, 2) final frame

    hid = hidden[0].cpu().numpy()
    truth = sample.inputs.entity[-1, :, 0:2]
    present = sample.anchor_valid

    pitch, fig, ax = new_pitch(
        "reconstruction — hidden players: solid = actual, hollow = model's guess")
    draw_players(pitch, ax, to_kloppy(truth * POS_SCALE), present & ~hid, alpha=0.35)
    draw_players(pitch, ax, to_kloppy(truth * POS_SCALE), present & hid, alpha=1.0)

    guess_xy = to_kloppy(recon * POS_SCALE)
    true_xy = to_kloppy(truth * POS_SCALE)
    errors = []
    for n in np.flatnonzero(hid & present):
        pitch.scatter(guess_xy[n, 0], guess_xy[n, 1], s=240, facecolors="none",
                      edgecolors="#ffd166", linewidth=2.4, zorder=7, ax=ax)
        ax.plot([true_xy[n, 0], guess_xy[n, 0]], [true_xy[n, 1], guess_xy[n, 1]],
                color="#ffd166", lw=1.6, ls="--", zorder=6)
        d = float(np.linalg.norm(recon[n] - truth[n]) * POS_SCALE)
        errors.append(d)
        ax.annotate(f"{d:.1f}m", xy=((true_xy[n, 0] + guess_xy[n, 0]) / 2,
                                     (true_xy[n, 1] + guess_xy[n, 1]) / 2),
                    color="#ffd166", fontsize=8, ha="center", zorder=8)

    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=GRASS)
    plt.close(fig)
    print(f"wrote {path}")
    return {"n_hidden": len(errors), "mean_error_m": float(np.mean(errors)) if errors else None}


@torch.no_grad()
def reconstruction_benchmark(model, dataset, device, batch_size=32, limit=40) -> dict:
    """Mean placement error in metres when a player is hidden entirely."""
    gen = torch.Generator(device=device).manual_seed(0)
    errs, centroid_errs = [], []
    for i, batch in enumerate(dataset.batches(batch_size, shuffle=False, drop_last=False)):
        if i >= limit:
            break
        entity = torch.from_numpy(batch["entity"]).to(device)
        masked, hidden = apply_agent_masking(entity, p=0.15, generator=gen)
        out = model(masked,
                    torch.from_numpy(batch["context"]).to(device),
                    torch.from_numpy(batch["role_ids"]).to(device),
                    torch.from_numpy(batch["player_ids"]).to(device))
        truth = entity[..., 0:2]
        err = torch.linalg.vector_norm(out["recon"] - truth, dim=-1) * POS_SCALE
        sel = hidden.unsqueeze(1) & (entity[..., 22] > 0.5)
        if sel.any():
            errs.append(err[sel].mean().item())
            # Baseline: guess the centroid of the visible players.
            vis = (entity[..., 22] > 0.5).unsqueeze(-1).float()
            centroid = (truth * vis).sum(2, keepdim=True) / vis.sum(2, keepdim=True).clamp(min=1)
            cerr = torch.linalg.vector_norm(centroid - truth, dim=-1) * POS_SCALE
            centroid_errs.append(cerr[sel].mean().item())
    return {"model_m": float(np.mean(errs)), "centroid_m": float(np.mean(centroid_errs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "s1_latest.pt"))
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    model = Footrack(ModelConfig(**ck["cfg"])).to(args.device).eval()
    model.load_state_dict(ck["state"])

    seqs = []
    for i in (1, 2):
        seqs.extend(cached_metrica(i, root=ROOT / "data" / "processed"))
    train_seqs, val_seqs = split_by_match(seqs, val_matches=["metrica_2"])
    val = WindowDataset(val_seqs, history=25, horizon=20, stride=25, target_rate=10.0,
                        augment=False, profiles=estimate_profiles(train_seqs), seed=0)

    OUT.mkdir(exist_ok=True)
    sample = val[args.window]
    plot_forecast(model, sample, args.device, OUT / "predict_forecast.png")
    info = plot_reconstruction(model, sample, args.device, OUT / "predict_reconstruction.png")
    print(f"\nthis frame: {info['n_hidden']} hidden, mean error {info['mean_error_m']:.1f} m")

    bench = reconstruction_benchmark(model, val, args.device)
    print(f"\nreconstruction over validation set:")
    print(f"   model            {bench['model_m']:5.1f} m")
    print(f"   centroid guess   {bench['centroid_m']:5.1f} m   (the trivial baseline)")


if __name__ == "__main__":
    main()
