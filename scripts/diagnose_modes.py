"""Diagnose mode usage and diversity in a trained forecast head.

Run:  .venv\\Scripts\\python.exe scripts\\diagnose_modes.py

**Read the winner histogram and the spread together.** A concentrated winner
distribution does not prove the modes collapsed — it may mean the selection
metric cannot see the diversity that exists. That distinction cost a debugging
cycle: after the relaxed-WTA fix, selecting over all 23 entities used 1 of 6
modes while selecting on the ball alone used all 6. The modes were fine; the
metric was blind (MODEL.md §3.2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from footrack.dataset import standard_datasets
from footrack.features import POS_SCALE
from footrack.model import Footrack, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def histogram(name: str, wins: np.ndarray, k: int) -> None:
    counts = np.bincount(wins, minlength=k)
    used = int((counts > 0).sum())
    print(f"\n{name}  —  {used}/{k} modes used")
    top = max(counts.max(), 1)
    for i, c in enumerate(counts):
        bar = "#" * int(48 * c / top)
        print(f"   mode {i}: {c:6d} ({c / len(wins):5.1%}) {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "s1_latest.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    model = Footrack(cfg).to(args.device).eval()
    model.load_state_dict(ck["state"])
    print(f"checkpoint from epoch {ck.get('epoch', '?')}, K={cfg.n_modes}")

    _, val, _ = standard_datasets(ROOT / "data" / "processed")

    wins = {"all entities": [], "ball only": [], "players only": []}
    err_ball, err_players, spreads = [], [], []

    with torch.no_grad():
        for b in val.batches(32, shuffle=False, drop_last=False):
            g = {k: torch.from_numpy(v).to(args.device) for k, v in b.items()}
            out = model(g["entity"], g["context"], g["role_ids"], g["player_ids"])
            err = torch.linalg.vector_norm(
                out["deltas"] - g["target_deltas"].unsqueeze(1), dim=-1
            )
            w = g["target_mask"].unsqueeze(1).float()

            def pooled(sel: slice) -> torch.Tensor:
                e, ww = err[..., sel], w[..., sel]
                return (e * ww).sum((2, 3)) / ww.sum((2, 3)).clamp(min=1)

            pm_all, pm_ball, pm_play = pooled(slice(None)), pooled(slice(0, 1)), pooled(slice(1, None))
            wins["all entities"].append(pm_all.argmin(1).cpu().numpy())
            wins["ball only"].append(pm_ball.argmin(1).cpu().numpy())
            wins["players only"].append(pm_play.argmin(1).cpu().numpy())
            err_ball.append((pm_ball * POS_SCALE).cpu().numpy())
            err_players.append((pm_play * POS_SCALE).cpu().numpy())
            final = out["deltas"][:, :, -1]  # (B, K, N, 2)
            spreads.append((final.std(dim=1).mean((1, 2)) * POS_SCALE).cpu().numpy())

    for name, w in wins.items():
        histogram(name, np.concatenate(w), cfg.n_modes)

    eb, ep = np.concatenate(err_ball), np.concatenate(err_players)
    print(f"\nmean error per mode (m)")
    print(f"   ball    : {np.round(eb.mean(0), 2)}")
    print(f"   players : {np.round(ep.mean(0), 2)}")
    print(f"\ndiscriminative spread — std across modes of per-mode mean error")
    print(f"   ball    : {eb.mean(0).std():.3f} m")
    print(f"   players : {ep.mean(0).std():.3f} m")
    print(f"   (whichever is larger dominates a pooled selection metric)")

    share = 22 * ep.mean() / (22 * ep.mean() + eb.mean())
    print(f"\nplayers carry {share:.1%} of an unweighted selection metric")
    print(f"mean inter-mode spread at final step: {np.concatenate(spreads).mean():.2f} m")

    # Effective mode count: how many modes a balanced usage would amount to.
    counts = np.bincount(np.concatenate(wins["ball only"]), minlength=cfg.n_modes)
    p = counts / counts.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    print(f"effective modes (perplexity of ball-only winners): {eff:.2f} of {cfg.n_modes}")
    print("  low values were once read as 'the horizon is too short for this many modes'.")
    print("  That was wrong: on 2 matches this sat at 2.4, and on 29 it reached 4.2 with")
    print("  no code change. Suspect too little data before suspecting K or the horizon.")

    # Per-entity spread understates ball divergence, because 22 of 23 entities
    # are players the modes agree about. Report the ball separately.
    print(f"\nper-mode ball error spans {eb.mean(0).min():.2f}-{eb.mean(0).max():.2f} m, "
          f"players only {ep.mean(0).min():.2f}-{ep.mean(0).max():.2f} m")
    print("  a wide ball range with a narrow player range is the intended behaviour:")
    print("  modes should encode different ball futures and agree about the players")


if __name__ == "__main__":
    main()
