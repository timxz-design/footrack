"""Stage S1 pretraining: trajectory forecasting + masked agent modelling.

Run:  .venv\\Scripts\\python.exe scripts\\train.py --epochs 10

The gate for this stage is beating the constant-velocity baseline (MODEL.md §9),
which is reported alongside the model every evaluation. Watch the ball and the
fast-player rows: the pooled figure is flattered by stationary players and can
improve while the model learns nothing useful.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from footrack import (
    WindowDataset,
    build_player_index,
    cached_metrica,
    estimate_profiles,
    infer_roles,
    split_by_match,
    summarise,
)
from footrack.dataset import constant_velocity_batch
from footrack.features import POS_SCALE
from footrack.model import (
    Footrack,
    ModelConfig,
    apply_agent_masking,
    reconstruction_loss,
    variety_loss,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
TARGET_RATE = 10.0
EVAL_STEPS = (4, 9, 19)  # 0.5s, 1.0s, 2.0s at 10Hz


def to_device(batch: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v)
        if t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    return out


@torch.no_grad()
def evaluate(model: Footrack, dataset: WindowDataset, device: str, batch_size: int) -> dict:
    """Displacement error in metres, model versus constant velocity."""
    model.eval()
    sums: dict[tuple[str, str, int], float] = {}
    counts: dict[tuple[str, str, int], float] = {}

    def add(who: str, key: str, step: int, err: np.ndarray, sel: np.ndarray) -> None:
        if sel.any():
            sums[(who, key, step)] = sums.get((who, key, step), 0.0) + float(err[sel].sum())
            counts[(who, key, step)] = counts.get((who, key, step), 0.0) + float(sel.sum())

    for batch in dataset.batches(batch_size, shuffle=False, drop_last=False):
        gpu = to_device(batch, device)
        out = model(gpu["entity"], gpu["context"], gpu["role_ids"], gpu["player_ids"])

        # Best-of-K, the standard way multi-modal forecasters are scored: the
        # model is asked to cover the space of futures, not to commit to one.
        err_k = torch.linalg.vector_norm(
            out["deltas"] - gpu["target_deltas"].unsqueeze(1), dim=-1
        )  # (B,K,F,N)
        m = gpu["target_mask"].unsqueeze(1).float()
        per_mode = (err_k * m).sum(dim=(2, 3)) / m.sum(dim=(2, 3)).clamp(min=1.0)
        pick = per_mode.argmin(1)
        chosen = err_k[torch.arange(len(pick), device=device), pick]  # (B,F,N)
        model_err = (chosen * POS_SCALE).cpu().numpy()

        vmask = batch["target_mask"] & batch["anchor_valid"][:, None, :]
        vvalid = batch["entity"][:, -1, :, 23].astype(bool)
        speed = np.linalg.norm(batch["entity"][:, -1, :, 2:4], axis=-1) * 10.0
        valid = vmask & vvalid[:, None, :]

        cv = constant_velocity_batch(
            batch["entity"], batch["anchor"], batch["target_positions"].shape[1], TARGET_RATE
        )
        cv_err = np.linalg.norm(cv - batch["target_positions"], axis=-1) * POS_SCALE

        for step in EVAL_STEPS:
            v = valid[:, step]
            for who, err in (("model", model_err[:, step]), ("cv", cv_err[:, step])):
                add(who, "all", step, err, v)
                add(who, "ball", step, err[:, :1], v[:, :1])
                add(who, "players", step, err[:, 1:], v[:, 1:])
                add(who, "fast", step, err[:, 1:], (v & (speed > 4.0))[:, 1:])

    model.train()
    return {k: sums[k] / counts[k] for k in sums}


def report(metrics: dict) -> None:
    header = "".join(f"{f't+{(s + 1) / TARGET_RATE:.1f}s':>18s}" for s in EVAL_STEPS)
    print(f"{'':16s}{header}")
    print(f"{'':16s}" + "".join(f"{'model':>9s}{'cv':>9s}" for _ in EVAL_STEPS))
    print("-" * (16 + 18 * len(EVAL_STEPS)))
    for key in ("all", "ball", "players", "fast"):
        row = ""
        for s in EVAL_STEPS:
            mv = metrics.get(("model", key, s))
            cv = metrics.get(("cv", key, s))
            row += f"{mv:9.2f}" if mv is not None else f"{'-':>9s}"
            row += f"{cv:9.2f}" if cv is not None else f"{'-':>9s}"
        print(f"{key:16s}{row}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--mode-weight", type=float, default=0.2)
    ap.add_argument("--wta-epsilon", type=float, default=0.05,
                    help="loser share in relaxed winner-take-all; 0 = pure WTA")
    ap.add_argument("--ball-selection-weight", type=float, default=1.0,
                    help="ball's share of the mode-selection metric; 0 = pooled, which collapses")
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--weighted-sampling", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    sequences = []
    for m in (1, 2):
        sequences.extend(cached_metrica(m, root=CACHE))

    train_seqs, val_seqs = split_by_match(sequences, val_matches=["metrica_2"])
    profiles = estimate_profiles(train_seqs)

    # Roles are inferred per match from average position: public data carries no
    # role labels, and without them every player looks identical to the model
    # apart from their team and physical profile.
    roles = infer_roles(sequences)
    player_index = build_player_index(sequences)
    print(f"roles inferred for {len(roles)} players: {summarise(sequences, roles)}")

    shared = dict(history=25, horizon=20, target_rate=TARGET_RATE, profiles=profiles,
                  roles=roles, player_index=player_index, seed=0)
    train = WindowDataset(train_seqs, stride=5, augment=True, **shared)
    val = WindowDataset(val_seqs, stride=25, augment=False, **shared)
    print(f"train {len(train)} windows / val {len(val)} windows")

    cfg = ModelConfig(d_model=args.d_model, depth=args.depth, horizon=20)
    model = Footrack(cfg).to(args.device)
    print(f"model  {model.n_parameters():,} parameters on {args.device}")

    opt = torch.optim.AdamW(model.param_groups(args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps_per_epoch), pct_start=0.15
    )
    gen = torch.Generator(device=args.device).manual_seed(0)

    print("\nbefore training:")
    report(evaluate(model, val, args.device, args.batch_size))

    CKPT.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        totals = np.zeros(3)
        mode_wins = np.zeros(cfg.n_modes)
        n = 0
        for batch in train.batches(args.batch_size, weighted=args.weighted_sampling):
            gpu = to_device(batch, args.device)
            masked, hidden = apply_agent_masking(gpu["entity"], args.mask_prob, gen)
            out = model(masked, gpu["context"], gpu["role_ids"], gpu["player_ids"])

            traj, mode, best = variety_loss(
                out["deltas"], out["mode_logits"], gpu["target_deltas"], gpu["target_mask"],
                epsilon=args.wta_epsilon, ball_selection_weight=args.ball_selection_weight,
            )
            mode_wins += np.bincount(best.cpu().numpy(), minlength=cfg.n_modes)
            recon = reconstruction_loss(
                out["recon"], gpu["entity"][..., 0:2], hidden, gpu["input_mask"]
            )
            loss = traj + args.mode_weight * mode + args.recon_weight * recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            totals += [traj.item(), mode.item(), recon.item()]
            n += 1

        traj_m, mode_m, recon_m = totals / max(n, 1)
        share = mode_wins / max(mode_wins.sum(), 1)
        alive = int((share > 0.01).sum())
        print(f"\nepoch {epoch:2d}  {time.time() - t0:5.1f}s  "
              f"traj {traj_m:.4f}  mode {mode_m:.4f}  recon {recon_m:.4f}")
        print(f"          modes alive {alive}/{cfg.n_modes}  "
              f"shares [{' '.join(f'{s:.2f}' for s in share)}]")
        report(evaluate(model, val, args.device, args.batch_size))

        torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "epoch": epoch},
                   CKPT / "s1_latest.pt")


if __name__ == "__main__":
    main()
