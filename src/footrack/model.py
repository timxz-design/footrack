"""The world model: spatio-temporal encoder with forecast and reconstruction heads.

Architecture follows MODEL.md §3. Four decisions carry most of the weight:

- **No positional encoding on the agent axis.** That is what makes the encoder
  permutation-equivariant within a team. Slots are distinguished by their
  features — team flag, role, tendency, position — not by their index, which is
  arbitrary.
- **Factorised attention.** Agent attention and time attention alternate. Full
  joint spatio-temporal attention costs (23 x T)^2 for no real benefit at this
  scale.
- **Scene-level modes, not per-agent.** All agents share one set of K mode
  weights, so mode k is a single coherent future. Per-agent modes would let the
  defender commit to the ball going left while the midfielder commits to right,
  producing scenes that never happen.
- **Tendency embeddings start at zero.** With weight decay pulling them back,
  that is the hierarchical shrinkage of MODEL.md §5.2 in its simplest form: the
  role embedding carries the mean, the tendency carries a regularised residual,
  and an unseen player sits at their role's average until data moves them.

Attention is unmasked. Absent entities are communicated through the ``present``
feature rather than an attention mask — the BERT arrangement, where a masked
token still attends and is attended to, which is exactly what reconstruction
needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from footrack.features import N_CONTEXT_FEATURES, N_ENTITY_FEATURES
from footrack.state import N_ENTITIES

# Feature-column indices the model needs to manipulate directly.
OBSERVATION_SLICE = slice(0, 19)
PRESENT_COL = 22


@dataclass
class ModelConfig:
    d_model: int = 128
    depth: int = 4
    heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    horizon: int = 20
    n_modes: int = 6
    n_roles: int = 24  # ROLES.md base x modifier combinations, plus unknown
    n_players: int = 1024  # tendency table size
    tendency_dim: int = 8
    max_history: int = 64

    @property
    def n_entities(self) -> int:
        return N_ENTITIES


class FactorisedBlock(nn.Module):
    """Agent attention, then time attention, then a feed-forward. Pre-norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_agent = nn.LayerNorm(d)
        self.attn_agent = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.norm_time = nn.LayerNorm(d)
        self.attn_time = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * cfg.ffn_mult, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, n, d = x.shape

        h = self.norm_agent(x).reshape(b * t, n, d)
        h, _ = self.attn_agent(h, h, h, need_weights=False)
        x = x + h.reshape(b, t, n, d)

        h = self.norm_time(x).permute(0, 2, 1, 3).reshape(b * n, t, d)
        h, _ = self.attn_time(h, h, h, need_weights=False)
        x = x + h.reshape(b, n, t, d).permute(0, 2, 1, 3)

        return x + self.ffn(self.norm_ffn(x))


class Encoder(nn.Module):
    """Entity features + context -> per-entity, per-timestep representations."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.entity_proj = nn.Linear(N_ENTITY_FEATURES, d)
        self.context_proj = nn.Linear(N_CONTEXT_FEATURES, d)

        self.role_emb = nn.Embedding(cfg.n_roles, d)
        # +1 row for "unknown player", which is where every new player starts.
        self.tendency_emb = nn.Embedding(cfg.n_players + 1, cfg.tendency_dim)
        self.tendency_proj = nn.Linear(cfg.tendency_dim, d, bias=False)

        # Time gets a positional encoding; agents deliberately do not.
        self.time_emb = nn.Parameter(torch.zeros(1, cfg.max_history, 1, d))

        self.blocks = nn.ModuleList(FactorisedBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.role_emb.weight, std=0.02)
        # Zero init: a player with no observed tendency contributes nothing, so
        # they inherit their role's behaviour exactly (MODEL.md §5.2).
        nn.init.zeros_(self.tendency_emb.weight)

    def forward(self, entity: Tensor, context: Tensor, role_ids: Tensor, player_ids: Tensor) -> Tensor:
        b, t, n, _ = entity.shape

        x = self.entity_proj(entity)
        x = x + self.context_proj(context).unsqueeze(2)
        x = x + self.role_emb(role_ids).unsqueeze(1)

        # -1 marks an unknown player; route those to the final table row.
        pid = torch.where(player_ids < 0, self.cfg.n_players, player_ids)
        x = x + self.tendency_proj(self.tendency_emb(pid)).unsqueeze(1)

        if t > self.cfg.max_history:
            raise ValueError(f"history {t} exceeds max_history {self.cfg.max_history}")
        x = x + self.time_emb[:, :t]

        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ForecastHead(nn.Module):
    """K coherent scene futures, as per-entity displacement from the anchor."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.traj = nn.Linear(cfg.d_model, cfg.n_modes * cfg.horizon * 2)
        self.mode = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_modes)
        )

    def forward(self, h: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
        """h: (B, N, D) at the final history step. present: (B, N).

        Returns deltas (B, K, F, N, 2) and mode logits (B, K).
        """
        b, n, _ = h.shape
        k, f = self.cfg.n_modes, self.cfg.horizon

        deltas = self.traj(h).view(b, n, k, f, 2).permute(0, 2, 3, 1, 4)

        # Scene representation: mean over observed entities only, so a match
        # with missing players is not dragged toward zero.
        w = present.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        return deltas, self.mode(pooled)


class ReconstructionHead(nn.Module):
    """Per-entity position at every timestep, for masked agent modelling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 2)
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class Footrack(nn.Module):
    """Encoder plus the stage S1 heads (MODEL.md §9)."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = Encoder(self.cfg)
        self.forecast = ForecastHead(self.cfg)
        self.reconstruct = ReconstructionHead(self.cfg)

    def forward(
        self, entity: Tensor, context: Tensor, role_ids: Tensor, player_ids: Tensor
    ) -> dict[str, Tensor]:
        h = self.encoder(entity, context, role_ids, player_ids)
        present = entity[:, -1, :, PRESENT_COL] > 0.5
        deltas, logits = self.forecast(h[:, -1], present)
        return {"hidden": h, "deltas": deltas, "mode_logits": logits,
                "recon": self.reconstruct(h)}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_groups(self, weight_decay: float = 0.01) -> list[dict]:
        """Weight decay everywhere except norms, biases and positional terms.

        Tendency embeddings are decayed deliberately: that decay *is* the
        shrinkage toward the role mean.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "time_emb" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


# -- masking ------------------------------------------------------------------


def apply_agent_masking(
    entity: Tensor, p: float, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Hide a random subset of entities for the reconstruction task.

    Zeroes the observation block and clears the ``present`` flag, leaving the
    structural features intact — the model must still know it is being asked
    about an opponent with a given physical profile.

    Returns the masked features and a (B, N) flag of which entities were hidden.
    Only entities that were actually observed can be chosen, since there is
    nothing to reconstruct for one that was already missing.
    """
    b, t, n, _ = entity.shape
    observed = entity[:, -1, :, PRESENT_COL] > 0.5

    draw = torch.rand(b, n, device=entity.device, generator=generator)
    hidden = (draw < p) & observed

    out = entity.clone()
    flag = hidden.unsqueeze(1).unsqueeze(-1)
    out[..., OBSERVATION_SLICE] = torch.where(
        flag, torch.zeros_like(out[..., OBSERVATION_SLICE]), out[..., OBSERVATION_SLICE]
    )
    out[..., PRESENT_COL] = torch.where(
        hidden.unsqueeze(1), torch.zeros_like(out[..., PRESENT_COL]), out[..., PRESENT_COL]
    )
    return out, hidden


# -- losses -------------------------------------------------------------------


def variety_loss(
    deltas: Tensor,
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    epsilon: float = 0.05,
    ball_selection_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Relaxed winner-take-all trajectory loss plus mode classification.

    Two failure modes sit either side of this, both measured rather than feared.

    **Equal averaging across modes** collapses them onto the mean future — a
    position no one ever occupies, since the ball goes left *or* right
    (MODEL.md §3.2).

    **Pure winner-take-all** starves the losers: whichever mode is best early
    wins every sample, takes all the gradient, and the rest never catch up. A
    2-epoch run left one mode winning 100% of windows with five stranded 28m
    away. Hence ``epsilon``: the winner takes ``1 - epsilon``, the losers share
    the rest, which is enough to keep them trainable. ``epsilon=0`` restores
    pure WTA.

    **A pooled selection metric drowns out the ball**, which is the subtler one.
    Averaging error over all 23 entities lets the 22 predictable players carry
    ~87% of the signal, so the mode that fits *them* best wins regardless of what
    it does with the ball. Measured after the epsilon fix: selecting on all
    entities used 1 of 6 modes, while selecting on the ball alone used all 6 —
    the modes were genuinely diverse and the metric could not see it.

    So selection scores the ball and the players separately and combines them
    with ``ball_selection_weight``, which also happens to be the semantically
    right choice: the ball's future largely determines the players' futures, so
    "which mode is this scene in" mostly means "where does the ball go".
    The trajectory *loss* still covers every entity — only the choice of winner
    is reweighted.

    **This is a trade-off, not a free win.** Measured at matched epochs on two
    matches, ``1.0`` raised effective mode count from 1.0 to ~2.4 and cost ~10%
    displacement accuracy: the winner is picked for ball fit but then trained on
    all 23 entities, so its gradient is less focused. The default is ``1.0``
    regardless, because collapse is not a mild inefficiency — it discards
    multimodality entirely, wasting five of six modes and leaving no uncertainty
    estimate. Revisit on the full corpus; the gap may close with more data.

    A weight of ``0.0`` restores the pooled metric that collapses.

    deltas (B, K, F, N, 2), logits (B, K), target (B, F, N, 2), mask (B, F, N).
    Returns (trajectory loss, mode loss, winning mode per sample).
    """
    err = torch.linalg.vector_norm(deltas - target.unsqueeze(1), dim=-1)  # (B,K,F,N)
    w = mask.unsqueeze(1).float()

    def pooled(sel: slice) -> Tensor:
        e, ww = err[..., sel], w[..., sel]
        return (e * ww).sum(dim=(2, 3)) / ww.sum(dim=(2, 3)).clamp(min=1.0)

    per_mode = pooled(slice(None))  # (B, K) — what the loss optimises
    a = float(ball_selection_weight)
    selection = a * pooled(slice(0, 1)) + (1.0 - a) * pooled(slice(1, None))

    # The ball is untracked in ~40% of frames (Metrica: 60% present, players
    # 100%). Without this, those samples score 0 for every mode — the ball term
    # divides by a clamped zero — and argmin silently returns mode 0, piling a
    # large share of windows onto one mode and faking a diversity result. Fall
    # back to the pooled metric whenever the ball has no valid target.
    has_ball = mask[:, :, 0].any(dim=1, keepdim=True)
    selection = torch.where(has_ball, selection, per_mode)

    best = selection.argmin(dim=1)
    k = per_mode.shape[1]

    if epsilon > 0 and k > 1:
        weights = torch.full_like(per_mode, epsilon / (k - 1))
        weights.scatter_(1, best.unsqueeze(1), 1.0 - epsilon)
        traj = (per_mode * weights).sum(dim=1).mean()
    else:
        traj = per_mode.gather(1, best.unsqueeze(1)).squeeze(1).mean()

    mode = F.cross_entropy(logits, best.detach())
    return traj, mode, best


def reconstruction_loss(
    recon: Tensor, target: Tensor, hidden: Tensor, observed: Tensor
) -> Tensor:
    """Masked-agent position loss, scored only on entities that were hidden.

    recon/target (B, T, N, 2), hidden (B, N), observed (B, T, N).
    """
    scored = hidden.unsqueeze(1) & observed
    if not scored.any():
        return recon.new_zeros(())
    err = torch.linalg.vector_norm(recon - target, dim=-1)
    return (err * scored.float()).sum() / scored.float().sum()
