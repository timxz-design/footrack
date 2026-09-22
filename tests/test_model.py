"""Model architecture and loss invariants.

The permutation-equivariance test is the important one: it verifies the claim
that makes the agent axis meaningful. Slot order is arbitrary, so relabelling
players within a team must permute the outputs and change nothing else. A stray
positional encoding on the agent axis would break it silently.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from footrack.features import N_CONTEXT_FEATURES, N_ENTITY_FEATURES
from footrack.model import (
    OBSERVATION_SLICE,
    PRESENT_COL,
    Footrack,
    ModelConfig,
    apply_agent_masking,
    reconstruction_loss,
    variety_loss,
)
from footrack.state import N_ENTITIES

B, T, F, K = 3, 12, 8, 4


def make_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    entity = torch.randn(B, T, N_ENTITIES, N_ENTITY_FEATURES, generator=g)
    # Valid one-hot team flags and present bits, as the feature builder emits.
    entity[..., 19:22] = 0.0
    entity[:, :, 0, 19] = 1.0
    entity[:, :, 1:12, 20] = 1.0
    entity[:, :, 12:, 21] = 1.0
    entity[..., PRESENT_COL] = (torch.rand(B, T, N_ENTITIES, generator=g) > 0.15).float()
    context = torch.randn(B, T, N_CONTEXT_FEATURES, generator=g)
    roles = torch.randint(0, 20, (B, N_ENTITIES), generator=g)
    players = torch.randint(-1, 40, (B, N_ENTITIES), generator=g)
    return entity, context, roles, players


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Footrack(ModelConfig(d_model=32, depth=2, heads=2, horizon=F, n_modes=K)).eval()


class TestArchitecture:
    def test_output_shapes(self, model):
        out = model(*make_batch())
        assert out["deltas"].shape == (B, K, F, N_ENTITIES, 2)
        assert out["mode_logits"].shape == (B, K)
        assert out["recon"].shape == (B, T, N_ENTITIES, 2)

    def test_parameter_count_in_documented_range(self):
        m = Footrack(ModelConfig(d_model=128, depth=4, horizon=20, n_modes=6))
        assert 2e5 < m.n_parameters() < 1e7, "MODEL.md §3.1 specifies 2-10M at full size"

    def test_all_parameters_receive_gradient(self):
        m = Footrack(ModelConfig(d_model=32, depth=2, heads=2, horizon=F, n_modes=K))
        out = m(*make_batch())
        target = torch.randn(B, F, N_ENTITIES, 2)
        mask = torch.ones(B, F, N_ENTITIES, dtype=torch.bool)
        traj, mode, _ = variety_loss(out["deltas"], out["mode_logits"], target, mask)
        (traj + mode + out["recon"].sum()).backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_history_longer_than_max_raises(self, model):
        entity, context, roles, players = make_batch()
        long = entity.repeat(1, 20, 1, 1)
        with pytest.raises(ValueError, match="max_history"):
            model(long, context.repeat(1, 20, 1), roles, players)

    def test_tendency_embedding_starts_at_zero(self):
        """An unseen player must contribute nothing and inherit their role."""
        m = Footrack(ModelConfig(d_model=32, depth=1))
        assert torch.all(m.encoder.tendency_emb.weight == 0)

    def test_param_groups_exclude_norms_and_biases(self):
        m = Footrack(ModelConfig(d_model=32, depth=2))
        groups = m.param_groups(weight_decay=0.01)
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0
        assert sum(p.numel() for g in groups for p in g["params"]) == m.n_parameters()


class TestPermutationEquivariance:
    def test_permuting_a_team_permutes_outputs(self, model):
        entity, context, roles, players = make_batch()
        with torch.no_grad():
            base = model(entity, context, roles, players)["deltas"]

            perm = torch.randperm(11) + 1
            idx = torch.cat([torch.tensor([0]), perm, torch.arange(12, N_ENTITIES)])
            permuted = model(entity[:, :, idx], context, roles[:, idx], players[:, idx])["deltas"]

        torch.testing.assert_close(base[:, :, :, idx], permuted, atol=1e-4, rtol=1e-3)

    def test_mode_logits_are_permutation_invariant(self, model):
        """Scene-level modes must not depend on slot ordering."""
        entity, context, roles, players = make_batch()
        with torch.no_grad():
            base = model(entity, context, roles, players)["mode_logits"]
            perm = torch.randperm(11) + 1
            idx = torch.cat([torch.tensor([0]), perm, torch.arange(12, N_ENTITIES)])
            after = model(entity[:, :, idx], context, roles[:, idx], players[:, idx])["mode_logits"]
        torch.testing.assert_close(base, after, atol=1e-4, rtol=1e-3)


class TestMasking:
    def test_observation_block_is_zeroed(self):
        entity, *_ = make_batch()
        masked, hidden = apply_agent_masking(entity, p=0.5)
        sel = hidden.unsqueeze(1).expand(B, T, N_ENTITIES)
        assert torch.all(masked[..., OBSERVATION_SLICE][sel] == 0)

    def test_present_flag_is_cleared(self):
        entity, *_ = make_batch()
        masked, hidden = apply_agent_masking(entity, p=0.5)
        assert torch.all(masked[..., PRESENT_COL][hidden.unsqueeze(1).expand(B, T, N_ENTITIES)] == 0)

    def test_structural_features_survive(self):
        """A hidden entity must stay identifiable, or there is nothing to
        condition the reconstruction on."""
        entity, *_ = make_batch()
        masked, hidden = apply_agent_masking(entity, p=0.5)
        torch.testing.assert_close(masked[..., 19:22], entity[..., 19:22])
        torch.testing.assert_close(masked[..., 24:27], entity[..., 24:27])

    def test_unmasked_entities_are_untouched(self):
        entity, *_ = make_batch()
        masked, hidden = apply_agent_masking(entity, p=0.5)
        keep = ~hidden.unsqueeze(1).expand(B, T, N_ENTITIES)
        torch.testing.assert_close(masked[keep], entity[keep])

    def test_only_observed_entities_can_be_hidden(self):
        entity, *_ = make_batch()
        entity[..., PRESENT_COL] = 0.0
        _, hidden = apply_agent_masking(entity, p=1.0)
        assert not hidden.any()

    def test_zero_probability_hides_nothing(self):
        entity, *_ = make_batch()
        masked, hidden = apply_agent_masking(entity, p=0.0)
        assert not hidden.any()
        torch.testing.assert_close(masked, entity)


class TestVarietyLoss:
    def setup_batch(self):
        deltas = torch.randn(B, K, F, N_ENTITIES, 2, requires_grad=True)
        logits = torch.randn(B, K, requires_grad=True)
        target = torch.randn(B, F, N_ENTITIES, 2)
        mask = torch.ones(B, F, N_ENTITIES, dtype=torch.bool)
        return deltas, logits, target, mask

    def test_winner_is_the_closest_mode(self):
        deltas, logits, target, mask = self.setup_batch()
        deltas = deltas.detach()
        deltas[:, 2] = target  # mode 2 is exact
        _, _, best = variety_loss(deltas, logits, target, mask)
        assert torch.all(best == 2)

    def test_relaxed_wta_keeps_losing_modes_trainable(self):
        """The failure this guards against: with pure WTA one mode won 100% of
        windows and the rest stayed stranded at initialisation."""
        deltas, logits, target, mask = self.setup_batch()
        traj, _, best = variety_loss(deltas, logits, target, mask, epsilon=0.05)
        traj.backward()
        for k in range(K):
            assert deltas.grad[:, k].abs().sum() > 0, f"mode {k} received no gradient"

    def test_pure_wta_starves_losers(self):
        deltas, logits, target, mask = self.setup_batch()
        traj, _, best = variety_loss(deltas, logits, target, mask, epsilon=0.0)
        traj.backward()
        for b in range(B):
            for k in range(K):
                got = deltas.grad[b, k].abs().sum() > 0
                assert got == (k == best[b].item())

    def test_winner_dominates_the_weighting(self):
        deltas, logits, target, mask = self.setup_batch()
        traj, _, best = variety_loss(deltas, logits, target, mask, epsilon=0.05)
        traj.backward()
        for b in range(B):
            w = best[b].item()
            losers = [k for k in range(K) if k != w]
            win_grad = deltas.grad[b, w].abs().sum()
            for k in losers:
                assert win_grad > deltas.grad[b, k].abs().sum()

    def test_masked_targets_are_ignored(self):
        deltas, logits, target, mask = self.setup_batch()
        mask[:, :, 5:] = False
        deltas = deltas.detach()
        deltas[:, 1, :, :5] = target[:, :, :5]
        _, _, best = variety_loss(deltas, logits, target, mask)
        assert torch.all(best == 1)


class TestBallWeightedSelection:
    """Guards the second collapse: with selection pooled over all 23 entities,
    the 22 predictable players carry ~87% of the signal and the ball's
    multimodality becomes invisible."""

    def setup_batch(self):
        torch.manual_seed(1)
        deltas = torch.randn(B, K, F, N_ENTITIES, 2)
        logits = torch.randn(B, K)
        target = torch.randn(B, F, N_ENTITIES, 2)
        mask = torch.ones(B, F, N_ENTITIES, dtype=torch.bool)
        return deltas, logits, target, mask

    def test_ball_weight_one_selects_on_ball_alone(self):
        deltas, logits, target, mask = self.setup_batch()
        deltas[:, 3, :, 0] = target[:, :, 0]  # mode 3 nails the ball only
        _, _, best = variety_loss(deltas, logits, target, mask, ball_selection_weight=1.0)
        assert torch.all(best == 3)

    def test_ball_weight_zero_ignores_the_ball(self):
        deltas, logits, target, mask = self.setup_batch()
        deltas[:, 3, :, 0] = target[:, :, 0]  # ball-perfect mode
        deltas[:, 1, :, 1:] = target[:, :, 1:]  # player-perfect mode
        _, _, best = variety_loss(deltas, logits, target, mask, ball_selection_weight=0.0)
        assert torch.all(best == 1)

    def test_ball_weight_shifts_the_winner(self):
        """The same predictions must select differently under the two metrics —
        otherwise the parameter is doing nothing."""
        deltas, logits, target, mask = self.setup_batch()
        deltas[:, 3, :, 0] = target[:, :, 0]
        deltas[:, 1, :, 1:] = target[:, :, 1:]
        _, _, pooled = variety_loss(deltas, logits, target, mask, ball_selection_weight=0.0)
        _, _, ball = variety_loss(deltas, logits, target, mask, ball_selection_weight=1.0)
        assert not torch.equal(pooled, ball)

    def test_loss_still_covers_every_entity(self):
        """Only selection is reweighted; the trajectory loss trains on all 23."""
        deltas, logits, target, mask = self.setup_batch()
        deltas.requires_grad_(True)
        traj, _, _ = variety_loss(deltas, logits, target, mask, ball_selection_weight=1.0)
        traj.backward()
        for n in range(N_ENTITIES):
            assert deltas.grad[:, :, :, n].abs().sum() > 0, f"entity {n} got no gradient"


class TestReconstructionLoss:
    def test_scores_only_hidden_entities(self):
        recon = torch.zeros(B, T, N_ENTITIES, 2)
        target = torch.ones(B, T, N_ENTITIES, 2)
        observed = torch.ones(B, T, N_ENTITIES, dtype=torch.bool)
        hidden = torch.zeros(B, N_ENTITIES, dtype=torch.bool)
        hidden[:, 3] = True
        loss = reconstruction_loss(recon, target, hidden, observed)
        assert loss.item() == pytest.approx(2.0**0.5, rel=1e-4)

    def test_no_hidden_entities_gives_zero(self):
        recon = torch.zeros(B, T, N_ENTITIES, 2)
        target = torch.ones(B, T, N_ENTITIES, 2)
        observed = torch.ones(B, T, N_ENTITIES, dtype=torch.bool)
        hidden = torch.zeros(B, N_ENTITIES, dtype=torch.bool)
        assert reconstruction_loss(recon, target, hidden, observed).item() == 0.0

    def test_unobserved_frames_excluded(self):
        recon = torch.zeros(B, T, N_ENTITIES, 2)
        target = torch.ones(B, T, N_ENTITIES, 2)
        observed = torch.zeros(B, T, N_ENTITIES, dtype=torch.bool)
        hidden = torch.ones(B, N_ENTITIES, dtype=torch.bool)
        assert reconstruction_loss(recon, target, hidden, observed).item() == 0.0
