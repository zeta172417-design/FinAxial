import unittest

import torch

from finmodel.decision_rl import (
    DecisionValueHead, critic_observations, discounted_return_to_go, exact_daily_score,
    generalized_advantage,
)
from finmodel.grpo import hard_composite_score
from finmodel.models import FinAxialDecisionPolicy
from scripts.train_decoupled_decision_rl import explained_variance, normalize_advantages


class DecisionRLTests(unittest.TestCase):
    def test_daily_rewards_mean_matches_existing_exact_block_reward(self):
        torch.manual_seed(211)
        dates, stocks = 6, 120
        score = torch.randn(dates, stocks)
        target = torch.randn(dates, stocks) * 0.02
        labelled = torch.ones(dates, stocks, dtype=torch.bool)
        labelled[1, :5] = False
        tradable = torch.ones_like(labelled)
        tradable[:, :4] = False
        daily = exact_daily_score(score, target, labelled, tradable)
        aggregate = hard_composite_score(score, target, labelled, tradable)
        torch.testing.assert_close(daily.reward.mean(), aggregate.final_score, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(daily.rank_ic.mean(), aggregate.rank_ic, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(daily.annual_excess.mean(), aggregate.annual_excess, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(daily.stability[1:].mean(), aggregate.stability, rtol=1e-5, atol=1e-6)

    def test_gae_one_one_matches_returns_to_go(self):
        reward = torch.tensor([0.2, -0.1, 0.5])
        value = torch.tensor([0.1, 0.3, -0.2])
        advantage, returns = generalized_advantage(reward, value, gamma=1, lam=1)
        torch.testing.assert_close(returns, torch.tensor([0.6, 0.4, 0.5]), rtol=0, atol=1e-6)
        torch.testing.assert_close(advantage, returns - value, rtol=0, atol=1e-6)

    def test_discounted_future_reward_matches_zero_value_gae(self):
        reward = torch.tensor([0.2, -0.1, 0.5])
        future = discounted_return_to_go(reward, gamma=0.5)
        torch.testing.assert_close(future, torch.tensor([0.275, 0.15, 0.5]))
        gae, _ = generalized_advantage(
            reward, torch.zeros_like(reward), gamma=0.5, lam=1.0,
        )
        torch.testing.assert_close(future, gae)

    def test_same_date_normalization_ignores_market_wide_shift(self):
        values = torch.tensor([[1.0, 3.0, 5.0], [2.0, 5.0, 7.0], [3.0, 7.0, 9.0]])
        shifted = values + torch.tensor([100.0, -20.0, 500.0])
        for algorithm in ("daily_group", "ppo_gae_group", "rtg_group"):
            original = normalize_advantages(
                values, algorithm=algorithm, epsilon=1e-6, clip=5.0,
            )
            changed = normalize_advantages(
                shifted, algorithm=algorithm, epsilon=1e-6, clip=5.0,
            )
            torch.testing.assert_close(original, changed, atol=1e-5, rtol=0)
            torch.testing.assert_close(original.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        legacy = normalize_advantages(
            shifted, algorithm="ppo_gae", epsilon=1e-6, clip=5.0,
        )
        self.assertGreater(float((legacy - original).abs().max()), 0.1)

    def test_explained_variance_handles_constant_target(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(explained_variance(target, target)), 1.0)
        self.assertEqual(float(explained_variance(target, torch.ones_like(target))), 0.0)

    def test_critic_observation_excludes_future_and_value_is_finite(self):
        torch.manual_seed(212)
        hidden = torch.randn(4, 120, 16)
        base = torch.randn(4, 120)
        mask = torch.ones(4, 120, dtype=torch.bool)
        selected = torch.zeros_like(mask)
        selected[:, :12] = True
        score = torch.randn(4, 120)
        first = critic_observations(hidden, base, mask, mask, selected, score)
        hidden[-1] += 100
        base[-1] += 100
        second = critic_observations(hidden, base, mask, mask, selected, score)
        torch.testing.assert_close(first[:-1], second[:-1], rtol=0, atol=0)
        value = DecisionValueHead(d_model=16)(first)
        self.assertEqual(tuple(value.shape), (4,))
        self.assertTrue(bool(torch.isfinite(value).all()))

    def test_hysteresis_ppo_gae_replay_has_finite_gradients(self):
        torch.manual_seed(213)
        dates, stocks, width = 4, 120, 16
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        target = torch.randn(dates, stocks) * 0.02
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        actor = FinAxialDecisionPolicy(d_model=width, action_mode="hysteresis")
        critic = DecisionValueHead(d_model=width)

        with torch.no_grad():
            sampled = actor(hidden, base, eligible, eligible, sample=True)
            daily = exact_daily_score(
                sampled.decision_score, target, eligible, eligible,
            )
            observations = critic_observations(
                hidden, base, eligible, eligible,
                sampled.selected, sampled.decision_score,
            )
            old_log_prob = sampled.log_prob.detach()
            fixed_actions = sampled.raw_action.detach()
            advantage, returns = generalized_advantage(
                daily.reward, critic(observations), gamma=0.99, lam=0.95,
            )

        replay = actor(
            hidden, base, eligible, eligible, actions=fixed_actions,
        )
        ratio = torch.exp(replay.log_prob - old_log_prob)
        clipped = ratio.clamp(0.8, 1.2)
        actor_loss = -torch.minimum(
            ratio * advantage, clipped * advantage,
        ).mean() + 0.01 * replay.reference_kl
        actor_loss.backward()
        self.assertTrue(torch.isfinite(actor_loss).item())
        self.assertTrue(all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in actor.parameters()
        ))

        critic_loss = torch.nn.functional.mse_loss(
            critic(observations), returns,
        )
        critic_loss.backward()
        self.assertTrue(torch.isfinite(critic_loss).item())
        self.assertTrue(all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in critic.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
