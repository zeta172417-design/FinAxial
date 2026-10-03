import unittest

import torch

from finmodel.decision_rl import (
    DecisionValueHead, center_daily_rewards_against_group,
    critic_observations, discounted_return_to_go, exact_daily_score,
    generalized_advantage,
)
from finmodel.grpo import hard_composite_score
from finmodel.models import FinAxialDecisionPolicy
from scripts.train_decoupled_decision_rl import explained_variance, normalize_advantages
from finmodel.sft import load_config


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

    def test_discounted_horizon_group_rewards(self):
        reward = torch.tensor([0.2, -0.1, 0.5, 0.3])
        torch.testing.assert_close(
            discounted_return_to_go(reward, gamma=0.5, horizon=1), reward,
        )
        torch.testing.assert_close(
            discounted_return_to_go(reward, gamma=0.5, horizon=2),
            torch.tensor([0.15, 0.15, 0.65, 0.3]),
        )
        torch.testing.assert_close(
            discounted_return_to_go(reward, gamma=0.5, horizon=4),
            discounted_return_to_go(reward, gamma=0.5),
        )
        changed = reward.clone()
        changed[-1] += 100
        torch.testing.assert_close(
            discounted_return_to_go(changed, gamma=0.5, horizon=2)[:2],
            discounted_return_to_go(reward, gamma=0.5, horizon=2)[:2],
        )

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

    def test_group_centered_critic_target_removes_common_market_shock(self):
        group = torch.tensor([
            [10.0, -4.0], [10.2, -4.1], [9.8, -3.9], [10.1, -4.2],
        ], dtype=torch.float64)
        local = group[:2]
        centered = center_daily_rewards_against_group(local, group)
        torch.testing.assert_close(
            centered,
            center_daily_rewards_against_group(
                local + torch.tensor([500.0, -70.0]),
                group + torch.tensor([500.0, -70.0]),
            ), atol=1e-5, rtol=0,
        )
        torch.testing.assert_close(centered, local - group.mean(dim=0), rtol=0, atol=1e-6)
        self.assertLess(float(centered.std()), float(group.std()))
        with self.assertRaises(ValueError):
            center_daily_rewards_against_group(local, group[:, :1])

    def test_global_scale_keeps_small_action_differences_small(self):
        values = torch.tensor([
            [-0.01, -1.0], [0.01, 1.0], [-0.01, -1.0], [0.01, 1.0],
        ])
        normalized = normalize_advantages(
            values, algorithm="ppo_gae_group", epsilon=1e-6, clip=5.0,
            mode="same_date_center_global_scale", scale_floor=0.05,
        )
        legacy = normalize_advantages(
            values, algorithm="ppo_gae_group", epsilon=1e-6, clip=5.0,
        )
        self.assertGreater(float(normalized[:, 1].std()),
                           50 * float(normalized[:, 0].std()))
        torch.testing.assert_close(legacy[:, 0], legacy[:, 1], rtol=0, atol=1e-6)
        shifted = values + torch.tensor([200.0, -70.0])
        torch.testing.assert_close(
            normalized,
            normalize_advantages(
                shifted, algorithm="ppo_gae_group", epsilon=1e-6, clip=5.0,
                mode="same_date_center_global_scale", scale_floor=0.05,
            ), atol=1e-5, rtol=0,
        )

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

    def test_critic_can_use_dual_head_return_forecast(self):
        torch.manual_seed(214)
        hidden = torch.randn(3, 20, 16)
        base = torch.randn(3, 20)
        forecast = torch.randn(3, 20) * 0.02
        mask = torch.ones(3, 20, dtype=torch.bool)
        selected = torch.zeros_like(mask)
        selected[:, :2] = True
        proxy = critic_observations(hidden, base, mask, mask, selected, base)
        explicit = critic_observations(
            hidden, base, mask, mask, selected, base,
            predicted_return=forecast,
        )
        self.assertEqual(tuple(explicit.shape), (3, 22))
        torch.testing.assert_close(explicit[:, :16], proxy[:, :16], rtol=0, atol=0)
        self.assertGreater(float((explicit[:, 16:18] - proxy[:, 16:18]).abs().max()), 0.01)
        later = forecast.clone()
        later[-1] += 0.5
        changed = critic_observations(
            hidden, base, mask, mask, selected, base,
            predicted_return=later,
        )
        torch.testing.assert_close(explicit[:-1], changed[:-1], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "predicted_return"):
            critic_observations(
                hidden, base, mask, mask, selected, base,
                predicted_return=forecast[:, :-1],
            )





    def test_portfolio_detail_preserves_initial_actor_and_receives_gradients(self):
        options = dict(
            d_model=16, action_mode="hysteresis_no_alpha",
            return_feature_mode="explicit", margin_mode="predicted_return",
            initial_swap_budget=10, candidate_max_count=40,
        )
        torch.manual_seed(216)
        plain = FinAxialDecisionPolicy(**options)
        torch.manual_seed(216)
        detail = FinAxialDecisionPolicy(**options, observation_mode="portfolio_detail")
        for name, parameter in plain.state_dict().items():
            torch.testing.assert_close(parameter, detail.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(bool((detail.detail_fusion.weight == 0).all()))
        hidden = torch.randn(3, 120, 16)
        base = torch.randn(3, 120)
        predicted_return = torch.randn(3, 120) * 0.02
        eligible = torch.ones(3, 120, dtype=torch.bool)
        with torch.no_grad():
            first = plain(hidden, base, eligible, eligible,
                          predicted_return=predicted_return, sample=False)
            second = detail(hidden, base, eligible, eligible,
                            predicted_return=predicted_return, sample=False)
        torch.testing.assert_close(first.action_mean, second.action_mean, rtol=0, atol=0)
        torch.testing.assert_close(first.decision_score, second.decision_score, rtol=0, atol=0)
        self.assertGreater(float(second.candidate_count.mean()), 0.0)
        with torch.no_grad():
            detail.action_head.weight.fill_(0.01)
        detail(hidden, base, eligible, eligible,
               predicted_return=predicted_return, sample=False).action_mean.square().sum().backward()
        self.assertGreater(float(detail.detail_fusion.weight.grad.abs().sum()), 0.0)

    def test_portfolio_detail_value_sees_previous_but_not_current_holding(self):
        torch.manual_seed(217)
        hidden = torch.randn(3, 30, 16)
        base = torch.randn(3, 30)
        mask = torch.ones(3, 30, dtype=torch.bool)
        selected = torch.zeros_like(mask)
        selected[:, :3] = True
        original = critic_observations(
            hidden, base, mask, mask, selected, base,
            predicted_return=base * 0.02, portfolio_detail=True,
        )
        self.assertEqual(tuple(original.shape), (3, 70))
        modified = selected.clone()
        modified[1, :3] = False
        modified[1, 5:8] = True
        changed = critic_observations(
            hidden, base, mask, mask, modified, base,
            predicted_return=base * 0.02, portfolio_detail=True,
        )
        torch.testing.assert_close(original[:2], changed[:2], rtol=0, atol=0)
        self.assertGreater(float((original[2] - changed[2]).abs().sum()), 0.0)
        value = DecisionValueHead(d_model=16, portfolio_detail=True)(original)
        self.assertEqual(tuple(value.shape), (3,))

    def test_factor_branch_preserves_shared_initialization(self):
        options = dict(
            d_model=16, action_mode="hysteresis_no_alpha",
            return_feature_mode="explicit", margin_mode="predicted_return",
            initial_swap_budget=10,
        )
        torch.manual_seed(215)
        plain = FinAxialDecisionPolicy(**options)
        for dimension in (14, 16):
            torch.manual_seed(215)
            factored = FinAxialDecisionPolicy(**options, decision_factor_dim=dimension)
            for name, parameter in plain.state_dict().items():
                torch.testing.assert_close(parameter, factored.state_dict()[name], rtol=0, atol=0)

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
