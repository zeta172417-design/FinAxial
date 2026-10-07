import unittest

import torch

from finmodel.decision_rl import exact_daily_score
from finmodel.grpo import hard_composite_score
from finmodel.models import FinAxialDecisionPolicy
from scripts.train_decoupled_decision_rl import normalize_advantages


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

    def test_global_scale_keeps_small_action_differences_small(self):
        values = torch.tensor([
            [-0.01, -1.0], [0.01, 1.0], [-0.01, -1.0], [0.01, 1.0],
        ])
        normalized = normalize_advantages(
            values, algorithm="daily_group", epsilon=1e-6, clip=5.0,
            mode="same_date_center_global_scale", scale_floor=0.05,
        )
        legacy = normalize_advantages(
            values, algorithm="daily_group", epsilon=1e-6, clip=5.0,
        )
        self.assertGreater(float(normalized[:, 1].std()),
                           50 * float(normalized[:, 0].std()))
        torch.testing.assert_close(legacy[:, 0], legacy[:, 1], rtol=0, atol=1e-6)
        shifted = values + torch.tensor([200.0, -70.0])
        torch.testing.assert_close(
            normalized,
            normalize_advantages(
                shifted, algorithm="daily_group", epsilon=1e-6, clip=5.0,
                mode="same_date_center_global_scale", scale_floor=0.05,
            ), atol=1e-5, rtol=0,
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


    def test_same_date_normalization_ignores_market_wide_shift(self):
        values = torch.tensor([[1., 3., 5.], [2., 5., 7.], [3., 7., 9.]])
        shifted = values + torch.tensor([100., -20., 500.])
        original = normalize_advantages(values, algorithm="daily_group", epsilon=1e-6, clip=5)
        changed = normalize_advantages(shifted, algorithm="daily_group", epsilon=1e-6, clip=5)
        torch.testing.assert_close(original, changed, atol=1e-5, rtol=0)
        torch.testing.assert_close(original.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)

    def test_grpo_replay_has_finite_gradients(self):
        torch.manual_seed(213)
        hidden = torch.randn(4, 40, 8)
        score = torch.randn(4, 40)
        target = torch.randn(4, 40) * .02
        mask = torch.ones(4, 40, dtype=torch.bool)
        actor = FinAxialDecisionPolicy(d_model=8, action_mode="hysteresis_no_alpha",
            return_feature_mode="explicit", margin_mode="predicted_return")
        with torch.no_grad():
            sampled = [actor(hidden, score, mask, mask, predicted_return=target, sample=True)
                       for _ in range(4)]
            reward = torch.stack([exact_daily_score(out.decision_score, target, mask, mask).reward
                                  for out in sampled])
            advantage = normalize_advantages(reward, algorithm="daily_group", epsilon=1e-6, clip=5)
        losses = []
        for index, out in enumerate(sampled):
            replay = actor(hidden, score, mask, mask, predicted_return=target, actions=out.raw_action)
            ratio = torch.exp((replay.log_prob - out.log_prob).clamp(-10, 10))
            loss = -torch.minimum(ratio * advantage[index], ratio.clamp(.8, 1.2) * advantage[index]).mean()
            losses.append(loss)
        objective = torch.stack(losses).mean() - .001 * replay.entropy
        objective.backward()
        self.assertTrue(bool(torch.isfinite(objective)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in actor.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
