import unittest

import torch

from finmodel.models import FinAxialDecisionPolicy
from finmodel.models.finaxial_decision_policy import (
    causal_ewma_scores,
    percentile_rank_scores,
)


class DecisionPolicyTests(unittest.TestCase):
    def test_initial_policy_exactly_matches_ewma_quarter(self):
        torch.manual_seed(101)
        dates, stocks, width = 5, 120, 32
        policy = FinAxialDecisionPolicy(d_model=width).eval()
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        output = policy(hidden, base, eligible, eligible, sample=False)
        rank = percentile_rank_scores(base, eligible)
        expected = causal_ewma_scores(rank, alpha=0.25)
        torch.testing.assert_close(output.decision_score, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            output.alpha, torch.full((dates,), 0.25), rtol=0, atol=1e-7,
        )
        torch.testing.assert_close(
            output.delta, torch.zeros(dates), rtol=0, atol=1e-7,
        )
        self.assertEqual(output.selected.sum(dim=1).tolist(), [12] * dates)

    def test_initial_policy_preserves_ewma_through_missing_then_reentry(self):
        torch.manual_seed(102)
        dates, stocks, width = 5, 120, 32
        policy = FinAxialDecisionPolicy(d_model=width).eval()
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        eligible[1:3, 4] = False
        output = policy(hidden, base, eligible, eligible, sample=False)
        expected = causal_ewma_scores(
            percentile_rank_scores(base, eligible), alpha=0.25,
        )
        torch.testing.assert_close(output.decision_score, expected, rtol=1e-6, atol=1e-6)

    def test_low_dimensional_action_replay_and_policy_gradient(self):
        torch.manual_seed(103)
        dates, stocks, width = 4, 120, 32
        policy = FinAxialDecisionPolicy(d_model=width)
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        noise = torch.randn(dates, 2)
        rollout = policy(
            hidden, base, eligible, eligible, sample=True, noise=noise,
        )
        replay = policy(
            hidden, base, eligible, eligible, actions=rollout.raw_action,
        )
        torch.testing.assert_close(
            replay.decision_score, rollout.decision_score, rtol=0, atol=0,
        )
        torch.testing.assert_close(replay.log_prob, rollout.log_prob, rtol=1e-6, atol=1e-6)
        self.assertEqual(tuple(rollout.raw_action.shape), (dates, 2))
        self.assertEqual(tuple(rollout.log_prob.shape), (dates,))
        loss = -0.7 * replay.log_prob.mean() + 0.01 * replay.reference_kl
        loss.backward()
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertLess(sum(p.numel() for p in policy.parameters()), 20_000)

    def test_four_action_selective_policy_starts_from_fixed_ewma(self):
        torch.manual_seed(104)
        dates, stocks, width = 5, 120, 32
        policy = FinAxialDecisionPolicy(
            d_model=width, action_mode="selective4", maximum_rank_bonus=0.10,
        ).eval()
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        output = policy(hidden, base, eligible, eligible, sample=False)
        expected = causal_ewma_scores(percentile_rank_scores(base, eligible), 0.25)
        torch.testing.assert_close(output.decision_score, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(tuple(output.raw_action.shape), (dates, 4))
        self.assertEqual(tuple(output.action_value.shape), (dates, 4))
        self.assertTrue(bool((output.hold_temperature > 0).all()))

    def test_six_action_bucket_policy_can_treat_rank_buckets_differently(self):
        torch.manual_seed(105)
        dates, stocks, width = 4, 120, 32
        policy = FinAxialDecisionPolicy(
            d_model=width, action_mode="bucketed6", maximum_rank_bonus=0.10,
        ).eval()
        hidden = torch.randn(dates, stocks, width)
        base = torch.randn(dates, stocks)
        eligible = torch.ones(dates, stocks, dtype=torch.bool)
        actions = policy.reference_action_mean.repeat(dates, 1)
        actions[:, 1:] = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        output = policy(hidden, base, eligible, eligible, actions=actions)
        self.assertEqual(tuple(output.raw_action.shape), (dates, 6))
        self.assertEqual(tuple(output.action_value.shape), (dates, 6))
        self.assertTrue(bool(torch.isfinite(output.decision_score).all()))
        self.assertLess(float(output.action_value[:, 1].mean()), 0.0)
        self.assertGreater(float(output.action_value[:, -1].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
