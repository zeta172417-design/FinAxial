import unittest

import torch

from finmodel.models import FinAxialDecisionPolicy
from finmodel.models.finaxial_policy import hard_top_fraction_mask
from finmodel.models.topk_actions import (
    project_selection_to_scores, select_hysteresis, select_swap_budget,
)
from finmodel.sft import load_config


class TopKActionTests(unittest.TestCase):
    def test_uncapped_budget_can_grow_above_initial_until_portfolio_size(self):
        policy = FinAxialDecisionPolicy(
            d_model=8, action_mode="hysteresis_no_alpha",
            return_feature_mode="explicit", margin_mode="predicted_return",
            initial_swap_budget=50, maximum_swap_budget=None,
        ).eval()
        hidden = torch.zeros(1, 1000, 8)
        score = torch.arange(1000, dtype=torch.float32)[None]
        valid = torch.ones(1, 1000, dtype=torch.bool)
        forecast = torch.zeros(1, 1000)
        actions = policy.reference_action_mean[None].clone()
        actions[0, 1] = torch.log(torch.tensor(2.0))
        output = policy(hidden, score, valid, valid,
                        predicted_return=forecast, actions=actions)
        self.assertEqual(float(output.swap_budget[0]), 100.0)
        actions[0, 1] = 8.0
        capped_by_portfolio = policy(hidden, score, valid, valid,
                                     predicted_return=forecast, actions=actions)
        self.assertEqual(float(capped_by_portfolio.swap_budget[0]), 100.0)

    def test_budget_retains_strongest_old_and_fills_from_outside(self):
        scores = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.6, 0.5])
        old = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        trade = torch.ones(6, dtype=torch.bool)
        selected = select_swap_budget(scores, trade, old, k=3, swap_budget=1)
        self.assertEqual(torch.nonzero(selected).flatten().tolist(), [0, 2, 3])
        self.assertEqual(int((selected & ~old).sum()), 1)

    def test_budget_handles_forced_exit(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        old = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        trade = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.bool)
        selected = select_swap_budget(scores, trade, old, k=3, swap_budget=0)
        self.assertEqual(torch.nonzero(selected).flatten().tolist(), [1, 2, 3])

    def test_hysteresis_only_swaps_when_margin_cleared(self):
        scores = torch.tensor([0.90, 0.80, 0.70, 0.72, 0.69, 0.20])
        old = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        trade = torch.ones(6, dtype=torch.bool)
        no_swap = select_hysteresis(scores, trade, old, k=3, max_swaps=2, margin=0.03)
        one_swap = select_hysteresis(scores, trade, old, k=3, max_swaps=2, margin=0.01)
        self.assertEqual(torch.nonzero(no_swap).flatten().tolist(), [0, 1, 2])
        self.assertEqual(torch.nonzero(one_swap).flatten().tolist(), [0, 1, 3])

    def test_return_margin_uses_forecast_spread_not_rank_gap(self):
        scores = torch.tensor([0.90, 0.80, 0.70, 0.72, 0.69, 0.20])
        old = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        trade = torch.ones(6, dtype=torch.bool)
        forecast = torch.tensor([0.0, 0.0, -0.01, 0.0, 0.0, 0.0])
        selected = select_hysteresis(
            scores, trade, old, k=3, max_swaps=1,
            margin=0.008, predicted_return=forecast,
        )
        self.assertEqual(torch.nonzero(selected).flatten().tolist(), [0, 1, 3])
        rejected = select_hysteresis(
            scores, trade, old, k=3, max_swaps=1,
            margin=0.012, predicted_return=forecast,
        )
        self.assertEqual(torch.nonzero(rejected).flatten().tolist(), [0, 1, 2])

    def test_return_margin_checks_each_pair_independently(self):
        scores = torch.tensor([0.95, 0.80, 0.70, 0.90, 0.85])
        old = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
        forecast = torch.tensor([0.0, 0.0, 0.01, 0.0, 0.02])
        selected = select_hysteresis(
            scores, torch.ones(5, dtype=torch.bool), old,
            k=3, max_swaps=2, margin=0.008,
            predicted_return=forecast,
        )
        self.assertEqual(torch.nonzero(selected).flatten().tolist(), [0, 2, 4])

    def test_explicit_return_margin_reaches_policy_and_replays(self):
        policy = FinAxialDecisionPolicy(
            d_model=8, action_mode="hysteresis", return_feature_mode="explicit",
            margin_mode="predicted_return", initial_hysteresis_margin=0.008,
            maximum_hysteresis_margin=0.04, initial_swap_budget=1,
            maximum_swap_budget=3, top_fraction=0.5,
        ).eval()
        hidden = torch.zeros(2, 6, 8)
        score = torch.tensor([[6., 5., 4., 3., 2., 1.], [6., 5., 3., 4., 2., 1.]])
        eligible = torch.ones(2, 6, dtype=torch.bool)
        forecast = torch.tensor([[0., 0., 0., 0., 0., 0.], [0., 0., -0.01, 0., 0., 0.]])
        output = policy(hidden, score, eligible, eligible, predicted_return=forecast)
        self.assertTrue(bool(output.selected[1, 3]))
        forecast[1, 3] = -0.009
        rejected = policy(hidden, score, eligible, eligible, predicted_return=forecast)
        self.assertFalse(bool(rejected.selected[1, 3]))
        replay = policy(hidden, score, eligible, eligible,
                        predicted_return=forecast, actions=rejected.raw_action)
        torch.testing.assert_close(rejected.decision_score, replay.decision_score)
        sampled = policy(hidden, score, eligible, eligible,
                         predicted_return=forecast, sample=True)
        sampled_replay = policy(hidden, score, eligible, eligible,
                                predicted_return=forecast, actions=sampled.raw_action)
        torch.testing.assert_close(sampled.log_prob, sampled_replay.log_prob)
        (-sampled_replay.log_prob.mean() + 0.01 * sampled_replay.reference_kl).backward()
        self.assertTrue(all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in policy.parameters() if parameter.grad is not None
        ))
        with self.assertRaises(ValueError):
            policy(hidden, score, eligible, eligible)

    def test_projection_matches_exact_topk_without_touching_remote_scores(self):
        scores = torch.tensor([0.99, 0.94, 0.91, 0.90, 0.89, 0.11, 0.01])
        trade = torch.ones(7, dtype=torch.bool)
        selected = torch.tensor([1, 0, 1, 1, 0, 0, 0], dtype=torch.bool)
        projected = project_selection_to_scores(scores, selected, trade)
        actual = hard_top_fraction_mask(projected[None], trade[None], 3 / 7)[0]
        self.assertTrue(torch.equal(actual, selected))
        self.assertEqual(projected[0].item(), scores[0].item())
        self.assertEqual(projected[-1].item(), scores[-1].item())

    def test_each_new_mode_replays_actions_and_has_finite_gradients(self):
        torch.manual_seed(2030)
        hidden = torch.randn(5, 120, 32)
        base = torch.randn(5, 120)
        eligible = torch.ones(5, 120, dtype=torch.bool)
        trade = eligible.clone()
        trade[2, 3] = False
        for mode, n_actions in (
            ("swap_budget_only", 1),
            ("swap_budget_raw", 1),
            ("swap_budget_alpha", 2),
            ("hysteresis", 3),
            ("boundary4", 4),
        ):
            with self.subTest(mode=mode):
                policy = FinAxialDecisionPolicy(d_model=32, action_mode=mode)
                sampled = policy(hidden, base, eligible, trade, sample=True)
                replay = policy(hidden, base, eligible, trade, actions=sampled.raw_action)
                self.assertEqual(tuple(sampled.raw_action.shape), (5, n_actions))
                torch.testing.assert_close(sampled.decision_score, replay.decision_score)
                torch.testing.assert_close(sampled.log_prob, replay.log_prob)
                self.assertTrue(torch.equal(
                    sampled.selected,
                    hard_top_fraction_mask(sampled.decision_score, trade, 0.1),
                ))
                if mode == "swap_budget_only":
                    torch.testing.assert_close(sampled.alpha, torch.full((5,), 0.25))
                if mode == "swap_budget_raw":
                    torch.testing.assert_close(sampled.alpha, torch.ones(5))
                loss = -sampled.log_prob.mean() + 0.01 * sampled.reference_kl
                loss.backward()
                self.assertTrue(all(
                    bool(torch.isfinite(p.grad).all())
                    for p in policy.parameters() if p.grad is not None
                ))


if __name__ == "__main__":
    unittest.main()
