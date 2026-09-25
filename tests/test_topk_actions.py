import unittest

import torch

from finmodel.models import FinAxialDecisionPolicy
from finmodel.models.finaxial_policy import hard_top_fraction_mask
from finmodel.models.topk_actions import (
    project_selection_to_scores, select_hysteresis, select_swap_budget,
)


class TopKActionTests(unittest.TestCase):
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
