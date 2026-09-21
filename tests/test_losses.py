import unittest

import torch

from finmodel.losses import (
    masked_huber,
    masked_mse,
    pairwise_logistic_loss,
    pearson_loss,
    stockmixer_official_loss,
    stockmixer_soft_score_loss,
)


class LossTests(unittest.TestCase):
    def test_degenerate_inputs_are_finite(self):
        cases = [
            (torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.bool)),
            (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([True])),
            (torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0]), torch.tensor([True, True])),
            (torch.tensor([1e8, -1e8]), torch.tensor([-1e6, 1e6]), torch.tensor([True, True])),
        ]
        for prediction, target, mask in cases:
            prediction.requires_grad_(True)
            values = [
                masked_mse(prediction, target, mask),
                masked_huber(prediction, target, mask),
                pearson_loss(prediction, target, mask),
                pairwise_logistic_loss(prediction, target, mask),
            ]
            self.assertTrue(all(torch.isfinite(value).item() for value in values))
            self.assertTrue(
                all(torch.isfinite(value).item() for value in stockmixer_official_loss(prediction, target, mask))
            )

    def test_official_pairwise_orientation(self):
        target = torch.tensor([-1.0, 0.0, 1.0])
        correctly_ordered = torch.tensor([-2.0, 0.0, 2.0])
        _, _, correct_rank = stockmixer_official_loss(correctly_ordered, target)
        _, _, reversed_rank = stockmixer_official_loss(-correctly_ordered, target)
        self.assertEqual(float(correct_rank), 0.0)
        self.assertGreater(float(reversed_rank), 0.0)

    def test_soft_final_only_has_no_mse_term_and_backpropagates(self):
        target = torch.linspace(-0.02, 0.02, 16)
        previous = torch.linspace(-1.0, 1.0, 16, requires_grad=True)
        current = torch.linspace(-0.8, 1.2, 16, requires_grad=True)
        mask = torch.ones(16, dtype=torch.bool)
        result = stockmixer_soft_score_loss(
            previous,
            current,
            target,
            target.roll(1),
            mask,
            mask,
            mask,
            mask,
            rank_temperature=0.1,
            top_temperature=0.02,
            excess_scale=0.5,
            mse_weight=0.0,
        )
        self.assertTrue(torch.isfinite(result.loss).item())
        self.assertAlmostEqual(float(result.loss.detach()), -float(result.score.detach()), places=7)
        result.loss.backward()
        self.assertTrue(torch.isfinite(previous.grad).all())
        self.assertTrue(torch.isfinite(current.grad).all())

    def test_soft_score_bounds_excess_and_rewards_stability(self):
        target = torch.linspace(-0.02, 0.02, 20)
        previous = target.clone().requires_grad_(True)
        stable = target.clone().requires_grad_(True)
        unstable = (-target).clone().requires_grad_(True)
        mask = torch.ones(20, dtype=torch.bool)
        stable_result = stockmixer_soft_score_loss(
            previous, stable, target, target, mask, mask, mask, mask,
            excess_scale=0.5, mse_weight=0.0,
        )
        unstable_result = stockmixer_soft_score_loss(
            previous, unstable, target, target, mask, mask, mask, mask,
            excess_scale=0.5, mse_weight=0.0,
        )
        self.assertLessEqual(abs(float(stable_result.annual_excess_objective.detach())), 0.5)
        self.assertGreater(
            float(stable_result.stability.detach()), float(unstable_result.stability.detach())
        )

    def test_ic_and_portfolio_use_official_universes(self):
        target = torch.linspace(-0.03, 0.03, 12)
        prediction = target.clone()
        changed = prediction.clone()
        changed[-1] = -1.0
        labelled = torch.ones(12, dtype=torch.bool)
        tradable = labelled.clone()
        tradable[-1] = False
        original = stockmixer_soft_score_loss(
            prediction, prediction, target, target, labelled, labelled, tradable, tradable,
        )
        modified = stockmixer_soft_score_loss(
            changed, changed, target, target, labelled, labelled, tradable, tradable,
        )
        self.assertNotAlmostEqual(float(original.rank_ic), float(modified.rank_ic), places=5)
        self.assertAlmostEqual(float(original.annual_excess_raw), float(modified.annual_excess_raw), places=7)
        self.assertAlmostEqual(float(original.stability), float(modified.stability), places=7)


if __name__ == "__main__":
    unittest.main()
