from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from finmodel.decision_cache import DecisionFeatureCache
from finmodel.decision_validation import with_validation_burnin
from finmodel.models import FinAxialDecisionPolicy
from scripts.train_decoupled_decision_rl import evaluate_cached


def cache(dates):
    days, stocks = len(dates), 20
    rng = np.random.default_rng(2026)
    return DecisionFeatureCache(
        root=Path('.'), date_indices=np.array(dates),
        hidden=rng.normal(size=(days, stocks, 8)).astype('float32'),
        base_score=rng.normal(size=(days, stocks)).astype('float32'),
        eligible=np.ones((days, stocks), bool), tradable=np.ones((days, stocks), bool),
        predicted_return=rng.normal(size=(days, stocks)).astype('float32') * .02,
        manifest={'backbone_sha256': 'a', 'panel_manifest_sha256': 'b'},
    )


class DecisionValidationTests(unittest.TestCase):
    def test_prefix_only_reads_past_X_and_keeps_validation_unchanged(self):
        train, validation = cache([1, 2, 3]), cache([4, 5])
        # No labels or label_valid attributes: warming must be X-only.
        panel = SimpleNamespace(limit_flags=np.zeros((6, 20, 2), bool))
        combined = with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 2)
        np.testing.assert_array_equal(combined.date_indices, [2, 3, 4, 5])
        np.testing.assert_array_equal(combined.hidden[:2], train.hidden[-2:])
        np.testing.assert_array_equal(combined.hidden[2:], validation.hidden)
        np.testing.assert_array_equal(validation.date_indices, [4, 5])
        self.assertIs(with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 0), validation)
        with self.assertRaises(ValueError):
            with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 5)

    def test_missing_boundary_uses_X_only_bridge_and_checks_hashes(self):
        train, validation = cache([1, 2]), cache([4, 5])
        panel = SimpleNamespace(limit_flags=np.zeros((6, 20, 2), bool))
        backbone = SimpleNamespace(
            encode_hidden=lambda *args, **kwargs: torch.ones(1, 20, 8),
            score_hidden=lambda *args: torch.ones(1, 20),
            predict_return_hidden=lambda *args: torch.ones(1, 20) * .03,
        )
        bridge = (torch.zeros(20, 1, 6), torch.ones(20, 1, dtype=torch.bool),
                  torch.ones(1, 20, dtype=torch.bool))
        with patch('finmodel.decision_validation.load_backbone', return_value=(backbone, {})), \
             patch('finmodel.decision_validation.causal_predictor_bridge_batch', return_value=bridge) as mocked:
            combined = with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 2)
            mocked.assert_called_once_with(panel, 3, {})
            np.testing.assert_allclose(combined.predicted_return[1], .03)
        def random_load(*args):
            torch.rand(10)
            return backbone, {}
        before = torch.random.get_rng_state().clone()
        with patch('finmodel.decision_validation.load_backbone', side_effect=random_load), \
             patch('finmodel.decision_validation.causal_predictor_bridge_batch', return_value=bridge):
            with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 2)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        validation.manifest['backbone_sha256'] = 'different'
        with self.assertRaises(ValueError):
            with_validation_burnin(train, validation, panel, {}, torch.device('cpu'), 2)

    def test_evaluator_scores_only_suffix_after_one_continuous_actor_call(self):
        data = cache([1, 2, 3, 4])
        panel = SimpleNamespace(limit_flags=np.zeros((5, 20, 2), bool))
        actor = FinAxialDecisionPolicy(d_model=8, market_dim=4, recurrent_dim=4,
                                      action_mode='hysteresis_no_alpha',
                                      return_feature_mode='explicit', margin_mode='predicted_return')
        with patch('scripts.train_decoupled_decision_rl.score_numpy_predictions', return_value=({}, None)) as scorer:
            result = evaluate_cached(actor, data, panel, torch.device('cpu'),
                                     route='test', base_metrics={}, score_prefix=2)
        arguments = scorer.call_args.kwargs
        np.testing.assert_array_equal(arguments['indices'], [3, 4])
        with torch.inference_mode():
            reference = actor(torch.from_numpy(data.hidden), torch.from_numpy(data.base_score),
                              torch.from_numpy(data.eligible), torch.from_numpy(data.tradable),
                              predicted_return=torch.from_numpy(data.predicted_return), sample=False)
        np.testing.assert_array_equal(arguments['predictions'], reference.decision_score[2:].numpy())
        self.assertAlmostEqual(result['diagnostics']['swap_budget_mean'],
                               float(reference.swap_budget[2:].mean()), places=6)


if __name__ == '__main__':
    unittest.main()
