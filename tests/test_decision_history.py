import unittest
from types import SimpleNamespace

import numpy as np
import torch

from finmodel.decision_history import causal_c0_bridge_batch, observed_return_features
from finmodel.models import FinAxialDecisionPolicy
from finmodel.sequence import MultiDateCrossSectionDataset


class DecisionHistoryTests(unittest.TestCase):
    def test_c0_bridge_matches_dataset_inputs_without_using_y(self):
        rng = np.random.default_rng(232)
        features = rng.normal(10, 1, size=(20, 3, 6)).astype(np.float32)
        features[:, :, 3] += 10
        panel = SimpleNamespace(
            features=features, feature_valid=np.ones((20, 3), bool),
            labels=np.zeros((20, 3), np.float32),
            label_valid=np.ones((20, 3), bool),
            limit_flags=np.zeros((20, 3, 2), bool),
        )
        config = {
            "model": {"lookback": 4, "context_days": 2, "output_steps": 3},
            "data": {"feature_mode": "temporal", "normalization_epsilon": 1e-5,
                     "normalization_clip": 5.0},
            "min_history": 2,
        }
        dataset = MultiDateCrossSectionDataset(
            panel, np.arange(10, 13), lookback=4, context_days=2,
            output_steps=3, stride=3, min_history=2, feature_mode="temporal",
        )
        original = dataset[0]
        bridge = causal_c0_bridge_batch(panel, 12, config)
        for key, value in zip(("x", "token_valid", "eligible"), bridge):
            torch.testing.assert_close(original[key], value, atol=0, rtol=0)
        # The production bridge has no dependency on target values.
        panel.labels[:] = 999
        for expected, actual in zip(bridge, causal_c0_bridge_batch(panel, 12, config)):
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_observed_return_is_causal_and_never_reads_labels(self):
        close = np.array([[10., 20.], [11., 20.], [12., 18.], [90., 90.]], dtype=np.float32)
        feature = np.zeros((4, 2, 6), dtype=np.float32)
        feature[:, :, 3] = close
        panel = SimpleNamespace(features=feature, feature_valid=np.ones((4, 2), bool))
        prior = observed_return_features(panel, np.array([1, 2]))
        feature[-1] = 1000.0
        np.testing.assert_array_equal(prior, observed_return_features(panel, np.array([1, 2])))
        np.testing.assert_allclose(prior, [[5., 0.], [4.5454545, -5.]], atol=1e-5)

    def test_history_actor_preserves_initial_hysteresis_actions_and_causality(self):
        torch.manual_seed(230)
        policy = FinAxialDecisionPolicy(
            d_model=16, action_mode="hysteresis", history_days=16,
        ).eval()
        hidden = torch.randn(20, 120, 16)
        base = torch.randn(20, 120)
        eligible = torch.ones(20, 120, dtype=torch.bool)
        observed = torch.randn(20, 120)
        initial = policy(
            hidden, base, eligible, eligible, observed_return=observed,
        )
        torch.testing.assert_close(
            initial.alpha, torch.full((20,), 0.25), atol=1e-7, rtol=0,
        )
        self.assertTrue(bool((initial.swap_budget == 10).all()))
        with torch.no_grad():
            policy.action_head.weight.normal_(0, 0.02)
        original = policy(
            hidden, base, eligible, eligible, observed_return=observed,
        )
        changed = observed.clone()
        changed[-1] += 100
        later = policy(
            hidden, base, eligible, eligible, observed_return=changed,
        )
        torch.testing.assert_close(
            original.action_mean[:-1], later.action_mean[:-1], atol=0, rtol=0,
        )

    def test_prefix_and_history_parameters_affect_log_probability(self):
        torch.manual_seed(231)
        policy = FinAxialDecisionPolicy(
            d_model=16, action_mode="hysteresis", history_days=4,
        )
        with torch.no_grad():
            policy.action_head.weight.normal_(0, 0.02)
        hidden = torch.randn(3, 120, 16)
        base = torch.randn(3, 120)
        eligible = torch.ones(3, 120, dtype=torch.bool)
        observed = torch.randn(3, 120)
        prefix = (torch.randn(3, 120), eligible.clone(), torch.randn(3, 120))
        first = policy(
            hidden, base, eligible, eligible, observed_return=observed,
        )
        with_prefix = policy(
            hidden, base, eligible, eligible, observed_return=observed,
            history_prefix=prefix,
        )
        self.assertGreater(
            float((first.action_mean[0] - with_prefix.action_mean[0]).abs().max().detach()), 0,
        )
        actions = with_prefix.raw_action.detach() + 0.1
        replay = policy(
            hidden, base, eligible, eligible, actions=actions,
            observed_return=observed, history_prefix=prefix,
        )
        (-replay.log_prob.mean()).backward()
        gradient = policy.history_encoder.weight_ih_l0.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
