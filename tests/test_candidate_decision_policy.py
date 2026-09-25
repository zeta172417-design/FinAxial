import unittest
from types import SimpleNamespace

import numpy as np
import torch

from finmodel.decision_history import observed_market_features
from finmodel.models import FinAxialDecisionPolicy


class CandidateDecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.hidden = torch.randn(5, 80, 16)
        self.score = torch.randn(5, 80) * 0.1
        self.eligible = torch.ones(5, 80, dtype=torch.bool)
        self.features = torch.randn(5, 80, 4) * 0.05

    def policy(self, action="hysteresis"):
        return FinAxialDecisionPolicy(
            d_model=16, action_mode=action, observation_mode="candidate_attention",
            history_days=4, candidate_max_count=32,
        )

    def test_market_features_use_only_known_prices_and_not_labels(self):
        feature = np.ones((7, 3, 6), np.float32)
        feature[..., 0] = 10
        feature[..., 1] = 12
        feature[..., 2] = 9
        feature[..., 3] = np.arange(7, dtype=np.float32)[:, None] + 10
        feature[..., 4] = np.arange(7, dtype=np.float32)[:, None] + 100
        panel = SimpleNamespace(
            features=feature, feature_valid=np.ones((7, 3), bool),
            labels=np.zeros((7, 3), np.float32),
        )
        original = observed_market_features(panel, np.array([2, 3]))
        panel.labels[:] = 1000
        feature[4:] = 1000
        np.testing.assert_array_equal(original, observed_market_features(panel, np.array([2, 3])))
        self.assertEqual(original.shape, (2, 3, 4))
        self.assertTrue(np.isfinite(original).all())

    def test_attention_has_same_zero_action_start_as_legacy(self):
        candidate = self.policy().eval()
        legacy = FinAxialDecisionPolicy(d_model=16, action_mode="hysteresis").eval()
        newer = candidate(
            self.hidden, self.score, self.eligible, self.eligible,
            observed_return=self.features,
        )
        older = legacy(self.hidden, self.score, self.eligible, self.eligible)
        torch.testing.assert_close(newer.decision_score, older.decision_score, atol=0, rtol=0)
        self.assertTrue(torch.equal(newer.selected, older.selected))

    def test_attention_is_causal_and_replay_backpropagates(self):
        policy = self.policy()
        with torch.no_grad():
            policy.action_head.weight.normal_(0, 0.05)
        first = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            observed_return=self.features,
        )
        changed = self.features.clone()
        changed[-1] += 3
        second = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            observed_return=changed,
        )
        torch.testing.assert_close(first.action_mean[:-1], second.action_mean[:-1], atol=0, rtol=0)
        self.assertGreater(float((first.action_mean[-1] - second.action_mean[-1]).abs().max().detach()), 0)
        replay = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            actions=first.raw_action.detach() + 0.1, observed_return=self.features,
        )
        (-replay.log_prob.mean()).backward()
        self.assertIsNotNone(policy.candidate_query.grad)
        self.assertTrue(bool(torch.isfinite(policy.candidate_query.grad).all()))
        self.assertGreater(float(policy.candidate_query.grad.abs().sum()), 0)

    def test_candidate_residual_is_bounded_and_changes_stock_selection(self):
        policy = self.policy("candidate_residual")
        initial = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            observed_return=self.features,
        )
        self.assertEqual(initial.raw_action.shape, (5, 9))
        self.assertEqual(initial.action_value.shape, (5, 9))
        fixed = initial.raw_action.clone()
        fixed[:, 3] = 4.0
        fixed[:, 4] = -4.0
        shifted = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            actions=fixed, observed_return=self.features,
        )
        self.assertTrue(bool(torch.isfinite(shifted.decision_score).all()))
        self.assertGreater(float((shifted.decision_score - initial.decision_score).abs().sum()), 0)
        self.assertTrue(bool((shifted.action_value[:, 3:].abs() <= 1).all()))
        replay = policy(
            self.hidden, self.score, self.eligible, self.eligible,
            actions=fixed, observed_return=self.features,
        )
        torch.testing.assert_close(shifted.log_prob, replay.log_prob, atol=0, rtol=0)

    def test_invalid_configuration_and_feature_shape(self):
        with self.assertRaises(ValueError):
            FinAxialDecisionPolicy(action_mode="candidate_residual")
        policy = self.policy()
        with self.assertRaises(ValueError):
            policy(
                self.hidden, self.score, self.eligible, self.eligible,
                observed_return=self.features[..., 0],
            )


if __name__ == "__main__":
    unittest.main()
