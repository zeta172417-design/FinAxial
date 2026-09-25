import unittest

import numpy as np
import pandas as pd
import torch

from finmodel.grpo import group_relative_advantage, hard_composite_score
from finmodel.metrics import evaluate_frame
from finmodel.models import FinAxialPolicyHead, StockTimeTransformer


def tiny_model(**overrides):
    values = {
        "stocks": 8,
        "lookback": 8,
        "output_steps": 3,
        "d_model": 32,
        "heads": 4,
        "temporal_layers": 2,
        "stock_layers": 2,
        "ffn_dim": 64,
        "stock_embedding_dim": 8,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "stock_id_dropout": 0.0,
    }
    values.update(overrides)
    return StockTimeTransformer(**values)


class GRPOPolicyTests(unittest.TestCase):
    def test_backbone_hidden_projection_matches_forward(self):
        torch.manual_seed(31)
        model = tiny_model(architecture="interleaved_axial").eval()
        values = torch.randn(8, 10, 6)
        token_valid = torch.ones(8, 10, dtype=torch.bool)
        eligible = torch.ones(3, 8, dtype=torch.bool)
        hidden = model.encode_hidden(values, token_valid, eligible)
        expected = model(values, token_valid, eligible)
        actual = model.score_hidden(hidden, eligible)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_policy_is_small_finite_and_future_causal(self):
        torch.manual_seed(37)
        policy = FinAxialPolicyHead(
            d_model=32, adapter_dim=16, state_dim=8, gate_hidden_dim=16,
        ).eval()
        hidden = torch.randn(4, 120, 32)
        base = torch.randn(4, 120)
        eligible = torch.ones(4, 120, dtype=torch.bool)
        tradable = eligible.clone()
        output = policy(hidden, base, eligible, tradable, sample=False)
        self.assertEqual(output.mean_score.shape, (4, 120))
        self.assertEqual(output.selected.sum(dim=1).tolist(), [12, 12, 12, 12])
        self.assertTrue(bool(torch.isfinite(output.mean_score).all()))
        self.assertGreaterEqual(float(output.gate_new.detach().min()), 0.05)
        self.assertLessEqual(float(output.gate_new.detach().max()), 1.0)
        self.assertLess(sum(parameter.numel() for parameter in policy.parameters()), 10_000)

        changed = hidden.clone()
        changed[-1] += 1_000.0
        changed_base = base.clone()
        changed_base[-1] += 1_000.0
        actual = policy(changed, changed_base, eligible, tradable, sample=False)
        torch.testing.assert_close(
            output.mean_score[:-1], actual.mean_score[:-1], rtol=0, atol=0,
        )

    def test_score_function_loss_backpropagates(self):
        torch.manual_seed(41)
        policy = FinAxialPolicyHead(
            d_model=32, adapter_dim=16, state_dim=8, gate_hidden_dim=16,
        )
        hidden = torch.randn(3, 120, 32)
        base = torch.randn(3, 120)
        eligible = torch.ones(3, 120, dtype=torch.bool)
        first = policy(
            hidden, base, eligible, eligible, sample=True,
            noise=torch.randn(3, 120),
        )
        second = policy(
            hidden, base, eligible, eligible, sample=True,
            noise=torch.randn(3, 120),
        )
        advantages = group_relative_advantage(torch.tensor([0.2, 0.4]))
        loss = -advantages[0] * first.log_prob - advantages[1] * second.log_prob
        loss = loss + 0.02 * (first.reference_kl + second.reference_kl)
        loss.backward()
        gradients = [
            parameter.grad for parameter in policy.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_hard_reward_matches_dataframe_evaluator(self):
        rng = np.random.default_rng(43)
        dates, stocks = 4, 120
        scores = rng.normal(size=(dates, stocks)).astype(np.float32)
        target = rng.normal(scale=0.02, size=(dates, stocks)).astype(np.float32)
        tradable = np.ones((dates, stocks), dtype=bool)
        tradable[:, ::37] = False
        label = np.ones_like(tradable)
        hard = hard_composite_score(
            torch.from_numpy(scores),
            torch.from_numpy(target),
            torch.from_numpy(label),
            torch.from_numpy(tradable),
        )
        rows = []
        for date in range(dates):
            for stock in range(stocks):
                rows.append({
                    "ts_code": f"S{stock:04d}",
                    "trade_date": 20250101 + date,
                    "pred_raw": float(scores[date, stock]),
                    "y_ret_1d": float(target[date, stock]),
                    "flag_limit_up": int(not tradable[date, stock]),
                    "eligible": True,
                })
        exact = evaluate_frame(pd.DataFrame(rows), "pred_raw")
        self.assertAlmostEqual(float(hard.rank_ic), exact["ic_mean"], places=6)
        self.assertAlmostEqual(float(hard.annual_excess), exact["annual_excess"], places=6)
        self.assertAlmostEqual(float(hard.stability), 1.0 - exact["mean_turnover"], places=6)
        self.assertAlmostEqual(float(hard.final_score), exact["final_score"], places=6)


if __name__ == "__main__":
    unittest.main()
