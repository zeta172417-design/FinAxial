import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from finmodel.models.stock_time_transformer import StockTimeTransformer
from finmodel.objective import compose_bounded_final_score, multi_date_soft_components
from finmodel.sequence import rolling_causal_zscore


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


class StockTimeTransformerTests(unittest.TestCase):
    def test_rolling_normalization_is_future_independent(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(15, 4, 6)).astype(np.float32)
        valid = np.ones((15, 4), dtype=bool)
        expected, counts = rolling_causal_zscore(raw, valid, lookback=8)
        changed = raw.copy()
        changed[10:] += 1000.0
        actual, changed_counts = rolling_causal_zscore(changed, valid, lookback=8)
        np.testing.assert_allclose(expected[:10], actual[:10], rtol=0, atol=0)
        np.testing.assert_array_equal(counts, changed_counts)

    def test_output_shape_centering_and_shared_stock_parameters(self):
        model = tiny_model().eval()
        values = torch.randn(8, 10, 6)
        token_valid = torch.ones(8, 10, dtype=torch.bool)
        eligible = torch.ones(3, 8, dtype=torch.bool)
        prediction = model(values, token_valid, eligible)
        self.assertEqual(prediction.shape, (3, 8))
        torch.testing.assert_close(prediction.mean(dim=1), torch.zeros(3), atol=1e-6, rtol=0)
        names = [name for name, _ in model.named_parameters() if "stock_blocks.0.attention.qkv" in name]
        self.assertEqual(names, [
            "stock_blocks.0.attention.qkv.weight",
            "stock_blocks.0.attention.qkv.bias",
        ])

    def test_sixteen_date_output_shape(self):
        model = tiny_model(output_steps=16).eval()
        sequence_length = model.lookback + model.output_steps - 1
        values = torch.randn(8, sequence_length, 6)
        token_valid = torch.ones(8, sequence_length, dtype=torch.bool)
        eligible = torch.ones(16, 8, dtype=torch.bool)
        prediction = model(values, token_valid, eligible)
        self.assertEqual(prediction.shape, (16, 8))
        torch.testing.assert_close(
            prediction.mean(dim=1), torch.zeros(16), atol=1e-6, rtol=0,
        )

    def test_explicit_burnin_shape(self):
        model = tiny_model(
            lookback=8, context_days=4, temporal_window=8, output_steps=6,
        ).eval()
        self.assertEqual(model.sequence_length, 10)
        prediction = model(
            torch.randn(8, 10, 6),
            torch.ones(8, 10, dtype=torch.bool),
            torch.ones(6, 8, dtype=torch.bool),
        )
        self.assertEqual(prediction.shape, (6, 8))

    def test_future_token_cannot_change_earlier_predictions(self):
        torch.manual_seed(9)
        model = tiny_model().eval()
        values = torch.randn(8, 10, 6)
        token_valid = torch.ones(8, 10, dtype=torch.bool)
        eligible = torch.ones(3, 8, dtype=torch.bool)
        expected = model(values, token_valid, eligible)
        changed = values.clone()
        changed[:, -1] += 1e4
        actual = model(changed, token_valid, eligible)
        torch.testing.assert_close(expected[:-1], actual[:-1], rtol=1e-5, atol=1e-5)

    def test_stock_permutation_equivariance(self):
        torch.manual_seed(13)
        model = tiny_model().eval()
        values = torch.randn(8, 10, 6)
        token_valid = torch.ones(8, 10, dtype=torch.bool)
        eligible = torch.ones(3, 8, dtype=torch.bool)
        expected = model(values, token_valid, eligible)
        permutation = torch.randperm(8)
        actual = model(
            values[permutation], token_valid[permutation], eligible[:, permutation], permutation,
        )
        torch.testing.assert_close(actual, expected[:, permutation], rtol=1e-5, atol=1e-5)

    def test_sequence_final_loss_is_finite_and_backpropagates(self):
        prediction = torch.randn(3, 8, requires_grad=True)
        target = torch.randn(3, 8) * 0.02
        mask = torch.ones(3, 8, dtype=torch.bool)
        components = multi_date_soft_components(prediction, target, mask, mask)
        loss, score, excess = compose_bounded_final_score(
            components,
            global_rank_ic=components.rank_ic,
            global_annual_excess_raw=components.annual_excess_raw,
            global_stability=components.stability,
        )
        self.assertTrue(bool(torch.isfinite(torch.stack([loss, score, excess])).all()))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_checkpoint_round_trip_is_exact(self):
        torch.manual_seed(23)
        model = tiny_model().eval()
        values = torch.randn(8, 10, 6)
        token_valid = torch.ones(8, 10, dtype=torch.bool)
        eligible = torch.ones(3, 8, dtype=torch.bool)
        expected = model(values, token_valid, eligible)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = tiny_model().eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        actual = restored(values, token_valid, eligible)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
