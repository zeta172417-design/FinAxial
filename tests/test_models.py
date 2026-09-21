import tempfile
import unittest
from pathlib import Path

import torch

from finmodel.models.stockmixer import StockMixerReturn


class ModelTests(unittest.TestCase):
    def test_stockmixer_save_reload(self):
        torch.manual_seed(3)
        model = StockMixerReturn(stocks=8, lookback=16).eval()
        values = torch.randn(8, 16, 6)
        expected = model(values).detach()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = StockMixerReturn(stocks=8, lookback=16).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(expected, restored(values))

    def test_stockmixer_supported_lookbacks(self):
        for lookback in (16, 32, 64, 128):
            model = StockMixerReturn(stocks=8, lookback=lookback)
            self.assertEqual(model(torch.randn(8, lookback, 6)).shape, (8,))

    def test_stockmixer_masks_ineligible_stocks(self):
        model = StockMixerReturn(stocks=8, lookback=64).eval()
        values = torch.randn(8, 64, 6)
        eligible = torch.ones(8, dtype=torch.bool)
        eligible[-1] = False
        changed = values.clone()
        changed[-1] = 1e6
        torch.testing.assert_close(model(values, eligible), model(changed, eligible))


if __name__ == "__main__":
    unittest.main()
