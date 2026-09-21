import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from finmodel.panel import Panel, build_panel, build_phase1_panel
from finmodel.sft import cosine_learning_rate, sft_split
from finmodel.splits import assert_disjoint, make_folds, make_phase1_split
from finmodel.windows import normalized_cross_section, normalized_window


class PanelTests(unittest.TestCase):
    def test_sft_split_needs_no_test_period(self):
        dates = np.asarray(
            list(range(20240601, 20240706))
            + list(range(20240708, 20240718)),
        )
        config = {
            "tuning_train_end": 20240704,
            "boundary_excluded_date": 20240705,
            "validation_start": 20240708,
            "validation_end": 20240717,
            "validation_days": 10,
        }
        split = sft_split(SimpleNamespace(dates=dates), config)
        self.assertEqual(len(split.validation_dates), 10)
        self.assertNotIn(20240705, split.training_dates)
        self.assertLess(split.training_dates[-1], split.validation_dates[0])
        self.assertAlmostEqual(
            cosine_learning_rate(1e-3, update=2, total_updates=10, warmup_updates=2, eta_min_ratio=0.01),
            1e-3,
        )

    def test_causal_fill_and_future_independence(self):
        dates = [20180102, 20180103, 20180104, 20180105]
        rows = []
        for code in ("A", "B"):
            for idx, date in enumerate(dates):
                missing = code == "B" and idx == 0
                suspended = code == "A" and idx == 2
                close = np.nan if missing or suspended else 10 + idx + (code == "B")
                rows.append({
                    "ts_code": code, "trade_date": date,
                    "open": close, "high": close, "low": close, "close": close,
                    "vol": np.nan if suspended else 100, "amount": np.nan if suspended else 1000,
                    "flag_limit_up": 0, "flag_limit_down": 0, "y_ret_1d": idx / 100,
                })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv = root / "train.csv"
            pd.DataFrame(rows).to_csv(csv, index=False)
            build_panel(csv, root / "panel", chunksize=3)
            panel = Panel.open(root / "panel")
            self.assertEqual(panel.features.shape, (4, 2, 6))
            self.assertFalse(panel.feature_valid[0, 1])
            self.assertEqual(float(panel.features[2, 0, 3]), float(panel.features[1, 0, 3]))
            self.assertEqual(float(panel.features[2, 0, 4]), 0.0)
            before = normalized_window(panel, 1, 0, 4).values.copy()
            writable = Panel.open(root / "panel", mode="r+")
            writable.features[3, 0] = 9999
            writable.features.flush()
            after = normalized_window(panel, 1, 0, 4).values
            np.testing.assert_array_equal(before, after)
            cross_section, eligible = normalized_cross_section(panel, 3, 2, min_history=2)
            self.assertTrue(np.isfinite(cross_section).all())
            self.assertTrue(eligible.all())

    def test_walk_forward_boundary_exclusion(self):
        dates = []
        for year in range(2018, 2025):
            dates.extend(year * 10000 + 100 + day for day in range(1, 52))
        folds = make_folds(np.asarray(dates), validation_days=40)
        self.assertEqual([fold.evaluation_year for fold in folds], [2021, 2022, 2023, 2024])
        for fold in folds:
            assert_disjoint(fold)
            self.assertLess(fold.train_end, fold.validation_dates[0])
            self.assertLess(fold.validation_dates[-1], fold.excluded_boundary_date)
            self.assertLess(fold.excluded_boundary_date, fold.evaluation_dates[0])

    def test_phase1_boundary_and_validation_size(self):
        dates = np.asarray(
            list(range(20241001, 20241106))
            + list(range(20241106, 20241146))
            + [20241231, 20250102, 20250103],
        )
        # Use explicit cutoffs in this synthetic calendar so exactly 40 dates
        # are reserved and the boundary is absent from every selection segment.
        split = make_phase1_split(
            dates,
            tuning_train_end=20241104,
            boundary_excluded_date=20241105,
            validation_start=20241106,
            validation_end=20241145,
            final_train_end=20241231,
            test_start=20250102,
            test_end=20250103,
        )
        self.assertEqual(len(split.validation_dates), 40)
        self.assertNotIn(20241105, split.tuning_train_dates)
        self.assertNotIn(20241105, split.validation_dates)

    def test_phase1_panel_continues_history_without_mutating_sources(self):
        train_rows = []
        test_rows = []
        labels = []
        for code in ("A", "B"):
            for offset, date in enumerate((20241230, 20241231)):
                value = 10.0 + offset + (code == "B")
                train_rows.append({
                    "ts_code": code, "trade_date": date,
                    "open": value, "high": value, "low": value, "close": value,
                    "vol": 100, "amount": 1000, "flag_limit_up": 0,
                    "flag_limit_down": 0, "y_ret_1d": 0.01,
                })
            for offset, date in enumerate((20250102, 20250103)):
                suspended = code == "A" and offset == 0
                value = np.nan if suspended else 12.0 + offset + (code == "B")
                test_rows.append({
                    "ts_code": code, "trade_date": date,
                    "open": value, "high": value, "low": value, "close": value,
                    "vol": np.nan if suspended else 100, "amount": np.nan if suspended else 1000,
                    "flag_limit_up": 0, "flag_limit_down": 0,
                })
                labels.append({"ts_code": code, "trade_date": date, "y_ret_1d": 0.02})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_csv, test_csv, label_csv = root / "train.csv", root / "x.csv", root / "y.csv"
            pd.DataFrame(train_rows).to_csv(train_csv, index=False)
            pd.DataFrame(test_rows).to_csv(test_csv, index=False)
            pd.DataFrame(labels).to_csv(label_csv, index=False)
            originals = {path: path.read_bytes() for path in (train_csv, test_csv, label_csv)}
            build_panel(train_csv, root / "train_panel", chunksize=2)
            build_phase1_panel(root / "train_panel", test_csv, label_csv, root / "phase1", chunksize=2)
            panel = Panel.open(root / "phase1")
            self.assertEqual(panel.shape, (4, 2))
            self.assertEqual(float(panel.features[2, 0, 3]), float(panel.features[1, 0, 3]))
            self.assertTrue(panel.feature_valid[2, 0])
            self.assertEqual(float(panel.features[2, 0, 4]), 0.0)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
