import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import evaluate as official_evaluate
from finmodel.metrics import add_causal_ewma, evaluate_frame, percentile_rank
from finmodel.sft import metric_summary


class MetricTests(unittest.TestCase):
    def test_metric_summary_presents_official_scores_first(self):
        metrics = {
            "final_score": 0.12, "ic_mean": 0.1, "annual_excess": 0.2,
            "mean_turnover": 0.3, "mse": 0.01, "mae": 0.02,
            "icir": 0.5, "ic_positive_ratio": 0.6, "coverage": 0.9,
        }
        self.assertEqual(
            list(metric_summary(metrics))[:4],
            ["final_score", "rank_ic_x_0.4", "annual_excess_x_0.3", "stability_x_0.3"],
        )

    def test_exact_official_parity(self):
        rng = np.random.default_rng(17)
        rows = []
        for date in (20220104, 20220105, 20220106):
            for stock in range(120):
                rows.append({
                    "ts_code": f"S{stock:04d}", "trade_date": date,
                    "pred_raw": rng.normal(), "y_ret_1d": rng.normal(scale=0.02),
                    "flag_limit_up": int(stock % 37 == 0),
                    # Deliberately present to prove that the official scorer
                    # ignores limit-down flags rather than silently filtering
                    # them in our local implementation.
                    "flag_limit_down": int(stock % 41 == 0), "eligible": True,
                })
        frame = pd.DataFrame(rows)
        ours = evaluate_frame(frame)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame[["ts_code", "trade_date", "pred_raw"]].rename(columns={"pred_raw": "pred"}).to_csv(root / "submission.csv", index=False)
            frame[["ts_code", "trade_date", "y_ret_1d"]].to_csv(root / "测试集_Y.csv", index=False)
            frame[["ts_code", "trade_date", "flag_limit_up"]].to_csv(root / "测试集_X.csv", index=False)
            official = official_evaluate(str(root / "submission.csv"), str(root))
        for key in official:
            self.assertAlmostEqual(ours[key], official[key], places=12, msg=key)

    def test_rank_and_ewma_are_causal(self):
        frame = pd.DataFrame({
            "trade_date": [1, 1, 2, 2], "ts_code": ["A", "B", "A", "B"],
            "pred_raw": [1.0, 2.0, 4.0, 3.0],
        })
        frame["pred_rank"] = percentile_rank(frame)
        smooth = add_causal_ewma(frame, 0.5)
        self.assertEqual(list(frame["pred_rank"]), [0.5, 1.0, 1.0, 0.5])
        np.testing.assert_allclose(smooth, [0.5, 1.0, 0.75, 0.75])


if __name__ == "__main__":
    unittest.main()
