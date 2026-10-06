from io import StringIO
import unittest
import numpy as np
import pandas as pd
from finmodel.metrics import evaluate_frame
from finmodel.score_calibration import apply_score_calibration, fit_positive_affine, verify_order_preserved


class ScoreCalibrationTests(unittest.TestCase):
    def test_known_fit_ignores_invalid_and_nonfinite(self):
        score = np.array([0., .2, .4, .6, .8, 100., np.nan])
        label = .01 * score - .003
        label[-2] = -999
        mask = np.array([1, 1, 1, 1, 1, 0, 1], bool)
        result = fit_positive_affine(score, label, mask)
        self.assertAlmostEqual(result['slope'], .01)
        self.assertAlmostEqual(result['intercept'], -.003)
        self.assertEqual(result['observations'], 5)
        self.assertLess(result['training_errors']['calibrated_decision_score']['mse'], 1e-30)

    def test_reject_degenerate_and_reversed_calibration(self):
        for x, y, mask in (([1., 1.], [1., 2.], [1, 1]),
                           ([1., 2.], [2., 1.], [1, 1]),
                           ([1., 2.], [1., 2.], [0, 0])):
            with self.assertRaises(ValueError):
                fit_positive_affine(x, y, mask)
        with self.assertRaises(ValueError):
            apply_score_calibration([1.], {'slope': 0, 'intercept': 0})

    def test_csv_preserves_ties_rank_topk_turnover_and_score(self):
        rng = np.random.default_rng(2026)
        scores = rng.normal(size=(4, 300)).astype(np.float32)
        scores[:, :8] = .5
        labels = .005 * scores + rng.normal(scale=.02, size=scores.shape)
        fitted = fit_positive_affine(scores, labels, np.ones(scores.shape, bool))
        transformed = apply_score_calibration(scores, fitted)
        decoded = pd.read_csv(StringIO(pd.DataFrame({'pred': transformed.ravel()}).to_csv(
            index=False, float_format='%.17g')), float_precision='round_trip').pred.to_numpy().reshape(scores.shape)
        verify_order_preserved(scores, decoded)
        frame = pd.DataFrame(dict(ts_code=np.tile(np.arange(300), 4),
            trade_date=np.repeat(np.arange(20200101, 20200105), 300), pred_raw=scores.ravel(),
            y_ret_1d=labels.ravel(), flag_limit_up=np.tile(np.arange(300) % 31 == 0, 4)))
        before = evaluate_frame(frame)
        frame['pred_raw'] = decoded.ravel()
        after = evaluate_frame(frame)
        for key in ('ic_mean', 'annual_excess', 'mean_turnover', 'final_score'):
            self.assertEqual(before[key], after[key])
        with self.assertRaises(AssertionError):
            verify_order_preserved(scores, -transformed)
