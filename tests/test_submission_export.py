import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from finmodel.io import sha256_file
from scripts.calibrate_submission import export_submission


class SubmissionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw.csv'
        self.features = self.root / 'features.csv'
        self.calibration = self.root / 'calibration.json'
        self.output = self.root / 'submission.csv'
        scores = np.tile(np.array([0., .3, .3, .9], dtype=np.float32), 3)
        frame = pd.DataFrame({'ts_code': np.tile(['A','B','C','D'], 3),
                              'trade_date': np.repeat([20250102,20250103,20250106], 4), 'pred': scores})
        frame.to_csv(self.raw, index=False, float_format='%.17g')
        frame[['ts_code','trade_date']].iloc[::-1].to_csv(self.features, index=False)
        self.raw.with_suffix('.csv.manifest.json').write_text(json.dumps({
            'csv_sha256': sha256_file(self.raw), 'evaluation_labels_read': False,
            'predictor_sha256': 'predictor', 'policy_sha256': 'policy'}))
        self.settings = {'slope': .008, 'intercept': -.004, 'fit_date_end': 20241230,
                         'predictor_sha256': 'predictor', 'policy_sha256': 'policy',
                         'evaluation_labels_read': False}
        self.calibration.write_text(json.dumps(self.settings))

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self):
        return export_submission(self.raw, self.features, self.calibration, self.output, expected_days=3)

    def test_complete_keys_final_date_ties_and_no_overwrite(self):
        result = self.run_export()
        self.assertEqual((result['rows'], result['dates'], result['stocks']), (12,3,4))
        self.assertEqual(result['date_end'], 20250106)
        self.assertTrue(result['all_days_order_and_ties_preserved_after_csv_reload'])
        frame = pd.read_csv(self.output, float_precision='round_trip')
        raw = pd.read_csv(self.raw, float_precision='round_trip')
        np.testing.assert_array_equal(frame.pred, raw.pred.to_numpy() * .008 - .004)
        with self.assertRaises(FileExistsError):
            self.run_export()

    def test_reject_weight_mismatch_and_test_period_fit(self):
        for changes in ({'policy_sha256':'wrong'}, {'fit_date_end':20250102},
                        {'evaluation_labels_read':True}, {'slope':-1}):
            self.calibration.write_text(json.dumps({**self.settings, **changes}))
            with self.assertRaises(ValueError):
                self.run_export()
            self.assertFalse(self.output.exists())

    def test_reject_incomplete_coverage_or_tampered_raw(self):
        expected = pd.read_csv(self.features)
        expected.iloc[:-1].to_csv(self.features, index=False)
        with self.assertRaises(ValueError):
            self.run_export()
        self.raw.write_text('ts_code,trade_date,pred\nA,20250102,1\n')
        with self.assertRaises(ValueError):
            self.run_export()


if __name__ == '__main__':
    unittest.main()
