import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from finmodel.factors import FACTOR_PROFILES, FACTOR_VERSION
from finmodel.inference import build_inference_panel, infer_final_model
from finmodel.io import atomic_json_dump, sha256_file
from finmodel.models import StockTimeTransformer, build_decision_policy, stock_vocab_sha256
from finmodel.panel import Panel, build_panel


class FinalInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        records = []
        for day in range(23):
            for stock in range(20):
                close = 10 + day * .01 + stock * .1
                records.append(dict(ts_code=f'S{stock:03d}', trade_date=20200101 + day,
                    open=close, high=close + .02, low=close - .02, close=close,
                    vol=100 + day, amount=1000 + day, flag_limit_up=0, flag_limit_down=0,
                    y_ret_1d=.001))
        frame = pd.DataFrame(records)
        frame[frame.trade_date < 20200121].to_csv(self.root / 'train.csv', index=False)
        self.test = frame[frame.trade_date >= 20200121].drop(columns='y_ret_1d').copy()
        self.test.to_csv(self.root / 'test.csv', index=False)
        build_panel(self.root / 'train.csv', self.root / 'history')

    def tearDown(self):
        self.temporary.cleanup()

    def test_unlabelled_final_date_causal_fill_and_duplicate_detection(self):
        x = self.test.copy()
        x.loc[x.index[0], ['open', 'high', 'low', 'close', 'vol', 'amount']] = np.nan
        x.to_csv(self.root / 'missing.csv', index=False)
        panel = Panel.open(build_inference_panel(self.root / 'history', self.root / 'missing.csv', self.root / 'panel', chunksize=21))
        self.assertEqual(panel.shape, (23, 20))
        self.assertFalse(panel.label_valid[20:].any())
        self.assertTrue(np.isnan(panel.labels[20:]).all())
        self.assertEqual(panel.dates[-1], 20200123)
        self.assertEqual(panel.features[20, 0, 3], panel.features[19, 0, 3])
        self.assertEqual(panel.features[20, 0, 4], 0)
        duplicate = pd.concat((self.test, self.test.iloc[:1]))
        duplicate.to_csv(self.root / 'duplicate.csv', index=False)
        with self.assertRaises(ValueError):
            build_inference_panel(self.root / 'history', self.root / 'duplicate.csv', self.root / 'bad')

    def test_end_to_end_all_keys_and_labels_cannot_change_predictions(self):
        panel = Panel.open(build_inference_panel(self.root / 'history', self.root / 'test.csv', self.root / 'panel'))
        architecture = dict(lookback=3, output_steps=8, context_days=0, temporal_window=8,
            channels=128, d_model=16, heads=4, temporal_layers=2, stock_layers=2,
            ffn_dim=32, stock_embedding_dim=4, dropout=0, attention_dropout=0,
            stock_id_dropout=0, architecture='interleaved_axial', head_mode='dual', return_scale=.02)
        predictor = StockTimeTransformer(stocks=20, **architecture)
        torch.save(predictor.state_dict(), self.root / 'predictor.pt')
        atomic_json_dump({'architecture': architecture, 'stock_vocab_sha256': stock_vocab_sha256(panel.codes)},
                         self.root / 'predictor.json')
        policy_config = dict(action_mode='hysteresis_no_alpha', market_dim=4, recurrent_dim=4,
                             return_feature_mode='explicit', margin_mode='predicted_return')
        policy = build_decision_policy(d_model=16, **policy_config)
        torch.save(policy.state_dict(), self.root / 'policy.pt')
        atomic_json_dump({'policy_config': policy_config, 'backbone_sha256': sha256_file(self.root / 'predictor.pt'),
            'stock_vocab_sha256': stock_vocab_sha256(panel.codes), 'validation_burnin_days': 2}, self.root / 'policy.json')
        atomic_json_dump({'factor_version': FACTOR_VERSION, 'feature_names': FACTOR_PROFILES['f128'][0],
                          'centers': [0.] * 122, 'scales': [1.] * 122}, self.root / 'calibration.json')
        config = dict(model=architecture, policy=policy_config, min_history=2,
            backbone_checkpoint=str(self.root / 'predictor.pt'), backbone_metadata=str(self.root / 'predictor.json'),
            validation={'decision_burnin_days': 2, 'score_output_positions': [4, 8]},
            data={'normalization_epsilon': 1e-5, 'normalization_clip': 5.})
        def run(name, chosen):
            return infer_final_model(panel=chosen, config=config, policy_path=self.root / 'policy.pt',
                policy_metadata=self.root / 'policy.json', calibration_path=self.root / 'calibration.json',
                workdir=self.root / name, output=self.root / (name + '.csv'), device=torch.device('cpu'), expected_days=3)
        reference = run('first', panel)
        self.assertEqual(len(reference), 60)
        self.assertEqual(list(reference.columns), ['ts_code', 'trade_date', 'pred'])
        self.assertEqual(reference.trade_date.max(), 20200123)
        atomic_json_dump({'slope': .008, 'intercept': -.004, 'fit_date_end': 20200120,
            'predictor_sha256': sha256_file(self.root / 'predictor.pt'),
            'policy_sha256': sha256_file(self.root / 'policy.pt')}, self.root / 'score_calibration.json')
        calibrated = infer_final_model(panel=panel, config=config, policy_path=self.root / 'policy.pt',
            policy_metadata=self.root / 'policy.json', calibration_path=self.root / 'calibration.json',
            workdir=self.root / 'calibrated', output=self.root / 'calibrated.csv', device=torch.device('cpu'),
            expected_days=3, score_calibration_path=self.root / 'score_calibration.json')
        np.testing.assert_allclose(calibrated.pred, reference.pred.to_numpy().astype(np.float64) * .008 - .004,
                                   rtol=0, atol=0)
        writable = Panel.open(self.root / 'panel', mode='r+')
        writable.labels[:] = 999
        writable.label_valid[:] = True
        second = run('second', writable)
        np.testing.assert_array_equal(reference.pred.to_numpy(), second.pred.to_numpy())
        changed = Panel.open(self.root / 'panel', mode='r+')
        changed.features[-1] *= 1.2
        third = run('third', changed)
        np.testing.assert_array_equal(reference[reference.trade_date < 20200123].pred.to_numpy(),
                                      third[third.trade_date < 20200123].pred.to_numpy())
        manifest = json.loads((self.root / 'first.csv.manifest.json').read_text())
        self.assertFalse(manifest['evaluation_labels_read'])
        self.assertTrue(manifest['final_feature_date_included'])
        with self.assertRaises(FileExistsError):
            run('first', panel)


if __name__ == '__main__':
    unittest.main()
