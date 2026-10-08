#!/usr/bin/env python3
"""Apply train-only affine calibration to a complete label-free prediction CSV."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from finmodel.io import atomic_json_dump, sha256_file
from finmodel.score_calibration import apply_score_calibration, verify_order_preserved


def export_submission(raw, features, calibration, output, expected_days=344):
    raw, features, calibration, output = map(Path, (raw, features, calibration, output))
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    temporary = output.with_suffix(output.suffix + '.partial')
    if any(path.exists() for path in (output, temporary, manifest_path)):
        raise FileExistsError('refusing to overwrite an existing submission or partial output')
    source = json.loads(raw.with_suffix(raw.suffix + '.manifest.json').read_text())
    fitted = json.loads(calibration.read_text())
    if source['csv_sha256'] != sha256_file(raw):
        raise ValueError('raw CSV does not match its inference manifest')
    if source.get('evaluation_labels_read') is not False or fitted.get('evaluation_labels_read') is not False:
        raise ValueError('inference and calibration must not use evaluation labels')
    for name in ('predictor_sha256', 'policy_sha256'):
        if source[name] != fitted[name]:
            raise ValueError('calibration belongs to different model weights')
    frame = pd.read_csv(raw, dtype={'ts_code': str}, float_precision='round_trip')
    keys = ['ts_code', 'trade_date']
    if list(frame.columns) != keys + ['pred'] or frame[keys].isna().any().any():
        raise ValueError('submission requires exactly ts_code,trade_date,pred and finite keys')
    if frame[keys].duplicated().any() or not np.isfinite(frame.pred).all():
        raise ValueError('duplicate keys or nonfinite predictions')
    expected = pd.read_csv(features, usecols=keys, dtype={'ts_code': str})
    observed_keys = pd.MultiIndex.from_frame(frame[keys])
    expected_keys = pd.MultiIndex.from_frame(expected[keys])
    if (expected_keys.has_duplicates or len(expected) != len(frame)
            or (observed_keys.get_indexer(expected_keys) < 0).any()):
        raise ValueError('prediction keys do not exactly cover the original test features')
    dates = np.sort(frame.trade_date.unique())
    stocks = frame.ts_code.nunique()
    if len(dates) != expected_days or not (frame.groupby('trade_date').size() == stocks).all():
        raise ValueError('submission does not cover the expected complete rectangular test panel')
    if int(fitted['fit_date_end']) >= int(dates[0]):
        raise ValueError('calibration fit must end before the test period')
    raw_values = frame.pred.to_numpy(copy=True)
    frame['pred'] = apply_score_calibration(raw_values, fitted)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temporary, index=False, float_format='%.17g', encoding='utf-8')
    # Check the actual parser used by the official evaluator, not just memory.
    decoded = pd.read_csv(temporary, dtype={'ts_code': str})
    for indices in frame.groupby('trade_date', sort=True).indices.values():
        verify_order_preserved(raw_values[indices][None, :], decoded.pred.to_numpy()[indices][None, :])
    if list(decoded.columns) != keys + ['pred'] or not np.isfinite(decoded.pred).all():
        raise AssertionError('exported CSV format or finite-value verification failed')
    temporary.rename(output)
    manifest = {
        **source,
        'csv_sha256': sha256_file(output), 'rows': len(frame), 'dates': len(dates), 'stocks': stocks,
        'date_start': int(dates[0]), 'date_end': int(dates[-1]),
        'raw_csv': str(raw), 'raw_csv_sha256': sha256_file(raw),
        'test_features_sha256': sha256_file(features), 'score_calibration': str(calibration),
        'score_calibration_sha256': sha256_file(calibration),
        'slope': fitted['slope'], 'intercept': fitted['intercept'],
        'columns': keys + ['pred'], 'csv_precision': '%.17g',
        'pred_semantics': 'positive affine training-calibrated portfolio score in decimal return units',
        'all_keys_verified': True, 'all_days_order_and_ties_preserved_after_csv_reload': True,
        'final_feature_date_included': True, 'evaluation_labels_read': False,
    }
    atomic_json_dump(manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='artifacts/raw_predictions.csv')
    parser.add_argument('--features', default='data/test_features.csv')
    parser.add_argument('--calibration', default='weights/score_calibration.json')
    parser.add_argument('--output', default='artifacts/submission.csv')
    parser.add_argument('--expected-days', type=int, default=344)
    args = parser.parse_args()
    print(json.dumps(export_submission(args.raw, args.features, args.calibration,
                                      args.output, args.expected_days), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
