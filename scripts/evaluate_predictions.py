#!/usr/bin/env python3
"""Score predictions only when the user independently supplies evaluation labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from finmodel.metrics import evaluate_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', default='artifacts/evaluation.json')
    args = parser.parse_args()
    keys = ['ts_code', 'trade_date']
    pred = pd.read_csv(args.predictions, usecols=keys + ['pred'])
    features = pd.read_csv(args.features, usecols=keys + ['flag_limit_up'])
    labels = pd.read_csv(args.labels, usecols=keys + ['y_ret_1d'])
    for name, data in (('predictions', pred), ('features', features), ('labels', labels)):
        if data[keys].isna().any().any() or data[keys].duplicated().any():
            raise ValueError(f'invalid keys: {name}')
    # A final unlabelled feature day may have no corresponding label row.
    frame = pred.merge(labels, on=keys, validate='one_to_one').merge(features, on=keys, validate='one_to_one')
    if len(frame) != len(labels):
        raise ValueError('predictions/features do not cover every supplied label key')
    metrics = evaluate_frame(frame, 'pred')
    from finmodel.io import atomic_json_dump
    atomic_json_dump(metrics, args.output)
    print(json.dumps({name: metrics[name] for name in
                     ('final_score', 'ic_mean', 'annual_excess', 'mean_turnover')}, indent=2))


if __name__ == '__main__':
    main()
