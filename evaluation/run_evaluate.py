#!/usr/bin/env python3
"""Validate a submission's keys and then call the official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evaluation.evaluate import evaluate
except ModuleNotFoundError:  # Allow direct execution from a synchronized folder.
    from evaluate import evaluate


KEYS = ["ts_code", "trade_date"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="CSV with ts_code, trade_date, pred")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    submission_path = Path(args.submission)
    data_dir = Path(args.data_dir)
    prediction = pd.read_csv(submission_path)
    if list(prediction.columns) != [*KEYS, "pred"]:
        raise ValueError("submission columns must be exactly: ts_code,trade_date,pred")
    if prediction.duplicated(KEYS).any():
        raise ValueError("submission contains duplicate keys")
    if not np.isfinite(prediction["pred"].to_numpy(dtype=np.float64)).all():
        raise ValueError("submission pred contains NaN or Inf")

    expected = pd.read_csv(data_dir / "测试集_Y.csv", usecols=KEYS)
    if len(prediction) != len(expected):
        raise ValueError(f"row count differs: submission={len(prediction)}, expected={len(expected)}")
    merged = expected.merge(prediction[KEYS], on=KEYS, how="outer", indicator=True)
    mismatch = int((merged["_merge"] != "both").sum())
    if mismatch:
        raise ValueError(f"submission key set differs from evaluation data: mismatches={mismatch}")

    result = evaluate(str(submission_path), str(data_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
