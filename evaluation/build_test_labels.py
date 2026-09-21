#!/usr/bin/env python3
"""Reconstruct next-day returns from the following row's close price."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["ts_code", "trade_date"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/测试集_X.csv")
    parser.add_argument("--output-dir", default="evaluation")
    parser.add_argument("--train-source", default="data/raw/训练集.csv")
    parser.add_argument("--verify-rows", type=int, default=20_000)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    required = {*KEYS, "close", "flag_limit_up"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing test columns: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError("test keys are not unique")

    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    next_close = frame.groupby("ts_code", sort=False)["close"].shift(-1)
    reconstructed = next_close.div(frame["close"]).sub(1.0)
    last_date = int(frame["trade_date"].max())
    keep = frame["trade_date"] < last_date
    test_x = frame.loc[keep].copy()
    test_y = frame.loc[keep, KEYS].copy()
    test_y["y_ret_1d"] = reconstructed.loc[keep].to_numpy()

    expected_rows = int(test_x["ts_code"].nunique() * test_x["trade_date"].nunique())
    if len(test_x) != expected_rows or len(test_y) != expected_rows:
        raise ValueError("trimmed test data is not a complete rectangular panel")
    if not test_x[KEYS].equals(test_y[KEYS]):
        raise ValueError("X/Y key order differs")

    train = pd.read_csv(
        args.train_source,
        nrows=args.verify_rows,
        usecols=["ts_code", "trade_date", "close", "y_ret_1d"],
    )
    expected = train.groupby("ts_code", sort=False)["close"].shift(-1).div(train["close"]).sub(1.0)
    comparable = train["y_ret_1d"].notna() & expected.notna()
    difference = (train.loc[comparable, "y_ret_1d"] - expected.loc[comparable]).abs()
    max_difference = float(difference.max())
    if not np.allclose(
        train.loc[comparable, "y_ret_1d"], expected.loc[comparable], rtol=0, atol=1e-12
    ):
        raise ValueError(f"training labels do not match the reconstruction formula: {max_difference}")

    x_path = output / "测试集_X.csv"
    y_path = output / "测试集_Y.csv"
    atomic_csv(test_x, x_path)
    atomic_csv(test_y, y_path)
    manifest = {
        "version": 1,
        "source": str(source),
        "source_sha256": sha256(source),
        "formula": "y_ret_1d(t) = close(t+1) / close(t) - 1",
        "excluded_date": last_date,
        "date_min": int(test_x["trade_date"].min()),
        "date_max": int(test_x["trade_date"].max()),
        "dates": int(test_x["trade_date"].nunique()),
        "stocks": int(test_x["ts_code"].nunique()),
        "rows": int(len(test_x)),
        "label_missing": int(test_y["y_ret_1d"].isna().sum()),
        "train_formula_check_rows": int(comparable.sum()),
        "train_formula_max_abs_diff": max_difference,
        "x_sha256": sha256(x_path),
        "y_sha256": sha256(y_path),
    }
    temporary = output / "label_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "label_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
