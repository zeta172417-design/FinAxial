from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .io import atomic_json_dump, sha256_file

FEATURE_COLUMNS = ("open", "high", "low", "close", "vol", "amount")
KEY_COLUMNS = ("ts_code", "trade_date")
PROCESSING_VERSION = "panel-v1-causal-prev-close"
PHASE1_PROCESSING_VERSION = "phase1-panel-v1-train-plus-reconstructed-test"


@dataclass(frozen=True)
class Panel:
    root: Path
    features: np.ndarray
    labels: np.ndarray
    feature_valid: np.ndarray
    label_valid: np.ndarray
    limit_flags: np.ndarray
    dates: np.ndarray
    codes: np.ndarray
    manifest: dict

    @classmethod
    def open(cls, root: str | Path, mode: str = "r") -> "Panel":
        root = Path(root)
        with (root / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        return cls(
            root=root,
            features=np.load(root / "features.npy", mmap_mode=mode),
            labels=np.load(root / "labels.npy", mmap_mode=mode),
            feature_valid=np.load(root / "feature_valid.npy", mmap_mode=mode),
            label_valid=np.load(root / "label_valid.npy", mmap_mode=mode),
            limit_flags=np.load(root / "limit_flags.npy", mmap_mode=mode),
            dates=np.load(root / "dates.npy", mmap_mode=mode),
            codes=np.load(root / "codes.npy", mmap_mode=mode),
            manifest=manifest,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.features.shape[0]), int(self.features.shape[1])


def _chunks(path: Path, columns: list[str], chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=columns, chunksize=chunksize)


def _discover_axes(path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray, int]:
    codes: set[str] = set()
    dates: set[int] = set()
    rows = 0
    for chunk in _chunks(path, list(KEY_COLUMNS), chunksize):
        if chunk[list(KEY_COLUMNS)].isna().any().any():
            raise ValueError("primary keys contain missing values")
        codes.update(chunk["ts_code"].astype(str).unique())
        dates.update(chunk["trade_date"].astype(np.int32).unique().tolist())
        rows += len(chunk)
    return np.asarray(sorted(dates), dtype=np.int32), np.asarray(sorted(codes), dtype="U16"), rows


def _causal_fill(features: np.ndarray, feature_valid: np.ndarray) -> dict[str, int]:
    """Fill missing post-listing rows using only the previous finite close."""
    n_dates, n_codes, _ = features.shape
    filled = 0
    prelisting = 0
    for code_idx in range(n_codes):
        previous_close = np.nan
        for date_idx in range(n_dates):
            row = features[date_idx, code_idx]
            raw_close = float(row[3])
            prices_finite = bool(np.isfinite(row[:4]).all())
            if prices_finite:
                feature_valid[date_idx, code_idx] = True
                previous_close = raw_close
            elif np.isfinite(previous_close):
                row[:4] = previous_close
                feature_valid[date_idx, code_idx] = True
                filled += 1
            else:
                prelisting += 1
                continue
            if not np.isfinite(row[4]):
                row[4] = 0.0
            if not np.isfinite(row[5]):
                row[5] = 0.0
    return {"causally_filled_rows": filled, "prelisting_invalid_rows": prelisting}


def build_panel(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    chunksize: int = 250_000,
    overwrite: bool = False,
) -> Path:
    """Convert the immutable training CSV into mmap-friendly NPY arrays.

    A manifest is written last and acts as the completion marker. Existing output is
    reused only when its source hash and processing version match.
    """
    source_csv = Path(source_csv).resolve()
    output_dir = Path(output_dir).resolve()
    source_hash = sha256_file(source_csv)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        with manifest_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("source", {}).get("sha256") == source_hash and old.get("processing_version") == PROCESSING_VERSION:
            return output_dir
        raise FileExistsError(f"stale panel exists at {output_dir}; pass --overwrite explicitly")

    output_dir.mkdir(parents=True, exist_ok=True)
    dates, codes, row_count = _discover_axes(source_csv, chunksize)
    expected_rows = len(dates) * len(codes)
    if row_count != expected_rows:
        raise ValueError(f"input is not a rectangular panel: rows={row_count}, dates*codes={expected_rows}")

    shape = (len(dates), len(codes))
    features = open_memmap(output_dir / "features.npy", mode="w+", dtype=np.float32, shape=shape + (6,))
    labels = open_memmap(output_dir / "labels.npy", mode="w+", dtype=np.float32, shape=shape)
    feature_valid = open_memmap(output_dir / "feature_valid.npy", mode="w+", dtype=np.bool_, shape=shape)
    label_valid = open_memmap(output_dir / "label_valid.npy", mode="w+", dtype=np.bool_, shape=shape)
    limit_flags = open_memmap(output_dir / "limit_flags.npy", mode="w+", dtype=np.bool_, shape=shape + (2,))
    features[:] = np.nan
    labels[:] = np.nan
    feature_valid[:] = False
    label_valid[:] = False
    limit_flags[:] = False

    date_index = pd.Index(dates)
    code_index = pd.Index(codes)
    columns = list(KEY_COLUMNS + FEATURE_COLUMNS) + ["flag_limit_up", "flag_limit_down", "y_ret_1d"]
    seen = np.zeros(shape, dtype=np.bool_)
    for chunk in _chunks(source_csv, columns, chunksize):
        date_ids = date_index.get_indexer(chunk["trade_date"].to_numpy(dtype=np.int32))
        code_ids = code_index.get_indexer(chunk["ts_code"].astype(str))
        if (date_ids < 0).any() or (code_ids < 0).any():
            raise RuntimeError("axis discovery and population disagree")
        if seen[date_ids, code_ids].any():
            raise ValueError("duplicate (trade_date, ts_code) key")
        seen[date_ids, code_ids] = True
        values = chunk[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        features[date_ids, code_ids] = values
        target = chunk["y_ret_1d"].to_numpy(dtype=np.float32)
        labels[date_ids, code_ids] = target
        label_valid[date_ids, code_ids] = np.isfinite(target)
        limit_flags[date_ids, code_ids, 0] = chunk["flag_limit_up"].fillna(0).to_numpy(dtype=np.int8) != 0
        limit_flags[date_ids, code_ids, 1] = chunk["flag_limit_down"].fillna(0).to_numpy(dtype=np.int8) != 0
    if not seen.all():
        raise ValueError(f"panel is missing {int((~seen).sum())} keys")

    fill_stats = _causal_fill(features, feature_valid)
    for array in (features, labels, feature_valid, label_valid, limit_flags):
        array.flush()
    np.save(output_dir / "dates.npy", dates, allow_pickle=False)
    np.save(output_dir / "codes.npy", codes, allow_pickle=False)
    manifest = {
        "processing_version": PROCESSING_VERSION,
        "source": {"path": str(source_csv), "sha256": source_hash, "rows": row_count},
        "keys": list(KEY_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": "y_ret_1d",
        "limit_columns": ["flag_limit_up", "flag_limit_down"],
        "shape": {"dates": len(dates), "codes": len(codes), "features": [*shape, 6]},
        "date_range": [int(dates[0]), int(dates[-1])],
        "dtypes": {
            "features": "float32", "labels": "float32", "feature_valid": "bool",
            "label_valid": "bool", "limit_flags": "bool",
        },
        "rules": {
            "missing_ohlc": "previous finite close only; pre-listing remains invalid",
            "missing_volume_amount": "zero after first valid OHLC row",
            "raw_csv_modified": False,
        },
        **fill_stats,
    }
    atomic_json_dump(manifest, manifest_path)
    return output_dir


def build_phase1_panel(
    train_panel_dir: str | Path,
    test_features_csv: str | Path,
    test_labels_csv: str | Path,
    output_dir: str | Path,
    *,
    chunksize: int = 250_000,
    overwrite: bool = False,
) -> Path:
    """Append the reconstructed evaluation period to the immutable train panel.

    The result is a continuous read-only-by-convention panel.  In particular, a
    2025 window can use the end of 2024 as history.  The original panel and CSVs
    are never modified.  A manifest is written last and is the completion marker.
    """
    train_panel_dir = Path(train_panel_dir).resolve()
    test_features_csv = Path(test_features_csv).resolve()
    test_labels_csv = Path(test_labels_csv).resolve()
    output_dir = Path(output_dir).resolve()
    sources = {
        "train_manifest_sha256": sha256_file(train_panel_dir / "manifest.json"),
        "test_features_sha256": sha256_file(test_features_csv),
        "test_labels_sha256": sha256_file(test_labels_csv),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        with manifest_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("processing_version") == PHASE1_PROCESSING_VERSION and old.get("source_hashes") == sources:
            return output_dir
        raise FileExistsError(f"stale phase-1 panel exists at {output_dir}; pass --overwrite explicitly")

    train = Panel.open(train_panel_dir)
    test_dates, test_codes, feature_rows = _discover_axes(test_features_csv, chunksize)
    if not np.array_equal(test_codes, np.asarray(train.codes)):
        raise ValueError("train and evaluation code axes differ")
    if len(test_dates) == 0 or int(test_dates[0]) <= int(train.dates[-1]):
        raise ValueError("evaluation dates must start strictly after the train panel")
    expected_test_rows = len(test_dates) * len(test_codes)
    if feature_rows != expected_test_rows:
        raise ValueError("evaluation features are not a rectangular panel")

    label_dates, label_codes, label_rows = _discover_axes(test_labels_csv, chunksize)
    if not np.array_equal(label_dates, test_dates) or not np.array_equal(label_codes, test_codes):
        raise ValueError("evaluation X/Y axes differ")
    if label_rows != expected_test_rows:
        raise ValueError("evaluation labels are not a rectangular panel")

    output_dir.mkdir(parents=True, exist_ok=True)
    dates = np.concatenate([np.asarray(train.dates, dtype=np.int32), test_dates])
    codes = np.asarray(train.codes).copy()
    train_count = len(train.dates)
    shape = (len(dates), len(codes))
    features = open_memmap(output_dir / "features.npy", mode="w+", dtype=np.float32, shape=shape + (6,))
    labels = open_memmap(output_dir / "labels.npy", mode="w+", dtype=np.float32, shape=shape)
    feature_valid = open_memmap(output_dir / "feature_valid.npy", mode="w+", dtype=np.bool_, shape=shape)
    label_valid = open_memmap(output_dir / "label_valid.npy", mode="w+", dtype=np.bool_, shape=shape)
    limit_flags = open_memmap(output_dir / "limit_flags.npy", mode="w+", dtype=np.bool_, shape=shape + (2,))
    features[:] = np.nan
    labels[:] = np.nan
    feature_valid[:] = False
    label_valid[:] = False
    limit_flags[:] = False
    features[:train_count] = train.features
    labels[:train_count] = train.labels
    feature_valid[:train_count] = train.feature_valid
    label_valid[:train_count] = train.label_valid
    limit_flags[:train_count] = train.limit_flags

    date_index = pd.Index(test_dates)
    code_index = pd.Index(codes)
    seen_x = np.zeros((len(test_dates), len(codes)), dtype=np.bool_)
    x_columns = list(KEY_COLUMNS + FEATURE_COLUMNS) + ["flag_limit_up", "flag_limit_down"]
    for chunk in _chunks(test_features_csv, x_columns, chunksize):
        local_dates = date_index.get_indexer(chunk["trade_date"].to_numpy(dtype=np.int32))
        code_ids = code_index.get_indexer(chunk["ts_code"].astype(str))
        if (local_dates < 0).any() or (code_ids < 0).any() or seen_x[local_dates, code_ids].any():
            raise ValueError("invalid or duplicate key in evaluation features")
        seen_x[local_dates, code_ids] = True
        date_ids = local_dates + train_count
        features[date_ids, code_ids] = chunk[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        limit_flags[date_ids, code_ids, 0] = chunk["flag_limit_up"].fillna(0).to_numpy(dtype=np.int8) != 0
        limit_flags[date_ids, code_ids, 1] = chunk["flag_limit_down"].fillna(0).to_numpy(dtype=np.int8) != 0
    if not seen_x.all():
        raise ValueError(f"evaluation features miss {int((~seen_x).sum())} keys")

    seen_y = np.zeros_like(seen_x)
    for chunk in _chunks(test_labels_csv, list(KEY_COLUMNS) + ["y_ret_1d"], chunksize):
        local_dates = date_index.get_indexer(chunk["trade_date"].to_numpy(dtype=np.int32))
        code_ids = code_index.get_indexer(chunk["ts_code"].astype(str))
        if (local_dates < 0).any() or (code_ids < 0).any() or seen_y[local_dates, code_ids].any():
            raise ValueError("invalid or duplicate key in evaluation labels")
        seen_y[local_dates, code_ids] = True
        date_ids = local_dates + train_count
        target = chunk["y_ret_1d"].to_numpy(dtype=np.float32)
        labels[date_ids, code_ids] = target
        label_valid[date_ids, code_ids] = np.isfinite(target)
    if not seen_y.all():
        raise ValueError(f"evaluation labels miss {int((~seen_y).sum())} keys")

    fill_stats = _causal_fill(features, feature_valid)
    for array in (features, labels, feature_valid, label_valid, limit_flags):
        array.flush()
    np.save(output_dir / "dates.npy", dates, allow_pickle=False)
    np.save(output_dir / "codes.npy", codes, allow_pickle=False)
    manifest = {
        "processing_version": PHASE1_PROCESSING_VERSION,
        "source_hashes": sources,
        "sources": {
            "train_panel": str(train_panel_dir),
            "test_features": str(test_features_csv),
            "test_labels": str(test_labels_csv),
        },
        "keys": list(KEY_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": "y_ret_1d",
        "limit_columns": ["flag_limit_up", "flag_limit_down"],
        "shape": {"dates": len(dates), "codes": len(codes), "features": [*shape, 6]},
        "date_range": [int(dates[0]), int(dates[-1])],
        "segments": {
            "train": [int(train.dates[0]), int(train.dates[-1])],
            "reconstructed_test": [int(test_dates[0]), int(test_dates[-1])],
        },
        "rules": {
            "missing_ohlc": "previous finite close only across the train/test boundary",
            "missing_volume_amount": "zero after first valid OHLC row",
            "source_files_modified": False,
            "test_labels": "post-hoc reconstruction; final date without next close excluded upstream",
        },
        **fill_stats,
    }
    atomic_json_dump(manifest, manifest_path)
    return output_dir
