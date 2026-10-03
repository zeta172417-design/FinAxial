"""Label-free causal multi-scale market features for the C0 memory branch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from .io import atomic_json_dump, sha256_file


MEMORY_VERSION = "finaxial-causal-ewm-memory-v1"
FEATURE_NAMES = (
    "ewm_return_times_horizon", "ewm_realized_volatility",
    "log_close_vs_ewm", "log_volume_vs_ewm", "log_amount_vs_ewm",
    "ewm_valid_coverage",
)


def iter_causal_multiscale_features(
    features: np.ndarray, feature_valid: np.ndarray, scales: Sequence[int],
) -> Iterator[np.ndarray]:
    """Yield [stocks, scales, 6] using only rows up to the yielded date."""
    if len(features.shape) != 3 or features.shape[-1] != 6:
        raise ValueError("features must be [dates, stocks, 6]")
    if feature_valid.shape != features.shape[:2]:
        raise ValueError("feature_valid shape does not match features")
    horizons = np.asarray(scales, dtype=np.float32)
    if not len(horizons) or np.any(horizons < 2) or len(set(scales)) != len(scales):
        raise ValueError("memory scales must be distinct and at least two dates")
    stocks = features.shape[1]
    alpha = (2.0 / (horizons + 1.0))[:, None]
    shape = (len(horizons), stocks)
    mean_return = np.zeros(shape, dtype=np.float32)
    mean_return_sq = np.zeros(shape, dtype=np.float32)
    mean_log_close = np.zeros(shape, dtype=np.float32)
    mean_log_volume = np.zeros(shape, dtype=np.float32)
    mean_log_amount = np.zeros(shape, dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.float32)
    initialized = np.zeros(stocks, dtype=bool)
    previous_close = np.zeros(stocks, dtype=np.float32)
    for date in range(features.shape[0]):
        row = np.asarray(features[date], dtype=np.float32)
        valid = np.asarray(feature_valid[date], dtype=bool)
        close, volume, amount = row[:, 3], row[:, 4], row[:, 5]
        valid = valid & np.isfinite(close) & (close > 1e-8)
        log_close = np.log(np.maximum(np.nan_to_num(close), 1e-8))
        log_volume = np.log1p(np.maximum(np.nan_to_num(volume), 0.0))
        log_amount = np.log1p(np.maximum(np.nan_to_num(amount), 0.0))
        first = valid & ~initialized
        mean_log_close[:, first] = log_close[first]
        mean_log_volume[:, first] = log_volume[first]
        mean_log_amount[:, first] = log_amount[first]
        price_pair = valid & (previous_close > 1e-8)
        daily_return = np.where(
            price_pair, log_close - np.log(np.maximum(previous_close, 1e-8)), 0.0,
        ).astype(np.float32)
        daily_return = np.clip(daily_return, -0.5, 0.5)
        active = valid[None, :]
        mean_return = np.where(
            active, (1.0 - alpha) * mean_return + alpha * daily_return, mean_return,
        )
        mean_return_sq = np.where(
            active, (1.0 - alpha) * mean_return_sq + alpha * daily_return ** 2,
            mean_return_sq,
        )
        mean_log_close = np.where(
            active, (1.0 - alpha) * mean_log_close + alpha * log_close,
            mean_log_close,
        )
        mean_log_volume = np.where(
            active, (1.0 - alpha) * mean_log_volume + alpha * log_volume,
            mean_log_volume,
        )
        mean_log_amount = np.where(
            active, (1.0 - alpha) * mean_log_amount + alpha * log_amount,
            mean_log_amount,
        )
        coverage = (1.0 - alpha) * coverage + alpha * active.astype(np.float32)
        variance = np.maximum(mean_return_sq - mean_return ** 2, 0.0)
        output = np.stack((
            np.clip(mean_return * horizons[:, None], -2.0, 2.0),
            np.clip(np.sqrt(variance) * np.sqrt(horizons[:, None]), 0.0, 2.0),
            np.clip(log_close[None, :] - mean_log_close, -2.0, 2.0),
            np.clip(log_volume[None, :] - mean_log_volume, -5.0, 5.0),
            np.clip(log_amount[None, :] - mean_log_amount, -5.0, 5.0),
            coverage,
        ), axis=-1).transpose(1, 0, 2)
        output[~valid, :, :5] = 0.0
        yield np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        initialized |= valid
        previous_close = np.where(np.isfinite(close) & (close > 0), close, previous_close)


@dataclass(frozen=True)
class LongMemoryFeatureStore:
    root: Path
    features: np.ndarray
    scales: tuple[int, ...]
    manifest: dict

    @classmethod
    def open(cls, root: str | Path, panel, scales: Sequence[int]) -> "LongMemoryFeatureStore":
        root = Path(root)
        with (root / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_hash = sha256_file(panel.root / "manifest.json")
        if manifest.get("version") != MEMORY_VERSION:
            raise ValueError("unsupported long-memory feature version")
        if manifest.get("panel_manifest_sha256") != expected_hash:
            raise ValueError("long-memory panel hash mismatch")
        if tuple(manifest.get("scales", ())) != tuple(int(x) for x in scales):
            raise ValueError("long-memory scales do not match model")
        values = np.load(root / "features.npy", mmap_mode="r")
        expected_shape = (panel.shape[0], panel.shape[1], len(scales), len(FEATURE_NAMES))
        if tuple(values.shape) != expected_shape or tuple(manifest.get("shape", ())) != expected_shape:
            raise ValueError("long-memory feature shape mismatch")
        return cls(root, values, tuple(int(x) for x in scales), manifest)


def build_long_memory_store(root: str | Path, panel, scales: Sequence[int]) -> LongMemoryFeatureStore:
    root = Path(root)
    if (root / "manifest.json").exists():
        return LongMemoryFeatureStore.open(root, panel, scales)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"incomplete long-memory cache: {root}")
    root.mkdir(parents=True, exist_ok=True)
    scales = tuple(int(x) for x in scales)
    shape = (panel.shape[0], panel.shape[1], len(scales), len(FEATURE_NAMES))
    values = open_memmap(root / "features.npy", mode="w+", dtype=np.float16, shape=shape)
    for date, feature in enumerate(iter_causal_multiscale_features(
        panel.features, panel.feature_valid, scales,
    )):
        values[date] = feature.astype(np.float16)
        if (date + 1) % 128 == 0:
            values.flush()
    values.flush()
    atomic_json_dump({
        "version": MEMORY_VERSION,
        "panel_manifest_sha256": sha256_file(panel.root / "manifest.json"),
        "shape": list(shape),
        "dtype": "float16",
        "scales": list(scales),
        "features": list(FEATURE_NAMES),
        "causal_rule": "each date reads only current and previous panel.features rows",
        "uses_labels": False,
    }, root / "manifest.json")
    return LongMemoryFeatureStore.open(root, panel, scales)
