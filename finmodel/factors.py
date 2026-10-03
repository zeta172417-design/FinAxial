"""Label-free, causal factor channels for the predictor ablations.

The first six model channels remain the existing rolling-z-scored OHLCVA.
This module supplies only the additional channels. F32/F64/F128 are nested
prefixes of one F128 cache; F32-short uses a separate <=60-day definition.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib.format import open_memmap

from .io import atomic_json_dump, sha256_file


FACTOR_VERSION = "finaxial-factors-v3-train-only-robust-calibration"
RAW_CHANNELS = ("open", "high", "low", "close", "vol", "amount")


def _specifications(short: bool = False) -> tuple[str, ...]:
    candle = tuple(f"candle:{name}" for name in (
        "body", "gap", "range", "upper_wick", "lower_wick", "close_location",
    ))
    base = candle
    base += tuple(f"logret:{h}" for h in (1, 5, 20, 60 if short else 120))
    base += tuple(f"ema_gap:{h}" for h in ((5, 10, 20, 60) if short else (5, 20, 60, 120)))
    base += tuple(f"realized_vol:{h}" for h in (5, 20, 60 if short else 120))
    base += tuple(f"relative_volume:{h}" for h in (5, 20, 60 if short else 120))
    rank_h = 60 if short else 120
    base += tuple(f"rank:{name}" for name in (
        "logret:1", "logret:5", f"logret:{rank_h}",
        "realized_vol:20", "relative_volume:20", "candle:close_location",
    ))
    assert len(base) == 26
    if short:
        return base
    extra64 = tuple(f"{kind}:{h}" for kind, horizons in (
        ("logret", (2, 10, 40, 252)),
        ("ema_gap", (2, 10, 40, 252)),
        ("mean_range", (10, 40, 60, 252)),
        ("realized_vol", (10, 40, 60, 252)),
        ("relative_amount", (5, 20, 60, 252)),
        ("up_fraction", (5, 20, 60, 252)),
    ) for h in horizons)
    extra64 += tuple(f"rank:{name}" for name in (
        "logret:2", "logret:10", "logret:40", "logret:252",
        "realized_vol:60", "realized_vol:252",
        "relative_amount:20", "up_fraction:20",
    ))
    assert len(base + extra64) == 58
    horizons = (3, 7, 14, 30, 90, 180)
    extra128 = tuple(f"{kind}:{h}" for kind in (
        "logret", "ema_gap", "realized_vol", "relative_volume",
        "relative_amount", "mean_range", "up_fraction", "downside_vol",
    ) for h in horizons)
    extra128 += tuple(f"rank:{kind}:{h}" for kind in (
        "logret", "realized_vol", "relative_volume", "downside_vol",
    ) for h in (14, 90, 180))
    extra128 += tuple(f"market:{name}" for name in (
        "equal_weight_return", "positive_breadth", "return_dispersion", "median_relative_volume",
    ))
    specs = base + extra64 + extra128
    assert len(specs) == 122 and len(set(specs)) == 122
    return specs


FACTOR_PROFILES = {
    "f32": (_specifications()[:26], "f32_derived.npy"),
    "f64": (_specifications()[:58], "f64_derived.npy"),
    "f128": (_specifications(), "f128_derived.npy"),
    "f32_short": (_specifications(short=True), "f32_short_derived.npy"),
}


def _percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.float32)
    chosen = np.flatnonzero(valid & np.isfinite(values))
    if len(chosen) < 2:
        return output
    ordered = np.argsort(values[chosen], kind="stable")
    sorted_values = values[chosen][ordered]
    _, group, count = np.unique(sorted_values, return_inverse=True, return_counts=True)
    beginnings = np.cumsum(count) - count
    mean_position = beginnings[group] + 0.5 * (count[group] - 1)
    output[chosen[ordered]] = 2.0 * mean_position / (len(chosen) - 1) - 1.0
    return output


def iter_factor_rows(
    features: np.ndarray, feature_valid: np.ndarray, *, short: bool = False,
    centers: np.ndarray | None = None, scales: np.ndarray | None = None,
    clip: float | None = 5.0, return_available: bool = False,
) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Yield factor rows using data through that date only.

    Calibration is applied only to valid continuous channels. Rank-percentile
    channels retain their centered [-1, 1] definition. Unavailable histories
    remain exact zero even when a continuous channel is median-centered.
    """
    if features.ndim != 3 or features.shape[-1] != 6 or feature_valid.shape != features.shape[:2]:
        raise ValueError("features and feature_valid must be [dates, stocks, 6]/[dates, stocks]")
    specs = _specifications(short)
    if (centers is None) != (scales is None):
        raise ValueError("centers and scales must be supplied together")
    if centers is not None and (
        np.shape(centers) != (len(specs),) or np.shape(scales) != (len(specs),)
        or not np.isfinite(centers).all() or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("invalid factor calibration arrays")
    if clip is not None and clip <= 0:
        raise ValueError("clip must be positive")
    stocks = features.shape[1]
    horizons = sorted({int(s.rsplit(":", 1)[-1]) for s in specs
                       if s.rsplit(":", 1)[-1].isdigit()})
    max_lag = max(horizons)
    log_ring = np.zeros((max_lag + 1, stocks), dtype=np.float32)
    ema = {h: {name: np.zeros(stocks, dtype=np.float32) for name in (
        "log_close", "log_volume", "log_amount", "return", "return_sq",
        "down_sq", "range", "up",
    )} for h in horizons}
    age = np.zeros(stocks, dtype=np.int32)
    previous_log_close = np.zeros(stocks, dtype=np.float32)
    for date in range(features.shape[0]):
        raw = np.asarray(features[date], dtype=np.float32)
        valid = np.asarray(feature_valid[date], dtype=bool).copy()
        valid &= np.isfinite(raw[:, :4]).all(axis=1) & (raw[:, 3] > 1e-8)
        safe = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        opn, high, low, close, volume, amount = safe.T
        log_close = np.log(np.maximum(close, 1e-8))
        log_volume = np.log1p(np.maximum(volume, 0.0))
        log_amount = np.log1p(np.maximum(amount, 0.0))
        previous_valid = age > 0
        daily_return = np.where(valid & previous_valid,
                                log_close - previous_log_close, 0.0).astype(np.float32)
        daily_return = np.clip(daily_return, -0.5, 0.5)
        day_range = np.where(valid, (high - low) / np.maximum(close, 1e-8), 0.0)
        for horizon in horizons:
            state = ema[horizon]
            alpha = 2.0 / (horizon + 1.0)
            for name, observation in (
                ("log_close", log_close), ("log_volume", log_volume),
                ("log_amount", log_amount), ("return", daily_return),
                ("return_sq", daily_return * daily_return),
                ("down_sq", np.minimum(daily_return, 0.0) ** 2),
                ("range", day_range), ("up", (daily_return > 0).astype(np.float32)),
            ):
                current = state[name]
                # First valid observation seeds nonzero-level series; earlier
                # pre-listing rows never enter an EMA.
                current[valid & ~previous_valid] = observation[valid & ~previous_valid]
                update = valid & previous_valid
                current[update] += alpha * (observation[update] - current[update])
        age += valid.astype(np.int32)
        log_ring[date % len(log_ring)] = np.where(valid, log_close, 0.0)
        row: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        candle = {
            "body": (close - opn) / np.maximum(opn, 1e-8),
            "gap": np.expm1(log_close - previous_log_close),
            "range": day_range,
            "upper_wick": (high - np.maximum(opn, close)) / np.maximum(close, 1e-8),
            "lower_wick": (np.minimum(opn, close) - low) / np.maximum(close, 1e-8),
            "close_location": (close - low) / np.maximum(high - low, 1e-8) - 0.5,
        }
        for name, value in candle.items():
            key = f"candle:{name}"
            row[key] = np.asarray(value * (1.0 if name == "close_location" else 10.0), dtype=np.float32)
            available[key] = valid & (previous_valid if name == "gap" else True)
        for spec in specs:
            if spec in row or spec.startswith(("rank:", "market:")):
                continue
            kind, horizon_text = spec.split(":")
            horizon = int(horizon_text)
            state = ema[horizon]
            if kind == "logret":
                lagged = log_ring[(date - horizon) % len(log_ring)]
                result = (log_close - lagged) * 10.0
            elif kind == "ema_gap":
                result = (log_close - state["log_close"]) * 10.0
            elif kind == "realized_vol":
                result = np.sqrt(np.maximum(state["return_sq"] - state["return"] ** 2, 0.0)) * 20.0
            elif kind == "downside_vol":
                result = np.sqrt(np.maximum(state["down_sq"], 0.0)) * 20.0
            elif kind == "relative_volume":
                result = log_volume - state["log_volume"]
            elif kind == "relative_amount":
                result = log_amount - state["log_amount"]
            elif kind == "mean_range":
                result = state["range"] * 10.0
            elif kind == "up_fraction":
                result = 2.0 * state["up"] - 1.0
            else:
                raise AssertionError(spec)
            row[spec] = np.asarray(result, dtype=np.float32)
            available[spec] = valid & (age >= horizon + (1 if kind == "logret" else 0))
        market_valid = valid & (age >= 2)
        if market_valid.any():
            market_return = float(np.mean(daily_return[market_valid]))
            breadth = float(2 * np.mean(daily_return[market_valid] > 0) - 1)
            dispersion = float(np.std(daily_return[market_valid]))
            relvol20 = log_volume - ema[20]["log_volume"]
            median_volume = float(np.median(relvol20[market_valid]))
        else:
            market_return = breadth = dispersion = median_volume = 0.0
        market = {
            "market:equal_weight_return": market_return * 10.0,
            "market:positive_breadth": breadth,
            "market:return_dispersion": dispersion * 20.0,
            "market:median_relative_volume": median_volume,
        }
        output = np.zeros((stocks, len(specs)), dtype=np.float32)
        output_available = np.zeros((stocks, len(specs)), dtype=bool)
        for channel, spec in enumerate(specs):
            if spec.startswith("rank:"):
                base_name = spec.removeprefix("rank:")
                values = _percentile(row[base_name], available[base_name])
                channel_available = available[base_name]
            elif spec.startswith("market:"):
                values = np.full(stocks, market[spec], dtype=np.float32)
                channel_available = market_valid
            else:
                values = row[spec]
                channel_available = available[spec]
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = np.clip(values, -1e6, 1e6)
            if centers is not None and not spec.startswith("rank:"):
                values = (values - centers[channel]) / scales[channel]
            if clip is not None:
                values = np.clip(values, -clip, clip)
            output[:, channel] = np.where(channel_available, values, 0.0)
            output_available[:, channel] = channel_available
        output[~valid] = 0.0
        output_available[~valid] = False
        yield (output, output_available) if return_available else output
        previous_log_close = np.where(valid, log_close, previous_log_close)


def fit_factor_calibration(
    panel, *, train_end_date: int, short: bool = False,
    sample_dates: int = 256, sample_stocks: int = 512,
) -> dict:
    """Fit robust per-channel centers/scales from features on training dates only."""
    if not hasattr(panel, "dates"):
        raise ValueError("panel dates are required for train-only calibration")
    dates = np.asarray(panel.dates, dtype=np.int64)
    train_indices = np.flatnonzero(dates <= int(train_end_date))
    if not len(train_indices) or dates[train_indices[-1]] != int(train_end_date):
        raise ValueError("train_end_date must be an exact panel trading date")
    if sample_dates <= 0 or sample_stocks <= 0:
        raise ValueError("sample_dates and sample_stocks must be positive")
    chosen_dates = np.unique(train_indices[
        np.linspace(0, len(train_indices) - 1, min(sample_dates, len(train_indices)), dtype=int)
    ])
    chosen_stocks = np.sort(np.random.default_rng(2026).choice(
        panel.shape[1], size=min(sample_stocks, panel.shape[1]), replace=False,
    ))
    rows, masks = [], []
    wanted = set(chosen_dates.tolist())
    for date, (row, available) in enumerate(iter_factor_rows(
        panel.features, panel.feature_valid, short=short,
        clip=None, return_available=True,
    )):
        if date > chosen_dates[-1]:
            break
        if date in wanted:
            rows.append(row[chosen_stocks])
            masks.append(available[chosen_stocks])
    samples = np.concatenate(rows, axis=0)
    available_samples = np.concatenate(masks, axis=0)
    names = _specifications(short)
    centers = np.zeros(len(names), dtype=np.float32)
    scales = np.ones(len(names), dtype=np.float32)
    counts = np.zeros(len(names), dtype=np.int64)
    for channel, name in enumerate(names):
        if name.startswith("rank:"):
            continue
        values = samples[available_samples[:, channel], channel]
        values = values[np.isfinite(values)]
        counts[channel] = len(values)
        if len(values) < 32:
            raise ValueError(f"not enough training observations for {name}")
        center = float(np.median(values))
        q25, q75 = np.percentile(values, (25, 75))
        tail_scale = float(np.percentile(np.abs(values - center), 99)) / 3.0
        robust_scale = max(float(q75 - q25) / 1.349, tail_scale)
        centers[channel] = center
        scales[channel] = robust_scale if robust_scale >= 1e-3 else 1.0
    return {
        "feature_names": list(names),
        "train_end_date": int(train_end_date),
        "last_sampled_date": int(dates[chosen_dates[-1]]),
        "sample_dates": int(len(chosen_dates)),
        "sample_stocks": int(len(chosen_stocks)),
        "sample_seed": 2026,
        "method": "median center; max(IQR/1.349, P99_abs_deviation/3, 1e-3) scale; rank unchanged; missing exact zero",
        "centers": centers.tolist(),
        "scales": scales.tolist(),
        "non_rank_observation_counts": counts.tolist(),
    }


def _panel_fingerprint(panel) -> str:
    return sha256_file(panel.root / "manifest.json")


def build_factor_cache(panel, root: str | Path, *, train_end_date: int) -> Path:
    """Build train-calibrated caches once before torchrun; manifest is completion marker."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        for profile in FACTOR_PROFILES:
            FactorFeatureStore.open(root, panel, profile, expected_train_end=train_end_date)
        return root
    root.mkdir(parents=True, exist_ok=True)
    calibration = {
        "f128": fit_factor_calibration(panel, train_end_date=train_end_date),
        "f32_short": fit_factor_calibration(panel, train_end_date=train_end_date, short=True),
    }
    generated = (("f128_derived.npy", False), ("f32_short_derived.npy", True))
    for filename, short in generated:
        building = root / f"{filename}.building.npy"
        count = 26 if short else 122
        array = open_memmap(building, mode="w+", dtype=np.float16,
                            shape=(*panel.shape, count))
        settings = calibration["f32_short" if short else "f128"]
        for date, row in enumerate(iter_factor_rows(
            panel.features, panel.feature_valid, short=short,
            centers=np.asarray(settings["centers"], dtype=np.float32),
            scales=np.asarray(settings["scales"], dtype=np.float32),
        )):
            array[date] = row.astype(np.float16)
        array.flush()
        del array
        os.replace(building, root / filename)
    full = np.load(root / "f128_derived.npy", mmap_mode="r")
    for filename, count in (("f32_derived.npy", 26), ("f64_derived.npy", 58)):
        building = root / f"{filename}.building.npy"
        narrow = open_memmap(building, mode="w+", dtype=np.float16,
                             shape=(*panel.shape, count))
        for start in range(0, panel.shape[0], 32):
            stop = min(start + 32, panel.shape[0])
            narrow[start:stop] = full[start:stop, :, :count]
        narrow.flush()
        del narrow
        os.replace(building, root / filename)
    del full
    files = (("f32_derived.npy", 26, False), ("f64_derived.npy", 58, False),
             ("f128_derived.npy", 122, False), ("f32_short_derived.npy", 26, True))
    atomic_json_dump({
        "version": FACTOR_VERSION,
        "train_end_date": int(train_end_date),
        "calibration": calibration,
        "panel_manifest_sha256": _panel_fingerprint(panel),
        "panel_source_sha256": panel.manifest.get("source", {}).get("sha256"),
        "panel_shape": list(panel.shape),
        "date_range": (
            [int(panel.dates[0]), int(panel.dates[-1])]
            if hasattr(panel, "dates") else None
        ),
        "dtype": "float16",
        "files": {name: {"shape": [*panel.shape, count],
                         "feature_names": list(_specifications(short)[:count])}
                  for name, count, short in files},
        "normalization": "raw six: existing trailing causal z-score; derived continuous: train-only robust center/scale then clip [-5,5]; rank: unchanged [-1,1]",
        "history": "EWM plus lagged close; maximum horizon 252 (F32 short maximum 60)",
        "labels_used": False,
    }, manifest_path)
    return root


@dataclass(frozen=True)
class FactorFeatureStore:
    root: Path
    features: np.ndarray
    names: tuple[str, ...]
    profile: str
    manifest: dict

    @classmethod
    def open(
        cls, root: str | Path, panel, profile: str,
        *, expected_train_end: int | None = None,
    ) -> "FactorFeatureStore":
        if profile not in FACTOR_PROFILES:
            raise ValueError(f"unknown factor profile: {profile}")
        root = Path(root)
        with (root / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version") != FACTOR_VERSION or manifest.get("panel_manifest_sha256") != _panel_fingerprint(panel):
            raise ValueError("factor cache version or panel fingerprint mismatch")
        if expected_train_end is not None and manifest.get("train_end_date") != int(expected_train_end):
            raise ValueError("factor cache training cutoff does not match configuration")
        names, filename = FACTOR_PROFILES[profile]
        file_info = manifest["files"][filename]
        if tuple(file_info["feature_names"][:len(names)]) != names:
            raise ValueError("factor cache feature order mismatch")
        array = np.load(root / filename, mmap_mode="r")
        if tuple(array.shape) != tuple(file_info["shape"]) or array.dtype != np.float16:
            raise ValueError("factor cache shape or dtype mismatch")
        return cls(root, array[..., :len(names)], names, profile, manifest)
