"""Versioned, causal factor extensions; the existing F128 cache stays immutable."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .factors import FACTOR_PROFILES, FactorFeatureStore, _percentile
from .io import atomic_json_dump, sha256_file


EXTENDED_FACTOR_VERSION = "finaxial-factors-v4-diverse-train-only-calibration"
BASE_NAMES = FACTOR_PROFILES["f128"][0]

EXTRA32 = (
    "overnight_logret:1", "intraday_logret:1", "overnight_mean:5", "overnight_mean:20",
    *(f"range_position:{h}" for h in (20, 60, 120, 252)),
    *(f"distance_to_high:{h}" for h in (20, 60, 120, 252)),
    *(f"trend_efficiency:{h}" for h in (5, 20, 60, 120)),
    *(f"trend_r2:{h}" for h in (20, 60, 120)),
    "trend_residual:20",
    *(f"return_volume_corr:{h}" for h in (10, 20, 60)),
    *(f"up_down_volume_imbalance:{h}" for h in (5, 20, 60)),
    *(f"volume_weighted_return:{h}" for h in (20, 60)),
    *(f"market_relative_logret:{h}" for h in (5, 20, 60, 120)),
)
EXTRA64 = EXTRA32 + (
    *(f"rank:{name}" for name in (
        "overnight_logret:1", "range_position:20", "range_position:60",
        "distance_to_high:20", "distance_to_high:60", "trend_efficiency:20",
        "trend_r2:60", "return_volume_corr:20", "up_down_volume_imbalance:20",
        "volume_weighted_return:20", "market_relative_logret:20",
        "market_relative_logret:60",
    )),
    "rank:log_amount_level", "rank:log_volume_level", "rank:amihud20",
    "rank:avg_amount20",
    *(f"drawdown_to_peak:{h}" for h in (20, 60, 120, 252)),
    *(f"down_up_vol_ratio:{h}" for h in (20, 60, 120, 252)),
    *(f"market_beta:{h}" for h in (20, 60, 120, 252)),
    *(f"residual_vol:{h}" for h in (20, 60, 120, 252)),
)
assert len(EXTRA32) == 32 and len(EXTRA64) == 64 and len(set(EXTRA64)) == 64


def _compact_names() -> tuple[str, ...]:
    """Keep separate short, medium and long scales; remove dense adjacent copies."""
    keep_horizons = {
        "ema_gap": {2, 5, 10, 20, 40, 60, 120, 252},
        "realized_vol": {5, 10, 20, 60, 120, 252},
        "relative_volume": {3, 5, 20, 60, 120},
        "relative_amount": {20, 60, 252},
        "mean_range": {10, 40, 60, 252},
    }
    selected = []
    for name in BASE_NAMES:
        if name == "candle:gap":  # Old definition is close/previous close, not overnight gap.
            continue
        kind, _, horizon = name.partition(":")
        if kind in keep_horizons and horizon.isdigit() and int(horizon) not in keep_horizons[kind]:
            continue
        selected.append(name)
    if len(selected) != 90:
        raise AssertionError(f"compact derived feature count is {len(selected)}, expected 90")
    return tuple(selected)


COMPACT_NAMES = _compact_names()

# Keep the E0 raw OHLCVA channels and 256-day causal sequence unchanged. Only
# the *additional* predictor factors are limited to observations from 1–20
# trading days, so this is an input-information ablation rather than a new
# architecture or label definition.
FAST_BASE_NAMES = (
    "candle:body", "candle:range", "candle:upper_wick",
    "candle:lower_wick", "candle:close_location",
    *(f"logret:{h}" for h in (1, 2, 3, 5, 7, 10, 14, 20)),
    *(f"ema_gap:{h}" for h in (2, 5, 10, 20)),
    *(f"realized_vol:{h}" for h in (5, 10, 20)),
    *(f"relative_volume:{h}" for h in (3, 5, 20)),
    *(f"relative_amount:{h}" for h in (5, 20)),
    *(f"rank:logret:{h}" for h in (1, 5, 10)),
    "rank:relative_volume:20",
)
FAST_EXTRA_NAMES = (
    "overnight_logret:1", "intraday_logret:1", "overnight_mean:5",
)
assert len(FAST_BASE_NAMES) == 29 and len(FAST_EXTRA_NAMES) == 3
assert len(set((*FAST_BASE_NAMES, *FAST_EXTRA_NAMES))) == 32
assert all(name in BASE_NAMES for name in FAST_BASE_NAMES)
assert all(name in EXTRA64 for name in FAST_EXTRA_NAMES)
EXTENDED_PROFILES = {
    "f96_compact": (COMPACT_NAMES, ()),
    "f128_refresh": (COMPACT_NAMES, EXTRA32),
    "f160_diverse": (BASE_NAMES, EXTRA32),
    "f192_extended": (BASE_NAMES, EXTRA64),
    "f32_fast": (FAST_BASE_NAMES, FAST_EXTRA_NAMES),
}


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    result = cumulative.copy()
    if window < len(result):
        result[window:] -= cumulative[:-window]
    result[:window - 1] = 0.0
    return result.astype(np.float32)


class _Calculator:
    def __init__(self, panel) -> None:
        raw = np.asarray(panel.features, dtype=np.float32)
        self.valid = np.asarray(panel.feature_valid, dtype=bool).copy()
        self.valid &= np.isfinite(raw[..., :4]).all(axis=-1) & (raw[..., 3] > 1e-8)
        self.raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.open, self.high, self.low, self.close, self.volume, self.amount = (
            self.raw[..., channel] for channel in range(6)
        )
        self.log_close = np.log(np.maximum(self.close, 1e-8))
        self.log_volume = np.log1p(np.maximum(self.volume, 0.0))
        self.log_amount = np.log1p(np.maximum(self.amount, 0.0))
        previous = np.zeros_like(self.log_close)
        previous[1:] = self.log_close[:-1]
        self.previous_valid = np.zeros_like(self.valid)
        self.previous_valid[1:] = self.valid[:-1]
        self.pair_valid = self.valid & self.previous_valid
        self.ret = np.where(self.pair_valid, self.log_close - previous, 0.0)
        self.ret = np.clip(self.ret, -0.5, 0.5).astype(np.float32)
        self.overnight = np.where(
            self.pair_valid & (self.open > 0),
            np.log(np.maximum(self.open, 1e-8)) - previous, 0.0,
        ).astype(np.float32)
        self.delta_volume = np.zeros_like(self.log_volume)
        self.delta_volume[1:] = self.log_volume[1:] - self.log_volume[:-1]
        self.delta_volume[~self.pair_valid] = 0.0
        counts = self.pair_valid.sum(axis=1).clip(min=1)
        self.market_ret = self.ret.sum(axis=1) / counts
        self.valid_cumulative = np.cumsum(self.valid, axis=0, dtype=np.int32)

    def available(self, window: int) -> np.ndarray:
        count = self.valid_cumulative.copy()
        if window < len(count):
            count[window:] -= self.valid_cumulative[:-window]
        count[:window - 1] = 0
        return (count >= window) & self.valid

    @lru_cache(maxsize=2)
    def extrema(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        minimum = pd.DataFrame(self.low).rolling(window, min_periods=window).min().to_numpy(dtype=np.float32)
        maximum = pd.DataFrame(self.high).rolling(window, min_periods=window).max().to_numpy(dtype=np.float32)
        return minimum, maximum

    @lru_cache(maxsize=2)
    def trend(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        y = self.log_close
        time = np.arange(len(y), dtype=np.float32)[:, None]
        sum_y = _rolling_sum(y, window)
        sum_yy = _rolling_sum(y * y, window)
        sum_xy = _rolling_sum(time * y, window)
        mean_x = time - (window - 1) / 2.0
        sxx = window * (window * window - 1) / 12.0
        slope = (sum_xy - mean_x * sum_y) / max(sxx, 1e-8)
        fitted = sum_y / window + slope * (time - mean_x)
        residual = y - fitted
        sst = np.maximum(sum_yy - sum_y * sum_y / window, 0.0)
        r2 = np.clip(slope * slope * sxx / np.maximum(sst, 1e-8), 0.0, 1.0)
        return r2.astype(np.float32), residual.astype(np.float32)

    @lru_cache(maxsize=2)
    def beta_residual_vol(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        market = np.broadcast_to(self.market_ret[:, None], self.ret.shape)
        x_mean = _rolling_sum(market, window) / window
        y_mean = _rolling_sum(self.ret, window) / window
        xx = np.maximum(_rolling_sum(market * market, window) / window - x_mean * x_mean, 0)
        yy = np.maximum(_rolling_sum(self.ret * self.ret, window) / window - y_mean * y_mean, 0)
        xy = _rolling_sum(market * self.ret, window) / window - x_mean * y_mean
        beta = xy / np.maximum(xx, 1e-8)
        residual = np.sqrt(np.maximum(yy - xy * xy / np.maximum(xx, 1e-8), 0.0))
        return beta.astype(np.float32), residual.astype(np.float32)

    def feature(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name.startswith("rank:"):
            underlying, available = self.feature(name.removeprefix("rank:"))
            ranked = np.zeros_like(underlying, dtype=np.float32)
            for date in range(len(ranked)):
                ranked[date] = _percentile(underlying[date], available[date])
            return ranked, available
        if name == "log_amount_level":
            return self.log_amount, self.valid & (self.amount > 0)
        if name == "log_volume_level":
            return self.log_volume, self.valid & (self.volume > 0)
        if name == "amihud20":
            impact = np.abs(self.ret) / np.maximum(self.amount, 1.0)
            return _rolling_sum(impact, 20) / 20, self.available(21)
        if name == "avg_amount20":
            return _rolling_sum(self.amount, 20) / 20, self.available(20)

        kind, horizon_text = name.split(":")
        horizon = int(horizon_text)
        available = self.available(max(horizon, 2 if kind.startswith("overnight") else 1))
        if kind == "overnight_logret":
            values = self.overnight * 10.0
            available &= self.pair_valid
        elif kind == "intraday_logret":
            values = np.log(np.maximum(self.close, 1e-8) / np.maximum(self.open, 1e-8)) * 10.0
        elif kind == "overnight_mean":
            values = _rolling_sum(self.overnight, horizon) * (10.0 / horizon)
            available &= self.available(horizon + 1)
        elif kind in {"range_position", "distance_to_high"}:
            low, high = self.extrema(horizon)
            if kind == "range_position":
                values = (self.close - low) / np.maximum(high - low, 1e-8) - 0.5
            else:
                values = np.log(np.maximum(self.close, 1e-8) / np.maximum(high, 1e-8)) * 10.0
        elif kind == "trend_efficiency":
            values = _rolling_sum(self.ret, horizon) / np.maximum(
                _rolling_sum(np.abs(self.ret), horizon), 1e-6,
            )
            available &= self.available(horizon + 1)
        elif kind in {"trend_r2", "trend_residual"}:
            r2, residual = self.trend(horizon)
            values = r2 if kind == "trend_r2" else residual * 10.0
        elif kind == "return_volume_corr":
            x, y = self.ret, self.delta_volume
            xm = _rolling_sum(x, horizon) / horizon
            ym = _rolling_sum(y, horizon) / horizon
            covariance = _rolling_sum(x * y, horizon) / horizon - xm * ym
            xvar = np.maximum(_rolling_sum(x * x, horizon) / horizon - xm * xm, 0)
            yvar = np.maximum(_rolling_sum(y * y, horizon) / horizon - ym * ym, 0)
            values = covariance / np.maximum(np.sqrt(xvar * yvar), 1e-8)
            available &= self.available(horizon + 1)
        elif kind == "up_down_volume_imbalance":
            signed = np.sign(self.ret) * self.volume
            values = _rolling_sum(signed, horizon) / np.maximum(
                _rolling_sum(self.volume, horizon), 1e-8,
            )
            available &= self.available(horizon + 1)
        elif kind == "volume_weighted_return":
            weighted = _rolling_sum(self.ret * self.volume, horizon) / np.maximum(
                _rolling_sum(self.volume, horizon), 1e-8,
            )
            values = (weighted - _rolling_sum(self.ret, horizon) / horizon) * 10.0
            available &= self.available(horizon + 1)
        elif kind == "market_relative_logret":
            relative = self.ret - self.market_ret[:, None]
            values = _rolling_sum(relative, horizon) * 10.0
            available &= self.available(horizon + 1)
        elif kind == "drawdown_to_peak":
            peak = pd.DataFrame(self.close).rolling(horizon, min_periods=horizon).max().to_numpy(dtype=np.float32)
            values = np.log(np.maximum(self.close, 1e-8) / np.maximum(peak, 1e-8)) * 10.0
        elif kind == "down_up_vol_ratio":
            down = _rolling_sum(np.minimum(self.ret, 0) ** 2, horizon)
            up = _rolling_sum(np.maximum(self.ret, 0) ** 2, horizon)
            values = np.sqrt(down / np.maximum(up, 1e-8))
            available &= self.available(horizon + 1)
        elif kind in {"market_beta", "residual_vol"}:
            beta, residual = self.beta_residual_vol(horizon)
            values = beta if kind == "market_beta" else residual * 20.0
            available &= self.available(horizon + 1)
        else:
            raise ValueError(f"unknown extended factor: {name}")
        available &= np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        values[~available] = 0.0
        return values, available


class CompositeFactorArray:
    """Read-only joined views, materialized only for the requested date slice."""

    ndim = 3
    dtype = np.dtype("float16")

    def __init__(self, base: np.ndarray, extra: np.ndarray,
                 base_indices: tuple[int, ...], extra_indices: tuple[int, ...]):
        self.base = base
        self.extra = extra
        self.base_indices = base_indices
        self.extra_indices = extra_indices
        self.shape = (*base.shape[:2], len(base_indices) + len(extra_indices))

    def __getitem__(self, index):
        parts = []
        if self.base_indices:
            parts.append(np.take(self.base[index], self.base_indices, axis=-1))
        if self.extra_indices:
            parts.append(np.take(self.extra[index], self.extra_indices, axis=-1))
        return np.concatenate(parts, axis=-1)


@dataclass(frozen=True)
class ExtendedFactorFeatureStore:
    root: Path
    features: CompositeFactorArray
    names: tuple[str, ...]
    profile: str
    manifest: dict

    @classmethod
    def open(cls, root: str | Path, base_root: str | Path, panel, profile: str,
             *, expected_train_end: int | None = None) -> "ExtendedFactorFeatureStore":
        if profile not in EXTENDED_PROFILES:
            raise ValueError(f"unknown extended factor profile: {profile}")
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != EXTENDED_FACTOR_VERSION:
            raise ValueError("extended factor cache version mismatch")
        if manifest.get("panel_manifest_sha256") != sha256_file(panel.root / "manifest.json"):
            raise ValueError("extended factor panel mismatch")
        if manifest.get("base_manifest_sha256") != sha256_file(Path(base_root) / "manifest.json"):
            raise ValueError("extended factor base cache mismatch")
        if expected_train_end is not None and manifest.get("train_end_date") != int(expected_train_end):
            raise ValueError("extended factor calibration cutoff mismatch")
        if tuple(manifest["feature_names"]) != EXTRA64:
            raise ValueError("extended factor order mismatch")
        base = FactorFeatureStore.open(
            base_root, panel, "f128", expected_train_end=expected_train_end,
        ).features
        extra = np.load(root / "extra64_derived.npy", mmap_mode="r")
        if extra.shape != (*panel.shape, 64) or extra.dtype != np.float16:
            raise ValueError("extended factor shape or dtype mismatch")
        base_names, extra_names = EXTENDED_PROFILES[profile]
        features = CompositeFactorArray(
            base, extra,
            tuple(BASE_NAMES.index(name) for name in base_names),
            tuple(EXTRA64.index(name) for name in extra_names),
        )
        return cls(root, features, (*base_names, *extra_names), profile, manifest)


def build_extended_factor_cache(panel, base_root: str | Path, root: str | Path,
                                *, train_end_date: int) -> Path:
    """Stream 64 extra channels without touching the existing F128 cache."""
    root = Path(root)
    if (root / "manifest.json").exists():
        ExtendedFactorFeatureStore.open(
            root, base_root, panel, "f192_extended",
            expected_train_end=train_end_date,
        )
        return root
    FactorFeatureStore.open(base_root, panel, "f128", expected_train_end=train_end_date)
    root.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(np.asarray(panel.dates) <= int(train_end_date))
    if not len(train) or int(panel.dates[train[-1]]) != int(train_end_date):
        raise ValueError("training cutoff must be an exact panel date")
    sample_dates = np.unique(train[np.linspace(0, len(train) - 1, min(256, len(train)), dtype=int)])
    sample_stocks = np.sort(np.random.default_rng(2026).choice(
        panel.shape[1], min(512, panel.shape[1]), replace=False,
    ))
    calculator = _Calculator(panel)
    building = root / "extra64_derived.building.npy"
    output = open_memmap(building, mode="w+", dtype=np.float16, shape=(*panel.shape, 64))
    calibration = {}
    for channel, name in enumerate(EXTRA64):
        values, available = calculator.feature(name)
        values = np.clip(values, -1e6, 1e6)
        if name.startswith("rank:"):
            center, scale = 0.0, 1.0
        else:
            sample = values[np.ix_(sample_dates, sample_stocks)]
            sample_mask = available[np.ix_(sample_dates, sample_stocks)]
            chosen = sample[sample_mask & np.isfinite(sample)]
            if len(chosen) < 32:
                raise ValueError(f"not enough train observations for {name}")
            center = float(np.median(chosen))
            q25, q75 = np.percentile(chosen, (25, 75))
            scale = max(float(q75 - q25) / 1.349,
                        float(np.percentile(np.abs(chosen - center), 99)) / 3.0)
            if scale < 1e-3:
                scale = 1.0
        calibrated = np.clip((values - center) / scale, -5.0, 5.0)
        calibrated[~available] = 0.0
        output[..., channel] = calibrated.astype(np.float16)
        calibration[name] = {"center": center, "scale": scale}
        print(f"FACTOR_V4 {channel + 1}/64 {name}", flush=True)
    output.flush()
    del output
    os.replace(building, root / "extra64_derived.npy")
    atomic_json_dump({
        "version": EXTENDED_FACTOR_VERSION,
        "panel_manifest_sha256": sha256_file(panel.root / "manifest.json"),
        "base_manifest_sha256": sha256_file(Path(base_root) / "manifest.json"),
        "train_end_date": int(train_end_date),
        "sample_dates": int(len(sample_dates)),
        "sample_stocks": int(len(sample_stocks)),
        "sample_seed": 2026,
        "feature_names": list(EXTRA64),
        "shape": [*panel.shape, 64],
        "dtype": "float16",
        "calibration": calibration,
        "normalization": "train-only robust center/scale, clip [-5,5]; rank stays [-1,1]; unavailable stays zero",
        "labels_used": False,
    }, root / "manifest.json")
    return root
