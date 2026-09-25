"""Causal, label-free observed returns for the decision policy."""

from __future__ import annotations

import numpy as np
import torch

from .sequence import rolling_causal_zscore
from .long_memory import LongMemoryFeatureStore


def observed_return_features(panel, date_indices: np.ndarray, *, scale: float = 0.02) -> np.ndarray:
    """Return realized from yesterday's close to today's known close.

    This is available when predicting the return labelled at today's date.
    It deliberately never reads ``panel.labels`` or any later feature row.
    """
    dates = np.asarray(date_indices, dtype=np.int64)
    if dates.ndim != 1 or not len(dates) or (dates <= 0).any():
        raise ValueError("date indices must be one-dimensional and have a previous date")
    if scale <= 0:
        raise ValueError("return scale must be positive")
    current = np.asarray(panel.features[dates, :, 3], dtype=np.float32)
    previous = np.asarray(panel.features[dates - 1, :, 3], dtype=np.float32)
    valid = (
        np.asarray(panel.feature_valid[dates], dtype=bool)
        & np.asarray(panel.feature_valid[dates - 1], dtype=bool)
        & np.isfinite(current) & np.isfinite(previous)
        & (previous > 1e-8)
    )
    result = np.zeros(current.shape, dtype=np.float32)
    np.divide(current, previous, out=result, where=valid)
    result[valid] -= 1.0
    result[~valid] = 0.0
    # Bound erroneous split adjustments without altering the rank signal.
    return np.clip(result / float(scale), -10.0, 10.0).astype(np.float32)


def observed_market_features(panel, date_indices: np.ndarray, *, scale: float = 0.02) -> np.ndarray:
    """Known-at-close price/volume features, without labels or future rows.

    Channels are close-to-close return, open-to-close return, intraday range,
    and log-volume change. Invalid observations are zeroed channel-wise.
    """
    dates = np.asarray(date_indices, dtype=np.int64)
    if dates.ndim != 1 or not len(dates) or (dates <= 0).any():
        raise ValueError("date indices must be one-dimensional and have a previous date")
    if scale <= 0:
        raise ValueError("return scale must be positive")
    current = np.asarray(panel.features[dates], dtype=np.float32)
    previous = np.asarray(panel.features[dates - 1], dtype=np.float32)
    valid_now = np.asarray(panel.feature_valid[dates], dtype=bool)
    valid_previous = np.asarray(panel.feature_valid[dates - 1], dtype=bool)
    result = np.zeros((*current.shape[:2], 4), dtype=np.float32)
    result[..., 0] = observed_return_features(panel, dates, scale=scale)
    open_price, high, low, close, volume = (
        current[..., channel] for channel in (0, 1, 2, 3, 4)
    )
    price_valid = valid_now & np.isfinite(open_price) & (open_price > 1e-8)
    np.divide(close - open_price, open_price, out=result[..., 1], where=price_valid)
    np.divide(high - low, open_price, out=result[..., 2], where=price_valid)
    result[..., 1:3] = np.clip(result[..., 1:3] / float(scale), -10.0, 10.0)
    old_volume = previous[..., 4]
    volume_valid = (
        valid_now & valid_previous & np.isfinite(volume) & np.isfinite(old_volume)
        & (volume > 0) & (old_volume > 0)
    )
    result[..., 3] = np.where(
        volume_valid,
        np.log1p(np.maximum(volume, 0)) - np.log1p(np.maximum(old_volume, 0)),
        0.0,
    )
    result[..., 3] = np.clip(result[..., 3], -5.0, 5.0)
    result[~valid_now] = 0.0
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def causal_c0_bridge_batch(panel, date_index: int, config: dict) -> tuple[torch.Tensor, ...]:
    """Construct one C0 window from X only; never inspect label values or masks."""
    model, data = config["model"], config["data"]
    if data.get("feature_mode", "temporal") != "temporal":
        raise ValueError("history bridge currently supports the temporal C0 feature mode")
    lookback = int(model["lookback"])
    output_steps = int(model["output_steps"])
    context_days = int(model["context_days"])
    sequence_length = context_days + output_steps
    start = int(date_index) - sequence_length + 1
    if start < 0:
        raise ValueError("not enough dates for the C0 bridge window")
    normalization_start = max(0, start - lookback + 1)
    raw = np.asarray(panel.features[normalization_start:date_index + 1], dtype=np.float32)
    valid = np.asarray(panel.feature_valid[normalization_start:date_index + 1], dtype=bool)
    normalized, counts = rolling_causal_zscore(
        raw, valid, lookback=lookback,
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
    )
    offset = start - normalization_start
    output_positions = np.arange(date_index - output_steps + 1, date_index + 1) - normalization_start
    eligible = (
        valid[output_positions]
        & (counts[output_positions] >= lookback)
        & (counts[output_positions] >= int(config["min_history"]))
    )
    result = (
        torch.from_numpy(normalized[offset:].transpose(1, 0, 2).copy()),
        torch.from_numpy(valid[offset:].T.copy()),
        torch.from_numpy(eligible.copy()),
    )
    if model.get("long_memory_scales"):
        store = LongMemoryFeatureStore.open(
            data["long_memory_cache"], panel, model["long_memory_scales"],
        )
        long_values = np.asarray(
            store.features[start:date_index + 1], dtype=np.float32,
        ).transpose(1, 0, 2, 3).copy()
        return (*result, torch.from_numpy(long_values))
    return result
