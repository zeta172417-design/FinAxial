from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel


def temporal_features(date_values: np.ndarray) -> np.ndarray:
    return date_stamps(date_values)


def date_stamps(date_values: np.ndarray) -> np.ndarray:
    import pandas as pd
    index = pd.to_datetime(np.asarray(date_values).astype(str), format="%Y%m%d")
    return np.column_stack([
        np.zeros(len(index)), np.zeros(len(index)), index.weekday, index.day, index.month,
    ]).astype(np.float32)


@dataclass(frozen=True)
class Window:
    values: np.ndarray
    stamps: np.ndarray
    valid: np.ndarray
    eligible: bool


def normalized_window(
    panel: Panel, date_idx: int, code_idx: int, lookback: int, min_history: int = 32,
    history_count: int | None = None,
) -> Window:
    if date_idx < 0 or date_idx >= panel.shape[0]:
        raise IndexError(date_idx)
    start = max(0, date_idx - lookback + 1)
    raw = np.asarray(panel.features[start:date_idx + 1, code_idx], dtype=np.float32)
    valid = np.asarray(panel.feature_valid[start:date_idx + 1, code_idx], dtype=bool)
    out = np.zeros((lookback, 6), dtype=np.float32)
    out_valid = np.zeros(lookback, dtype=bool)
    pad = lookback - len(raw)
    if valid.any():
        selected = raw[valid]
        mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = selected.std(axis=0, dtype=np.float64).astype(np.float32)
        normalized = (raw - mean) / (std + 1e-5)
        normalized[~valid] = 0.0
        out[pad:] = np.clip(normalized, -5.0, 5.0)
    out_valid[pad:] = valid
    stamps = np.zeros((lookback, 5), dtype=np.float32)
    stamps[pad:] = date_stamps(panel.dates[start:date_idx + 1])
    total_history = history_count
    if total_history is None:
        total_history = int(np.asarray(panel.feature_valid[:date_idx + 1, code_idx], dtype=np.int32).sum())
    return Window(out, stamps, out_valid, total_history >= min_history and bool(valid[-1]))


def normalized_cross_section(
    panel: Panel, date_idx: int, lookback: int, min_history: int = 32,
    history_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [stocks, lookback, 6] data and a stock eligibility mask."""
    start = max(0, date_idx - lookback + 1)
    raw = np.asarray(panel.features[start:date_idx + 1], dtype=np.float32).transpose(1, 0, 2)
    valid = np.asarray(panel.feature_valid[start:date_idx + 1], dtype=bool).T
    out = np.zeros((panel.shape[1], lookback, 6), dtype=np.float32)
    pad = lookback - raw.shape[1]
    count = valid.sum(axis=1)
    safe_count = np.maximum(count, 1)[:, None]
    finite_raw = np.where(valid[..., None], raw, 0.0)
    mean = finite_raw.sum(axis=1) / safe_count
    centered = np.where(valid[..., None], raw - mean[:, None, :], 0.0)
    var = (centered * centered).sum(axis=1) / safe_count
    normalized = centered / (np.sqrt(var)[:, None, :] + 1e-5)
    normalized[~valid] = 0.0
    out[:, pad:] = np.clip(normalized, -5.0, 5.0)
    total_history = history_counts
    if total_history is None:
        total_history = np.asarray(panel.feature_valid[:date_idx + 1], dtype=np.int32).sum(axis=0)
    full_window = raw.shape[1] == lookback and valid.all(axis=1)
    eligible = (total_history >= min_history) & full_window & valid[:, -1]
    return out, eligible
