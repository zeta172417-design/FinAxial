from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .panel import Panel

FEATURE_MODES = ("temporal", "dual_normalization", "causal_returns", "factors")


def rolling_causal_zscore(
    raw: np.ndarray,
    valid: np.ndarray,
    *,
    lookback: int,
    epsilon: float = 1e-5,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each date using only its trailing window, never future rows.

    Args:
        raw: ``[dates, stocks, channels]`` values.
        valid: ``[dates, stocks]`` feature-valid flags.

    Returns:
        Current-date rolling z-scores with the same shape as ``raw`` and the
        number of valid observations in every date/stock trailing window.
    """
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if raw.ndim != 3 or valid.shape != raw.shape[:2]:
        raise ValueError("raw must be [dates, stocks, channels] with matching valid")
    if lookback <= 0 or epsilon <= 0 or clip <= 0:
        raise ValueError("lookback, epsilon and clip must be positive")

    masked = np.where(valid[..., None], raw, 0.0).astype(np.float64, copy=False)
    prefix = np.concatenate([
        np.zeros((1, raw.shape[1], raw.shape[2]), dtype=np.float64),
        np.cumsum(masked, axis=0, dtype=np.float64),
    ])
    prefix_sq = np.concatenate([
        np.zeros((1, raw.shape[1], raw.shape[2]), dtype=np.float64),
        np.cumsum(masked * masked, axis=0, dtype=np.float64),
    ])
    count_prefix = np.concatenate([
        np.zeros((1, raw.shape[1]), dtype=np.int32),
        np.cumsum(valid, axis=0, dtype=np.int32),
    ])
    ends = np.arange(1, raw.shape[0] + 1)
    starts = np.maximum(0, ends - int(lookback))
    sums = prefix[ends] - prefix[starts]
    sums_sq = prefix_sq[ends] - prefix_sq[starts]
    counts = count_prefix[ends] - count_prefix[starts]
    safe = np.maximum(counts, 1)[..., None]
    mean = sums / safe
    variance = np.maximum(sums_sq / safe - mean * mean, 0.0)
    normalized = (raw.astype(np.float64) - mean) / (np.sqrt(variance) + epsilon)
    normalized[~valid] = 0.0
    normalized = np.clip(normalized, -clip, clip).astype(np.float32)
    return normalized, counts


def cross_sectional_zscore(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    epsilon: float = 1e-5,
    clip: float = 5.0,
) -> np.ndarray:
    """Normalize each date/channel across currently valid stocks only."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("values must be [dates, stocks, channels] with matching valid")
    mask = valid[..., None]
    count = mask.sum(axis=1, keepdims=True).clip(min=1)
    masked = np.where(mask, values, 0.0).astype(np.float64, copy=False)
    mean = masked.sum(axis=1, keepdims=True) / count
    variance = np.maximum(
        (masked * masked).sum(axis=1, keepdims=True) / count - mean * mean,
        0.0,
    )
    normalized = (values.astype(np.float64) - mean) / (np.sqrt(variance) + epsilon)
    normalized[~mask.repeat(values.shape[2], axis=2)] = 0.0
    return np.clip(normalized, -clip, clip).astype(np.float32)


def causal_return_features(
    raw: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build six features using only the current row and previous filled close."""
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if raw.ndim != 3 or raw.shape[2] != 6 or valid.shape != raw.shape[:2]:
        raise ValueError("raw must be [dates, stocks, 6] with matching valid")
    derived = np.zeros_like(raw, dtype=np.float32)
    derived_valid = np.zeros_like(valid)
    if len(raw) < 2:
        return derived, derived_valid
    previous_close = raw[:-1, :, 3]
    current = raw[1:]
    usable = valid[1:] & np.isfinite(previous_close) & (np.abs(previous_close) > 1e-8)
    ratios = current[..., :4] / previous_close[..., None] - 1.0
    derived[1:, :, :4] = np.where(usable[..., None], ratios, 0.0)
    derived[1:, :, 4] = np.where(usable, np.log1p(np.maximum(current[..., 4], 0.0)), 0.0)
    derived[1:, :, 5] = np.where(usable, np.log1p(np.maximum(current[..., 5], 0.0)), 0.0)
    derived_valid[1:] = usable
    return derived, derived_valid


class MultiDateCrossSectionDataset(Dataset):
    """Causal rolling windows supervising several consecutive market dates.

    By default each item contains ``lookback + output_steps - 1`` date tokens.
    ``context_days`` can instead specify the exact number of unsupervised
    burn-in tokens before the ``output_steps`` labelled tokens.  Token ``j``
    predicts the return attached to the same trading date without observing
    later tokens.
    """

    def __init__(
        self,
        panel: Panel,
        date_indices: Sequence[int],
        *,
        lookback: int = 64,
        output_steps: int = 8,
        context_days: int | None = None,
        stride: int = 7,
        min_history: int = 32,
        epsilon: float = 1e-5,
        clip: float = 5.0,
        feature_mode: str = "temporal",
        long_memory_features: np.ndarray | None = None,
        factor_features: np.ndarray | None = None,
        inference_future_padding: int = 0,
        require_trainable: bool = True,
    ) -> None:
        self.panel = panel
        self.lookback = int(lookback)
        self.output_steps = int(output_steps)
        self.context_days = self.lookback - 1 if context_days is None else int(context_days)
        self.sequence_length = self.context_days + self.output_steps
        self.min_history = int(min_history)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.feature_mode = str(feature_mode)
        self.long_memory_features = long_memory_features
        self.factor_features = factor_features
        self.inference_future_padding = int(inference_future_padding)
        self.require_trainable = bool(require_trainable)
        if self.inference_future_padding < 0:
            raise ValueError("inference_future_padding must be non-negative")
        if self.inference_future_padding and self.require_trainable:
            raise ValueError("future padding is inference-only")
        if factor_features is not None and (
            factor_features.ndim != 3
            or factor_features.shape[:2] != panel.features.shape[:2]
        ):
            raise ValueError("factor features must be [panel dates, stocks, channels]")
        if long_memory_features is not None and (
            long_memory_features.ndim != 4
            or long_memory_features.shape[:2] != panel.features.shape[:2]
        ):
            raise ValueError("long memory must be [panel dates, stocks, scales, channels]")
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {FEATURE_MODES}, got {self.feature_mode!r}"
            )
        if self.feature_mode == "factors":
            if factor_features is None:
                raise ValueError("factors feature mode needs a factor cache")
            self.channels = 6 + int(factor_features.shape[2])
        else:
            if factor_features is not None:
                raise ValueError("factor cache supplied without factors feature mode")
            self.channels = 6 if self.feature_mode == "temporal" else 12
        requested = np.asarray(date_indices, dtype=np.int64)
        if self.lookback <= 0 or self.context_days < 0 or self.output_steps <= 1 or stride <= 0:
            raise ValueError(
                "lookback/stride must be positive, context_days non-negative, "
                "and output_steps must exceed one"
            )
        if requested.ndim != 1 or len(requested) < self.output_steps:
            raise ValueError("not enough requested dates for one multi-date sample")
        if not np.all(np.diff(requested) == 1):
            raise ValueError("requested dates must be consecutive panel indices")
        panel_dates = panel.features.shape[0]
        if requested[0] < 0 or requested[-1] > panel_dates - 1 + self.inference_future_padding:
            raise ValueError("requested dates exceed available data or allowed future padding")

        feature_valid = np.asarray(panel.feature_valid, dtype=bool)
        label_valid = np.asarray(panel.label_valid, dtype=bool)
        limit_up = np.asarray(panel.limit_flags[..., 0], dtype=bool)
        cumulative = np.cumsum(feature_valid, axis=0, dtype=np.int32)
        rolling_count = cumulative.copy()
        if self.lookback < len(rolling_count):
            rolling_count[self.lookback:] -= cumulative[:-self.lookback]
        full_window = rolling_count >= self.lookback
        eligible = feature_valid & full_window & (cumulative >= self.min_history)
        labelled = eligible & label_valid
        tradable = eligible & ~limit_up
        trainable = (
            (labelled.sum(axis=1) >= 2)
            & (tradable.sum(axis=1) >= 2)
            & ((labelled & tradable).sum(axis=1) >= 1)
        )

        last_start = len(requested) - self.output_steps
        relative_starts = list(range(0, last_start + 1, int(stride)))
        if relative_starts[-1] != last_start:
            relative_starts.append(last_start)
        blocks = []
        for relative in relative_starts:
            outputs = requested[relative:relative + self.output_steps]
            input_start = int(outputs[0]) - self.context_days
            real_outputs = outputs[outputs < panel_dates]
            if input_start < 0 or not len(real_outputs):
                continue
            if self.require_trainable and not bool(trainable[real_outputs].all()):
                continue
            blocks.append(outputs.copy())
        if not blocks:
            raise ValueError("multi-date dataset has no trainable blocks")
        self.output_blocks = np.stack(blocks)
        self.requested_date_indices = requested
        self.dropped_candidate_blocks = len(relative_starts) - len(blocks)

    def __len__(self) -> int:
        return len(self.output_blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        output_dates = self.output_blocks[index]
        sequence_end = int(output_dates[-1])
        real_end = min(sequence_end, self.panel.features.shape[0] - 1)
        future_count = sequence_end - real_end
        sequence_start = sequence_end - self.sequence_length + 1
        normalization_start = max(0, sequence_start - self.lookback + 1)
        feature_start = max(0, normalization_start - 1)
        full_raw = np.asarray(
            self.panel.features[feature_start:real_end + 1], dtype=np.float32,
        )
        full_valid = np.asarray(
            self.panel.feature_valid[feature_start:real_end + 1], dtype=bool,
        )
        feature_offset = normalization_start - feature_start
        raw = full_raw[feature_offset:]
        valid = full_valid[feature_offset:]
        normalized, counts = rolling_causal_zscore(
            raw, valid, lookback=self.lookback,
            epsilon=self.epsilon, clip=self.clip,
        )
        if self.feature_mode == "dual_normalization":
            cross_sectional = cross_sectional_zscore(
                normalized, valid, epsilon=self.epsilon, clip=self.clip,
            )
            normalized = np.concatenate([normalized, cross_sectional], axis=2)
        elif self.feature_mode == "causal_returns":
            derived, derived_valid = causal_return_features(full_raw, full_valid)
            derived = derived[feature_offset:]
            derived_valid = derived_valid[feature_offset:]
            derived_normalized, _ = rolling_causal_zscore(
                derived, derived_valid, lookback=self.lookback,
                epsilon=self.epsilon, clip=self.clip,
            )
            normalized = np.concatenate([normalized, derived_normalized], axis=2)
        offset = sequence_start - normalization_start
        values = normalized[offset:].transpose(1, 0, 2).copy()
        token_valid = valid[offset:].T.copy()
        if future_count:
            values = np.pad(values, ((0, 0), (0, future_count), (0, 0)))
            token_valid = np.pad(token_valid, ((0, 0), (0, future_count)))
        if self.factor_features is not None:
            derived = np.asarray(
                self.factor_features[sequence_start:real_end + 1], dtype=np.float32,
            ).transpose(1, 0, 2)
            if future_count:
                derived = np.pad(derived, ((0, 0), (0, future_count), (0, 0)))
            values = np.concatenate((values, derived), axis=2)
        real_output = output_dates <= real_end
        stock_count = self.panel.features.shape[1]
        eligible = np.zeros((self.output_steps, stock_count), dtype=bool)
        targets = np.zeros((self.output_steps, stock_count), dtype=np.float32)
        label_valid = np.zeros_like(eligible)
        limit_up = np.zeros_like(eligible)
        output_offset = output_dates[real_output] - normalization_start
        output_valid = valid[output_offset]
        output_counts = counts[output_offset]
        eligible[real_output] = (
            output_valid
            & (output_counts >= self.lookback)
            & (output_counts >= self.min_history)
        )
        targets[real_output] = np.asarray(
            self.panel.labels[output_dates[real_output]], dtype=np.float32,
        )
        label_valid[real_output] = np.asarray(
            self.panel.label_valid[output_dates[real_output]], dtype=bool,
        )
        limit_up[real_output] = np.asarray(
            self.panel.limit_flags[output_dates[real_output], :, 0], dtype=bool,
        )
        item = {
            "x": torch.from_numpy(values),
            "token_valid": torch.from_numpy(token_valid),
            "target": torch.from_numpy(targets),
            "mask": torch.from_numpy(eligible & label_valid),
            "eligible": torch.from_numpy(eligible),
            "tradable": torch.from_numpy(eligible & ~limit_up),
            "date_indices": torch.from_numpy(output_dates.copy()),
        }
        if self.long_memory_features is not None:
            long_values = np.asarray(
                self.long_memory_features[sequence_start:real_end + 1],
                dtype=np.float32,
            ).transpose(1, 0, 2, 3).copy()
            if future_count:
                long_values = np.pad(
                    long_values, ((0, 0), (0, future_count), (0, 0), (0, 0)),
                )
            item["long_memory"] = torch.from_numpy(long_values)
        return item
