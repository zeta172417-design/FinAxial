from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .panel import Panel


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
    ) -> None:
        self.panel = panel
        self.lookback = int(lookback)
        self.output_steps = int(output_steps)
        self.context_days = self.lookback - 1 if context_days is None else int(context_days)
        self.sequence_length = self.context_days + self.output_steps
        self.min_history = int(min_history)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
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
            if input_start < 0 or not bool(trainable[outputs].all()):
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
        sequence_start = sequence_end - self.sequence_length + 1
        normalization_start = max(0, sequence_start - self.lookback + 1)
        raw = np.asarray(
            self.panel.features[normalization_start:sequence_end + 1], dtype=np.float32,
        )
        valid = np.asarray(
            self.panel.feature_valid[normalization_start:sequence_end + 1], dtype=bool,
        )
        normalized, counts = rolling_causal_zscore(
            raw, valid, lookback=self.lookback,
            epsilon=self.epsilon, clip=self.clip,
        )
        offset = sequence_start - normalization_start
        values = normalized[offset:].transpose(1, 0, 2).copy()
        token_valid = valid[offset:].T.copy()
        output_offset = output_dates - normalization_start
        output_valid = valid[output_offset]
        output_counts = counts[output_offset]
        eligible = (
            output_valid
            & (output_counts >= self.lookback)
            & (output_counts >= self.min_history)
        )
        targets = np.asarray(self.panel.labels[output_dates], dtype=np.float32).copy()
        label_valid = np.asarray(self.panel.label_valid[output_dates], dtype=bool)
        limit_up = np.asarray(self.panel.limit_flags[output_dates, :, 0], dtype=bool)
        return {
            "x": torch.from_numpy(values),
            "token_valid": torch.from_numpy(token_valid),
            "target": torch.from_numpy(targets),
            "mask": torch.from_numpy(eligible & label_valid),
            "eligible": torch.from_numpy(eligible),
            "tradable": torch.from_numpy(eligible & ~limit_up),
            "date_indices": torch.from_numpy(output_dates.copy()),
        }
