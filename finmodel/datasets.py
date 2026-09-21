from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .panel import Panel
from .windows import normalized_cross_section, normalized_window


class SeriesWindowDataset(Dataset):
    """Single-stock windows; a sample can never cross a stock boundary."""

    def __init__(
        self,
        panel: Panel,
        date_indices: Sequence[int],
        *,
        lookback: int = 256,
        min_history: int = 32,
        require_label: bool = False,
    ) -> None:
        self.panel = panel
        self.lookback = lookback
        self.min_history = min_history
        self.indices: list[tuple[int, int]] = []
        cumulative_history = np.cumsum(np.asarray(panel.feature_valid, dtype=np.int16), axis=0)
        for date_idx in date_indices:
            histories = cumulative_history[date_idx]
            valid = np.asarray(panel.feature_valid[date_idx]) & (histories >= min_history)
            if require_label:
                valid &= np.asarray(panel.label_valid[date_idx])
            self.indices.extend((int(date_idx), int(code_idx)) for code_idx in np.flatnonzero(valid))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        date_idx, code_idx = self.indices[index]
        window = normalized_window(
            self.panel, date_idx, code_idx, self.lookback, self.min_history,
            history_count=self.min_history,
        )
        return {
            "x": torch.from_numpy(window.values),
            "stamp": torch.from_numpy(window.stamps),
            "valid": torch.from_numpy(window.valid),
            "target": torch.tensor(float(self.panel.labels[date_idx, code_idx]), dtype=torch.float32),
            "date_idx": torch.tensor(date_idx),
            "code_idx": torch.tensor(code_idx),
        }


class CrossSectionDataset(Dataset):
    """One item is one trading date, preserving the true cross-section."""

    def __init__(self, panel: Panel, date_indices: Sequence[int], *, lookback: int, min_history: int = 32) -> None:
        self.panel = panel
        self.date_indices = np.asarray(date_indices, dtype=np.int64)
        self.lookback = lookback
        self.min_history = min_history
        self.cumulative_history = np.cumsum(np.asarray(panel.feature_valid, dtype=np.int16), axis=0)

    def __len__(self) -> int:
        return len(self.date_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        date_idx = int(self.date_indices[index])
        values, eligible = normalized_cross_section(
            self.panel, date_idx, self.lookback, self.min_history,
            history_counts=self.cumulative_history[date_idx],
        )
        label_valid = np.asarray(self.panel.label_valid[date_idx], dtype=bool)
        mask = eligible & label_valid
        return {
            "x": torch.from_numpy(values),
            "target": torch.from_numpy(np.asarray(self.panel.labels[date_idx], dtype=np.float32).copy()),
            "mask": torch.from_numpy(mask),
            "eligible": torch.from_numpy(eligible),
            "date_idx": torch.tensor(date_idx),
        }


class ConsecutiveCrossSectionDataset(Dataset):
    """Adjacent market-wide dates for differentiable turnover objectives."""

    def __init__(self, panel: Panel, date_indices: Sequence[int], *, lookback: int, min_history: int = 32) -> None:
        self.base = CrossSectionDataset(
            panel, date_indices, lookback=lookback, min_history=min_history,
        )
        if len(self.base.date_indices) < 2:
            raise ValueError("consecutive cross-section training requires at least two dates")
        if not np.all(np.diff(self.base.date_indices) == 1):
            raise ValueError("date indices must be consecutive trading dates")
        self.panel = panel
        self.date_indices = self.base.date_indices[1:]
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.base) - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        previous = self.base[index]
        current = self.base[index + 1]
        result: dict[str, torch.Tensor] = {}
        for prefix, item in (("previous", previous), ("current", current)):
            date_idx = int(item["date_idx"])
            not_limit_up = ~torch.from_numpy(
                np.asarray(self.panel.limit_flags[date_idx, :, 0], dtype=bool).copy()
            )
            result.update({f"{prefix}_{key}": value for key, value in item.items()})
            result[f"{prefix}_tradable"] = item["eligible"] & not_limit_up
        return result


class BalancedReplaySampler(Sampler[int]):
    """Draw 50% new-year samples and 50% historical replay, deterministically."""

    def __init__(self, dataset: SeriesWindowDataset, new_year: int, num_samples: int, seed: int) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        dates = np.asarray(dataset.panel.dates)
        new, history = [], []
        for sample_idx, (date_idx, _) in enumerate(dataset.indices):
            (new if int(dates[date_idx]) // 10000 == new_year else history).append(sample_idx)
        if not new or not history:
            raise ValueError("balanced replay requires both new-year and historical samples")
        self.new = np.asarray(new)
        self.history = np.asarray(history)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for index in range(self.num_samples):
            pool = self.new if index % 2 == 0 else self.history
            yield int(pool[rng.integers(len(pool))])


class SameDateBatchSampler(Sampler[list[int]]):
    """Random batches whose members always share one trading date."""

    def __init__(
        self, dataset: SeriesWindowDataset, batch_size: int, num_batches: int,
        seed: int = 2026, new_year: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        grouped: dict[int, list[int]] = {}
        for sample_idx, (date_idx, _) in enumerate(dataset.indices):
            grouped.setdefault(date_idx, []).append(sample_idx)
        self.groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values() if indices]
        if not self.groups:
            raise ValueError("dataset has no date groups")
        self.new_groups = []
        self.history_groups = []
        if new_year is not None:
            for date_idx, indices in grouped.items():
                target = self.new_groups if int(dataset.panel.dates[date_idx]) // 10000 == new_year else self.history_groups
                target.append(np.asarray(indices, dtype=np.int64))
            if not self.new_groups or not self.history_groups:
                raise ValueError("same-date balanced replay needs new and historical dates")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for batch_index in range(self.num_batches):
            if self.new_groups:
                pool = self.new_groups if batch_index % 2 == 0 else self.history_groups
            else:
                pool = self.groups
            group = pool[int(rng.integers(len(pool)))]
            replace = len(group) < self.batch_size
            yield rng.choice(group, self.batch_size, replace=replace).tolist()
