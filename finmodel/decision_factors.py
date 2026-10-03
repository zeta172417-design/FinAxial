"""Read-only, label-free factor sidecar for the D0 decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .factors import FACTOR_PROFILES, FactorFeatureStore
from .factors_v4 import EXTRA64, ExtendedFactorFeatureStore


DECISION_FACTOR_NAMES = (
    "logret:5", "logret:20", "realized_vol:20", "downside_vol:30",
    "relative_volume:20", "rank:relative_volume:20",
    "range_position:20", "distance_to_high:60", "market_relative_logret:20",
    "rank:log_amount_level", "drawdown_to_peak:60", "market_beta:60",
    "flag_limit_up", "flag_limit_down",
)
assert len(DECISION_FACTOR_NAMES) == 14

MEDIUM_LONG_FACTOR_NAMES = (
    "logret:20", "logret:90", "logret:252",
    "ema_gap:60", "ema_gap:120",
    "realized_vol:60", "downside_vol:90", "relative_volume:90",
    "rank:logret:90", "rank:realized_vol:90",
    "distance_to_high:60", "drawdown_to_peak:120",
    "market_beta:120", "rank:avg_amount20",
    "flag_limit_up", "flag_limit_down",
)
DECISION_FACTOR_PROFILES = {
    "legacy14": DECISION_FACTOR_NAMES,
    "medium_long16": MEDIUM_LONG_FACTOR_NAMES,
}
assert len(MEDIUM_LONG_FACTOR_NAMES) == 16 and len(set(MEDIUM_LONG_FACTOR_NAMES)) == 16
assert all(name in FACTOR_PROFILES["f128"][0] for name in MEDIUM_LONG_FACTOR_NAMES[:10])
assert all(name in EXTRA64 for name in MEDIUM_LONG_FACTOR_NAMES[10:14])


@dataclass(frozen=True)
class DecisionFactorFeatureStore:
    panel: object
    base: np.ndarray
    extra: np.ndarray
    base_indices: tuple[int, ...]
    extra_indices: tuple[int, ...]
    names: tuple[str, ...] = DECISION_FACTOR_NAMES

    @classmethod
    def open(cls, panel, base_root: str | Path, extra_root: str | Path,
             *, expected_train_end: int,
             profile: str = "legacy14") -> "DecisionFactorFeatureStore":
        if profile not in DECISION_FACTOR_PROFILES:
            raise ValueError(f"unknown decision factor profile: {profile}")
        base = FactorFeatureStore.open(
            base_root, panel, "f128", expected_train_end=expected_train_end,
        ).features
        # Validates panel, base-cache identity, cutoff, and feature order.
        extended = ExtendedFactorFeatureStore.open(
            extra_root, base_root, panel, "f192_extended",
            expected_train_end=expected_train_end,
        )
        old_names = FACTOR_PROFILES["f128"][0]
        names = DECISION_FACTOR_PROFILES[profile]
        factor_names = names[:-2]
        base_names = tuple(name for name in factor_names if name in old_names)
        extra_names = tuple(name for name in factor_names if name in EXTRA64)
        if (*base_names, *extra_names) != factor_names:
            raise ValueError("decision factor names must list base then extended channels")
        base_indices = tuple(old_names.index(name) for name in base_names)
        extra_indices = tuple(EXTRA64.index(name) for name in extra_names)
        return cls(panel, base, extended.features.extra, base_indices, extra_indices, names)

    def rows(self, date_indices: np.ndarray) -> np.ndarray:
        dates = np.asarray(date_indices, dtype=np.int64)
        if dates.ndim != 1 or np.any(dates < 0) or np.any(dates >= self.panel.shape[0]):
            raise ValueError("decision factor date indices must lie in the panel")
        old = np.take(self.base[dates], self.base_indices, axis=-1).astype(np.float32)
        new = np.take(self.extra[dates], self.extra_indices, axis=-1).astype(np.float32)
        flags = np.asarray(self.panel.limit_flags[dates], dtype=np.float32)
        result = np.concatenate((old, new, flags), axis=-1)
        if result.shape != (len(dates), self.panel.shape[1], len(self.names)):
            raise ValueError("decision factor shape mismatch")
        if not np.isfinite(result).all():
            raise ValueError("non-finite decision factors")
        return result
