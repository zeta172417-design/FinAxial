"""Read-only, date-indexed frozen C0 outputs for sequential decision learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import sha256_file


CACHE_VERSION = "finaxial-c0-daily-fp32-v1"
CACHE_VERSION_DUAL = "finaxial-c0-daily-fp32-dual-v2"


@dataclass(frozen=True)
class DecisionFeatureCache:
    root: Path
    date_indices: np.ndarray
    hidden: np.ndarray
    base_score: np.ndarray
    eligible: np.ndarray
    tradable: np.ndarray
    manifest: dict
    predicted_return: np.ndarray | None = None

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        checkpoint_sha256: str,
        panel_manifest_sha256: str,
    ) -> "DecisionFeatureCache":
        root = Path(root)
        with (root / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("cache_version")
        if version not in {CACHE_VERSION, CACHE_VERSION_DUAL}:
            raise ValueError(f"unsupported decision cache version: {root}")
        if manifest.get("backbone_sha256") != checkpoint_sha256:
            raise ValueError(f"decision cache backbone hash mismatch: {root}")
        if manifest.get("panel_manifest_sha256") != panel_manifest_sha256:
            raise ValueError(f"decision cache panel hash mismatch: {root}")
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in ("date_indices", "hidden", "base_score", "eligible", "tradable")
        }
        predicted_return = (
            np.load(root / "predicted_return.npy", mmap_mode="r")
            if version == CACHE_VERSION_DUAL else None
        )
        dates = arrays["date_indices"]
        if dates.ndim != 1 or len(dates) == 0 or not np.all(np.diff(dates) == 1):
            raise ValueError(f"decision cache dates must be consecutive: {root}")
        shape = tuple(manifest["hidden_shape"])
        if tuple(arrays["hidden"].shape) != shape:
            raise ValueError(f"decision cache hidden shape mismatch: {root}")
        if any(tuple(arrays[name].shape) != shape[:2] for name in ("base_score", "eligible", "tradable")):
            raise ValueError(f"decision cache stock axis mismatch: {root}")
        if predicted_return is not None and tuple(predicted_return.shape) != shape[:2]:
            raise ValueError(f"decision cache return shape mismatch: {root}")
        return cls(root=root, manifest=manifest, predicted_return=predicted_return, **arrays)

    def rows_for_dates(self, dates: np.ndarray) -> np.ndarray:
        requested = np.asarray(dates, dtype=np.int64)
        rows = requested - int(self.date_indices[0])
        if not len(rows) or (rows < 0).any() or (rows >= len(self.date_indices)).any():
            raise IndexError("requested dates are outside the C0 cache")
        if not np.array_equal(np.asarray(self.date_indices[rows]), requested):
            raise ValueError("requested date indices do not match C0 cache")
        return rows


def cache_source_hashes(panel, backbone_checkpoint: str | Path) -> tuple[str, str]:
    return (
        sha256_file(backbone_checkpoint),
        sha256_file(panel.root / "manifest.json"),
    )
