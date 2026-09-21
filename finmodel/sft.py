from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .io import atomic_json_dump
from .tracking import SwanLabRun


@dataclass(frozen=True)
class SFTSplit:
    training_dates: tuple[int, ...]
    validation_dates: tuple[int, ...]
    boundary_excluded_date: int | None


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sft_split(panel, config: dict[str, Any]) -> SFTSplit:
    dates = np.asarray(panel.dates, dtype=np.int64)
    if dates.ndim != 1 or not len(dates) or not np.all(dates[:-1] < dates[1:]):
        raise ValueError("panel dates must be non-empty and strictly increasing")
    boundary_value = config.get("boundary_excluded_date")
    boundary = None if boundary_value is None else int(boundary_value)
    if boundary is None and not bool(config.get("allow_adjacent_train_validation", False)):
        raise ValueError(
            "boundary_excluded_date may be null only when "
            "allow_adjacent_train_validation=true"
        )
    if boundary is not None and boundary not in set(int(value) for value in dates):
        raise ValueError(f"boundary date {boundary} is absent from panel")
    training = dates[dates <= int(config["tuning_train_end"])]
    validation = dates[
        (dates >= int(config["validation_start"]))
        & (dates <= int(config["validation_end"]))
    ]
    expected = int(config["validation_days"])
    if len(validation) != expected:
        raise ValueError(f"expected {expected} validation dates, found {len(validation)}")
    if not len(training):
        raise ValueError("training split is empty")
    if boundary is not None and (boundary in training or boundary in validation):
        raise AssertionError("label-leaking boundary date was not isolated")
    if int(training[-1]) >= int(validation[0]):
        raise AssertionError("training and validation are not temporally disjoint")
    return SFTSplit(
        training_dates=tuple(int(value) for value in training),
        validation_dates=tuple(int(value) for value in validation),
        boundary_excluded_date=boundary,
    )


def panel_indices(panel, dates: Iterable[int]) -> np.ndarray:
    positions = {int(value): index for index, value in enumerate(panel.dates)}
    try:
        return np.asarray([positions[int(value)] for value in dates], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"date is absent from panel: {exc.args[0]}") from exc


def metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    stability = 1.0 - float(metrics["mean_turnover"])
    return {
        "final_score": float(metrics["final_score"]),
        "rank_ic_x_0.4": 0.4 * float(metrics["ic_mean"]),
        "annual_excess_x_0.3": 0.3 * float(metrics["annual_excess"]),
        "stability_x_0.3": 0.3 * stability,
        "rank_ic": float(metrics["ic_mean"]),
        "annual_excess": float(metrics["annual_excess"]),
        "one_minus_turnover": stability,
        "mse": float(metrics["mse"]),
        "mae": float(metrics["mae"]),
        "icir": float(metrics["icir"]),
        "ic_positive_ratio": float(metrics["ic_positive_ratio"]),
        "coverage": float(metrics["coverage"]),
    }


def swan_settings(config: dict[str, Any], *, name: str, tags: list[str], extra: dict[str, Any]):
    settings = config["swanlab"]
    return SwanLabRun(
        enabled=bool(settings["enabled"]),
        project=str(settings["project"]),
        name=name,
        logdir=settings["logdir"],
        config={"protocol": config["protocol"], "seed": config["seed"], **extra},
        tags=tags,
        mode=settings.get("mode", "online"),
    )


def save_torch_checkpoint(model, directory: str | Path, metadata: dict[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    atomic_json_dump(metadata, directory / "metadata.json")


def cosine_learning_rate(
    base_lr: float,
    *,
    update: int,
    total_updates: int,
    warmup_updates: int,
    eta_min_ratio: float,
) -> float:
    if not 0.0 <= eta_min_ratio <= 1.0:
        raise ValueError("cosine eta-min ratio must be in [0, 1]")
    update = max(1, min(int(update), int(total_updates)))
    if warmup_updates > 0 and update <= warmup_updates:
        return float(base_lr) * update / warmup_updates
    remaining = max(total_updates - warmup_updates, 1)
    progress = (update - warmup_updates - 1) / max(remaining - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (eta_min_ratio + (1.0 - eta_min_ratio) * cosine)


def reset_peak_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        if "Invalid device argument" not in str(exc):
            raise
        torch.cuda.reset_peak_memory_stats()


def job_name(
    stage: str, model: str, route: str, lookback: int, seed: int, *, budget: str | None = None,
) -> str:
    budget_part = f"-{budget}" if budget else ""
    return f"{stage}-{model}-{route}-lb{lookback}{budget_part}-seed{seed}"
