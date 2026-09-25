"""Shared D0 dataset, checkpoint loading, and score-evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import sha256_file
from .long_memory import LongMemoryFeatureStore
from .metrics import add_causal_ewma, evaluate_frame, make_prediction_frame
from .models import StockTimeTransformer, stock_vocab_sha256
from .panel import Panel
from .sequence import MultiDateCrossSectionDataset
from .sft import load_config, metric_summary


def build_dataset(
    panel: Panel,
    indices: np.ndarray,
    config: dict[str, Any],
    *,
    stride: int,
) -> MultiDateCrossSectionDataset:
    model, data = config["model"], config["data"]
    memory = (
        LongMemoryFeatureStore.open(
            data["long_memory_cache"], panel, model["long_memory_scales"],
        ).features if model.get("long_memory_scales") else None
    )
    dataset = MultiDateCrossSectionDataset(
        panel, indices,
        lookback=int(model["lookback"]),
        output_steps=int(model["output_steps"]),
        context_days=model.get("context_days"),
        stride=int(stride),
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
        feature_mode=str(data.get("feature_mode", "temporal")),
        long_memory_features=memory,
    )
    if dataset.channels != int(model["channels"]):
        raise ValueError("dataset and model channel counts differ")
    return dataset


def load_backbone(
    config: dict[str, Any], panel: Panel, device: torch.device,
) -> tuple[StockTimeTransformer, str]:
    checkpoint = Path(config["backbone_checkpoint"])
    metadata = load_config(config["backbone_metadata"])
    vocabulary_hash = stock_vocab_sha256(panel.codes)
    if metadata.get("stock_vocab_sha256") != vocabulary_hash:
        raise ValueError("predictor checkpoint stock vocabulary does not match panel")
    backbone = StockTimeTransformer(stocks=panel.shape[1], **config["model"]).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone, sha256_file(checkpoint)


def score_numpy_predictions(
    *,
    panel: Panel,
    indices: np.ndarray,
    predictions: np.ndarray,
    eligible: np.ndarray,
    route: str,
    ewma_alphas: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    frame = make_prediction_frame(
        panel=panel,
        date_indices=indices,
        predictions=predictions,
        eligible=eligible,
        model="finaxial_c0_grpo",
        route=route,
        fold="post_hoc_full_test_validation",
        alpha=1.0,
    )
    raw = metric_summary(evaluate_frame(frame, "pred_raw"))
    ewma: dict[str, dict[str, float]] = {}
    for alpha in ewma_alphas:
        if alpha >= 1.0:
            continue
        frame["pred_smoothed"] = add_causal_ewma(frame, alpha, source="pred_rank")
        ewma[f"alpha_{alpha:g}"] = metric_summary(
            evaluate_frame(frame, "pred_smoothed")
        )
    return raw, ewma
