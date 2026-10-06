#!/usr/bin/env python3
"""Train the canonical FinAxial predictor model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from finmodel.io import atomic_json_dump, seed_everything
from finmodel.factors import FactorFeatureStore
from finmodel.factors_v4 import EXTENDED_PROFILES, ExtendedFactorFeatureStore
from finmodel.long_memory import LongMemoryFeatureStore
from finmodel.metrics import add_causal_ewma, evaluate_frame, make_prediction_frame
from finmodel.models.stock_time_transformer import StockTimeTransformer, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.objective import (
    absolute_return_regression_loss,
    compose_bounded_final_score,
    fixed_scale_excess_huber_loss,
    multi_date_soft_components,
    multi_date_soft_rank_ic,
    multi_date_soft_top10_excess,
    objective_settings,
    standardized_huber_loss,
)
from finmodel.losses import masked_mse
from finmodel.sequence import MultiDateCrossSectionDataset
from finmodel.sft import (
    cosine_learning_rate,
    dual_weighted_selection,
    job_name,
    load_config,
    metric_summary,
    mirror_last_checkpoint,
    panel_indices,
    raw_checkpoint_policy,
    reset_peak_memory,
    save_torch_checkpoint,
    use_last_checkpoint_policy,
    sft_split,
    swan_settings,
)


class NullTracker:
    def log(self, values, *, step=None) -> None:
        del values, step


def global_mean_with_local_gradient(value: torch.Tensor) -> torch.Tensor:
    """Cross-rank forward mean whose backward edge remains local for DDP."""
    mean = value.detach().clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    mean /= dist.get_world_size()
    return value + (mean - value.detach())


def build_model(config: dict[str, Any], stocks: int) -> StockTimeTransformer:
    return StockTimeTransformer(stocks=stocks, **config["model"])


def save_checkpoint(
    model: StockTimeTransformer,
    directory: Path,
    metadata: dict[str, Any],
    *,
    architecture: dict[str, Any],
    vocabulary_hash: str,
) -> None:
    save_torch_checkpoint(model, directory, {
        **metadata,
        "model": "finaxial_predictor",
        "architecture": architecture,
        "stock_vocab_sha256": vocabulary_hash,
    })


@torch.inference_mode()
def distributed_validation(
    model: StockTimeTransformer,
    dataset: MultiDateCrossSectionDataset,
    panel: Panel,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    route: str,
    ewma_alphas: tuple[float, ...],
    prediction_scale: float = 1.0,
    score_return_head: bool = False,
    score_position_start: int = 0,
    score_position_end: int | None = None,
) -> dict[str, Any] | None:
    model.eval()
    stocks = panel.shape[1]
    steps = dataset.output_steps
    position_end = steps if score_position_end is None else int(score_position_end)
    if not 0 <= score_position_start < position_end <= steps:
        raise ValueError("validation output-position range is invalid")
    positions = list(range(rank, len(dataset), world_size))
    max_count = (len(dataset) + world_size - 1) // world_size
    prediction_pad = torch.full((max_count, steps, stocks), float("nan"), device=device)
    return_pad = (
        torch.full_like(prediction_pad, float("nan"))
        if model.head_mode == "dual" else None
    )
    eligible_pad = torch.zeros(
        (max_count, steps, stocks), dtype=torch.uint8, device=device,
    )
    date_pad = torch.full((max_count, steps), -1, dtype=torch.int64, device=device)
    for slot, position in enumerate(positions):
        item = dataset[position]
        values = item["x"].to(device)
        token_valid = item["token_valid"].to(device)
        eligible = item["eligible"].to(device)
        model_output = model(
            values, token_valid, eligible,
            long_memory=(item["long_memory"].to(device) if "long_memory" in item else None),
            return_heads=model.head_mode == "dual",
        )
        if model.head_mode == "dual":
            prediction_pad[slot], return_pad[slot] = model_output
        else:
            prediction_pad[slot] = model_output
        eligible_pad[slot] = eligible.to(torch.uint8)
        date_pad[slot] = item["date_indices"].to(device)

    gathered_predictions = [torch.empty_like(prediction_pad) for _ in range(world_size)]
    gathered_returns = (
        [torch.empty_like(return_pad) for _ in range(world_size)]
        if return_pad is not None else None
    )
    gathered_eligible = [torch.empty_like(eligible_pad) for _ in range(world_size)]
    gathered_dates = [torch.empty_like(date_pad) for _ in range(world_size)]
    dist.all_gather(gathered_predictions, prediction_pad)
    if return_pad is not None:
        dist.all_gather(gathered_returns, return_pad)
    dist.all_gather(gathered_eligible, eligible_pad)
    dist.all_gather(gathered_dates, date_pad)
    if rank != 0:
        return None

    requested = np.asarray(dataset.requested_date_indices, dtype=np.int64)
    lookup = {int(date): offset for offset, date in enumerate(requested)}
    predictions = np.full((len(requested), stocks), np.nan, dtype=np.float32)
    eligibility = np.zeros((len(requested), stocks), dtype=bool)
    returns = np.full_like(predictions, np.nan) if gathered_returns is not None else None
    selected_output_position = np.full(len(requested), -1, dtype=np.int16)
    for source_rank, (gathered, eligible, dates) in enumerate(zip(
        gathered_predictions, gathered_eligible, gathered_dates,
    )):
        for block in range(max_count):
            for step, date_idx in enumerate(dates[block].cpu().tolist()):
                if not score_position_start <= step < position_end:
                    continue
                position = lookup.get(int(date_idx))
                if position is None or step <= selected_output_position[position]:
                    continue
                predictions[position] = gathered[block, step].cpu().numpy()
                if returns is not None:
                    returns[position] = gathered_returns[source_rank][block, step].cpu().numpy()
                eligibility[position] = eligible[block, step].cpu().numpy().astype(bool)
                selected_output_position[position] = step
    if not np.isfinite(predictions).all():
        raise RuntimeError("distributed validation did not cover every requested date")
    if returns is not None and not np.isfinite(returns).all():
        raise RuntimeError("distributed validation did not cover every return date")

    def score(predictions: np.ndarray, suffix: str):
        frame = make_prediction_frame(
            panel=panel,
            date_indices=requested,
            predictions=predictions,
            eligible=eligibility,
            model="finaxial_predictor",
            route=f"{route}_{suffix}",
            fold="validation",
            alpha=1.0,
        )
        raw = evaluate_frame(frame, "pred_raw")
        smoothed = {}
        for alpha in ewma_alphas:
            if alpha >= 1.0:
                continue
            frame["pred_smoothed"] = add_causal_ewma(frame, alpha, source="pred_rank")
            smoothed[f"alpha_{alpha:g}"] = evaluate_frame(frame, "pred_smoothed")
        return raw, smoothed

    raw_metrics, raw_ewma = score(
        predictions * float(prediction_scale), "raw",
    )
    result = {
        "raw_metrics": raw_metrics,
        "raw_ewma_metrics": raw_ewma,
    }
    if model.head_mode == "return_only":
        returns = predictions
    if returns is not None:
        if score_return_head:
            return_raw, return_ewma = score(returns, "return_head")
            result["return_head_raw_metrics"] = return_raw
            result["return_head_ewma_metrics"] = return_ewma
        valid = eligibility & np.asarray(panel.label_valid[requested], dtype=bool)
        truth = np.asarray(panel.labels[requested], dtype=np.float32)
        valid &= np.isfinite(truth)
        error = returns[valid] - truth[valid]
        daily_pearson = []
        for date in range(len(requested)):
            if int(valid[date].sum()) < 2:
                continue
            predicted = returns[date, valid[date]].astype(np.float64)
            actual = truth[date, valid[date]].astype(np.float64)
            predicted -= predicted.mean()
            actual -= actual.mean()
            denominator = float(np.linalg.norm(predicted) * np.linalg.norm(actual))
            if denominator > 1e-12:
                daily_pearson.append(float(np.dot(predicted, actual) / denominator))
        result["return_metrics"] = {
            "mae": float(np.mean(np.abs(error))) if len(error) else 0.0,
            "mse": float(np.mean(error * error)) if len(error) else 0.0,
            "bias": float(np.mean(error)) if len(error) else 0.0,
            "pearson_ic_mean": float(np.mean(daily_pearson)) if daily_pearson else 0.0,
            "pearson_ic_std": float(np.std(daily_pearson)) if daily_pearson else 0.0,
            "prediction_std": float(np.std(returns[valid])) if len(error) else 0.0,
            "target_std": float(np.std(truth[valid])) if len(error) else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-train-blocks", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--limit-validation-blocks", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--disable-swanlab", action="store_true")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch training with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("FINMODEL_DDP_TIMEOUT_SECONDS", "180"))),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    config = load_config(args.config)
    training = config["training"]
    use_last_checkpoint_policy(training)
    objective_name = str(training.get("objective", "multi_date_final_global_excess"))
    return_loss_type = str(training.get("return_loss_type", "huber"))
    if return_loss_type not in {"huber", "mae"}:
        raise ValueError(f"unknown return loss type: {return_loss_type}")
    if return_loss_type == "mae" and objective_name != "dual_rank_return":
        raise ValueError("MAE ablation requires the dual Rank-IC/return objective")
    if objective_name not in {
        "multi_date_final_global_excess", "rank_ic_huber", "rank_ic_only",
        "rank_ic_excess_huber", "dual_rank_return", "return_huber",
    }:
        raise ValueError(f"unknown training objective: {objective_name}")
    top10_excess_weight = float(training.get("top10_excess_weight", 0.0))
    if top10_excess_weight < 0:
        raise ValueError("top10_excess_weight must be non-negative")
    if top10_excess_weight and objective_name not in {
        "rank_ic_huber", "rank_ic_only", "rank_ic_excess_huber", "dual_rank_return",
    }:
        raise ValueError("top10_excess_weight requires a Rank-IC objective")
    selection_metric = str(training.get("selection_metric", "exact_final_score"))
    if selection_metric not in {"exact_final_score", "exact_rank_ic", "return_mse", "dual_weighted"}:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    max_epochs = int(training["max_epochs"])
    epochs = int(args.epochs or max_epochs)
    if not 1 <= epochs <= max_epochs:
        raise ValueError(f"epochs must be in [1, {max_epochs}]")
    seed = int(config["seed"])
    seed_everything(seed)
    panel = Panel.open(args.panel)
    split = sft_split(panel, config)
    train_indices = panel_indices(panel, split.training_dates)
    validation_indices = panel_indices(panel, split.validation_dates)
    data = config["data"]
    validation_position = config.get("validation", {}).get("score_output_positions")
    if validation_position is None:
        position_start, position_end = 0, int(config["model"]["output_steps"])
        validation_requested = validation_indices
        validation_future_padding = 0
    else:
        position_start, position_end = map(int, validation_position)
        output_steps = int(config["model"]["output_steps"])
        if not 0 <= position_start < position_end <= output_steps:
            raise ValueError("validation score_output_positions must lie in output steps")
        if int(validation_indices[-1]) != panel.shape[0] - 1:
            raise ValueError(
                "position-banded validation requires the validation split to end "
                "at the final panel date, so padding cannot expose later real data"
            )
        validation_future_padding = output_steps - position_end
        validation_requested = np.arange(
            int(validation_indices[0]) - position_start,
            int(validation_indices[-1]) + validation_future_padding + 1,
            dtype=np.int64,
        )
        if validation_requested[0] < 0:
            raise ValueError("validation needs more prefix history than panel provides")
    memory_store = (
        LongMemoryFeatureStore.open(
            data["long_memory_cache"], panel,
            config["model"].get("long_memory_scales", ()),
        ) if config["model"].get("long_memory_scales") else None
    )
    long_features = memory_store.features if memory_store is not None else None
    if data.get("feature_mode") == "factors":
        if data["factor_profile"] in EXTENDED_PROFILES:
            factor_store = ExtendedFactorFeatureStore.open(
                data["factor_cache"], data["base_factor_cache"], panel,
                data["factor_profile"],
                expected_train_end=int(config["tuning_train_end"]),
            )
        else:
            factor_store = FactorFeatureStore.open(
                data["factor_cache"], panel, data["factor_profile"],
                expected_train_end=int(config["tuning_train_end"]),
            )
    else:
        factor_store = None
    factor_features = factor_store.features if factor_store is not None else None
    train_dataset = MultiDateCrossSectionDataset(
        panel, train_indices,
        lookback=int(config["model"]["lookback"]),
        output_steps=int(config["model"]["output_steps"]),
        context_days=config["model"].get("context_days"),
        stride=int(data["train_stride"]),
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
        feature_mode=str(data.get("feature_mode", "temporal")),
        long_memory_features=long_features,
        factor_features=factor_features,
    )
    validation_dataset = MultiDateCrossSectionDataset(
        panel, validation_requested,
        lookback=int(config["model"]["lookback"]),
        output_steps=int(config["model"]["output_steps"]),
        context_days=config["model"].get("context_days"),
        stride=position_end - position_start,
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
        feature_mode=str(data.get("feature_mode", "temporal")),
        long_memory_features=long_features,
        factor_features=factor_features,
        inference_future_padding=validation_future_padding,
        require_trainable=validation_position is None,
    )
    if validation_position is not None:
        # The dataset uses earlier inputs and masked future padding to place
        # every validation date in the same causal output-position band.
        validation_dataset.requested_date_indices = validation_indices
    configured_channels = int(config["model"]["channels"])
    if train_dataset.channels != configured_channels:
        raise ValueError(
            f"feature_mode {train_dataset.feature_mode!r} produces "
            f"{train_dataset.channels} channels, but model.channels={configured_channels}"
        )
    if args.limit_train_blocks:
        train_dataset.output_blocks = train_dataset.output_blocks[-args.limit_train_blocks:]
    if args.limit_validation_blocks:
        validation_dataset.output_blocks = validation_dataset.output_blocks[:args.limit_validation_blocks]
        if validation_position is None:
            kept = np.unique(validation_dataset.output_blocks.reshape(-1))
        else:
            scored = validation_dataset.output_blocks[:, position_start:position_end]
            kept = np.intersect1d(scored.reshape(-1), validation_indices)
        validation_dataset.requested_date_indices = kept

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError("distributed shard is empty")

    model = build_model(config, panel.shape[1]).to(device)
    raw_policy = raw_checkpoint_policy(training, model.head_mode)
    if (objective_name == "dual_rank_return") != (model.head_mode == "dual"):
        raise ValueError("dual_rank_return objective and dual model head must be configured together")
    if (objective_name == "return_huber") != (model.head_mode == "return_only"):
        raise ValueError("return_huber objective and return_only head must be configured together")
    if (selection_metric == "return_mse") != (objective_name == "return_huber"):
        raise ValueError("return_mse checkpoint selection requires return_huber and vice versa")
    if selection_metric == "dual_weighted" and objective_name != "dual_rank_return":
        raise ValueError("dual_weighted checkpoint selection requires dual_rank_return")
    if selection_metric == "dual_weighted":
        dual_weighted_selection(
            {"rank_ic": 0.0, "annual_excess": 0.0}, {"mse": 0.0}, training,
        )
    ddp_model = DDP(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    updates_per_epoch = len(loader)
    total_updates = epochs * updates_per_epoch
    warmup_updates = int(training["warmup_epochs"]) * updates_per_epoch
    smoothing_epochs = int(training["validation_smoothing_epochs"])
    ewma_alphas = tuple(float(x) for x in training["validation_ewma_alphas"])
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    reset_peak_memory(device)

    route = str(training["route_name"])
    name = job_name(
        "training", "finaxial-predictor", route,
        int(config["model"]["lookback"]), seed,
        budget=f"k{config['model']['output_steps']}-ep{epochs}-ddp{world_size}",
    )
    tracker_context = swan_settings(
        config,
        name=name,
        tags=["training", "finaxial-predictor", "multi-date",
              "return-only" if objective_name == "return_huber" else "final-global"],
        extra={
            "world_size": world_size,
            "epochs": epochs,
            "train_blocks": len(train_dataset),
            "dates_per_block": int(config["model"]["output_steps"]),
            "validation_score_output_positions": [position_start, position_end],
            "validation_future_padding_days": validation_future_padding,
            "architecture": config["model"],
            **training,
        },
    ) if rank == 0 and not args.disable_swanlab else nullcontext(NullTracker())

    vocabulary_hash = stock_vocab_sha256(panel.codes)
    history: list[dict[str, Any]] = []
    global_update = 0
    best_stable = -float("inf")
    best_endpoint = -float("inf")
    best_epoch = 0
    best_update = 0
    best_raw, best_raw_epoch = -float("inf"), 0
    peak_raw_final, peak_raw_epoch = -float("inf"), 0
    started = time.perf_counter()
    log_every = int(config["swanlab"]["log_interval_updates"])

    try:
        with tracker_context as tracker:
            for epoch in range(1, epochs + 1):
                sampler.set_epoch(epoch)
                ddp_model.train()
                loss_settings = objective_settings(training, epoch)
                interval = np.zeros(11, dtype=np.float64)
                interval_count = 0
                interval_started = time.perf_counter()
                for step, item in enumerate(loader, start=1):
                    optimizer.zero_grad(set_to_none=True)
                    target = item["target"].to(device).squeeze(0)
                    label_mask = item["mask"].to(device).squeeze(0)
                    model_output = ddp_model(
                        item["x"].to(device),
                        item["token_valid"].to(device),
                        item["eligible"].to(device),
                        long_memory=(
                            item["long_memory"].to(device)
                            if "long_memory" in item else None
                        ),
                        return_heads=objective_name == "dual_rank_return",
                    )
                    if objective_name == "dual_rank_return":
                        prediction, return_prediction = model_output
                    else:
                        prediction = model_output
                    tradable_mask = item["tradable"].to(device).squeeze(0)
                    if objective_name == "rank_ic_only":
                        global_huber = prediction.sum() * 0.0
                    elif objective_name == "rank_ic_excess_huber":
                        local_huber = fixed_scale_excess_huber_loss(
                            prediction, target, label_mask, tradable_mask,
                            return_scale=float(training["return_scale"]),
                            target_clip=float(training.get("return_target_clip", 5.0)),
                            delta=float(training.get("huber_delta", 0.5)),
                        )
                        global_huber = global_mean_with_local_gradient(local_huber)
                    elif objective_name == "dual_rank_return":
                        local_huber = absolute_return_regression_loss(
                            return_prediction, target, label_mask,
                            loss_type=return_loss_type,
                            return_scale=float(training["return_scale"]),
                            target_clip=float(training.get("return_target_clip", 5.0)),
                            delta=float(training.get("huber_delta", 0.5)),
                        )
                        global_huber = global_mean_with_local_gradient(local_huber)
                    elif objective_name == "return_huber":
                        local_huber = absolute_return_regression_loss(
                            prediction, target, label_mask,
                            loss_type=return_loss_type,
                            return_scale=float(training["return_scale"]),
                            target_clip=float(training.get("return_target_clip", 5.0)),
                            delta=float(training.get("huber_delta", 0.5)),
                        )
                        global_huber = global_mean_with_local_gradient(local_huber)
                    else:
                        local_huber = standardized_huber_loss(
                            prediction, target, label_mask,
                            delta=float(training.get("huber_delta", 0.5)),
                        )
                        global_huber = global_mean_with_local_gradient(local_huber)
                    if objective_name == "return_huber":
                        global_rank = prediction.sum() * 0.0
                        global_excess = global_rank
                        bounded_excess = global_rank
                        global_stability = global_rank
                        mse_diagnostic = masked_mse(prediction, target, label_mask)
                        loss = global_huber
                        soft_score = -global_huber
                    elif objective_name in {
                        "rank_ic_huber", "rank_ic_only", "rank_ic_excess_huber", "dual_rank_return",
                    }:
                        local_rank = multi_date_soft_rank_ic(
                            prediction, target, label_mask,
                            temperature=loss_settings.rank_temperature,
                        )
                        global_rank = global_mean_with_local_gradient(local_rank)
                        if top10_excess_weight:
                            local_excess = multi_date_soft_top10_excess(
                                prediction, target, label_mask, tradable_mask,
                                temperature=loss_settings.top_temperature,
                            )
                            global_excess = global_mean_with_local_gradient(local_excess)
                            excess_bound = float(training["excess_bound"])
                            bounded_excess = excess_bound * torch.tanh(
                                global_excess / excess_bound
                            )
                        else:
                            # Preserve the old Rank-IC route's compute cost.
                            global_excess = prediction.sum() * 0.0
                            bounded_excess = global_excess
                        global_stability = prediction.sum() * 0.0
                        mse_diagnostic = masked_mse(
                            return_prediction if objective_name == "dual_rank_return" else prediction,
                            target, label_mask,
                        )
                        loss = -global_rank
                        loss = loss - top10_excess_weight * bounded_excess
                        if objective_name in {"rank_ic_huber", "rank_ic_excess_huber", "dual_rank_return"}:
                            loss = loss + float(
                                training.get("huber_weight", 0.1)
                            ) * global_huber
                        soft_score = global_rank + top10_excess_weight * bounded_excess
                    else:
                        components = multi_date_soft_components(
                            prediction, target, label_mask,
                            tradable_mask,
                            rank_temperature=loss_settings.rank_temperature,
                            top_temperature=loss_settings.top_temperature,
                        )
                        global_rank = global_mean_with_local_gradient(components.rank_ic)
                        global_excess = global_mean_with_local_gradient(
                            components.annual_excess_raw
                        )
                        global_stability = global_mean_with_local_gradient(
                            components.stability
                        )
                        loss, soft_score, bounded_excess = compose_bounded_final_score(
                            components,
                            global_rank_ic=global_rank,
                            global_annual_excess_raw=global_excess,
                            global_stability=global_stability,
                            excess_bound=float(training["excess_bound"]),
                            mse_weight=float(training["mse_weight"]),
                            component_weights=loss_settings.component_weights,
                            range_balance_beta=loss_settings.range_balance_beta,
                        )
                        mse_diagnostic = components.mse
                    official_soft_score = (
                        0.4 * global_rank
                        + 0.3 * bounded_excess
                        + 0.3 * global_stability
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss epoch={epoch} step={step}")
                    if step == 1 and objective_name != "return_huber":
                        proxy_values = []
                        for component in (global_rank, bounded_excess, global_stability):
                            gradient = torch.autograd.grad(
                                component, prediction,
                                retain_graph=True, allow_unused=True,
                            )[0]
                            proxy_values.append(
                                torch.zeros((), device=device)
                                if gradient is None else gradient.float().norm()
                                / max(gradient.numel() ** 0.5, 1.0)
                            )
                        proxy = torch.stack(proxy_values)
                        dist.all_reduce(proxy, op=dist.ReduceOp.SUM)
                        proxy /= world_size
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), float(training["gradient_clip"]),
                    )
                    next_update = global_update + 1
                    learning_rate = cosine_learning_rate(
                        float(training["learning_rate"]),
                        update=next_update,
                        total_updates=total_updates,
                        warmup_updates=warmup_updates,
                        eta_min_ratio=float(training["cosine_eta_min_ratio"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    optimizer.step()
                    global_update = next_update
                    if rank == 0 and global_update == 1:
                        print(
                            "FIRST_OPTIMIZER_UPDATE_COMPLETE "
                            f"world_size={world_size} dates_per_rank={config['model']['output_steps']}",
                            flush=True,
                        )

                    values = np.asarray([
                        float(loss.detach()), float(soft_score.detach()),
                        float(official_soft_score.detach()), float(global_rank.detach()),
                        float(global_excess.detach()), float(bounded_excess.detach()),
                        float(global_stability.detach()), float(mse_diagnostic.detach()),
                        float(global_huber.detach()), float(grad_norm), learning_rate,
                    ])
                    interval += values
                    interval_count += 1
                    if global_update == 1 or interval_count >= log_every or step == len(loader):
                        reduced = torch.tensor(
                            [*interval.tolist(), float(interval_count)],
                            dtype=torch.float64, device=device,
                        )
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        denominator = max(float(reduced[-1]), 1.0)
                        means = (reduced[:-1] / denominator).cpu().tolist()
                        if rank == 0:
                            training_logs = {
                                "train/loss": means[0],
                                "train/objective_score": means[1],
                                "train/mse_diagnostic": means[7],
                                "train/grad_norm": means[9],
                                "train/lr": means[10],
                                "train/epoch": epoch,
                                "train/optimizer_update": global_update,
                                "train/dates_per_global_update": (
                                    world_size * int(config["model"]["output_steps"])
                                ),
                                "train/updates_per_second": interval_count / max(
                                    time.perf_counter() - interval_started, 1e-9,
                                ),
                            }
                            if objective_name == "return_huber":
                                training_logs["train/return_huber_loss"] = means[8]
                            else:
                                training_logs.update({
                                    (
                                        "train/soft_final_official"
                                        if objective_name == "multi_date_final_global_excess"
                                        else "train/soft_rank_excess_no_turnover"
                                    ): means[2],
                                    "train/soft_rank_ic": means[3],
                                    "train/soft_annual_excess_raw": means[4],
                                    "train/soft_annual_excess_bounded": means[5],
                                    "train/soft_one_minus_turnover": means[6],
                                    ("train/auxiliary_mae" if return_loss_type == "mae"
                                     else "train/auxiliary_huber"): means[8],
                                    "train/rank_temperature": loss_settings.rank_temperature,
                                    "train/top_temperature": loss_settings.top_temperature,
                                    "train/rank_weight": loss_settings.component_weights[0],
                                    "train/excess_weight": loss_settings.component_weights[1],
                                    "train/top10_excess_weight": top10_excess_weight,
                                    "train/stability_weight": loss_settings.component_weights[2],
                                    "train/range_balance_beta": loss_settings.range_balance_beta,
                                })
                            tracker.log(training_logs, step=global_update)
                            if step == 1 and objective_name != "return_huber":
                                tracker.log({
                                    "train/gradient_proxy/rank_prediction": float(proxy[0]),
                                    "train/gradient_proxy/excess_prediction": float(proxy[1]),
                                    "train/gradient_proxy/stability_prediction": float(proxy[2]),
                                    "train/gradient_proxy/epoch": epoch,
                                }, step=global_update)
                        interval[:] = 0
                        interval_count = 0
                        interval_started = time.perf_counter()

                validation = distributed_validation(
                    model, validation_dataset, panel, device,
                    rank=rank, world_size=world_size, route=route,
                    ewma_alphas=ewma_alphas,
                    prediction_scale=(
                        float(training["return_scale"])
                        if objective_name == "rank_ic_excess_huber" else 1.0
                    ),
                    score_position_start=position_start,
                    score_position_end=position_end,
                )
                if rank == 0:
                    assert validation is not None
                    raw_summary = metric_summary(validation["raw_metrics"])
                    raw_ewma = {
                        alpha: metric_summary(metrics)
                        for alpha, metrics in validation["raw_ewma_metrics"].items()
                    }
                    row = {
                        "epoch": epoch,
                        "optimizer_update": global_update,
                        "raw": raw_summary,
                        "raw_ewma": raw_ewma,
                        **({"return_metrics": validation["return_metrics"]}
                           if "return_metrics" in validation else {}),
                    }
                    if selection_metric == "dual_weighted":
                        row["selection_components"] = dual_weighted_selection(
                            raw_summary, validation["return_metrics"], training,
                        )
                    history.append(row)
                    window = history[-smoothing_epochs:]
                    if selection_metric == "return_mse":
                        selection_key = "negative_return_mse"
                        selection_values = [-float(x["return_metrics"]["mse"]) for x in window]
                    elif selection_metric == "dual_weighted":
                        selection_key = "dual_weighted_score"
                        selection_values = [float(x["selection_components"]["score"]) for x in window]
                    else:
                        selection_key = (
                            "final_score" if selection_metric == "exact_final_score" else "rank_ic"
                        )
                        selection_values = [float(x["raw"][selection_key]) for x in window]
                    stable = float(np.mean(selection_values))
                    stable_std = float(np.std(selection_values))
                    row.update({
                        "stable_selection_mean": stable,
                        "stable_selection_std": stable_std,
                        "stable_window_size": len(window),
                    })
                    atomic_json_dump(history, output / "validation_history.json")
                    ewma_logs = {}
                    for alpha, summary in raw_ewma.items():
                        for key, value in summary.items():
                            ewma_logs[f"validation/raw/ewma/{alpha}/{key}"] = value
                    return_logs = (
                        {f"validation/return/{key}": value
                         for key, value in validation["return_metrics"].items()}
                        if "return_metrics" in validation else {}
                    )
                    validation_logs = {
                        **(return_logs if objective_name == "return_huber" else {}),
                        **{f"validation/raw/{key}": value for key, value in raw_summary.items()},
                        **(return_logs if objective_name != "return_huber" else {}),
                        **ewma_logs,
                        f"validation/stability/{selection_key}_ma{smoothing_epochs}": stable,
                        f"validation/stability/{selection_key}_window_std": stable_std,
                        "validation/epoch": epoch,
                        "validation/optimizer_update": global_update,
                    }
                    if selection_metric == "dual_weighted":
                        validation_logs.update({
                            f"validation/checkpoint_selection/{key}": value
                            for key, value in row["selection_components"].items()
                        })
                    if selection_metric == "return_mse":
                        validation_logs[f"validation/return/mse_ma{smoothing_epochs}"] = -stable
                    if model.attention_mode in {"identity_init", "identity_tanh"}:
                        def effective_gate(block):
                            raw = block.attention.self_gate.detach()
                            return float(
                                raw.tanh() if model.attention_mode == "identity_tanh"
                                else raw.clamp(0, 1)
                            )
                        temporal_gates = [
                            effective_gate(block)
                            for block in model.temporal_blocks
                        ]
                        stock_gates = [
                            effective_gate(block)
                            for block in model.stock_blocks
                        ]
                        validation_logs.update({
                            "validation/attention_gate/temporal_mean": float(np.mean(temporal_gates)),
                            "validation/attention_gate/stock_mean": float(np.mean(stock_gates)),
                            "validation/attention_gate/temporal_abs_mean": float(np.mean(np.abs(temporal_gates))),
                            "validation/attention_gate/stock_abs_mean": float(np.mean(np.abs(stock_gates))),
                        })
                    tracker.log(validation_logs, step=global_update)

                    raw_score = float(raw_summary["final_score"])
                    if raw_score > peak_raw_final:
                        peak_raw_final, peak_raw_epoch = raw_score, epoch
                    # Keep only the latest completed epoch. All validation
                    # scores (including weighted scores) are diagnostics only.
                    if epoch > 0:
                        best_raw, best_raw_epoch = raw_score, epoch
                        best_stable, best_endpoint = stable, raw_score
                        best_epoch, best_update = epoch, global_update
                        save_checkpoint(
                            model, output / "last",
                            {"epoch": epoch, "optimizer_update": global_update,
                             "checkpoint_selection": "last_completed_epoch",
                             "validation_used_for_checkpoint_selection": False,
                             "raw_checkpoint_policy": "last_epoch",
                             "validation_final_score": raw_score,
                             "validation_rank_ic": float(raw_summary["rank_ic"]),
                             "validation_return_mse": validation.get("return_metrics", {}).get("mse"),
                             "selection_metric": selection_metric,
                             "selection_components": row.get("selection_components"),
                             "selection_weights": training.get("selection_weights") if selection_metric == "dual_weighted" else None,
                             "selection_scales": training.get("selection_scales") if selection_metric == "dual_weighted" else None,
                             "selection_mse_reference": training.get("selection_mse_reference") if selection_metric == "dual_weighted" else None,
                             "training_objective": objective_name,
                             "return_loss_type": return_loss_type,
                             "return_scale": training.get("return_scale"),
                             "stable_validation_selection_value": stable,
                             "stable_validation_return_mse": -stable if selection_metric == "return_mse" else None,
                             "stable_validation_selection_std": stable_std,
                             "parameter_source": "raw",
                             "score_components": raw_summary,
                             "return_metrics": validation.get("return_metrics")},
                            architecture=config["model"], vocabulary_hash=vocabulary_hash,
                        )
                        mirror_last_checkpoint(output / "last", ("best", "best_raw"))
                dist.barrier()

            peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                selected = next(row for row in history if row["epoch"] == best_epoch)
                summary = {
                    "experiment_protocol": config["protocol"],
                    "panel": str(Path(args.panel).resolve()),
                    "world_size": world_size,
                    "epochs_trained": epochs,
                    "optimizer_updates": global_update,
                    "updates_per_epoch": updates_per_epoch,
                    "train_blocks": len(train_dataset),
                    "validation_blocks": len(validation_dataset),
                    "validation_days_scored": len(validation_dataset.requested_date_indices),
                    "validation_score_output_positions": [position_start, position_end],
                    "validation_future_padding_days": validation_future_padding,
                    "lookback": int(config["model"]["lookback"]),
                    "context_days": int(model.context_days),
                    "temporal_window": int(model.temporal_window),
                    "output_steps": int(config["model"]["output_steps"]),
                    "sequence_length": model.sequence_length,
                    "dates_per_global_update": world_size * int(config["model"]["output_steps"]),
                    "training_objective": objective_name,
                    "return_loss_type": return_loss_type,
                    "top10_excess_weight": top10_excess_weight,
                    "head_mode": model.head_mode,
                    "attention_mode": model.attention_mode,
                    "long_memory_scales": list(model.long_memory_scales),
                    "long_memory_cache": data.get("long_memory_cache"),
                    "return_scale": training.get("return_scale"),
                    "initialization": str(
                        training.get("initialization", "random_from_scratch")
                    ),
                    "loss_variant": str(training.get("loss_variant", "official")),
                    "feature_mode": str(data.get("feature_mode", "temporal")),
                    "factor_profile": data.get("factor_profile"),
                    "factor_cache": data.get("factor_cache"),
                    "base_factor_cache": data.get("base_factor_cache"),
                    "architecture": str(config["model"].get("architecture", "stacked")),
                    "checkpoint_selection": "last_completed_epoch",
                    "checkpoint_path": str(output / "last"),
                    "validation_used_for_checkpoint_selection": False,
                    "last_epoch": best_epoch,
                    "last_score_components": selected["raw"],
                    "checkpoint_parameter_source": "raw",
                    "selection_metric": selection_metric,
                    "selection_weights": training.get("selection_weights") if selection_metric == "dual_weighted" else None,
                    "selection_scales": training.get("selection_scales") if selection_metric == "dual_weighted" else None,
                    "selection_mse_reference": training.get("selection_mse_reference") if selection_metric == "dual_weighted" else None,
                    "best_raw_checkpoint_policy": raw_policy,
                    "parameter_ema_enabled": False,
                    "best_epoch": best_epoch,
                    "best_optimizer_update": best_update,
                    "best_validation_final_score": best_endpoint,
                    "best_validation_rank_ic": selected["raw"]["rank_ic"],
                    "best_validation_return_mse": selected.get("return_metrics", {}).get("mse"),
                    "best_stable_validation_selection_value": best_stable,
                    "best_stable_validation_return_mse": -best_stable if selection_metric == "return_mse" else None,
                    "best_checkpoint_score_components": selected["raw"],
                    "best_checkpoint_return_metrics": selected.get("return_metrics"),
                    "best_checkpoint_selection_components": selected.get("selection_components"),
                    "best_checkpoint_ewma_components": selected["raw_ewma"],
                    "best_raw_validation_final_score": best_raw,
                    "best_raw_epoch": best_raw_epoch,
                    "peak_raw_validation_final_score": peak_raw_final,
                    "peak_raw_validation_final_score_epoch": peak_raw_epoch,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "peak_memory_bytes_max_rank": int(peak.item()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "normalization": str(data.get(
                        "normalization", "per-token trailing-window causal z-score",
                    )),
                    "train_stride": int(data["train_stride"]),
                }
                atomic_json_dump(summary, output / "train_summary.json")
                atomic_json_dump({"complete": True, "summary": summary}, output / "complete.json")
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
