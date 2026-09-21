#!/usr/bin/env python3
"""Train the canonical B0 Stock-Time Transformer."""

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
from finmodel.metrics import add_causal_ewma, evaluate_frame, make_prediction_frame
from finmodel.models.stock_time_transformer import StockTimeTransformer, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.objective import compose_bounded_final_score, multi_date_soft_components
from finmodel.sequence import MultiDateCrossSectionDataset
from finmodel.sft import (
    cosine_learning_rate,
    job_name,
    load_config,
    metric_summary,
    panel_indices,
    reset_peak_memory,
    save_torch_checkpoint,
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
        "model": "stock_time_transformer_b0",
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
) -> dict[str, Any] | None:
    model.eval()
    stocks = panel.shape[1]
    steps = dataset.output_steps
    positions = list(range(rank, len(dataset), world_size))
    max_count = (len(dataset) + world_size - 1) // world_size
    prediction_pad = torch.full((max_count, steps, stocks), float("nan"), device=device)
    eligible_pad = torch.zeros(
        (max_count, steps, stocks), dtype=torch.uint8, device=device,
    )
    date_pad = torch.full((max_count, steps), -1, dtype=torch.int64, device=device)
    for slot, position in enumerate(positions):
        item = dataset[position]
        values = item["x"].to(device)
        token_valid = item["token_valid"].to(device)
        eligible = item["eligible"].to(device)
        prediction_pad[slot] = model(values, token_valid, eligible)
        eligible_pad[slot] = eligible.to(torch.uint8)
        date_pad[slot] = item["date_indices"].to(device)

    gathered_predictions = [torch.empty_like(prediction_pad) for _ in range(world_size)]
    gathered_eligible = [torch.empty_like(eligible_pad) for _ in range(world_size)]
    gathered_dates = [torch.empty_like(date_pad) for _ in range(world_size)]
    dist.all_gather(gathered_predictions, prediction_pad)
    dist.all_gather(gathered_eligible, eligible_pad)
    dist.all_gather(gathered_dates, date_pad)
    if rank != 0:
        return None

    requested = np.asarray(dataset.requested_date_indices, dtype=np.int64)
    lookup = {int(date): offset for offset, date in enumerate(requested)}
    predictions = np.full((len(requested), stocks), np.nan, dtype=np.float32)
    eligibility = np.zeros((len(requested), stocks), dtype=bool)
    for gathered, eligible, dates in zip(
        gathered_predictions, gathered_eligible, gathered_dates,
    ):
        for block in range(max_count):
            for step, date_idx in enumerate(dates[block].cpu().tolist()):
                position = lookup.get(int(date_idx))
                if position is None:
                    continue
                predictions[position] = gathered[block, step].cpu().numpy()
                eligibility[position] = eligible[block, step].cpu().numpy().astype(bool)
    if not np.isfinite(predictions).all():
        raise RuntimeError("distributed validation did not cover every requested date")

    def score(predictions: np.ndarray, suffix: str):
        frame = make_prediction_frame(
            panel=panel,
            date_indices=requested,
            predictions=predictions,
            eligible=eligibility,
            model="stock_time_transformer_b0",
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

    raw_metrics, raw_ewma = score(predictions, "raw")
    return {
        "raw_metrics": raw_metrics,
        "raw_ewma_metrics": raw_ewma,
    }


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
    train_dataset = MultiDateCrossSectionDataset(
        panel, train_indices,
        lookback=int(config["model"]["lookback"]),
        output_steps=int(config["model"]["output_steps"]),
        context_days=config["model"].get("context_days"),
        stride=int(data["train_stride"]),
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
    )
    validation_dataset = MultiDateCrossSectionDataset(
        panel, validation_indices,
        lookback=int(config["model"]["lookback"]),
        output_steps=int(config["model"]["output_steps"]),
        context_days=config["model"].get("context_days"),
        stride=int(config["model"]["output_steps"]),
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
    )
    if args.limit_train_blocks:
        train_dataset.output_blocks = train_dataset.output_blocks[-args.limit_train_blocks:]
    if args.limit_validation_blocks:
        validation_dataset.output_blocks = validation_dataset.output_blocks[:args.limit_validation_blocks]
        kept = np.unique(validation_dataset.output_blocks.reshape(-1))
        validation_dataset.requested_date_indices = kept

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError("distributed shard is empty")

    model = build_model(config, panel.shape[1]).to(device)
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
        "tuning", "stock-time-transformer-b0", route,
        int(config["model"]["lookback"]), seed,
        budget=f"k{config['model']['output_steps']}-ep{epochs}-ddp{world_size}",
    )
    tracker_context = swan_settings(
        config,
        name=name,
        tags=["tuning", "stock-time-transformer-b0", "multi-date", "final-global"],
        extra={
            "world_size": world_size,
            "epochs": epochs,
            "train_blocks": len(train_dataset),
            "dates_per_block": int(config["model"]["output_steps"]),
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
    started = time.perf_counter()
    log_every = int(config["swanlab"]["log_interval_updates"])

    try:
        with tracker_context as tracker:
            for epoch in range(1, epochs + 1):
                sampler.set_epoch(epoch)
                ddp_model.train()
                interval = np.zeros(9, dtype=np.float64)
                interval_count = 0
                interval_started = time.perf_counter()
                for step, item in enumerate(loader, start=1):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = ddp_model(
                        item["x"].to(device),
                        item["token_valid"].to(device),
                        item["eligible"].to(device),
                    )
                    components = multi_date_soft_components(
                        prediction,
                        item["target"].to(device).squeeze(0),
                        item["mask"].to(device).squeeze(0),
                        item["tradable"].to(device).squeeze(0),
                        rank_temperature=float(training["rank_temperature"]),
                        top_temperature=float(training["top_temperature"]),
                    )
                    global_rank = global_mean_with_local_gradient(components.rank_ic)
                    global_excess = global_mean_with_local_gradient(components.annual_excess_raw)
                    global_stability = global_mean_with_local_gradient(components.stability)
                    loss, soft_score, bounded_excess = compose_bounded_final_score(
                        components,
                        global_rank_ic=global_rank,
                        global_annual_excess_raw=global_excess,
                        global_stability=global_stability,
                        excess_bound=float(training["excess_bound"]),
                        mse_weight=float(training["mse_weight"]),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss epoch={epoch} step={step}")
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
                        float(global_rank.detach()), float(global_excess.detach()),
                        float(bounded_excess.detach()), float(global_stability.detach()),
                        float(components.mse.detach()), float(grad_norm), learning_rate,
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
                            tracker.log({
                                "train/loss": means[0],
                                "train/soft_final": means[1],
                                "train/soft_rank_ic": means[2],
                                "train/soft_annual_excess_raw": means[3],
                                "train/soft_annual_excess_bounded": means[4],
                                "train/soft_one_minus_turnover": means[5],
                                "train/mse_diagnostic": means[6],
                                "train/grad_norm": means[7],
                                "train/lr": means[8],
                                "train/epoch": epoch,
                                "train/optimizer_update": global_update,
                                "train/dates_per_global_update": (
                                    world_size * int(config["model"]["output_steps"])
                                ),
                                "train/updates_per_second": interval_count / max(
                                    time.perf_counter() - interval_started, 1e-9,
                                ),
                            }, step=global_update)
                        interval[:] = 0
                        interval_count = 0
                        interval_started = time.perf_counter()

                validation = distributed_validation(
                    model, validation_dataset, panel, device,
                    rank=rank, world_size=world_size, route=route,
                    ewma_alphas=ewma_alphas,
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
                    }
                    history.append(row)
                    window = history[-smoothing_epochs:]
                    stable = float(np.mean([
                        x["raw"]["final_score"] for x in window
                    ]))
                    stable_std = float(np.std([
                        x["raw"]["final_score"] for x in window
                    ]))
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
                    validation_logs = {
                        **{f"validation/raw/{key}": value for key, value in raw_summary.items()},
                        **ewma_logs,
                        "validation/stability/final_score_ma3": stable,
                        "validation/stability/final_score_window_std": stable_std,
                        "validation/epoch": epoch,
                        "validation/optimizer_update": global_update,
                    }
                    tracker.log(validation_logs, step=global_update)

                    raw_score = float(raw_summary["final_score"])
                    if raw_score > best_raw:
                        best_raw, best_raw_epoch = raw_score, epoch
                        save_checkpoint(
                            model, output / "best_raw",
                            {"epoch": epoch, "optimizer_update": global_update,
                             "validation_final_score": raw_score,
                             "score_components": raw_summary},
                            architecture=config["model"], vocabulary_hash=vocabulary_hash,
                        )
                    eligible_selection = len(window) == smoothing_epochs or epoch == epochs
                    if eligible_selection and stable > best_stable:
                        best_stable, best_endpoint = stable, raw_score
                        best_epoch, best_update = epoch, global_update
                        save_checkpoint(
                            model, output / "best",
                            {"epoch": epoch, "optimizer_update": global_update,
                             "validation_final_score": raw_score,
                             "stable_validation_final_score": stable,
                             "stable_validation_final_score_std": stable_std,
                             "parameter_source": "raw",
                             "score_components": raw_summary},
                            architecture=config["model"], vocabulary_hash=vocabulary_hash,
                        )
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
                    "lookback": int(config["model"]["lookback"]),
                    "context_days": int(model.context_days),
                    "temporal_window": int(model.temporal_window),
                    "output_steps": int(config["model"]["output_steps"]),
                    "sequence_length": model.sequence_length,
                    "dates_per_global_update": world_size * int(config["model"]["output_steps"]),
                    "training_objective": "multi_date_final_global_excess",
                    "checkpoint_selection": (
                        f"highest_{smoothing_epochs}_epoch_trailing_mean_"
                        "raw_final_score"
                    ),
                    "checkpoint_parameter_source": "raw",
                    "parameter_ema_enabled": False,
                    "best_epoch": best_epoch,
                    "best_optimizer_update": best_update,
                    "best_validation_final_score": best_endpoint,
                    "best_stable_validation_final_score": best_stable,
                    "best_checkpoint_score_components": selected["raw"],
                    "best_checkpoint_ewma_components": selected["raw_ewma"],
                    "best_raw_validation_final_score": best_raw,
                    "best_raw_epoch": best_raw_epoch,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "peak_memory_bytes_max_rank": int(peak.item()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "normalization": "per-token trailing-window causal z-score",
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
