#!/usr/bin/env python3
"""Current StockMixer supervised training with a differentiable competition score."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from finmodel.datasets import ConsecutiveCrossSectionDataset, CrossSectionDataset
from finmodel.io import atomic_json_dump, seed_everything
from finmodel.losses import stockmixer_official_loss, stockmixer_soft_score_loss
from finmodel.metrics import evaluate_frame, make_prediction_frame
from finmodel.models.stockmixer import StockMixerReturn
from finmodel.panel import Panel
from finmodel.sft import (
    cosine_learning_rate, update_ema_model, job_name, load_config,
    metric_summary, panel_indices, sft_split, reset_peak_memory,
    save_torch_checkpoint, swan_settings,
)


class PairModel(nn.Module):
    """Keep both adjacent-date forwards inside one DDP forward call."""

    def __init__(self, stockmixer: StockMixerReturn) -> None:
        super().__init__()
        self.stockmixer = stockmixer

    def forward(
        self,
        previous_x: torch.Tensor,
        previous_eligible: torch.Tensor,
        current_x: torch.Tensor,
        current_eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.stockmixer(previous_x, previous_eligible),
            self.stockmixer(current_x, current_eligible),
        )


class NullTracker:
    def log(self, values, *, step=None) -> None:
        del values, step


@torch.inference_mode()
def distributed_validation(
    raw_model: StockMixerReturn,
    ema_model: StockMixerReturn,
    dataset: CrossSectionDataset,
    panel: Panel,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    rank_weight: float,
) -> dict[str, Any] | None:
    raw_model.eval()
    ema_model.eval()
    stocks = panel.shape[1]
    positions = list(range(rank, len(dataset), world_size))
    max_count = (len(dataset) + world_size - 1) // world_size
    raw_pad = torch.full((max_count, stocks), float("nan"), device=device)
    ema_pad = torch.full((max_count, stocks), float("nan"), device=device)
    eligible_pad = torch.zeros((max_count, stocks), dtype=torch.uint8, device=device)
    position_pad = torch.full((max_count,), -1, dtype=torch.int64, device=device)
    local_sums = torch.zeros(7, dtype=torch.float64, device=device)

    for slot, position in enumerate(positions):
        item = dataset[position]
        values = item["x"].to(device)
        eligible = item["eligible"].to(device)
        target = item["target"].to(device)
        mask = item["mask"].to(device)
        raw_prediction = raw_model(values, eligible)
        ema_prediction = ema_model(values, eligible)
        raw_total, raw_mse, raw_rank = stockmixer_official_loss(
            raw_prediction, target, mask, rank_weight=rank_weight,
        )
        ema_total, ema_mse, ema_rank = stockmixer_official_loss(
            ema_prediction, target, mask, rank_weight=rank_weight,
        )
        raw_pad[slot] = raw_prediction
        ema_pad[slot] = ema_prediction
        eligible_pad[slot] = eligible.to(torch.uint8)
        position_pad[slot] = position
        local_sums[:6] += torch.stack([
            raw_total, raw_mse, raw_rank, ema_total, ema_mse, ema_rank,
        ]).to(dtype=torch.float64)
        local_sums[6] += 1.0

    gathered_raw = [torch.empty_like(raw_pad) for _ in range(world_size)]
    gathered_ema = [torch.empty_like(ema_pad) for _ in range(world_size)]
    gathered_eligible = [torch.empty_like(eligible_pad) for _ in range(world_size)]
    gathered_position = [torch.empty_like(position_pad) for _ in range(world_size)]
    dist.all_gather(gathered_raw, raw_pad)
    dist.all_gather(gathered_ema, ema_pad)
    dist.all_gather(gathered_eligible, eligible_pad)
    dist.all_gather(gathered_position, position_pad)
    dist.all_reduce(local_sums, op=dist.ReduceOp.SUM)
    if rank != 0:
        return None

    raw_predictions = np.empty((len(dataset), stocks), dtype=np.float32)
    ema_predictions = np.empty_like(raw_predictions)
    eligibility = np.empty((len(dataset), stocks), dtype=bool)
    for raw_values, ema_values, eligible_values, position_values in zip(
        gathered_raw, gathered_ema, gathered_eligible, gathered_position,
    ):
        for slot, position in enumerate(position_values.cpu().tolist()):
            if position < 0:
                continue
            raw_predictions[position] = raw_values[slot].cpu().numpy()
            ema_predictions[position] = ema_values[slot].cpu().numpy()
            eligibility[position] = eligible_values[slot].cpu().numpy().astype(bool)

    def score(predictions: np.ndarray, route: str) -> dict[str, Any]:
        frame = make_prediction_frame(
            panel=panel, date_indices=dataset.date_indices,
            predictions=predictions, eligible=eligibility,
            model="stockmixer", route=route, fold="validation", alpha=1.0,
        )
        return evaluate_frame(frame, "pred_raw")

    divisor = max(float(local_sums[6].item()), 1.0)
    return {
        "raw_metrics": score(raw_predictions, "sft_soft_final_raw"),
        "ema_metrics": score(ema_predictions, "sft_soft_final_ema"),
        "raw_losses": [float(value) / divisor for value in local_sums[:3].cpu().tolist()],
        "ema_losses": [float(value) / divisor for value in local_sums[3:6].cpu().tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stockmixer_sft.json")
    parser.add_argument("--panel", default="artifacts/panel/train")
    parser.add_argument("--lookback", type=int, choices=[16, 32, 64, 128], required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-train-dates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--limit-validation-dates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-swanlab", action="store_true",
        help="disable tracking for local smoke tests",
    )
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun so LOCAL_RANK is defined")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    config = load_config(args.config)
    soft_config = config["stockmixer"]["sft"]
    max_epochs = int(config["stockmixer"]["max_epochs"])
    epochs = int(args.epochs or max_epochs)
    if not 1 <= epochs <= max_epochs:
        raise ValueError(
            f"epochs must be between 1 and the configured SFT cap ({max_epochs}), got {epochs}"
        )
    seed = int(config["seed"])
    seed_everything(seed)
    panel = Panel.open(args.panel)
    split = sft_split(panel, config)
    training_dates = panel_indices(panel, split.training_dates)
    validation_dates = panel_indices(panel, split.validation_dates)
    if args.limit_train_dates:
        training_dates = training_dates[-args.limit_train_dates:]
    if args.limit_validation_dates:
        validation_dates = validation_dates[:args.limit_validation_dates]
    train_dataset = ConsecutiveCrossSectionDataset(
        panel, training_dates, lookback=args.lookback, min_history=int(config["min_history"]),
    )
    validation_dataset = CrossSectionDataset(
        panel, validation_dates, lookback=args.lookback, min_history=int(config["min_history"]),
    )
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed, drop_last=True,
    )
    loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError("distributed shard is empty; increase training dates")

    stockmixer = StockMixerReturn(stocks=panel.shape[1], lookback=args.lookback).to(device)
    pair_model = PairModel(stockmixer).to(device)
    ddp_model = DDP(
        pair_model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    )
    ema_model = copy.deepcopy(stockmixer).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=float(config["stockmixer"]["learning_rate"]),
        weight_decay=float(soft_config["weight_decay"]),
    )
    updates_per_epoch = len(loader)
    total_updates = epochs * updates_per_epoch
    warmup_epochs = int(soft_config["warmup_epochs"])
    warmup_updates = warmup_epochs * updates_per_epoch
    ema_decay = float(soft_config["ema_decay"])
    smoothing_epochs = int(soft_config["validation_smoothing_epochs"])
    route = str(soft_config["route_name"])
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    reset_peak_memory(device)

    name = job_name(
        "tuning", "stockmixer", route, args.lookback, seed,
        budget=f"ep{epochs}-ddp{world_size}",
    )
    tracker_context = swan_settings(
        config, name=name,
        tags=["tuning", "stockmixer", "soft-final", f"lb{args.lookback}", f"ddp{world_size}"],
        extra={
            "lookback": args.lookback, "epochs": epochs, "world_size": world_size,
            "local_date_pairs_per_update": 1, "global_date_pairs_per_update": world_size,
            "objective": "bounded_differentiable_final_score",
            "selection": "highest_5_epoch_trailing_mean_exact_ema_final_score",
            **soft_config,
        },
    ) if rank == 0 and not args.disable_swanlab else nullcontext(NullTracker())

    started = time.perf_counter()
    global_update = 0
    best_stable_score = -float("inf")
    best_endpoint_score = -float("inf")
    best_epoch = 0
    best_update = 0
    best_raw_score = -float("inf")
    best_raw_epoch = 0
    best_ema_score = -float("inf")
    best_ema_epoch = 0
    validation_history: list[dict[str, Any]] = []
    log_every = int(config["swanlab"]["log_interval_updates"])

    try:
        with tracker_context as tracker:
            dist.barrier()
            for epoch in range(1, epochs + 1):
                sampler.set_epoch(epoch)
                ddp_model.train()
                interval = np.zeros(7, dtype=np.float64)
                interval_count = 0
                interval_started = time.perf_counter()
                for step, item in enumerate(loader, start=1):
                    optimizer.zero_grad(set_to_none=True)
                    previous_prediction, current_prediction = ddp_model(
                        item["previous_x"].to(device), item["previous_eligible"].to(device),
                        item["current_x"].to(device), item["current_eligible"].to(device),
                    )
                    soft = stockmixer_soft_score_loss(
                        previous_prediction, current_prediction,
                        item["previous_target"].to(device).reshape(-1),
                        item["current_target"].to(device).reshape(-1),
                        item["previous_mask"].to(device).reshape(-1),
                        item["current_mask"].to(device).reshape(-1),
                        item["previous_tradable"].to(device).reshape(-1),
                        item["current_tradable"].to(device).reshape(-1),
                        rank_temperature=float(soft_config["rank_temperature"]),
                        top_temperature=float(soft_config["top_temperature"]),
                        excess_scale=float(soft_config["excess_bound"]),
                        mse_weight=float(soft_config["mse_weight"]),
                    )
                    if not torch.isfinite(soft.loss):
                        raise FloatingPointError(f"non-finite soft-final loss at epoch={epoch} step={step}")
                    soft.loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                    next_update = global_update + 1
                    learning_rate = cosine_learning_rate(
                        float(config["stockmixer"]["learning_rate"]),
                        update=next_update, total_updates=total_updates,
                        warmup_updates=warmup_updates,
                        eta_min_ratio=float(soft_config["cosine_eta_min_ratio"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    optimizer.step()
                    decay = ema_decay if next_update > warmup_updates else 0.0
                    update_ema_model(ema_model, stockmixer, decay)
                    global_update = next_update
                    values = np.asarray([
                        float(soft.loss.detach()), float(soft.score.detach()),
                        float(soft.rank_ic.detach()), float(soft.annual_excess_raw.detach()),
                        float(soft.annual_excess_objective.detach()),
                        float(soft.stability.detach()), float(soft.mse_anchor.detach()),
                    ])
                    interval += values
                    interval_count += 1

                    if interval_count >= log_every or step == len(loader):
                        reduced = torch.tensor(
                            [*interval.tolist(), float(interval_count)],
                            dtype=torch.float64, device=device,
                        )
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        denominator = max(float(reduced[-1]), 1.0)
                        means = (reduced[:-1] / denominator).cpu().tolist()
                        grad_value = torch.tensor(float(grad_norm), device=device)
                        dist.all_reduce(grad_value, op=dist.ReduceOp.SUM)
                        if rank == 0:
                            tracker.log({
                                "train/loss": means[0],
                                "train/soft_final": means[1],
                                "train/soft_rank_ic": means[2],
                                "train/soft_annual_excess_raw": means[3],
                                "train/soft_annual_excess_bounded": means[4],
                                "train/soft_one_minus_turnover": means[5],
                                "train/mse_anchor": means[6],
                                "train/weighted_rank_ic": 0.4 * means[2],
                                "train/weighted_annual_excess": 0.3 * means[4],
                                "train/weighted_stability": 0.3 * means[5],
                                "train/lr": learning_rate,
                                "train/grad_norm": float(grad_value) / world_size,
                                "train/epoch": epoch,
                                "train/epoch_progress": step / len(loader),
                                "train/optimizer_update": global_update,
                                "train/global_date_pairs_per_update": world_size,
                                "train/updates_per_second": interval_count / max(time.perf_counter() - interval_started, 1e-9),
                            }, step=global_update)
                        interval[:] = 0
                        interval_count = 0
                        interval_started = time.perf_counter()

                validation = distributed_validation(
                    stockmixer, ema_model, validation_dataset, panel, device,
                    rank=rank, world_size=world_size,
                    rank_weight=float(config["stockmixer"]["rank_weight"]),
                )
                if rank == 0:
                    assert validation is not None
                    raw_metrics = validation["raw_metrics"]
                    ema_metrics = validation["ema_metrics"]
                    raw_summary = metric_summary(raw_metrics)
                    ema_summary = metric_summary(ema_metrics)
                    history_row = {
                        "epoch": epoch, "optimizer_update": global_update,
                        "raw": raw_summary, "ema": ema_summary,
                        "raw_losses": validation["raw_losses"],
                        "ema_losses": validation["ema_losses"],
                    }
                    validation_history.append(history_row)
                    window = validation_history[-smoothing_epochs:]
                    stable_score = float(np.mean([row["ema"]["final_score"] for row in window]))
                    stable_std = float(np.std([row["ema"]["final_score"] for row in window]))
                    history_row.update({
                        "stable_final_score_mean": stable_score,
                        "stable_final_score_std": stable_std,
                        "stable_window_size": len(window),
                    })
                    atomic_json_dump(validation_history, output / "validation_history.json")
                    tracker.log({
                        **{f"validation/ema/{key}": value for key, value in ema_summary.items()},
                        **{f"validation/raw/{key}": value for key, value in raw_summary.items()},
                        "validation/stability/final_score_ma5": stable_score,
                        "validation/stability/final_score_window_std": stable_std,
                        "validation/ema/loss_total": validation["ema_losses"][0],
                        "validation/ema/loss_mse": validation["ema_losses"][1],
                        "validation/ema/loss_rank": validation["ema_losses"][2],
                        "validation/raw/loss_total": validation["raw_losses"][0],
                        "validation/raw/loss_mse": validation["raw_losses"][1],
                        "validation/raw/loss_rank": validation["raw_losses"][2],
                        "validation/epoch": epoch,
                        "validation/optimizer_update": global_update,
                    }, step=global_update)

                    raw_score = float(raw_summary["final_score"])
                    ema_score = float(ema_summary["final_score"])
                    if raw_score > best_raw_score:
                        best_raw_score, best_raw_epoch = raw_score, epoch
                        save_torch_checkpoint(stockmixer, output / "best_raw", {
                            "epoch": epoch, "optimizer_update": global_update,
                            "validation_final_score": raw_score,
                        })
                    if ema_score > best_ema_score:
                        best_ema_score, best_ema_epoch = ema_score, epoch
                        save_torch_checkpoint(ema_model, output / "best_ema", {
                            "epoch": epoch, "optimizer_update": global_update,
                            "validation_final_score": ema_score,
                        })
                    eligible_for_selection = len(window) == smoothing_epochs or epoch == epochs
                    if eligible_for_selection and stable_score > best_stable_score:
                        best_stable_score, best_endpoint_score = stable_score, ema_score
                        best_epoch, best_update = epoch, global_update
                        save_torch_checkpoint(ema_model, output / "best", {
                            "epoch": epoch, "optimizer_update": global_update,
                            "validation_final_score": ema_score,
                            "stable_validation_final_score": stable_score,
                            "stable_validation_final_score_std": stable_std,
                            "stable_window_size": len(window),
                            "score_components": ema_summary,
                            "parameter_source": "ema",
                            "selection_metric": "exact_final_score",
                        })
                dist.barrier()

            peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                last_ten = [float(row["ema"]["final_score"]) for row in validation_history[-10:]]
                summary = {
                    "experiment_protocol": config["protocol"],
                    "lookback": args.lookback,
                    "world_size": world_size,
                    "epochs_trained": epochs,
                    "optimizer_updates": global_update,
                    "updates_per_epoch": updates_per_epoch,
                    "global_date_pairs_per_optimizer_update": world_size,
                    "optimizer": "adamw", "learning_rate": float(config["stockmixer"]["learning_rate"]),
                    "weight_decay": float(soft_config["weight_decay"]),
                    "lr_scheduler": "linear_warmup_then_cosine",
                    "warmup_epochs": warmup_epochs,
                    "ema_decay": ema_decay,
                    "soft_rank_temperature": float(soft_config["rank_temperature"]),
                    "soft_top_temperature": float(soft_config["top_temperature"]),
                    "soft_excess_bound": float(soft_config["excess_bound"]),
                    "soft_mse_weight": float(soft_config["mse_weight"]),
                    "checkpoint_selection": "highest_5_epoch_trailing_mean_exact_ema_final_score",
                    "best_epoch": best_epoch, "best_optimizer_update": best_update,
                    "best_validation_final_score": best_endpoint_score,
                    "best_stable_validation_final_score": best_stable_score,
                    "best_raw_validation_final_score": best_raw_score,
                    "best_raw_epoch": best_raw_epoch,
                    "best_ema_validation_final_score": best_ema_score,
                    "best_ema_epoch": best_ema_epoch,
                    "last_10_ema_final_score_mean": float(np.mean(last_ten)),
                    "last_10_ema_final_score_std": float(np.std(last_ten)),
                    "parameters": sum(parameter.numel() for parameter in stockmixer.parameters()),
                    "peak_memory_bytes_max_rank": int(peak.item()),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                atomic_json_dump(summary, output / "train_summary.json")
                atomic_json_dump({"complete": True, "summary": summary}, output / "complete.json")
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
