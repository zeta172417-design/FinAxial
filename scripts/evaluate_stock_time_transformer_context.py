#!/usr/bin/env python3
"""Compare one checkpoint at different causal output positions on identical dates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from finmodel.io import atomic_json_dump
from finmodel.metrics import add_causal_ewma, evaluate_frame, make_prediction_frame
from finmodel.panel import Panel
from finmodel.sequence import MultiDateCrossSectionDataset
from finmodel.sft import load_config, metric_summary, panel_indices, sft_split
from scripts.train_stock_time_transformer import build_model


def parse_positions(value: str) -> tuple[int, ...]:
    positions = tuple(int(item) for item in value.split(",") if item.strip())
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("positions must be a non-empty unique comma-separated list")
    return positions


def eligibility_for_dates(
    panel: Panel, date_indices: np.ndarray, *, lookback: int, min_history: int,
) -> np.ndarray:
    feature_valid = np.asarray(panel.feature_valid, dtype=bool)
    cumulative = np.cumsum(feature_valid, axis=0, dtype=np.int32)
    rolling_count = cumulative.copy()
    if lookback < len(rolling_count):
        rolling_count[lookback:] -= cumulative[:-lookback]
    eligible = (
        feature_valid
        & (rolling_count >= int(lookback))
        & (cumulative >= int(min_history))
    )
    return eligible[date_indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", default="0,7,15,23,31,47,63")
    parser.add_argument("--signal-ewma-alpha", type=float, default=0.25)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch context evaluation with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("FINMODEL_DDP_TIMEOUT_SECONDS", "1800"))),
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    started = time.perf_counter()

    try:
        config = load_config(args.config)
        panel = Panel.open(args.panel)
        split = sft_split(panel, config)
        validation_indices = panel_indices(panel, split.validation_dates)
        architecture = config["model"]
        positions = parse_positions(args.positions)
        output_steps = int(architecture["output_steps"])
        if min(positions) < 0 or max(positions) >= output_steps:
            raise ValueError(f"positions must be in [0, {output_steps - 1}]")

        dataset = MultiDateCrossSectionDataset(
            panel,
            validation_indices,
            lookback=int(architecture["lookback"]),
            context_days=architecture.get("context_days"),
            output_steps=output_steps,
            stride=1,
            min_history=int(config["min_history"]),
            epsilon=float(config["data"]["normalization_epsilon"]),
            clip=float(config["data"]["normalization_clip"]),
        )
        model = build_model(config, panel.shape[1]).to(device)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        assigned = list(range(rank, len(dataset), world_size))
        max_count = (len(dataset) + world_size - 1) // world_size
        stocks = panel.shape[1]
        prediction_pad = torch.full(
            (max_count, output_steps, stocks), float("nan"),
            dtype=torch.float32, device=device,
        )
        block_pad = torch.full((max_count,), -1, dtype=torch.int64, device=device)
        with torch.inference_mode():
            for slot, block_index in enumerate(assigned):
                item = dataset[block_index]
                prediction_pad[slot] = model(
                    item["x"].to(device),
                    item["token_valid"].to(device),
                    item["eligible"].to(device),
                ).float()
                block_pad[slot] = block_index

        gathered_predictions = [torch.empty_like(prediction_pad) for _ in range(world_size)]
        gathered_blocks = [torch.empty_like(block_pad) for _ in range(world_size)]
        dist.all_gather(gathered_predictions, prediction_pad)
        dist.all_gather(gathered_blocks, block_pad)

        if rank == 0:
            by_position: dict[int, dict[int, np.ndarray]] = {
                position: {} for position in positions
            }
            output_blocks = np.asarray(dataset.output_blocks, dtype=np.int64)
            for predictions, block_indices in zip(gathered_predictions, gathered_blocks):
                predictions_cpu = predictions.cpu().numpy()
                for slot, block_index in enumerate(block_indices.cpu().tolist()):
                    if block_index < 0:
                        continue
                    for position in positions:
                        date_index = int(output_blocks[block_index, position])
                        by_position[position][date_index] = predictions_cpu[slot, position].copy()

            common_dates = sorted(set.intersection(*(
                set(values) for values in by_position.values()
            )))
            common_indices = np.asarray(common_dates, dtype=np.int64)
            if len(common_indices) < 30 or not np.all(np.diff(common_indices) == 1):
                raise RuntimeError("common position evaluation dates are insufficient or non-consecutive")
            eligible = eligibility_for_dates(
                panel, common_indices,
                lookback=int(architecture["lookback"]),
                min_history=int(config["min_history"]),
            )

            rows, details = [], {}
            context_days = int(architecture.get("context_days", int(architecture["lookback"]) - 1))
            temporal_window = int(architecture.get("temporal_window", architecture["lookback"]))
            alpha = float(args.signal_ewma_alpha)
            for position in positions:
                predictions = np.stack([
                    by_position[position][int(date)] for date in common_indices
                ])
                frame = make_prediction_frame(
                    panel=panel,
                    date_indices=common_indices,
                    predictions=predictions,
                    eligible=eligible,
                    model="stock_time_transformer_b0",
                    route=f"context_position_{position}",
                    fold="common_validation_dates",
                    alpha=1.0,
                )
                raw = metric_summary(evaluate_frame(frame, "pred_raw"))
                frame["pred_smoothed"] = add_causal_ewma(frame, alpha, source="pred_rank")
                smoothed = metric_summary(evaluate_frame(frame, "pred_smoothed"))
                prefix_tokens = context_days + position + 1
                descriptor = {
                    "output_position": position,
                    "prior_input_days": context_days + position,
                    "prefix_tokens_including_current": prefix_tokens,
                    "direct_attention_tokens": min(temporal_window, prefix_tokens),
                    "raw": raw,
                    f"signal_ewma_alpha_{alpha:g}": smoothed,
                }
                details[str(position)] = descriptor
                rows.append({
                    "output_position": position,
                    "prior_input_days": context_days + position,
                    "prefix_tokens_including_current": prefix_tokens,
                    "direct_attention_tokens": min(temporal_window, prefix_tokens),
                    **{f"raw_{key}": raw[key] for key in (
                        "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
                        "mse", "mae", "icir", "ic_positive_ratio", "coverage",
                    )},
                    **{f"ewma_{key}": smoothed[key] for key in (
                        "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
                    )},
                })

            result = {
                "protocol": "same-date-causal-context-position-evaluation",
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "world_size": world_size,
                "validation_blocks_stride_1": len(dataset),
                "common_validation_days": len(common_indices),
                "common_date_start": int(panel.dates[common_indices[0]]),
                "common_date_end": int(panel.dates[common_indices[-1]]),
                "positions": list(positions),
                "signal_ewma_alpha": alpha,
                "normalization_lookback": int(architecture["lookback"]),
                "temporal_window": temporal_window,
                "context_days": context_days,
                "elapsed_seconds": time.perf_counter() - started,
                "results": details,
            }
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(result, output / "context_position_metrics.json")
            pd.DataFrame(rows).to_csv(output / "context_position_metrics.csv", index=False)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
