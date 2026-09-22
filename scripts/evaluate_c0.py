#!/usr/bin/env python3
"""Evaluate the frozen FinAxial C0 seeds on the complete labelled test period."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from finmodel.io import atomic_json_dump, sha256_file, torch_environment
from finmodel.metrics import evaluate_frame, make_prediction_frame
from finmodel.models import StockTimeTransformer, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.sequence import MultiDateCrossSectionDataset
from finmodel.sft import load_config, metric_summary


def evaluation_indices(panel: Panel, start: int, end: int, expected: int) -> np.ndarray:
    dates = np.asarray(panel.dates, dtype=np.int64)
    indices = np.flatnonzero((dates >= int(start)) & (dates <= int(end))).astype(np.int64)
    if len(indices) != int(expected):
        raise ValueError(f"expected {expected} evaluation dates, found {len(indices)}")
    if not np.all(np.diff(indices) == 1):
        raise ValueError("evaluation dates must be consecutive panel indices")
    return indices


def dataset_key(config: dict[str, Any]) -> tuple[Any, ...]:
    model, data = config["model"], config["data"]
    return (
        int(model["lookback"]), int(model["output_steps"]),
        int(model.get("context_days", int(model["lookback"]) - 1)),
        int(config["min_history"]), float(data["normalization_epsilon"]),
        float(data["normalization_clip"]), str(data.get("feature_mode", "temporal")),
    )


def build_dataset(
    panel: Panel, indices: np.ndarray, config: dict[str, Any],
) -> MultiDateCrossSectionDataset:
    model, data = config["model"], config["data"]
    dataset = MultiDateCrossSectionDataset(
        panel, indices,
        lookback=int(model["lookback"]),
        output_steps=int(model["output_steps"]),
        context_days=model.get("context_days"),
        stride=int(model["output_steps"]),
        min_history=int(config["min_history"]),
        epsilon=float(data["normalization_epsilon"]),
        clip=float(data["normalization_clip"]),
        feature_mode=str(data.get("feature_mode", "temporal")),
    )
    if dataset.channels != int(model["channels"]):
        raise ValueError("dataset and model channel counts differ")
    return dataset


@torch.inference_mode()
def predict(
    panel: Panel,
    dataset: MultiDateCrossSectionDataset,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = StockTimeTransformer(stocks=panel.shape[1], **config["model"]).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    requested = np.asarray(dataset.requested_date_indices, dtype=np.int64)
    lookup = {int(date): offset for offset, date in enumerate(requested)}
    predictions = np.full((len(requested), panel.shape[1]), np.nan, dtype=np.float32)
    eligibility = np.zeros((len(requested), panel.shape[1]), dtype=bool)
    selected_position = np.full(len(requested), -1, dtype=np.int16)
    for block in range(len(dataset)):
        item = dataset[block]
        output = model(
            item["x"].to(device),
            item["token_valid"].to(device),
            item["eligible"].to(device),
        ).float().cpu().numpy()
        for position, date_idx in enumerate(item["date_indices"].tolist()):
            offset = lookup.get(int(date_idx))
            if offset is None or position <= selected_position[offset]:
                continue
            predictions[offset] = output[position]
            eligibility[offset] = item["eligible"][position].numpy().astype(bool)
            selected_position[offset] = position
    if not np.isfinite(predictions).all() or (selected_position < 0).any():
        raise RuntimeError("evaluation inference did not cover every requested date")
    del model, state
    torch.cuda.empty_cache()
    return predictions, eligibility


def score_predictions(
    panel: Panel,
    indices: np.ndarray,
    predictions: np.ndarray,
    eligibility: np.ndarray,
    *,
    model: str,
    route: str,
    alpha: float,
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    frame = make_prediction_frame(
        panel=panel, date_indices=indices, predictions=predictions,
        eligible=eligibility, model=model, route=route,
        fold="full_labelled_test", alpha=alpha,
    )
    raw = metric_summary(evaluate_frame(frame, "pred_raw"))
    ewma = metric_summary(evaluate_frame(frame, "pred_smoothed"))
    ranks = frame["pred_rank"].to_numpy(dtype=np.float32).reshape(
        len(indices), panel.shape[1],
    )
    del frame
    gc.collect()
    return raw, ewma, ranks


def aggregate_seed_metrics(members: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metric_names = (
        "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
        "mse", "mae", "icir", "ic_positive_ratio", "coverage",
    )
    result: dict[str, Any] = {}
    for metric in metric_names:
        values = np.asarray([member[key][metric] for member in members], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/c0_evaluation.json")
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--output", default="artifacts/finaxial_c0/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-rerun", action="store_true",
        help="Explicitly allow overwriting an existing C0 evaluation result.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    manifest_path = Path(args.manifest)
    manifest = load_config(manifest_path)
    if manifest.get("protocol") != "c0_full_test_v1":
        raise ValueError("refusing to evaluate a manifest not marked as C0 full test")
    panel = Panel.open(args.panel)
    indices = evaluation_indices(
        panel,
        manifest["evaluation_start"],
        manifest["evaluation_end"],
        manifest["evaluation_days"],
    )
    alpha = float(manifest["signal_ewma_alpha"])
    output = Path(args.output)
    result_path = output / "evaluation_results.json"
    if result_path.exists() and not args.allow_rerun:
        raise FileExistsError(
            f"C0 full test has already been evaluated: {result_path}; "
            "use --allow-rerun only for an intentional audit"
        )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary_hash = stock_vocab_sha256(panel.codes)
    dataset_cache: dict[tuple[Any, ...], MultiDateCrossSectionDataset] = {}
    group_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for group in manifest["groups"]:
        group_name = str(group["name"])
        members: list[dict[str, Any]] = []
        rank_predictions, raw_predictions = [], []
        group_eligibility: np.ndarray | None = None
        for member in group["members"]:
            seed = int(member["seed"])
            config_path = Path(member["config"])
            checkpoint = Path(member["checkpoint"])
            config = load_config(config_path)
            metadata = load_config(Path(member["metadata"]))
            if metadata.get("stock_vocab_sha256") != vocabulary_hash:
                raise ValueError(f"stock vocabulary mismatch: {checkpoint}")
            key = dataset_key(config)
            if key not in dataset_cache:
                dataset_cache[key] = build_dataset(panel, indices, config)
            dataset = dataset_cache[key]
            predictions, eligibility = predict(panel, dataset, config, checkpoint, device)
            if group_eligibility is None:
                group_eligibility = eligibility
            elif not np.array_equal(group_eligibility, eligibility):
                raise RuntimeError(f"eligibility differs within group {group_name}")
            raw, ewma, ranks = score_predictions(
                panel, indices, predictions, eligibility,
                model=group_name, route=f"seed_{seed}", alpha=alpha,
            )
            record = {
                "seed": seed,
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "raw": raw,
                "ewma_alpha_0.25": ewma,
            }
            members.append(record)
            rank_predictions.append(ranks)
            raw_predictions.append(predictions)
            csv_rows.append({
                "group": group_name, "member": f"seed_{seed}",
                **{f"raw_{name}": raw[name] for name in (
                    "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
                )},
                **{f"ewma_{name}": ewma[name] for name in (
                    "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
                )},
            })
            print(
                f"{group_name} seed={seed} raw_final={raw['final_score']:.6f} "
                f"ic={raw['rank_ic']:.6f} excess={raw['annual_excess']:.6f} "
                f"stability={raw['one_minus_turnover']:.6f}", flush=True,
            )

        assert group_eligibility is not None
        ensemble_predictions = np.mean(np.stack(rank_predictions), axis=0, dtype=np.float64)
        ensemble_raw, ensemble_ewma, _ = score_predictions(
            panel, indices, ensemble_predictions.astype(np.float32), group_eligibility,
            model=group_name, route="three_seed_equal_rank_ensemble", alpha=alpha,
        )
        csv_rows.append({
            "group": group_name, "member": "three_seed_equal_rank_ensemble",
            **{f"raw_{name}": ensemble_raw[name] for name in (
                "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
            )},
            **{f"ewma_{name}": ensemble_ewma[name] for name in (
                "final_score", "rank_ic", "annual_excess", "one_minus_turnover",
            )},
        })
        group_directory = output / group_name
        group_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            group_directory / "predictions.npz",
            date_indices=indices,
            dates=np.asarray(panel.dates[indices]),
            eligible=group_eligibility,
            seed_2026=raw_predictions[0],
            seed_2027=raw_predictions[1],
            seed_2028=raw_predictions[2],
            ensemble_equal_rank=ensemble_predictions.astype(np.float32),
        )
        result = {
            "name": group_name,
            "members": members,
            "seed_raw_summary": aggregate_seed_metrics(members, "raw"),
            "seed_ewma_summary": aggregate_seed_metrics(members, "ewma_alpha_0.25"),
            "equal_rank_ensemble": {
                "raw": ensemble_raw,
                "ewma_alpha_0.25": ensemble_ewma,
            },
        }
        group_results.append(result)
        atomic_json_dump(result, group_directory / "metrics.json")
        print(
            f"{group_name} ensemble raw_final={ensemble_raw['final_score']:.6f}", flush=True,
        )

    result = {
        "protocol": manifest["protocol"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "panel": str(Path(args.panel)),
        "panel_manifest_sha256": sha256_file(Path(args.panel) / "manifest.json"),
        "evaluation_start": int(manifest["evaluation_start"]),
        "evaluation_end": int(manifest["evaluation_end"]),
        "evaluation_days": len(indices),
        "signal_ewma_alpha": alpha,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": torch_environment(),
        "groups": group_results,
    }
    atomic_json_dump(result, result_path)
    columns = [
        "group", "member", "raw_final_score", "raw_rank_ic", "raw_annual_excess",
        "raw_one_minus_turnover", "ewma_final_score", "ewma_rank_ic",
        "ewma_annual_excess", "ewma_one_minus_turnover",
    ]
    with (output / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
