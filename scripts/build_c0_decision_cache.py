#!/usr/bin/env python3
"""Compute frozen C0 hidden and raw score once for each decision date."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from numpy.lib.format import open_memmap

from finmodel.decision_cache import (
    CACHE_VERSION, CACHE_VERSION_DUAL, DecisionFeatureCache, cache_source_hashes,
)
from finmodel.io import atomic_json_dump, seed_everything
from finmodel.panel import Panel
from finmodel.pipeline import build_dataset, load_backbone
from finmodel.sft import load_config, panel_indices, sft_split


@torch.inference_mode()
def build_one(*, name, dataset, panel, backbone, output, checkpoint_hash, panel_hash, device):
    output = Path(output) / name
    if (output / "manifest.json").exists():
        DecisionFeatureCache.open(
            output, checkpoint_sha256=checkpoint_hash,
            panel_manifest_sha256=panel_hash,
        )
        print(f"REUSE {name}: {output}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"incomplete cache directory exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    requested = np.unique(dataset.output_blocks.reshape(-1)).astype(np.int64)
    if not np.all(np.diff(requested) == 1):
        raise ValueError(f"{name} output dates are not consecutive")
    lookup = {int(date): row for row, date in enumerate(requested)}
    owner_block = np.full(len(requested), -1, dtype=np.int32)
    owner_position = np.full(len(requested), -1, dtype=np.int16)
    for block, dates in enumerate(dataset.output_blocks):
        for position, date in enumerate(dates):
            row = lookup[int(date)]
            if position > owner_position[row]:
                owner_block[row] = block
                owner_position[row] = position
    if (owner_block < 0).any():
        raise RuntimeError("some decision dates have no C0 owner block")

    shape = (len(requested), panel.shape[1])
    hidden = open_memmap(output / "hidden.npy", mode="w+", dtype=np.float32,
                         shape=(*shape, backbone.d_model))
    base_score = open_memmap(output / "base_score.npy", mode="w+", dtype=np.float32,
                             shape=shape)
    predicted_return = (
        open_memmap(output / "predicted_return.npy", mode="w+", dtype=np.float32, shape=shape)
        if backbone.head_mode == "dual" else None
    )
    eligible = open_memmap(output / "eligible.npy", mode="w+", dtype=np.bool_, shape=shape)
    tradable = open_memmap(output / "tradable.npy", mode="w+", dtype=np.bool_, shape=shape)
    np.save(output / "date_indices.npy", requested)
    started = time.perf_counter()
    for block in range(len(dataset)):
        chosen = np.flatnonzero(owner_block == block)
        if not len(chosen):
            continue
        item = dataset[block]
        values = item["x"].to(device)
        token_valid = item["token_valid"].to(device)
        block_eligible = item["eligible"].to(device)
        block_hidden = backbone.encode_hidden(
            values, token_valid, block_eligible,
            long_memory=(item["long_memory"].to(device) if "long_memory" in item else None),
        )
        block_score = backbone.score_hidden(block_hidden, block_eligible)
        block_return = (
            backbone.predict_return_hidden(block_hidden, block_eligible)
            if predicted_return is not None else None
        )
        positions = owner_position[chosen].astype(np.int64)
        hidden[chosen] = block_hidden[positions].float().cpu().numpy()
        base_score[chosen] = block_score[positions].float().cpu().numpy()
        if predicted_return is not None:
            predicted_return[chosen] = block_return[positions].float().cpu().numpy()
        eligible[chosen] = item["eligible"][positions].numpy()
        tradable[chosen] = item["tradable"][positions].numpy()
        if (block + 1) % 5 == 0 or block + 1 == len(dataset):
            print(
                f"CACHE {name} block={block + 1}/{len(dataset)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    for array in (hidden, base_score, eligible, tradable):
        array.flush()
    if predicted_return is not None:
        predicted_return.flush()
    atomic_json_dump({
        "cache_version": CACHE_VERSION_DUAL if predicted_return is not None else CACHE_VERSION,
        "split": name,
        "backbone_sha256": checkpoint_hash,
        "panel_manifest_sha256": panel_hash,
        "hidden_shape": list(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "predicted_return_units": "decimal_absolute_next_day" if predicted_return is not None else None,
        "head_mode": backbone.head_mode,
        "long_memory_scales": list(backbone.long_memory_scales),
        "date_start": int(panel.dates[requested[0]]),
        "date_end": int(panel.dates[requested[-1]]),
        "dates": len(requested),
        "stocks": panel.shape[1],
        "owner_selection": "largest_causal_position_within_overlapping_C0_windows",
        "elapsed_seconds": time.perf_counter() - started,
    }, output / "manifest.json")
    print(f"COMPLETE {name}: {len(requested)} dates at {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d0_decision_grpo.json")
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--output", default="artifacts/d0/cache")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    panel = Panel.open(args.panel)
    split = sft_split(panel, config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone, checkpoint_hash = load_backbone(config, panel, device)
    cache_hash, panel_hash = cache_source_hashes(panel, config["backbone_checkpoint"])
    if cache_hash != checkpoint_hash:
        raise AssertionError("C0 checkpoint hash changed while loading")
    datasets = {
        "train": build_dataset(
            panel, panel_indices(panel, split.training_dates), config,
            stride=int(config["data"]["train_stride"]),
        ),
        "validation": build_dataset(
            panel, panel_indices(panel, split.validation_dates), config,
            stride=int(config["model"]["output_steps"]),
        ),
    }
    for name, dataset in datasets.items():
        build_one(
            name=name, dataset=dataset, panel=panel, backbone=backbone,
            output=args.output, checkpoint_hash=checkpoint_hash,
            panel_hash=panel_hash, device=device,
        )


if __name__ == "__main__":
    main()
