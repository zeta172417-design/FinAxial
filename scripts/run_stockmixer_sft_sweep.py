#!/usr/bin/env python3
"""Run the current StockMixer SFT lookbacks sequentially through DDP."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_stockmixer_sft.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stockmixer_sft.json")
    parser.add_argument("--panel", default="artifacts/panel/train")
    parser.add_argument("--nproc-per-node", type=int, default=16)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--root", default="artifacts/stockmixer_sft_runs")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--limit-train-dates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--limit-validation-dates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-swanlab", action="store_true",
        help="disable tracking for local smoke tests",
    )
    args = parser.parse_args()
    with (ROOT / args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    max_epochs = int(config["stockmixer"]["max_epochs"])
    epochs = int(args.epochs or max_epochs)
    if not 1 <= epochs <= max_epochs:
        raise ValueError(
            f"epochs must be between 1 and the configured SFT cap ({max_epochs}), got {epochs}"
        )
    root = ROOT / args.root
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for offset, lookback in enumerate(args.lookbacks):
        output = root / "tuning" / f"stockmixer_sft_lb{lookback}"
        summary_path = output / "train_summary.json"
        if summary_path.is_file() and (output / "complete.json").is_file():
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            if (
                summary.get("experiment_protocol") == config["protocol"]
                and summary.get("epochs_trained") == epochs
                and summary.get("world_size") == args.nproc_per_node
            ):
                print(f"SKIP complete lb={lookback}", flush=True)
                continue
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={args.nproc_per_node}",
            str(TRAIN), "--config", args.config, "--panel", args.panel,
            "--lookback", str(lookback), "--epochs", str(epochs), "--output", str(output),
        ]
        if args.limit_train_dates:
            command += ["--limit-train-dates", str(args.limit_train_dates)]
        if args.limit_validation_dates:
            command += ["--limit-validation-dates", str(args.limit_validation_dates)]
        if args.disable_swanlab:
            command += ["--disable-swanlab"]
        log_path = logs / f"stockmixer_sft_lb{lookback}.log"
        print(f"START lb={lookback} ddp={args.nproc_per_node} log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        if completed.returncode:
            raise RuntimeError(f"StockMixer SFT DDP failed at lb={lookback}; see {log_path}")
        print(f"DONE lb={lookback}", flush=True)


if __name__ == "__main__":
    main()
