#!/usr/bin/env python3
"""Build the continuous train + reconstructed-test panel for post-hoc validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finmodel.panel import build_phase1_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-panel", default="artifacts/panel/train")
    parser.add_argument("--test-features", default="evaluation/测试集_X.csv")
    parser.add_argument("--test-labels", default="evaluation/测试集_Y.csv")
    parser.add_argument("--output", default="artifacts/panel/phase1")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = build_phase1_panel(
        args.train_panel,
        args.test_features,
        args.test_labels,
        args.output,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
