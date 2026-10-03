#!/usr/bin/env python3
"""Build reusable label-free F32/F64/F128 causal feature caches on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finmodel.factors import build_factor_cache
from finmodel.panel import Panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--output", default="artifacts/factors/causal_v3")
    parser.add_argument("--train-end-date", type=int, required=True)
    args = parser.parse_args()
    result = build_factor_cache(
        Panel.open(args.panel), args.output, train_end_date=args.train_end_date,
    )
    print(f"Factor cache ready: {result / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
