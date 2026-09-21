#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finmodel.panel import build_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the causal read-only NPY panel")
    parser.add_argument("--source", default="data/raw/训练集.csv")
    parser.add_argument("--output", default="artifacts/panel/train")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = build_panel(
        args.source, args.output, chunksize=args.chunksize, overwrite=args.overwrite,
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
