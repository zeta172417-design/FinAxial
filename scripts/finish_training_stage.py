#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finmodel.io import atomic_json_dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    summary = json.loads((args.output / 'train_summary.json').read_text())
    metadata = json.loads((args.output / 'last/metadata.json').read_text())
    if summary.get('epochs_trained') != args.epochs or metadata['epoch'] != args.epochs:
        raise RuntimeError('trainer returned before the requested final epoch')
    if not (args.output / 'complete.json').is_file():
        raise RuntimeError('missing trainer completion record')
    atomic_json_dump({'epochs': args.epochs, 'checkpoint_policy': 'last_epoch'},
                     args.output / 'workflow_complete.json')


if __name__ == '__main__':
    main()
