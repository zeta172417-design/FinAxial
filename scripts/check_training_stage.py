#!/usr/bin/env python3
"""A per-epoch complete.json is not proof the requested run finished."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    marker = args.output / 'workflow_complete.json'
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record['epochs'] != args.epochs:
            raise RuntimeError('existing workflow epoch budget differs')
        summary = json.loads((args.output / 'train_summary.json').read_text())
        metadata = json.loads((args.output / 'last/metadata.json').read_text())
        if summary.get('epochs_trained') != args.epochs or metadata['epoch'] != args.epochs:
            raise RuntimeError('completed marker and actual epochs disagree')
        print('REUSE', args.output)
        return
    if args.output.exists() or args.output.with_suffix('.log').exists():
        raise RuntimeError(f'refusing to overwrite an incomplete report stage: {args.output}')
    print('FRESH', args.output)


if __name__ == '__main__':
    main()
