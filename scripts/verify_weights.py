#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='weights', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        path = args.root / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'weight integrity check failed: {filename}')
        print('OK', filename)


if __name__ == '__main__':
    main()
