#!/usr/bin/env python3
"""Export complete test-feature predictions without reading test labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from finmodel.inference import build_inference_panel, infer_final_model
from finmodel.io import seed_everything
from finmodel.panel import Panel
from finmodel.sft import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-panel', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--config', default='configs/inference.json')
    parser.add_argument('--policy', default='weights/decision/policy.pt')
    parser.add_argument('--policy-metadata', default='weights/decision/metadata.json')
    parser.add_argument('--calibration', default='weights/factor_calibration.json')
    parser.add_argument('--workdir', default='artifacts/inference')
    parser.add_argument('--output', default='artifacts/predictions.csv')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-days', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config['seed'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested accelerator is unavailable; no silent CPU fallback')
    panel = Panel.open(build_inference_panel(args.history_panel, args.features, Path(args.workdir) / 'panel'))
    result = infer_final_model(panel=panel, config=config, policy_path=args.policy,
        policy_metadata=args.policy_metadata, calibration_path=args.calibration,
        workdir=args.workdir, output=args.output, device=device, expected_days=args.expected_days)
    print(f'EXPORTED {len(result)} rows to {args.output}; no evaluation labels read')


if __name__ == '__main__':
    main()
