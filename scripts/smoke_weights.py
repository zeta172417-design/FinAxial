#!/usr/bin/env python3
"""Strict weight load and small synthetic forward; no market data or labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from finmodel.models import StockTimeTransformer, build_decision_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested accelerator unavailable')
    predictor_metadata = json.loads(Path('weights/predictor/metadata.json').read_text())
    decision_metadata = json.loads(Path('weights/decision/metadata.json').read_text())
    state = torch.load('weights/predictor/model.pt', map_location='cpu', weights_only=True)
    stocks = state['stock_embedding.weight'].shape[0] - 1
    architecture = dict(predictor_metadata['architecture'])
    # Window length changes no learned tensor shape; use four dates for a cheap
    # functional smoke only. Real inference always retains the published 256.
    architecture.update(output_steps=4, context_days=0)
    predictor = StockTimeTransformer(stocks=stocks, **architecture).to(device).eval()
    predictor.load_state_dict(state, strict=True)
    policy = build_decision_policy(d_model=architecture['d_model'], **decision_metadata['policy_config']).to(device).eval()
    policy.load_state_dict(torch.load('weights/decision/policy.pt', map_location='cpu', weights_only=True), strict=True)
    with torch.inference_mode():
        hidden = predictor.encode_hidden(torch.zeros(stocks, 4, 128, device=device),
            torch.ones(stocks, 4, dtype=torch.bool, device=device),
            torch.ones(4, stocks, dtype=torch.bool, device=device))
        valid = torch.ones(4, stocks, dtype=torch.bool, device=device)
        ranking = predictor.score_hidden(hidden, valid)
        returns = predictor.predict_return_hidden(hidden, valid)
        result = policy(hidden, ranking, valid, valid, predicted_return=returns, sample=False)
        assert torch.isfinite(result.decision_score).all()
        assert torch.isfinite(result.action_value).all()
    print(json.dumps({'device': str(device), 'predictor_parameters': sum(p.numel() for p in predictor.parameters()),
        'decision_parameters': sum(p.numel() for p in policy.parameters()), 'synthetic_dates': 4,
        'stocks': stocks, 'strict_weight_load': True, 'forward_finite': True}))


if __name__ == '__main__':
    main()
