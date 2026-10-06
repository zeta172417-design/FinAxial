#!/usr/bin/env python3
"""Replay frozen actor and fit export scale using TRAIN cache/labels only."""
import argparse
import json
from io import StringIO
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from finmodel.decision_cache import DecisionFeatureCache, cache_source_hashes
from finmodel.io import atomic_json_dump, sha256_file
from finmodel.metrics import evaluate_frame, make_prediction_frame
from finmodel.models import build_decision_policy, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.score_calibration import apply_score_calibration, fit_positive_affine, return_errors, verify_order_preserved
from finmodel.sft import load_config


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/decision_grpo.json')
    parser.add_argument('--train-cache', default='artifacts/reproduction/cache/train')
    parser.add_argument('--policy', default='artifacts/reproduction/decision/full/last/policy.pt')
    parser.add_argument('--policy-metadata', default='artifacts/reproduction/decision/full/last/metadata.json')
    parser.add_argument('--output', default='artifacts/score_calibration/calibration.json')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = load_config(args.config)
    panel = Panel.open(config['train_panel'])
    predictor_hash, panel_hash = cache_source_hashes(panel, config['backbone_checkpoint'])
    cache = DecisionFeatureCache.open(args.train_cache, checkpoint_sha256=predictor_hash, panel_manifest_sha256=panel_hash)
    # Check all dates BEFORE reading any labels. Never fit on evaluation caches.
    dates = np.asarray(panel.dates[cache.date_indices])
    if cache.manifest['split'] != 'train' or dates[-1] > config['tuning_train_end']:
        raise ValueError('calibration must use train cache ending at the training cutoff')
    metadata = load_config(args.policy_metadata)
    if (metadata['backbone_sha256'] != predictor_hash
            or metadata['stock_vocab_sha256'] != stock_vocab_sha256(panel.codes)
            or metadata['policy_config'] != config['policy']):
        raise ValueError('actor, predictor, vocabulary or architecture mismatch')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested accelerator unavailable; no silent CPU fallback')
    torch.set_num_threads(4)
    policy = build_decision_policy(d_model=config['model']['d_model'], **config['policy']).to(device).eval()
    policy.load_state_dict(torch.load(args.policy, map_location='cpu', weights_only=True), strict=True)
    print(f'Replaying frozen actor: {len(dates)} TRAIN dates, {panel.shape[1]} stocks', flush=True)
    values = [torch.from_numpy(np.array(array, copy=True)).to(device) for array in
              (cache.hidden, cache.base_score, cache.eligible)]
    tradable = torch.from_numpy(~np.asarray(panel.limit_flags[cache.date_indices, :, 0], bool)).to(device)
    predicted = torch.from_numpy(np.array(cache.predicted_return, copy=True)).to(device)
    actor = policy(*values, tradable, predicted_return=predicted, sample=False)
    burnin = int(config['validation']['decision_burnin_days'])
    scores = actor.decision_score[burnin:].cpu().numpy().astype(np.float64)
    indices = np.asarray(cache.date_indices[burnin:])
    eligible = np.asarray(cache.eligible[burnin:])
    for row in range(len(scores)):
        usable = eligible[row] & np.isfinite(scores[row])
        scores[row, ~usable] = np.median(scores[row, usable]) if usable.any() else 0.
    truth = np.asarray(panel.labels[indices])
    valid = np.asarray(panel.label_valid[indices]) & np.isfinite(truth)
    fitted = fit_positive_affine(scores, truth, valid)
    transformed = apply_score_calibration(scores, fitted)
    verify_order_preserved(scores, transformed)
    # Verify ACTUAL CSV precision as well as in-memory arithmetic.
    sample = pd.DataFrame({'pred': transformed[-32:].reshape(-1)})
    decoded = pd.read_csv(StringIO(sample.to_csv(index=False, float_format='%.17g')),
                          float_precision='round_trip').pred.to_numpy().reshape(32, panel.shape[1])
    verify_order_preserved(scores[-32:], decoded)
    frame = make_prediction_frame(panel=panel, date_indices=indices[-32:], predictions=scores[-32:],
        eligible=eligible[-32:], model='FinAxial', route='train_calibration', fold='train', alpha=1)
    before = evaluate_frame(frame)
    frame['pred_raw'] = decoded.reshape(-1)
    after = evaluate_frame(frame)
    keys = ('ic_mean', 'annual_excess', 'mean_turnover', 'final_score')
    if any(before[key] != after[key] for key in keys):
        raise AssertionError('official score changed after calibration/CSV round trip')
    fitted['training_errors']['return_head'] = return_errors(cache.predicted_return[burnin:], truth, valid)
    fitted.update(predictor_sha256=predictor_hash, policy_sha256=sha256_file(args.policy),
        panel_manifest_sha256=panel_hash, train_cache_manifest_sha256=sha256_file(Path(args.train_cache) / 'manifest.json'),
        fit_date_start=int(panel.dates[indices[0]]), fit_date_end=int(panel.dates[indices[-1]]),
        fit_days=len(indices), replay_warmup_days=burnin, evaluation_labels_read=False,
        full_training_order_and_ties_preserved=True, csv_precision='%.17g', official_parity_sample_days=32,
        official_parity_before={key: before[key] for key in keys}, official_parity_after={key: after[key] for key in keys})
    atomic_json_dump(fitted, output)
    print(json.dumps(fitted, indent=2), flush=True)


if __name__ == '__main__':
    main()
