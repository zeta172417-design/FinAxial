"""Label-free final-model inference; evaluation labels are never inputs."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from .factors import FACTOR_PROFILES, FACTOR_VERSION, iter_factor_rows
from .io import atomic_json_dump, sha256_file
from .models import build_decision_policy, stock_vocab_sha256
from .panel import Panel, FEATURE_COLUMNS, KEY_COLUMNS, _causal_fill, _discover_axes
from .pipeline import load_backbone
from .sequence import MultiDateCrossSectionDataset


INFERENCE_VERSION = 'finaxial-label-free-inference-v1'


def build_inference_panel(history_panel, features_csv, output, *, chunksize=250_000):
    """Append X only, including its final day. Never manufacture future labels."""
    history = Panel.open(history_panel)
    source = Path(features_csv).resolve()
    output = Path(output)
    hashes = {'history_manifest': sha256_file(history.root / 'manifest.json'),
              'features_csv': sha256_file(source)}
    marker = output / 'manifest.json'
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached.get('source_hashes') != hashes or cached.get('processing_version') != INFERENCE_VERSION:
            raise FileExistsError('inference panel source mismatch')
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'incomplete inference panel: {output}')
    dates, codes, rows = _discover_axes(source, chunksize)
    if not np.array_equal(codes, history.codes):
        raise ValueError('inference stock vocabulary/order differs from training')
    if not len(dates) or dates[0] <= history.dates[-1] or rows != len(dates) * len(codes):
        raise ValueError('inference must be a later rectangular date/stock panel')
    output.mkdir(parents=True, exist_ok=True)
    all_dates = np.concatenate((history.dates, dates))
    shape = (len(all_dates), len(codes))
    arrays = {}
    specifications = {'features': (np.float32, shape + (6,), np.nan),
        'labels': (np.float32, shape, np.nan), 'feature_valid': (np.bool_, shape, False),
        'label_valid': (np.bool_, shape, False), 'limit_flags': (np.bool_, shape + (2,), False)}
    for name, (dtype, array_shape, fill) in specifications.items():
        arrays[name] = open_memmap(output / f'{name}.npy', mode='w+', dtype=dtype, shape=array_shape)
        arrays[name][:] = fill
        arrays[name][:len(history.dates)] = getattr(history, name)
    seen = np.zeros((len(dates), len(codes)), dtype=bool)
    columns = list(KEY_COLUMNS + FEATURE_COLUMNS) + ['flag_limit_up', 'flag_limit_down']
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        if chunk[list(KEY_COLUMNS)].duplicated().any():
            raise ValueError('duplicate inference key within CSV chunk')
        day = pd.Index(dates).get_indexer(chunk.trade_date.to_numpy(dtype=np.int32))
        stock = pd.Index(codes).get_indexer(chunk.ts_code.astype(str))
        if (day < 0).any() or (stock < 0).any() or seen[day, stock].any():
            raise ValueError('duplicate or unknown inference key')
        seen[day, stock] = True
        arrays['features'][day + len(history.dates), stock] = chunk[list(FEATURE_COLUMNS)].to_numpy(np.float32)
        for index, name in enumerate(('flag_limit_up', 'flag_limit_down')):
            arrays['limit_flags'][day + len(history.dates), stock, index] = chunk[name].fillna(0).to_numpy() != 0
    if not seen.all():
        raise ValueError('missing inference keys')
    statistics = _causal_fill(arrays['features'], arrays['feature_valid'])
    for array in arrays.values():
        array.flush()
    np.save(output / 'dates.npy', all_dates, allow_pickle=False)
    np.save(output / 'codes.npy', codes, allow_pickle=False)
    atomic_json_dump({'processing_version': INFERENCE_VERSION, 'source_hashes': hashes,
        'shape': {'dates': shape[0], 'codes': shape[1], 'features': [*shape, 6]},
        'inference_dates': len(dates), 'inference_start': int(dates[0]), 'inference_end': int(dates[-1]),
        'history_dates': len(history.dates), 'evaluation_labels_read': False,
        'date_range': [int(all_dates[0]), int(all_dates[-1])], **statistics}, marker)
    return output


def build_inference_factors(panel, calibration_path, output):
    """Use the published TRAIN-only calibration, never fit on inference data."""
    settings = json.loads(Path(calibration_path).read_text())
    names = FACTOR_PROFILES['f128'][0]
    if settings['factor_version'] != FACTOR_VERSION or tuple(settings['feature_names']) != names:
        raise ValueError('published factor definition/calibration mismatch')
    output = Path(output)
    marker = output / 'manifest.json'
    hashes = {'panel': sha256_file(panel.root / 'manifest.json'), 'calibration': sha256_file(calibration_path)}
    if marker.exists():
        if json.loads(marker.read_text())['source_hashes'] != hashes:
            raise FileExistsError('inference factor source mismatch')
        return np.load(output / 'f128_derived.npy', mmap_mode='r')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('incomplete inference factor directory')
    output.mkdir(parents=True, exist_ok=True)
    factors = open_memmap(output / 'f128_derived.npy', mode='w+', dtype=np.float16,
                          shape=(*panel.shape, len(names)))
    for date, row in enumerate(iter_factor_rows(panel.features, panel.feature_valid,
            centers=np.asarray(settings['centers'], np.float32), scales=np.asarray(settings['scales'], np.float32))):
        factors[date] = row
    factors.flush()
    atomic_json_dump({'source_hashes': hashes, 'labels_used': False, 'shape': list(factors.shape)}, marker)
    return factors


def inference_dataset(panel, config, factors, first, last):
    start, end = map(int, config['validation']['score_output_positions'])
    steps = int(config['model']['output_steps'])
    padding = steps - end
    return MultiDateCrossSectionDataset(panel, np.arange(first - start, last + padding + 1),
        lookback=int(config['model']['lookback']), output_steps=steps,
        context_days=config['model'].get('context_days'), stride=end - start,
        min_history=int(config['min_history']), epsilon=float(config['data']['normalization_epsilon']),
        clip=float(config['data']['normalization_clip']), feature_mode='factors', factor_features=factors,
        inference_future_padding=padding, require_trainable=False)


@torch.inference_mode()
def infer_final_model(*, panel, config, policy_path, policy_metadata, calibration_path,
                      workdir, output, device, expected_days=None):
    """Infer every supplied X date, using action means and continuous decision state."""
    from scripts.build_c0_decision_cache import build_one
    from .decision_cache import DecisionFeatureCache, cache_source_hashes
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + '.manifest.json').exists():
        raise FileExistsError(f'refusing to overwrite prediction export: {output}')
    first = int(panel.manifest['history_dates'])
    days = int(panel.manifest['inference_dates'])
    if expected_days is not None and days != expected_days:
        raise ValueError(f'expected {expected_days} inference dates, got {days}')
    burnin = int(config['validation']['decision_burnin_days'])
    if first < burnin + config['model']['output_steps']:
        raise ValueError('insufficient pre-inference history')
    metadata = json.loads(Path(policy_metadata).read_text())
    predictor_hash, panel_hash = cache_source_hashes(panel, config['backbone_checkpoint'])
    if metadata['backbone_sha256'] != predictor_hash or metadata['stock_vocab_sha256'] != stock_vocab_sha256(panel.codes):
        raise ValueError('predictor/policy/stock vocabulary pairing mismatch')
    if metadata['policy_config'] != config['policy']:
        raise ValueError('decision architecture does not match its checkpoint')
    if int(metadata['validation_burnin_days']) != burnin:
        raise ValueError('decision warmup differs from published protocol')
    factors = build_inference_factors(panel, calibration_path, Path(workdir) / 'factors')
    backbone, _ = load_backbone(config, panel, device)
    indices = np.arange(first - burnin, first + days)
    # Every cache date is predicted once; labels have no effect on inference.
    dataset = inference_dataset(panel, config, factors, int(indices[0]), int(indices[-1]))
    cache_root = Path(workdir) / 'cache'
    build_one(name='inference', dataset=dataset, panel=panel, backbone=backbone,
        output=cache_root, checkpoint_hash=predictor_hash, panel_hash=panel_hash, device=device,
        score_positions=config['validation']['score_output_positions'], expected_dates=indices)
    del backbone, factors, dataset
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    cache = DecisionFeatureCache.open(cache_root / 'inference', checkpoint_sha256=predictor_hash,
                                      panel_manifest_sha256=panel_hash)
    policy = build_decision_policy(d_model=config['model']['d_model'], **config['policy']).to(device).eval()
    policy.load_state_dict(torch.load(policy_path, map_location='cpu', weights_only=True), strict=True)
    result = policy(*[torch.from_numpy(np.array(x, copy=True)).to(device) for x in
                     (cache.hidden, cache.base_score, cache.eligible)],
        torch.from_numpy(~np.asarray(panel.limit_flags[indices, :, 0], dtype=bool)).to(device),
        predicted_return=torch.from_numpy(np.array(cache.predicted_return, copy=True)).to(device), sample=False)
    scores = result.decision_score[burnin:].float().cpu().numpy()
    for row in range(days):
        valid = np.asarray(cache.eligible[burnin + row]) & np.isfinite(scores[row])
        scores[row, ~valid] = np.median(scores[row, valid]) if valid.any() else 0.0
    frame = pd.DataFrame({'ts_code': np.tile(panel.codes, days),
        'trade_date': np.repeat(panel.dates[first:], len(panel.codes)), 'pred': scores.reshape(-1)})
    if len(frame) != days * panel.shape[1] or frame[list(KEY_COLUMNS)].duplicated().any() or not np.isfinite(frame.pred).all():
        raise AssertionError('prediction coverage/key/finite-value check failed')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    frame.to_csv(temporary, index=False, float_format='%.10g')
    temporary.rename(output)
    atomic_json_dump({'model': 'FinAxial E0', 'rows': len(frame), 'dates': days, 'stocks': panel.shape[1],
        'date_start': int(panel.dates[first]), 'date_end': int(panel.dates[-1]), 'decision_burnin_days': burnin,
        'predictor_sha256': predictor_hash, 'policy_sha256': sha256_file(policy_path),
        'calibration_sha256': sha256_file(calibration_path), 'csv_sha256': sha256_file(output),
        'evaluation_labels_read': False, 'final_feature_date_included': True,
        'pred_semantics': 'portfolio-aware ranking score; not a calibrated return estimate'},
        output.with_suffix(output.suffix + '.manifest.json'))
    return frame
