"""Positive affine export calibration; fit only on known training labels."""
from __future__ import annotations
import numpy as np


def apply_score_calibration(scores, calibration):
    slope, intercept = float(calibration['slope']), float(calibration['intercept'])
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        raise ValueError('calibration requires a finite intercept and strictly positive slope')
    result = np.asarray(scores, dtype=np.float64) * slope + intercept
    if not np.isfinite(result).all():
        raise ValueError('nonfinite calibrated scores')
    return result


def return_errors(prediction, labels, valid):
    prediction, labels = np.broadcast_arrays(np.asarray(prediction, np.float64), np.asarray(labels, np.float64))
    mask = np.asarray(valid, bool) & np.isfinite(prediction) & np.isfinite(labels)
    if not mask.any():
        raise ValueError('no finite labelled observations')
    error = prediction[mask] - labels[mask]
    return {'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
            'bias': float(np.mean(error)), 'observations': int(mask.sum())}


def fit_positive_affine(scores, labels, valid):
    scores, labels, mask = np.asarray(scores, np.float64), np.asarray(labels, np.float64), np.asarray(valid, bool)
    if scores.shape != labels.shape or mask.shape != scores.shape:
        raise ValueError('calibration arrays must have matching shapes')
    mask = mask & np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < 2:
        raise ValueError('at least two finite labelled observations required')
    x, y = scores[mask], labels[mask]
    centered = x - x.mean()
    variance = float(np.mean(centered ** 2))
    if variance <= 0:
        raise ValueError('constant scores cannot be calibrated')
    slope = float(np.mean(centered * (y - y.mean())) / variance)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError('OLS slope is not positive; refusing to reverse or collapse ranking')
    result = {'method': 'global_positive_affine_OLS', 'slope': slope,
        'intercept': float(y.mean() - slope * x.mean()), 'units': 'decimal_next_day_return',
        'observations': int(mask.sum()), 'fit_role': 'training_in_sample_not_generalization_evidence'}
    result['training_errors'] = {
        'raw_decision_score': return_errors(x, y, np.ones(x.shape, bool)),
        'calibrated_decision_score': return_errors(apply_score_calibration(x, result), y, np.ones(x.shape, bool)),
        'constant_training_mean': return_errors(np.full(x.shape, y.mean()), y, np.ones(x.shape, bool))}
    return result


def verify_order_preserved(scores, transformed):
    """Verify complete order and exact tie boundaries for every date."""
    scores, transformed = np.asarray(scores), np.asarray(transformed)
    if scores.ndim != 2 or transformed.shape != scores.shape:
        raise ValueError('rank verification requires matching [date, stock] arrays')
    for original, calibrated in zip(scores, transformed):
        order = np.argsort(original, kind='stable')
        other = np.argsort(calibrated, kind='stable')
        if not np.array_equal(order, other) or not np.array_equal(
                np.diff(original[order]) == 0, np.diff(calibrated[order]) == 0):
            raise AssertionError('calibration changed ordering or ties')
