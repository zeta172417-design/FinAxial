from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .losses import (
    _correlation,
    _hard_percentile_rank,
    _soft_day_components,
    _soft_percentile_rank,
    masked_mse,
)


class SequenceSoftScore(NamedTuple):
    rank_ic: torch.Tensor
    annual_excess_raw: torch.Tensor
    stability: torch.Tensor
    mse: torch.Tensor


class ObjectiveSettings(NamedTuple):
    rank_temperature: float
    top_temperature: float
    component_weights: tuple[float, float, float]
    range_balance_beta: float


def standardized_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 0.5,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Daily cross-sectional Huber after causal-label-free standardization.

    Standardizing prediction and target independently makes this auxiliary
    calibration term insensitive to the arbitrary daily score scale while
    retaining substantially denser gradients than a rank-only objective.
    """
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [dates, stocks]")
    if mask.shape != prediction.shape:
        raise ValueError("mask must match prediction")
    if delta <= 0 or epsilon <= 0:
        raise ValueError("delta and epsilon must be positive")
    losses = []
    for date in range(prediction.shape[0]):
        valid = mask[date].bool() & torch.isfinite(target[date])
        if int(valid.sum()) < 2:
            continue
        pred = prediction[date, valid]
        truth = target[date, valid]
        pred = (pred - pred.mean()) / (pred.std(unbiased=False) + float(epsilon))
        truth = (truth - truth.mean()) / (truth.std(unbiased=False) + float(epsilon))
        losses.append(F.huber_loss(pred, truth, delta=float(delta)))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def fixed_scale_excess_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    return_scale: float = 0.02,
    target_clip: float = 5.0,
    delta: float = 0.5,
) -> torch.Tensor:
    """Predict next-day excess return in fixed, cross-date-comparable units.

    The daily tradable-universe mean is removed because the downstream return
    metric is excess over that same market.  Dividing by one fixed scale keeps
    magnitude information that would be erased by daily z-scoring.
    """
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [dates, stocks]")
    if label_mask.shape != prediction.shape or tradable_mask.shape != prediction.shape:
        raise ValueError("masks must match prediction")
    if return_scale <= 0 or target_clip <= 0 or delta <= 0:
        raise ValueError("return_scale, target_clip and delta must be positive")
    losses = []
    for date in range(prediction.shape[0]):
        labelled = label_mask[date].bool() & torch.isfinite(target[date])
        market = labelled & tradable_mask[date].bool()
        if int(labelled.sum()) < 2 or not bool(market.any()):
            continue
        market_return = target[date, market].mean().detach()
        scaled_excess = (
            (target[date, labelled] - market_return) / float(return_scale)
        ).clamp(-float(target_clip), float(target_clip))
        losses.append(F.huber_loss(
            prediction[date, labelled], scaled_excess, delta=float(delta),
        ))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def absolute_return_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    return_scale: float = 0.02,
    target_clip: float = 5.0,
    delta: float = 0.5,
) -> torch.Tensor:
    """Calibrate an independent head to absolute next-day returns."""
    if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target and mask must be [dates, stocks]")
    if return_scale <= 0 or target_clip <= 0 or delta <= 0:
        raise ValueError("return scale, target clip and Huber delta must be positive")
    valid = mask.bool() & torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    scaled_target = (target[valid] / float(return_scale)).clamp(
        -float(target_clip), float(target_clip),
    )
    return F.huber_loss(
        prediction[valid] / float(return_scale), scaled_target,
        delta=float(delta),
    )


def multi_date_soft_rank_ic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Mean differentiable daily Rank IC without building Top-10 portfolios."""
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [dates, stocks]")
    if mask.shape != prediction.shape or temperature <= 0:
        raise ValueError("mask must match prediction and temperature must be positive")
    values = []
    for date in range(prediction.shape[0]):
        valid = mask[date].bool() & torch.isfinite(target[date])
        ranks, indices = _soft_percentile_rank(
            prediction[date], valid, float(temperature),
        )
        if indices.numel() < 2:
            values.append(prediction[date].sum() * 0.0)
            continue
        truth_rank = _hard_percentile_rank(target[date, indices]).detach()
        values.append(_correlation(ranks[indices], truth_rank))
    return torch.stack(values).mean()


def multi_date_soft_top10_excess(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    temperature: float = 0.02,
    min_universe: int = 100,
) -> torch.Tensor:
    """Differentiable annualized Top-10% excess return, without turnover.

    The detached cutoff is halfway between the last selected and first
    unselected tradable score. This makes the forward limit match exact Top-10%
    selection while the sigmoid supplies gradients around the selection edge.
    Scores are standardized per day so the temperature is scale-independent.
    """
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [dates, stocks]")
    if label_mask.shape != prediction.shape or tradable_mask.shape != prediction.shape:
        raise ValueError("masks must match prediction")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_universe < 2:
        raise ValueError("min_universe must be at least 2")
    values = []
    for date in range(prediction.shape[0]):
        valid = (
            label_mask[date].bool() & tradable_mask[date].bool()
            & torch.isfinite(target[date]) & torch.isfinite(prediction[date])
        )
        scores = prediction[date, valid]
        truth = target[date, valid]
        if scores.numel() < min_universe:
            values.append(prediction[date].sum() * 0.0)
            continue
        top_count = max(scores.numel() // 10, 1)
        ranked = torch.topk(scores.detach(), top_count + 1).values
        threshold = (ranked[top_count - 1] + ranked[top_count]) * 0.5
        score_scale = scores.detach().std(unbiased=False).clamp_min(1e-6)
        weights = torch.sigmoid(
            ((scores - threshold) / score_scale) / float(temperature)
        )
        top_return = (weights * truth).sum() / weights.sum().clamp_min(1e-6)
        values.append(252.0 * (top_return - truth.mean()))
    if not values:
        return prediction.sum() * 0.0
    return torch.stack(values).mean()


def objective_settings(training: dict, epoch: int) -> ObjectiveSettings:
    """Resolve the controlled loss ablation scheduled for one epoch."""
    variant = str(training.get("loss_variant", "official"))
    rank_temperature = float(training["rank_temperature"])
    top_temperature = float(training["top_temperature"])
    weights = (0.4, 0.3, 0.3)
    range_beta = 0.0
    if variant == "official":
        pass
    elif variant == "temperature_anneal":
        end_epoch = int(training.get("temperature_anneal_end_epoch", 20))
        progress = min(max((int(epoch) - 1) / max(end_epoch - 1, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        rank_start = float(training.get("rank_temperature_start", 0.20))
        rank_end = float(training.get("rank_temperature_end", 0.05))
        top_start = float(training.get("top_temperature_start", 0.04))
        top_end = float(training.get("top_temperature_end", 0.01))
        rank_temperature = rank_end + (rank_start - rank_end) * cosine
        top_temperature = top_end + (top_start - top_end) * cosine
    elif variant == "turnover_curriculum":
        no_turnover_end = int(training.get("curriculum_no_turnover_end_epoch", 5))
        transition_end = int(training.get("curriculum_transition_end_epoch", 15))
        start = (0.6, 0.4, 0.0)
        if epoch <= no_turnover_end:
            weights = start
        elif epoch < transition_end:
            progress = (epoch - no_turnover_end) / max(
                transition_end - no_turnover_end, 1,
            )
            official = (0.4, 0.3, 0.3)
            weights = tuple(
                left + progress * (right - left)
                for left, right in zip(start, official)
            )
    elif variant == "range_balanced":
        range_beta = float(training.get("range_balance_beta", 0.1))
    else:
        raise ValueError(f"unknown loss_variant: {variant}")
    if rank_temperature <= 0 or top_temperature <= 0:
        raise ValueError("loss temperatures must be positive")
    if not 0.0 <= range_beta <= 1.0:
        raise ValueError("range_balance_beta must be in [0, 1]")
    return ObjectiveSettings(rank_temperature, top_temperature, weights, range_beta)


def multi_date_soft_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    rank_temperature: float = 0.1,
    top_temperature: float = 0.02,
) -> SequenceSoftScore:
    """Compute differentiable Final-score components over consecutive dates."""
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [dates, stocks]")
    if label_mask.shape != prediction.shape or tradable_mask.shape != prediction.shape:
        raise ValueError("sequence masks must match prediction")
    daily_ic, daily_excess, portfolios, daily_mse = [], [], [], []
    for date in range(prediction.shape[0]):
        ic, excess, portfolio = _soft_day_components(
            prediction[date], target[date], label_mask[date], tradable_mask[date],
            rank_temperature=rank_temperature,
            top_temperature=top_temperature,
        )
        daily_ic.append(ic)
        daily_excess.append(excess)
        portfolios.append(portfolio)
        daily_mse.append(masked_mse(prediction[date], target[date], label_mask[date]))
    portfolio = torch.stack(portfolios)
    intersections = (portfolio[:-1] * portfolio[1:]).sum(dim=1)
    unions = (
        portfolio[:-1] + portfolio[1:] - portfolio[:-1] * portfolio[1:]
    ).sum(dim=1)
    stability = (intersections / unions.clamp_min(1e-6)).mean()
    return SequenceSoftScore(
        torch.stack(daily_ic).mean(),
        torch.stack(daily_excess).mean(),
        stability,
        torch.stack(daily_mse).mean(),
    )


def compose_bounded_final_score(
    components: SequenceSoftScore,
    *,
    global_rank_ic: torch.Tensor,
    global_annual_excess_raw: torch.Tensor,
    global_stability: torch.Tensor,
    excess_bound: float = 0.5,
    mse_weight: float = 0.0,
    component_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    range_balance_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, soft Final score and bounded annual excess."""
    if excess_bound <= 0 or mse_weight < 0:
        raise ValueError("excess_bound must be positive and mse_weight non-negative")
    if len(component_weights) != 3 or any(weight < 0 for weight in component_weights):
        raise ValueError("component_weights must contain three non-negative values")
    if not 0.0 <= range_balance_beta <= 1.0:
        raise ValueError("range_balance_beta must be in [0, 1]")
    bounded_excess = float(excess_bound) * torch.tanh(
        global_annual_excess_raw / float(excess_bound)
    )
    rank_weight, excess_weight, stability_weight = component_weights
    score = (
        rank_weight * global_rank_ic
        + excess_weight * bounded_excess
        + stability_weight * global_stability
    )
    if range_balance_beta:
        range_normalized = (
            0.4 * global_rank_ic / 0.1
            + 0.3 * bounded_excess / 0.5
            + 0.3 * global_stability
        )
        score = (1.0 - range_balance_beta) * score + range_balance_beta * range_normalized
    return -score + float(mse_weight) * components.mse, score, bounded_excess
