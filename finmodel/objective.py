from __future__ import annotations

import math
from typing import NamedTuple

import torch

from .losses import _soft_day_components, masked_mse


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
