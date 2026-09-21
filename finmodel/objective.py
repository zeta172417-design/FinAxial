from __future__ import annotations

from typing import NamedTuple

import torch

from .losses import _soft_day_components, masked_mse


class SequenceSoftScore(NamedTuple):
    rank_ic: torch.Tensor
    annual_excess_raw: torch.Tensor
    stability: torch.Tensor
    mse: torch.Tensor


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, soft Final score and bounded annual excess."""
    if excess_bound <= 0 or mse_weight < 0:
        raise ValueError("excess_bound must be positive and mse_weight non-negative")
    bounded_excess = float(excess_bound) * torch.tanh(
        global_annual_excess_raw / float(excess_bound)
    )
    score = 0.4 * global_rank_ic + 0.3 * bounded_excess + 0.3 * global_stability
    return -score + float(mse_weight) * components.mse, score, bounded_excess
