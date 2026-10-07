"""Exact daily score decomposition for predictor-based GRPO decisions."""

from __future__ import annotations
from typing import NamedTuple

import torch
from .grpo import _average_rank, _correlation
from .models.finaxial_policy import hard_top_fraction_mask


class DailyScore(NamedTuple):
    reward: torch.Tensor
    rank_ic: torch.Tensor
    annual_excess: torch.Tensor
    stability: torch.Tensor


@torch.no_grad()
def exact_daily_score(
    scores: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable: torch.Tensor,
    *,
    previous_selected: torch.Tensor | None = None,
    previous_selected_by_date: torch.Tensor | None = None,
    top_fraction: float = 0.10,
) -> DailyScore:
    """Daily contributions whose mean equals the exact block Final Score.

    An optional pre-block portfolio gives the first day a real transition.
    Different valid-day denominators follow the official aggregation.
    """
    if scores.ndim != 2 or any(
        value.shape != scores.shape for value in (target, label_mask, tradable)
    ):
        raise ValueError("all daily score arrays must be [dates, stocks]")
    dates = scores.shape[0]
    if previous_selected_by_date is not None and previous_selected_by_date.shape != scores.shape:
        raise ValueError("previous_selected_by_date must match scores")
    labelled = label_mask.bool() & torch.isfinite(target)
    trade = tradable.bool()
    selected = hard_top_fraction_mask(scores, trade, top_fraction)
    ic = scores.new_zeros(dates)
    excess = scores.new_zeros(dates)
    stability = scores.new_zeros(dates)
    ic_valid = torch.zeros(dates, dtype=torch.bool, device=scores.device)
    excess_valid = torch.zeros_like(ic_valid)
    stability_valid = torch.zeros_like(ic_valid)
    for date in range(dates):
        labelled_indices = torch.nonzero(labelled[date], as_tuple=False).reshape(-1)
        if labelled_indices.numel() >= 2:
            ic[date] = _correlation(
                _average_rank(scores[date, labelled_indices].float()),
                _average_rank(target[date, labelled_indices].float()),
            )
            ic_valid[date] = True
        return_indices = torch.nonzero(labelled[date] & trade[date], as_tuple=False).reshape(-1)
        if return_indices.numel():
            count = max(int(return_indices.numel() * top_fraction), 1)
            chosen = torch.topk(
                scores[date, return_indices],
                k=min(count, return_indices.numel()),
            ).indices
            returns = target[date, return_indices]
            excess[date] = returns[chosen].mean() - returns.mean()
            excess_valid[date] = True
        if previous_selected_by_date is not None:
            previous = previous_selected_by_date[date].bool()
        elif date > 0:
            previous = selected[date - 1]
        elif previous_selected is not None:
            previous = previous_selected.bool()
        else:
            continue
        union = (selected[date] | previous).sum().clamp_min(1)
        stability[date] = (
            (selected[date] & previous).sum().float() / union.float()
        )
        stability_valid[date] = True

    scale = float(dates)
    reward = (
        0.4 * ic * (scale / ic_valid.sum().clamp_min(1))
        + 0.3 * 252.0 * excess * (scale / excess_valid.sum().clamp_min(1))
        + 0.3 * stability * (scale / stability_valid.sum().clamp_min(1))
    )
    return DailyScore(reward, ic, 252.0 * excess, stability)
