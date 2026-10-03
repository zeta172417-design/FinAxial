from __future__ import annotations

from typing import NamedTuple

import torch

from .models.finaxial_policy import hard_top_fraction_mask


class HardScore(NamedTuple):
    final_score: torch.Tensor
    rank_ic: torch.Tensor
    annual_excess: torch.Tensor
    stability: torch.Tensor


def _average_rank(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware zero-based ranks matching scipy's average-rank convention."""
    values = values.reshape(-1)
    if values.numel() < 2:
        return torch.zeros_like(values)
    sorted_values, order = torch.sort(values, stable=True)
    group_start = torch.ones_like(sorted_values, dtype=torch.bool)
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group = group_start.cumsum(dim=0) - 1
    group_count = int(group[-1].item()) + 1
    positions = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    sums = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, group, positions)
    counts.scatter_add_(0, group, torch.ones_like(positions))
    sorted_ranks = (sums / counts.clamp_min(1.0))[group]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() < 2:
        return left.new_zeros(())
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(
        (left.square().sum() * right.square().sum()).clamp_min(1e-12)
    )
    correlation = (left * right).sum() / denominator
    return torch.where(torch.isfinite(correlation), correlation, correlation.new_zeros(()))


@torch.no_grad()
def hard_composite_score(
    scores: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    top_fraction: float = 0.10,
) -> HardScore:
    """Exact block reward: mean Rank IC, annual excess and hard Jaccard stability."""
    if scores.ndim != 2 or target.shape != scores.shape:
        raise ValueError("scores and target must both be [dates, stocks]")
    if label_mask.shape != scores.shape or tradable_mask.shape != scores.shape:
        raise ValueError("masks must match scores")
    label_mask = label_mask.bool() & torch.isfinite(target)
    tradable_mask = tradable_mask.bool()
    ic_values, daily_excess = [], []
    for date in range(scores.shape[0]):
        labelled = torch.nonzero(label_mask[date], as_tuple=False).reshape(-1)
        if labelled.numel() >= 2:
            prediction_rank = _average_rank(scores[date, labelled].float())
            target_rank = _average_rank(target[date, labelled].float())
            ic_values.append(_correlation(prediction_rank, target_rank))
        return_mask = label_mask[date] & tradable_mask[date]
        return_indices = torch.nonzero(return_mask, as_tuple=False).reshape(-1)
        if return_indices.numel():
            count = max(int(return_indices.numel() * float(top_fraction)), 1)
            chosen = torch.topk(
                scores[date, return_indices], k=min(count, return_indices.numel()),
            ).indices
            returns = target[date, return_indices]
            top_return = returns[chosen].mean()
            daily_excess.append(top_return - returns.mean())
    rank_ic = torch.stack(ic_values).mean() if ic_values else scores.new_zeros(())
    annual_excess = (
        252.0 * torch.stack(daily_excess).mean()
        if daily_excess else scores.new_zeros(())
    )
    portfolios = hard_top_fraction_mask(scores, tradable_mask, top_fraction)
    intersections = (portfolios[:-1] & portfolios[1:]).sum(dim=1).float()
    unions = (portfolios[:-1] | portfolios[1:]).sum(dim=1).float()
    stability = (
        (intersections / unions.clamp_min(1.0)).mean()
        if len(portfolios) > 1 else scores.new_zeros(())
    )
    final_score = 0.4 * rank_ic + 0.3 * annual_excess + 0.3 * stability
    return HardScore(final_score, rank_ic, annual_excess, stability)


def group_relative_advantage(
    rewards: torch.Tensor, *, epsilon: float = 1e-6, clip: float | None = 5.0,
) -> torch.Tensor:
    """Normalize one same-state rollout group without a learned critic."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two rewards in one group")
    advantage = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(epsilon)
    if clip is not None:
        advantage = advantage.clamp(-float(clip), float(clip))
    return advantage
