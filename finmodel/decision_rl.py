"""Exact daily score decomposition and value estimates for C0 decision RL."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .grpo import _average_rank, _correlation
from .models.finaxial_decision_policy import percentile_rank_scores
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
        if date > 0:
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


@torch.no_grad()
def critic_observations(
    hidden: torch.Tensor,
    base_score: torch.Tensor,
    eligible: torch.Tensor,
    tradable: torch.Tensor,
    selected: torch.Tensor,
    decision_score: torch.Tensor,
    *,
    return_scale: float = 0.02,
    top_fraction: float = 0.10,
) -> torch.Tensor:
    """Pre-action state; never reads labels or today's sampled action."""
    if hidden.shape[:2] != base_score.shape or selected.shape != base_score.shape:
        raise ValueError("critic state axes disagree")
    dates, stocks = base_score.shape
    mask = eligible.bool()
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = (hidden * mask[..., None]).sum(dim=1) / counts
    base_rank = percentile_rank_scores(base_score, mask)
    previous_selection = hard_top_fraction_mask(
        base_rank[:1], tradable[:1].bool(), top_fraction,
    )[0]
    previous_score = base_rank[0]
    scalar_rows = []
    predicted_return = base_score * return_scale
    for date in range(dates):
        values = predicted_return[date, mask[date]]
        dispersion = values.std(unbiased=False) if values.numel() else base_score.new_zeros(())
        previous_values = predicted_return[date, previous_selection]
        previous_mean = previous_values.mean() if previous_values.numel() else base_score.new_zeros(())
        naive_selection = hard_top_fraction_mask(
            base_rank[date:date + 1], tradable[date:date + 1].bool(), top_fraction,
        )[0]
        union = (naive_selection | previous_selection).sum().clamp_min(1)
        naive_jaccard = (naive_selection & previous_selection).sum().float() / union.float()
        scalar_rows.append(torch.stack((
            dispersion / return_scale,
            previous_mean / return_scale,
            1.0 - naive_jaccard,
            previous_selection.float().mean(),
            previous_score[previous_selection].mean() if bool(previous_selection.any())
                else base_score.new_zeros(()),
            previous_score.float().std(unbiased=False),
        )))
        previous_selection = selected[date].bool()
        previous_score = decision_score[date]
    return torch.cat((pooled, torch.stack(scalar_rows)), dim=-1).float()


class DecisionValueHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model + 6),
            nn.Linear(d_model + 6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


def generalized_advantage(
    reward: torch.Tensor,
    value: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reward.shape != value.shape or reward.ndim != 1:
        raise ValueError("reward and value must be matching one-dimensional tensors")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0, 1]")
    advantage = torch.zeros_like(reward)
    carry = reward.new_zeros(())
    for date in range(len(reward) - 1, -1, -1):
        next_value = value[date + 1] if date + 1 < len(reward) else value.new_zeros(())
        residual = reward[date] + gamma * next_value - value[date]
        carry = residual + gamma * lam * carry
        advantage[date] = carry
    return advantage, advantage + value


def discounted_return_to_go(reward: torch.Tensor, *, gamma: float = 0.99) -> torch.Tensor:
    """Finite-block future reward without a value baseline or bootstrapping."""
    if reward.ndim != 1 or not 0 <= gamma <= 1:
        raise ValueError("reward must be one-dimensional and gamma in [0, 1]")
    returns = torch.zeros_like(reward)
    carry = reward.new_zeros(())
    for date in range(len(reward) - 1, -1, -1):
        carry = reward[date] + gamma * carry
        returns[date] = carry
    return returns
