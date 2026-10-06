"""Exact daily score decomposition and value estimates for predictor decision RL."""

from __future__ import annotations

import math
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


@torch.no_grad()
def critic_observations(
    hidden: torch.Tensor,
    base_score: torch.Tensor,
    eligible: torch.Tensor,
    tradable: torch.Tensor,
    selected: torch.Tensor,
    decision_score: torch.Tensor,
    *,
    predicted_return: torch.Tensor | None = None,
    return_scale: float = 0.02,
    top_fraction: float = 0.10,
    portfolio_detail: bool = False,
) -> torch.Tensor:
    """Pre-action state; never reads labels or today's sampled action."""
    if hidden.shape[:2] != base_score.shape or selected.shape != base_score.shape:
        raise ValueError("critic state axes disagree")
    if return_scale <= 0:
        raise ValueError("return_scale must be positive")
    if predicted_return is not None and (
        predicted_return.shape != base_score.shape
        or not bool(torch.isfinite(predicted_return).all())
    ):
        raise ValueError("predicted_return must be finite and match base_score")
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
    detail_rows = []
    if predicted_return is None:
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
        if portfolio_detail:
            tradable_today = tradable[date].bool() & mask[date]
            top_today = hard_top_fraction_mask(
                base_rank[date:date + 1], tradable_today[None, :], top_fraction,
            )[0]
            cutoff = 1.0 - top_fraction
            boundary_today = (
                (base_rank[date] - cutoff).abs() <= 0.05
            ) & tradable_today

            def pool(group: torch.Tensor) -> torch.Tensor:
                weight = group.to(hidden.dtype)
                return (hidden[date] * weight[:, None]).sum(dim=0) / weight.sum().clamp_min(1.0)

            detail_rows.append(torch.cat((
                pool(previous_selection & mask[date]),
                pool(top_today),
                pool(boundary_today),
            )))
        previous_selection = selected[date].bool()
        previous_score = decision_score[date]
    parts = (pooled, torch.stack(scalar_rows))
    if portfolio_detail:
        parts += (torch.stack(detail_rows),)
    return torch.cat(parts, dim=-1).float()


class DecisionValueHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 64,
                 portfolio_detail: bool = False) -> None:
        super().__init__()
        input_dim = d_model * (4 if portfolio_detail else 1) + 6
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class DecisionStateValueHead(DecisionValueHead):
    """Value of pre-action state, including detached actor memory/stock ages.

    The base network is unchanged. A zero-initialized residual branch preserves
    its initial predictions, and never backpropagates into the frozen predictor
    or recurrent actor. Every held eligible stock is retained before adding
    fresh top-15% candidates (up to candidate_max_count).
    """

    def __init__(self, d_model=128, hidden_dim=64, recurrent_dim=32,
                 candidate_max_count=1024, return_scale=0.02):
        super().__init__(d_model=d_model, hidden_dim=hidden_dim, portfolio_detail=True)
        self.candidate_max_count = candidate_max_count
        self.return_scale = return_scale
        with torch.random.fork_rng(devices=[]):
            self.stock_encoder = nn.Sequential(
                nn.LayerNorm(d_model + 5), nn.Linear(d_model + 5, 64), nn.Tanh(),
            )
            self.keys = nn.Linear(64, 64)
            self.values = nn.Linear(64, 64)
            self.queries = nn.Parameter(torch.randn(4, 64) / 8.0)
            self.pool = nn.Sequential(nn.Linear(4 * 64, 16), nn.Tanh())
            self.state_norm = nn.LayerNorm(recurrent_dim + 16)
            self.state_fusion = nn.Linear(recurrent_dim + 16, hidden_dim)
        nn.init.zeros_(self.state_fusion.weight)
        nn.init.zeros_(self.state_fusion.bias)

    def forward(self, observation, *, hidden, base_rank, eligible, tradable,
                predicted_return, previous_selected, holding_age, actor_state):
        dates, stocks = base_rank.shape
        if (hidden.shape[:2] != (dates, stocks)
                or any(x.shape != base_rank.shape for x in (
                    eligible, tradable, predicted_return, previous_selected, holding_age))
                or actor_state.shape != (dates, self.state_norm.normalized_shape[0] - 16)
                or observation.shape[0] != dates):
            raise ValueError("enhanced critic state axes disagree")
        rows = []
        for date in range(dates):
            held = torch.nonzero(previous_selected[date] & eligible[date], as_tuple=False).flatten()
            if len(held) > self.candidate_max_count:
                raise ValueError("enhanced critic cap would discard held stocks")
            fresh = torch.nonzero(
                eligible[date] & tradable[date] & ~previous_selected[date]
                & (base_rank[date] >= 0.85), as_tuple=False,
            ).flatten()
            if len(fresh):
                fresh = fresh[base_rank[date, fresh].argsort(descending=True)]
            rows.append(torch.cat((held, fresh[:self.candidate_max_count - len(held)])))
        width = max(max(len(row) for row in rows), 1)
        indices = torch.zeros((dates, width), dtype=torch.long, device=hidden.device)
        valid = torch.zeros_like(indices, dtype=torch.bool)
        for date, row in enumerate(rows):
            indices[date, :len(row)] = row
            valid[date, :len(row)] = True
            if not len(row):
                valid[date, 0] = True  # masked empty output below; avoid NaN softmax.
        age = holding_age.detach().gather(1, indices)
        metadata = torch.stack((
            previous_selected.gather(1, indices).to(hidden.dtype),
            torch.log1p(age) / math.log(257.0),  # no clipping at 20 holding days
            base_rank.detach().gather(1, indices),
            (predicted_return.detach().gather(1, indices) / self.return_scale).clamp(-5, 5),
            tradable.gather(1, indices).to(hidden.dtype),
        ), dim=-1)
        day = torch.arange(dates, device=hidden.device)[:, None]
        tokens = self.stock_encoder(torch.cat((hidden.detach()[day, indices], metadata), dim=-1))
        attention = torch.softmax(
            ((self.queries[None] @ self.keys(tokens).transpose(1, 2)) / 8.0)
            .masked_fill(~valid[:, None, :], -torch.inf), dim=-1,
        )
        portfolio = self.pool((attention @ self.values(tokens)).reshape(dates, -1))
        nonempty = torch.tensor([len(row) > 0 for row in rows], device=hidden.device)
        portfolio = torch.where(nonempty[:, None], portfolio, torch.zeros_like(portfolio))
        state = self.state_norm(torch.cat((actor_state.detach(), portfolio), dim=-1))
        encoded = self.network[1](self.network[0](observation.detach())) + self.state_fusion(state)
        return self.network[3](self.network[2](encoded)).squeeze(-1)


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


def discounted_return_to_go(
    reward: torch.Tensor, *, gamma: float = 0.99, horizon: int | None = None,
) -> torch.Tensor:
    """Discounted future reward, optionally truncated to ``horizon`` days.

    The final dates of a block use only rewards that are actually in the block;
    no future labels or value bootstrap are read across its boundary.
    """
    if reward.ndim != 1 or not 0 <= gamma <= 1:
        raise ValueError("reward must be one-dimensional and gamma in [0, 1]")
    if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
        raise ValueError("horizon must be a positive integer or None")
    returns = torch.zeros_like(reward)
    carry = reward.new_zeros(())
    for date in range(len(reward) - 1, -1, -1):
        carry = reward[date] + gamma * carry
        if horizon is not None and date + horizon < len(reward):
            carry = carry - gamma ** horizon * reward[date + horizon]
        returns[date] = carry
    return returns


def center_daily_rewards_against_group(
    local_rewards: torch.Tensor, group_rewards: torch.Tensor,
) -> torch.Tensor:
    """Remove the shared date shock before fitting a within-day action critic.

    The baseline is formed from all same-block rollouts across ranks. It is
    used only for training targets; the exact reward and validation score stay
    unchanged. The actor's own rollout contributes to its group baseline, as
    it already does in the existing same-date GRPO normalization.
    """
    if (
        local_rewards.ndim != 2 or group_rewards.ndim != 2
        or local_rewards.shape[1] != group_rewards.shape[1]
        or group_rewards.shape[0] < local_rewards.shape[0]
        or group_rewards.shape[0] < 2
    ):
        raise ValueError("rewards must be [local/group rollouts, matching dates]")
    return local_rewards - group_rewards.mean(dim=0, keepdim=True)
