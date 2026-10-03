from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn


def masked_cross_sectional_standardize(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Standardize every date over currently eligible stocks only."""
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must both be [dates, stocks]")
    mask = mask.to(values.device, dtype=torch.bool)
    weights = mask.to(values.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=1, keepdim=True) / count
    centered = (values - mean) * weights
    variance = centered.square().sum(dim=1, keepdim=True) / count
    normalized = centered / (variance.sqrt() + float(epsilon))
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def hard_top_fraction_mask(
    scores: torch.Tensor, tradable: torch.Tensor, fraction: float = 0.1,
) -> torch.Tensor:
    """Select the hard top fraction independently for each date."""
    if scores.ndim != 2 or tradable.shape != scores.shape:
        raise ValueError("scores and tradable must both be [dates, stocks]")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    selected = torch.zeros_like(tradable, dtype=torch.bool)
    for date in range(scores.shape[0]):
        indices = torch.nonzero(tradable[date].bool(), as_tuple=False).reshape(-1)
        count = max(int(indices.numel() * float(fraction)), 1)
        if indices.numel() == 0:
            continue
        chosen = torch.topk(scores[date, indices], k=min(count, indices.numel())).indices
        selected[date, indices[chosen]] = True
    return selected


class PolicyOutput(NamedTuple):
    fresh_score: torch.Tensor
    gate_new: torch.Tensor
    mean_score: torch.Tensor
    sampled_score: torch.Tensor
    selected: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reference_kl: torch.Tensor
    policy_std: torch.Tensor


class FinAxialPolicyHead(nn.Module):
    """Small recurrent decision policy placed on top of frozen FinAxial states."""

    def __init__(
        self,
        d_model: int = 128,
        *,
        adapter_dim: int = 32,
        state_dim: int = 16,
        gate_hidden_dim: int = 32,
        residual_scale: float = 0.1,
        initial_retention: float = 0.02,
        maximum_retention: float = 0.95,
        initial_std: float = 0.10,
        minimum_std: float = 0.01,
        maximum_std: float = 0.30,
        top_fraction: float = 0.10,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 <= initial_retention < maximum_retention < 1.0:
            raise ValueError("retention must satisfy 0 <= initial < maximum < 1")
        if not 0.0 < minimum_std < initial_std < maximum_std:
            raise ValueError("std must satisfy minimum < initial < maximum")
        self.residual_scale = float(residual_scale)
        self.maximum_retention = float(maximum_retention)
        self.minimum_std = float(minimum_std)
        self.maximum_std = float(maximum_std)
        self.top_fraction = float(top_fraction)
        self.epsilon = float(epsilon)

        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, 1),
        )
        self.stock_state = nn.Linear(d_model, state_dim)
        self.market_state = nn.Linear(d_model, state_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * state_dim + 4, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        retention_probability = max(initial_retention / maximum_retention, 1e-6)
        retention_logit = math.log(
            retention_probability / (1.0 - retention_probability)
        )
        std_probability = (initial_std - minimum_std) / (maximum_std - minimum_std)
        std_logit = math.log(std_probability / (1.0 - std_probability))
        self.logit_std = nn.Parameter(torch.tensor(std_logit, dtype=torch.float32))

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, retention_logit)

    @property
    def policy_std(self) -> torch.Tensor:
        return self.minimum_std + (
            self.maximum_std - self.minimum_std
        ) * torch.sigmoid(self.logit_std)

    def forward(
        self,
        hidden: torch.Tensor,
        base_score: torch.Tensor,
        eligible: torch.Tensor,
        tradable: torch.Tensor,
        *,
        sample: bool,
        noise: torch.Tensor | None = None,
    ) -> PolicyOutput:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [dates, stocks, d_model]")
        if base_score.shape != hidden.shape[:2]:
            raise ValueError("base_score must match hidden dates/stocks")
        if eligible.shape != base_score.shape or tradable.shape != base_score.shape:
            raise ValueError("eligible/tradable must match base_score")
        eligible = eligible.to(hidden.device, dtype=torch.bool)
        tradable = tradable.to(hidden.device, dtype=torch.bool)
        base_score = base_score.to(hidden.device)
        reference = masked_cross_sectional_standardize(
            base_score, eligible, epsilon=self.epsilon,
        )
        residual = self.adapter(hidden).squeeze(-1)
        fresh = masked_cross_sectional_standardize(
            base_score + self.residual_scale * residual,
            eligible,
            epsilon=self.epsilon,
        )
        weights = eligible.to(hidden.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        market_hidden = (hidden * weights[..., None]).sum(dim=1) / count
        stock_state = self.stock_state(hidden)
        market_state = self.market_state(market_hidden)

        if noise is not None and noise.shape != base_score.shape:
            raise ValueError("noise must match base_score")
        std = self.policy_std.to(dtype=hidden.dtype, device=hidden.device)
        previous_score = fresh[0].detach()
        previous_selected = torch.zeros_like(eligible[0], dtype=torch.bool)
        means, actions, gates, selections = [], [], [], []
        log_prob_sum = hidden.new_zeros(())
        entropy_sum = hidden.new_zeros(())
        kl_sum = hidden.new_zeros(())
        normal_constant = 0.5 * math.log(2.0 * math.pi)

        for date in range(hidden.shape[0]):
            scalars = torch.stack((
                fresh[date],
                previous_score,
                fresh[date] - previous_score,
                previous_selected.to(hidden.dtype),
            ), dim=-1)
            gate_input = torch.cat((
                stock_state[date],
                market_state[date][None, :].expand(hidden.shape[1], -1),
                scalars,
            ), dim=-1)
            retention = self.maximum_retention * torch.sigmoid(
                self.gate(gate_input).squeeze(-1)
            )
            gate_new = 1.0 - retention
            mean = gate_new * fresh[date] + retention * previous_score
            mean = masked_cross_sectional_standardize(
                mean[None, :], eligible[date][None, :], epsilon=self.epsilon,
            )[0]
            epsilon = (
                torch.randn_like(mean) if noise is None else noise[date].to(mean)
            )
            prestandardized_action = mean + std * epsilon if sample else mean
            # Detach sampled actions for the score-function estimator.
            action_for_log_prob = prestandardized_action.detach()
            action = masked_cross_sectional_standardize(
                action_for_log_prob[None, :],
                eligible[date][None, :],
                epsilon=self.epsilon,
            )[0]
            mask = eligible[date].to(hidden.dtype)
            denominator = mask.sum().clamp_min(1.0)
            standardized_error = (action_for_log_prob - mean) / std
            log_probability = -(
                0.5 * standardized_error.square() + std.log() + normal_constant
            )
            log_prob_sum = log_prob_sum + (log_probability * mask).sum() / denominator
            entropy_sum = entropy_sum + std.log() + normal_constant + 0.5
            kl_sum = kl_sum + (
                0.5 * (mean - reference[date]).square() * mask / std.square()
            ).sum() / denominator
            selected = hard_top_fraction_mask(
                action[None, :], tradable[date][None, :], self.top_fraction,
            )[0]
            means.append(mean)
            actions.append(action)
            gates.append(gate_new)
            selections.append(selected)
            previous_score = action.detach()
            previous_selected = selected.detach()

        dates = float(hidden.shape[0])
        return PolicyOutput(
            fresh_score=fresh,
            gate_new=torch.stack(gates),
            mean_score=torch.stack(means),
            sampled_score=torch.stack(actions),
            selected=torch.stack(selections),
            log_prob=log_prob_sum / dates,
            entropy=entropy_sum / dates,
            reference_kl=kl_sum / dates,
            policy_std=std,
        )
