from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from .finaxial_policy import hard_top_fraction_mask
from .topk_actions import (
    project_selection_to_scores, select_hysteresis, select_swap_budget,
)


def _average_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1)
    if values.numel() < 2:
        return torch.zeros_like(values)
    sorted_values, order = torch.sort(values, stable=True)
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    groups = starts.cumsum(dim=0) - 1
    group_count = int(groups[-1].item()) + 1
    positions = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    sums = torch.zeros(group_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, groups, positions)
    counts.scatter_add_(0, groups, torch.ones_like(positions))
    sorted_ranks = (sums / counts.clamp_min(1.0))[groups]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def percentile_rank_scores(
    scores: torch.Tensor, eligible: torch.Tensor,
) -> torch.Tensor:
    """Tie-aware daily percentile ranks, including invalid median fallbacks."""
    if scores.ndim != 2 or eligible.shape != scores.shape:
        raise ValueError("scores and eligible must both be [dates, stocks]")
    rows = []
    for date in range(scores.shape[0]):
        valid = eligible[date].bool() & torch.isfinite(scores[date])
        if bool(valid.any()):
            ordered = scores[date, valid].sort().values
            middle = ordered.numel() // 2
            fallback = (
                ordered[middle]
                if ordered.numel() % 2
                else (ordered[middle - 1] + ordered[middle]) * 0.5
            )
        else:
            fallback = scores.new_zeros(())
        filled = torch.where(valid, scores[date], fallback)
        rows.append((_average_rank(filled.float()) + 1.0) / float(scores.shape[1]))
    return torch.stack(rows).to(scores.dtype)


def causal_ewma_scores(base_rank: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
    if base_rank.ndim != 2 or not 0.0 < alpha <= 1.0:
        raise ValueError("base_rank must be [dates, stocks] and alpha in (0, 1]")
    rows = [base_rank[0]]
    for date in range(1, base_rank.shape[0]):
        rows.append(float(alpha) * base_rank[date] + (1.0 - float(alpha)) * rows[-1])
    return torch.stack(rows)


class DecisionPolicyOutput(NamedTuple):
    decision_score: torch.Tensor
    selected: torch.Tensor
    raw_action: torch.Tensor
    alpha: torch.Tensor
    delta: torch.Tensor
    hold_bonus: torch.Tensor
    hold_threshold: torch.Tensor
    hold_temperature: torch.Tensor
    action_mean: torch.Tensor
    action_value: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reference_kl: torch.Tensor
    policy_std: torch.Tensor
    swap_budget: torch.Tensor
    realized_swaps: torch.Tensor
    candidate_bonus_abs_mean: torch.Tensor
    candidate_count: torch.Tensor


class FinAxialDecisionPolicy(nn.Module):
    """Low-dimensional recurrent policy for turnover-aware Top-10 decisions."""

    def __init__(
        self,
        d_model: int = 128,
        *,
        market_dim: int = 16,
        recurrent_dim: int = 32,
        action_mode: str = "uniform2",
        alpha_minimum: float = 0.05,
        alpha_maximum: float = 0.95,
        initial_alpha: float = 0.25,
        maximum_delta: float = 0.02,
        maximum_rank_bonus: float = 0.15,
        hold_threshold_minimum: float = 0.70,
        hold_threshold_maximum: float = 0.99,
        initial_hold_threshold: float = 0.90,
        hold_temperature_minimum: float = 0.01,
        hold_temperature_maximum: float = 0.15,
        initial_hold_temperature: float = 0.05,
        rank_bonus_buckets: int = 5,
        initial_action_std: float = 0.20,
        minimum_action_std: float = 0.05,
        maximum_action_std: float = 0.50,
        return_scale: float = 0.02,
        top_fraction: float = 0.10,
        epsilon: float = 1e-6,
        initial_swap_budget: int = 10,
        maximum_swap_budget: int = 60,
        initial_hysteresis_margin: float = 0.002,
        maximum_hysteresis_margin: float = 0.02,
        boundary_width: float = 0.08,
        history_days: int = 0,
        history_hidden_dim: int = 32,
        history_context_dim: int = 32,
        history_boundary_width: float = 0.05,
        observation_mode: str = "mean_pool",
        candidate_top_fraction: float = 0.15,
        candidate_max_count: int = 1024,
        candidate_context_dim: int = 64,
        candidate_queries: int = 4,
        maximum_candidate_bonus: float = 0.04,
        candidate_boundary_width: float = 0.08,
        return_feature_mode: str = "proxy",
    ) -> None:
        super().__init__()
        if not 0 < alpha_minimum < initial_alpha < alpha_maximum < 1:
            raise ValueError("alpha bounds must contain initial_alpha inside (0, 1)")
        if not 0 < minimum_action_std < initial_action_std < maximum_action_std:
            raise ValueError("action std bounds are invalid")
        if maximum_delta <= 0 or maximum_rank_bonus <= 0 or return_scale <= 0:
            raise ValueError("delta, rank bonus and return scale must be positive")
        if action_mode not in {
            "uniform2", "selective4", "bucketed6", "swap_budget_only",
            "swap_budget_raw",
            "swap_budget_alpha", "hysteresis", "boundary4", "candidate_residual",
        }:
            raise ValueError(f"unsupported action_mode: {action_mode!r}")
        if not 0 < initial_swap_budget < maximum_swap_budget:
            raise ValueError("swap budget bounds are invalid")
        if not 0 < initial_hysteresis_margin < maximum_hysteresis_margin:
            raise ValueError("hysteresis margin bounds are invalid")
        if boundary_width <= 0:
            raise ValueError("boundary width must be positive")
        if history_days < 0 or (history_days == 1) or history_hidden_dim <= 0 or history_context_dim <= 0:
            raise ValueError("history_days must be 0 or >= 2, with positive context dimensions")
        if history_boundary_width <= 0:
            raise ValueError("history boundary width must be positive")
        if observation_mode not in {"mean_pool", "candidate_attention"}:
            raise ValueError("unsupported observation mode")
        if observation_mode == "candidate_attention" and history_days < 2:
            raise ValueError("candidate attention needs at least two history days")
        if action_mode == "candidate_residual" and observation_mode != "candidate_attention":
            raise ValueError("candidate residual actions need candidate attention")
        if return_feature_mode not in {"proxy", "explicit"}:
            raise ValueError("return feature mode must be proxy or explicit")
        if return_feature_mode == "explicit" and observation_mode != "candidate_attention":
            raise ValueError("explicit returns need candidate attention")
        if observation_mode == "candidate_attention" and not top_fraction < candidate_top_fraction <= 1.0:
            raise ValueError("candidate top fraction must exceed portfolio fraction")
        if candidate_max_count < 1 or candidate_context_dim < 1 or candidate_queries < 1:
            raise ValueError("candidate capacity must be positive")
        if maximum_candidate_bonus <= 0 or candidate_boundary_width <= 0:
            raise ValueError("candidate bonus and boundary width must be positive")
        if not (
            0.0 < hold_threshold_minimum < initial_hold_threshold
            < hold_threshold_maximum < 1.0
        ):
            raise ValueError("hold threshold bounds are invalid")
        if not (
            0.0 < hold_temperature_minimum < initial_hold_temperature
            < hold_temperature_maximum
        ):
            raise ValueError("hold temperature bounds are invalid")
        if action_mode == "bucketed6" and rank_bonus_buckets != 5:
            raise ValueError("bucketed6 requires exactly five rank bonus buckets")
        self.action_mode = str(action_mode)
        self.alpha_minimum = float(alpha_minimum)
        self.alpha_maximum = float(alpha_maximum)
        self.maximum_delta = float(maximum_delta)
        self.maximum_rank_bonus = float(maximum_rank_bonus)
        self.hold_threshold_minimum = float(hold_threshold_minimum)
        self.hold_threshold_maximum = float(hold_threshold_maximum)
        self.hold_temperature_minimum = float(hold_temperature_minimum)
        self.hold_temperature_maximum = float(hold_temperature_maximum)
        self.rank_bonus_buckets = int(rank_bonus_buckets)
        self.minimum_action_std = float(minimum_action_std)
        self.maximum_action_std = float(maximum_action_std)
        self.return_scale = float(return_scale)
        self.top_fraction = float(top_fraction)
        self.epsilon = float(epsilon)
        self.initial_alpha = float(initial_alpha)
        self.initial_swap_budget = int(initial_swap_budget)
        self.maximum_swap_budget = int(maximum_swap_budget)
        self.initial_hysteresis_margin = float(initial_hysteresis_margin)
        self.maximum_hysteresis_margin = float(maximum_hysteresis_margin)
        self.boundary_width = float(boundary_width)
        self.history_days = int(history_days)
        self.history_boundary_width = float(history_boundary_width)
        self.observation_mode = str(observation_mode)
        self.candidate_top_fraction = float(candidate_top_fraction)
        self.candidate_max_count = int(candidate_max_count)
        self.maximum_candidate_bonus = float(maximum_candidate_bonus)
        self.candidate_boundary_width = float(candidate_boundary_width)
        self.return_feature_mode = str(return_feature_mode)

        self.market_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, market_dim),
            nn.Tanh(),
        )
        # Six interpretable market/portfolio scalars accompany pooled C0 state.
        if self.observation_mode == "candidate_attention":
            # Encode only the held/high-scoring candidate union. The queries
            # attend to stocks, never to future dates or the whole N^2 universe.
            fusion_width = d_model + history_hidden_dim + (
                5 if self.return_feature_mode == "explicit" else 4
            )
            self.history_encoder = nn.GRU(
                input_size=7, hidden_size=history_hidden_dim, batch_first=True,
            )
            self.stock_history_fusion = nn.Sequential(
                nn.LayerNorm(fusion_width),
                nn.Linear(fusion_width, candidate_context_dim),
                nn.Tanh(),
            )
            self.candidate_keys = nn.Linear(candidate_context_dim, candidate_context_dim)
            self.candidate_values = nn.Linear(candidate_context_dim, candidate_context_dim)
            self.candidate_query = nn.Parameter(torch.randn(
                candidate_queries, candidate_context_dim,
            ) / math.sqrt(candidate_context_dim))
            self.portfolio_encoder = nn.Sequential(
                nn.Linear(candidate_queries * candidate_context_dim, market_dim),
                nn.Tanh(),
            )
        elif self.history_days:
            # Each date/stock is encoded from exactly its trailing history_days
            # observations. The module is shared across every stock and date.
            self.history_encoder = nn.GRU(
                input_size=4, hidden_size=history_hidden_dim, batch_first=True,
            )
            self.stock_history_fusion = nn.Sequential(
                nn.LayerNorm(d_model + history_hidden_dim),
                nn.Linear(d_model + history_hidden_dim, history_context_dim),
                nn.Tanh(),
            )
            self.portfolio_encoder = nn.Sequential(
                nn.Linear(3 * history_context_dim, market_dim),
                nn.Tanh(),
            )
        self.recurrent = nn.GRUCell(
            market_dim * (2 if self.history_days else 1) + 6, recurrent_dim,
        )
        action_count = {
            "uniform2": 2,
            "selective4": 4,
            "bucketed6": 1 + self.rank_bonus_buckets,
            "swap_budget_only": 1,
            "swap_budget_raw": 1,
            "swap_budget_alpha": 2,
            "hysteresis": 3,
            "candidate_residual": 9,
            "boundary4": 4,
        }[self.action_mode]
        self.action_count = int(action_count)
        self.action_head = nn.Linear(recurrent_dim, self.action_count)

        alpha_probability = (
            (float(initial_alpha) - self.alpha_minimum)
            / (self.alpha_maximum - self.alpha_minimum)
        )
        initial_alpha_logit = math.log(alpha_probability / (1.0 - alpha_probability))
        reference_action_mean = torch.zeros(self.action_count, dtype=torch.float32)
        if self.action_mode not in {"swap_budget_only", "swap_budget_raw"}:
            reference_action_mean[0] = initial_alpha_logit
        if self.action_mode in {"hysteresis", "candidate_residual"}:
            margin_probability = (
                self.initial_hysteresis_margin / self.maximum_hysteresis_margin
            )
            reference_action_mean[1] = math.log(
                margin_probability / (1.0 - margin_probability)
            )
        if self.action_mode == "selective4":
            threshold_probability = (
                (float(initial_hold_threshold) - self.hold_threshold_minimum)
                / (self.hold_threshold_maximum - self.hold_threshold_minimum)
            )
            temperature_probability = (
                (float(initial_hold_temperature) - self.hold_temperature_minimum)
                / (self.hold_temperature_maximum - self.hold_temperature_minimum)
            )
            reference_action_mean[2] = math.log(
                threshold_probability / (1.0 - threshold_probability)
            )
            reference_action_mean[3] = math.log(
                temperature_probability / (1.0 - temperature_probability)
            )
        self.register_buffer("reference_action_mean", reference_action_mean)
        std_probability = (
            (float(initial_action_std) - self.minimum_action_std)
            / (self.maximum_action_std - self.minimum_action_std)
        )
        self.logit_std = nn.Parameter(torch.full(
            (self.action_count,), math.log(std_probability / (1.0 - std_probability)),
            dtype=torch.float32,
        ))
        self.register_buffer(
            "reference_std", torch.full((self.action_count,), float(initial_action_std)),
        )
        nn.init.zeros_(self.action_head.weight)
        with torch.no_grad():
            self.action_head.bias.copy_(self.reference_action_mean)

    @property
    def policy_std(self) -> torch.Tensor:
        return self.minimum_action_std + (
            self.maximum_action_std - self.minimum_action_std
        ) * torch.sigmoid(self.logit_std)

    def _alpha_delta(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.alpha_minimum + (
            self.alpha_maximum - self.alpha_minimum
        ) * torch.sigmoid(raw_action[..., 0])
        # Signed delta is useful: positive values discourage turnover, while a
        # negative value deliberately retires stale holdings.
        delta = self.maximum_delta * torch.tanh(raw_action[..., 1])
        return alpha, delta

    def _selective_actions(
        self, raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, _ = self._alpha_delta(raw_action)
        bonus = self.maximum_rank_bonus * torch.tanh(raw_action[..., 1])
        threshold = self.hold_threshold_minimum + (
            self.hold_threshold_maximum - self.hold_threshold_minimum
        ) * torch.sigmoid(raw_action[..., 2])
        temperature = self.hold_temperature_minimum + (
            self.hold_temperature_maximum - self.hold_temperature_minimum
        ) * torch.sigmoid(raw_action[..., 3])
        return alpha, bonus, threshold, temperature

    def _encode_stock_history(
        self,
        hidden: torch.Tensor,
        base_score: torch.Tensor,
        base_rank: torch.Tensor,
        eligible: torch.Tensor,
        observed_return: torch.Tensor,
        history_prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if observed_return.shape != base_score.shape:
            raise ValueError("observed_return must match [dates, stocks]")
        current = torch.stack((
            base_rank, base_score, observed_return, eligible.to(base_score.dtype),
        ), dim=-1)
        required_prefix = self.history_days - 1
        if history_prefix is None:
            prefix = current.new_zeros((required_prefix, base_score.shape[1], 4))
        else:
            prefix_score, prefix_eligible, prefix_return = history_prefix
            if (
                prefix_score.shape != (required_prefix, base_score.shape[1])
                or prefix_eligible.shape != prefix_score.shape
                or prefix_return.shape != prefix_score.shape
            ):
                raise ValueError("history prefix must cover exactly history_days - 1 dates")
            prefix_score = prefix_score.to(base_score)
            prefix_eligible = prefix_eligible.to(device=base_score.device, dtype=torch.bool)
            prefix_return = prefix_return.to(base_score)
            prefix_rank = percentile_rank_scores(prefix_score, prefix_eligible)
            prefix = torch.stack((
                prefix_rank, prefix_score, prefix_return,
                prefix_eligible.to(base_score.dtype),
            ), dim=-1)
        timeline = torch.cat((prefix, current), dim=0)
        windows = timeline.unfold(0, self.history_days, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous().reshape(
            -1, self.history_days, 4,
        )
        _, state = self.history_encoder(windows)
        encoded = state[-1].reshape(*base_score.shape, -1)
        return self.stock_history_fusion(torch.cat((hidden, encoded), dim=-1))

    def _candidate_timeline(
        self,
        base_score: torch.Tensor,
        base_rank: torch.Tensor,
        eligible: torch.Tensor,
        observed_features: torch.Tensor,
        history_prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if observed_features.shape != (*base_score.shape, 4):
            raise ValueError("candidate attention needs [dates, stocks, 4] observed features")
        current = torch.cat((
            base_rank[..., None], base_score[..., None],
            eligible.to(base_score.dtype)[..., None], observed_features,
        ), dim=-1)
        needed = self.history_days - 1
        if history_prefix is None:
            prefix = current.new_zeros((needed, base_score.shape[1], 7))
        else:
            prefix_score, prefix_eligible, prefix_features = history_prefix
            if (
                prefix_score.shape != (needed, base_score.shape[1])
                or prefix_eligible.shape != prefix_score.shape
                or prefix_features.shape != (*prefix_score.shape, 4)
            ):
                raise ValueError("candidate history prefix has incorrect shape")
            prefix_score = prefix_score.to(base_score)
            prefix_eligible = prefix_eligible.to(base_score.device, dtype=torch.bool)
            prefix_rank = percentile_rank_scores(prefix_score, prefix_eligible)
            prefix = torch.cat((
                prefix_rank[..., None], prefix_score[..., None],
                prefix_eligible.to(base_score.dtype)[..., None],
                prefix_features.to(base_score),
            ), dim=-1)
        return torch.cat((prefix, current), dim=0)

    def _candidate_context(
        self,
        date: int,
        timeline: torch.Tensor,
        hidden: torch.Tensor,
        base_rank: torch.Tensor,
        eligible: torch.Tensor,
        tradable: torch.Tensor,
        previous_selected: torch.Tensor,
        holding_age: torch.Tensor,
        predicted_excess: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_trade = eligible[date] & tradable[date]
        available = torch.nonzero(valid_trade, as_tuple=False).flatten()
        if available.numel():
            count = min(
                self.candidate_max_count,
                max(1, int(math.ceil(float(available.numel()) * self.candidate_top_fraction))),
            )
            top = available[torch.topk(base_rank[date, available], k=count).indices]
        else:
            top = available
        held = torch.nonzero(previous_selected & tradable[date], as_tuple=False).flatten()
        # Keep held names first if a small candidate cap is configured.
        if held.numel() >= self.candidate_max_count:
            held = held[torch.topk(base_rank[date, held], k=self.candidate_max_count).indices]
            indices = held
        else:
            held_mask = torch.zeros_like(previous_selected)
            held_mask[held] = True
            fresh = top[~held_mask[top]]
            indices = torch.cat((held, fresh[:self.candidate_max_count - held.numel()]))
        if not indices.numel():
            return (
                hidden.new_zeros(self.portfolio_encoder[0].out_features),
                indices, hidden.new_zeros((0, 6)),
            )
        window = timeline[date:date + self.history_days, indices].permute(1, 0, 2)
        _, state = self.history_encoder(window)
        latest = window[:, -1]
        cutoff_rank = 1.0 - self.top_fraction
        held_float = previous_selected[indices].to(hidden.dtype)
        gap = ((base_rank[date, indices] - cutoff_rank)
               / self.candidate_boundary_width).clamp(-2.0, 2.0)
        metadata_parts = [
            held_float, (holding_age[indices] / 20.0).clamp(0.0, 1.0),
            base_rank[date, indices], gap,
        ]
        if self.return_feature_mode == "explicit":
            forecast = (
                predicted_excess[date, indices] / self.return_scale
            ).clamp(-5.0, 5.0)
            metadata_parts.append(forecast)
        metadata = torch.stack(metadata_parts, dim=-1)
        stock_context = self.stock_history_fusion(torch.cat((
            hidden[date, indices], state[-1], metadata,
        ), dim=-1))
        keys = self.candidate_keys(stock_context)
        values = self.candidate_values(stock_context)
        attention = torch.softmax(
            (self.candidate_query @ keys.transpose(0, 1)) / math.sqrt(keys.shape[-1]),
            dim=-1,
        )
        pooled = attention @ values
        context = self.portfolio_encoder(pooled.reshape(-1))
        recent = window[:, -min(5, self.history_days):]
        rank_momentum = (latest[:, 0] - recent[:, 0, 0]) * 5.0
        close_momentum = recent[:, :, 3].mean(dim=1) / 3.0
        basis = torch.stack((
            held_float * 2.0 - 1.0,
            rank_momentum.clamp(-1.0, 1.0),
            close_momentum.clamp(-1.0, 1.0),
            (
                forecast / 3.0 if self.return_feature_mode == "explicit"
                else latest[:, 4] / 3.0
            ).clamp(-1.0, 1.0),
            (latest[:, 6] / 3.0).clamp(-1.0, 1.0),
            gap.clamp(-1.0, 1.0),
        ), dim=-1)
        return context, indices, basis

    def forward(
        self,
        hidden: torch.Tensor,
        base_score: torch.Tensor,
        eligible: torch.Tensor,
        tradable: torch.Tensor,
        *,
        sample: bool = False,
        actions: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        observed_return: torch.Tensor | None = None,
        history_prefix: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        predicted_return: torch.Tensor | None = None,
    ) -> DecisionPolicyOutput:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [dates, stocks, d_model]")
        if base_score.shape != hidden.shape[:2]:
            raise ValueError("base_score must match hidden dates/stocks")
        if eligible.shape != base_score.shape or tradable.shape != base_score.shape:
            raise ValueError("eligible/tradable must match base_score")
        action_shape = (*base_score.shape[:1], self.action_count)
        if actions is not None and actions.shape != action_shape:
            raise ValueError(f"actions must be {action_shape}")
        if noise is not None and noise.shape != action_shape:
            raise ValueError(f"noise must be {action_shape}")
        if self.return_feature_mode == "explicit" and (
            predicted_return is None or predicted_return.shape != base_score.shape
        ):
            raise ValueError("explicit return forecast must match [dates, stocks]")
        if self.return_feature_mode == "proxy" and predicted_return is not None:
            raise ValueError("proxy policy does not accept an explicit return forecast")
        if actions is not None and (sample or noise is not None):
            raise ValueError("fixed actions cannot be combined with sampling/noise")

        eligible = eligible.to(hidden.device, dtype=torch.bool)
        tradable = tradable.to(hidden.device, dtype=torch.bool)
        base_score = base_score.to(hidden.device)
        base_rank = percentile_rank_scores(base_score, eligible)
        candidate_timeline = None
        if self.observation_mode == "candidate_attention":
            if observed_return is None:
                raise ValueError("candidate attention requires observed market features")
            candidate_timeline = self._candidate_timeline(
                base_score, base_rank, eligible, observed_return.to(base_score),
                history_prefix,
            )
            stock_history = None
        elif self.history_days:
            if observed_return is None:
                raise ValueError("history-aware policy requires observed_return")
            stock_history = self._encode_stock_history(
                hidden, base_score, base_rank, eligible,
                observed_return.to(base_score), history_prefix,
            )
        else:
            if observed_return is not None or history_prefix is not None:
                raise ValueError("legacy policy does not accept history inputs")
            stock_history = None
        if predicted_return is None:
            predicted_excess = base_score * self.return_scale
        else:
            absolute = predicted_return.to(base_score)
            valid_weight = eligible.to(absolute.dtype)
            market_prediction = (absolute * valid_weight).sum(dim=1, keepdim=True) / (
                valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            predicted_excess = absolute - market_prediction
        weights = eligible.to(hidden.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        market_hidden = (hidden * weights[..., None]).sum(dim=1) / counts
        market_encoded = self.market_encoder(market_hidden)

        actor_hidden = hidden.new_zeros(self.recurrent.hidden_size)
        previous_score = base_rank[0].detach()
        previous_selected = hard_top_fraction_mask(
            base_rank[0:1], tradable[0:1], self.top_fraction,
        )[0]
        holding_age = base_score.new_zeros(base_score.shape[1])
        std = self.policy_std.to(device=hidden.device, dtype=hidden.dtype)
        reference_mean = self.reference_action_mean.to(hidden)
        reference_std = self.reference_std.to(hidden)
        normal_constant = 0.5 * math.log(2.0 * math.pi)
        scores, selected_rows, raw_actions, means = [], [], [], []
        action_values, alphas, deltas = [], [], []
        hold_bonuses, hold_thresholds, hold_temperatures = [], [], []
        swap_budgets, realized_swaps = [], []
        candidate_bonus_abs_means, candidate_counts = [], []
        log_prob_rows = []
        entropy_sum = hidden.new_zeros(())
        kl_sum = hidden.new_zeros(())

        for date in range(hidden.shape[0]):
            valid = eligible[date]
            trade = tradable[date]
            valid_mu = predicted_excess[date, valid]
            mu_std = valid_mu.std(unbiased=False).clamp_min(self.epsilon)
            trade_indices = torch.nonzero(trade, as_tuple=False).reshape(-1)
            count = max(int(trade_indices.numel() * self.top_fraction), 1)
            if trade_indices.numel():
                top_positions = torch.topk(
                    predicted_excess[date, trade_indices],
                    k=min(count, trade_indices.numel()),
                ).indices
                top_indices = trade_indices[top_positions]
                cutoff = predicted_excess[date, top_indices].min()
                top_mean = predicted_excess[date, top_indices].mean()
            else:
                cutoff = predicted_excess.new_zeros(())
                top_mean = predicted_excess.new_zeros(())
            if bool(previous_selected.any()):
                previous_mean = predicted_excess[date, previous_selected].mean()
            else:
                previous_mean = predicted_excess.new_zeros(())
            naive_selected = hard_top_fraction_mask(
                base_rank[date:date + 1], tradable[date:date + 1], self.top_fraction,
            )[0]
            union = (naive_selected | previous_selected).sum().clamp_min(1)
            jaccard = (naive_selected & previous_selected).sum().float() / union.float()
            scalars = torch.stack((
                mu_std / self.return_scale,
                cutoff / self.return_scale,
                top_mean / self.return_scale,
                previous_mean / self.return_scale,
                1.0 - jaccard,
                previous_selected.float().mean(),
            )).to(hidden.dtype)
            if candidate_timeline is not None:
                portfolio_context, candidate_indices, candidate_basis = self._candidate_context(
                    date, candidate_timeline, hidden, base_rank, eligible,
                    tradable, previous_selected, holding_age, predicted_excess,
                )
                actor_input = torch.cat((
                    market_encoded[date], portfolio_context, scalars,
                ))
            elif stock_history is not None:
                def masked_pool(mask: torch.Tensor) -> torch.Tensor:
                    weight = mask.to(stock_history.dtype)
                    return (stock_history[date] * weight[:, None]).sum(dim=0) / (
                        weight.sum().clamp_min(1.0)
                    )

                cutoff_rank = 1.0 - self.top_fraction
                near_boundary = (
                    (base_rank[date] - cutoff_rank).abs()
                    <= self.history_boundary_width
                ) & valid & trade
                portfolio_context = self.portfolio_encoder(torch.cat((
                    masked_pool(valid),
                    masked_pool(previous_selected & valid),
                    masked_pool(near_boundary),
                )))
                actor_input = torch.cat((
                    market_encoded[date], portfolio_context, scalars,
                ))
            else:
                actor_input = torch.cat((market_encoded[date], scalars))
            actor_hidden = self.recurrent(
                actor_input, actor_hidden,
            )
            action_mean = self.action_head(actor_hidden)
            if actions is not None:
                raw_action = actions[date].to(action_mean).detach()
            elif sample:
                epsilon = (
                    torch.randn_like(action_mean)
                    if noise is None else noise[date].to(action_mean)
                )
                raw_action = (action_mean + std * epsilon).detach()
            else:
                raw_action = action_mean.detach()
            standardized_error = (raw_action - action_mean) / std
            log_prob = -(
                0.5 * standardized_error.square() + std.log() + normal_constant
            ).sum()
            log_prob_rows.append(log_prob)
            entropy_sum = entropy_sum + (
                std.log() + normal_constant + 0.5
            ).sum()
            kl = (
                torch.log(reference_std / std)
                + (std.square() + (action_mean - reference_mean).square())
                / (2.0 * reference_std.square())
                - 0.5
            )
            kl_sum = kl_sum + kl.mean()
            current_swap_budget = 0
            if self.action_mode == "uniform2":
                alpha, delta = self._alpha_delta(raw_action)
                rank_bonus = (delta / mu_std).clamp(
                    -self.maximum_rank_bonus, self.maximum_rank_bonus,
                )
                stock_rank_bonus = rank_bonus * previous_selected.to(hidden.dtype)
                hold_bonus = rank_bonus
                hold_threshold = rank_bonus.new_zeros(())
                hold_temperature = rank_bonus.new_zeros(())
                action_value = torch.stack((alpha, delta))
            elif self.action_mode == "selective4":
                alpha, hold_bonus, hold_threshold, hold_temperature = (
                    self._selective_actions(raw_action)
                )
                quality = torch.sigmoid(
                    (base_rank[date] - hold_threshold) / hold_temperature
                )
                stock_rank_bonus = (
                    hold_bonus * quality * previous_selected.to(hidden.dtype)
                )
                delta = hold_bonus.new_zeros(())
                action_value = torch.stack((
                    alpha, hold_bonus, hold_threshold, hold_temperature,
                ))
            elif self.action_mode == "bucketed6":
                alpha, _ = self._alpha_delta(raw_action)
                bucket_bonus = self.maximum_rank_bonus * torch.tanh(raw_action[1:])
                bucket_index = torch.clamp(
                    (base_rank[date] * self.rank_bonus_buckets).long(),
                    min=0, max=self.rank_bonus_buckets - 1,
                )
                per_stock_bonus = bucket_bonus[bucket_index]
                stock_rank_bonus = (
                    per_stock_bonus * previous_selected.to(hidden.dtype)
                )
                delta = alpha.new_zeros(())
                selected_previous = previous_selected.bool()
                hold_bonus = (
                    per_stock_bonus[selected_previous].mean()
                    if bool(selected_previous.any()) else alpha.new_zeros(())
                )
                hold_threshold = alpha.new_zeros(())
                hold_temperature = alpha.new_zeros(())
                action_value = torch.cat((alpha.reshape(1), bucket_bonus))
            else:
                if self.action_mode in {"swap_budget_only", "swap_budget_raw"}:
                    alpha = raw_action.new_tensor(
                        1.0 if self.action_mode == "swap_budget_raw" else self.initial_alpha
                    )
                    budget_raw = raw_action[0]
                else:
                    alpha = self.alpha_minimum + (
                        self.alpha_maximum - self.alpha_minimum
                    ) * torch.sigmoid(raw_action[0])
                    budget_raw = raw_action[1] if self.action_mode == "swap_budget_alpha" else raw_action[2]
                delta = raw_action.new_zeros(())
                hold_bonus = raw_action.new_zeros(())
                hold_threshold = raw_action.new_zeros(())
                hold_temperature = raw_action.new_zeros(())
                stock_rank_bonus = torch.zeros_like(base_rank[date])
                if self.action_mode == "boundary4":
                    action_value = torch.cat((alpha.reshape(1),
                        self.maximum_rank_bonus * torch.tanh(raw_action[1:])))
                else:
                    current_swap_budget = min(
                        self.maximum_swap_budget,
                        max(0, int(torch.round(
                            self.initial_swap_budget * torch.exp(budget_raw.clamp(-8.0, 8.0))
                        ).item())),
                    )
                    if self.action_mode in {"hysteresis", "candidate_residual"}:
                        margin = self.maximum_hysteresis_margin * torch.sigmoid(raw_action[1])
                        global_values = torch.stack((
                            alpha, margin, raw_action.new_tensor(float(current_swap_budget)),
                        ))
                        action_value = (
                            torch.cat((global_values, torch.tanh(raw_action[3:])))
                            if self.action_mode == "candidate_residual" else global_values
                        )
                    elif self.action_mode in {"swap_budget_only", "swap_budget_raw"}:
                        action_value = raw_action.new_tensor([float(current_swap_budget)])
                    else:
                        action_value = torch.stack((
                            alpha, raw_action.new_tensor(float(current_swap_budget)),
                        ))
            current_signal = (
                base_score[date] if self.action_mode == "swap_budget_raw"
                else base_rank[date]
            )
            base_decision_score = (
                alpha * current_signal
                + (1.0 - alpha) * previous_score
                + stock_rank_bonus
            )
            candidate_bonus_abs_mean = base_decision_score.new_zeros(())
            if self.action_mode == "candidate_residual" and candidate_indices.numel():
                count = min(max(int(trade.sum().item() * self.top_fraction), 1), int(trade.sum().item()))
                cutoff = torch.topk(base_decision_score[trade], k=count).values.min()
                coefficients = torch.tanh(raw_action[3:])
                adjustment = self.maximum_candidate_bonus * torch.tanh(
                    (candidate_basis @ coefficients) / math.sqrt(candidate_basis.shape[-1])
                ) * torch.exp(-(
                    (base_decision_score[candidate_indices] - cutoff).abs()
                    / self.candidate_boundary_width
                ))
                bonus = torch.zeros_like(base_decision_score).scatter(
                    0, candidate_indices, adjustment,
                )
                base_decision_score = base_decision_score + bonus
                candidate_bonus_abs_mean = adjustment.abs().mean().detach()
            decision_score = base_decision_score
            if self.action_mode in {"swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "candidate_residual"}:
                count = min(max(int(trade.sum().item() * self.top_fraction), 1), int(trade.sum().item()))
                if self.action_mode in {"hysteresis", "candidate_residual"}:
                    desired = select_hysteresis(
                        base_decision_score, trade, previous_selected,
                        k=count, max_swaps=current_swap_budget, margin=float(margin.item()),
                    )
                else:
                    desired = select_swap_budget(
                        base_decision_score, trade, previous_selected,
                        k=count, swap_budget=current_swap_budget,
                    )
                decision_score = project_selection_to_scores(
                    base_decision_score, desired, trade,
                )
            elif self.action_mode == "boundary4" and bool(trade.any()):
                count = min(max(int(trade.sum().item() * self.top_fraction), 1), int(trade.sum().item()))
                cutoff = torch.topk(base_decision_score[trade], k=count).values.min()
                near_boundary = torch.exp(-(
                    (base_decision_score - cutoff).abs() / self.boundary_width
                ))
                momentum = base_rank[date] - base_rank[max(date - 1, 0)]
                held = previous_selected.to(base_decision_score.dtype)
                coefficients = action_value[1:]
                decision_score = base_decision_score + near_boundary * (
                    coefficients[0] * held
                    + coefficients[1] * (1.0 - held)
                    + coefficients[2] * momentum
                )
            # Keep the median-rank fallback in recurrent state for temporarily
            # ineligible stocks. The official EWMA evaluator ranks every code
            # after applying that same fallback; zeroing it here would alter
            # the history when a stock later becomes eligible again.
            selected = hard_top_fraction_mask(
                decision_score[None, :], trade[None, :], self.top_fraction,
            )[0]
            if self.action_mode in {"swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "candidate_residual"}:
                if not torch.equal(selected, desired):
                    raise AssertionError("projected Top-K does not match the requested portfolio")
            scores.append(decision_score)
            selected_rows.append(selected)
            raw_actions.append(raw_action)
            means.append(action_mean)
            action_values.append(action_value)
            alphas.append(alpha)
            deltas.append(delta)
            hold_bonuses.append(hold_bonus)
            hold_thresholds.append(hold_threshold)
            hold_temperatures.append(hold_temperature)
            swap_budgets.append(raw_action.new_tensor(float(current_swap_budget)))
            realized_swaps.append((selected & ~previous_selected).sum().to(raw_action.dtype))
            candidate_bonus_abs_means.append(candidate_bonus_abs_mean)
            candidate_counts.append(raw_action.new_tensor(
                float(candidate_indices.numel()) if candidate_timeline is not None else 0.0,
            ))
            previous_score = (
                base_decision_score if self.action_mode in {
                    "swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "candidate_residual", "boundary4",
                } else decision_score
            ).detach()
            previous_selected = selected.detach()
            holding_age = torch.where(
                previous_selected, holding_age + 1.0, torch.zeros_like(holding_age),
            ).detach()

        dates = float(hidden.shape[0])
        return DecisionPolicyOutput(
            decision_score=torch.stack(scores),
            selected=torch.stack(selected_rows),
            raw_action=torch.stack(raw_actions),
            alpha=torch.stack(alphas),
            delta=torch.stack(deltas),
            hold_bonus=torch.stack(hold_bonuses),
            hold_threshold=torch.stack(hold_thresholds),
            hold_temperature=torch.stack(hold_temperatures),
            action_mean=torch.stack(means),
            action_value=torch.stack(action_values),
            # Keep one joint action log probability per date. PPO ratios over
            # a full trajectory would otherwise exponentiate dozens of terms
            # and saturate the clip after the first optimizer step.
            log_prob=torch.stack(log_prob_rows),
            entropy=entropy_sum / dates,
            reference_kl=kl_sum / dates,
            policy_std=std,
            swap_budget=torch.stack(swap_budgets),
            realized_swaps=torch.stack(realized_swaps),
            candidate_bonus_abs_mean=torch.stack(candidate_bonus_abs_means),
            candidate_count=torch.stack(candidate_counts),
        )
