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
    # Optional pre-action state for a detached critic; no new actor parameters.
    actor_state: torch.Tensor | None = None
    pre_action_age: torch.Tensor | None = None


class DecisionStaticCache(NamedTuple):
    """Action-independent daily inputs shared by all rollouts of one block."""

    base_rank: torch.Tensor
    predicted_excess: torch.Tensor
    market_hidden: torch.Tensor
    naive_selected: torch.Tensor
    candidate_top: tuple[torch.Tensor, ...]
    mu_std: torch.Tensor
    cutoff: torch.Tensor
    top_mean: torch.Tensor


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
        alpha_parameterization: str = "sigmoid",
        alpha_linear_scale: float = 1.0,
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
        kl_mean_reference_std: float | None = None,
        minimum_action_std: float = 0.05,
        maximum_action_std: float = 0.50,
        return_scale: float = 0.02,
        top_fraction: float = 0.10,
        epsilon: float = 1e-6,
        initial_swap_budget: int = 10,
        maximum_swap_budget: int | None = 60,
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
        margin_mode: str = "score",
        decision_factor_dim: int = 0,
    ) -> None:
        super().__init__()
        if alpha_parameterization == "sigmoid":
            if not 0 < alpha_minimum < initial_alpha < alpha_maximum < 1:
                raise ValueError("sigmoid alpha bounds must contain initial_alpha inside (0, 1)")
        elif alpha_parameterization == "upper_clipped_linear":
            if not 0 < alpha_minimum < initial_alpha < alpha_maximum <= 1.0:
                raise ValueError("upper-clipped alpha requires interior initial_alpha and maximum <= 1")
            if alpha_linear_scale <= 0:
                raise ValueError("alpha_linear_scale must be positive")
        else:
            raise ValueError(f"unsupported alpha_parameterization: {alpha_parameterization!r}")
        if not 0 < minimum_action_std < initial_action_std < maximum_action_std:
            raise ValueError("action std bounds are invalid")
        if kl_mean_reference_std is not None and kl_mean_reference_std <= 0:
            raise ValueError("KL mean reference std must be positive")
        if maximum_delta <= 0 or maximum_rank_bonus <= 0 or return_scale <= 0:
            raise ValueError("delta, rank bonus and return scale must be positive")
        if action_mode not in {
            "uniform2", "selective4", "bucketed6", "swap_budget_only",
            "swap_budget_raw",
            "swap_budget_alpha", "hysteresis", "hysteresis_no_alpha",
            "hysteresis_margin_only",
            "boundary4", "candidate_residual",
        }:
            raise ValueError(f"unsupported action_mode: {action_mode!r}")
        if initial_swap_budget <= 0 or (
            maximum_swap_budget is not None and initial_swap_budget >= maximum_swap_budget
        ):
            raise ValueError("swap budget bounds are invalid")
        if not 0 < initial_hysteresis_margin < maximum_hysteresis_margin:
            raise ValueError("hysteresis margin bounds are invalid")
        if boundary_width <= 0:
            raise ValueError("boundary width must be positive")
        if history_days < 0 or (history_days == 1) or history_hidden_dim <= 0 or history_context_dim <= 0:
            raise ValueError("history_days must be 0 or >= 2, with positive context dimensions")
        if history_boundary_width <= 0:
            raise ValueError("history boundary width must be positive")
        if observation_mode not in {"mean_pool", "candidate_attention", "portfolio_detail"}:
            raise ValueError("unsupported observation mode")
        if observation_mode == "candidate_attention" and history_days < 2:
            raise ValueError("candidate attention needs at least two history days")
        if action_mode == "candidate_residual" and observation_mode != "candidate_attention":
            raise ValueError("candidate residual actions need candidate attention")
        if return_feature_mode not in {"proxy", "explicit"}:
            raise ValueError("return feature mode must be proxy or explicit")
        if margin_mode not in {"score", "predicted_return"}:
            raise ValueError("margin mode must be score or predicted_return")
        if margin_mode == "predicted_return" and return_feature_mode != "explicit":
            raise ValueError("return-based margin requires explicit return forecasts")
        if observation_mode in {"candidate_attention", "portfolio_detail"} and not top_fraction < candidate_top_fraction <= 1.0:
            raise ValueError("candidate top fraction must exceed portfolio fraction")
        if candidate_max_count < 1 or candidate_context_dim < 1 or candidate_queries < 1:
            raise ValueError("candidate capacity must be positive")
        if maximum_candidate_bonus <= 0 or candidate_boundary_width <= 0:
            raise ValueError("candidate bonus and boundary width must be positive")
        if decision_factor_dim < 0:
            raise ValueError("decision_factor_dim must be non-negative")
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
        self.alpha_parameterization = str(alpha_parameterization)
        self.alpha_linear_scale = float(alpha_linear_scale)
        self.maximum_delta = float(maximum_delta)
        self.maximum_rank_bonus = float(maximum_rank_bonus)
        self.hold_threshold_minimum = float(hold_threshold_minimum)
        self.hold_threshold_maximum = float(hold_threshold_maximum)
        self.hold_temperature_minimum = float(hold_temperature_minimum)
        self.hold_temperature_maximum = float(hold_temperature_maximum)
        self.rank_bonus_buckets = int(rank_bonus_buckets)
        self.minimum_action_std = float(minimum_action_std)
        self.maximum_action_std = float(maximum_action_std)
        self.kl_mean_reference_std = float(
            initial_action_std if kl_mean_reference_std is None
            else kl_mean_reference_std
        )
        self.return_scale = float(return_scale)
        self.top_fraction = float(top_fraction)
        self.epsilon = float(epsilon)
        self.initial_alpha = float(initial_alpha)
        self.initial_swap_budget = int(initial_swap_budget)
        self.maximum_swap_budget = (
            None if maximum_swap_budget is None else int(maximum_swap_budget)
        )
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
        self.margin_mode = str(margin_mode)
        self.decision_factor_dim = int(decision_factor_dim)

        self.market_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, market_dim),
            nn.Tanh(),
        )
        if self.decision_factor_dim:
            # Keep shared-module initialization identical to a factor-free
            # policy under the same seed, so factor ablations start together.
            with torch.random.fork_rng(devices=[]):
                self.decision_factor_encoder = nn.Sequential(
                    nn.LayerNorm(self.decision_factor_dim),
                    nn.Linear(self.decision_factor_dim, market_dim),
                    nn.Tanh(),
                )
                self.decision_factor_fusion = nn.Linear(3 * market_dim, market_dim)
            # Adding factors leaves the initial, untrained action policy unchanged.
            nn.init.zeros_(self.decision_factor_fusion.weight)
            nn.init.zeros_(self.decision_factor_fusion.bias)
        # Six interpretable market/portfolio scalars accompany pooled predictor state.
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
        elif self.observation_mode == "portfolio_detail":
            # Read current-day, per-stock predictor state for held names and the high
            # ranked candidate set. The zero-initialized fusion preserves the
            # legacy action policy at initialization under an identical seed.
            with torch.random.fork_rng(devices=[]):
                self.detail_token_encoder = nn.Sequential(
                    nn.LayerNorm(d_model + 5),
                    nn.Linear(d_model + 5, candidate_context_dim),
                    nn.Tanh(),
                )
                self.detail_keys = nn.Linear(candidate_context_dim, candidate_context_dim)
                self.detail_values = nn.Linear(candidate_context_dim, candidate_context_dim)
                self.detail_queries = nn.Parameter(torch.randn(
                    candidate_queries, candidate_context_dim,
                ) / math.sqrt(candidate_context_dim))
                self.detail_pool = nn.Sequential(
                    nn.Linear(candidate_queries * candidate_context_dim, market_dim),
                    nn.Tanh(),
                )
                self.detail_fusion = nn.Linear(market_dim, market_dim)
            nn.init.zeros_(self.detail_fusion.weight)
            nn.init.zeros_(self.detail_fusion.bias)
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
            "hysteresis_no_alpha": 2,
            "hysteresis_margin_only": 1,
            "candidate_residual": 9,
            "boundary4": 4,
        }[self.action_mode]
        self.action_count = int(action_count)
        self.action_head = nn.Linear(recurrent_dim, self.action_count)

        reference_action_mean = torch.zeros(self.action_count, dtype=torch.float32)
        if (
            self.action_mode not in {
                "swap_budget_only", "swap_budget_raw", "hysteresis_no_alpha",
                "hysteresis_margin_only",
            }
            and self.alpha_parameterization == "sigmoid"
        ):
            alpha_probability = (
                (float(initial_alpha) - self.alpha_minimum)
                / (self.alpha_maximum - self.alpha_minimum)
            )
            reference_action_mean[0] = math.log(
                alpha_probability / (1.0 - alpha_probability)
            )
        if self.action_mode in {
            "hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only",
            "candidate_residual",
        }:
            margin_probability = (
                self.initial_hysteresis_margin / self.maximum_hysteresis_margin
            )
            margin_index = 0 if self.action_mode in {
                "hysteresis_no_alpha", "hysteresis_margin_only",
            } else 1
            reference_action_mean[margin_index] = math.log(
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

    def _alpha_from_raw(self, raw_alpha: torch.Tensor) -> torch.Tensor:
        if self.alpha_parameterization == "upper_clipped_linear":
            return (
                self.initial_alpha + self.alpha_linear_scale * raw_alpha
            ).clamp(self.alpha_minimum, self.alpha_maximum)
        return self.alpha_minimum + (
            self.alpha_maximum - self.alpha_minimum
        ) * torch.sigmoid(raw_alpha)

    def _alpha_delta(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._alpha_from_raw(raw_action[..., 0])
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

    def _portfolio_detail_context(
        self,
        date: int,
        hidden: torch.Tensor,
        base_rank: torch.Tensor,
        eligible: torch.Tensor,
        tradable: torch.Tensor,
        previous_selected: torch.Tensor,
        holding_age: torch.Tensor,
        predicted_excess: torch.Tensor,
        candidate_top: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal attention over held names and current high-ranked candidates."""
        if candidate_top is None:
            available = torch.nonzero(eligible[date] & tradable[date], as_tuple=False).flatten()
            if available.numel():
                count = min(
                    self.candidate_max_count,
                    max(1, int(math.ceil(float(available.numel()) * self.candidate_top_fraction))),
                )
                top = available[torch.topk(base_rank[date, available], k=count).indices]
            else:
                top = available
        else:
            top = candidate_top
        indices = self._portfolio_detail_indices(
            top, previous_selected, tradable[date], base_rank[date],
        )
        if not indices.numel():
            return hidden.new_zeros(self.market_encoder[1].out_features), indices
        cutoff_rank = 1.0 - self.top_fraction
        metadata = torch.stack((
            previous_selected[indices].to(hidden.dtype),
            (holding_age[indices] / 20.0).clamp(0.0, 1.0),
            base_rank[date, indices],
            ((base_rank[date, indices] - cutoff_rank)
             / self.candidate_boundary_width).clamp(-2.0, 2.0),
            (predicted_excess[date, indices] / self.return_scale).clamp(-5.0, 5.0),
        ), dim=-1)
        tokens = self._encode_detail_tokens(torch.cat((hidden[date, indices], metadata), dim=-1))
        return self._pool_detail_tokens(tokens, previous_selected[indices],
                                        recurrent_state=recurrent_state), indices

    def _select_hysteresis(self, *args, **kwargs):
        # Default delegation preserves published execution exactly.
        return select_hysteresis(*args, **kwargs)

    def _encode_detail_tokens(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        return self.detail_token_encoder(raw_tokens)

    def _pool_detail_tokens(self, tokens, held, valid=None, recurrent_state=None):
        """Overridable pooling; default keeps the published query attention."""
        keys, values = self.detail_keys(tokens), self.detail_values(tokens)
        logits = (self.detail_queries @ keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1])
        if valid is not None:
            logits = logits.masked_fill(~valid[..., None, :], -torch.inf)
        pooled = torch.softmax(logits, dim=-1) @ values
        return self.detail_pool(pooled.reshape(*tokens.shape[:-2], -1))

    def _fuse_detail_input(self, market, detail, scalars):
        return torch.cat((market + self.detail_fusion(detail), scalars), dim=-1)

    def _portfolio_detail_indices(
        self, top: torch.Tensor, previous_selected: torch.Tensor,
        tradable: torch.Tensor, base_rank: torch.Tensor,
    ) -> torch.Tensor:
        """Exact held-first candidate union, shared by serial and batched paths."""
        held = torch.nonzero(previous_selected & tradable, as_tuple=False).flatten()
        if held.numel() >= self.candidate_max_count:
            indices = held[torch.topk(
                base_rank[held], k=self.candidate_max_count,
            ).indices]
        else:
            held_mask = torch.zeros_like(previous_selected)
            held_mask[held] = True
            fresh = top[~held_mask[top]]
            indices = torch.cat((held, fresh[:self.candidate_max_count - held.numel()]))
        return indices

    def _portfolio_detail_context_batch(
        self, date: int, hidden: torch.Tensor, cache: DecisionStaticCache,
        previous_selected: torch.Tensor, holding_age: torch.Tensor,
        tradable: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run candidate MLP and query attention across sampled trajectories."""
        rollout_count = previous_selected.shape[0]
        rows = [self._portfolio_detail_indices(
            cache.candidate_top[date], previous_selected[row], tradable[date],
            cache.base_rank[date],
        ) for row in range(rollout_count)]
        lengths = torch.tensor([len(row) for row in rows], device=hidden.device)
        width = max(max(len(row) for row in rows), 1)
        indices = torch.zeros((rollout_count, width), dtype=torch.long, device=hidden.device)
        valid = torch.zeros((rollout_count, width), dtype=torch.bool, device=hidden.device)
        for row, stock_indices in enumerate(rows):
            if stock_indices.numel():
                indices[row, :len(stock_indices)] = stock_indices
                valid[row, :len(stock_indices)] = True
            else:
                valid[row, 0] = True  # A dummy token prevents an all-masked softmax.
        rank = cache.base_rank[date, indices]
        metadata = torch.stack((
            previous_selected.gather(1, indices).to(hidden.dtype),
            (holding_age.gather(1, indices) / 20.0).clamp(0.0, 1.0),
            rank,
            ((rank - (1.0 - self.top_fraction)) / self.candidate_boundary_width).clamp(-2.0, 2.0),
            (cache.predicted_excess[date, indices] / self.return_scale).clamp(-5.0, 5.0),
        ), dim=-1)
        stock_hidden = hidden[date, indices]
        tokens = self._encode_detail_tokens(torch.cat((stock_hidden, metadata), dim=-1))
        context = self._pool_detail_tokens(tokens, previous_selected.gather(1, indices), valid,
                                           recurrent_state=recurrent_state)
        context = torch.where(lengths[:, None] > 0, context, torch.zeros_like(context))
        return context, lengths

    @torch.no_grad()
    def prepare_static(
        self, hidden: torch.Tensor, base_score: torch.Tensor,
        eligible: torch.Tensor, tradable: torch.Tensor,
        *, predicted_return: torch.Tensor,
    ) -> DecisionStaticCache:
        """Cache only quantities that are independent of actions and weights."""
        if (self.observation_mode != "portfolio_detail"
                or self.action_mode != "hysteresis_no_alpha"
                or self.return_feature_mode != "explicit"
                or self.decision_factor_dim):
            raise ValueError("static cache currently supports the portfolio-detail two-action policy")
        if (hidden.ndim != 3 or base_score.shape != hidden.shape[:2]
                or eligible.shape != base_score.shape or tradable.shape != base_score.shape
                or predicted_return.shape != base_score.shape):
            raise ValueError("static cache inputs have inconsistent dimensions")
        if hidden.requires_grad or base_score.requires_grad or predicted_return.requires_grad:
            raise ValueError("static cache requires a frozen predictor")
        base_rank = percentile_rank_scores(base_score, eligible)
        mask = eligible.bool()
        weights = mask.to(hidden.dtype)
        market_hidden = (hidden * weights[..., None]).sum(dim=1) / (
            weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        return_values = predicted_return.to(base_score)
        market_prediction = (return_values * mask.to(return_values.dtype)).sum(
            dim=1, keepdim=True,
        ) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        predicted_excess = return_values - market_prediction
        naive_selected = hard_top_fraction_mask(base_rank, tradable, self.top_fraction)
        tops, mu_stds, cutoffs, top_means = [], [], [], []
        for date in range(base_score.shape[0]):
            available = torch.nonzero(mask[date] & tradable[date].bool(), as_tuple=False).flatten()
            if available.numel():
                candidate_count = min(
                    self.candidate_max_count,
                    max(1, int(math.ceil(float(available.numel()) * self.candidate_top_fraction))),
                )
                top = available[torch.topk(base_rank[date, available], k=candidate_count).indices]
            else:
                top = available
            tops.append(top)
            valid_mu = predicted_excess[date, mask[date]]
            mu_stds.append(valid_mu.std(unbiased=False).clamp_min(self.epsilon))
            trade_indices = torch.nonzero(tradable[date].bool(), as_tuple=False).flatten()
            if trade_indices.numel():
                count = max(int(trade_indices.numel() * self.top_fraction), 1)
                top_positions = torch.topk(
                    predicted_excess[date, trade_indices],
                    k=min(count, trade_indices.numel()),
                ).indices
                top_indices = trade_indices[top_positions]
                cutoffs.append(predicted_excess[date, top_indices].min())
                top_means.append(predicted_excess[date, top_indices].mean())
            else:
                cutoffs.append(base_score.new_zeros(()))
                top_means.append(base_score.new_zeros(()))
        return DecisionStaticCache(
            base_rank, predicted_excess, market_hidden, naive_selected,
            tuple(tops), torch.stack(mu_stds), torch.stack(cutoffs), torch.stack(top_means),
        )

    def _forward_group(
        self, hidden: torch.Tensor, base_score: torch.Tensor,
        eligible: torch.Tensor, tradable: torch.Tensor,
        *, rollouts: int, static_cache: DecisionStaticCache,
        predicted_return: torch.Tensor, sample: bool,
        actions: torch.Tensor | None, noise: torch.Tensor | None,
        capture_state: bool = False,
        anchor_selected: torch.Tensor | None = None,
        execution_score: torch.Tensor | None = None,
    ) -> DecisionPolicyOutput:
        """Parallelize the neural policy over rollouts, keeping causal dates serial.

        Exact hard Top-K execution remains per trajectory because its candidate
        counts and held stocks are data-dependent. The expensive candidate MLP,
        attention, and recurrent actor are batched over the rollout axis.
        """
        if (self.action_mode != "hysteresis_no_alpha"
                or self.observation_mode != "portfolio_detail"
                or self.return_feature_mode != "explicit"
                or self.decision_factor_dim):
            raise ValueError("batched rollouts support the portfolio-detail two-action route only")
        dates, stocks = base_score.shape
        if anchor_selected is not None and anchor_selected.shape != base_score.shape:
            raise ValueError("anchor_selected must be [dates, stocks]")
        if (hidden.shape[:2] != (dates, stocks)
                or eligible.shape != base_score.shape or tradable.shape != base_score.shape
                or predicted_return.shape != base_score.shape
                or static_cache.base_rank.shape != base_score.shape
                or len(static_cache.candidate_top) != dates):
            raise ValueError("batched rollout inputs do not match the static cache")
        expected = (rollouts, dates, self.action_count)
        if actions is not None and actions.shape != expected:
            raise ValueError(f"batched actions must be {expected}")
        if noise is not None and noise.shape != expected:
            raise ValueError(f"batched noise must be {expected}")
        if actions is not None and (sample or noise is not None):
            raise ValueError("fixed actions cannot be combined with sampling/noise")
        trade = tradable.to(device=hidden.device, dtype=torch.bool)
        base_rank = static_cache.base_rank
        # Observation features/cache keep the original two heads. Only the
        # environment's ordering and portfolio projection use the optional mix.
        execution_rank = (base_rank if execution_score is None else
            percentile_rank_scores(execution_score.to(base_score), eligible))
        predicted_excess = static_cache.predicted_excess
        market_encoded = self.market_encoder(static_cache.market_hidden)
        recurrent = hidden.new_zeros((rollouts, self.recurrent.hidden_size))
        initial_selected = (static_cache.naive_selected[0] if execution_score is None else
            hard_top_fraction_mask(execution_rank[:1], trade[:1], self.top_fraction)[0])
        previous_selected = initial_selected.expand(rollouts, -1).clone()
        holding_age = hidden.new_zeros((rollouts, stocks))
        std = self.policy_std.to(hidden)
        reference_mean = self.reference_action_mean.to(hidden)
        reference_std = self.reference_std.to(hidden)
        normal_constant = 0.5 * math.log(2.0 * math.pi)
        entropy_sum = hidden.new_zeros((rollouts,))
        kl_sum = hidden.new_zeros((rollouts,))
        output_rows = {name: [] for name in (
            "decision_score", "selected", "raw_action", "alpha", "delta",
            "hold_bonus", "hold_threshold", "hold_temperature", "action_mean",
            "action_value", "log_prob", "swap_budget", "realized_swaps",
            "candidate_bonus_abs_mean", "candidate_count",
        )}
        actor_states, pre_action_ages = [], []

        for date in range(dates):
            if anchor_selected is not None and date > 0:
                previous_selected = anchor_selected[date - 1].bool().expand(rollouts, -1)
                holding_age = torch.where(
                    previous_selected, holding_age + 1.0, torch.zeros_like(holding_age),
                ).detach()
            today_trade = trade[date]
            prev_count = previous_selected.sum(dim=1).clamp_min(1)
            previous_mean = (
                predicted_excess[date][None, :] * previous_selected
            ).sum(dim=1) / prev_count
            naive = static_cache.naive_selected[date]
            union = (previous_selected | naive).sum(dim=1).clamp_min(1)
            jaccard = (previous_selected & naive).sum(dim=1).to(hidden.dtype) / union
            scalars = torch.stack((
                static_cache.mu_std[date].expand(rollouts) / self.return_scale,
                static_cache.cutoff[date].expand(rollouts) / self.return_scale,
                static_cache.top_mean[date].expand(rollouts) / self.return_scale,
                previous_mean / self.return_scale,
                1.0 - jaccard,
                previous_selected.to(hidden.dtype).mean(dim=1),
            ), dim=-1)
            detail, candidate_lengths = self._portfolio_detail_context_batch(
                date, hidden, static_cache, previous_selected, holding_age, trade,
                recurrent_state=recurrent,
            )
            actor_input = self._fuse_detail_input(
                market_encoded[date][None, :].expand(rollouts, -1), detail, scalars,
            )
            recurrent = self.recurrent(actor_input, recurrent)
            if capture_state:
                actor_states.append(recurrent.detach())
                pre_action_ages.append(holding_age.detach())
            action_mean = self.action_head(recurrent)
            if actions is not None:
                raw_action = actions[:, date].to(action_mean).detach()
            elif sample:
                epsilon = torch.randn_like(action_mean) if noise is None else noise[:, date].to(action_mean)
                raw_action = (action_mean + std * epsilon).detach()
            else:
                raw_action = action_mean.detach()
            standardized_error = (raw_action - action_mean) / std
            log_prob = -(
                0.5 * standardized_error.square() + std.log() + normal_constant
            ).sum(dim=-1)
            entropy_sum = entropy_sum + (std.log() + normal_constant + 0.5).sum()
            kl = (
                torch.log(reference_std / std)
                + std.square() / (2.0 * reference_std.square())
                + (action_mean - reference_mean).square()
                / (2.0 * self.kl_mean_reference_std ** 2)
                - 0.5
            )
            kl_sum = kl_sum + kl.mean(dim=-1)
            margin = self.maximum_hysteresis_margin * torch.sigmoid(raw_action[:, 0])
            trade_count = int(today_trade.sum().item())
            k = min(max(int(trade_count * self.top_fraction), 1), trade_count)
            scores, selections, budgets = [], [], []
            for row in range(rollouts):
                budget = max(0, int(torch.round(
                    self.initial_swap_budget * torch.exp(raw_action[row, 1].clamp(-8.0, 8.0))
                ).item()))
                budget = min(
                    budget, k if self.maximum_swap_budget is None else self.maximum_swap_budget,
                )
                desired = self._select_hysteresis(
                    execution_rank[date], today_trade, previous_selected[row],
                    k=k, max_swaps=budget, margin=float(margin[row].item()),
                    predicted_return=(predicted_return[date]
                                      if self.margin_mode == "predicted_return" else None),
                )
                score = project_selection_to_scores(execution_rank[date], desired, today_trade)
                selected = hard_top_fraction_mask(
                    score[None, :], today_trade[None, :], self.top_fraction,
                )[0]
                if not torch.equal(selected, desired):
                    raise AssertionError("batched projected Top-K differs from requested portfolio")
                scores.append(score)
                selections.append(selected)
                budgets.append(budget)
            decision_score = torch.stack(scores)
            selected = torch.stack(selections)
            budget_values = raw_action.new_tensor(budgets)
            realized_swaps = (selected & ~previous_selected).sum(dim=1).to(raw_action.dtype)
            zeros = torch.zeros_like(margin)
            day = {
                "decision_score": decision_score,
                "selected": selected,
                "raw_action": raw_action,
                "alpha": torch.ones_like(margin),
                "delta": zeros,
                "hold_bonus": zeros,
                "hold_threshold": zeros,
                "hold_temperature": zeros,
                "action_mean": action_mean,
                "action_value": torch.stack((margin, budget_values), dim=-1),
                "log_prob": log_prob,
                "swap_budget": budget_values,
                "realized_swaps": realized_swaps,
                "candidate_bonus_abs_mean": zeros,
                "candidate_count": candidate_lengths.to(raw_action.dtype),
            }
            for name, value in day.items():
                output_rows[name].append(value)
            previous_selected = selected.detach()
            if anchor_selected is None:
                holding_age = torch.where(
                    previous_selected, holding_age + 1.0, torch.zeros_like(holding_age),
                ).detach()

        stacked = {name: torch.stack(rows, dim=1) for name, rows in output_rows.items()}
        return DecisionPolicyOutput(
            **stacked,
            entropy=entropy_sum / float(dates),
            reference_kl=kl_sum / float(dates),
            policy_std=std,
            actor_state=torch.stack(actor_states, dim=1) if capture_state else None,
            pre_action_age=torch.stack(pre_action_ages, dim=1) if capture_state else None,
        )

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
        decision_factors: torch.Tensor | None = None,
        static_cache: DecisionStaticCache | None = None,
        rollouts: int = 1,
        capture_state: bool = False,
        anchor_selected: torch.Tensor | None = None,
        execution_score: torch.Tensor | None = None,
    ) -> DecisionPolicyOutput:
        """Observe original heads; optionally execute against a separate ranking.

        Actual holdings/ages still feed back after execution. With a mix, future
        actions can change because the portfolio changes, not because we replaced
        the observed ranking/return forecasts. Default behavior is unchanged.
        """
        if rollouts < 1:
            raise ValueError("rollouts must be positive")
        if execution_score is not None:
            if (self.action_mode != "hysteresis_no_alpha" or self.margin_mode != "predicted_return"
                    or self.return_feature_mode != "explicit"):
                raise ValueError("separate execution ranking requires the explicit-return two-action route")
            if execution_score.shape != base_score.shape or not bool(torch.isfinite(execution_score).all()):
                raise ValueError("execution_score must match base_score and be finite")
        if rollouts > 1:
            if static_cache is None or predicted_return is None or decision_factors is not None:
                raise ValueError("batched rollouts require static cache and explicit returns")
            return self._forward_group(
                hidden, base_score, eligible, tradable,
                rollouts=rollouts, static_cache=static_cache,
                predicted_return=predicted_return, sample=sample,
                actions=actions, noise=noise,
                capture_state=capture_state,
                anchor_selected=anchor_selected,
                execution_score=execution_score,
            )
        if anchor_selected is not None:
            raise ValueError("same-state branches require batched rollouts > 1")
        if hidden.ndim != 3:
            raise ValueError("hidden must be [dates, stocks, d_model]")
        if base_score.shape != hidden.shape[:2]:
            raise ValueError("base_score must match hidden dates/stocks")
        if eligible.shape != base_score.shape or tradable.shape != base_score.shape:
            raise ValueError("eligible/tradable must match base_score")
        if self.decision_factor_dim:
            if decision_factors is None or decision_factors.shape != (*base_score.shape, self.decision_factor_dim):
                raise ValueError("decision_factors must match [dates, stocks, decision_factor_dim]")
        elif decision_factors is not None:
            raise ValueError("decision_factors require a nonzero decision_factor_dim")
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
        if predicted_return is not None:
            predicted_return = predicted_return.to(base_score)
        base_rank = (
            percentile_rank_scores(base_score, eligible)
            if static_cache is None else static_cache.base_rank
        )
        execution_rank = (base_rank if execution_score is None else
            percentile_rank_scores(execution_score.to(base_score), eligible))
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
        if static_cache is not None:
            predicted_excess = static_cache.predicted_excess
        elif predicted_return is None:
            predicted_excess = base_score * self.return_scale
        else:
            absolute = predicted_return
            valid_weight = eligible.to(absolute.dtype)
            market_prediction = (absolute * valid_weight).sum(dim=1, keepdim=True) / (
                valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            predicted_excess = absolute - market_prediction
        weights = eligible.to(hidden.dtype)
        counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        market_hidden = (
            (hidden * weights[..., None]).sum(dim=1) / counts
            if static_cache is None else static_cache.market_hidden
        )
        market_encoded = self.market_encoder(market_hidden)
        encoded_factors = (
            self.decision_factor_encoder(decision_factors.to(hidden))
            if self.decision_factor_dim else None
        )

        actor_hidden = hidden.new_zeros(self.recurrent.hidden_size)
        previous_score = execution_rank[0].detach()
        previous_selected = hard_top_fraction_mask(
            execution_rank[0:1], tradable[0:1], self.top_fraction,
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
        actor_states, pre_action_ages = [], []
        entropy_sum = hidden.new_zeros(())
        kl_sum = hidden.new_zeros(())

        for date in range(hidden.shape[0]):
            valid = eligible[date]
            trade = tradable[date]
            trade_indices = torch.nonzero(trade, as_tuple=False).reshape(-1)
            if static_cache is not None:
                mu_std = static_cache.mu_std[date]
                cutoff = static_cache.cutoff[date]
                top_mean = static_cache.top_mean[date]
            else:
                valid_mu = predicted_excess[date, valid]
                mu_std = valid_mu.std(unbiased=False).clamp_min(self.epsilon)
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
            naive_selected = (
                hard_top_fraction_mask(
                    base_rank[date:date + 1], tradable[date:date + 1], self.top_fraction,
                )[0] if static_cache is None else static_cache.naive_selected[date]
            )
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
            day_market = market_encoded[date]
            if encoded_factors is not None:
                def factor_pool(mask: torch.Tensor) -> torch.Tensor:
                    weight = mask.to(encoded_factors.dtype)
                    return (encoded_factors[date] * weight[:, None]).sum(dim=0) / (
                        weight.sum().clamp_min(1.0)
                    )

                trade_valid = valid & trade
                candidates = torch.nonzero(trade_valid, as_tuple=False).flatten()
                top_mask = torch.zeros_like(trade_valid)
                if candidates.numel():
                    count_top = max(1, int(math.ceil(candidates.numel() * self.candidate_top_fraction)))
                    chosen = candidates[torch.topk(base_rank[date, candidates], k=count_top).indices]
                    top_mask[chosen] = True
                factor_context = torch.cat((
                    factor_pool(valid),
                    factor_pool(previous_selected & valid),
                    factor_pool(top_mask),
                ))
                day_market = day_market + self.decision_factor_fusion(factor_context)
            candidate_indices = torch.empty(0, dtype=torch.long, device=hidden.device)
            if candidate_timeline is not None:
                portfolio_context, candidate_indices, candidate_basis = self._candidate_context(
                    date, candidate_timeline, hidden, base_rank, eligible,
                    tradable, previous_selected, holding_age, predicted_excess,
                )
                actor_input = torch.cat((
                    day_market, portfolio_context, scalars,
                ))
            elif self.observation_mode == "portfolio_detail":
                portfolio_context, candidate_indices = self._portfolio_detail_context(
                    date, hidden, base_rank, eligible, tradable,
                    previous_selected, holding_age, predicted_excess,
                    candidate_top=(static_cache.candidate_top[date]
                                   if static_cache is not None else None),
                    recurrent_state=actor_hidden,
                )
                actor_input = self._fuse_detail_input(day_market, portfolio_context, scalars)
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
                    day_market, portfolio_context, scalars,
                ))
            else:
                actor_input = torch.cat((day_market, scalars))
            actor_hidden = self.recurrent(
                actor_input, actor_hidden,
            )
            if capture_state:
                actor_states.append(actor_hidden.detach())
                pre_action_ages.append(holding_age.detach())
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
                + std.square() / (2.0 * reference_std.square())
                + (action_mean - reference_mean).square()
                / (2.0 * self.kl_mean_reference_std ** 2)
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
                elif self.action_mode == "hysteresis_no_alpha":
                    alpha = raw_action.new_tensor(1.0)
                    budget_raw = raw_action[1]
                elif self.action_mode == "hysteresis_margin_only":
                    alpha = raw_action.new_tensor(1.0)
                    budget_raw = None
                else:
                    alpha = self._alpha_from_raw(raw_action[0])
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
                    if self.action_mode == "hysteresis_margin_only":
                        available = int(trade.sum().item())
                        current_swap_budget = min(
                            max(int(available * self.top_fraction), 1), available,
                        )
                    else:
                        current_swap_budget = max(0, int(torch.round(
                            self.initial_swap_budget * torch.exp(budget_raw.clamp(-8.0, 8.0))
                        ).item()))
                        if self.maximum_swap_budget is None:
                            available = int(trade.sum().item())
                            physical_limit = min(
                                max(int(available * self.top_fraction), 1), available,
                            )
                            current_swap_budget = min(current_swap_budget, physical_limit)
                        else:
                            current_swap_budget = min(
                                current_swap_budget, self.maximum_swap_budget,
                            )
                    if self.action_mode in {"hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"}:
                        margin_index = 0 if self.action_mode in {
                            "hysteresis_no_alpha", "hysteresis_margin_only",
                        } else 1
                        margin = self.maximum_hysteresis_margin * torch.sigmoid(raw_action[margin_index])
                        global_values = (
                            margin.reshape(1) if self.action_mode == "hysteresis_margin_only" else
                            torch.stack((margin, raw_action.new_tensor(float(current_swap_budget))))
                            if self.action_mode == "hysteresis_no_alpha" else
                            torch.stack((alpha, margin, raw_action.new_tensor(float(current_swap_budget))))
                        )
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
                else execution_rank[date]
            )
            if self.action_mode in {"hysteresis_no_alpha", "hysteresis_margin_only"}:
                # Current policies never smooth the predictor score.  Keep the
                # historical alpha modes above only for loading old checkpoints.
                base_decision_score = current_signal + stock_rank_bonus
            else:
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
            if self.action_mode in {"swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"}:
                count = min(max(int(trade.sum().item() * self.top_fraction), 1), int(trade.sum().item()))
                if self.action_mode in {"hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"}:
                    desired = self._select_hysteresis(
                        base_decision_score, trade, previous_selected,
                        k=count, max_swaps=current_swap_budget, margin=float(margin.item()),
                        predicted_return=(predicted_return[date]
                                          if self.margin_mode == "predicted_return" else None),
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
            if self.action_mode in {"swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"}:
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
            candidate_counts.append(raw_action.new_tensor(float(candidate_indices.numel())))
            previous_score = (
                base_decision_score if self.action_mode in {
                    "swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual", "boundary4",
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
            actor_state=torch.stack(actor_states) if capture_state else None,
            pre_action_age=torch.stack(pre_action_ages) if capture_state else None,
        )
