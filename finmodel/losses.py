from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class SoftScoreResult(NamedTuple):
    loss: torch.Tensor
    score: torch.Tensor
    rank_ic: torch.Tensor
    annual_excess_raw: torch.Tensor
    annual_excess_objective: torch.Tensor
    stability: torch.Tensor
    mse_anchor: torch.Tensor


def _valid(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    if mask is not None:
        finite &= mask.bool()
    return prediction[finite], target[finite]


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred, truth = _valid(prediction, target, mask)
    return ((pred - truth) ** 2).mean() if pred.numel() else prediction.sum() * 0.0


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, delta: float = 0.02) -> torch.Tensor:
    pred, truth = _valid(prediction, target, mask)
    return F.huber_loss(pred, truth, delta=delta) if pred.numel() else prediction.sum() * 0.0


def pearson_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred, truth = _valid(prediction, target, mask)
    if pred.numel() < 2:
        return prediction.sum() * 0.0
    pred = pred - pred.mean()
    truth = truth - truth.mean()
    denominator = torch.sqrt((pred.square().sum() * truth.square().sum()).clamp_min(1e-12))
    correlation = (pred * truth).sum() / denominator
    # Constant labels/predictions carry no correlation signal and remain finite.
    correlation = torch.where(torch.isfinite(correlation), correlation, torch.zeros_like(correlation))
    return 1.0 - correlation


def pairwise_logistic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    seed: int = 2026,
    permutations: int = 4,
    min_target_gap: float = 1e-4,
) -> torch.Tensor:
    pred, truth = _valid(prediction, target, mask)
    if pred.numel() < 2:
        return prediction.sum() * 0.0
    generator = torch.Generator(device=pred.device).manual_seed(seed)
    losses = []
    for _ in range(permutations):
        permutation = torch.randperm(pred.numel(), generator=generator, device=pred.device)
        target_gap = truth - truth[permutation]
        keep = target_gap.abs() >= min_target_gap
        if keep.any():
            signed_margin = (pred - pred[permutation]) * target_gap.sign()
            losses.append(F.softplus(-signed_margin[keep]).mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def stockmixer_official_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    rank_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream StockMixer MSE plus full cross-sectional hinge ranking loss.

    The upstream implementation forms ``(p_i-p_j) * (y_j-y_i)`` and applies
    ReLU.  Filtering first is equivalent to its outer-product mask while avoiding
    invalid labels and preventing masked NaNs from contaminating the loss.
    """
    pred, truth = _valid(prediction.reshape(-1), target.reshape(-1), None if mask is None else mask.reshape(-1))
    if pred.numel() == 0:
        zero = prediction.sum() * 0.0
        return zero, zero, zero
    mse = F.mse_loss(pred, truth)
    if pred.numel() < 2:
        rank = prediction.sum() * 0.0
    else:
        prediction_difference = pred[:, None] - pred[None, :]
        target_reverse_difference = truth[None, :] - truth[:, None]
        rank = F.relu(prediction_difference * target_reverse_difference).mean()
    return mse + float(rank_weight) * rank, mse, rank


def _soft_percentile_rank(
    prediction: torch.Tensor, mask: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable cross-sectional percentile ranks on the valid universe."""
    indices = torch.nonzero(
        mask.reshape(-1).bool() & torch.isfinite(prediction.reshape(-1)), as_tuple=False,
    ).reshape(-1)
    full = prediction.new_zeros(prediction.numel())
    if indices.numel() < 2:
        return full, indices
    values = prediction.reshape(-1)[indices]
    scale = values.std(unbiased=False).clamp_min(1e-6)
    standardized = (values - values.mean()) / scale
    pairwise = torch.sigmoid(
        (standardized[:, None] - standardized[None, :]) / float(temperature)
    )
    full = full.scatter(0, indices, pairwise.mean(dim=1))
    return full, indices


def _hard_percentile_rank(target: torch.Tensor) -> torch.Tensor:
    if target.numel() < 2:
        return target.new_zeros(target.shape)
    order = torch.argsort(target)
    ranks = torch.empty_like(target)
    ranks[order] = torch.arange(target.numel(), device=target.device, dtype=target.dtype)
    return ranks / float(target.numel() - 1)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() < 2:
        return left.sum() * 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt((left.square().sum() * right.square().sum()).clamp_min(1e-12))
    value = (left * right).sum() / denominator
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def _soft_day_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_mask: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    rank_temperature: float,
    top_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    label_mask = label_mask.reshape(-1).bool() & torch.isfinite(target)
    tradable_mask = tradable_mask.reshape(-1).bool()

    # The official evaluator uses two different universes: Rank IC includes
    # every labelled stock (including limit-up stocks), while Top 10% return
    # and turnover exclude limit-up stocks.  Keep the two ranks separate;
    # sharing the tradable rank here silently changed the IC objective.
    ic_ranks, labelled_indices = _soft_percentile_rank(
        prediction, label_mask, rank_temperature,
    )
    soft_ranks, tradable_indices = _soft_percentile_rank(
        prediction, tradable_mask, rank_temperature,
    )
    if labelled_indices.numel() < 2:
        zero = prediction.sum() * 0.0
        soft_ic = zero
    else:
        truth_rank = _hard_percentile_rank(target[labelled_indices]).detach()
        soft_ic = _correlation(ic_ranks[labelled_indices], truth_rank)

    return_mask = label_mask & tradable_mask
    return_indices = torch.nonzero(return_mask, as_tuple=False).reshape(-1)
    if return_indices.numel() == 0:
        soft_excess = prediction.sum() * 0.0
    else:
        weights = torch.sigmoid(
            (soft_ranks[return_indices] - 0.9) / float(top_temperature)
        )
        truth = target[return_indices]
        soft_top_return = (weights * truth).sum() / weights.sum().clamp_min(1e-6)
        soft_excess = 252.0 * (soft_top_return - truth.mean())

    portfolio = prediction.new_zeros(prediction.shape)
    if tradable_indices.numel():
        portfolio = portfolio.scatter(
            0,
            tradable_indices,
            torch.sigmoid(
                (soft_ranks[tradable_indices] - 0.9) / float(top_temperature)
            ),
        )
    return soft_ic, soft_excess, portfolio


def stockmixer_soft_score_loss(
    previous_prediction: torch.Tensor,
    current_prediction: torch.Tensor,
    previous_target: torch.Tensor,
    current_target: torch.Tensor,
    previous_label_mask: torch.Tensor,
    current_label_mask: torch.Tensor,
    previous_tradable_mask: torch.Tensor,
    current_tradable_mask: torch.Tensor,
    *,
    rank_temperature: float = 0.1,
    top_temperature: float = 0.02,
    excess_scale: float | None = None,
    mse_weight: float = 0.0,
) -> SoftScoreResult:
    """Stabilized differentiable proxy of the official three-component score.

    ``excess_scale`` is the symmetric bound of a tanh transform applied to
    noisy single-day annualized excess returns.  ``b*tanh(x/b)`` preserves a
    unit slope around zero while matching the competition's typical excess
    range when ``b=0.5``. ``mse_weight`` anchors the otherwise scale-invariant
    soft-rank prediction to raw return magnitudes.
    """
    if rank_temperature <= 0 or top_temperature <= 0 or mse_weight < 0:
        raise ValueError("soft-score temperatures must be positive and mse_weight non-negative")
    if excess_scale is not None and excess_scale <= 0:
        raise ValueError("excess_scale must be positive when enabled")
    previous_ic, previous_excess, previous_portfolio = _soft_day_components(
        previous_prediction, previous_target, previous_label_mask, previous_tradable_mask,
        rank_temperature=rank_temperature, top_temperature=top_temperature,
    )
    current_ic, current_excess, current_portfolio = _soft_day_components(
        current_prediction, current_target, current_label_mask, current_tradable_mask,
        rank_temperature=rank_temperature, top_temperature=top_temperature,
    )
    soft_ic = (previous_ic + current_ic) / 2
    soft_excess_raw = (previous_excess + current_excess) / 2
    soft_excess_objective = (
        float(excess_scale) * torch.tanh(soft_excess_raw / float(excess_scale))
        if excess_scale is not None else soft_excess_raw
    )
    intersection = (previous_portfolio * current_portfolio).sum()
    union = (
        previous_portfolio + current_portfolio - previous_portfolio * current_portfolio
    ).sum()
    soft_stability = intersection / union.clamp_min(1e-6)
    previous_mse = masked_mse(previous_prediction, previous_target, previous_label_mask)
    current_mse = masked_mse(current_prediction, current_target, current_label_mask)
    mse_anchor = (previous_mse + current_mse) / 2
    soft_score = 0.4 * soft_ic + 0.3 * soft_excess_objective + 0.3 * soft_stability
    loss = -soft_score + float(mse_weight) * mse_anchor
    return SoftScoreResult(
        loss, soft_score, soft_ic, soft_excess_raw,
        soft_excess_objective, soft_stability, mse_anchor,
    )


def return_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    kind: str,
    *,
    seed: int = 2026,
) -> torch.Tensor:
    if kind == "mse":
        return masked_mse(prediction, target, mask)
    if kind == "huber":
        return masked_huber(prediction, target, mask)
    if kind == "mse_ic":
        return masked_mse(prediction, target, mask) + 1e-3 * pearson_loss(prediction, target, mask)
    if kind == "mse_rank":
        return masked_mse(prediction, target, mask) + 1e-3 * pairwise_logistic_loss(prediction, target, mask, seed=seed)
    if kind == "mse_ic_rank":
        return (
            masked_mse(prediction, target, mask)
            + 5e-4 * pearson_loss(prediction, target, mask)
            + 5e-4 * pairwise_logistic_loss(prediction, target, mask, seed=seed)
        )
    raise ValueError(f"unknown loss: {kind}")
