from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    valid = torch.isfinite(target)
    if mask is not None:
        valid &= mask.reshape(-1).bool()
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target[valid])


def _soft_percentile_rank(
    prediction: torch.Tensor, mask: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable cross-sectional percentile rank on a masked universe."""
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
    return full.scatter(0, indices, pairwise.mean(dim=1)), indices


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
    """Return soft Rank IC, annualized Top-10% excess and portfolio weights."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    label_mask = label_mask.reshape(-1).bool() & torch.isfinite(target)
    tradable_mask = tradable_mask.reshape(-1).bool()

    # Official Rank IC includes every labelled stock. Top-10% return and
    # turnover exclude limit-up stocks, so the two ranks use separate masks.
    ic_ranks, labelled_indices = _soft_percentile_rank(
        prediction, label_mask, rank_temperature,
    )
    soft_ranks, tradable_indices = _soft_percentile_rank(
        prediction, tradable_mask, rank_temperature,
    )
    if labelled_indices.numel() < 2:
        soft_ic = prediction.sum() * 0.0
    else:
        truth_rank = _hard_percentile_rank(target[labelled_indices]).detach()
        soft_ic = _correlation(ic_ranks[labelled_indices], truth_rank)

    return_indices = torch.nonzero(
        label_mask & tradable_mask, as_tuple=False,
    ).reshape(-1)
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
