"""Hard, causal Top-K portfolio actions and score-vector projection."""

from __future__ import annotations

import torch


def project_selection_to_scores(
    scores: torch.Tensor,
    selected: torch.Tensor,
    tradable: torch.Tensor,
    *,
    gap: float = 1e-4,
) -> torch.Tensor:
    """Make a chosen portfolio the exact Top-K with minimal boundary changes.

    Values far from the original K-th score stay unchanged. The small offsets
    preserve the ordering of adjusted stocks within each side of the boundary.
    """
    trade = tradable.bool()
    chosen = selected.bool() & trade
    if not bool(trade.any()) or not bool(chosen.any()):
        return scores.clone()
    k = int(chosen.sum().item())
    threshold = torch.topk(scores[trade], k=k).values.min()
    lower = scores[trade].min()
    upper = scores[trade].max()
    span = (upper - lower).clamp_min(1.0)
    step = float(gap) * span
    selected_floor = threshold + step + step * (scores - lower) / span
    excluded_ceiling = threshold - step + step * (scores - upper) / span
    projected = torch.where(chosen, torch.maximum(scores, selected_floor), scores)
    projected = torch.where(trade & ~chosen, torch.minimum(projected, excluded_ceiling), projected)
    return projected


def select_swap_budget(
    scores: torch.Tensor,
    tradable: torch.Tensor,
    previous_selected: torch.Tensor,
    *,
    k: int,
    swap_budget: int,
) -> torch.Tensor:
    """Retain the strongest previous names, then fill with strongest entrants."""
    trade = tradable.bool()
    old = previous_selected.bool() & trade
    k = min(max(int(k), 0), int(trade.sum().item()))
    keep = min(int(old.sum().item()), max(0, k - max(int(swap_budget), 0)))
    selected = torch.zeros_like(trade)
    if keep:
        indices = torch.nonzero(old, as_tuple=False).flatten()
        chosen = indices[torch.topk(scores[indices], k=keep).indices]
        selected[chosen] = True
    fill = k - keep
    if fill:
        indices = torch.nonzero(trade & ~selected, as_tuple=False).flatten()
        chosen = indices[torch.topk(scores[indices], k=fill).indices]
        selected[chosen] = True
    return selected


def select_hysteresis(
    scores: torch.Tensor,
    tradable: torch.Tensor,
    previous_selected: torch.Tensor,
    *,
    k: int,
    max_swaps: int,
    margin: float,
) -> torch.Tensor:
    """Replace incumbents only when a challenger clears a quality margin."""
    trade = tradable.bool()
    k = min(max(int(k), 0), int(trade.sum().item()))
    old_indices = torch.nonzero(previous_selected.bool() & trade, as_tuple=False).flatten()
    if old_indices.numel() > k:
        old_indices = old_indices[torch.topk(scores[old_indices], k=k).indices]
    selected = torch.zeros_like(trade)
    selected[old_indices] = True
    mandatory = k - int(old_indices.numel())
    entrants = torch.nonzero(trade & ~selected, as_tuple=False).flatten()
    if entrants.numel():
        entrants = entrants[torch.argsort(scores[entrants], descending=True, stable=True)]
    if mandatory:
        selected[entrants[:mandatory]] = True
        entrants = entrants[mandatory:]
    incumbents = old_indices[torch.argsort(scores[old_indices], stable=True)]
    candidate_count = min(int(incumbents.numel()), int(entrants.numel()), max(0, int(max_swaps)))
    if candidate_count:
        gains = scores[entrants[:candidate_count]] - scores[incumbents[:candidate_count]]
        # Gains are nonincreasing: entrants descend while incumbents ascend.
        approved = int((gains > float(margin)).sum().item())
        selected[incumbents[:approved]] = False
        selected[entrants[:approved]] = True
    return selected
