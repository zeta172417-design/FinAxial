from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PREDICTION_COLUMNS = (
    "ts_code", "trade_date", "pred_raw", "pred_rank", "pred_smoothed",
    "y_ret_1d", "flag_limit_up", "eligible", "model", "route", "fold",
)


def percentile_rank(frame: pd.DataFrame, source: str = "pred_raw") -> pd.Series:
    return frame.groupby("trade_date", sort=False)[source].rank(method="average", pct=True)


def add_causal_ewma(frame: pd.DataFrame, alpha: float, source: str = "pred_rank") -> pd.Series:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if alpha == 1.0:
        return frame[source].astype(float).copy()
    ordered = frame.sort_values(["trade_date", "ts_code"])
    smoothed = ordered.groupby("ts_code", sort=False)[source].transform(
        lambda values: values.ewm(alpha=alpha, adjust=False).mean()
    )
    result = pd.Series(index=frame.index, dtype=float)
    result.loc[ordered.index] = smoothed.to_numpy()
    return result


def make_prediction_frame(
    *, panel, date_indices: np.ndarray, predictions: np.ndarray, eligible: np.ndarray,
    model: str, route: str, fold: str, alpha: float,
) -> pd.DataFrame:
    rows = []
    for offset, date_idx in enumerate(date_indices):
        pred = np.asarray(predictions[offset], dtype=np.float64).copy()
        valid = np.asarray(eligible[offset], dtype=bool)
        fallback = float(np.nanmedian(pred[valid])) if valid.any() else 0.0
        pred[~valid | ~np.isfinite(pred)] = fallback
        rows.append(pd.DataFrame({
            "ts_code": np.asarray(panel.codes), "trade_date": int(panel.dates[date_idx]),
            "pred_raw": pred, "y_ret_1d": np.asarray(panel.labels[date_idx]),
            "flag_limit_up": np.asarray(panel.limit_flags[date_idx, :, 0], dtype=np.int8),
            "eligible": valid, "model": model, "route": route, "fold": fold,
        }))
    frame = pd.concat(rows, ignore_index=True)
    frame["pred_rank"] = percentile_rank(frame)
    frame["pred_smoothed"] = add_causal_ewma(frame, alpha)
    return frame[list(PREDICTION_COLUMNS)]


def evaluate_frame(frame: pd.DataFrame, prediction_column: str = "pred_raw") -> dict[str, Any]:
    """Replicate evaluate.py exactly, plus diagnostics requested by the plan."""
    ic_values: list[float] = []
    excess: list[float] = []
    absolute: list[float] = []
    daily: dict[int, dict[str, float]] = defaultdict(dict)
    squared, absolute_errors = [], []
    for date, group in frame.groupby("trade_date", sort=True):
        labelled = group.dropna(subset=["y_ret_1d"])
        if len(labelled) >= 30:
            ic = float(spearmanr(labelled[prediction_column], labelled["y_ret_1d"]).statistic)
            ic_values.append(ic)
            daily[int(date)]["rank_ic"] = ic
        error = labelled[prediction_column].to_numpy() - labelled["y_ret_1d"].to_numpy()
        squared.extend(np.square(error).tolist())
        absolute_errors.extend(np.abs(error).tolist())
        tradable = group[(group["flag_limit_up"] == 0) & group["y_ret_1d"].notna()].copy()
        if len(tradable) >= 100:
            tradable = tradable.sort_values(prediction_column, ascending=False).reset_index(drop=True)
            n_top = max(len(tradable) // 10, 1)
            top_return = float(tradable["y_ret_1d"].iloc[:n_top].mean())
            market_return = float(tradable["y_ret_1d"].mean())
            absolute.append(top_return)
            excess.append(top_return - market_return)
            daily[int(date)].update(top_return=top_return, market_return=market_return, excess=top_return - market_return)

    turnovers: list[float] = []
    previous: set[str] | None = None
    for date, group in frame.groupby("trade_date", sort=True):
        tradable = group[group["flag_limit_up"] == 0].sort_values(prediction_column, ascending=False)
        if len(tradable) < 100:
            previous = None
            continue
        current = set(tradable["ts_code"].iloc[:max(len(tradable) // 10, 1)])
        if previous:
            turnover = 1.0 - len(current & previous) / len(current | previous)
            turnovers.append(turnover)
            daily[int(date)]["turnover"] = turnover
        previous = current

    ic_array = np.asarray(ic_values, dtype=float)
    ic_mean = float(np.mean(ic_array)) if len(ic_array) else float("nan")
    ic_std = float(np.std(ic_array, ddof=1)) if len(ic_array) > 1 else float("nan")
    annual_excess = float(np.mean(excess) * 252) if excess else float("nan")
    top_annual = float(np.mean(absolute) * 252) if absolute else float("nan")
    turnover = float(np.mean(turnovers)) if turnovers else float("nan")
    result: dict[str, Any] = {
        "mse": float(np.mean(squared)) if squared else float("nan"),
        "mae": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else 0.0,
        "ic_positive_ratio": float(np.mean(ic_array > 0)) if len(ic_array) else float("nan"),
        "annual_excess": annual_excess,
        "top1_annual_ret": top_annual,
        "mean_turnover": turnover,
        "final_score": 0.4 * ic_mean + 0.3 * annual_excess + 0.3 * (1.0 - turnover),
        "coverage": float(frame["eligible"].mean()) if "eligible" in frame else 1.0,
        "daily": daily,
    }
    yearly = {}
    for year, group in frame.groupby(frame["trade_date"] // 10000):
        daily_group = [daily[int(date)] for date in sorted(group["trade_date"].unique()) if int(date) in daily]
        top = [item["top_return"] for item in daily_group if "top_return" in item]
        exc = [item["excess"] for item in daily_group if "excess" in item]
        yearly[str(int(year))] = {
            "top_cumulative": float(np.prod(1 + np.asarray(top)) - 1) if top else float("nan"),
            "excess_cumulative": float(np.prod(1 + np.asarray(exc)) - 1) if exc else float("nan"),
        }
    result["yearly"] = yearly
    return result
