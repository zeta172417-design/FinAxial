#!/usr/bin/env python3
"""Compare daily group advantages against value-based GAE/PPO on cached C0."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from finmodel.decision_cache import DecisionFeatureCache, cache_source_hashes
from finmodel.decision_validation import with_validation_burnin
from finmodel.decision_factors import (
    DECISION_FACTOR_PROFILES, DecisionFactorFeatureStore,
)
from finmodel.decision_history import (
    causal_c0_bridge_batch, observed_market_features, observed_return_features,
)
from finmodel.decision_rl import (
    DecisionStateValueHead,
    DecisionValueHead,
    center_daily_rewards_against_group,
    critic_observations,
    discounted_return_to_go,
    exact_daily_score,
    generalized_advantage,
)
from finmodel.io import atomic_json_dump, seed_everything
from finmodel.models import build_decision_policy, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.pipeline import build_dataset, load_backbone, score_numpy_predictions
from finmodel.sft import (
    mirror_last_checkpoint,
    use_last_checkpoint_policy,
    cosine_learning_rate, job_name, load_config, panel_indices,
    reset_peak_memory, sft_split, swan_settings,
)


class NullTracker:
    def log(self, values, *, step=None):
        del values, step


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(array, copy=True)).to(device)


def cached_block(cache, panel, output_dates, burnin, device, *, official_trade_universe=False,
                 history_days=0, observation_mode="mean_pool", return_feature_mode="proxy",
                 return_scale=0.02, factor_store=None):
    score_rows = cache.rows_for_dates(output_dates)
    if not np.all(np.diff(score_rows) == 1):
        raise ValueError("one training block must have consecutive cached dates")
    first = max(0, int(score_rows[0]) - int(burnin))
    last = int(score_rows[-1]) + 1
    date_indices = np.asarray(cache.date_indices[first:last], dtype=np.int64)
    scored_start = int(score_rows[0]) - first
    hidden = to_device(cache.hidden[first:last], device)
    base = to_device(cache.base_score[first:last], device)
    eligible = to_device(cache.eligible[first:last], device)
    tradable = to_device(
        ~np.asarray(panel.limit_flags[date_indices, :, 0], dtype=bool)
        if official_trade_universe else cache.tradable[first:last], device,
    )
    target = to_device(panel.labels[date_indices[scored_start:]], device)
    label_valid = to_device(panel.label_valid[date_indices[scored_start:]], device)
    label_mask = label_valid if official_trade_universe else eligible[scored_start:] & label_valid
    observed = (
        to_device(
            observed_market_features(panel, date_indices)
            if observation_mode == "candidate_attention"
            else observed_return_features(panel, date_indices), device,
        )
        if history_days else None
    )
    if return_feature_mode == "explicit" and cache.predicted_return is None:
        raise ValueError("explicit return mode requires a dual-head predicted_return cache")
    return_prediction = (
        to_device(cache.predicted_return[first:last], device)
        if return_feature_mode == "explicit" else None
    )
    result = (hidden, base, eligible, tradable, target, label_mask, scored_start, observed, return_prediction)
    if factor_store is None:
        return result
    return (*result, to_device(factor_store.rows(date_indices), device))


@torch.inference_mode()
def validation_history_prefix(train_cache, validation_cache, panel, config, device, days,
                              observation_mode="mean_pool"):
    """Build only the causal score history preceding validation; never load its labels."""
    if days <= 0:
        return None
    first = int(validation_cache.date_indices[0])
    dates = np.arange(first - days + 1, first, dtype=np.int64)
    if len(dates) != days - 1 or dates[0] <= 0:
        raise ValueError("not enough panel history before validation")
    scores = np.empty((len(dates), panel.shape[1]), dtype=np.float32)
    eligibility = np.empty(scores.shape, dtype=bool)
    source_first = int(train_cache.date_indices[0])
    source_last = int(train_cache.date_indices[-1])
    gap_dates = []
    for position, date in enumerate(dates):
        if source_first <= date <= source_last:
            row = date - source_first
            scores[position] = train_cache.base_score[row]
            eligibility[position] = train_cache.eligible[row]
        else:
            gap_dates.append((position, int(date)))
    if gap_dates:
        if len(gap_dates) != 1 or int(panel.dates[gap_dates[0][1]]) != int(config["boundary_excluded_date"]):
            raise ValueError("unexpected missing C0 score in validation history")
        position, date = gap_dates[0]
        backbone, source_hash = load_backbone(config, panel, device)
        if source_hash != train_cache.manifest["backbone_sha256"]:
            raise ValueError("validation bridge C0 checkpoint hash mismatch")
        bridge = causal_c0_bridge_batch(panel, date, config)
        values, token_valid, bridge_eligible = bridge[:3]
        hidden = backbone.encode_hidden(
            values.to(device), token_valid.to(device), bridge_eligible.to(device),
            long_memory=(bridge[3].to(device) if len(bridge) == 4 else None),
        )
        score = backbone.score_hidden(hidden, bridge_eligible.to(device))
        scores[position] = score[-1].float().cpu().numpy()
        eligibility[position] = bridge_eligible[-1].numpy()
        del backbone, hidden, score
        torch.cuda.empty_cache()
    observed = (
        observed_market_features(panel, dates)
        if observation_mode == "candidate_attention" else observed_return_features(panel, dates)
    )
    return (
        to_device(scores, device), to_device(eligibility, device),
        to_device(observed, device),
    )


@torch.inference_mode()
def evaluate_cached(policy, cache, panel, device, *, route, base_metrics,
                    official_trade_universe=False, history_prefix=None, factor_store=None,
                    score_prefix=0):
    if score_prefix < 0 or score_prefix >= len(cache.date_indices):
        raise ValueError("invalid validation score prefix")
    policy.eval()
    hidden = to_device(cache.hidden, device)
    base = to_device(cache.base_score, device)
    eligible = to_device(cache.eligible, device)
    tradable = to_device(
        ~np.asarray(panel.limit_flags[cache.date_indices, :, 0], dtype=bool)
        if official_trade_universe else cache.tradable, device,
    )
    history_kwargs = (
        {
            "observed_return": to_device(
                observed_market_features(panel, np.asarray(cache.date_indices))
                if policy.observation_mode == "candidate_attention"
                else observed_return_features(panel, np.asarray(cache.date_indices)), device,
            ),
            "history_prefix": history_prefix,
        } if policy.history_days else {}
    )
    if policy.return_feature_mode == "explicit" and cache.predicted_return is None:
        raise ValueError("explicit return mode requires a dual-head predicted_return cache")
    return_kwargs = (
        {"predicted_return": to_device(cache.predicted_return, device)}
        if policy.return_feature_mode == "explicit" else {}
    )
    output = policy(
        hidden, base, eligible, tradable, sample=False,
        **history_kwargs, **return_kwargs,
        **({"decision_factors": to_device(factor_store.rows(np.asarray(cache.date_indices)), device)}
           if factor_store is not None else {}),
    )
    policy_raw, _ = score_numpy_predictions(
        panel=panel,
        indices=np.asarray(cache.date_indices[score_prefix:]),
        predictions=output.decision_score[score_prefix:].float().cpu().numpy(),
        eligible=eligible[score_prefix:].cpu().numpy(),
        route=route,
        ewma_alphas=(1.0,),
    )
    diagnostics = {
        "swap_budget_mean": float(output.swap_budget[score_prefix:].mean().cpu()),
        "swap_budget_std": float(output.swap_budget[score_prefix:].std().cpu()),
        "realized_swaps_mean": float(output.realized_swaps[score_prefix:].mean().cpu()),
        "realized_swaps_std": float(output.realized_swaps[score_prefix:].std().cpu()),
        "candidate_count_mean": float(output.candidate_count[score_prefix:].mean().cpu()),
        "candidate_bonus_abs_mean": float(output.candidate_bonus_abs_mean[score_prefix:].mean().cpu()),
        "action_value_mean": output.action_value[score_prefix:].float().mean(dim=0).cpu().tolist(),
        "action_value_std": output.action_value[score_prefix:].float().std(dim=0).cpu().tolist(),
        "policy_std": output.policy_std.float().cpu().tolist(),
        "reference_kl": float(output.reference_kl.cpu()),
    }
    if policy.action_mode in {"hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"}:
        margin_index = 0 if policy.action_mode in {
            "hysteresis_no_alpha", "hysteresis_margin_only",
        } else 1
        diagnostics["hysteresis_margin_mean"] = float(output.action_value[score_prefix:, margin_index].mean().cpu())
        diagnostics["hysteresis_margin_std"] = float(output.action_value[score_prefix:, margin_index].std(unbiased=False).cpu())
    del hidden, base, eligible, tradable, output
    torch.cuda.empty_cache()
    return {"policy_raw": policy_raw, "diagnostics": diagnostics, **base_metrics}


@torch.inference_mode()
def evaluate_base(cache, panel):
    base, _ = score_numpy_predictions(
        panel=panel,
        indices=np.asarray(cache.date_indices),
        predictions=np.asarray(cache.base_score, dtype=np.float32),
        eligible=np.asarray(cache.eligible, dtype=bool),
        route="stage2_daily_cached_c0",
        ewma_alphas=(1.0,),
    )
    return {"stage1_raw": base}


ACTIVE_ACTION_MODES = frozenset({"hysteresis_no_alpha", "hysteresis_margin_only"})
LEGACY_ALPHA_KEYS = frozenset({
    "alpha_minimum", "alpha_maximum", "initial_alpha",
    "alpha_parameterization", "alpha_linear_scale",
})


def require_unsmoothed_policy(config):
    """Reject learnable/fixed EWMA routes and strip obsolete alpha settings."""
    policy = config["policy"]
    action_mode = str(policy["action_mode"])
    if action_mode not in ACTIVE_ACTION_MODES:
        raise ValueError(
            f"new decision training requires an unsmoothed action mode "
            f"{sorted(ACTIVE_ACTION_MODES)}; got {action_mode!r}"
        )
    for key in LEGACY_ALPHA_KEYS:
        policy.pop(key, None)
    return action_mode


def decision_blocks_from_cache_dates(date_indices, *, block_days, stride):
    """Use cached predictor dates without tying RL rollout length to its window."""
    dates = np.asarray(date_indices, dtype=np.int64)
    if dates.ndim != 1 or len(dates) < block_days or block_days < 2 or stride < 1:
        raise ValueError("invalid decision block length, stride, or cache dates")
    if not np.all(np.diff(dates) == 1):
        raise ValueError("decision cache dates must be consecutive")
    last_start = len(dates) - block_days
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return np.stack([dates[start:start + block_days] for start in starts])


def group_normalize_by_date(values, epsilon, clip):
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("daily advantages require [rollouts >= 2, dates]")
    normalized = (values - values.mean(dim=0, keepdim=True)) / (
        values.std(dim=0, unbiased=False, keepdim=True).clamp_min(epsilon)
    )
    return normalized.clamp(-clip, clip)


def normalize_advantages(values, *, algorithm, epsilon, clip,
                         mode="legacy", scale_floor=0.0):
    """Keep the legacy PPO normalization while enabling matched-date controls."""
    if mode == "same_date_center_global_scale":
        if values.ndim != 2 or values.shape[0] < 2 or scale_floor < 0:
            raise ValueError("global-scale advantages need [rollouts >= 2, dates] and nonnegative floor")
        centered = values - values.mean(dim=0, keepdim=True)
        scale = centered.std(unbiased=False).clamp_min(max(epsilon, scale_floor))
        return (centered / scale).clamp(-clip, clip)
    if mode == "none":
        # Raw GAE: no centering, rescaling or advantage clipping. PPO's
        # probability-ratio clipping and gradient clipping remain enabled.
        return values
    if mode == "global_rollout_date":
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("global advantages require [rollouts >= 2, dates]")
        return ((values - values.mean()) /
                values.std(unbiased=False).clamp_min(epsilon)).clamp(-clip, clip)
    if mode != "legacy":
        raise ValueError(f"unsupported advantage normalization mode: {mode}")
    if algorithm == "ppo_gae":
        return ((values - values.mean()) /
                values.std(unbiased=False).clamp_min(epsilon)).clamp(-clip, clip)
    return group_normalize_by_date(values, epsilon, clip)


def explained_variance(prediction, target, epsilon=1e-8):
    """Value-fit diagnostic; zero when target variance is not informative."""
    target_variance = target.float().var(unbiased=False)
    if bool(target_variance <= epsilon):
        return target.new_zeros(())
    return 1.0 - (target.float() - prediction.float()).var(unbiased=False) / target_variance


def split_batched_policy_output(output):
    """Expose batched trajectory rows to the unchanged reward/critic code."""
    count = output.raw_action.shape[0]
    return [type(output)(**{
        name: (getattr(output, name) if name == "policy_std" or getattr(output, name) is None
               else getattr(output, name)[row])
        for name in output._fields
    }) for row in range(count)]


def rollout_minibatches(rollouts_per_rank, minibatch_size, passes, *, seed):
    """Partition whole local trajectories; use each exactly once per pass.

    Keep legacy full-buffer updates in their original order. For minibatches,
    shuffle independently from the policy's sampling RNG. All ranks make the
    same number of optimizer steps, each with the same local batch size.
    """
    if (rollouts_per_rank < 1 or minibatch_size < 1 or passes < 1
            or rollouts_per_rank % minibatch_size):
        raise ValueError("rollout count must be divisible by a positive minibatch size")
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(passes):
        order = (np.arange(rollouts_per_rank) if minibatch_size == rollouts_per_rank
                 else rng.permutation(rollouts_per_rank))
        batches.extend(order[start:start + minibatch_size]
                       for start in range(0, rollouts_per_rank, minibatch_size))
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision_grpo.json")
    parser.add_argument("--algorithm", required=True, choices=("daily_group",))
    parser.add_argument("--action-mode", choices=sorted(ACTIVE_ACTION_MODES))
    parser.add_argument("--observation-mode", choices=(
        "mean_pool", "candidate_attention", "portfolio_detail",
    ))
    parser.add_argument("--panel", default="artifacts/reproduction/panel")
    parser.add_argument("--cache", default="artifacts/reproduction/cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gamma", type=float, help="Override the configured reward discount")
    parser.add_argument("--gae-lambda", type=float, help="Override the configured GAE lambda")
    parser.add_argument("--run-suffix", default="", help="Unique SwanLab name suffix for an ablation")
    parser.add_argument("--initial-hysteresis-margin", type=float,
                        help="Override the decision policy's initial margin")
    parser.add_argument("--initial-swap-budget", type=int,
                        help="Override the decision policy's initial active-swap budget")
    parser.add_argument("--history-days", type=int, default=0,
                        help="Use a per-stock causal history encoder; 0 retains the legacy actor")
    parser.add_argument("--swanlab-logdir", help="Override only the local SwanLab log directory")
    parser.add_argument("--limit-train-blocks", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--limit-validation-days", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--disable-swanlab", action="store_true")
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch decision RL with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("FINMODEL_DDP_TIMEOUT_SECONDS", "1800"))),
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    config = load_config(args.config)
    if args.swanlab_logdir:
        config["swanlab"]["logdir"] = args.swanlab_logdir
    if args.action_mode:
        config["policy"]["action_mode"] = args.action_mode
    if args.initial_hysteresis_margin is not None:
        config["policy"]["initial_hysteresis_margin"] = args.initial_hysteresis_margin
    if args.initial_swap_budget is not None:
        config["policy"]["initial_swap_budget"] = args.initial_swap_budget
    if args.observation_mode:
        config["policy"]["observation_mode"] = args.observation_mode
    if args.history_days:
        config["policy"]["history_days"] = args.history_days
    history_days = int(config["policy"].get("history_days", 0))
    observation_mode = str(config["policy"].get("observation_mode", "mean_pool"))
    if observation_mode == "candidate_attention":
        for key, value in {
            "history_hidden_dim": 32,
            "candidate_top_fraction": 0.15,
            "candidate_max_count": 1024,
            "candidate_context_dim": 64,
            "candidate_queries": 4,
            "maximum_candidate_bonus": 0.04,
            "candidate_boundary_width": 0.08,
        }.items():
            config["policy"].setdefault(key, value)
    if history_days < 0 or history_days == 1:
        raise ValueError("history days must be 0 or at least 2")
    action_mode = require_unsmoothed_policy(config)
    training = config["training"]
    use_last_checkpoint_policy(training)
    rollout_execution = str(training.get("rollout_execution", "legacy"))
    same_state_branches = bool(training.get("same_state_branches", False))
    if same_state_branches and (
        args.algorithm != "daily_group" or rollout_execution != "cached_batched"
        or int(training["minibatch_rollouts_per_rank"]) < 2
    ):
        raise ValueError("same-state branching requires batched daily_group with minibatch > 1")
    if rollout_execution not in {"legacy", "cached_serial", "cached_batched"}:
        raise ValueError("rollout_execution must be legacy, cached_serial, or cached_batched")
    if rollout_execution != "legacy" and not (
        action_mode == "hysteresis_no_alpha"
        and observation_mode == "portfolio_detail"
        and config["policy"].get("return_feature_mode") == "explicit"
        and not config["policy"].get("decision_factor_dim", 0)
        and history_days == 0
    ):
        raise ValueError("cached rollout execution supports only portfolio-detail two-action route")
    critic_return_mode = str(training.get("critic_return_mode", "proxy"))
    if critic_return_mode not in {"proxy", "predicted_return"}:
        raise ValueError("critic_return_mode must be proxy or predicted_return")
    if critic_return_mode == "predicted_return" and config["policy"].get("return_feature_mode") != "explicit":
        raise ValueError("predicted_return critic requires an explicit dual-head return cache")
    critic_target_mode = str(training.get("critic_target_mode", "raw"))
    if critic_target_mode not in {"raw", "group_centered"}:
        raise ValueError("critic_target_mode must be raw or group_centered")
    if critic_target_mode == "group_centered" and args.algorithm not in {"ppo_gae", "ppo_gae_group"}:
        raise ValueError("group-centered critic targets require PPO with a critic")
    critic_observation_mode = str(training.get("critic_observation_mode", "legacy"))
    if critic_observation_mode not in {"legacy", "portfolio_detail", "portfolio_detail_memory_age"}:
        raise ValueError("unsupported critic_observation_mode")
    enhanced_critic = critic_observation_mode == "portfolio_detail_memory_age"
    if enhanced_critic and (rollout_execution != "cached_batched" or args.algorithm not in {"ppo_gae", "ppo_gae_group"}):
        raise ValueError("memory/age critic requires cached_batched PPO")
    advantage_mode = str(training.get("advantage_normalization_mode", "legacy"))
    advantage_scale_floor = float(training.get("advantage_scale_floor", 0.0))
    if advantage_mode not in {"legacy", "same_date_center_global_scale", "none", "global_rollout_date"}:
        raise ValueError(f"unsupported advantage normalization mode: {advantage_mode}")
    if not np.isfinite(advantage_scale_floor) or advantage_scale_floor < 0:
        raise ValueError("advantage_scale_floor must be finite and non-negative")
    if advantage_mode == "same_date_center_global_scale" and args.algorithm == "ppo_gae":
        raise ValueError("same-date centering is incompatible with global PPO advantages")
    if args.gamma is not None:
        training["gamma"] = args.gamma
    if args.gae_lambda is not None:
        training["gae_lambda"] = args.gae_lambda
    if not 0 <= float(training["gamma"]) <= 1 or not 0 <= float(training["gae_lambda"]) <= 1:
        raise ValueError("gamma and GAE lambda must be in [0, 1]")
    reward_horizon_days = training.get("reward_horizon_days")
    if reward_horizon_days is not None:
        if args.algorithm != "rtg_group" or isinstance(reward_horizon_days, bool) or not isinstance(reward_horizon_days, int):
            raise ValueError("reward_horizon_days requires rtg_group and an integer horizon")
        if not 1 <= reward_horizon_days <= int(training["decision_block_days"]):
            raise ValueError("reward_horizon_days must fit within decision_block_days")
    if args.run_suffix and not re.fullmatch(r"[a-z0-9_-]+", args.run_suffix):
        raise ValueError("run suffix must contain only lowercase letters, digits, - or _")
    if world_size != int(training["expected_world_size"]):
        raise ValueError(f"expected {training['expected_world_size']} ranks, found {world_size}")
    if history_days and args.algorithm != "daily_group":
        raise ValueError("history-aware ablations use daily_group and hysteresis-family actions")
    if observation_mode == "candidate_attention" and history_days < 2:
        raise ValueError("candidate attention requires history_days >= 2")
    epochs = int(args.epochs or training["max_epochs"])
    if not 1 <= epochs <= int(training["max_epochs"]):
        raise ValueError("epochs exceed configured range")
    rollouts_per_rank = int(training["rollouts_per_rank"])
    if rollouts_per_rank < 1 or world_size * rollouts_per_rank < 2:
        raise ValueError("at least two same-block rollouts are required")
    minibatch_rollouts_per_rank = int(training.get("minibatch_rollouts_per_rank", rollouts_per_rank))
    updates_per_rollout = int(training["policy_updates_per_rollout"])
    if (minibatch_rollouts_per_rank < 1
            or rollouts_per_rank % minibatch_rollouts_per_rank
            or updates_per_rollout < 1):
        raise ValueError("invalid rollout minibatch size or policy pass count")
    updates_per_collection = updates_per_rollout * (rollouts_per_rank // minibatch_rollouts_per_rank)
    seed = int(config["seed"])
    seed_everything(seed + rank)
    panel = Panel.open(args.panel)
    factor_store = None
    if config["policy"].get("decision_factor_dim", 0):
        factor_profile = str(config["data"].get("decision_factor_profile", "legacy14"))
        if factor_profile not in DECISION_FACTOR_PROFILES:
            raise ValueError(f"unknown decision factor profile: {factor_profile}")
        if int(config["policy"]["decision_factor_dim"]) != len(DECISION_FACTOR_PROFILES[factor_profile]):
            raise ValueError("decision_factor_dim does not match decision_factor_profile")
        factor_store = DecisionFactorFeatureStore.open(
            panel, config["data"].get("base_factor_cache", config["data"]["factor_cache"]),
            config["data"]["decision_factor_cache"],
            expected_train_end=int(config["tuning_train_end"]),
            profile=factor_profile,
        )
    split = sft_split(panel, config)
    checkpoint_hash, panel_hash = cache_source_hashes(
        panel, config["backbone_checkpoint"],
    )
    train_cache = DecisionFeatureCache.open(
        Path(args.cache) / "train", checkpoint_sha256=checkpoint_hash,
        panel_manifest_sha256=panel_hash,
    )
    validation_cache = DecisionFeatureCache.open(
        Path(args.cache) / "validation", checkpoint_sha256=checkpoint_hash,
        panel_manifest_sha256=panel_hash,
    )
    if "decision_block_days" in training:
        blocks = decision_blocks_from_cache_dates(
            train_cache.date_indices,
            block_days=int(training["decision_block_days"]),
            stride=int(training.get("decision_train_stride", config["data"]["train_stride"])),
        )
    else:
        expected_train = build_dataset(
            panel, panel_indices(panel, split.training_dates), config,
            stride=int(config["data"]["train_stride"]),
        )
        blocks = expected_train.output_blocks
    if args.limit_train_blocks:
        blocks = blocks[-args.limit_train_blocks:]
    if args.limit_validation_days:
        # Smoke mode limits only this process-local in-memory view.
        from dataclasses import replace
        days = int(args.limit_validation_days)
        validation_cache = replace(
            validation_cache,
            date_indices=validation_cache.date_indices[:days],
            hidden=validation_cache.hidden[:days],
            base_score=validation_cache.base_score[:days],
            eligible=validation_cache.eligible[:days],
            tradable=validation_cache.tradable[:days],
            predicted_return=(
                validation_cache.predicted_return[:days]
                if validation_cache.predicted_return is not None else None
            ),
        )
    if len(validation_cache.date_indices) != int(config["validation_days"]) and not args.limit_validation_days:
        raise ValueError("validation cache does not cover the configured test period")

    validation_burnin = int(config.get("validation", {}).get("decision_burnin_days", 0))
    evaluation_cache = (
        with_validation_burnin(train_cache, validation_cache, panel, config, device, validation_burnin)
        if rank == 0 else validation_cache
    )

    policy = build_decision_policy(
        d_model=int(config["model"]["d_model"]), **config["policy"],
    ).to(device)
    history_prefix = (
        validation_history_prefix(
            train_cache, validation_cache, panel, config, device, history_days,
            observation_mode=observation_mode,
        ) if rank == 0 and history_days and not args.limit_validation_days else None
    )
    actor = DDP(
        policy, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    )
    critic = None
    critic_ddp = None
    critic_optimizer = None
    if args.algorithm in {"ppo_gae", "ppo_gae_group"}:
        if enhanced_critic:
            critic = DecisionStateValueHead(
                d_model=int(config["model"]["d_model"]),
                hidden_dim=int(training["critic_hidden_dim"]),
                recurrent_dim=policy.recurrent.hidden_size,
                candidate_max_count=policy.candidate_max_count,
                return_scale=policy.return_scale,
            ).to(device)
        else:
            critic = DecisionValueHead(
                d_model=int(config["model"]["d_model"]),
                hidden_dim=int(training["critic_hidden_dim"]),
                portfolio_detail=critic_observation_mode == "portfolio_detail",
            ).to(device)
        critic_ddp = DDP(
            critic, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )
        critic_optimizer = torch.optim.AdamW(
            critic_ddp.parameters(), lr=float(training["critic_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    output_dir = Path(args.output)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "complete.json").exists():
            raise FileExistsError(f"experiment already complete: {output_dir}")
        atomic_json_dump({
            "algorithm": args.algorithm,
            "run_suffix": args.run_suffix,
            "config": config,
        }, output_dir / "config_resolved.json")
    dist.barrier()
    reset_peak_memory(device)
    total_updates = len(blocks) * epochs * updates_per_collection
    warmup_updates = len(blocks) * int(training["warmup_epochs"]) * updates_per_collection
    run_action = action_mode if args.algorithm == "daily_group" else f"{action_mode}-{args.algorithm}"
    if history_days:
        run_action = f"{run_action}-history{history_days}"
        if observation_mode != "mean_pool":
            run_action = f"{run_action}-{observation_mode}"
    if args.run_suffix:
        run_action = f"{run_action}-{args.run_suffix}"
    advantage_normalization = (
        advantage_mode if advantage_mode in {"same_date_center_global_scale", "none", "global_rollout_date"}
        else ("global_rollout_date" if args.algorithm == "ppo_gae" else "same_date_group")
    )
    name = job_name(
        "rl", "finaxial-stage2-topk", run_action,
        int(config["model"]["lookback"]), seed,
        budget=f"k{config['model']['output_steps']}-g{world_size * rollouts_per_rank}-ep{epochs}-ddp{world_size}",
    )
    tracker_context = swan_settings(
        config, name=name,
        tags=["stage2", "daily-reward", "cached-c0", args.algorithm, action_mode],
        extra={
            "algorithm": args.algorithm,
            "action_mode": action_mode,
            "cache": str(args.cache),
            "backbone_sha256": checkpoint_hash,
            "rollouts_per_group": world_size * rollouts_per_rank,
            "minibatch_rollouts_global": world_size * minibatch_rollouts_per_rank,
            "policy_passes_per_collection": updates_per_rollout,
            "optimizer_updates_per_collection": updates_per_collection,
            "train_blocks": len(blocks),
            "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic else 0,
            "advantage_normalization": advantage_normalization,
            "critic_target_mode": critic_target_mode if critic is not None else None,
            "critic_observation_mode": critic_observation_mode if critic is not None else None,
            "history_days": history_days,
            "observation_mode": observation_mode,
            "policy_parameters": sum(p.numel() for p in policy.parameters()),
            "policy_config": config["policy"],
            **training,
        },
    ) if rank == 0 and not args.disable_swanlab else nullcontext(NullTracker())

    history = []
    best_stable = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    global_update = 0
    started = time.perf_counter()
    base_metrics = evaluate_base(validation_cache, panel) if rank == 0 else {}

    def validate(epoch, tracker):
        nonlocal best_stable, best_epoch, bad_epochs
        dist.barrier()
        if rank == 0:
            result = evaluate_cached(
                policy, evaluation_cache, panel, device,
                route=f"stage2_daily_{args.algorithm}_{action_mode}", base_metrics=base_metrics,
                official_trade_universe=bool(config.get("official_trade_universe", False)),
                history_prefix=history_prefix,
                factor_store=factor_store,
                score_prefix=validation_burnin,
            )
            row = {"epoch": epoch, **result}
            history.append(row)
            window = history[-int(training["validation_smoothing_epochs"]):]
            stable = float(np.mean([item["policy_raw"]["final_score"] for item in window]))
            row["stable_selection_mean"] = stable
            atomic_json_dump(history, output_dir / "validation_history.json")
            metrics = result["policy_raw"]
            diagnostics = result["diagnostics"]
            tracker.log({
                "validation/final_score": metrics["final_score"],
                "validation/rank_ic": metrics["rank_ic"],
                "validation/annual_excess": metrics["annual_excess"],
                "validation/one_minus_turnover": metrics["one_minus_turnover"],
                "validation/stable_final_mean": stable,
                "validation/stage1_raw/final_score": base_metrics["stage1_raw"]["final_score"],
                "validation/policy/swap_budget_mean": diagnostics["swap_budget_mean"],
                "validation/policy/swap_budget_std": diagnostics["swap_budget_std"],
                "validation/policy/realized_swaps_mean": diagnostics["realized_swaps_mean"],
                "validation/policy/realized_swaps_std": diagnostics["realized_swaps_std"],
                "validation/policy/candidate_count_mean": diagnostics["candidate_count_mean"],
                "validation/policy/candidate_bonus_abs_mean": diagnostics["candidate_bonus_abs_mean"],
                **({
                    "validation/policy/hysteresis_margin_mean": diagnostics["hysteresis_margin_mean"],
                    "validation/policy/hysteresis_margin_std": diagnostics["hysteresis_margin_std"],
                } if action_mode in {"hysteresis", "hysteresis_no_alpha", "hysteresis_margin_only", "candidate_residual"} else {}),
                "validation/policy/reference_kl": diagnostics["reference_kl"],
                "validation/epoch": epoch,
                "validation/optimizer_update": global_update,
            }, step=global_update)
            # Save the last completed training epoch regardless of validation.
            # best/ remains a compatibility copy, not a validation-selected run.
            if epoch > 0:
                best_stable, best_epoch, bad_epochs = stable, epoch, 0
                best_dir = output_dir / "last"
                best_dir.mkdir(exist_ok=True)
                torch.save(policy.state_dict(), best_dir / "policy.pt")
                if critic is not None:
                    torch.save(critic.state_dict(), best_dir / "critic.pt")
                atomic_json_dump({
                    "algorithm": args.algorithm,
                    "checkpoint_selection": "last_completed_epoch",
                    "validation_used_for_checkpoint_selection": False,
                    "action_mode": action_mode,
                    "gamma": float(training["gamma"]),
                    "gae_lambda": float(training["gae_lambda"]) if critic is not None else None,
                    "critic_return_mode": critic_return_mode if critic is not None else None,
                    "critic_target_mode": critic_target_mode if critic is not None else None,
                    "critic_observation_mode": critic_observation_mode if critic is not None else None,
                    "advantage_normalization": advantage_normalization,
                    "history_days": history_days,
                    "observation_mode": observation_mode,
                    "policy_parameters": sum(p.numel() for p in policy.parameters()),
                    "policy_config": config["policy"],
                    "rollouts_per_group": world_size * rollouts_per_rank,
                    "minibatch_rollouts_global": world_size * minibatch_rollouts_per_rank,
                    "policy_passes_per_collection": updates_per_rollout,
                    "optimizer_updates_per_collection": updates_per_collection,
                    "epoch": epoch,
                    "optimizer_update": global_update,
                    "stable_validation_final_score": stable,
                    "score_components": metrics,
                    "validation_burnin_days": validation_burnin,
                    "stage1_raw": base_metrics["stage1_raw"],
                    "policy_diagnostics": diagnostics,
                    "backbone_sha256": checkpoint_hash,
                    "stock_vocab_sha256": stock_vocab_sha256(panel.codes),
                    "cache": str(args.cache),
                }, best_dir / "metadata.json")
                mirror_last_checkpoint(best_dir, ("best",))
            print(
                f"VALIDATION {args.algorithm} epoch={epoch} final={metrics['final_score']:.6f} "
                f"ic={metrics['rank_ic']:.6f} excess={metrics['annual_excess']:.6f} "
                f"stability={metrics['one_minus_turnover']:.6f} ", flush=True,
            )
        stop = torch.zeros((), dtype=torch.uint8, device=device)
        # Validation is monitoring only: it never stops or selects training.
        dist.broadcast(stop, src=0)
        dist.barrier()
        return bool(stop.item())

    try:
        with tracker_context as tracker:
            validate(0, tracker)
            stopped_early = False
            for epoch in range(1, epochs + 1):
                order = np.random.default_rng(seed + epoch).permutation(len(blocks))
                policy.train()
                if critic is not None:
                    critic.train()
                logged = []
                for block_number, block_index in enumerate(order, start=1):
                    block = cached_block(
                        train_cache, panel, blocks[int(block_index)],
                        int(training["policy_burnin_days"]), device,
                        official_trade_universe=bool(config.get("official_trade_universe", False)),
                        history_days=history_days,
                        observation_mode=observation_mode,
                        return_feature_mode=policy.return_feature_mode,
                        return_scale=policy.return_scale,
                        factor_store=factor_store,
                    )
                    hidden, base, eligible, tradable, target, label_mask, prefix, observed, return_prediction = block[:9]
                    decision_factors = block[9] if factor_store is not None else None
                    static_cache = (
                        policy.prepare_static(
                            hidden, base, eligible, tradable,
                            predicted_return=return_prediction,
                        ) if rollout_execution != "legacy" else None
                    )
                    with torch.no_grad():
                        anchor_selected = None
                        anchor_previous = None
                        if same_state_branches:
                            anchor_selected = policy(
                                hidden, base, eligible, tradable,
                                predicted_return=return_prediction, static_cache=static_cache,
                            ).selected.detach()
                            # Share the full causal portfolio timeline across ranks and
                            # reuse it unchanged during all minibatch likelihood replays.
                            dist.broadcast(anchor_selected, src=0)
                            anchor_previous = torch.cat((
                                static_cache.naive_selected[:1], anchor_selected[:-1],
                            ))[prefix:]
                        rollouts, daily_rewards, critic_states = [], [], []
                        critic_detail_states = []
                        old_values, local_future, local_returns = [], [], []
                        if rollout_execution == "cached_batched" and rollouts_per_rank > 1:
                            # Architecture-independent common random numbers.
                            # Mainline runs without this experimental setting
                            # retain their original sampling behavior.
                            noise_kwargs = {}
                            noise_seed = config.get("experiment", {}).get("rollout_noise_seed")
                            if noise_seed is not None:
                                noise_rng = torch.Generator(device="cpu")
                                noise_rng.manual_seed(int(noise_seed) + rank * 100000000
                                                      + epoch * 100000 + block_number)
                                noise_kwargs["noise"] = torch.randn(
                                    rollouts_per_rank, len(base), policy.action_count,
                                    generator=noise_rng,
                                ).to(device)
                            sampled = split_batched_policy_output(policy(
                                hidden, base, eligible, tradable, sample=True,
                                predicted_return=return_prediction,
                                static_cache=static_cache, rollouts=rollouts_per_rank,
                                capture_state=enhanced_critic,
                                anchor_selected=anchor_selected,
                                **noise_kwargs,
                            ))
                        else:
                            sampled = [policy(
                                hidden, base, eligible, tradable, sample=True,
                                capture_state=enhanced_critic,
                                **({"observed_return": observed} if observed is not None else {}),
                                **({"predicted_return": return_prediction}
                                   if return_prediction is not None else {}),
                                **({"decision_factors": decision_factors}
                                   if decision_factors is not None else {}),
                                **({"static_cache": static_cache}
                                   if static_cache is not None else {}),
                            ) for _ in range(rollouts_per_rank)]
                        for output in sampled:
                            prior = output.selected[prefix - 1] if prefix else None
                            reward = exact_daily_score(
                                output.decision_score[prefix:], target,
                                label_mask, tradable[prefix:],
                                previous_selected=prior,
                                previous_selected_by_date=anchor_previous,
                                top_fraction=float(config["policy"]["top_fraction"]),
                            )
                            rollouts.append(output)
                            daily_rewards.append(reward)
                            if critic is not None:
                                obs = critic_observations(
                                    hidden, base, eligible, tradable,
                                    output.selected, output.decision_score,
                                    predicted_return=(return_prediction
                                                      if critic_return_mode == "predicted_return" else None),
                                    return_scale=float(config["policy"]["return_scale"]),
                                    top_fraction=float(config["policy"]["top_fraction"]),
                                    portfolio_detail=critic_observation_mode != "legacy",
                                )[prefix:]
                                detail_state = {}
                                if enhanced_critic:
                                    if output.actor_state is None or output.pre_action_age is None:
                                        raise RuntimeError("enhanced critic requires captured pre-action actor state")
                                    previous_selected = torch.cat((
                                        static_cache.naive_selected[:1], output.selected[:-1],
                                    ))
                                    detail_state = {
                                        "hidden": hidden[prefix:],
                                        "base_rank": static_cache.base_rank[prefix:],
                                        "eligible": eligible[prefix:], "tradable": tradable[prefix:],
                                        "predicted_return": (
                                            return_prediction if critic_return_mode == "predicted_return"
                                            else base * policy.return_scale
                                        )[prefix:],
                                        "previous_selected": previous_selected[prefix:],
                                        "holding_age": output.pre_action_age[prefix:],
                                        "actor_state": output.actor_state[prefix:],
                                    }
                                critic_detail_states.append(detail_state)
                                value = critic(obs, **detail_state)
                                critic_states.append(obs.detach())
                                old_values.append(value.detach())
                                if critic_target_mode == "raw":
                                    gae, returns = generalized_advantage(
                                        reward.reward, value,
                                        gamma=float(training["gamma"]),
                                        lam=float(training["gae_lambda"]),
                                    )
                                    local_future.append(gae.detach())
                                    local_returns.append(returns.detach())
                            elif args.algorithm == "rtg_group":
                                local_future.append(discounted_return_to_go(
                                    reward.reward, gamma=float(training["gamma"]),
                                    horizon=reward_horizon_days,
                                ).detach())
                        local_daily = torch.stack([row.reward for row in daily_rewards])
                        grouped = [torch.empty_like(local_daily) for _ in range(world_size)]
                        dist.all_gather(grouped, local_daily)
                        group_daily = torch.cat(grouped)
                        if critic is not None and critic_target_mode == "group_centered":
                            centered_rewards = center_daily_rewards_against_group(
                                local_daily, group_daily,
                            )
                            for reward_row, value in zip(centered_rewards, old_values):
                                gae, returns = generalized_advantage(
                                    reward_row, value,
                                    gamma=float(training["gamma"]),
                                    lam=float(training["gae_lambda"]),
                                )
                                local_future.append(gae.detach())
                                local_returns.append(returns.detach())
                        local_action_values = torch.stack([
                            row.action_value[prefix:] for row in rollouts
                        ])
                        gathered_action_values = [
                            torch.empty_like(local_action_values)
                            for _ in range(world_size)
                        ]
                        dist.all_gather(gathered_action_values, local_action_values)
                        group_action_values = torch.cat(gathered_action_values)
                        rollout_reward_std = group_daily.std(
                            dim=0, unbiased=False,
                        ).mean()
                        rollout_action_std = group_action_values.std(
                            dim=0, unbiased=False,
                        ).mean(dim=0)
                        if args.algorithm == "daily_group":
                            all_advantage = group_daily
                        else:
                            local_raw_future = torch.stack(local_future)
                            gathered_future = [
                                torch.empty_like(local_raw_future)
                                for _ in range(world_size)
                            ]
                            dist.all_gather(gathered_future, local_raw_future)
                            all_advantage = torch.cat(gathered_future)
                        group_advantage = normalize_advantages(
                            all_advantage, algorithm=args.algorithm,
                            epsilon=float(training["advantage_epsilon"]),
                            clip=float(training["advantage_clip"]),
                            mode=advantage_mode,
                            scale_floor=advantage_scale_floor,
                        )
                        within_date_advantage_std = all_advantage.std(
                            dim=0, unbiased=False,
                        ).mean()
                        between_date_advantage_std = all_advantage.mean(
                            dim=0,
                        ).std(unbiased=False)
                        if critic is not None:
                            critic_explained_variance = explained_variance(
                                torch.stack(old_values), torch.stack(local_returns),
                            )
                            critic_target_std = torch.stack(local_returns).std(unbiased=False)
                        else:
                            critic_explained_variance = local_daily.new_zeros(())
                            critic_target_std = local_daily.new_zeros(())
                        local_advantage = group_advantage[
                            rank * rollouts_per_rank:(rank + 1) * rollouts_per_rank
                        ].detach()
                        fixed_actions = [row.raw_action.detach() for row in rollouts]
                        old_log_probs = [row.log_prob[prefix:].detach() for row in rollouts]

                    update_batches = rollout_minibatches(
                        rollouts_per_rank, minibatch_rollouts_per_rank,
                        updates_per_rollout, seed=[seed, rank, epoch, block_number],
                    )
                    for policy_update, rollout_indices in enumerate(update_batches):
                        batch_indices = rollout_indices.tolist()
                        batch_advantage = local_advantage[batch_indices]
                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_losses, ratios, kls, old_policy_kls, clip_fractions = [], [], [], [], []
                        if rollout_execution == "cached_batched" and minibatch_rollouts_per_rank > 1:
                            replay = actor(
                                hidden, base, eligible, tradable,
                                actions=torch.stack([fixed_actions[row] for row in batch_indices]),
                                predicted_return=return_prediction,
                                static_cache=static_cache, rollouts=minibatch_rollouts_per_rank,
                                anchor_selected=anchor_selected,
                            )
                            log_ratio = (
                                replay.log_prob[:, prefix:] - torch.stack([old_log_probs[row] for row in batch_indices])
                            ).clamp(-10, 10)
                            ratio = torch.exp(log_ratio)
                            clip = float(training["policy_ratio_clip"])
                            objective = torch.minimum(
                                ratio * batch_advantage,
                                ratio.clamp(1 - clip, 1 + clip) * batch_advantage,
                            ).mean()
                            loss = (
                                -objective
                                + float(training["kl_coefficient"]) * replay.reference_kl.mean()
                                - float(training["entropy_coefficient"]) * replay.entropy.mean()
                                / (policy.action_count if config.get("official_trade_universe", False) else 1)
                            )
                            if not torch.isfinite(loss):
                                raise FloatingPointError("non-finite actor loss")
                            loss.backward()
                            actor_losses.append(loss.detach())
                            ratios.append(ratio.mean().detach())
                            kls.append(replay.reference_kl.mean().detach())
                            old_policy_kls.append((ratio - 1.0 - log_ratio).mean().detach())
                            clip_fractions.append(
                                ((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean().detach()
                            )
                        else:
                            for batch_position, rollout_index in enumerate(batch_indices):
                                context = actor.no_sync() if batch_position + 1 < minibatch_rollouts_per_rank else nullcontext()
                                with context:
                                    replay = actor(
                                        hidden, base, eligible, tradable,
                                        actions=fixed_actions[rollout_index],
                                        **({"observed_return": observed} if observed is not None else {}),
                                        **({"predicted_return": return_prediction}
                                           if return_prediction is not None else {}),
                                        **({"decision_factors": decision_factors}
                                           if decision_factors is not None else {}),
                                        **({"static_cache": static_cache}
                                           if static_cache is not None else {}),
                                    )
                                    log_ratio = (
                                        replay.log_prob[prefix:] - old_log_probs[rollout_index]
                                    ).clamp(-10, 10)
                                    ratio = torch.exp(log_ratio)
                                    clip = float(training["policy_ratio_clip"])
                                    objective = torch.minimum(
                                        ratio * local_advantage[rollout_index],
                                        ratio.clamp(1 - clip, 1 + clip)
                                        * local_advantage[rollout_index],
                                    ).mean()
                                    loss = (
                                        -objective
                                        + float(training["kl_coefficient"]) * replay.reference_kl
                                        - float(training["entropy_coefficient"])
                                        * replay.entropy
                                        / (policy.action_count if config.get("official_trade_universe", False) else 1)
                                    )
                                    if not torch.isfinite(loss):
                                        raise FloatingPointError("non-finite actor loss")
                                    (loss / minibatch_rollouts_per_rank).backward()
                                actor_losses.append(loss.detach())
                                ratios.append(ratio.mean().detach())
                                kls.append(replay.reference_kl.detach())
                                old_policy_kls.append((ratio - 1.0 - log_ratio).mean().detach())
                                clip_fractions.append(
                                    ((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean().detach()
                                )
                        actor_norm = torch.nn.utils.clip_grad_norm_(
                            actor.parameters(), float(training["gradient_clip"]),
                        )
                        global_update += 1
                        actor_lr = cosine_learning_rate(
                            float(training["learning_rate"]), update=global_update,
                            total_updates=total_updates, warmup_updates=warmup_updates,
                            eta_min_ratio=float(training["cosine_eta_min_ratio"]),
                        )
                        for group in actor_optimizer.param_groups:
                            group["lr"] = actor_lr
                        actor_optimizer.step()

                        critic_loss = torch.zeros((), device=device)
                        if critic_ddp is not None:
                            critic_optimizer.zero_grad(set_to_none=True)
                            value_losses = []
                            for batch_position, rollout_index in enumerate(batch_indices):
                                context = critic_ddp.no_sync() if batch_position + 1 < minibatch_rollouts_per_rank else nullcontext()
                                with context:
                                    estimate = critic_ddp(
                                        critic_states[rollout_index], **critic_detail_states[rollout_index],
                                    )
                                    value_loss = torch.nn.functional.mse_loss(
                                        estimate, local_returns[rollout_index],
                                    )
                                    (float(training["value_coefficient"])
                                     * value_loss / minibatch_rollouts_per_rank).backward()
                                value_losses.append(value_loss.detach())
                            critic_norm = torch.nn.utils.clip_grad_norm_(
                                critic_ddp.parameters(), float(training["gradient_clip"]),
                            )
                            critic_loss = torch.stack(value_losses).mean()
                            for group in critic_optimizer.param_groups:
                                group["lr"] = actor_lr * (
                                    float(training["critic_learning_rate"])
                                    / float(training["learning_rate"])
                                )
                            critic_optimizer.step()
                        else:
                            critic_norm = torch.zeros((), device=device)

                        local_metrics = torch.stack((
                            torch.stack(actor_losses).mean(),
                            critic_loss,
                            local_daily.mean(),
                            torch.stack([row.rank_ic.mean() for row in daily_rewards]).mean(),
                            torch.stack([row.annual_excess.mean() for row in daily_rewards]).mean(),
                            torch.stack([row.stability.mean() for row in daily_rewards]).mean(),
                            torch.stack(ratios).mean(),
                            torch.stack(kls).mean(),
                            local_advantage.std(unbiased=False),
                            torch.stack([row.swap_budget[prefix:].mean() for row in rollouts]).mean(),
                            torch.stack([row.realized_swaps[prefix:].mean() for row in rollouts]).mean(),
                            torch.as_tensor(actor_norm, device=device),
                            torch.as_tensor(critic_norm, device=device),
                            torch.stack(clip_fractions).mean(),
                            critic_explained_variance,
                            critic_target_std,
                            within_date_advantage_std,
                            between_date_advantage_std,
                            rollout_reward_std,
                            rollout_action_std[0],
                            rollout_action_std[1] if policy.action_count > 1
                            else rollout_action_std.new_zeros(()),
                            torch.stack(old_policy_kls).mean(),
                            group_action_values[..., 0].mean(),
                            group_action_values[..., 1].mean() if policy.action_count > 1
                            else group_action_values.new_zeros(()),
                        )).detach().double()
                        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
                        logged.append((local_metrics / world_size).cpu().numpy())
                        if len(logged) >= int(config["swanlab"]["log_interval_updates"]) or (
                            block_number == len(order) and policy_update + 1 == updates_per_collection
                        ):
                            if rank == 0:
                                mean = np.mean(logged, axis=0)
                                tracker.log({
                                    "train/actor_loss": mean[0],
                                    "train/critic_loss": mean[1],
                                    "train/daily_final_reward": mean[2],
                                    "train/daily_rank_ic": mean[3],
                                    "train/daily_annual_excess": mean[4],
                                    "train/daily_stability": mean[5],
                                    "train/policy_ratio": mean[6],
                                    "train/reference_kl": mean[7],
                                    "train/advantage_std": mean[8],
                                    "train/swap_budget_mean": mean[9],
                                    "train/realized_swaps_mean": mean[10],
                                    "train/actor_grad_norm": mean[11],
                                    "train/critic_grad_norm": mean[12],
                                    "train/policy_clip_fraction": mean[13],
                                    "train/critic_explained_variance": mean[14],
                                    "train/critic_target_std": mean[15],
                                    "train/advantage_within_date_std": mean[16],
                                    "train/advantage_between_date_std": mean[17],
                                    "train/rollout_reward_std_same_date": mean[18],
                                    "train/rollout_margin_std": mean[19],
                                    "train/rollout_budget_std": mean[20],
                                    "train/approx_kl_to_rollout_policy": mean[21],
                                    "train/rollout_margin_mean": mean[22],
                                    "train/rollout_budget_mean": mean[23],
                                    "train/actor_lr": actor_lr,
                                    "train/epoch": epoch,
                                    "train/optimizer_update": global_update,
                                    "train/collection_pass": policy_update // (rollouts_per_rank // minibatch_rollouts_per_rank) + 1,
                                    "train/minibatch_in_pass": policy_update % (rollouts_per_rank // minibatch_rollouts_per_rank) + 1,
                                }, step=global_update)
                            logged.clear()
                    del hidden, base, eligible, tradable, rollouts
                stopped_early = validate(epoch, tracker)
                if stopped_early:
                    break

            peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                best = next(row for row in history if row["epoch"] == best_epoch)
                summary = {
                    "algorithm": args.algorithm,
                    "checkpoint_selection": "last_completed_epoch",
                    "checkpoint_path": str(output_dir / "last"),
                    "validation_used_for_checkpoint_selection": False,
                    "last_epoch": best_epoch,
                    "last_score_components": best["policy_raw"],
                    "last_policy_diagnostics": best["diagnostics"],
                    "action_mode": action_mode,
                    "gamma": float(training["gamma"]),
                    "gae_lambda": float(training["gae_lambda"]) if critic is not None else None,
                    "critic_return_mode": critic_return_mode if critic is not None else None,
                    "critic_target_mode": critic_target_mode if critic is not None else None,
                    "critic_observation_mode": critic_observation_mode if critic is not None else None,
                    "advantage_normalization": advantage_normalization,
                    "history_days": history_days,
                    "observation_mode": observation_mode,
                    "policy_parameters": sum(p.numel() for p in policy.parameters()),
                    "policy_config": config["policy"],
                    "protocol": config["protocol"],
                    "epochs_trained": history[-1]["epoch"],
                    "stage1_raw": base_metrics["stage1_raw"],
                    "optimizer_updates": global_update,
                    "train_blocks": len(blocks),
                    "validation_days": len(validation_cache.date_indices),
                    "validation_burnin_days": validation_burnin,
                    "rollouts_per_group": world_size * rollouts_per_rank,
                    "minibatch_rollouts_global": world_size * minibatch_rollouts_per_rank,
                    "policy_passes_per_collection": updates_per_rollout,
                    "optimizer_updates_per_collection": updates_per_collection,
                    "rollout_execution": rollout_execution,
                    "same_state_branches": same_state_branches,
                    "policy_burnin_days": int(training["policy_burnin_days"]),
                    "cache": str(args.cache),
                    "backbone_sha256": checkpoint_hash,
                    "peak_memory_bytes_max_rank": int(peak.item()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "stopped_early": stopped_early,
                }
                atomic_json_dump(summary, output_dir / "train_summary.json")
                atomic_json_dump({"complete": True, "summary": summary}, output_dir / "complete.json")
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
