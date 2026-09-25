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
from finmodel.decision_history import (
    causal_c0_bridge_batch, observed_market_features, observed_return_features,
)
from finmodel.decision_rl import (
    DecisionValueHead,
    critic_observations,
    discounted_return_to_go,
    exact_daily_score,
    generalized_advantage,
)
from finmodel.io import atomic_json_dump, seed_everything
from finmodel.models import FinAxialDecisionPolicy, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.pipeline import build_dataset, load_backbone, score_numpy_predictions
from finmodel.sft import (
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
                 return_scale=0.02):
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
    return_prediction = (
        to_device(
            cache.predicted_return[first:last]
            if cache.predicted_return is not None
            else np.asarray(cache.base_score[first:last], dtype=np.float32) * float(return_scale),
            device,
        ) if return_feature_mode == "explicit" else None
    )
    return hidden, base, eligible, tradable, target, label_mask, scored_start, observed, return_prediction


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
                    official_trade_universe=False, history_prefix=None):
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
    return_kwargs = (
        {"predicted_return": to_device(
            cache.predicted_return
            if cache.predicted_return is not None
            else np.asarray(cache.base_score, dtype=np.float32) * policy.return_scale,
            device,
        )} if policy.return_feature_mode == "explicit" else {}
    )
    output = policy(
        hidden, base, eligible, tradable, sample=False,
        **history_kwargs, **return_kwargs,
    )
    policy_raw, _ = score_numpy_predictions(
        panel=panel,
        indices=np.asarray(cache.date_indices),
        predictions=output.decision_score.float().cpu().numpy(),
        eligible=eligible.cpu().numpy(),
        route=route,
        ewma_alphas=(1.0,),
    )
    diagnostics = {
        "alpha_mean": float(output.alpha.mean().cpu()),
        "alpha_std": float(output.alpha.std().cpu()),
        "delta_mean": float(output.delta.mean().cpu()),
        "delta_std": float(output.delta.std().cpu()),
        "swap_budget_mean": float(output.swap_budget.mean().cpu()),
        "swap_budget_std": float(output.swap_budget.std().cpu()),
        "realized_swaps_mean": float(output.realized_swaps.mean().cpu()),
        "realized_swaps_std": float(output.realized_swaps.std().cpu()),
        "candidate_count_mean": float(output.candidate_count.mean().cpu()),
        "candidate_bonus_abs_mean": float(output.candidate_bonus_abs_mean.mean().cpu()),
        "action_value_mean": output.action_value.float().mean(dim=0).cpu().tolist(),
        "action_value_std": output.action_value.float().std(dim=0).cpu().tolist(),
        "policy_std": output.policy_std.float().cpu().tolist(),
        "reference_kl": float(output.reference_kl.cpu()),
    }
    if policy.action_mode in {"hysteresis", "candidate_residual"}:
        diagnostics["hysteresis_margin_mean"] = float(output.action_value[:, 1].mean().cpu())
        diagnostics["hysteresis_margin_std"] = float(output.action_value[:, 1].std(unbiased=False).cpu())
    del hidden, base, eligible, tradable, output
    torch.cuda.empty_cache()
    return {"policy_raw": policy_raw, "diagnostics": diagnostics, **base_metrics}


@torch.inference_mode()
def evaluate_base(cache, panel):
    base, ewma = score_numpy_predictions(
        panel=panel,
        indices=np.asarray(cache.date_indices),
        predictions=np.asarray(cache.base_score, dtype=np.float32),
        eligible=np.asarray(cache.eligible, dtype=bool),
        route="stage2_daily_cached_c0",
        ewma_alphas=(1.0, 0.25),
    )
    return {"stage1_raw": base, "fixed_ewma_025": ewma["alpha_0.25"]}


def group_normalize_by_date(values, epsilon, clip):
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("daily advantages require [rollouts >= 2, dates]")
    normalized = (values - values.mean(dim=0, keepdim=True)) / (
        values.std(dim=0, unbiased=False, keepdim=True).clamp_min(epsilon)
    )
    return normalized.clamp(-clip, clip)


def normalize_advantages(values, *, algorithm, epsilon, clip):
    """Keep the legacy PPO normalization while enabling matched-date controls."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d0_decision_grpo.json")
    parser.add_argument("--algorithm", required=True, choices=(
        "daily_group", "ppo_gae", "ppo_gae_group", "rtg_group",
    ))
    parser.add_argument("--action-mode", choices=(
        "uniform2", "swap_budget_only", "swap_budget_raw", "swap_budget_alpha", "hysteresis", "boundary4",
        "candidate_residual",
    ))
    parser.add_argument("--observation-mode", choices=("mean_pool", "candidate_attention"))
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--cache", default="artifacts/d0/cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gamma", type=float, help="Override the configured reward discount")
    parser.add_argument("--gae-lambda", type=float, help="Override the configured GAE lambda")
    parser.add_argument("--run-suffix", default="", help="Unique SwanLab name suffix for an ablation")
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
    action_mode = str(config["policy"]["action_mode"])
    training = config["training"]
    if args.gamma is not None:
        training["gamma"] = args.gamma
    if args.gae_lambda is not None:
        training["gae_lambda"] = args.gae_lambda
    if not 0 <= float(training["gamma"]) <= 1 or not 0 <= float(training["gae_lambda"]) <= 1:
        raise ValueError("gamma and GAE lambda must be in [0, 1]")
    if args.run_suffix and not re.fullmatch(r"[a-z0-9_-]+", args.run_suffix):
        raise ValueError("run suffix must contain only lowercase letters, digits, - or _")
    if world_size != int(training["expected_world_size"]):
        raise ValueError(f"expected {training['expected_world_size']} ranks, found {world_size}")
    if args.algorithm == "ppo_gae" and action_mode not in {"uniform2", "hysteresis"}:
        raise ValueError("PPO is validated only for uniform2 and hysteresis actions")
    if args.algorithm in {"ppo_gae_group", "rtg_group"} and action_mode != "hysteresis":
        raise ValueError("new long-credit ablations are validated only for hysteresis actions")
    if history_days and (args.algorithm != "daily_group" or action_mode not in {"hysteresis", "candidate_residual"}):
        raise ValueError("history-aware ablations use daily_group and hysteresis-family actions")
    if observation_mode == "candidate_attention" and history_days < 2:
        raise ValueError("candidate attention requires history_days >= 2")
    epochs = int(args.epochs or training["max_epochs"])
    if not 1 <= epochs <= int(training["max_epochs"]):
        raise ValueError("epochs exceed configured range")
    rollouts_per_rank = int(training["rollouts_per_rank"])
    if rollouts_per_rank < 1 or world_size * rollouts_per_rank < 2:
        raise ValueError("at least two same-block rollouts are required")
    seed = int(config["seed"])
    seed_everything(seed + rank)
    panel = Panel.open(args.panel)
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

    policy = FinAxialDecisionPolicy(
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
        critic = DecisionValueHead(
            d_model=int(config["model"]["d_model"]),
            hidden_dim=int(training["critic_hidden_dim"]),
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
    dist.barrier()
    reset_peak_memory(device)
    updates_per_rollout = int(training["policy_updates_per_rollout"])
    total_updates = len(blocks) * epochs * updates_per_rollout
    warmup_updates = len(blocks) * int(training["warmup_epochs"]) * updates_per_rollout
    run_action = action_mode if args.algorithm == "daily_group" else f"{action_mode}-{args.algorithm}"
    if history_days:
        run_action = f"{run_action}-history{history_days}"
        if observation_mode != "mean_pool":
            run_action = f"{run_action}-{observation_mode}"
    if args.run_suffix:
        run_action = f"{run_action}-{args.run_suffix}"
    advantage_normalization = (
        "global_rollout_date" if args.algorithm == "ppo_gae" else "same_date_group"
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
            "train_blocks": len(blocks),
            "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic else 0,
            "advantage_normalization": advantage_normalization,
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
                policy, validation_cache, panel, device,
                route=f"stage2_daily_{args.algorithm}_{action_mode}", base_metrics=base_metrics,
                official_trade_universe=bool(config.get("official_trade_universe", False)),
                history_prefix=history_prefix,
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
                "validation/fixed_ewma_025/final_score": base_metrics["fixed_ewma_025"]["final_score"],
                "validation/stable_final_mean": stable,
                "validation/policy/alpha_mean": diagnostics["alpha_mean"],
                "validation/policy/alpha_std": diagnostics["alpha_std"],
                "validation/policy/delta_mean": diagnostics["delta_mean"],
                "validation/policy/delta_std": diagnostics["delta_std"],
                "validation/policy/swap_budget_mean": diagnostics["swap_budget_mean"],
                "validation/policy/swap_budget_std": diagnostics["swap_budget_std"],
                "validation/policy/realized_swaps_mean": diagnostics["realized_swaps_mean"],
                "validation/policy/realized_swaps_std": diagnostics["realized_swaps_std"],
                "validation/policy/candidate_count_mean": diagnostics["candidate_count_mean"],
                "validation/policy/candidate_bonus_abs_mean": diagnostics["candidate_bonus_abs_mean"],
                **({
                    "validation/policy/hysteresis_margin_mean": diagnostics["hysteresis_margin_mean"],
                    "validation/policy/hysteresis_margin_std": diagnostics["hysteresis_margin_std"],
                } if action_mode in {"hysteresis", "candidate_residual"} else {}),
                "validation/policy/reference_kl": diagnostics["reference_kl"],
                "validation/epoch": epoch,
                "validation/optimizer_update": global_update,
            }, step=global_update)
            improved = (epoch == 0 or len(window) == int(training["validation_smoothing_epochs"])) and stable > best_stable
            if improved:
                best_stable, best_epoch, bad_epochs = stable, epoch, 0
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                torch.save(policy.state_dict(), best_dir / "policy.pt")
                if critic is not None:
                    torch.save(critic.state_dict(), best_dir / "critic.pt")
                atomic_json_dump({
                    "algorithm": args.algorithm,
                    "action_mode": action_mode,
                    "gamma": float(training["gamma"]),
                    "gae_lambda": float(training["gae_lambda"]) if critic is not None else None,
                    "advantage_normalization": advantage_normalization,
                    "history_days": history_days,
                    "observation_mode": observation_mode,
                    "policy_parameters": sum(p.numel() for p in policy.parameters()),
                    "policy_config": config["policy"],
                    "epoch": epoch,
                    "optimizer_update": global_update,
                    "stable_validation_final_score": stable,
                    "score_components": metrics,
                    "fixed_ewma_025": base_metrics["fixed_ewma_025"],
                    "policy_diagnostics": diagnostics,
                    "backbone_sha256": checkpoint_hash,
                    "stock_vocab_sha256": stock_vocab_sha256(panel.codes),
                    "cache": str(args.cache),
                }, best_dir / "metadata.json")
            elif epoch > 0:
                bad_epochs += 1
            print(
                f"VALIDATION {args.algorithm} epoch={epoch} final={metrics['final_score']:.6f} "
                f"ic={metrics['rank_ic']:.6f} excess={metrics['annual_excess']:.6f} "
                f"stability={metrics['one_minus_turnover']:.6f} "
                f"alpha_std={diagnostics['alpha_std']:.6f}", flush=True,
            )
        stop = torch.zeros((), dtype=torch.uint8, device=device)
        if rank == 0 and epoch >= int(training["minimum_epochs"]):
            stop.fill_(int(bad_epochs >= int(training["early_stopping_patience"])))
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
                    hidden, base, eligible, tradable, target, label_mask, prefix, observed, return_prediction = cached_block(
                        train_cache, panel, blocks[int(block_index)],
                        int(training["policy_burnin_days"]), device,
                        official_trade_universe=bool(config.get("official_trade_universe", False)),
                        history_days=history_days,
                        observation_mode=observation_mode,
                        return_feature_mode=policy.return_feature_mode,
                        return_scale=policy.return_scale,
                    )
                    with torch.no_grad():
                        rollouts, daily_rewards, critic_states = [], [], []
                        old_values, local_future, local_returns = [], [], []
                        for _ in range(rollouts_per_rank):
                            output = policy(
                                hidden, base, eligible, tradable, sample=True,
                                **({"observed_return": observed} if observed is not None else {}),
                                **({"predicted_return": return_prediction}
                                   if return_prediction is not None else {}),
                            )
                            prior = output.selected[prefix - 1] if prefix else None
                            reward = exact_daily_score(
                                output.decision_score[prefix:], target,
                                label_mask, tradable[prefix:],
                                previous_selected=prior,
                                top_fraction=float(config["policy"]["top_fraction"]),
                            )
                            rollouts.append(output)
                            daily_rewards.append(reward)
                            if critic is not None:
                                obs = critic_observations(
                                    hidden, base, eligible, tradable,
                                    output.selected, output.decision_score,
                                    return_scale=float(config["policy"]["return_scale"]),
                                    top_fraction=float(config["policy"]["top_fraction"]),
                                )[prefix:]
                                value = critic(obs)
                                gae, returns = generalized_advantage(
                                    reward.reward, value,
                                    gamma=float(training["gamma"]),
                                    lam=float(training["gae_lambda"]),
                                )
                                critic_states.append(obs.detach())
                                old_values.append(value.detach())
                                local_future.append(gae.detach())
                                local_returns.append(returns.detach())
                            elif args.algorithm == "rtg_group":
                                local_future.append(discounted_return_to_go(
                                    reward.reward, gamma=float(training["gamma"]),
                                ).detach())
                        local_daily = torch.stack([row.reward for row in daily_rewards])
                        grouped = [torch.empty_like(local_daily) for _ in range(world_size)]
                        dist.all_gather(grouped, local_daily)
                        group_daily = torch.cat(grouped)
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

                    for policy_update in range(updates_per_rollout):
                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_losses, ratios, kls, clip_fractions = [], [], [], []
                        for rollout_index in range(rollouts_per_rank):
                            context = actor.no_sync() if rollout_index + 1 < rollouts_per_rank else nullcontext()
                            with context:
                                replay = actor(
                                    hidden, base, eligible, tradable,
                                    actions=fixed_actions[rollout_index],
                                    **({"observed_return": observed} if observed is not None else {}),
                                    **({"predicted_return": return_prediction}
                                       if return_prediction is not None else {}),
                                )
                                ratio = torch.exp((
                                    replay.log_prob[prefix:] - old_log_probs[rollout_index]
                                ).clamp(-10, 10))
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
                                (loss / rollouts_per_rank).backward()
                            actor_losses.append(loss.detach())
                            ratios.append(ratio.mean().detach())
                            kls.append(replay.reference_kl.detach())
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
                            for rollout_index in range(rollouts_per_rank):
                                context = critic_ddp.no_sync() if rollout_index + 1 < rollouts_per_rank else nullcontext()
                                with context:
                                    estimate = critic_ddp(critic_states[rollout_index])
                                    value_loss = torch.nn.functional.mse_loss(
                                        estimate, local_returns[rollout_index],
                                    )
                                    (float(training["value_coefficient"])
                                     * value_loss / rollouts_per_rank).backward()
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
                            torch.stack([row.alpha[prefix:].mean() for row in rollouts]).mean(),
                            torch.stack([row.delta[prefix:].mean() for row in rollouts]).mean(),
                            torch.stack([row.swap_budget[prefix:].mean() for row in rollouts]).mean(),
                            torch.stack([row.realized_swaps[prefix:].mean() for row in rollouts]).mean(),
                            torch.as_tensor(actor_norm, device=device),
                            torch.as_tensor(critic_norm, device=device),
                            torch.stack(clip_fractions).mean(),
                            critic_explained_variance,
                            critic_target_std,
                            within_date_advantage_std,
                            between_date_advantage_std,
                        )).detach().double()
                        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
                        logged.append((local_metrics / world_size).cpu().numpy())
                        if len(logged) >= int(config["swanlab"]["log_interval_updates"]) or (
                            block_number == len(order) and policy_update + 1 == updates_per_rollout
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
                                    "train/alpha_mean": mean[9],
                                    "train/delta_mean": mean[10],
                                    "train/swap_budget_mean": mean[11],
                                    "train/realized_swaps_mean": mean[12],
                                    "train/actor_grad_norm": mean[13],
                                    "train/critic_grad_norm": mean[14],
                                    "train/policy_clip_fraction": mean[15],
                                    "train/critic_explained_variance": mean[16],
                                    "train/critic_target_std": mean[17],
                                    "train/advantage_within_date_std": mean[18],
                                    "train/advantage_between_date_std": mean[19],
                                    "train/actor_lr": actor_lr,
                                    "train/epoch": epoch,
                                    "train/optimizer_update": global_update,
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
                    "action_mode": action_mode,
                    "gamma": float(training["gamma"]),
                    "gae_lambda": float(training["gae_lambda"]) if critic is not None else None,
                    "advantage_normalization": advantage_normalization,
                    "history_days": history_days,
                    "observation_mode": observation_mode,
                    "policy_parameters": sum(p.numel() for p in policy.parameters()),
                    "policy_config": config["policy"],
                    "protocol": config["protocol"],
                    "epochs_trained": history[-1]["epoch"],
                    "best_epoch": best_epoch,
                    "best_stable_final_score": best_stable,
                    "best_score_components": best["policy_raw"],
                    "fixed_ewma_025": base_metrics["fixed_ewma_025"],
                    "stage1_raw": base_metrics["stage1_raw"],
                    "best_policy_diagnostics": best["diagnostics"],
                    "optimizer_updates": global_update,
                    "train_blocks": len(blocks),
                    "validation_days": len(validation_cache.date_indices),
                    "rollouts_per_group": world_size * rollouts_per_rank,
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
