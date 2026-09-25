#!/usr/bin/env python3
"""Fine-tune C0 and the existing daily-group hysteresis policy together.

The reference experiment reads frozen C0 features from a cache.  Here the
scored dates are encoded online so the action log-probability can train C0.
The preceding burn-in dates stay cached and label-free; validation is fully
re-encoded with the current C0 at every epoch.
"""

from __future__ import annotations

import argparse
import json
import os
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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from finmodel.decision_cache import DecisionFeatureCache, cache_source_hashes
from finmodel.decision_rl import (
    DecisionValueHead, critic_observations, exact_daily_score,
    generalized_advantage,
)
from finmodel.io import atomic_json_dump, seed_everything
from finmodel.models import FinAxialDecisionPolicy, stock_vocab_sha256
from finmodel.panel import Panel
from finmodel.pipeline import build_dataset, load_backbone, score_numpy_predictions
from finmodel.sft import (
    cosine_learning_rate, job_name, load_config, panel_indices, reset_peak_memory,
    sft_split, swan_settings,
)
from scripts.train_decoupled_decision_rl import (
    NullTracker, evaluate_base, explained_variance, group_normalize_by_date,
    normalize_advantages, to_device,
)


class JointHysteresis(nn.Module):
    def __init__(self, backbone: nn.Module, policy: FinAxialDecisionPolicy,
                 *, fixed_base_rank: bool = False):
        super().__init__()
        self.backbone = backbone
        self.policy = policy
        self.fixed_base_rank = fixed_base_rank

    def encode_state(self, owner_items, owner_row_map, scored_eligible, scored_tradable,
                     scored_base_score,
                     prefix_hidden, prefix_score, prefix_eligible, prefix_tradable):
        encoded = []
        scores = []
        for values, token_valid, owner_eligible in owner_items:
            owner_hidden = self.backbone.encode_hidden(
                values, token_valid, owner_eligible,
            )
            encoded.append(owner_hidden)
            if not self.fixed_base_rank:
                scores.append(self.backbone.score_hidden(owner_hidden, owner_eligible))
        hidden = torch.stack([
            encoded[source][position] for source, position in owner_row_map
        ])
        score = (scored_base_score if self.fixed_base_rank else torch.stack([
            scores[source][position] for source, position in owner_row_map
        ]))
        return (
            torch.cat((prefix_hidden, hidden), dim=0),
            torch.cat((prefix_score, score), dim=0),
            torch.cat((prefix_eligible, scored_eligible), dim=0),
            torch.cat((prefix_tradable, scored_tradable), dim=0),
        )

    def forward(self, owner_items, owner_row_map, scored_eligible, scored_tradable,
                scored_base_score,
                prefix_hidden, prefix_score, prefix_eligible, prefix_tradable,
                *, actions=None, sample=False, rollout_count=1):
        all_hidden, all_score, all_eligible, all_tradable = self.encode_state(
            owner_items, owner_row_map, scored_eligible, scored_tradable,
            scored_base_score, prefix_hidden, prefix_score, prefix_eligible,
            prefix_tradable,
        )
        if actions is not None and len(actions) != rollout_count:
            raise ValueError("action count differs from rollout count")
        return tuple(
            self.policy(
                all_hidden, all_score, all_eligible, all_tradable,
                actions=(actions[index] if actions is not None else None),
                sample=sample,
            )
            for index in range(rollout_count)
        )


def owner_lookup(dataset, cache):
    """Reproduce the cache builder's latest-causal-position ownership."""
    owner_block = np.full(len(cache.date_indices), -1, dtype=np.int32)
    owner_position = np.full(len(cache.date_indices), -1, dtype=np.int16)
    for block, dates in enumerate(dataset.output_blocks):
        for position, row in enumerate(cache.rows_for_dates(dates)):
            if position > owner_position[row]:
                owner_block[row] = block
                owner_position[row] = position
    if (owner_block < 0).any():
        raise RuntimeError("training cache contains a date without an owner block")
    return owner_block, owner_position


def online_block(block_index, dataset, cache, panel, device, owner_block, owner_position):
    item = dataset[block_index]
    output_dates = item["date_indices"].numpy()
    rows = cache.rows_for_dates(output_dates)
    if not np.all(np.diff(rows) == 1):
        raise ValueError("scored dates must be consecutive")
    prefix_first = max(0, int(rows[0]) - 32)
    prefix_last = int(rows[0])
    prefix_dates = np.asarray(cache.date_indices[prefix_first:prefix_last], dtype=np.int64)
    official_tradable = ~np.asarray(panel.limit_flags[..., 0], dtype=bool)
    source_blocks = tuple(sorted(set(int(value) for value in owner_block[rows])))
    source_lookup = {block: source for source, block in enumerate(source_blocks)}
    owner_items = []
    for source_block in source_blocks:
        source_item = item if source_block == block_index else dataset[source_block]
        owner_items.append((
            source_item["x"].to(device), source_item["token_valid"].to(device),
            source_item["eligible"].to(device),
        ))
    row_map = tuple(
        (source_lookup[int(owner_block[row])], int(owner_position[row]))
        for row in rows
    )
    args = (
        owner_items, row_map,
        item["eligible"].to(device),
        to_device(official_tradable[output_dates], device),
        to_device(cache.base_score[rows], device),
        to_device(cache.hidden[prefix_first:prefix_last], device),
        to_device(cache.base_score[prefix_first:prefix_last], device),
        to_device(cache.eligible[prefix_first:prefix_last], device),
        to_device(official_tradable[prefix_dates], device),
    )
    target = item["target"].to(device)
    mask = to_device(panel.label_valid[output_dates], device)
    return args, target, mask, prefix_last - prefix_first


@torch.inference_mode()
def verify_owner_alignment(backbone, block_index, dataset, cache, panel, device,
                           owner_block, owner_position):
    inputs, _, _, _ = online_block(
        block_index, dataset, cache, panel, device, owner_block, owner_position,
    )
    owner_items, row_map = inputs[:2]
    hidden_parts = []
    score_parts = []
    for values, token_valid, eligible in owner_items:
        current_hidden = backbone.encode_hidden(values, token_valid, eligible)
        hidden_parts.append(current_hidden)
        score_parts.append(backbone.score_hidden(current_hidden, eligible))
    hidden = torch.stack([hidden_parts[source][position] for source, position in row_map])
    score = torch.stack([score_parts[source][position] for source, position in row_map])
    rows = cache.rows_for_dates(dataset.output_blocks[block_index])
    hidden_difference = (hidden - to_device(cache.hidden[rows], device)).abs().max()
    score_difference = (score - to_device(cache.base_score[rows], device)).abs().max()
    if float(hidden_difference) > 1e-4 or float(score_difference) > 1e-4:
        raise AssertionError(
            f"owner-aligned online C0 differs from cache: "
            f"hidden={float(hidden_difference):.6g}, score={float(score_difference):.6g}"
        )
    return float(hidden_difference), float(score_difference)


@torch.inference_mode()
def validate_online(backbone, policy, dataset, cache, panel, device,
                    *, fixed_base_rank=False):
    backbone.eval()
    policy.eval()
    requested = np.asarray(cache.date_indices, dtype=np.int64)
    if not np.array_equal(requested, dataset.requested_date_indices):
        raise ValueError("validation dataset and cache date axes differ")
    lookup = {int(date): offset for offset, date in enumerate(requested)}
    hidden = torch.empty(
        (len(requested), panel.shape[1], backbone.d_model), device=device,
    )
    score = torch.empty((len(requested), panel.shape[1]), device=device)
    owner = np.full(len(requested), -1, dtype=np.int16)
    for block in range(len(dataset)):
        item = dataset[block]
        eligible = item["eligible"].to(device)
        current_hidden = backbone.encode_hidden(
            item["x"].to(device), item["token_valid"].to(device), eligible,
        )
        current_score = backbone.score_hidden(current_hidden, eligible)
        for position, date in enumerate(item["date_indices"].tolist()):
            offset = lookup.get(int(date))
            if offset is None or position <= owner[offset]:
                continue
            hidden[offset] = current_hidden[position]
            score[offset] = current_score[position]
            owner[offset] = position
    if (owner < 0).any():
        raise RuntimeError("online validation did not encode every date")
    eligibility = to_device(cache.eligible, device)
    tradable = to_device(
        ~np.asarray(panel.limit_flags[requested, :, 0], dtype=bool), device,
    )
    decision_base_score = (
        to_device(cache.base_score, device) if fixed_base_rank else score
    )
    output = policy(hidden, decision_base_score, eligibility, tradable, sample=False)
    metrics, _ = score_numpy_predictions(
        panel=panel, indices=requested,
        predictions=output.decision_score.float().cpu().numpy(),
        eligible=np.asarray(cache.eligible, dtype=bool),
        route="stage2_joint_c0_hysteresis", ewma_alphas=(1.0,),
    )
    score_delta = (score - to_device(cache.base_score, device)).abs()
    diagnostics = {
        "alpha_mean": float(output.alpha.mean().cpu()),
        "swap_budget_mean": float(output.swap_budget.mean().cpu()),
        "hysteresis_margin_mean": float(output.action_value[:, 1].mean().cpu()),
        "c0_score_mean_absolute_change": float(score_delta.mean().cpu()),
        "c0_score_max_absolute_change": float(score_delta.max().cpu()),
    }
    del hidden, score, output
    torch.cuda.empty_cache()
    return metrics, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d0_decision_ppo_group.json")
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--cache", default="artifacts/d0/cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-train-blocks", type=int, default=0)
    parser.add_argument("--limit-validation-blocks", type=int, default=0)
    parser.add_argument("--disable-swanlab", action="store_true")
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("FINMODEL_DDP_TIMEOUT_SECONDS", "1800"))),
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    try:
        config = load_config(args.config)
        training = config["training"]
        if world_size != int(training["expected_world_size"]):
            raise ValueError("world size differs from the hysteresis reference")
        if config["policy"]["action_mode"] != "hysteresis":
            raise ValueError("this controlled experiment requires hysteresis actions")
        epochs = int(args.epochs or training["max_epochs"])
        if not 1 <= epochs <= int(training["max_epochs"]):
            raise ValueError("epochs outside configured range")
        seed = int(config["seed"])
        seed_everything(seed + rank)
        panel = Panel.open(args.panel)
        split = sft_split(panel, config)
        train = build_dataset(
            panel, panel_indices(panel, split.training_dates), config,
            stride=int(config["data"]["train_stride"]),
        )
        validation = build_dataset(
            panel, panel_indices(panel, split.validation_dates), config,
            stride=int(config["model"]["output_steps"]),
        )
        if args.limit_validation_blocks:
            validation.output_blocks = validation.output_blocks[:args.limit_validation_blocks]
            validation.requested_date_indices = np.unique(validation.output_blocks.reshape(-1))
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
        owner_block, owner_position = owner_lookup(train, train_cache)
        train_block_indices = np.arange(len(train), dtype=np.int32)
        if args.limit_train_blocks:
            train_block_indices = train_block_indices[-args.limit_train_blocks:]
        if args.limit_validation_blocks:
            from dataclasses import replace
            days = len(validation.requested_date_indices)
            validation_cache = replace(
                validation_cache,
                date_indices=validation_cache.date_indices[:days],
                hidden=validation_cache.hidden[:days],
                base_score=validation_cache.base_score[:days],
                eligible=validation_cache.eligible[:days],
                tradable=validation_cache.tradable[:days],
            )
        if not np.array_equal(
            validation.requested_date_indices, validation_cache.date_indices,
        ):
            raise ValueError("validation cache must cover the exact validation dates")
        backbone, loaded_hash = load_backbone(config, panel, device)
        if loaded_hash != checkpoint_hash:
            raise ValueError("C0 checkpoint hash changed")
        backbone_trainable = bool(training.get("backbone_trainable", True))
        fixed_base_rank = bool(training.get("fixed_base_rank", False))
        algorithm = str(training.get("algorithm", "daily_group"))
        if algorithm not in {"daily_group", "ppo_gae", "ppo_gae_group"}:
            raise ValueError(f"unsupported joint algorithm: {algorithm}")
        if fixed_base_rank and not backbone_trainable:
            raise ValueError("fixed_base_rank is only meaningful with trainable C0 hidden states")
        backbone.requires_grad_(backbone_trainable)
        if backbone_trainable and backbone.head_mode == "dual":
            # The proxy-mode decision policy consumes the rank score and
            # hidden states, never the independent absolute-return output.
            # Keep that unused head fixed while allowing RL gradients through
            # the shared encoder and ranking head.
            backbone.absolute_return_norm.requires_grad_(False)
            backbone.absolute_return_head.requires_grad_(False)
        if fixed_base_rank:
            backbone.output_norm.requires_grad_(False)
            backbone.return_head.requires_grad_(False)
        backbone.eval()  # deterministic C0 features; gradients remain enabled
        if rank == 0:
            hidden_error, score_error = verify_owner_alignment(
                backbone, int(train_block_indices[0]), train, train_cache,
                panel, device, owner_block, owner_position,
            )
            print(
                f"OWNER_ALIGNMENT hidden_max_error={hidden_error:.8f} "
                f"score_max_error={score_error:.8f}", flush=True,
            )
        # Loading C0 constructs modules and consumes Torch RNG. Reset before
        # initializing a fresh policy so its rank-0 weights match the frozen
        # four-card hysteresis experiment's epoch-0 policy exactly.
        seed_everything(seed + rank)
        policy = FinAxialDecisionPolicy(
            d_model=int(config["model"]["d_model"]), **config["policy"],
        ).to(device)
        initial_policy_checkpoint = config.get("initial_policy_checkpoint")
        if initial_policy_checkpoint is not None:
            state = torch.load(
                initial_policy_checkpoint, map_location="cpu", weights_only=True,
            )
            policy.load_state_dict(state, strict=True)
        joint = JointHysteresis(
            backbone, policy, fixed_base_rank=fixed_base_rank,
        ).to(device)
        actor = DDP(
            joint, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )
        optimizer = torch.optim.AdamW([
            {"params": backbone.parameters(), "lr": float(training["backbone_learning_rate"]),
             "initial_lr": float(training["backbone_learning_rate"])},
            {"params": policy.parameters(), "lr": float(training["learning_rate"]),
             "initial_lr": float(training["learning_rate"])},
        ], weight_decay=float(training["weight_decay"]))
        critic = None
        critic_ddp = None
        critic_optimizer = None
        if algorithm in {"ppo_gae", "ppo_gae_group"}:
            critic = DecisionValueHead(
                d_model=int(config["model"]["d_model"]),
                hidden_dim=int(training["critic_hidden_dim"]),
            ).to(device)
            critic_ddp = DDP(
                critic, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            )
            critic_optimizer = torch.optim.AdamW(
                critic_ddp.parameters(),
                lr=float(training["critic_learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
        output_dir = Path(args.output)
        if rank == 0:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"output is not empty: {output_dir}")
            output_dir.mkdir(parents=True)
        dist.barrier()
        reset_peak_memory(device)
        base_metrics = evaluate_base(validation_cache, panel) if rank == 0 else None
        if rank == 0:
            reference_path = Path(
                config.get(
                    "reference_validation_history",
                    "artifacts/d0/decision_grpo/validation_history.json",
                )
            )
            reference_history = json.loads(reference_path.read_text())
            reference_epoch = (
                0 if initial_policy_checkpoint is None
                else int(json.loads(Path(
                    initial_policy_checkpoint
                ).with_name("metadata.json").read_text())["epoch"])
            )
            reference = next(
                row["policy_raw"] for row in reference_history
                if row["epoch"] == reference_epoch
            )
        else:
            reference = None
        initialization = str(training.get(
            "route_name",
            ("unfrozen" if backbone_trainable else "frozen")
            + ("-random-policy" if initial_policy_checkpoint is None else "-warmstart"),
        ))
        run_name = job_name(
            "rl", f"finaxial-stage2-{algorithm}-joint-c0", initialization,
            int(config["model"]["lookback"]), seed,
            budget=f"k{config['model']['output_steps']}-g{world_size * int(training['rollouts_per_rank'])}-ep{epochs}-ddp{world_size}",
        )
        tracker_context = swan_settings(
            config, name=run_name,
            tags=["stage2", "hysteresis", "joint-c0", "daily-group",
                  algorithm, "fixed-base-rank" if fixed_base_rank else "live-base-rank",
                  "backbone-trainable" if backbone_trainable else "backbone-frozen"],
            extra={"backbone_sha256": checkpoint_hash,
                   "initial_policy_checkpoint": config["initial_policy_checkpoint"],
                   "backbone_trainable": backbone_trainable,
                   "fixed_base_rank": fixed_base_rank,
                   "algorithm": algorithm,
                   "backbone_lr": training["backbone_learning_rate"],
                   "policy_lr": training["learning_rate"], **training},
        ) if rank == 0 and not args.disable_swanlab else nullcontext(NullTracker())
        history = []
        best_stable = -float("inf")
        best_epoch = 0
        bad_epochs = 0
        global_update = 0
        nonzero_c0_grad_updates = 0
        updates_per_rollout = int(training["policy_updates_per_rollout"])
        rollouts_per_rank = int(training["rollouts_per_rank"])
        total_updates = len(train_block_indices) * epochs * updates_per_rollout
        warmup_updates = len(train_block_indices) * int(training["warmup_epochs"]) * updates_per_rollout
        started = time.perf_counter()

        def validate(epoch, tracker):
            nonlocal best_stable, best_epoch, bad_epochs
            dist.barrier()
            stop = torch.zeros((), dtype=torch.uint8, device=device)
            if rank == 0:
                metrics, diagnostic = validate_online(
                    backbone, policy, validation, validation_cache, panel, device,
                    fixed_base_rank=fixed_base_rank,
                )
                row = {"epoch": epoch, "optimizer_update": global_update,
                       "policy_raw": metrics, "diagnostics": diagnostic}
                history.append(row)
                window = history[-int(training["validation_smoothing_epochs"]):]
                stable = float(np.mean([entry["policy_raw"]["final_score"] for entry in window]))
                row["stable_selection_mean"] = stable
                atomic_json_dump(history, output_dir / "validation_history.json")
                tracker.log({
                    "validation/final_score": metrics["final_score"],
                    "validation/rank_ic": metrics["rank_ic"],
                    "validation/annual_excess": metrics["annual_excess"],
                    "validation/one_minus_turnover": metrics["one_minus_turnover"],
                    "validation/stable_final_mean": stable,
                    "validation/fixed_ewma_025/final_score": base_metrics["fixed_ewma_025"]["final_score"],
                    "validation/reference_hysteresis/final_score": reference["final_score"],
                    "validation/c0_score_mean_absolute_change": diagnostic["c0_score_mean_absolute_change"],
                    "validation/c0_score_max_absolute_change": diagnostic["c0_score_max_absolute_change"],
                    "validation/policy/alpha_mean": diagnostic["alpha_mean"],
                    "validation/policy/swap_budget_mean": diagnostic["swap_budget_mean"],
                    "validation/policy/hysteresis_margin_mean": diagnostic["hysteresis_margin_mean"],
                    "validation/epoch": epoch,
                }, step=global_update)
                if epoch == 0 and not args.limit_validation_blocks:
                    delta = abs(metrics["final_score"] - reference["final_score"])
                    # PPU SDPA can vary the exact annual-excess reduction by a
                    # few 1e-4 while Rank IC and holdings remain identical.
                    if delta > float(training.get("reference_score_tolerance", 5e-4)):
                        raise AssertionError(
                            f"joint initial score differs from frozen reference by {delta:.6g}"
                        )
                improved = (epoch == 0 or len(window) == int(training["validation_smoothing_epochs"])) and stable > best_stable
                if improved:
                    best_stable, best_epoch, bad_epochs = stable, epoch, 0
                    best_dir = output_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    torch.save(backbone.state_dict(), best_dir / "model.pt")
                    torch.save(policy.state_dict(), best_dir / "policy.pt")
                    if critic is not None:
                        torch.save(critic.state_dict(), best_dir / "critic.pt")
                    atomic_json_dump({
                        "epoch": epoch, "optimizer_update": global_update,
                        "score_components": metrics, "stable_validation_final_score": stable,
                        "backbone_initial_sha256": checkpoint_hash,
                        "stock_vocab_sha256": stock_vocab_sha256(panel.codes),
                        "initial_policy_checkpoint": config["initial_policy_checkpoint"],
                        "algorithm": algorithm,
                        "fixed_base_rank": fixed_base_rank,
                        "policy_config": config["policy"],
                    }, best_dir / "metadata.json")
                elif epoch > 0:
                    bad_epochs += 1
                print(
                    f"VALIDATION epoch={epoch} final={metrics['final_score']:.6f} "
                    f"ic={metrics['rank_ic']:.6f} excess={metrics['annual_excess']:.6f} "
                    f"stability={metrics['one_minus_turnover']:.6f} "
                    f"c0_delta={diagnostic['c0_score_mean_absolute_change']:.6f}",
                    flush=True,
                )
                if epoch >= int(training["minimum_epochs"]) and bad_epochs >= int(training["early_stopping_patience"]):
                    stop.fill_(1)
            dist.broadcast(stop, src=0)
            dist.barrier()
            return bool(stop.item())

        with tracker_context as tracker:
            validate(0, tracker)
            stopped_early = False
            for epoch in range(1, epochs + 1):
                order = np.random.default_rng(seed + epoch).permutation(train_block_indices)
                policy.train()
                if critic is not None:
                    critic.train()
                backbone.eval()
                for block_number, block_index in enumerate(order, start=1):
                    inputs, target, label_mask, prefix = online_block(
                        int(block_index), train, train_cache, panel, device,
                        owner_block, owner_position,
                    )
                    with torch.no_grad():
                        state = joint.encode_state(*inputs)
                        rollouts = tuple(
                            policy(*state, sample=True)
                            for _ in range(rollouts_per_rank)
                        )
                        rewards = []
                        critic_states = []
                        old_values = []
                        local_future = []
                        local_returns = []
                        for sampled in rollouts:
                            prior = sampled.selected[prefix - 1] if prefix else None
                            reward = exact_daily_score(
                                sampled.decision_score[prefix:], target,
                                label_mask, inputs[3], previous_selected=prior,
                                top_fraction=float(config["policy"]["top_fraction"]),
                            )
                            rewards.append(reward)
                            if critic is not None:
                                observation = critic_observations(
                                    *state, sampled.selected,
                                    sampled.decision_score,
                                    return_scale=float(config["policy"]["return_scale"]),
                                    top_fraction=float(config["policy"]["top_fraction"]),
                                )[prefix:]
                                value = critic(observation)
                                future, returns = generalized_advantage(
                                    reward.reward, value,
                                    gamma=float(training["gamma"]),
                                    lam=float(training["gae_lambda"]),
                                )
                                critic_states.append(observation.detach())
                                old_values.append(value.detach())
                                local_future.append(future.detach())
                                local_returns.append(returns.detach())
                        local_daily = torch.stack([reward.reward for reward in rewards])
                        raw_advantage = (
                            torch.stack(local_future) if critic is not None
                            else local_daily
                        )
                        gathered = [torch.empty_like(raw_advantage) for _ in range(world_size)]
                        dist.all_gather(gathered, raw_advantage)
                        advantages = normalize_advantages(
                            torch.cat(gathered), algorithm=algorithm,
                            epsilon=float(training["advantage_epsilon"]),
                            clip=float(training["advantage_clip"]),
                        )[rank * rollouts_per_rank:(rank + 1) * rollouts_per_rank].detach()
                        critic_ev = (
                            explained_variance(
                                torch.stack(old_values), torch.stack(local_returns),
                            ) if critic is not None else local_daily.new_zeros(())
                        )
                        actions = [row.raw_action.detach() for row in rollouts]
                        old_log_prob = [row.log_prob[prefix:].detach() for row in rollouts]
                    for update in range(updates_per_rollout):
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        loss_tensors = []
                        ratios = []
                        clip_fractions = []
                        replays = actor(
                            *inputs, actions=actions, rollout_count=rollouts_per_rank,
                        )
                        for rollout_index, replay in enumerate(replays):
                            ratio = torch.exp((
                                replay.log_prob[prefix:] - old_log_prob[rollout_index]
                            ).clamp(-10, 10))
                            clip = float(training["policy_ratio_clip"])
                            advantage = advantages[rollout_index]
                            objective = torch.minimum(
                                ratio * advantage,
                                ratio.clamp(1 - clip, 1 + clip) * advantage,
                            ).mean()
                            loss = (
                                -objective
                                + float(training["kl_coefficient"]) * replay.reference_kl
                                - float(training["entropy_coefficient"])
                                * replay.entropy / policy.action_count
                            )
                            if not torch.isfinite(loss):
                                raise FloatingPointError("non-finite joint actor loss")
                            losses.append(loss.detach())
                            loss_tensors.append(loss)
                            ratios.append(ratio.mean().detach())
                            clip_fractions.append(
                                ((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean().detach()
                            )
                        torch.stack(loss_tensors).mean().backward()
                        backbone_norm = torch.nn.utils.clip_grad_norm_(
                            backbone.parameters(), float(training["backbone_gradient_clip"]),
                        )
                        policy_norm = torch.nn.utils.clip_grad_norm_(
                            policy.parameters(), float(training["gradient_clip"]),
                        )
                        if not bool(torch.isfinite(backbone_norm)):
                            raise FloatingPointError("C0 gradient is non-finite")
                        if bool(backbone_norm > 0):
                            nonzero_c0_grad_updates += 1
                        if global_update == 0 and rank == 0:
                            print(
                                f"FIRST_JOINT_UPDATE c0_grad_norm={float(backbone_norm):.6f} "
                                f"policy_grad_norm={float(policy_norm):.6f}",
                                flush=True,
                            )
                        global_update += 1
                        ratio = cosine_learning_rate(
                            1.0, update=global_update, total_updates=total_updates,
                            warmup_updates=warmup_updates,
                            eta_min_ratio=float(training["cosine_eta_min_ratio"]),
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = group["initial_lr"] * ratio
                        optimizer.step()
                        critic_loss = torch.zeros((), device=device)
                        critic_norm = torch.zeros((), device=device)
                        if critic_ddp is not None:
                            critic_optimizer.zero_grad(set_to_none=True)
                            value_losses = []
                            for rollout_index in range(rollouts_per_rank):
                                context = (
                                    critic_ddp.no_sync()
                                    if rollout_index + 1 < rollouts_per_rank
                                    else nullcontext()
                                )
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
                            if not bool(torch.isfinite(critic_norm)):
                                raise FloatingPointError("critic gradient is non-finite")
                            critic_loss = torch.stack(value_losses).mean()
                            for group in critic_optimizer.param_groups:
                                group["lr"] = float(training["critic_learning_rate"]) * ratio
                            critic_optimizer.step()
                        metrics = torch.stack((
                            torch.stack(losses).mean(), local_daily.mean(),
                            torch.stack([reward.rank_ic.mean() for reward in rewards]).mean(),
                            torch.stack([reward.annual_excess.mean() for reward in rewards]).mean(),
                            torch.stack([reward.stability.mean() for reward in rewards]).mean(),
                            torch.as_tensor(backbone_norm, device=device),
                            torch.as_tensor(policy_norm, device=device),
                            critic_loss, torch.as_tensor(critic_norm, device=device),
                            torch.stack(ratios).mean(),
                            torch.stack(clip_fractions).mean(), critic_ev,
                        )).detach().double()
                        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                        if rank == 0 and (
                            global_update % int(config["swanlab"]["log_interval_updates"]) == 0
                            or (block_number == len(order) and update + 1 == updates_per_rollout)
                        ):
                            row = (metrics / world_size).cpu().tolist()
                            tracker.log({
                                "train/actor_loss": row[0], "train/daily_final_reward": row[1],
                                "train/daily_rank_ic": row[2], "train/daily_annual_excess": row[3],
                                "train/daily_stability": row[4], "train/c0_grad_norm": row[5],
                                "train/policy_grad_norm": row[6],
                                "train/critic_loss": row[7],
                                "train/critic_grad_norm": row[8],
                                "train/policy_ratio": row[9],
                                "train/clip_fraction": row[10],
                                "train/critic_explained_variance": row[11],
                                "train/c0_lr": optimizer.param_groups[0]["lr"],
                                "train/policy_lr": optimizer.param_groups[1]["lr"],
                                "train/critic_lr": (
                                    critic_optimizer.param_groups[0]["lr"]
                                    if critic_optimizer is not None else 0.0
                                ),
                                "train/optimizer_update": global_update, "train/epoch": epoch,
                            }, step=global_update)
                    del rollouts, inputs, state
                if backbone_trainable and nonzero_c0_grad_updates == 0:
                    raise RuntimeError(
                        "C0 never received a gradient; check the action head and policy inputs"
                    )
                stopped_early = validate(epoch, tracker)
                if stopped_early:
                    break
            peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                best = next(row for row in history if row["epoch"] == best_epoch)
                summary = {
                    "protocol": config["protocol"], "epochs_trained": history[-1]["epoch"],
                    "best_epoch": best_epoch, "best_stable_final_score": best_stable,
                    "best_score_components": best["policy_raw"],
                    "reference_hysteresis": reference,
                    "optimizer_updates": global_update, "train_blocks": len(train_block_indices),
                    "validation_days": len(validation_cache.date_indices),
                    "backbone_initial_sha256": checkpoint_hash,
                    "initial_policy_checkpoint": config["initial_policy_checkpoint"],
                    "backbone_trainable": backbone_trainable,
                    "fixed_base_rank": fixed_base_rank,
                    "algorithm": algorithm,
                    "nonzero_c0_grad_updates": nonzero_c0_grad_updates,
                    "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
                    "policy_parameters": sum(p.numel() for p in policy.parameters()),
                    "critic_parameters": sum(p.numel() for p in critic.parameters()) if critic else 0,
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
