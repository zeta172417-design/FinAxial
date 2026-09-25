#!/usr/bin/env python3
"""Evaluate a saved D0 decision-maker on the labelled validation cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from finmodel.decision_cache import DecisionFeatureCache, cache_source_hashes
from finmodel.io import atomic_json_dump
from finmodel.models import FinAxialDecisionPolicy
from finmodel.panel import Panel
from finmodel.sft import load_config
from scripts.train_decoupled_decision_rl import evaluate_base, evaluate_cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("grpo", "ppo_group"), default="grpo")
    parser.add_argument("--panel", default="artifacts/panel/phase1")
    parser.add_argument("--cache", default="artifacts/d0/cache/validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Optional JSON result path")
    args = parser.parse_args()

    config_path = (
        "configs/d0_decision_grpo.json" if args.route == "grpo"
        else "configs/d0_decision_ppo_group.json"
    )
    checkpoint = Path("artifacts/d0") / (
        "decision_grpo" if args.route == "grpo" else "decision_ppo_group"
    ) / "best/policy.pt"
    config = load_config(config_path)
    panel = Panel.open(args.panel)
    checkpoint_hash, panel_hash = cache_source_hashes(
        panel, config["backbone_checkpoint"],
    )
    cache = DecisionFeatureCache.open(
        args.cache, checkpoint_sha256=checkpoint_hash,
        panel_manifest_sha256=panel_hash,
    )
    device = torch.device(args.device)
    policy = FinAxialDecisionPolicy(
        d_model=int(config["model"]["d_model"]), **config["policy"],
    ).to(device)
    policy.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True,
    )
    metrics = evaluate_cached(
        policy, cache, panel, device, route=f"d0_{args.route}",
        base_metrics=evaluate_base(cache, panel),
        official_trade_universe=bool(config.get("official_trade_universe", False)),
    )
    result = {
        "route": args.route,
        "checkpoint": str(checkpoint),
        "validation_days": len(cache.date_indices),
        "metrics": metrics["policy_raw"],
        "diagnostics": metrics["diagnostics"],
    }
    if args.output:
        atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
