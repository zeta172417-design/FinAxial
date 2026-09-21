#!/usr/bin/env python3
"""Two-device NCCL/PCCL collective and 20-step DDP smoke test."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from finmodel.io import atomic_json_dump, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="artifacts/reports/distributed_smoke.json")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(2026 + rank)

    collective = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(collective)
    expected = world * (world + 1) / 2
    if float(collective) != expected:
        raise AssertionError(f"collective mismatch: {float(collective)} != {expected}")

    model = DistributedDataParallel(torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 1),
    ).to(device), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(args.steps):
        values = torch.randn(128, 32, device=device)
        target = torch.randn(128, 1, device=device)
        loss = torch.nn.functional.mse_loss(model(values), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite distributed smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    dist.barrier()
    if rank == 0:
        atomic_json_dump({
            "backend": dist.get_backend(), "world_size": world, "steps": args.steps,
            "collective": float(collective), "loss_first": losses[0], "loss_last": losses[-1],
        }, args.output)
        print(json.dumps({"world_size": world, "steps": args.steps, "collective": float(collective)}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
