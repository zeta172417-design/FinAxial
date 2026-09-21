#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python"
PANEL_PATH="artifacts/panel/phase1"
CONFIG_PATH="configs/stock_time_transformer.json"

cd "$PROJECT_ROOT"
source /usr/local/PPU_SDK/envsetup.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=ERROR
export PCCL_DEBUG=ERROR
export FINMODEL_DDP_TIMEOUT_SECONDS=300

test -f "$PANEL_PATH/manifest.json"

echo "Running one-PPU B0 smoke"
(
  export CUDA_VISIBLE_DEVICES=0
  "$PYTHON_BIN" -m torch.distributed.run --master-port=29710 --nproc-per-node=1 \
    scripts/train_stock_time_transformer.py \
    --config "$CONFIG_PATH" --panel "$PANEL_PATH" \
    --output /tmp/stock_time_transformer_b0_smoke \
    --epochs 1 --limit-train-blocks 1 --limit-validation-blocks 1 \
    --disable-swanlab
)

echo "Smoke passed; starting the reproducible eight-PPU run"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$PYTHON_BIN" -c 'import torch; count=torch.cuda.device_count(); assert count == 8, f"expected 8 visible PPUs, found {count}"'
"$PYTHON_BIN" -m torch.distributed.run --master-port=29720 --nproc-per-node=8 \
  scripts/train_stock_time_transformer.py \
  --config "$CONFIG_PATH" --panel "$PANEL_PATH" \
  --output artifacts/stock_time_transformer_b0 \
  --epochs 30
