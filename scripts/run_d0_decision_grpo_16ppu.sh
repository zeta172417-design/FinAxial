#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FINAXIAL_PYTHON:-/mnt/workspace/zhaozetao/envs/multimodel-ppu/bin/python}"
OUTPUT="artifacts/d0/decision_grpo"
CACHE="artifacts/d0/cache"
cd "$PROJECT_ROOT"
source /usr/local/PPU_SDK/envsetup.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=ERROR
export PCCL_DEBUG=ERROR
export FINMODEL_DDP_TIMEOUT_SECONDS=1800
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

test -f artifacts/panel/phase1/manifest.json
test -f artifacts/d0/predictor/best/model.pt
test -f artifacts/d0/predictor/best/metadata.json
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.device_count() == 16'
if test -f "$OUTPUT/complete.json"; then
  test -f "$OUTPUT/best/policy.pt"
  echo "SKIP completed: $OUTPUT"
  exit 0
fi
if test -d "$OUTPUT" && test -n "$(find "$OUTPUT" -mindepth 1 -print -quit)"; then
  echo "ERROR incomplete output already exists: $OUTPUT" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/build_c0_decision_cache.py \
  --config configs/d0_decision_grpo.json \
  --panel artifacts/panel/phase1 --output "$CACHE"
"$PYTHON_BIN" -m torch.distributed.run \
  --master-port=29994 --nproc-per-node=16 \
  scripts/train_decoupled_decision_rl.py \
  --config configs/d0_decision_grpo.json \
  --algorithm daily_group --action-mode hysteresis \
  --run-suffix d0_grpo \
  --panel artifacts/panel/phase1 --cache "$CACHE" --output "$OUTPUT"
