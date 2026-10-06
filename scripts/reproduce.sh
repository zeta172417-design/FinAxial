#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_csv="${1:?usage: bash scripts/reproduce.sh path/to/train.csv}"
PYTHON_BIN="${FINAXIAL_PYTHON:-python}"
if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then source /usr/local/PPU_SDK/envsetup.sh; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
root=artifacts/reproduction
mkdir -p "$root"
exec 9>"$root/.workflow.lock"
flock -n 9 || { echo "Another reproduction workflow is using $root" >&2; exit 1; }
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.device_count() == 4, "reproduction requires exactly four devices"'
"$PYTHON_BIN" scripts/preprocess.py --source "$source_csv" --output "$root/panel"
"$PYTHON_BIN" scripts/build_factor_cache.py --panel "$root/panel" --output "$root/factors" --train-end-date 20231228
for stage in predictor decision; do
  output="$root/$stage/full"
  if [[ "$stage" == predictor ]]; then epochs=20; else epochs=8; fi
  "$PYTHON_BIN" scripts/check_training_stage.py --output "$output" --epochs "$epochs"
  if test -f "$output/workflow_complete.json"; then continue; fi
  mkdir -p "$root/$stage"
  if [[ "$stage" == predictor ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --nproc-per-node=4 --master-port=29980 \
      scripts/train_stock_time_transformer.py --config configs/predictor.json \
      --panel "$root/panel" --output "$output" --disable-swanlab > "$output.log" 2>&1
  else
    "$PYTHON_BIN" scripts/build_predictor_decision_cache.py --config configs/decision_grpo.json \
      --panel "$root/panel" --output "$root/cache"
    "$PYTHON_BIN" -m torch.distributed.run --nproc-per-node=4 --master-port=29981 \
      scripts/train_decoupled_decision_rl.py --algorithm daily_group --config configs/decision_grpo.json \
      --panel "$root/panel" --cache "$root/cache" --output "$output" --disable-swanlab > "$output.log" 2>&1
  fi
  "$PYTHON_BIN" scripts/finish_training_stage.py --output "$output" --epochs "$epochs"
done
echo "DONE: last-checkpoint validation metrics in $root/decision/full/train_summary.json"
