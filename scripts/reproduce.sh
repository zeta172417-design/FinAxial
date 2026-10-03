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
"$PYTHON_BIN" scripts/preprocess.py --source "$source_csv" --output "$root/panel"
"$PYTHON_BIN" scripts/build_factor_cache.py --panel "$root/panel" --output "$root/factors" --train-end-date 20231228
for stage in predictor decision; do
  output="$root/$stage/full"
  if test -f "$output/complete.json"; then echo "REUSE $output"; continue; fi
  if test -e "$output" || test -e "$output.log"; then echo "Refusing partial run: $output" >&2; exit 1; fi
  mkdir -p "$root/$stage"
  if [[ "$stage" == predictor ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --nproc-per-node=4 --master-port=29980 \
      scripts/train_stock_time_transformer.py --config configs/predictor.json \
      --panel "$root/panel" --output "$output" --disable-swanlab > "$output.log" 2>&1
  else
    "$PYTHON_BIN" scripts/build_c0_decision_cache.py --config configs/decision_grpo.json \
      --panel "$root/panel" --output "$root/cache"
    "$PYTHON_BIN" -m torch.distributed.run --nproc-per-node=4 --master-port=29981 \
      scripts/train_decoupled_decision_rl.py --algorithm daily_group --config configs/decision_grpo.json \
      --panel "$root/panel" --cache "$root/cache" --output "$output" --disable-swanlab > "$output.log" 2>&1
  fi
  test -f "$output/complete.json"
done
echo "DONE: last-checkpoint validation metrics in $root/decision/full/train_summary.json"
