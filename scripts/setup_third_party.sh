#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "$ROOT_DIR" submodule update --init --recursive

apply_once() {
  local repo="$1"
  local patch="$2"
  if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied: $patch"
    return
  fi
  git -C "$repo" apply --check "$patch"
  git -C "$repo" apply "$patch"
  echo "applied: $patch"
}

apply_once \
  "$ROOT_DIR/third_party/StockMixer" \
  "$ROOT_DIR/patches/stockmixer-variable-lookback.patch"
