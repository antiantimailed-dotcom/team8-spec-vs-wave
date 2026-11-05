#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS="${DATASETS_ROOT:-$HOME/Datasets}"

mkdir -p "$ROOT/data/gtzan" "$ROOT/data/fma"

link_one() {
  local src="$1" dest="$2"
  if [[ ! -e "$src" ]]; then
    echo "Missing: $src" >&2
    exit 1
  fi
  rm -rf "$dest"
  ln -s "$src" "$dest"
  echo "linked $dest -> $src"
}

link_one "$DATASETS/gtzan/genres" "$ROOT/data/gtzan/genres"
link_one "$DATASETS/fma/fma_small" "$ROOT/data/fma/fma_small"
link_one "$DATASETS/fma/fma_metadata" "$ROOT/data/fma/fma_metadata"
echo "Done."
