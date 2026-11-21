#!/usr/bin/env bash
# Run the full RQ1 sweep on one Mac, one job at a time:
# 2 models x 4 FMA-small training sizes x 3 seeds = 24 runs.
# Safe to stop (Ctrl-C) and re-run: finished full runs are skipped.
# Usage, from the repo root:
#   PYTHON=../team8-spec-vs-wave/.venv/bin/python bash scripts/run_rq1_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
WORKERS="${WORKERS:-8}"
mkdir -p logs

[[ -e data/fma/fma_small ]] || bash scripts/link_datasets.sh
if [[ ! -f splits/fma_small_train.txt ]]; then
  "$PY" scripts/preprocess.py --dataset fma_small --workers "$WORKERS"
  "$PY" scripts/make_fma_splits.py
fi

done_run() {  # true if a finished, non-quick run exists for this combo
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json, glob, sys
size, model, seed = sys.argv[1:]
for m in glob.glob(f"runs/fma_small_*_n{size}_{model}_s{seed}_*/metrics.json"):
    if not json.load(open(m)).get("args", {}).get("quick"):
        sys.exit(0)
sys.exit(1)
PYEOF
}

for SEED in 0 1 2; do
  for SIZE in 800 1600 3200 6400; do
    for MODEL in spectrogram_cnn sample_cnn; do
      if done_run "$SIZE" "$MODEL" "$SEED"; then
        echo "skip  $MODEL n=$SIZE seed=$SEED (already done)"; continue
      fi
      echo "start $MODEL n=$SIZE seed=$SEED  $(date '+%H:%M')"
      caffeinate -i "$PY" src/train.py --dataset fma_small --train-size "$SIZE" \
        --model "$MODEL" --seed "$SEED" --num-workers "$WORKERS" \
        2>&1 | tee "logs/rq1_${MODEL}_n${SIZE}_s${SEED}.log"
    done
  done
done

"$PY" scripts/plot_rq1_gap.py
echo "RQ1 sweep finished. See results/rq1_gap.png and results/rq1_gap.csv"
