#!/bin/bash
# RQ2 on GTZAN: 2 splits x 2 models x 3 seeds = 12 runs, one after another.
# Runs fine on a Mac with Apple silicon (SampleCNN is the slow part); each run appends to results/runs.csv.
#   bash scripts/run_gtzan_rq2.sh            # full training (up to 60 epochs, early stopping)
#   bash scripts/run_gtzan_rq2.sh --quick    # 15 epochs, for a fast check
set -euo pipefail
for SEED in 0 1 2; do
  for SPLIT in random filtered; do
    for MODEL in spectrogram_cnn sample_cnn; do
      python src/train.py --dataset gtzan --split "$SPLIT" --model "$MODEL" --seed "$SEED" "$@"
    done
  done
done
