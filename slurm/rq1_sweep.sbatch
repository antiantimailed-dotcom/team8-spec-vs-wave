#!/bin/bash
# RQ1 sweep on the UTRGV HPC cluster: 2 models x 4 FMA-small training sizes x 3 seeds = 24 runs.
# Each array task trains one combination on one GPU and appends a row to results/runs.csv.
#
# Before submitting (once):
#   1. Clone the repo on the cluster and create the virtualenv (see README).
#   2. Put the data where DATASETS_ROOT points, then run
#        python scripts/preprocess.py --dataset fma_small --workers 8
#      (or copy your already-built data/cache/fma_small folder over).
#   3. Make sure splits/fma_small_*.txt are committed and pulled.
#   4. Fill in the three lines marked EDIT with the cluster's partition name and setup.
# Submit from the repo root:
#   mkdir -p logs && sbatch slurm/rq1_sweep.sbatch
# Re-run only some tasks, e.g. the ones that failed:
#   sbatch --array=5,17 slurm/rq1_sweep.sbatch

#SBATCH --job-name=team8-rq1
#SBATCH --array=0-23
#SBATCH --partition=gpu               # EDIT: the cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rq1_%A_%a.out

set -euo pipefail

# EDIT: load the cluster's Python/CUDA modules if needed, e.g.  module load python cuda
source .venv/bin/activate              # EDIT: path to the virtualenv on the cluster
# export DATASETS_ROOT=/path/to/data   # EDIT if the data is not under ./data

MODELS=(spectrogram_cnn sample_cnn)
SIZES=(800 1600 3200 6400)
SEEDS=(0 1 2)

i=${SLURM_ARRAY_TASK_ID}
MODEL=${MODELS[$(( i % 2 ))]}
SIZE=${SIZES[$(( (i / 2) % 4 ))]}
SEED=${SEEDS[$(( i / 8 ))]}

echo "task $i: model=$MODEL train_size=$SIZE seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python src/train.py --dataset fma_small --train-size "$SIZE" --model "$MODEL" --seed "$SEED" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
