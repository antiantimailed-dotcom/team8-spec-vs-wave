# team8-spec-vs-wave

Team 8 (Amanda Aninagyei, Sebastian Kishore, Ian Fedele): does the accuracy gap between a
**log-mel spectrogram CNN** and a **raw-waveform CNN (SampleCNN)** shrink as training data grows?

- **RQ1:** gap vs. training size on FMA-small (800 / 1,600 / 3,200 / 6,400 clips)
- **RQ2:** GTZAN random split vs. fault-filtered (artist-disjoint) split
- **RQ3:** per-genre errors and what the waveform filters learn

## Setup (Mac with Apple silicon, or the cluster)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
./scripts/link_datasets.sh          # or: export DATASETS_ROOT=/path/to/data
```

## 1. Decode the audio once

Every clip is decoded to 22,050 Hz mono and stored as 11 slices of 59,049 samples (`data/cache/`).

```bash
python scripts/preprocess.py --dataset gtzan
python scripts/preprocess.py --dataset fma_small --workers 8
```

Clips that fail to decode or are shorter than 15 s are listed in `data/cache/<dataset>/failed.txt`.

## 2. Splits (made once, committed, never regenerated)

| Files | What | Made by |
|---|---|---|
| `splits/gtzan_filtered_*.txt` | Fault-filtered, artist-disjoint split (Kereliuk et al. 2015): 442 / 197 / 290 | `scripts/make_gtzan_splits.py` |
| `splits/gtzan_random_*.txt` | Random split, same training clips per genre: 442 / 150 / 150 | `scripts/make_gtzan_splits.py` |
| `splits/fma_small_{train,valid,test}.txt` | Official FMA split, minus failed clips | `scripts/make_fma_splits.py` |
| `splits/fma_small_train_{800,1600,3200,6400}.txt` | Nested, genre-balanced training subsets (6400 = all usable) | `scripts/make_fma_splits.py` |

```bash
python scripts/make_fma_splits.py   # after preprocessing FMA-small
```

## 3. Train

One run = one model on one dataset / split / size / seed:

```bash
python src/train.py --dataset gtzan --split filtered --model spectrogram_cnn --seed 0
python src/train.py --dataset fma_small --train-size 800 --model sample_cnn --seed 0
```

Useful options: `--quick` (15 epochs, for a fast check), `--epochs` / `--patience`,
`--num-workers 4`, `--crops-per-clip`. Models: `spectrogram_cnn`, `sample_cnn` (main
comparison), `small_spec_cnn`, `small_wave_cnn` (Ian's original smaller baselines).

Protocol (identical for every model): random 2.7 s crops for training, 8 per clip per
epoch; each validation/test clip is cut into 11 slices whose softmax outputs are averaged;
Adam (lr 1e-3, weight decay 1e-4), LR halved on a plateau, early stopping on validation
macro-F1 (up to 60 epochs, patience 10).

Batch scripts:

```bash
bash scripts/run_gtzan_rq2.sh                 # RQ2: 12 GTZAN runs (fine on a Mac)
mkdir -p logs && sbatch slurm/rq1_sweep.sbatch   # RQ1: 24 FMA-small runs on the HPC cluster
```

## Outputs

- `results/runs.csv`: one row per run (commit this; it is what the report is built from)
- `runs/<run_id>/`: `best.pt`, `history.csv`, `metrics.json` (confusion matrix, per-genre
  report, settings, git commit), `predictions.csv` (per-clip probabilities, for paired tests)

Audio, the cache, `runs/` and checkpoints are not committed.

## Layout

```text
src/common.py      shared settings, paths, split helpers
src/dataset.py     cached-clip datasets (random crops / 11-slice evaluation)
src/model.py       SpectrogramCNN, SampleCNN, SmallSpecCNN, SmallWaveCNN
src/train.py       train / validate / test one run
scripts/           preprocessing, split builders, batch runs, dataset linking
slurm/             cluster job scripts
splits/            committed clip-ID lists
```

If runs use the UTRGV HPC (Cradle) cluster, include the grant acknowledgment from hpc.utrgv.edu in the report.
