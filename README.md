# team8-spec-vs-wave

Team 8 starter for **music genre classification**: compare a **mel-spectrogram CNN** vs a **raw-waveform CNN** on GTZAN (then FMA-small).

## Setup (Mac Studio / Apple Silicon)

```bash
cd ~/Projects/team8-spec-vs-wave
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# link local datasets (GTZAN + FMA already under ~/Datasets)
./scripts/link_datasets.sh
```

## Train and monitor (GTZAN only)

After this change is merged, update your existing Mac checkout with `git pull --ff-only`
(on `main`). To try the PR before merging instead:

```bash
cd /Users/ianfedele/Projects/team8-spec-vs-wave
git status --short  # commit/stash any local edits before switching branches
git fetch origin
git switch --track origin/feat/tensorboard-monitoring
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run these from the Mac checkout, in its activated `.venv`. Use a **new output
folder** to preserve previous baseline checkpoints/JSON; choose another name for
future experiments. Run sequentially for a fair comparison on the same hardware.
Both commands use the same seed and file split:

```bash
python src/train.py --representation spectrogram --epochs 15 --seed 42 --out-dir checkpoints/tensorboard-comparison-01 --log-dir runs
python src/train.py --representation waveform --epochs 15 --seed 42 --out-dir checkpoints/tensorboard-comparison-01 --log-dir runs
```

Terminal progress bars remain. Each invocation creates a unique representation +
UTC timestamp + random suffix directory under `runs/`. Epoch-level graphs update
after every completed epoch: `loss/train`, `loss/validation`, `accuracy/train`,
`accuracy/validation` (accuracy is a fraction from 0 to 1). Select both runs in
TensorBoard's Scalars view to compare them. Settings and split sizes appear under
Text. Training accuracy is measured during training, with training-mode layers;
validation accuracy is measured in evaluation mode after the epoch.

The best validation checkpoint is still saved as `best_<representation>.pt` and
metrics as `metrics_<representation>.json` under `--out-dir`. JSON history now
also includes training accuracy. Reusing an output folder overwrites these files
but never reuses an event directory. Final `loss/test` and `accuracy/test` are
recorded **once**, after reloading the best validation checkpoint, at that
checkpoint's epoch; they are not a test learning curve. Ctrl-C flushes/closes the
event writer; incomplete epochs have no epoch metrics, and interrupted runs do
not claim final test results. Old baseline runs cannot be graphed retroactively.

### Open TensorBoard remotely

In a second terminal **on the Mac** (or a second Cursor Remote SSH terminal):

```bash
cd /Users/ianfedele/Projects/team8-spec-vs-wave
source .venv/bin/activate
python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

In the laptop's Cursor window already connected to the Mac via Remote SSH, open
the **Ports** panel and forward remote port **6006**, keeping it private/local.
Open `http://localhost:6006` on the laptop (or the local port Cursor assigns).
Alternatively, run this in a **local Mint laptop terminal**, keeping it open:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:6006:127.0.0.1:6006 ianfedele@100.75.160.71
```

Then browse to `http://localhost:6006`. Choose either Cursor forwarding or the
SSH tunnel, not both on the same local port. If local 6006 is occupied, use
`-L 127.0.0.1:16006:127.0.0.1:6006` and browse to `http://localhost:16006`.
No public server, TensorBoard account, or port forwarding on your router is needed.
Training can stay in a `tmux` session on the Mac if you disconnect.
Event files (including custom `--log-dir` locations) are ignored by Git.

### Interpretation and tests

This script trains **GTZAN only**; FMA-small is downloaded/linked for future work,
not supported by this training command. The seeded stratified 70/15/15 split is
file-level, **not artist-disjoint**. Treat it as a baseline, not paper-quality
leakage-controlled results. Unreadable WAVs and AppleDouble files are skipped.

CPU smoke/regression tests use generated short audio fixtures, not full training:

```bash
python -m unittest discover -s tests -v
```

## Data

| Dataset | Path after linking | Notes |
|---|---|---|
| GTZAN | `data/gtzan/genres/` | 1,000 × 30s WAV, 10 genres |
| FMA-small | `data/fma/fma_small/` | 8,000 × 30s MP3 (next) |
| FMA metadata | `data/fma/fma_metadata/` | `tracks.csv`, etc. |

Audio is **not** committed. Teammates should download GTZAN / FMA-small locally and run `scripts/link_datasets.sh` (or set `DATASETS_ROOT`).

## Project layout

```text
src/dataset.py   GTZAN loader (spectrogram | waveform)
src/model.py     SpecCNN + WaveCNN
src/train.py     train / val / test loop (MPS/CUDA/CPU)
scripts/link_datasets.sh
checkpoints/     best_*.pt + metrics_*.json
```

## Next steps

- Add FMA-small loader using `tracks.csv` (`subset == small`)
- Optional: confusion-matrix plots, longer clips, SpecAugment
