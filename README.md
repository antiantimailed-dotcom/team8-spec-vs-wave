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

## Train

Spectrogram model (default):

```bash
source .venv/bin/activate
python src/train.py --representation spectrogram --epochs 15
```

Waveform model:

```bash
python src/train.py --representation waveform --epochs 15
```

Checkpoints and metrics land in `checkpoints/`.

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
- Log side-by-side spectrogram vs waveform accuracy
- Optional: confusion-matrix plots, longer clips, SpecAugment
