"""Settings, paths and split helpers shared by every script.

Everything that must be identical across our runs lives here, so all three of us
train under the same conditions.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATASETS_ROOT", REPO / "data"))   # raw audio (linked by scripts/link_datasets.sh)
CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", DATA_ROOT / "cache"))  # decoded .npy clips
SPLITS_DIR = REPO / "splits"
RESULTS_CSV = REPO / "results" / "runs.csv"
RUNS_DIR = REPO / "runs"                                               # one folder per training run (git-ignored)

# ---- audio ----
SAMPLE_RATE = 22050
SLICE = 59049              # 3^10 samples, about 2.7 s: one model input
N_SLICES = 11              # every clip is stored as 11 slices (about 29.5 s)
CLIP_LEN = SLICE * N_SLICES
N_MELS, N_FFT, HOP = 128, 2048, 512

# ---- datasets ----
GENRES = {
    "gtzan": ["blues", "classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"],
    "fma_small": ["Electronic", "Experimental", "Folk", "Hip-Hop",
                  "Instrumental", "International", "Pop", "Rock"],
}
GTZAN_EXCLUDE = {"jazz.00054"}           # corrupt in most copies
FMA_TRAIN_SIZES = [800, 1600, 3200, 6400]


def raw_path(dataset: str, clip_id: str) -> Path:
    if dataset == "gtzan":
        genre = clip_id.split(".")[0]
        return DATA_ROOT / "gtzan" / "genres" / genre / f"{clip_id}.wav"
    if dataset == "fma_small":
        return DATA_ROOT / "fma" / "fma_small" / clip_id[:3] / f"{clip_id}.mp3"
    raise ValueError(dataset)


def cache_path(dataset: str, clip_id: str) -> Path:
    return CACHE_ROOT / dataset / f"{clip_id}.npy"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def fma_labels() -> dict[str, str]:
    """track id -> top genre, written by scripts/make_fma_splits.py."""
    f = SPLITS_DIR / "fma_small_labels.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f}: run scripts/make_fma_splits.py first.")
    out = {}
    for line in f.read_text().splitlines()[1:]:
        if line.strip():
            tid, genre = line.split(",", 1)
            out[tid] = genre
    return out


def label_of(dataset: str, clip_id: str, labels: dict[str, str] | None = None) -> int:
    if dataset == "gtzan":
        return GENRES["gtzan"].index(clip_id.split(".")[0])
    return GENRES["fma_small"].index(labels[clip_id])


def split_files(dataset: str, split: str, train_size: int | None = None) -> dict[str, Path]:
    """Paths of the train/valid/test id lists for one experiment.

    gtzan:     split = "random" or "filtered"
    fma_small: split = "official"; train_size = 800, 1600, 3200, 6400 or None (all training clips)
    """
    if dataset == "gtzan":
        if split not in ("random", "filtered"):
            raise SystemExit("GTZAN split must be 'random' or 'filtered'")
        files = {p: SPLITS_DIR / f"gtzan_{split}_{p}.txt" for p in ("train", "valid", "test")}
    elif dataset == "fma_small":
        train = (SPLITS_DIR / f"fma_small_train_{train_size}.txt" if train_size
                 else SPLITS_DIR / "fma_small_train.txt")
        files = {"train": train,
                 "valid": SPLITS_DIR / "fma_small_valid.txt",
                 "test": SPLITS_DIR / "fma_small_test.txt"}
    else:
        raise SystemExit(f"Unknown dataset {dataset}")
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise SystemExit("Missing split files:\n  " + "\n  ".join(missing))
    return files
