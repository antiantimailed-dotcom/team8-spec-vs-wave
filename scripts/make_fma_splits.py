"""Write the FMA-small split files used for RQ1.

Uses the official train / validation / test split from fma_metadata/tracks.csv
(artist-filtered by the FMA authors), drops clips that failed preprocessing, then builds
NESTED, genre-balanced training subsets: the 800 set is inside the 1,600 set, which is
inside the 3,200 set, and so on, so only the amount of data changes between sizes.

    python scripts/preprocess.py --dataset fma_small     # first, so failed clips are known
    python scripts/make_fma_splits.py

Writes to splits/:
    fma_small_labels.csv                    track_id,genre for every usable clip
    fma_small_train.txt / _valid / _test    the official split
    fma_small_train_{800,1600,3200,6400}.txt nested subsets (6400 = every usable training clip)
Run it once, commit the files, and never regenerate them.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CACHE_ROOT, DATA_ROOT, FMA_TRAIN_SIZES, GENRES, SPLITS_DIR  # noqa: E402

SEED = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks-csv", type=Path, default=DATA_ROOT / "fma" / "fma_metadata" / "tracks.csv")
    ap.add_argument("--failed", type=Path, default=CACHE_ROOT / "fma_small" / "failed.txt")
    args = ap.parse_args()

    tracks = pd.read_csv(args.tracks_csv, index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"]
    df = pd.DataFrame({
        "track_id": [f"{i:06d}" for i in small.index],
        "genre": small[("track", "genre_top")].astype(str).values,
        "split": small[("set", "split")].astype(str).values,
    })
    print(f"FMA-small in tracks.csv: {len(df)} tracks")

    failed = set()
    if args.failed.exists():
        failed = {line.split("\t")[0] for line in args.failed.read_text().splitlines() if line.strip()}
    else:
        print(f"warning: {args.failed} not found; run preprocess.py first so broken clips are excluded")
    df = df[~df.track_id.isin(failed)]
    unknown = set(df.genre) - set(GENRES["fma_small"])
    if unknown:
        raise SystemExit(f"Unexpected genres in tracks.csv: {unknown}")
    print(f"usable after dropping {len(failed)} failed clip(s): {len(df)}")

    SPLITS_DIR.mkdir(exist_ok=True)
    df.sort_values("track_id")[["track_id", "genre"]].to_csv(SPLITS_DIR / "fma_small_labels.csv", index=False)
    names = {"training": "train", "validation": "valid", "test": "test"}
    for official, ours in names.items():
        ids = sorted(df[df.split == official].track_id)
        (SPLITS_DIR / f"fma_small_{ours}.txt").write_text("\n".join(ids) + "\n")
        print(f"  {ours:5s} {len(ids):5d}  per genre {dict(sorted(Counter(df[df.split == official].genre).items()))}")

    # nested, genre-balanced training subsets
    train = df[df.split == "training"]
    rng = random.Random(SEED)
    order = {}
    for g in GENRES["fma_small"]:
        ids = sorted(train[train.genre == g].track_id)
        rng.shuffle(ids)
        order[g] = ids
    for size in FMA_TRAIN_SIZES:
        if size == max(FMA_TRAIN_SIZES):
            chosen = sorted(train.track_id)                      # every usable training clip
        else:
            per_genre = size // len(GENRES["fma_small"])
            short = [g for g, ids in order.items() if len(ids) < per_genre]
            if short:
                raise SystemExit(f"Not enough training clips in {short} for size {size}")
            chosen = sorted(c for g in order for c in order[g][:per_genre])
        (SPLITS_DIR / f"fma_small_train_{size}.txt").write_text("\n".join(chosen) + "\n")
        print(f"  train_{size}: {len(chosen)} clips")


if __name__ == "__main__":
    main()
