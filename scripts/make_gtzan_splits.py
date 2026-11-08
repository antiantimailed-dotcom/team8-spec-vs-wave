"""Build the two GTZAN splits used in RQ2 and write them to splits/ as clip-ID lists.

1. Fault-filtered split (Kereliuk, Sturm & Larsen 2015): converted from their published
   train/valid/test lists (github.com/coreyker/dnn-mgr, folder gtzan/). Pass the folder
   holding those three *_filtered.txt files with --filtered-dir.
2. Random stratified split: 70/15/15 per genre, then the training set is cut so that each
   genre has exactly as many training clips as in the fault-filtered training set. Both
   splits therefore train on the same number of clips per genre; only *which* clips differ.

The corrupt clip jazz.00054 is removed from every list.

Usage (from the repo root):
    python scripts/make_gtzan_splits.py --filtered-dir /path/to/dnn-mgr/gtzan

Run it once, commit the files in splits/, and never regenerate them.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
EXCLUDE = {"jazz.00054"}
SEED = 0


def read_filtered(path):
    # lines look like "blues/blues.00029.wav" -> "blues.00029"
    ids = []
    for line in Path(path).read_text().split():
        clip = Path(line.strip()).stem
        if clip and clip not in EXCLUDE:
            ids.append(clip)
    return ids


def genre_of(clip_id):
    return clip_id.split(".")[0]


def write_list(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered-dir", required=True,
                    help="folder with train_filtered.txt, valid_filtered.txt, test_filtered.txt")
    ap.add_argument("--out-dir", default="splits")
    args = ap.parse_args()
    out = Path(args.out_dir)

    # 1. fault-filtered split
    filtered = {}
    for part in ["train", "valid", "test"]:
        filtered[part] = read_filtered(Path(args.filtered_dir) / f"{part}_filtered.txt")
        write_list(out / f"gtzan_filtered_{part}.txt", filtered[part])
    train_counts = Counter(genre_of(c) for c in filtered["train"])

    # 2. random stratified split, training set matched per genre to the filtered one
    rng = random.Random(SEED)
    random_split = {"train": [], "valid": [], "test": []}
    for g in GENRES:
        clips = [f"{g}.{i:05d}" for i in range(100)]
        clips = [c for c in clips if c not in EXCLUDE]
        rng.shuffle(clips)
        n_valid = n_test = 15
        test = clips[:n_test]
        valid = clips[n_test:n_test + n_valid]
        train_pool = clips[n_test + n_valid:]          # about 70 clips
        train = train_pool[:train_counts[g]]           # cut to the filtered count
        random_split["train"] += train
        random_split["valid"] += valid
        random_split["test"] += test
    for part, ids in random_split.items():
        write_list(out / f"gtzan_random_{part}.txt", ids)

    for name, split in [("filtered", filtered), ("random", random_split)]:
        sizes = {p: len(v) for p, v in split.items()}
        print(f"{name:9s} {sizes}")
    print("training clips per genre:", dict(sorted(train_counts.items())))


if __name__ == "__main__":
    main()
