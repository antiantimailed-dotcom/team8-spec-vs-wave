"""Datasets that read pre-decoded clips (see scripts/preprocess.py).

Both models take raw audio in; the spectrogram model makes its own log-mel on the GPU
(model.py), so the two see exactly the same samples.

- TrainCrops: each item is one random 2.7 s crop of a training clip. The dataset
  repeats every clip `crops_per_clip` times, so one epoch draws that many crops per clip.
- EvalClips: each item is a whole clip cut into its 11 non-overlapping slices; the
  trainer averages the model's predictions over them.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CLIP_LEN, N_SLICES, SLICE, cache_path, label_of


class _CachedClips(Dataset):
    def __init__(self, dataset: str, clip_ids: list[str], labels: dict[str, str] | None = None):
        kept, missing = [], []
        for cid in clip_ids:
            (kept if cache_path(dataset, cid).exists() else missing).append(cid)
        if missing:
            print(f"  note: {len(missing)} clip(s) not in the cache were skipped, e.g. {missing[:3]}")
        if not kept:
            raise SystemExit(f"No cached clips found for {dataset}: run scripts/preprocess.py first.")
        self.dataset = dataset
        self.ids = kept
        self.y = [label_of(dataset, c, labels) for c in kept]

    def _load(self, i: int) -> np.ndarray:
        return np.load(cache_path(self.dataset, self.ids[i]), mmap_mode="r")


class TrainCrops(_CachedClips):
    def __init__(self, dataset, clip_ids, labels=None, crops_per_clip: int = 8):
        super().__init__(dataset, clip_ids, labels)
        self.crops_per_clip = crops_per_clip

    def __len__(self) -> int:
        return len(self.ids) * self.crops_per_clip

    def __getitem__(self, idx: int):
        i = idx % len(self.ids)
        # torch's RNG is seeded per DataLoader worker, so crops differ across workers
        off = int(torch.randint(0, CLIP_LEN - SLICE + 1, (1,)))
        x = np.array(self._load(i)[off:off + SLICE], dtype=np.float32)
        return torch.from_numpy(x), self.y[i]


class EvalClips(_CachedClips):
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        x = np.array(self._load(i), dtype=np.float32).reshape(N_SLICES, SLICE)
        return torch.from_numpy(x), self.y[i], i
