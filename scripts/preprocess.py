"""Decode every clip once to a 22,050 Hz mono float16 .npy file of exactly 11 slices.

Training then reads these small arrays instead of decoding MP3/WAV every epoch.

    python scripts/preprocess.py --dataset gtzan
    python scripts/preprocess.py --dataset fma_small --workers 8

Clips that fail to decode, or hold less than 15 s of audio, are listed in
<cache>/<dataset>/failed.txt and left out; make_fma_splits.py drops them from the splits.
Re-running skips clips already cached (use --overwrite to redo them).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CACHE_ROOT, CLIP_LEN, DATA_ROOT, GTZAN_EXCLUDE, SAMPLE_RATE, cache_path  # noqa: E402

MIN_SECONDS = 15


def list_clips(dataset: str) -> list[tuple[str, Path]]:
    if dataset == "gtzan":
        root = DATA_ROOT / "gtzan" / "genres"
        files = sorted(p for p in root.glob("*/*.wav") if not p.name.startswith("._"))
        return [(p.stem, p) for p in files if p.stem not in GTZAN_EXCLUDE]
    root = DATA_ROOT / "fma" / "fma_small"
    files = sorted(p for p in root.glob("*/*.mp3") if not p.name.startswith("._"))
    return [(p.stem, p) for p in files]


def decode(path: Path) -> np.ndarray:
    """Mono float32 audio at SAMPLE_RATE."""
    import torch
    import torchaudio
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)    # (samples, channels)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
    except Exception:
        wav, sr = torchaudio.load(str(path))                             # fallback decoder
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.numpy()


def work(item):
    dataset, clip_id, path, overwrite = item
    out = cache_path(dataset, clip_id)
    if out.exists() and not overwrite:
        return clip_id, "cached"
    try:
        y = decode(path)
    except Exception as e:
        return clip_id, f"decode error: {str(e)[:100]}"
    if len(y) < MIN_SECONDS * SAMPLE_RATE:
        return clip_id, f"too short: {len(y) / SAMPLE_RATE:.1f} s"
    clip = np.zeros(CLIP_LEN, dtype=np.float16)
    y = y[:CLIP_LEN]
    clip[:len(y)] = y
    np.save(out, clip)
    return clip_id, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["gtzan", "fma_small"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    clips = list_clips(args.dataset)
    if not clips:
        raise SystemExit(f"No audio found for {args.dataset} under {DATA_ROOT}. Run scripts/link_datasets.sh "
                         "or set DATASETS_ROOT.")
    out_dir = CACHE_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips)} clips -> {out_dir}")

    failed, done = [], 0
    items = [(args.dataset, cid, p, args.overwrite) for cid, p in clips]
    with Pool(args.workers) as pool:
        for k, (cid, status) in enumerate(pool.imap_unordered(work, items, chunksize=8), 1):
            if status in ("ok", "cached"):
                done += 1
            else:
                failed.append(f"{cid}\t{status}")
            if k % 500 == 0 or k == len(items):
                print(f"  {k}/{len(items)}  ok {done}  failed {len(failed)}", flush=True)

    (out_dir / "failed.txt").write_text("\n".join(sorted(failed)) + ("\n" if failed else ""))
    print(f"done: {done} cached, {len(failed)} failed (listed in {out_dir / 'failed.txt'})")
    for line in failed[:10]:
        print("  ", line)


if __name__ == "__main__":
    main()
