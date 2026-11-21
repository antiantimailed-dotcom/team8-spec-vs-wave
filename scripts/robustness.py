"""Extension: which input breaks first on degraded audio? (evaluation only, no training)

Loads the saved best.pt of every full run in the chosen settings and re-tests it on
degraded copies of the same test clips:

    clean            the original test audio (should match results/runs.csv)
    noise_snr20/10/0 white noise at 20, 10 and 0 dB signal-to-noise ratio
    lowpass_4k/2k    8th-order low-pass filter at 4 kHz and 2 kHz (removes treble)
    mp3_64k/32k      MP3 re-encoding at 64 and 32 kbit/s (needs ffmpeg: brew install ffmpeg)

Every model sees exactly the same degraded audio (fixed noise seed), so the comparison is
paired. Usage, from the repo root:

    python scripts/robustness.py                               # FMA 6,400 and GTZAN filtered
    python scripts/robustness.py --settings fma_6400 --limit 100   # quick check on 100 clips

Writes results/robustness.csv (one row per run x condition), results/robustness.md
(mean ± SD over seeds and the drop from clean) and figures/robustness_<setting>.png.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, correlate, sosfiltfilt  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CLIP_LEN, GENRES, N_SLICES, SAMPLE_RATE, SLICE, cache_path, fma_labels, label_of, read_ids, split_files  # noqa: E402
from results_io import REPO, load_runs  # noqa: E402

SPEC, WAVE = "spectrogram_cnn", "sample_cnn"
LABEL = {SPEC: "Spectrogram CNN", WAVE: "SampleCNN"}
SETTINGS = {
    "fma_6400": dict(dataset="fma_small", split="official", size=6400, title="FMA-small, 6,400 training clips"),
    "gtzan_filtered": dict(dataset="gtzan", split="filtered", size=None, title="GTZAN fault-filtered"),
    "gtzan_random": dict(dataset="gtzan", split="random", size=None, title="GTZAN random split"),
}
CONDITIONS = ["clean", "noise_snr20", "noise_snr10", "noise_snr0", "lowpass_4k", "lowpass_2k", "mp3_64k", "mp3_32k"]


# ---------------- degradations (numpy only) ----------------
def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(x ** 2))
    if power == 0:
        return x.copy()
    noise = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(power / 10 ** (snr_db / 10))
    return (x + noise).astype(np.float32)


def lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = butter(8, cutoff_hz, btype="low", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def mp3(x: np.ndarray, kbps: int) -> np.ndarray:
    """Encode to MP3 and decode back with ffmpeg, through pipes (no temp files)."""
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    raw = ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1"]
    enc = subprocess.run(base + raw + ["-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"],
                         input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), capture_output=True, check=True)
    dec = subprocess.run(base + ["-f", "mp3", "-i", "pipe:0"] + raw + ["pipe:1"],
                         input=enc.stdout, capture_output=True, check=True)
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    # MP3 adds a fixed delay at the start (about 1,105 samples, 50 ms); line the audio back up
    # so every slice covers the same moment as in the clean clip.
    lag = _delay(x, y)
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag, np.float32), y])
    out = np.zeros(len(x), dtype=np.float32)
    out[:min(len(x), len(y))] = y[:len(x)]
    return out


def _delay(x: np.ndarray, y: np.ndarray, max_lag: int = 4096) -> int:
    """Shift of y relative to x, from cross-correlating a 2**17-sample window."""
    n = min(len(x), len(y), 2 ** 17)
    if n <= 2 * max_lag or not np.any(x[:n]):
        return 0
    c = correlate(y[:n], x[:n], mode="full", method="fft")
    mid = n - 1                                             # index of zero lag
    window = c[mid - max_lag: mid + max_lag + 1]
    return int(np.argmax(window)) - max_lag


def degrade(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return x
    kind, level = condition.split("_")
    if kind == "noise":
        return add_noise(x, float(level.replace("snr", "")), rng)
    if kind == "lowpass":
        return lowpass(x, float(level.replace("k", "")) * 1000)
    if kind == "mp3":
        return mp3(x, int(level.replace("k", "")))
    raise ValueError(condition)


# ---------------- data and models ----------------
def test_set(setting: dict, limit: int | None):
    files = split_files(setting["dataset"], setting["split"], setting["size"])
    labels = fma_labels() if setting["dataset"] == "fma_small" else None
    ids = [c for c in read_ids(files["test"]) if cache_path(setting["dataset"], c).exists()]
    if limit:
        ids = ids[:limit]
    y = np.array([label_of(setting["dataset"], c, labels) for c in ids])
    return ids, y


def load_clean(dataset: str, ids: list[str]) -> np.ndarray:
    return np.stack([np.load(cache_path(dataset, c)).astype(np.float32) for c in ids])


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_models(runs: pd.DataFrame, n_classes: int, device):
    import torch
    from model import build_model
    models = {}
    for _, r in runs.iterrows():
        ckpt = torch.load(r["run_dir"] / "best.pt", map_location="cpu", weights_only=False)
        m = build_model(r["model"], n_classes)
        m.load_state_dict(ckpt["model"])
        models[r["run_id"]] = m.to(device).eval()
    return models


def predict(model, audio: np.ndarray, device, clips_per_batch: int = 8) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    preds = []
    with torch.no_grad():
        for k in range(0, len(audio), clips_per_batch):
            chunk = torch.from_numpy(np.ascontiguousarray(audio[k:k + clips_per_batch], dtype=np.float32))
            b = chunk.shape[0]
            p = F.softmax(model(chunk.view(-1, SLICE).to(device)), dim=1).view(b, N_SLICES, -1).mean(1)
            preds.append(p.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---------------- main loop ----------------
def run_setting(name: str, setting: dict, conditions: list[str], limit, seed: int, rows: list):
    runs = load_runs(setting["dataset"], setting["split"], models=[SPEC, WAVE])
    runs = runs[runs["size"].isna()] if setting["size"] is None else runs[runs["size"] == setting["size"]]
    runs = runs[[(d / "best.pt").exists() for d in runs["run_dir"]]]
    if runs.empty:
        print(f"{name}: no full runs with a best.pt here, skipping"); return
    genres = GENRES[setting["dataset"]]
    ids, y = test_set(setting, limit)
    print(f"{name}: {len(runs)} models, {len(ids)} test clips, conditions: {', '.join(conditions)}")
    device = pick_device()
    models = load_models(runs, len(genres), device)
    clean = load_clean(setting["dataset"], ids)
    for cond in conditions:
        t0 = time.time()
        audio = np.empty_like(clean, dtype=np.float16)
        for i in range(len(ids)):                              # the same degraded clip for every model
            rng = np.random.default_rng([seed, i])
            audio[i] = degrade(clean[i], cond, rng)
        for _, r in runs.iterrows():
            pred = predict(models[r["run_id"]], audio, device)
            rows.append({"setting": name, "condition": cond, "run_id": r["run_id"], "model": r["model"],
                         "seed": int(r["seed"]), "n_clips": len(ids), "accuracy": accuracy_score(y, pred),
                         "macro_f1": f1_score(y, pred, labels=list(range(len(genres))), average="macro", zero_division=0)})
        accs = {m: np.mean([x["accuracy"] for x in rows if x["setting"] == name and x["condition"] == cond and x["model"] == m])
                for m in (SPEC, WAVE)}
        print(f"  {cond:12s} spectrogram {accs[SPEC]:.3f}  SampleCNN {accs[WAVE]:.3f}  ({time.time() - t0:.0f} s)")


def summarize(df: pd.DataFrame, outdir: Path, figdir: Path):
    lines = ["# Robustness to degraded audio", "",
             "Test accuracy, mean ± SD over seeds. Drop = clean − degraded, paired by seed "
             "(larger = more fragile). 'Spectrogram drops more' counts seeds.", ""]
    for name, g in df.groupby("setting", sort=False):
        clean = g[g.condition == "clean"].set_index(["model", "seed"])["accuracy"]
        g = g.assign(acc_drop=[clean.get((m, s), np.nan) - a for m, s, a in zip(g.model, g.seed, g.accuracy)])
        lines += [f"## {SETTINGS[name]['title']}", "",
                  "| Condition | Spectrogram | SampleCNN | Spectrogram drop | SampleCNN drop | Seeds where spectrogram drops more |",
                  "|---|---|---|---|---|---|"]
        conds = [c for c in CONDITIONS if c in set(g.condition)]
        stats = {}
        for c in conds:
            s, w = g[(g.condition == c) & (g.model == SPEC)], g[(g.condition == c) & (g.model == WAVE)]
            both = sorted(set(s.seed) & set(w.seed))
            sd_, wd_ = s.set_index("seed")["acc_drop"], w.set_index("seed")["acc_drop"]
            more = sum(sd_[k] > wd_[k] for k in both)
            stats[c] = (s.accuracy.mean(), s.accuracy.std(), w.accuracy.mean(), w.accuracy.std())
            sd = lambda v: "" if np.isnan(v) else f" ± {100 * v:.1f}"  # noqa: E731
            lines.append(f"| {c} | {100 * s.accuracy.mean():.1f}{sd(s.accuracy.std())} | {100 * w.accuracy.mean():.1f}{sd(w.accuracy.std())} "
                         f"| {100 * s['acc_drop'].mean():+.1f} | {100 * w['acc_drop'].mean():+.1f} | {'—' if c == 'clean' else f'{more} of {len(both)}'} |")
        lines.append("")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(conds))
        for k, (m, col, off) in enumerate([(SPEC, "#2F6FB0", -0.2), (WAVE, "#E07A2F", 0.2)]):
            mean = [stats[c][2 * k] for c in conds]; err = [np.nan_to_num(stats[c][2 * k + 1]) for c in conds]
            ax.bar(x + off, mean, 0.4, yerr=err, capsize=3, color=col, label=LABEL[m])
        ax.set_xticks(x, [c.replace("_", "\n") for c in conds]); ax.set_ylim(0, 1)
        ax.set_ylabel("test accuracy (mean ± SD over seeds)")
        ax.set_title(f"Robustness, {SETTINGS[name]['title']}"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(figdir / f"robustness_{name}.png", dpi=150); plt.close(fig)
    (outdir / "robustness.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=["fma_6400", "gtzan_filtered"], choices=list(SETTINGS))
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N test clips (for a quick check)")
    ap.add_argument("--seed", type=int, default=0, help="noise seed (kept fixed so models are compared fairly)")
    args = ap.parse_args()
    conditions = list(args.conditions)
    if any(c.startswith("mp3") for c in conditions) and not shutil.which("ffmpeg"):
        print("ffmpeg not found, so the MP3 conditions are skipped (install it with: brew install ffmpeg)")
        conditions = [c for c in conditions if not c.startswith("mp3")]
    outdir, figdir = REPO / "results", REPO / "figures"
    outdir.mkdir(exist_ok=True); figdir.mkdir(exist_ok=True)
    rows = []
    for name in args.settings:
        run_setting(name, SETTINGS[name], conditions, args.limit, args.seed, rows)
        pd.DataFrame(rows).to_csv(outdir / "robustness.csv", index=False, float_format="%.4f")  # saved as it goes
    if rows:
        summarize(pd.DataFrame(rows), outdir, figdir)
        print(f"wrote {outdir / 'robustness.csv'}, {outdir / 'robustness.md'} and figures/robustness_*.png")


if __name__ == "__main__":
    main()
