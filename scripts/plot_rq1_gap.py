"""RQ1: does the spectrogram-vs-waveform gap shrink from 800 to 6,400 FMA-small training clips?

Reads results/runs.csv (one row per run, written by src/train.py), keeps the FMA-small runs of
the two main models, averages over seeds, and computes

    gap = spectrogram_cnn - sample_cnn        (positive = spectrogram ahead)

paired by seed where both models have that seed. Writes a table and a figure:

    results/rq1_gap.csv   one row per metric x training size
    results/rq1_gap.png   performance vs training size (top) and the gap (bottom)

Examples (from the repo root):
    python scripts/plot_rq1_gap.py
    python scripts/plot_rq1_gap.py --metric macro_f1
    python scripts/plot_rq1_gap.py --input results/runs.csv --outdir results --include-quick

Runs made with --quick (max_epochs 15) are left out unless --include-quick is given. If the
same model / size / seed was run more than once, the last row in the CSV is used.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from common import FMA_TRAIN_SIZES  # noqa: E402  (pure stdlib module, no torch import)

SPEC, WAVE = "spectrogram_cnn", "sample_cnn"
LABELS = {SPEC: "Spectrogram CNN", WAVE: "SampleCNN (waveform)"}
METRICS = {"accuracy": ("test_accuracy", "Test accuracy"),
           "macro_f1": ("test_macro_f1", "Test macro-F1")}
QUICK_EPOCHS = 15                     # what train.py --quick sets max_epochs to
REQUIRED = ["run_id", "model", "dataset", "seed", "train_size", "test_accuracy", "test_macro_f1"]


def nominal_size(run_id: str, train_size) -> int | None:
    """The requested size (800/1600/3200/6400). run_id carries it as _n<size>_; the train_size
    column holds the actual clip count, which for 6400 ("every usable clip") can be a bit lower."""
    m = re.search(r"_n(\d+)_", str(run_id))
    if m:
        return int(m.group(1))
    try:
        n = int(train_size)
    except (TypeError, ValueError):
        return None
    return min(FMA_TRAIN_SIZES, key=lambda s: abs(s - n))


def sd(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def load(args) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{args.input} is missing column(s): {', '.join(missing)}")
    n_all = len(df)
    df = df[(df["dataset"] == args.dataset) & (df["model"].isin([SPEC, WAVE]))].copy()
    n_ds = len(df)
    if not args.include_quick and "max_epochs" in df.columns:
        df = df[pd.to_numeric(df["max_epochs"], errors="coerce") != QUICK_EPOCHS]
    n_quick = n_ds - len(df)
    df["size"] = [nominal_size(r, t) for r, t in zip(df["run_id"], df["train_size"])]
    df = df.dropna(subset=["size"])
    df["size"] = df["size"].astype(int)
    df["seed"] = df["seed"].astype(int)
    before = len(df)
    df = df.drop_duplicates(subset=["model", "size", "seed"], keep="last")
    print(f"{args.input}: {n_all} rows, {n_ds} {args.dataset} rows for {SPEC}/{WAVE}"
          f"{f', {n_quick} quick run(s) excluded' if n_quick else ''}"
          f"{f', {before - len(df)} duplicate run(s) replaced by the latest' if before > len(df) else ''}"
          f" -> {len(df)} runs used")
    return df


def summarize(df: pd.DataFrame, metric: str) -> list[dict]:
    col = METRICS[metric][0]
    rows = []
    for size in sorted(df["size"].unique()):
        d = df[df["size"] == size]
        s = d[d["model"] == SPEC].set_index("seed")[col].astype(float)
        w = d[d["model"] == WAVE].set_index("seed")[col].astype(float)
        seeds = sorted(set(s.index) & set(w.index))
        if seeds:                                     # paired by seed
            gaps = (s.loc[seeds] - w.loc[seeds]).to_numpy()
            gap_mean, gap_std, paired = float(gaps.mean()), sd(gaps), True
        elif len(s) and len(w):                       # no common seeds: difference of means
            gap_mean, gap_std, paired = float(s.mean() - w.mean()), float("nan"), False
        else:
            gap_mean, gap_std, paired = float("nan"), float("nan"), False
        rows.append({
            "metric": metric, "train_size": int(size),
            "spec_n": len(s), "spec_mean": float(s.mean()) if len(s) else float("nan"), "spec_std": sd(s),
            "wave_n": len(w), "wave_mean": float(w.mean()) if len(w) else float("nan"), "wave_std": sd(w),
            "gap_mean": gap_mean, "gap_std": gap_std, "gap_n_pairs": len(seeds), "gap_paired": paired,
            "paired_seeds": " ".join(map(str, seeds)),
        })
    return rows


def plot(table: pd.DataFrame, metrics: list[str], out: Path, dataset: str) -> None:
    fig, axes = plt.subplots(2, len(metrics), figsize=(max(6.5, 5.5 * len(metrics)), 7), squeeze=False,
                             sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for j, metric in enumerate(metrics):
        t = table[table["metric"] == metric].sort_values("train_size")
        top, bot = axes[0, j], axes[1, j]
        for model, key, marker in [(SPEC, "spec", "o"), (WAVE, "wave", "s")]:
            ok = t[f"{key}_n"] > 0
            top.errorbar(t.loc[ok, "train_size"], t.loc[ok, f"{key}_mean"],
                         yerr=t.loc[ok, f"{key}_std"].fillna(0), marker=marker, capsize=4,
                         label=LABELS[model])
        top.set_ylabel(METRICS[metric][1])
        top.set_title(f"{METRICS[metric][1]} on {dataset} (mean ± SD over seeds)", fontsize=10)
        top.grid(alpha=0.3); top.legend(fontsize=9)
        ok = t["gap_mean"].notna()
        bot.errorbar(t.loc[ok, "train_size"], t.loc[ok, "gap_mean"], yerr=t.loc[ok, "gap_std"].fillna(0),
                     marker="D", color="tab:purple", capsize=4)
        bot.axhline(0, color="grey", lw=1, ls="--")
        bot.set_ylabel("Gap (spectrogram − waveform)")
        bot.set_xlabel("Training clips (log scale)")
        bot.grid(alpha=0.3)
        bot.set_xscale("log", base=2)
        bot.set_xticks(FMA_TRAIN_SIZES)
        bot.set_xticklabels([f"{s:,}" for s in FMA_TRAIN_SIZES])
        bot.set_xlim(FMA_TRAIN_SIZES[0] / 1.3, FMA_TRAIN_SIZES[-1] * 1.3)
    fig.suptitle("RQ1: spectrogram vs. waveform gap by training size")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fmt(m, s) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "   n/a        "
    return f"{m:+.3f}" if s is None else (f"{m:.3f} ± {s:.3f}" if not math.isnan(s) else f"{m:.3f}        ")


def report(table: pd.DataFrame, metrics: list[str]) -> None:
    lo, hi = FMA_TRAIN_SIZES[0], FMA_TRAIN_SIZES[-1]
    for metric in metrics:
        t = table[table["metric"] == metric].set_index("train_size")
        print(f"\n== {METRICS[metric][1]} ==")
        print(f"{'clips':>6}  {'spectrogram':<15} {'waveform':<15} {'gap (paired)':<22} seeds(spec/wave)")
        for size in FMA_TRAIN_SIZES:
            if size not in t.index:
                print(f"{size:>6}  no runs yet")
                continue
            r = t.loc[size]
            gap = fmt(r.gap_mean, r.gap_std)
            gap += "" if r.gap_paired or math.isnan(r.gap_mean) else " (unpaired)"
            print(f"{size:>6}  {fmt(r.spec_mean, r.spec_std):<15} {fmt(r.wave_mean, r.wave_std):<15} "
                  f"{gap:<22} {r.spec_n}/{r.wave_n}")
        g_lo = t.loc[lo, "gap_mean"] if lo in t.index else float("nan")
        g_hi = t.loc[hi, "gap_mean"] if hi in t.index else float("nan")
        if math.isnan(g_lo) or math.isnan(g_hi):
            print(f"-> cannot compare the gap at {lo:,} and {hi:,} clips yet (need both models at both sizes)")
        else:
            verdict = ("smaller (gap narrows)" if abs(g_hi) < abs(g_lo) else "not smaller (gap does not narrow)")
            print(f"-> gap at {hi:,} = {g_hi:+.3f} vs gap at {lo:,} = {g_lo:+.3f}: |gap| at {hi:,} is {verdict}")
            lead = "spectrogram leads" if g_lo > 0 else "waveform leads"
            print(f"   at {lo:,} clips the {lead}; with few seeds, check the ± spread before calling it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO / "results" / "runs.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "results")
    ap.add_argument("--metric", choices=["accuracy", "macro_f1", "both"], default="both")
    ap.add_argument("--dataset", default="fma_small")
    ap.add_argument("--include-quick", action="store_true", help="keep --quick runs (max_epochs 15)")
    args = ap.parse_args()
    if not args.input.exists():
        sys.exit(f"no results file at {args.input}")

    metrics = ["accuracy", "macro_f1"] if args.metric == "both" else [args.metric]
    df = load(args)
    if df.empty:
        sys.exit(f"no {args.dataset} runs of {SPEC}/{WAVE} found yet; run the RQ1 sweep first")
    extra = sorted(set(df["size"]) - set(FMA_TRAIN_SIZES))
    if extra:
        print(f"note: unexpected training size(s) {extra} are included in the table")

    table = pd.DataFrame([r for m in metrics for r in summarize(df, m)])
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_out, png_out = args.outdir / "rq1_gap.csv", args.outdir / "rq1_gap.png"
    table.to_csv(csv_out, index=False, float_format="%.4f")
    plot(table, metrics, png_out, args.dataset)
    report(table, metrics)
    print(f"\nwrote {csv_out} and {png_out}")


if __name__ == "__main__":
    main()
