"""Per-clip significance tests for RQ1 and RQ2, from the saved predictions of every run.

    python scripts/significance.py                  # both RQs, 10,000 bootstrap resamples
    python scripts/significance.py --bootstrap 2000 # faster

For every comparison of spectrogram_cnn vs. sample_cnn on the same test clips:
  - per seed: accuracy of each model and an exact McNemar test (do the two models get
    different clips right more often than chance?);
  - pooled over seeds: the mean accuracy gap with a 95% bootstrap CI and p-value, made by
    resampling test clips. The CI shows uncertainty from the finite test set; seed-to-seed
    spread is shown separately by the per-seed rows.

RQ1 (FMA-small): the gap at each training size, and the slope of the gap per doubling of the
training set (the "does it narrow?" test), with a bootstrap CI over the shared test clips.
RQ2 (GTZAN): the gap on each split, and the difference in accuracy drop between the models
(drop_spec - drop_wave = gap_random - gap_filtered), resampling each split's test set.

Writes results/significance_rq1.md/.csv (FMA) and results/significance_rq2.md/.csv (GTZAN).
Each pair is written only when that dataset's runs are present, so running this on a machine
with only one dataset never overwrites the other dataset's results.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from results_io import REPO, load_predictions, load_runs  # noqa: E402

SPEC, WAVE = "spectrogram_cnn", "sample_cnn"


# ---------------- core statistics ----------------
def mcnemar(s: np.ndarray, w: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired correctness vectors: b = only spec right, c = only wave right."""
    b = int(np.sum(s & ~w)); c = int(np.sum(~s & w))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, float(p)


def boot_p(samples: np.ndarray, n_boot: int) -> float:
    """Two-sided bootstrap p-value for 'the true value is 0'."""
    p = 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))
    return float(min(1.0, max(p, 1.0 / n_boot)))


def ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------- loading paired correctness ----------------
def paired_matrix(runs: pd.DataFrame):
    """For one setting (same test set), return seeds, clip ids and correctness arrays
    S, W of shape (n_seeds, n_clips) for the seeds both models have."""
    by = {}
    for _, r in runs.iterrows():
        by[(r["model"], r["seed"])] = load_predictions(r["run_dir"])["correct"]
    seeds = sorted({s for m, s in by if m == SPEC} & {s for m, s in by if m == WAVE})
    if not seeds:
        return [], [], None, None
    clips = None
    for k in [(m, s) for m in (SPEC, WAVE) for s in seeds]:
        ids = set(by[k].index)
        clips = ids if clips is None else clips & ids
    clips = sorted(clips)
    S = np.stack([by[(SPEC, s)].loc[clips].to_numpy(bool) for s in seeds])
    W = np.stack([by[(WAVE, s)].loc[clips].to_numpy(bool) for s in seeds])
    return seeds, clips, S, W


def compare(label: str, runs: pd.DataFrame, rng, n_boot: int, rows: list, lines: list):
    seeds, clips, S, W = paired_matrix(runs)
    if not seeds:
        lines.append(f"| {label} | — | — | — | no seed run by both models | | |")
        return None
    for k, s in enumerate(seeds):
        b, c, p = mcnemar(S[k], W[k])
        rows.append({"comparison": label, "seed": s, "n_clips": len(clips), "spec_acc": S[k].mean(),
                     "wave_acc": W[k].mean(), "gap": S[k].mean() - W[k].mean(),
                     "only_spec_right": b, "only_wave_right": c, "mcnemar_p": p})
    d = (S.astype(float) - W.astype(float)).mean(0)              # per-clip gap, averaged over seeds
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    gap, (lo, hi), p = float(d.mean()), ci(boots), boot_p(boots, n_boot)
    per_seed = (S.mean(1) - W.mean(1))
    rows.append({"comparison": label, "seed": "pooled", "n_clips": len(clips), "spec_acc": S.mean(),
                 "wave_acc": W.mean(), "gap": gap, "ci_low": lo, "ci_high": hi, "boot_p": p})
    seed_txt = ", ".join(f"{100 * g:+.1f}" for g in per_seed)
    mc = ", ".join(f"{r['mcnemar_p']:.3f}" for r in rows if r["comparison"] == label and r["seed"] != "pooled")
    lines.append(f"| {label} | {100 * S.mean():.1f}% | {100 * W.mean():.1f}% | **{100 * gap:+.1f}** "
                 f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {fmt_p(p)} | {seed_txt} | {mc} |")
    return {"seeds": seeds, "clips": clips, "d": d}


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


HEADER = ("| Comparison | Spectrogram | SampleCNN | Gap, pts [95% CI] | p (bootstrap) | Gap per seed | McNemar p per seed |\n"
          "|---|---|---|---|---|---|---|")


# ---------------- RQ1 ----------------
def rq1(rng, n_boot, rows, lines):
    runs = load_runs("fma_small", models=[SPEC, WAVE])
    if runs.empty:
        print("RQ1: no full FMA-small runs here, skipping (significance_rq1.* left untouched)")
        return False
    lines += ["## RQ1: gap by training size (FMA-small)", "", HEADER]
    per_size = {}
    for size in sorted(runs["size"].dropna().unique()):
        res = compare(f"FMA n={int(size):,}", runs[runs["size"] == size], rng, n_boot, rows, lines)
        if res:
            per_size[int(size)] = res
    lines.append("")
    if len(per_size) >= 2:
        common = sorted(set.intersection(*[set(r["clips"]) for r in per_size.values()]))
        sizes = sorted(per_size)
        D = np.stack([pd.Series(per_size[n]["d"], index=per_size[n]["clips"]).loc[common].to_numpy()
                      for n in sizes])                              # (n_sizes, n_clips)
        x = np.log2(sizes); xc = x - x.mean()
        slope = float((D.mean(1) * xc).sum() / (xc ** 2).sum())
        idx = rng.integers(0, len(common), size=(n_boot, len(common)))  # same clips for every size
        G = np.stack([D[j][idx].mean(1) for j in range(len(sizes))])   # (n_sizes, n_boot)
        slopes = (G * xc[:, None]).sum(0) / (xc ** 2).sum()
        lo, hi = ci(slopes); p = boot_p(slopes, n_boot)
        rows.append({"comparison": "FMA gap slope per doubling", "seed": "pooled", "n_clips": len(common),
                     "gap": slope, "ci_low": lo, "ci_high": hi, "boot_p": p})
        verdict = ("no evidence the gap changes with training size" if lo <= 0 <= hi else
                   "the gap narrows as data grows" if slope < 0 else "the gap widens as data grows")
        lines += [f"**Trend:** the gap changes by **{100 * slope:+.2f} pts per doubling** of the training set "
                  f"(95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}], p = {fmt_p(p)}, {len(common)} shared test clips): "
                  f"{verdict}.", ""]
    return True


# ---------------- RQ2 ----------------
def rq2(rng, n_boot, rows, lines):
    runs = load_runs("gtzan", models=[SPEC, WAVE])
    runs = runs[runs["split"].isin(["random", "filtered"])]
    if runs.empty:
        print("RQ2: no full GTZAN runs here, skipping (significance_rq2.* left untouched)")
        return False
    lines += ["## RQ2: GTZAN random vs. fault-filtered split", "", HEADER]
    res = {s: compare(f"GTZAN {s}", runs[runs["split"] == s], rng, n_boot, rows, lines)
           for s in ("random", "filtered")}
    lines.append("")
    if res["random"] and res["filtered"]:
        dr, df_ = res["random"]["d"], res["filtered"]["d"]
        diff = float(dr.mean() - df_.mean())
        boots = (dr[rng.integers(0, len(dr), size=(n_boot, len(dr)))].mean(1)
                 - df_[rng.integers(0, len(df_), size=(n_boot, len(df_)))].mean(1))
        lo, hi = ci(boots); p = boot_p(boots, n_boot)
        rows.append({"comparison": "GTZAN drop difference (spec drop - wave drop)", "seed": "pooled",
                     "gap": diff, "ci_low": lo, "ci_high": hi, "boot_p": p})
        lines += [f"**Drop difference:** the spectrogram CNN loses **{100 * diff:+.1f} pts more** than SampleCNN "
                  f"when moving to the fault-filtered split (= gap on random − gap on filtered; "
                  f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}], p = {fmt_p(p)}).", ""]
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=10000, help="number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="random seed for the bootstrap")
    ap.add_argument("--outdir", type=Path, default=REPO / "results")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    intro = ["# Significance tests", "",
             f"Accuracy on the test clips. Gap = spectrogram − SampleCNN (positive: spectrogram ahead). "
             f"CIs and p-values: {args.bootstrap:,} bootstrap resamples of test clips, seeds averaged; "
             f"McNemar: exact test per seed.", ""]
    args.outdir.mkdir(parents=True, exist_ok=True)
    for tag, fn in (("rq1", rq1), ("rq2", rq2)):
        rows, lines = [], list(intro)
        if fn(rng, args.bootstrap, rows, lines):
            (args.outdir / f"significance_{tag}.md").write_text("\n".join(lines) + "\n")
            pd.DataFrame(rows).to_csv(args.outdir / f"significance_{tag}.csv", index=False, float_format="%.5f")
            print("\n".join(lines[len(intro):]))
            print(f"wrote {args.outdir / f'significance_{tag}.md'} and .csv\n")

if __name__ == "__main__":
    main()
