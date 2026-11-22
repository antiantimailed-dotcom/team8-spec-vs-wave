"""RQ3: where do the two models make different mistakes, and what does SampleCNN learn?

    python scripts/rq3_analysis.py                    # per-genre analysis (no GPU needed)
    python scripts/rq3_analysis.py --filters          # also probe SampleCNN's learned filters (needs torch + best.pt)

Part 1, per genre (from runs/<run_id>/predictions.csv), for each setting:
    gtzan_filtered   GTZAN, fault-filtered split
    fma_6400         FMA-small, 6,400 training clips
  - confusion matrices averaged over seeds (rows = true genre, normalized to 1)
  - per-genre F1, mean ± SD over seeds, and spectrogram - SampleCNN difference
  - each model's most frequent confusions
  Writes results/rq3_<setting>.md, results/rq3_per_genre_<setting>.csv,
         figures/rq3_confusion_<setting>.png, figures/rq3_genre_f1_<setting>.png

Part 2, --filters: plays pure sine tones (30 Hz to 10 kHz) through each trained SampleCNN
and records which frequency every channel of the first layers responds to most, counting
only channels that are clearly tuned to one frequency range. The same probe on untrained
SampleCNNs (random weights) is the control. A learned, mel-like layout shows up as far more
tuned channels than the control, with about 30% of them below 1 kHz (mel) rather than the
9% of a linear scale or the ~60% a random test tone would give.
  Writes results/rq3_filter_peaks_<setting>.csv, figures/rq3_filter_response_<setting>.png,
         figures/rq3_filter_peaks_<setting>.png (setting = --filter-setting, default: first of --settings)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from results_io import REPO, load_predictions, load_runs  # noqa: E402

SPEC, WAVE = "spectrogram_cnn", "sample_cnn"
LABEL = {SPEC: "Spectrogram CNN", WAVE: "SampleCNN"}
SETTINGS = {
    "gtzan_filtered": dict(dataset="gtzan", split="filtered", size=None, title="GTZAN fault-filtered"),
    "gtzan_random": dict(dataset="gtzan", split="random", size=None, title="GTZAN random split"),
    "fma_6400": dict(dataset="fma_small", split="official", size=6400, title="FMA-small, 6,400 clips"),
    "fma_800": dict(dataset="fma_small", split="official", size=800, title="FMA-small, 800 clips"),
}
RESULTS, FIGURES = REPO / "results", REPO / "figures"


# ---------------- Part 1: per-genre errors ----------------
def select(setting: dict, model: str) -> pd.DataFrame:
    runs = load_runs(setting["dataset"], setting["split"], models=[model])
    if setting["size"] is None:
        return runs[runs["size"].isna()]
    return runs[runs["size"] == setting["size"]]


def genre_stats(runs: pd.DataFrame):
    mats, f1s, genres = [], [], None
    for _, r in runs.iterrows():
        p = load_predictions(r["run_dir"])
        genres = genres or [c[2:] for c in p.columns if c.startswith("p_")]
        cm = confusion_matrix(p["true"], p["pred"], labels=genres).astype(float)
        mats.append(cm / np.maximum(cm.sum(1, keepdims=True), 1))
        f1s.append(f1_score(p["true"], p["pred"], labels=genres, average=None, zero_division=0))
    return genres, np.stack(mats), np.stack(f1s)


def per_genre(name: str, setting: dict, lines: list):
    out = {}
    for m in (SPEC, WAVE):
        runs = select(setting, m)
        if runs.empty:
            print(f"  {name}: no full {m} runs found, skipping this setting")
            return
        out[m] = (runs["seed"].tolist(),) + genre_stats(runs)
    genres = out[SPEC][1]
    if out[WAVE][1] != genres:
        raise SystemExit(f"{name}: the two models' prediction files list genres in a different order")
    (s_seeds, _, s_cm, s_f1), (w_seeds, _, w_cm, w_f1) = out[SPEC], out[WAVE]
    sd = lambda a: a.std(0, ddof=1) if len(a) > 1 else np.full(a.shape[1], np.nan)  # noqa: E731
    tab = pd.DataFrame({
        "genre": genres,
        "spec_f1_mean": s_f1.mean(0), "spec_f1_sd": sd(s_f1),
        "wave_f1_mean": w_f1.mean(0), "wave_f1_sd": sd(w_f1),
        "spec_recall": np.diag(s_cm.mean(0)), "wave_recall": np.diag(w_cm.mean(0)),
    })
    tab["f1_diff"] = tab["spec_f1_mean"] - tab["wave_f1_mean"]
    common = sorted(set(s_seeds) & set(w_seeds))
    tab["seeds_spec_better"] = [
        sum(s_f1[s_seeds.index(s), g] > w_f1[w_seeds.index(s), g] for s in common) for g in range(len(genres))]
    tab = tab.sort_values("f1_diff", ascending=False)
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    tab.to_csv(RESULTS / f"rq3_per_genre_{name}.csv", index=False, float_format="%.4f")

    lines += [f"## {setting['title']} ({len(s_seeds)} spectrogram / {len(w_seeds)} SampleCNN seeds)", "",
              f"| Genre | Spectrogram F1 | SampleCNN F1 | Difference | Seeds spectrogram better (of {len(common)}) |",
              "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        lines.append(f"| {r.genre} | {r.spec_f1_mean:.2f} ± {r.spec_f1_sd:.2f} | {r.wave_f1_mean:.2f} ± {r.wave_f1_sd:.2f} "
                     f"| {r.f1_diff:+.2f} | {r.seeds_spec_better} |")
    lines.append("")
    for m, cm in ((SPEC, s_cm), (WAVE, w_cm)):
        mean = cm.mean(0).copy(); np.fill_diagonal(mean, 0)
        top = np.dstack(np.unravel_index(np.argsort(mean, axis=None)[::-1][:5], mean.shape))[0]
        lines.append(f"**{LABEL[m]}, most frequent confusions:** " + "; ".join(
            f"{genres[i]} → {genres[j]} ({100 * mean[i, j]:.0f}%)" for i, j in top))
    lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, m, cm in ((axes[0], SPEC, s_cm), (axes[1], WAVE, w_cm)):
        mean = cm.mean(0)
        ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(genres)), genres, rotation=45, ha="right")
        ax.set_yticks(range(len(genres)), genres)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{LABEL[m]} (mean over {len(cm)} seeds)")
        for i in range(len(genres)):
            for j in range(len(genres)):
                if mean[i, j] >= 0.05:
                    ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mean[i, j] > 0.55 else "#14213D")
    fig.suptitle(f"Confusion matrices, {setting['title']} (rows sum to 1)")
    fig.tight_layout(); fig.savefig(FIGURES / f"rq3_confusion_{name}.png", dpi=150); plt.close(fig)

    t = tab.iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(t) + 1.5))
    ax.barh(y + 0.2, t["spec_f1_mean"], 0.4, xerr=t["spec_f1_sd"].fillna(0), label=LABEL[SPEC], color="#2F6FB0", capsize=3)
    ax.barh(y - 0.2, t["wave_f1_mean"], 0.4, xerr=t["wave_f1_sd"].fillna(0), label=LABEL[WAVE], color="#E07A2F", capsize=3)
    ax.set_yticks(y, t["genre"]); ax.set_xlim(0, 1); ax.set_xlabel("test F1 (mean ± SD over seeds)")
    ax.set_title(f"Per-genre F1, {setting['title']}\n(sorted: biggest spectrogram advantage at top)", fontsize=10)
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / f"rq3_genre_f1_{name}.png", dpi=150); plt.close(fig)
    print(f"  {name}: wrote rq3_per_genre_{name}.csv and two figures")


# ---------------- Part 2: SampleCNN filter frequencies ----------------
def hz_to_mel(f):   # Slaney mel scale (what the spectrogram model uses)
    f = np.asarray(f, float)
    return np.where(f < 1000, f / (200 / 3), 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27))


def mel_to_hz(m):
    m = np.asarray(m, float)
    return np.where(m < 15, m * (200 / 3), 1000 * np.exp((m - 15) * np.log(6.4) / 27))


TUNED_RATIO = 2.0   # a channel counts as frequency-tuned if its best tone gets >= 2x its median response


def tone_responses(model, tones, n_layers: int, device):
    """Mean activation of every channel to every tone: one (channels, tones) array per layer."""
    import torch
    resp = [[] for _ in range(n_layers)]
    with torch.no_grad():
        for b in range(0, len(tones), 16):
            x = model.stem(tones[b:b + 16].to(device).unsqueeze(1))
            for L in range(n_layers):                     # layer 0 = stem, then blocks 1..n-1
                if L > 0:
                    x = model.blocks[L - 1](x)
                resp[L].append(x.mean(-1).cpu())          # mean over time: (batch, channels)
    return [torch.cat(r).numpy().T for r in resp]


def channel_table(R_layers, freqs, state: str, seed, run_id: str) -> list[dict]:
    rows = []
    for L, R in enumerate(R_layers):
        peak, med = R.max(1), np.median(R, 1)
        for c in range(R.shape[0]):
            active = bool(peak[c] > 1e-6)
            rows.append({"state": state, "run_id": run_id, "seed": seed, "layer": L, "channel": c,
                         "active": active, "tuned": bool(active and peak[c] >= TUNED_RATIO * max(med[c], 1e-9)),
                         "peak_hz": float(freqs[R[c].argmax()]) if active else np.nan})
    return rows


def probe_filters(name: str, n_layers: int, lines: list):
    setting = SETTINGS[name]
    import torch
    from common import N_MELS, SAMPLE_RATE, SLICE
    from model import build_model

    runs = select(setting, WAVE)
    if runs.empty:
        print("  --filters: no full SampleCNN runs for this setting"); return
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    freqs = np.geomspace(30, 10000, 160)
    t = np.arange(SLICE) / SAMPLE_RATE
    tones = torch.from_numpy((0.1 * np.sin(2 * np.pi * freqs[:, None] * t[None, :])).astype(np.float32))
    rows, examples = [], {}
    for i, (_, r) in enumerate(runs.iterrows()):
        ckpt = torch.load(r["run_dir"] / "best.pt", map_location="cpu", weights_only=False)
        n_classes = len(ckpt["genres"])
        trained = build_model(WAVE, n_classes); trained.load_state_dict(ckpt["model"])
        torch.manual_seed(1000 + i)                       # control: same architecture, random weights
        untrained = build_model(WAVE, n_classes)
        for state, m in (("trained", trained), ("untrained", untrained)):
            R = tone_responses(m.to(device).eval(), tones, n_layers, device)
            rows += channel_table(R, freqs, state, r["seed"], r["run_id"])
            examples.setdefault(state, (r["seed"], R))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"rq3_filter_peaks_{name}.csv", index=False, float_format="%.1f")

    mel_centers = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)[1:-1])
    ref_mel = np.mean(mel_centers < 1000)
    ref_lin = 1000 / (SAMPLE_RATE / 2)
    ref_grid = np.mean(freqs < 1000)
    lines += [f"## SampleCNN filter frequencies (pure-tone probe), {setting['title']}", "",
              f"Each channel's preferred frequency = the test tone it responds to most. Only **tuned** channels "
              f"count (best tone ≥ {TUNED_RATIO:g}× the channel's median response). Control: the same probe on "
              f"untrained SampleCNNs with random weights.", "",
              f"Reference shares below 1 kHz: mel scale ({N_MELS} bands) {100 * ref_mel:.0f}%, linear scale "
              f"{100 * ref_lin:.0f}%. A channel with no real preference lands on a random test tone, and "
              f"{100 * ref_grid:.0f}% of the tones are below 1 kHz.", "",
              "| Layer | Trained: tuned channels | Trained: median preferred | Trained: below 1 kHz "
              "| Untrained: tuned channels | Untrained: median preferred | Untrained: below 1 kHz |",
              "|---|---|---|---|---|---|---|"]
    for L in sorted(df.layer.unique()):
        cells = []
        for state in ("trained", "untrained"):
            g = df[(df.layer == L) & (df.state == state)]
            tg = g[g.tuned]
            cells += [f"{len(tg)} of {len(g)} ({100 * len(tg) / max(len(g), 1):.0f}%)",
                      f"{tg.peak_hz.median():,.0f} Hz" if len(tg) else "—",
                      f"{100 * np.mean(tg.peak_hz < 1000):.0f}%" if len(tg) else "—"]
        lines.append(f"| {'stem' if L == 0 else f'block {L}'} | " + " | ".join(cells) + " |")
    lines += ["", "Read it this way: a learned, mel-like frequency layout shows up as **many more tuned channels "
              "than the untrained control**, with a share below 1 kHz near the mel reference rather than the "
              "random-tone level.", ""]

    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8), squeeze=False)
    for row, state in enumerate(("trained", "untrained")):
        seed, Rs = examples[state]
        for L, ax in enumerate(axes[row]):
            R = Rs[L]; keep = R.max(1) > 1e-6
            if not keep.any():
                ax.set_axis_off(); continue
            Rn = R[keep] / R[keep].max(1, keepdims=True)
            order = np.argsort(Rn.argmax(1))
            ax.imshow(Rn[order], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                      extent=[np.log10(freqs[0]), np.log10(freqs[-1]), 0, len(order)])
            ticks = [100, 300, 1000, 3000, 10000]
            ax.set_xticks(np.log10(ticks), [f"{f:g}" if f < 1000 else f"{f // 1000:g}k" for f in ticks])
            ax.set_title(f"{state}: {'stem' if L == 0 else f'block {L}'}")
            if row == 1:
                ax.set_xlabel("tone frequency (Hz)")
            if L == 0:
                ax.set_ylabel("channel (sorted by preferred frequency)")
    fig.suptitle(f"SampleCNN channel responses to pure tones, {setting['title']}\n"
                 "normalized per channel (top: trained, bottom: random weights)")
    fig.tight_layout(); fig.savefig(FIGURES / f"rq3_filter_response_{name}.png", dpi=150); plt.close(fig)

    deepest = int(df.layer.max())
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bins = np.geomspace(30, 10000, 25)
    for state, col in (("trained", "#E07A2F"), ("untrained", "#8A8F98")):
        v = df[(df.layer == deepest) & (df.state == state) & df.tuned].peak_hz
        if len(v):
            ax.hist(v, bins=bins, density=True, alpha=0.55, color=col, label=f"{state} SampleCNN, block {deepest} (tuned channels)")
    ax.hist(mel_centers, bins=bins, density=True, histtype="step", lw=2, color="#2F6FB0", label=f"{N_MELS} mel band centers")
    ax.set_xscale("log"); ax.set_xlabel("preferred frequency (Hz)"); ax.set_ylabel("density")
    ax.set_title(f"Preferred frequencies, {setting['title']}: trained vs. random weights vs. mel scale", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / f"rq3_filter_peaks_{name}.png", dpi=150); plt.close(fig)
    print(f"  filters: wrote rq3_filter_peaks_{name}.csv and two figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=["gtzan_filtered", "fma_6400"], choices=list(SETTINGS))
    ap.add_argument("--filters", action="store_true", help="also probe SampleCNN filters (needs torch and best.pt files)")
    ap.add_argument("--filter-setting", default=None, choices=list(SETTINGS),
                    help="which models to probe (default: the first of --settings)")
    ap.add_argument("--layers", type=int, default=4, help="stem + this many minus one blocks to probe")
    args = ap.parse_args()
    lines = ["# RQ3: per-genre errors and learned filters", ""]
    for name in args.settings:
        per_genre(name, SETTINGS[name], lines)
    if args.filters:
        probe_filters(args.filter_setting or args.settings[0], args.layers, lines)
    out = RESULTS / ("rq3_" + "_".join(args.settings) + ".md")
    RESULTS.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
