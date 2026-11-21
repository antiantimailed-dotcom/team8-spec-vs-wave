"""Read finished runs back in for analysis (no torch needed).

Every run made by src/train.py has a row in results/runs.csv and a folder runs/<run_id>/
holding metrics.json, predictions.csv, history.csv and best.pt. These helpers pick the
runs an analysis should use:

- quick runs (--quick, 15 epochs) are dropped unless include_quick=True;
- if the same dataset / split / size / model / seed was run more than once, the last
  row in runs.csv wins (same rule as scripts/plot_rq1_gap.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO / "results" / "runs.csv"
RUNS_DIR = REPO / "runs"
QUICK_EPOCHS = 15


def nominal_size(run_id: str):
    """800 / 1600 / 3200 / 6400 from '..._n800_...', or None for full-split runs ('nall')."""
    m = re.search(r"_n(\d+)_", str(run_id))
    return int(m.group(1)) if m else None


def _is_quick(row) -> bool:
    mj = RUNS_DIR / row["run_id"] / "metrics.json"
    if mj.exists():
        try:
            return bool(json.loads(mj.read_text()).get("args", {}).get("quick", False))
        except (OSError, json.JSONDecodeError):
            pass
    return int(row.get("max_epochs", 0) or 0) == QUICK_EPOCHS


def load_runs(dataset: str | None = None, split: str | None = None, models=None,
              include_quick: bool = False, require_dir: bool = True, csv_path: Path = RESULTS_CSV) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise SystemExit(f"No results file at {csv_path}")
    df = pd.read_csv(csv_path)
    if dataset:
        df = df[df["dataset"] == dataset]
    if split:
        df = df[df["split"] == split]
    if models:
        df = df[df["model"].isin(models)]
    df = df.copy()
    df["size"] = [nominal_size(r) for r in df["run_id"]]
    df["seed"] = df["seed"].astype(int)
    if not include_quick:
        df = df[[not _is_quick(r) for _, r in df.iterrows()]]
    df["run_dir"] = [RUNS_DIR / r for r in df["run_id"]]
    df["has_dir"] = [d.is_dir() for d in df["run_dir"]]
    if require_dir:
        missing = df[~df["has_dir"]]
        if len(missing):
            print(f"note: {len(missing)} run(s) in runs.csv have no runs/ folder here and are skipped "
                  f"(e.g. {missing['run_id'].iloc[0]}). Copy the runs/ folders from whoever trained them.")
        df = df[df["has_dir"]]
    key = ["dataset", "split", "size", "model", "seed"]
    df = df.drop_duplicates(subset=key, keep="last") if len(df) else df
    return df.reset_index(drop=True)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Per-clip test predictions, indexed by clip_id, with a boolean 'correct' column."""
    p = pd.read_csv(Path(run_dir) / "predictions.csv", dtype={"clip_id": str})
    p["correct"] = p["true"] == p["pred"]
    return p.set_index("clip_id")


def load_metrics(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def describe(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 runs"
    parts = df.groupby(["dataset", "split", "model"], dropna=False).size()
    return f"{len(df)} runs: " + ", ".join(f"{d}/{s}/{m}={n}" for (d, s, m), n in parts.items())
