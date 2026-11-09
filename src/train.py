"""Train one model on one dataset/split/size/seed and record everything about the run.

Examples (from the repo root):
    python src/train.py --dataset gtzan --split filtered --model spectrogram_cnn --seed 0
    python src/train.py --dataset gtzan --split random   --model sample_cnn      --seed 1
    python src/train.py --dataset fma_small --train-size 800 --model sample_cnn --seed 2

Each run writes runs/<run_id>/ (best.pt, history.csv, metrics.json, predictions.csv)
and appends one row to results/runs.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import GENRES, RESULTS_CSV, RUNS_DIR, N_SLICES, SLICE, fma_labels, read_ids, split_files
from dataset import EvalClips, TrainCrops
from model import build_model, MODELS


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def predict(model, loader, device):
    """Clip-level probabilities: softmax averaged over each clip's 11 slices."""
    model.eval()
    probs, ys, idx = [], [], []
    for x, y, i in loader:                       # x: (B, 11, SLICE)
        b = x.shape[0]
        p = F.softmax(model(x.view(-1, SLICE).to(device)), dim=1).view(b, N_SLICES, -1).mean(1)
        probs.append(p.cpu()); ys += y.tolist(); idx += i.tolist()
    return torch.cat(probs).numpy(), np.array(ys), idx


def scores(probs, y, n_classes):
    pred = probs.argmax(1)
    return (accuracy_score(y, pred),
            f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0), pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["gtzan", "fma_small"], default="gtzan")
    ap.add_argument("--split", default=None, help="gtzan: random | filtered (default filtered); fma_small: official")
    ap.add_argument("--train-size", type=int, default=None, help="fma_small only: 800, 1600, 3200 or 6400")
    ap.add_argument("--model", choices=list(MODELS), default="spectrogram_cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs (early stopping usually ends sooner)")
    ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val macro-F1 gain")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crops-per-clip", type=int, default=8, help="random 2.7 s crops drawn per clip per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (try 4 on the cluster)")
    ap.add_argument("--quick", action="store_true", help="15 epochs, patience 5: for a fast first look")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.patience = 15, 5
    split = args.split or ("filtered" if args.dataset == "gtzan" else "official")

    set_seed(args.seed)
    device = pick_device()
    size_tag = f"n{args.train_size}" if args.train_size else "nall"
    run_id = f"{args.dataset}_{split}_{size_tag}_{args.model}_s{args.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}\ndevice={device}")

    # ---- data ----
    files = split_files(args.dataset, split, args.train_size)
    labels = fma_labels() if args.dataset == "fma_small" else None
    genres = GENRES[args.dataset]
    ids = {p: read_ids(f) for p, f in files.items()}
    train_ds = TrainCrops(args.dataset, ids["train"], labels, args.crops_per_clip)
    valid_ds = EvalClips(args.dataset, ids["valid"], labels)
    test_ds = EvalClips(args.dataset, ids["test"], labels)
    print(f"clips: train {len(train_ds.ids)}, valid {len(valid_ds)}, test {len(test_ds)}")
    gen = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    eval_kw = dict(batch_size=8, shuffle=False, num_workers=args.num_workers)
    valid_dl, test_dl = DataLoader(valid_ds, **eval_kw), DataLoader(test_ds, **eval_kw)

    # ---- model ----
    model = build_model(args.model, len(genres)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    # ---- train ----
    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0); n += y.size(0)
        probs, y_val, _ = predict(model, valid_dl, device)
        va_acc, va_f1, _ = scores(probs, y_val, len(genres))
        sched.step(va_f1)
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch:3d}  loss {total / n:.3f}  val acc {va_acc:.3f}  val F1 {va_f1:.3f}  "
              f"({(time.time() - t0) / 60:.1f} min)")
        if va_f1 > best_f1:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "args": vars(args), "genres": genres, "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break
    minutes = (time.time() - t0) / 60

    # ---- test with the best model ----
    model.load_state_dict(best_state)
    probs, y_test, idx = predict(model, test_dl, device)
    te_acc, te_f1, pred = scores(probs, y_test, len(genres))
    report = classification_report(y_test, pred, labels=list(range(len(genres))),
                                   target_names=genres, digits=3, zero_division=0)
    print(f"\nTEST  acc {te_acc:.3f}  macro-F1 {te_f1:.3f}  (best epoch {best_epoch})\n{report}")

    # ---- save everything ----
    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    with open(run_dir / "predictions.csv", "w", newline="") as f:     # per-clip, for paired tests
        w = csv.writer(f)
        w.writerow(["clip_id", "true", "pred"] + [f"p_{g}" for g in genres])
        for k, i in enumerate(idx):
            w.writerow([test_ds.ids[i], genres[y_test[k]], genres[pred[k]]] + [f"{v:.5f}" for v in probs[k]])
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "args": vars(args), "split": split, "device": str(device),
        "n_params": n_params, "train_clips": len(train_ds.ids), "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "test_accuracy": te_acc, "test_macro_f1": te_f1,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(len(genres)))).tolist(),
        "classification_report": report, "git_commit": git_commit()}, indent=2))

    row = {"run_id": run_id, "model": args.model, "dataset": args.dataset, "split": split,
           "train_size": len(train_ds.ids), "seed": args.seed, "test_accuracy": round(te_acc, 4),
           "test_macro_f1": round(te_f1, 4), "best_epoch": best_epoch, "max_epochs": args.epochs,
           "train_minutes": round(minutes, 1), "device": str(device), "git_commit": git_commit(),
           "date": datetime.now().strftime("%Y-%m-%d")}
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        try:                                   # cluster jobs may finish at the same moment
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
    print(f"saved {run_dir.relative_to(RUNS_DIR.parent)} and a row in {RESULTS_CSV.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
