from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import GTZAN_GENRES, GTZANDataset
from model import build_model


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().tolist())
        all_p.extend(pred.cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    return total_loss / max(len(all_y), 1), acc, all_y, all_p


def main() -> None:
    p = argparse.ArgumentParser(description="Train GTZAN spectrogram vs waveform models")
    p.add_argument("--data-root", type=Path, default=Path("data/gtzan/genres"))
    p.add_argument(
        "--representation",
        choices=["spectrogram", "waveform"],
        default="spectrogram",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"device={device} representation={args.representation}")

    full = GTZANDataset(args.data_root, representation=args.representation)
    labels = [lab for _, lab in full.items]
    idx = list(range(len(full)))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.3, random_state=args.seed, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    train_ds = Subset(full, train_idx)
    val_ds = Subset(full, val_idx)
    test_ds = Subset(full, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = build_model(args.representation, num_classes=len(GTZAN_GENRES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_path = args.out_dir / f"best_{args.representation}.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = running / max(n, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "representation": args.representation,
                    "genres": GTZAN_GENRES,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(
        y_true, y_pred, target_names=GTZAN_GENRES, digits=3
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"\nbest_val_acc={best_val:.3f} test_acc={test_acc:.3f}")
    print(report)

    metrics = {
        "representation": args.representation,
        "device": str(device),
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "confusion_matrix": cm,
        "classification_report": report,
        "checkpoint": str(best_path),
    }
    metrics_path = args.out_dir / f"metrics_{args.representation}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {best_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
