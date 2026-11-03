"""Train/evaluate SA3 loss features using the original CNN, split reader and metrics."""

import argparse
import json
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from music_gen_eval.meva.evaluate import metrics, print_results
from music_gen_eval.meva.nets import LossCurveCNN
from music_gen_eval.meva.train import predict, read_split
from music_gen_eval.utils import select_device

from .extract import MODELS, extract


def load_dataset(rows, feature_dir):
    inputs = []
    for row in rows:
        with np.load(feature_dir / f"{row['clip_id']}.npz", allow_pickle=False) as feature:
            loss = feature["loss"]
            if loss.ndim != 2 or min(loss.shape) == 0 or not np.isfinite(loss).all():
                raise ValueError(f"Invalid loss curves: {row['clip_id']}")
            # Same temporal averaging, zero padding and mask as MEva's pack_curves.
            if loss.shape[1] > 1500:
                edges = np.linspace(0, loss.shape[1], 1501).astype(int)
                loss = np.stack([loss[:, a:b].mean(1) for a, b in pairwise(edges)], axis=1)
            packed = np.zeros((loss.shape[0] + 1, 1500), dtype=np.float32)
            packed[:-1, : loss.shape[1]] = loss
            packed[-1, : loss.shape[1]] = 1
            inputs.append(packed)
    if not inputs:
        raise ValueError("No clips selected")
    return TensorDataset(
        torch.from_numpy(np.stack(inputs)), torch.tensor([row["score"] for row in rows], dtype=torch.float32)
    )


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Choose an empty --output-dir or use --overwrite")
    rows = {split: read_split(args.split_dir, split) for split in ("train", "val", "test")}
    extract([row for split in rows.values() for row in split], args.feature_dir, args.device, args.model)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    loaders = {
        split: DataLoader(
            load_dataset(rows[split], args.feature_dir),
            batch_size=args.batch_size,
            shuffle=split == "train",
            generator=torch.Generator().manual_seed(args.seed),
        )
        for split in ("train", "val")
    }
    model = LossCurveCNN(loaders["train"].dataset.tensors[0].shape[1], 1500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    metadata = {
        "format_version": 2,
        "signals": "loss",
        "target": "csv_score",
        "device": str(device),
        "config": {
            "variant": "loss",
            "model": args.model,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": 1500,
            "feature_dir": str(args.feature_dir.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "patience": 10,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in ("best.pt", "run.json", "steps.jsonl"):
            (args.output_dir / name).unlink(missing_ok=True)
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    print(f"train: {len(rows['train'])} clips; val: {len(rows['val'])} clips", flush=True)
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for inputs, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs.to(device)).squeeze(-1), targets.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        y, p = predict(model, loaders["val"], device)
        validation = metrics(y, p)
        step = {"epoch": epoch, "train_mse": total / len(loaders["train"].dataset), "val": validation}
        with (args.output_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(step, allow_nan=False) + "\n")
        print(
            f"Epoch {epoch}: train_mse={step['train_mse']:.4f} "
            f"val_mse={validation['mse']:.4f} val_pearson={validation['pearson']}",
            flush=True,
        )
        scheduler.step(validation["mse"])
        if validation["mse"] < best:
            best, stale = validation["mse"], 0
            torch.save({**metadata, "epoch": epoch, "model_state": model.state_dict()}, args.output_dir / "best.pt")
        else:
            stale += 1
        if stale >= 10:
            break
    print(args.output_dir / "best.pt", flush=True)


def evaluate(args):
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    rows = read_split(args.split_dir, "test")
    if args.systems:
        rows = [row for row in rows if row["system_id"] in args.systems]
    dataset = load_dataset(rows, args.feature_dir)
    device = select_device(args.device)
    model = LossCurveCNN(dataset.tensors[0].shape[1], 1500).to(device)
    model.load_state_dict(checkpoint["model_state"])
    y, p = predict(model, DataLoader(dataset, batch_size=64), device)
    report = {
        "experiment": args.output_dir.name,
        "variant": "loss",
        "split_dir": str(args.split_dir.resolve()),
        "systems": sorted({row["system_id"] for row in rows}),
        **metrics(y, p),
    }
    results = pd.DataFrame(rows)[["clip_id", "score"]]
    results["prediction"] = p
    output = args.output_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"test: {len(rows)} clips.", flush=True)
    print_results([report], args.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the existing best.pt without retraining")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/musiceval"))
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--systems", nargs="*", default=[])
    parser.add_argument("--format", choices=["terminal", "md"], default="terminal")
    args = parser.parse_args()
    args.feature_dir = args.feature_dir or Path("outputs/stable_audio") / args.model / "features"
    args.output_dir = args.output_dir or Path("outputs/stable_audio") / f"{args.model}-loss"
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
