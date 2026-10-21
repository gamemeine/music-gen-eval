import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich import box
from rich.console import Console
from rich.table import Table
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from music_gen_eval.utils import select_device

from .nets import LossCurveCNN
from .train import load_dataset, predict, read_split


def metrics(y, p):
    valid = len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0

    return {
        "n": len(y),
        "mse": float(np.mean((y - p) ** 2)),
        "mae": float(np.mean(np.abs(y - p))),
        "pearson": float(pearsonr(y, p).statistic) if valid else None,
        "spearman": float(spearmanr(y, p).statistic) if valid else None,
    }


def evaluate(checkpoint_path, dataset_dir, feature_dir, output_dir, device_name="auto", systems=(), show_table=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    variant = checkpoint["config"].get("variant", checkpoint.get("signals", "loss_entropy").replace("_", "-"))

    rows = read_split(dataset_dir, "test")
    if systems:
        print(f"Filtering test clips to {len(systems)} systems: {', '.join(systems)}")
        rows = [row for row in rows if row["system_id"] in systems]
    if not rows:
        raise ValueError("No test clips match the selected systems")

    print(f"test: {len(rows)} clips.", file=sys.stderr)
    device = select_device(device_name)
    dataset = load_dataset(rows, feature_dir, variant=variant)

    channels = 9 if variant == "loss-entropy" else 5
    model = LossCurveCNN(input_channels=channels, sequence_length=1500).to(device)
    model.load_state_dict(checkpoint["model_state"])
    targets, predictions = predict(model, DataLoader(dataset, batch_size=64), device)
    report = {
        "experiment": checkpoint_path.parent.name,
        "variant": variant,
        "systems": sorted({row["system_id"] for row in rows}),
        **metrics(targets, predictions),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)[["clip_id", "score"]]
    results["prediction"] = predictions
    results.to_csv(output_dir / "predictions.csv", index=False)

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    if show_table:
        print_results([report])

    return report


def print_results(reports, table_format="terminal"):
    rows = [
        [report["experiment"]]
        + [f"{report[key]:.4f}" if report[key] is not None else "—" for key in ("pearson", "spearman", "mse", "mae")]
        for report in reports
    ]

    if table_format == "md":
        table = Table("Experiment", "r ↑", "ρ ↑", "MSE ↓", "MAE ↓", box=box.MARKDOWN)
        for row in rows:
            table.add_row(*row)
        Console(force_terminal=False).print(table)
        return

    names = Table("Experiment", box=None)
    scores = Table("r ↑", "ρ ↑", "MSE ↓", "MAE ↓", box=None)

    for name, *values in rows:
        names.add_row(name)
        scores.add_row(*values)

    table = Table(box=box.SIMPLE)
    table.add_column("")
    table.add_column("MusicEval", justify="center")
    table.add_row(names, scores)
    Console().print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "--checkpoints", type=Path, nargs="+",
        default=[Path("outputs/meva/repro/loss-entropy/best.pt")],
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/MusicEval-full"))
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/meva/features"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--format", choices=["terminal", "md"], default="terminal", help="Output table format.")
    parser.add_argument(
        "--systems", nargs="*", default=[], metavar="SYSTEM_ID",
        help="Evaluate only these system IDs, e.g. S004 S029. Default: all test systems.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    reports = []
    for checkpoint in args.checkpoint:
        output = args.output_dir or checkpoint.parent / "evaluation"
        if args.output_dir and len(args.checkpoint) > 1:
            output = output / checkpoint.parent.name
        reports.append(
            evaluate(checkpoint, args.dataset_dir, args.feature_dir, output, args.device, args.systems, show_table=False)
        )

    print_results(reports, args.format)
