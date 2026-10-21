import argparse
import json
import random
import runpy
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from music_gen_eval.utils import select_device

from .nets import LossCurveCNN


def read_split(split_dir, split, exclude_systems=()):
    scores = pd.read_csv(split_dir / f"{split}.csv")
    rows = []

    for row in scores.itertuples(index=False):
        clip_id = Path(row.audio_path).stem
        system, prompt = clip_id.rsplit("-", 1)[1].split("_")

        if system in exclude_systems:
            continue

        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": row.audio_path,
                "system_id": system,
                "prompt_id": prompt,
                "score": float(row.score),
            }
        )

    return rows


def pack_curves(loss, entropy, length=1500, variant="loss-entropy"):
    curves = {"loss": loss, "entropy": entropy, "loss-entropy": np.concatenate((loss, entropy))}[variant]

    if curves.shape[1] > length:
        edges = np.linspace(0, curves.shape[1], length + 1).astype(int)
        curves = np.stack([curves[:, a:b].mean(1) for a, b in pairwise(edges)], axis=1)

    packed = np.zeros((curves.shape[0] + 1, length), dtype=np.float32)
    packed[:-1, : curves.shape[1]] = curves
    packed[-1, : curves.shape[1]] = 1

    return packed


def load_dataset(rows, feature_dir, length=1500, variant="loss-entropy"):
    inputs = []

    for row in rows:
        with np.load(feature_dir / f"{row['clip_id']}.npz", allow_pickle=False) as feature:
            loss, entropy = feature["loss"], feature["entropy"]
            if loss.shape != entropy.shape or loss.shape[0] != 4 or loss.shape[1] == 0:
                raise ValueError("Invalid loss/entropy shape")
            if not np.isfinite(loss).all() or not np.isfinite(entropy).all():
                raise ValueError("Nonfinite features")

            inputs.append(pack_curves(loss, entropy, length, variant))

    return TensorDataset(
        torch.from_numpy(np.stack(inputs)),
        torch.tensor([r["score"] for r in rows], dtype=torch.float32),
    )


@torch.inference_mode()
def extract_audio(model, waveform, device):
    path = Path(__file__).resolve().parents[2] / "external/meva/lib/repro/musicgen_pattern.py"
    aligned_logits = runpy.run_path(str(path))["delayed_aligned_logits"]
    losses, entropies = [], []

    for chunk in waveform.split(30 * 32000, dim=-1):
        if chunk.shape[-1] < 3200:
            continue

        codes = model.audio_encoder.encode(chunk.unsqueeze(0).to(device)).audio_codes.long()
        logits = aligned_logits(model, codes)[0].float()

        log_probs = logits.log_softmax(-1)
        loss = -log_probs.gather(-1, codes[0, 0, :, 1:].unsqueeze(-1)).squeeze(-1)

        probabilities = log_probs.exp()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)

        losses.append(loss.cpu().numpy())
        entropies.append(entropy.cpu().numpy())

    return np.concatenate(losses, axis=1), np.concatenate(entropies, axis=1)


def extract(
    rows,
    feature_dir,
    device_name="auto",
    model_name="facebook/musicgen-small",
    local_files_only=False,
):
    feature_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (row, feature_dir / f"{row['clip_id']}.npz")
        for row in rows
        if not (feature_dir / f"{row['clip_id']}.npz").exists()
    ]

    if not pending:
        print(f"Reusing {len(rows)} cached features.")
        return

    from transformers import MusicgenForConditionalGeneration
    from transformers.models.musicgen.modeling_musicgen import (
        MusicgenSinusoidalPositionalEmbedding,
    )

    device = select_device(device_name)
    model = (
        MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    for module in model.modules():
        if isinstance(module, MusicgenSinusoidalPositionalEmbedding):
            module.offset = 2

    for row, path in tqdm(pending, desc="Extracting signals"):
        samples, sample_rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())

        if sample_rate != 32000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)

        loss, entropy = extract_audio(model, waveform.mean(0, keepdim=True), device)
        np.savez_compressed(
            path,
            loss=loss,
            entropy=entropy,
        )


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions, targets = [], []

    for inputs, labels in loader:
        predictions.extend(model(inputs.to(device)).squeeze(-1).cpu().tolist())
        targets.extend(labels.tolist())

    return np.array(targets), np.array(predictions)


def train(
    split_dir,
    feature_dir,
    output_dir,
    epochs=100,
    batch_size=64,
    seed=42,
    device_name="auto",
    variant="loss-entropy",
    exclude_systems=(),
    overwrite=False,
):
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Choose an empty run directory or use --overwrite: {output_dir}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)

    records = {split: read_split(split_dir, split, exclude_systems) for split in ("train", "val")}

    for split, rows in records.items():
        if not rows:
            raise ValueError(f"No clips remain in {split} after excluding systems")
        print(f"{split}: {len(rows)} clips")

    loaders = {
        s: DataLoader(
            load_dataset(rows, feature_dir, variant=variant),
            batch_size=batch_size,
            shuffle=(s == "train"),
            generator=torch.Generator().manual_seed(seed),
        )
        for s, rows in records.items()
    }

    channels = 9 if variant == "loss-entropy" else 5
    model = LossCurveCNN(input_channels=channels, sequence_length=1500).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)

    config = {
        "variant": variant,
        "exclude_systems": list(exclude_systems),
        "split_dir": str(split_dir.resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "sequence_length": 1500,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "patience": 10,
    }

    metadata = {
        "format_version": 2,
        "config": config,
        "signals": variant.replace("-", "_"),
        "target": "csv_score",
        "device": str(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in ("best.pt", "run.json", "steps.jsonl", "history.json"):
            (output_dir / name).unlink(missing_ok=True)

    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))

    best_loss, stale = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for inputs, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs.to(device)).squeeze(-1), targets.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)

        y, p = predict(model, loaders["val"], device)
        val_loss = float(np.mean((y - p) ** 2))
        correlation = float(pearsonr(y, p).statistic) if np.ptp(y) and np.ptp(p) else None

        step = {
            "epoch": epoch,
            "train_mse": total / len(loaders["train"].dataset),
            "val": {"mse": val_loss, "pearson": correlation},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        with (output_dir / "steps.jsonl").open("a") as file:
            file.write(json.dumps(step, allow_nan=False) + "\n")

        print(
            f"Epoch {epoch}: train_mse={step['train_mse']:.4f} val_mse={val_loss:.4f} val_pearson={correlation}",
            flush=True,
        )
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(
                {**metadata, "epoch": epoch, "model_state": model.state_dict()},
                output_dir / "best.pt",
            )
        else:
            stale += 1

        if stale >= 10:
            break

    return output_dir / "best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/musiceval"), help="Directory with train.csv, val.csv and test.csv containing score and audio_path.")
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/meva/features"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing training outputs and start from scratch.")
    parser.add_argument("--variant", choices=["loss", "entropy", "loss-entropy"], default="loss-entropy")
    parser.add_argument(
        "--exclude-systems", nargs="*", default=[], metavar="SYSTEM_ID",
        help="System IDs to exclude from training and validation, e.g. S004 S029. Test set is unchanged.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path("outputs/meva/repro") / args.variant

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error("Choose an empty --output-dir or use --overwrite")

    rows = [
        row
        for split in ("train", "val", "test")
        for row in read_split(args.split_dir, split, args.exclude_systems if split != "test" else ())
    ]
    extract(rows, args.feature_dir, args.device)

    print(
        train(
            args.split_dir,
            args.feature_dir,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.seed,
            args.device,
            args.variant,
            args.exclude_systems,
            args.overwrite,
        )
    )
