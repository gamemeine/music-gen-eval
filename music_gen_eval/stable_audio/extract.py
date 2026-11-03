"""Stable Audio 3 intrinsic RF loss curves, compatible with the existing MEva CNN."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from music_gen_eval.utils import select_device

MODELS = {"base": "small-music-base", "small": "small-music"}
NOISE_LEVELS = np.array([0.1, 0.5, 0.9], dtype=np.float32)


@torch.inference_mode()
def loss_curves(model, latents, conditioning, noise, valid_frames):
    """RF target is noise - clean latent. Neither signal is a token likelihood."""
    clean = latents.float()
    target = noise - clean
    dtype = next(model.model.parameters()).dtype
    mask = torch.arange(clean.shape[-1], device=clean.device)[None] < valid_frames
    losses, denoising = [], []
    for level in NOISE_LEVELS:
        t = float(level)
        noised = clean * (1 - t) + noise * t
        timestep = torch.tensor([t], device=clean.device, dtype=torch.float32)
        prediction = model.model(
            noised.to(dtype),
            timestep,
            **conditioning,
            padding_mask=mask,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
        ).float()
        losses.append((prediction - target).square().mean(1)[0, :valid_frames].cpu().numpy())
        reconstructed = noised - t * prediction
        denoising.append((reconstructed - clean).square().mean(1)[0, :valid_frames].cpu().numpy())
    return np.stack(losses), np.stack(denoising)


@torch.inference_mode()
def extract_audio(generator, waveform, sample_rate, device, seed, empty_prompt):
    model = generator.model
    ratio = model.pretransform.downsampling_ratio
    all_loss, all_denoising = [], []
    for index, chunk in enumerate(waveform.split(30 * sample_rate, dim=-1)):
        torch.manual_seed(seed + index)
        seconds = chunk.shape[-1] / sample_rate
        # Use upstream resampling, stereo conversion, alignment and SAME encoding.
        size = generator._adapt_sample_size(
            [{"seconds_total": seconds}],
            generator.model_config["sample_size"],
            duration_padding_sec=0,
        )
        latents, _ = generator._encode_audio_input((sample_rate, chunk), size)
        valid = min(latents.shape[-1], math.ceil(seconds * model.sample_rate / ratio))
        tensors = {
            "prompt": empty_prompt,
            "seconds_total": model.conditioner.conditioners["seconds_total"]([seconds], "cpu"),
            "inpaint_mask": [torch.zeros(1, 1, latents.shape[-1])],
            "inpaint_masked_input": [torch.zeros(1, model.io_channels, latents.shape[-1])],
        }
        conditioning = {
            key: value.to(device) if value is not None else None
            for key, value in model.get_conditioning_inputs(tensors).items()
        }
        rng = torch.Generator(device="cpu").manual_seed(seed + index)
        noise = torch.randn(latents.shape, generator=rng, dtype=torch.float32).to(device)
        loss, denoising = loss_curves(model, latents, conditioning, noise, valid)
        all_loss.append(loss)
        all_denoising.append(denoising)
    return np.concatenate(all_loss, axis=1), np.concatenate(all_denoising, axis=1)


def extract(rows, feature_dir, device_name="auto", model_name="base", seed=42):
    feature_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"stabilityai/stable-audio-3-{MODELS[model_name]}"
    pending = []
    for row in rows:
        path = feature_dir / f"{row['clip_id']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as feature:
                if (
                    str(feature.get("model_id", "")) != model_id
                    or not np.array_equal(feature.get("noise_levels"), NOISE_LEVELS)
                    or int(feature.get("seed", -1)) != seed
                ):
                    raise ValueError(f"Different model/settings in {path}; choose a separate --feature-dir")
        else:
            pending.append((row, path))
    if not pending:
        print(f"Reusing {len(rows)} cached Stable Audio features.")
        return

    from stable_audio_3 import StableAudioModel

    device = select_device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    # Load on CPU first, keeping the text encoder there to fit a 6 GB GPU.
    generator = StableAudioModel.from_pretrained(MODELS[model_name], device="cpu", model_half=False)
    generator.device = str(device)
    model = generator.model
    if model.diffusion_objective not in ("rectified_flow", "rf_denoiser"):
        raise ValueError(f"Unsupported objective: {model.diffusion_objective}")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.model.to(device=device, dtype=dtype)
    model.pretransform.to(device=device, dtype=dtype)
    empty_prompt = model.conditioner.conditioners["prompt"]([""], "cpu")
    # Like the MusicGen extractor, use audio without its actual text prompt.
    for row, path in tqdm(pending, desc=f"Extracting Stable Audio {model_name}"):
        samples, sample_rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        clip_seed = seed + int(hashlib.sha256(row["clip_id"].encode()).hexdigest()[:8], 16)
        loss, denoising = extract_audio(generator, waveform, sample_rate, device, clip_seed, empty_prompt)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            loss=loss,
            denoising=denoising,
            noise_levels=NOISE_LEVELS,
            model_id=np.array(model_id),
            seed=np.array(seed),
            signal=np.array("rf_velocity_mse_proxy"),
        )
        temporary.replace(path)


if __name__ == "__main__":
    from music_gen_eval.meva.train import read_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits/musiceval"))
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    rows = [row for split in ("train", "val", "test") for row in read_split(args.split_dir, split)]
    extract(rows, args.feature_dir or Path("outputs/stable_audio") / args.model / "features", args.device, args.model)
