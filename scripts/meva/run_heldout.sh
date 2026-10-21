#!/bin/bash
set -euo pipefail

# Parameters
VARIANTS=(
    "loss"
    "entropy"
    "loss-entropy"
)
EPOCHS=100
BATCH_SIZE=64
SEED=42

HELDOUT_SYSTEMS=(
    S006
    S023
    S015
    S008
    S019
    S002
)

OUTPUT_DIR=outputs/meva/heldout

# Training
CHECKPOINTS=()
for VARIANT in "${VARIANTS[@]}"; do
    uv run -m music_gen_eval.meva.train \
        --dataset-dir data/MusicEval-full \
        --feature-dir outputs/meva/features \
        --output-dir ${OUTPUT_DIR}/${VARIANT} \
        --variant ${VARIANT} \
        --exclude-systems ${HELDOUT_SYSTEMS[@]} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --seed ${SEED} \
        --device auto \
        "$@"

    CHECKPOINTS+=("${OUTPUT_DIR}/${VARIANT}/best.pt")
done

# Evaluation
uv run python -m music_gen_eval.meva.evaluate \
    --checkpoints ${CHECKPOINTS[@]} \
    --dataset-dir data/MusicEval-full \
    --feature-dir outputs/meva/features \
    --systems "${HELDOUT_SYSTEMS[@]}" \
    --device auto \
    --format md
