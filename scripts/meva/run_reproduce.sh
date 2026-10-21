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

OUTPUT_DIR=outputs/meva/repro

# Training
CHECKPOINTS=()
for VARIANT in "${VARIANTS[@]}"; do
    uv run -m music_gen_eval.meva.train \
        --split-dir data/splits/musiceval \
        --feature-dir outputs/meva/features \
        --output-dir ${OUTPUT_DIR}/${VARIANT} \
        --variant ${VARIANT} \
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
    --split-dir data/splits/musiceval \
    --feature-dir outputs/meva/features \
    --device auto \
    --format md
