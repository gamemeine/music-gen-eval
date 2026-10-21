#!/bin/bash
set -euo pipefail

hf download BAAI/MusicEval MusicEval-full.zip \
    --repo-type dataset --local-dir data

unzip -o data/MusicEval-full.zip -d data
rm data/MusicEval-full.zip
