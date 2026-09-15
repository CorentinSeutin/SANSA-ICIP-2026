#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

# scripts/weights/download_weights.py downloads InternVL2.5-8B,
# Mistral-7B-Instruct-v0.3, LISA-7B-v1, CLIP ViT-L/14 and SAM ViT-H.
python3 -m scripts.weights.download_weights
