#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p ./cache

mode="${1:-train}"
if [[ "$mode" == "validate" || "$mode" == "validation" ]]; then
    dataset_names=("${@:2}")
    if [[ ${#dataset_names[@]} -eq 0 ]]; then
        dataset_names=(DISP-wo-LLM DISP EXSP EXSP-LLMJ)
    fi
    for dataset_name in "${dataset_names[@]}"; do
        deepspeed --master_port=24999 --num_gpus 1 scripts/fine_tune.py \
            "$dataset_name" --validate-only
    done
    exit 0
fi

if [[ "$mode" != "train" ]]; then
    echo "Usage: $0 [train|validate] [dataset ...]" >&2
    exit 2
fi

dataset_names=(DISP-wo-LLM DISP EXSP EXSP-LLMJ)
for dataset_name in "${dataset_names[@]}"; do
    deepspeed --master_port=24999 --num_gpus 1 scripts/fine_tune.py "$dataset_name"
done
