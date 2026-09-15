#!/usr/bin/env bash

set -euo pipefail

model_names=(base DISP-wo-LLM DISP EXSP EXSP-LLMJ)
test_dataset_names=(Baseline DISP-wo-LLM DISP EXSP EXSP-LLMJ HPD)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p ./cache

for model_name in "${model_names[@]}"; do
    for test_dataset_name in "${test_dataset_names[@]}"; do
        python scripts/eval.py \
            "${model_name}" \
            "${test_dataset_name}"
    done
done
