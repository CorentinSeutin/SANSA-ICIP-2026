#!/usr/bin/env bash

set -euo pipefail

splits=(train val)
objects_per_category=(125 25)
dataset_names=(Baseline DISP-wo-LLM DISP EXSP EXSP-LLMJ) 
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p ./cache

for split in "${splits[@]}"; do
    python scripts/generate_cropped_objects.py "$split"
done

for index in "${!splits[@]}"; do
    split="${splits[$index]}"
    n="${objects_per_category[$index]}"
    python scripts/sample_objects.py "$split" "$n"
done

for dataset_name in "${dataset_names[@]}"; do
    for split in "${splits[@]}"; do
        python scripts/describe_objects.py "$split" "$dataset_name"
        python scripts/generate_sansa_prompts.py "$split" "$dataset_name"
        python scripts/apply_LLMJ_filter.py "$split" "$dataset_name"
    done

    python scripts/split_objects.py "$dataset_name" 80

    for split in train val test; do
        python scripts/prepare_data.py "$split" "$dataset_name"
    done
done

python scripts/prepare_data.py "test" "HPD"
