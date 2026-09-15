#!/usr/bin/env bash

set -euo pipefail

dataset_names=(DISP-wo-LLM DISP EXSP EXSP-LLMJ)
max_shard_size="40GB"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for dataset_name in "${dataset_names[@]}"; do
    checkpoint_dir="./runs/${dataset_name}/deepspeed_checkpoint"
    weights_dir="./runs/${dataset_name}/consolidated_weights"
    model_dir="./runs/${dataset_name}/model"

    python "${checkpoint_dir}/zero_to_fp32.py" \
        "${checkpoint_dir}" \
        "${weights_dir}" \
        --max_shard_size "${max_shard_size}"

    python -m scripts.weights.merge_weights \
        --save_path "${model_dir}" \
        --weight "${weights_dir}/pytorch_model.bin"
done
