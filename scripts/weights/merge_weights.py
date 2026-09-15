from .local_weights import PROJECT_ROOT, local_weight, load_lisa_model

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

sys.path.insert(0, str(PROJECT_ROOT.parent / "LISA"))
from model.LISA import LISAForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()

    model_path = local_weight("lisa")
    vision_tower = local_weight("clip")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, model_max_length=2048,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model = load_lisa_model(
        LISAForCausalLM, model_path, vision_tower,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        train_mask_decoder=False, out_dim=256,
        seg_token_idx=tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
        use_mm_start_end=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch.float16)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    # Match the LoRA modules and hyperparameters in fine_tune.py.
    target_modules = sorted(
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and not any(part in name for part in (
            "visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"))
        and any(part in name for part in ("q_proj", "v_proj", "k_proj", "o_proj"))
    )
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=target_modules,
        bias="none", task_type="CAUSAL_LM",
    ))
    model.resize_token_embeddings(len(tokenizer))
    # The full training checkpoint already contains the SAM parameters.
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model = model.merge_and_unload()
    state_dict = {
        key: value for key, value in model.state_dict().items()
        if "vision_tower" not in key
    }
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
