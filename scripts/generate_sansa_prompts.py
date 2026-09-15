from weights.local_weights import local_weight

import argparse, json, re, sys, torch

from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
BATCH_SIZE = 32
MAX_NEW_TOKENS = 64

# Dictionary-based Segmentation Prompt (DISP) Instruction Prompt (IP)
DISP_IP =  """
                Instruction: Convert the input into a single, concise segmentation command.
                Start with "Segment ..." and use only the given words.
            """.strip()


def build_seg_prompt(obj_description, baseline=False, use_mistral=False, tokenizer=None):
    if use_mistral:
        obj_description = re.sub(r"\s+", " ", str(obj_description)).strip()

        messages = [
            {"role": "system", "content": "You reformulate visual descriptions into precise segmentation prompts."},
            {"role": "user", "content": f"{DISP_IP}\n\nInput: \"{obj_description}\"\nOutput:"},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif baseline:
        return obj_description
    else:
        prompt = f"Segment: {str(obj_description)}"
        return prompt
    

class PromptDataset(Dataset):
    def __init__(self, objects=None, baseline=False, use_mistral=False, tokenizer=None):
        self.indices = []
        self.descriptions = []
        self.use_mistral = use_mistral
        self.tokenizer = tokenizer
        self.baseline = baseline

        for idx, obj in enumerate(objects):
            self.indices.append(idx)
            self.descriptions.append(obj["description"])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return {
            "index": self.indices[idx],
            "prompt": build_seg_prompt(self.descriptions[idx], baseline=self.baseline, use_mistral=self.use_mistral, tokenizer=self.tokenizer),
        }


def load_tokenizer_and_pipe(device):
    model_path = local_weight("mistral")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        padding_side="left",
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
    )

    return tokenizer, pipe


def clean_seg_prompt(seg_prompt):
    """Clean a generated sentence and turn it into a segmentation command."""
    # Remove whitespace
    cleaned_prompt = seg_prompt.strip()

    # Remove quotation marks
    cleaned_prompt = cleaned_prompt.strip('"')
    cleaned_prompt = cleaned_prompt.strip()

    # Read the first word without punctuation
    first_word = cleaned_prompt.split(maxsplit=1)[0] if cleaned_prompt else ""
    first_word = first_word.strip(".:;,-—!?").lower()

    # Add "Segment" when the model returned only an object description
    if first_word != "segment":
        description = cleaned_prompt.lstrip(".:;,-—!? ")
        cleaned_prompt = f"Segment {description}"

    # Replace newlines, tabs, and repeated spaces with single spaces
    cleaned_prompt = " ".join(cleaned_prompt.split())

    return cleaned_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=("train", "val"))
    p.add_argument("dataset_name")
    args = p.parse_args()

    split = args.split
    dataset_name = args.dataset_name

    if "Baseline" in dataset_name:
        baseline = True
    else:
        baseline = False

    if "Baseline" in dataset_name or "wo-LLM" in dataset_name or "EXSP" in dataset_name:
        use_mistral = False
        print(f"[INFO][{split.upper()}][{dataset_name}] The LLM will not be used.")
    else:
        use_mistral = True
        print(f"[INFO][{split.upper()}][{dataset_name}] The LLM will be used.")

    objects_path = DATA_ROOT / "descriptions" / dataset_name / f"{split}_objects_described.json"
    objects_w_seg_prompt_path = DATA_ROOT / "descriptions" / dataset_name / f"{split}_objects_prompts.json"

    objects_w_seg_prompt_path.parent.mkdir(parents=True, exist_ok=True)

    if objects_w_seg_prompt_path.is_file():
        print(f"[INFO][{split.upper()}][{dataset_name}][SKIP] Output already exists at: {objects_w_seg_prompt_path}")
        return
    print(f"[INFO][{split.upper()}][{dataset_name}] Starting generate_sansa_prompts.py")
    
    if not objects_path.is_file():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Described objects file not found: {objects_path}")

    with open(objects_path, "r", encoding="utf-8") as file:
        objects = json.load(file)

    if use_mistral:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run the Mistral model; no CUDA GPU is available.")
        device = torch.device("cuda", torch.cuda.current_device())
        print(f"[INFO][{split.upper()}][{dataset_name}][DEVICE] GPU: {torch.cuda.get_device_name(device)}")
        tokenizer, pipe = load_tokenizer_and_pipe(device)
    else:
        tokenizer = None; pipe = None
    
    dataset = PromptDataset(objects=objects, baseline=baseline, use_mistral=use_mistral, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for batch in tqdm(dataloader, desc=f"[INFO][{split.upper()}][{dataset_name}] Generating SANSA Segmentation Prompts with the previously generated descriptions"):
        if use_mistral:
            outputs = pipe(
                batch["prompt"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            normalized = []
            for out in outputs if isinstance(outputs, list) else [outputs]:
                if isinstance(out, list):
                    out = out[0] if out else {}
                normalized.append(out)

            for i, out in enumerate(normalized):
                seg_prompt = out.get("generated_text", "")
                seg_prompt = clean_seg_prompt(seg_prompt)
                obj_index = batch["index"][i].item()
                objects[obj_index]["segmentation_prompt"] = seg_prompt
        else:
            for i in range(len(batch["prompt"])):
                obj_index = batch["index"][i].item()
                objects[obj_index]["segmentation_prompt"] = batch["prompt"][i]

    with open(objects_w_seg_prompt_path, "w", encoding="utf-8") as file:
        json.dump(objects, file, ensure_ascii=False, indent=2)

    print(f"[INFO][{split.upper()}][{dataset_name}] Saved {len(objects)} objects with SANSA segmentation prompts in: {objects_w_seg_prompt_path}")
    print(f"[INFO][{split.upper()}][{dataset_name}] Finished generate_sansa_prompts.py")

if __name__ == "__main__":
    sys.exit(main())
