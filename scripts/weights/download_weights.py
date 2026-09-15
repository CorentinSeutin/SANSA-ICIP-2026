"""Download all pretrained weights used by SANSA."""

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download
from torch.hub import download_url_to_file
from .config import MODEL_IDS, SAM_FILENAME, SAM_URL, WEIGHTS_ROOT


def main():
    print("[INFO] Starting download_weights.py", flush=True)
    for name, model_id in MODEL_IDS.items():
        print(f"[INFO] Preparing {model_id}", flush=True)

        snapshot_download(
            repo_id=model_id,
            cache_dir=WEIGHTS_ROOT / "hub",
            max_workers=1,
            allow_patterns=["*.json", "*.py", "*.model", "*.txt",
                            "pytorch_model*.bin" if name == "lisa" else "model*.safetensors"],
        )
    print(f"[INFO] Hugging Face models ready in {WEIGHTS_ROOT / 'hub'}")

    path = WEIGHTS_ROOT / SAM_FILENAME
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        download_url_to_file(SAM_URL, str(path))

    print(f"[INFO] SAM checkpoint ready at {path}")
    print("[INFO] Finished download_weights.py", flush=True)


if __name__ == "__main__":
    main()
