import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = Path(os.environ.get("SANSA_WEIGHTS_DIR", PROJECT_ROOT / "weights")).expanduser().resolve()
MODEL_IDS = {
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "lisa": "xinlai/LISA-7B-v1",
    "clip": "openai/clip-vit-large-patch14",
}
SAM_FILENAME = "sam_vit_h_4b8939.pth"
SAM_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{SAM_FILENAME}"
