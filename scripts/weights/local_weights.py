import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from huggingface_hub import snapshot_download

from .config import MODEL_IDS, WEIGHTS_ROOT, SAM_FILENAME


def local_weight(name):
    if name == "sam":
        path = WEIGHTS_ROOT / SAM_FILENAME
        return str(path)

    return snapshot_download(repo_id=MODEL_IDS[name], cache_dir=WEIGHTS_ROOT / "hub", local_files_only=True)


def load_lisa_model(model_class, model_path, vision_tower, **kwargs):
    config = model_class.config_class.from_pretrained(model_path, local_files_only=True)
    config.mm_vision_tower = vision_tower
    config.vision_tower = vision_tower

    return model_class.from_pretrained(
        model_path, config=config, vision_tower=vision_tower,
        local_files_only=True, **kwargs,
    )
