import argparse
import json

from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS_ROOT = PROJECT_ROOT / "data" / "descriptions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "val"))
    ap.add_argument("dataset_name")
    args = ap.parse_args()

    split = args.split
    dataset_name = args.dataset_name

    print(f"[INFO][{split.upper()}][{dataset_name}] Starting apply_LLMJ_filter.py")

    objects_path = DESCRIPTIONS_ROOT / dataset_name / f"{split}_objects_prompts.json"
    filtered_objects_path = DESCRIPTIONS_ROOT / dataset_name / f"{split}_objects_prompts_filtered.json"

    filtered_objects_path.parent.mkdir(parents=True, exist_ok=True)

    if filtered_objects_path.is_file():
        print(f"[INFO][{split.upper()}][{dataset_name}][SKIP] Output already exists at: {filtered_objects_path}")
        return
    print(f"[INFO][{split.upper()}][{dataset_name}] Starting apply_LLMJ_filter.py")
    
    if not objects_path.is_file():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Objects with prompts file not found: {objects_path}")

    with objects_path.open("r", encoding="utf-8") as f:
        objects = json.load(f)

    if "LLMJ" in dataset_name:
        print(f"[INFO][{split.upper()}][{dataset_name}] Applying LLMJ filtering method.")
        filtered_objects = []
        for obj in tqdm(objects, desc="Filtering with LLMJ method"):
            if "YES" not in obj.get("verification", ""):
                filtered_objects.append(obj)
    else:
        print(f"[INFO][{split.upper()}][{dataset_name}] No LLMJ filtering required.")
        filtered_objects = objects
    
    with filtered_objects_path.open("w", encoding="utf-8") as f:
        json.dump(filtered_objects, f, ensure_ascii=False, indent=2)

    removed_count = len(objects) - len(filtered_objects)
    print(f"[INFO][{split.upper()}][{dataset_name}] Saved [{len(filtered_objects)}/{len(objects)}] objects in: {filtered_objects_path}")
    print(f"[INFO][{split.upper()}][{dataset_name}] Removed [{removed_count}/{len(objects)}] of {len(objects)} objects")
    print(f"[INFO][{split.upper()}][{dataset_name}] Finished apply_LLMJ_filter.py")

if __name__ == "__main__":
    main()