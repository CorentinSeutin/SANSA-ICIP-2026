#!/usr/bin/env python3

import argparse, json, random

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS_ROOT = PROJECT_ROOT / "data" / "descriptions"


def sample_per_category(objects, n):
    objects_by_category = {}
    for obj in objects:
        category = obj["category"]
        objects_by_category.setdefault(category, []).append(obj)

    selected_objects = []
    for category_objects in objects_by_category.values():
        selected_objects.extend(random.sample(category_objects, min(n, len(category_objects))))

    random.shuffle(selected_objects)
    return selected_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "val"))
    parser.add_argument("n", type=int)
    args = parser.parse_args()

    split = args.split
    input_path = DESCRIPTIONS_ROOT / f"{split}_objects.json"
    output_path = DESCRIPTIONS_ROOT / f"{args.split}_objects_sampled.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.is_file():
        print(f"[INFO][{split.upper()}][SKIP] Output already exists at: {output_path}")
        return
    print(f"[INFO][{split.upper()}] Starting sample_objects.py with n={args.n}")

    if not input_path.is_file():
        raise FileNotFoundError(f"[INFO][{split.upper()}][ERROR] Input dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    selected_objects = sample_per_category(objects, args.n)
    
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(selected_objects, file, indent=2, ensure_ascii=False)

    print(f"[INFO][{split.upper()}] Saved {len(selected_objects)} objects in: {output_path}")
    print(f"[INFO][{split.upper()}] Finished sample_objects.py")

if __name__ == "__main__":
    main()
