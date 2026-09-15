#!/usr/bin/env python3

import argparse, json, random

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS_ROOT = PROJECT_ROOT / "data" / "descriptions"


def sample_percentage(objects, percent):
    n = int(len(objects) * (percent / 100))
    indices = set(random.sample(range(len(objects)), n))
    
    selected_objects = [objects[i] for i in indices]
    remaining_objects = [objects[i] for i in range(len(objects)) if i not in indices]
    return selected_objects, remaining_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("n", type=int)
    args = parser.parse_args()

    dataset_name = args.dataset_name

    input_path = DESCRIPTIONS_ROOT / dataset_name / f"train_objects_prompts_filtered.json"

    output_path1 = DESCRIPTIONS_ROOT / dataset_name / f"train_objects_prompts_{args.n}_filtered.json"
    output_path2 = DESCRIPTIONS_ROOT / dataset_name / f"train_objects_prompts_{100 - args.n}_filtered.json"

    output_path1.parent.mkdir(parents=True, exist_ok=True)
    output_path2.parent.mkdir(parents=True, exist_ok=True)

    if output_path1.is_file() and output_path2.is_file():
        print(f"[INFO][{dataset_name}][SKIP] Both split outputs already exist at: {output_path1} and {output_path2}")
        return
    print(f"[INFO][{dataset_name}] Starting split_objects.py with n={args.n}")

    if not input_path.is_file():
        raise FileNotFoundError(f"[INFO][{dataset_name}][ERROR] Input dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    selected_objects1, selected_objects2 = sample_percentage(objects, args.n)

    with output_path1.open("w", encoding="utf-8") as file:
        json.dump(selected_objects1, file, indent=2, ensure_ascii=False)

    with output_path2.open("w", encoding="utf-8") as file:
        json.dump(selected_objects2, file, indent=2, ensure_ascii=False)

    print(f"[INFO][{dataset_name}] Saved {len(selected_objects1)} objects in: {output_path1}")
    print(f"[INFO][{dataset_name}] Saved {len(selected_objects2)} objects in: {output_path2}")
    print(f"[INFO][{dataset_name}] Finished split_objects.py")

if __name__ == "__main__":
    main()
