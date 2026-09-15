import argparse, json, shutil

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
COCO_IMAGES_ROOT = PROJECT_ROOT / "coco" / "images"


def polygon_to_points(polygon):
    if polygon and isinstance(polygon[0], (list, tuple)):
        return [[int(round(point[0])), int(round(point[1]))] for point in polygon]
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("Polygon must contain at least three x/y pairs.")
    return [
        [int(round(polygon[index])), int(round(polygon[index + 1]))]
        for index in range(0, len(polygon), 2)
    ]


def get_input_paths(split, dataset_name):
    descriptions_dir = DATA_ROOT / "descriptions" / dataset_name

    if dataset_name == "HPD":
        if split != "test":
            raise ValueError(f"[INFO][{dataset_name}] HPD can only be prepared as the test split.")
        objects_path = DATA_ROOT / "descriptions" / "HPD_objects_sampled.json"
        images_path = COCO_IMAGES_ROOT / "val2017"
    elif split == "train":
        objects_path = descriptions_dir / "train_objects_prompts_80_filtered.json"
        images_path = COCO_IMAGES_ROOT / "train2017"
    elif split == "val":
        objects_path = descriptions_dir / "train_objects_prompts_20_filtered.json"
        images_path = COCO_IMAGES_ROOT / "train2017"
    else:
        objects_path = descriptions_dir / "val_objects_prompts_filtered.json"
        images_path = COCO_IMAGES_ROOT / "val2017"

    return objects_path, images_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "val", "test"))
    parser.add_argument("dataset_name")
    args = parser.parse_args()

    split = args.split
    dataset_name = args.dataset_name

    objects_path, images_path = get_input_paths(split, dataset_name)
    output_path = DATA_ROOT / "for_lisa" / dataset_name / split

    if output_path.is_dir() and any(output_path.iterdir()):
        print(f"[INFO][{split.upper()}][{dataset_name}][SKIP] Output directory already exists at: {output_path}")
        return
    print(f"[INFO][{split.upper()}][{dataset_name}] Starting prepare.py")

    if not objects_path.is_file():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Objects file not found: {objects_path}")
    if not images_path.is_dir():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] COCO images directory not found: {images_path}")

    with objects_path.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    if dataset_name == "HPD" and not any(
        str(obj.get("segmentation_prompt") or "").strip() for obj in objects
    ):
        raise ValueError(
            "HPD contains no manual segmentation prompts. Run "
            "the HPD annotation/export step first, or provide the final "
            "HPD_objects_sampled.json."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    valid_data_counter = 0
    for obj in objects:
        image_name = obj.get("img_name")
        polygon = obj.get("polygon")
        annotation_id = obj.get("annotation_id")
        prompt = obj.get("segmentation_prompt")

        if not image_name or not polygon or not annotation_id or not prompt:
            continue
        try:
            points = polygon_to_points(polygon)
        except (TypeError, ValueError) as error:
            raise ValueError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Invalid polygon for annotation {annotation_id}: {error}") from error

        img_path = images_path / image_name
        if not img_path.is_file():
            print(f"[INFO][{split.upper()}][{dataset_name}][SKIP] Image not found at: {img_path}")
            continue

        output_img_path = output_path / f"img_{valid_data_counter:06d}.jpg"
        shutil.copy2(img_path, output_img_path)

        annotation_path = output_img_path.with_suffix(".json")
        annotation = {
            "shapes": [
                {
                    "label": str(annotation_id),
                    "points": points,
                }
            ],
            "text": [prompt],
            "is_sentence": True,
        }

        with annotation_path.open("w", encoding="utf-8") as file:
            json.dump(annotation, file, indent=2)

        valid_data_counter += 1

    print(f"[INFO][{split.upper()}][{dataset_name}] Created [{valid_data_counter}/{len(objects)}] image/JSON pairs at: {output_path}")
    print(f"[INFO][{split.upper()}][{dataset_name}] Finished prepare.py")


if __name__ == "__main__":
    main()
