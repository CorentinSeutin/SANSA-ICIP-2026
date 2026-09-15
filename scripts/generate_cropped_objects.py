import argparse, json, cv2

import numpy as np

from tqdm import tqdm
from pathlib import Path

from pycocotools import mask as mask_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COCO_ROOT = PROJECT_ROOT / "coco"
DATA_ROOT = PROJECT_ROOT / "data"

MIN_FRAC = 0.005

BACKGROUND = "black"


def segmentation_to_mask(segmentation, h, w):
    """Decode a COCO polygon or RLE segmentation into a uint8 mask."""
    if isinstance(segmentation, list):
        # COCO polygons are converted to RLE and merged by pycocotools.
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
    elif isinstance(segmentation, dict):
        size = segmentation.get("size")
        counts = segmentation.get("counts")

        if isinstance(counts, list):
            # Convert uncompressed RLE to pycocotools' compressed form.
            rle = mask_utils.frPyObjects(segmentation, h, w)
        elif isinstance(counts, str):
            # JSON represents the compressed RLE bytes as a string.
            rle = {"size": size, "counts": counts.encode("ascii")}
        elif isinstance(counts, bytes):
            rle = segmentation
        else:
            raise ValueError("RLE 'counts' must be a list, string, or bytes.")
    else:
        raise ValueError("COCO segmentation must be a polygon list or an RLE object.")

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8) * 255


def has_one_CC(mask):
    comp_count, _ = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return (comp_count - 1) == 1


def get_mask_bbox(mask, h, w):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, 0, w - 1, h - 1
    
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    return x1, y1, x2, y2


def save_masked_obj(img, mask, obj_path):
    img = img.astype(np.float32)

    h, w = mask.shape

    # Smooth edges: Gaussian blur on alpha mask
    mask = (mask.astype(np.float32) / 255.0)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.clip(mask, 0, 1)
    mask = mask[:, :, None]

    background = np.full((h, w, 3), (0, 0, 0), dtype=np.uint8)
    background = background.astype(np.float32)

    out_img = (mask * img + (1 - mask) * background).astype(np.uint8)

    cv2.imwrite(str(obj_path), out_img)


def mask_to_polygon(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None

    return points.astype(int).ravel().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "val"))
    args = ap.parse_args()

    split = args.split

    annotations_path = COCO_ROOT / "annotations" / f"instances_{split}2017.json"
    imgs_path = COCO_ROOT / "images" / f"{split}2017"
    
    masked_obj_path = DATA_ROOT / "obj-cropped" / split
    objects_out_path = DATA_ROOT / "descriptions" / f"{split}_objects.json"

    masked_obj_path.mkdir(parents=True, exist_ok=True)
    objects_out_path.parent.mkdir(parents=True, exist_ok=True)

    if masked_obj_path.exists() and objects_out_path.exists():
        print(f"[INFO][{split.upper()}][SKIP] Output already exists at: {masked_obj_path} and {objects_out_path}")
        return
    print(f"[INFO][{split.upper()}] Starting generate_cropped_objects.py")

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations_file = json.load(f)

    images = annotations_file.get("images", [])
    annotations = annotations_file.get("annotations", [])
    categories = annotations_file.get("categories", [])

    annots_by_img_id = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        annots_by_img_id.setdefault(img_id, []).append(ann)

    id_to_category = {c.get("id"): c.get("name") for c in categories}

    results = []
    for img in tqdm(images):
        img_id = img["id"]
        fname = img["file_name"]
        img_name = Path(fname).stem

        img_path = imgs_path / fname
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h = int(img.shape[0])
        w = int(img.shape[1])

        annotations = annots_by_img_id.get(img_id, [])
        if not annotations:
            continue

        for ann in annotations:
            polygon = ann.get("segmentation") # E.g. : [[50,30,170,30,170,230,50,230]] = [[x1,y1,x2,y2,x3,y3,x4,y4]]
            mask = segmentation_to_mask(polygon, h, w)
            if mask.max() == 0:
                continue

            area = int(np.count_nonzero(mask))
            frac = area / float(h * w)

            if frac < MIN_FRAC or not has_one_CC(mask):
                continue

            x1, y1, x2, y2 = get_mask_bbox(mask, h, w)

            crop_img = img[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]

            ann_id = ann.get("id")
            cat_id = ann.get("category_id")
            cat_name = id_to_category.get(cat_id, str(cat_id))

            obj_filename = f"{img_name}-{ann_id}.png"
            obj_path = masked_obj_path / obj_filename

            save_masked_obj(crop_img, crop_mask, obj_path)

            if isinstance(polygon, list) and len(polygon) == 1:
                arr = np.asarray(polygon[0], dtype=float).reshape(-1, 2)
                arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
                arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)
                polygon = [int(round(v)) for xy in arr for v in xy]
            else:
                polygon = mask_to_polygon(mask)
                if polygon is None:
                    continue

            results.append(
                {
                    "img_name": Path(fname).name,
                    "obj_filename": obj_filename,
                    "annotation_id": ann_id,
                    "category": cat_name,
                    "category_id": cat_id,
                    "polygon": polygon,
                    "description": None,
                    "frac_area": frac,
                }
            )

    with objects_out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO][{split.upper()}] Saved [{len(results)}/{len(annotations)}] cropped objects in: {objects_out_path}")
    print(f"[INFO][{split.upper()}] Finished generate_cropped_objects.py")

if __name__ == "__main__":
    main()
