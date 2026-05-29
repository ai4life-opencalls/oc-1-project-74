"""
prepare_dataset.py — Merge per-set COCO annotations into train/val splits.

Two classes are created:
  - plant (category_id=1 → model class 0): convex hull of all leaf polygons per image
  - leaf  (category_id=2 → model class 1): individual leaf instance polygons

This matches the prediction notebook's convention: label==0 → plant, label==1 → leaf.

The split is done by set (not by image) so all time-series frames from the
same plant stay in the same partition.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --val-fraction 0.2 --seed 42
    python prepare_dataset.py --dataset-root /path/to/data --out-dir ./data
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

DATASET_ROOT = os.environ.get("OC1_DATASET_ROOT", "./dataset")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CATEGORIES = [
    {"id": 1, "name": "plant", "supercategory": "plant"},
    {"id": 2, "name": "leaf",  "supercategory": "plant"},
]


def make_plant_annotation(leaf_anns: list, image_id: int, ann_id: int) -> dict | None:
    """Build a single plant annotation as the convex hull of all leaf polygons."""
    all_points = []
    for ann in leaf_anns:
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            all_points.append(pts)

    if not all_points:
        return None

    all_points = np.concatenate(all_points, axis=0)
    hull = cv2.convexHull(all_points)          # (N, 1, 2)
    hull_pts = hull.squeeze(1)                 # (N, 2)

    seg = hull_pts.flatten().tolist()
    x, y = hull_pts[:, 0], hull_pts[:, 1]
    x_min, y_min, x_max, y_max = float(x.min()), float(y.min()), float(x.max()), float(y.max())

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,                      # plant
        "segmentation": [seg],
        "area": float(cv2.contourArea(hull)),
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "iscrowd": 0,
    }


def merge_sets(
    set_names: list[str],
    dataset_root: str,
    img_id_start: int = 0,
    ann_id_start: int = 0,
) -> tuple[dict, int, int]:
    images = []
    annotations = []
    img_id = img_id_start
    ann_id = ann_id_start

    for set_name in set_names:
        ann_path = os.path.join(dataset_root, set_name, "annotations.json")
        with open(ann_path) as f:
            data = json.load(f)

        local_id_map: dict[int, int] = {}

        for img in data["images"]:
            local_id_map[img["id"]] = img_id
            images.append(
                {
                    **img,
                    "id": img_id,
                    # prepend set dir so image_root=DATASET_ROOT resolves correctly
                    "file_name": os.path.join(set_name, img["file_name"]),
                }
            )
            img_id += 1

        # Group existing (leaf) annotations by image
        by_image: dict[int, list] = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)

        for old_img_id, leaf_anns in by_image.items():
            new_img_id = local_id_map[old_img_id]

            # Plant annotation (convex hull of all leaves)
            plant_ann = make_plant_annotation(leaf_anns, new_img_id, ann_id)
            if plant_ann is not None:
                annotations.append(plant_ann)
                ann_id += 1

            # Individual leaf annotations
            for ann in leaf_anns:
                annotations.append(
                    {
                        **ann,
                        "id": ann_id,
                        "image_id": new_img_id,
                        "category_id": 2,       # leaf (was 0 in original files)
                    }
                )
                ann_id += 1

    merged = {
        "info": {"description": "OC1 leaf instance segmentation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    return merged, img_id, ann_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sets = sorted(
        s for s in os.listdir(args.dataset_root)
        if os.path.isdir(os.path.join(args.dataset_root, s))
        and os.path.exists(os.path.join(args.dataset_root, s, "annotations.json"))
    )
    print(f"Found {len(sets)} sets.")

    rng = random.Random(args.seed)
    shuffled = sets[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_fraction))
    val_sets   = sorted(shuffled[:n_val])
    train_sets = sorted(shuffled[n_val:])

    print(f"Train sets ({len(train_sets)}): {train_sets}")
    print(f"Val   sets ({len(val_sets)}):   {val_sets}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_data, next_img, next_ann = merge_sets(train_sets, args.dataset_root)
    val_data, _, _ = merge_sets(val_sets, args.dataset_root, next_img, next_ann)

    for split, data in [("train", train_data), ("val", val_data)]:
        out_path = os.path.join(args.out_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(
            f"Wrote {out_path}: "
            f"{len(data['images'])} images, {len(data['annotations'])} annotations"
        )


if __name__ == "__main__":
    main()
