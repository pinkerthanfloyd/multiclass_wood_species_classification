"""
generate_prediction_coco_json.py

Runs a trained YOLO segmentation model over the FULL dataset and writes a
COCO-format JSON containing all predicted annotations.  The output mirrors
the structure of instances_selected.json so it can be consumed directly by
build_relative_vectors_from_coco_json.py, make_splits.py, or any downstream
pipeline that expects COCO instance segmentation format.

Image collection modes:
    1. With --split-dir: reads train.csv + val.csv + test.csv (original behaviour).
    2. Without --split-dir (default): recursively scans --dataset-root for all
       image files (jpg/jpeg/png/tif/tiff/bmp).  Species is inferred from the
       filename and _split is set to "all".

The output JSON is the input for the second-stage species classifier:
    image -> detector -> COCO annotations -> feature vectors -> species model

Usage:
    # Scan all images under dataset root (no CSVs needed):
    python generate_prediction_coco_json.py \
        --model runs_wood_feature_sem_loss_asym/split1/asym_bs4_img1024/weights/best.pt \
        [--dataset-root data/raw_selected] \
        [--data-yaml data/raw_selected/data.yaml] \
        [--output predicted_instances.json]

    # Use split CSVs (original behaviour):
    python generate_prediction_coco_json.py \
        --model runs_wood_feature_sem_loss_asym/split1/asym_bs4_img1024/weights/best.pt \
        --split-dir data/raw_selected/split1 \
        [--dataset-root data/raw_selected] \
        [--data-yaml data/raw_selected/data.yaml] \
        [--output predicted_instances.json]

Output COCO JSON schema (matches instances_selected.json):
    {
        "info": { ... },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "8",     "supercategory": "wood_feature"},
            {"id": 2, "name": "56",    "supercategory": "wood_feature"},
            ...
            {"id": 14, "name": "radio", "supercategory": "wood_feature"}
        ],
        "images": [
            {"id": 1, "file_name": "0003-1.jpg", "width": 1500, "height": 1500},
            ...
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 13,         # 1-indexed (YOLO class 12 + 1)
                "segmentation": [[x1,y1,x2,y2,...,xn,yn]],
                "bbox": [x, y, w, h],      # absolute pixels, COCO format
                "area": 1234.5,             # polygon area in pixels
                "iscrowd": 0,
                "score": 0.87              # prediction confidence (extra field)
            },
            ...
        ]
    }

Notes:
    - category_id is 1-indexed (COCO convention): YOLO class 0 -> category_id 1.
      This matches coco_to_yolo_seg.py which does class_id = category_id - 1.
    - image_id is a sequential integer starting at 1.
    - annotation_id is a sequential integer starting at 1.
    - "score" is an extra field not in standard COCO GT but useful for filtering.
      Downstream consumers that don't expect it will simply ignore it.
    - bbox is computed from the segmentation polygon (tight bounding box).
    - area is computed via the shoelace formula on the polygon vertices.
    - Images with zero predictions still appear in the "images" list (they just
      have no annotations pointing to them), matching the instances_selected.json
      convention.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_yolo_seg_baseline_csvs import (
    IMAGE_REF_COLUMNS,
    LABEL_REF_COLUMNS,
    SPECIES_COLUMNS,
    TASK_NAME_COLUMNS,
    first_existing_column,
    parse_task_name,
    resolve_image_path,
    load_class_names,
    get_yolo_class,
)


# ============================================================
# HELPERS
# ============================================================
def parse_species_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def polygon_area_shoelace(points: np.ndarray) -> float:
    """Compute area of a polygon using the shoelace formula.
    points: (N, 2) array of (x, y) coordinates in pixel space."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_bbox_coco(points: np.ndarray) -> List[float]:
    """Compute COCO-format bbox [x, y, width, height] from polygon vertices."""
    if len(points) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def load_split_csv_images(csv_path: Path, dataset_root: Path) -> List[dict]:
    """Load image records from a split CSV."""
    import pandas as pd

    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    image_col = first_existing_column(columns, IMAGE_REF_COLUMNS)
    species_col = first_existing_column(columns, SPECIES_COLUMNS)
    task_name_col = first_existing_column(columns, TASK_NAME_COLUMNS)

    if image_col is None:
        print(f"WARNING: {csv_path} has no image column. Skipping.")
        return []

    records = []
    for _, row in df.iterrows():
        try:
            image_path = resolve_image_path(str(row[image_col]), dataset_root)
        except FileNotFoundError:
            continue

        task_name = str(row[task_name_col]).strip() if task_name_col else ""
        _, species_from_task = parse_task_name(task_name)
        species_name = str(row[species_col]).strip() if species_col else ""
        if not species_name or species_name.lower() == "nan":
            species_name = species_from_task or parse_species_from_filename(image_path.name)

        records.append({
            "image_path": str(image_path),
            "image_name": image_path.name,
            "species": species_name,
        })
    return records


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_records_from_directory(dataset_root: Path) -> List[dict]:
    """Recursively scan *dataset_root* for image files and return records.

    No CSVs are needed.  Species is inferred from the filename and every
    record gets ``_split = "all"``.
    """
    all_records: List[dict] = []
    for p in sorted(dataset_root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        all_records.append({
            "image_path": str(p),
            "image_name": p.name,
            "species": parse_species_from_filename(p.name),
            "split": "all",
        })
    return all_records


def collect_all_records(dataset_root: Path, split_dir: Path) -> List[dict]:
    """Collect image records from train.csv + val.csv + test.csv, deduplicated."""
    all_records = []
    seen_paths = set()

    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        records = load_split_csv_images(csv_path, dataset_root)
        for rec in records:
            key = rec["image_path"]
            if key not in seen_paths:
                seen_paths.add(key)
                rec["split"] = split_name
                all_records.append(rec)

    return all_records


def build_categories(class_names: List[str]) -> List[dict]:
    """Build COCO categories list.  1-indexed to match instances_selected.json."""
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            "id": i + 1,           # 1-indexed
            "name": name,
            "supercategory": "wood_feature",
        })
    return categories


def _extract_annotations_from_result(result, image_id: int, ann_id_start: int) -> Tuple[List[dict], int]:
    """Extract COCO annotations from a single YOLO result.

    Returns (annotations_list, next_ann_id).  Moves tensors to CPU and
    converts to Python scalars immediately so the GPU result can be freed.
    """
    annotations: List[dict] = []
    ann_id = ann_id_start

    if result.boxes is None or len(result.boxes) == 0:
        return annotations, ann_id
    if result.masks is None or result.masks.xy is None:
        return annotations, ann_id

    # Pull everything to CPU / numpy in one shot, then drop the result ref
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy().astype(float)
    # .xy returns a list of numpy arrays (one per instance) already on CPU
    polygons = result.masks.xy

    for class_id, confidence, points in zip(class_ids, confidences, polygons):
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 3:
            continue

        flat_coords = []
        for x, y in points:
            flat_coords.append(round(float(x), 2))
            flat_coords.append(round(float(y), 2))

        bbox = polygon_bbox_coco(points)
        area = polygon_area_shoelace(points)
        category_id = int(class_id) + 1

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [flat_coords],
            "bbox": [round(b, 2) for b in bbox],
            "area": round(area, 2),
            "iscrowd": 0,
            "score": round(float(confidence), 4),
        })
        ann_id += 1

    return annotations, ann_id


def run_inference_to_coco(
    model_path: str,
    records: List[dict],
    class_names: List[str],
    imgsz: int,
    conf: float,
    iou_nms: float,
    device: int,
    chunk_size: int = 50,
) -> dict:
    """Run YOLO inference on all images and build a COCO JSON dict.

    Processes images in chunks of `chunk_size` to keep memory bounded.
    Between chunks the GPU cache and Python garbage collector are flushed.
    retina_masks is OFF -- the model-resolution masks are sufficient for
    polygon extraction and use ~4x less VRAM.
    """

    import torch

    YOLO = get_yolo_class()
    model = YOLO(model_path)

    categories = build_categories(class_names)

    images_list: List[dict] = []
    annotations_list: List[dict] = []
    ann_id_counter = 1

    # Build image entries.  Results are matched by POSITION (index in the
    # records list), not by filename, because Ultralytics renames images to
    # image0.jpg, image1.jpg, ... when source is a Python list.
    for idx, rec in enumerate(records):
        image_id = idx + 1
        images_list.append({
            "id": image_id,
            "file_name": rec["image_name"],
            "width": 0,
            "height": 0,
            "_species": rec["species"],
            "_split": rec.get("split", "unknown"),
        })

    image_paths = [rec["image_path"] for rec in records]
    n_total = len(image_paths)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(
        f"Running inference on {n_total} images in {n_chunks} chunks "
        f"of {chunk_size} (conf={conf}, imgsz={imgsz}, retina_masks=False)..."
    )

    n_processed = 0
    n_annotations = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_paths = image_paths[start:end]

        # Run inference with stream=True so each result is yielded and freed
        results = model.predict(
            source=chunk_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            max_det=300,
            retina_masks=False,
            device=device,
            verbose=False,
            stream=True,
        )

        # IMPORTANT: Ultralytics renames images to image0.jpg, image1.jpg, ...
        # when source is a Python list.  Results stream in the SAME ORDER as
        # the input list, so we track by position (start + local_idx) instead
        # of trying to match filenames.
        for local_idx, result in enumerate(results):
            global_idx = start + local_idx
            image_id = global_idx + 1  # our image IDs are 1-indexed

            # Fill in image dimensions
            h, w = result.orig_shape
            images_list[image_id - 1]["width"] = int(w)
            images_list[image_id - 1]["height"] = int(h)

            # Extract annotations (CPU-side, small python dicts)
            anns, ann_id_counter = _extract_annotations_from_result(
                result, image_id, ann_id_counter
            )
            annotations_list.extend(anns)
            n_annotations += len(anns)

            n_processed += 1

        # Free GPU memory between chunks
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(
            f"  chunk {chunk_idx + 1}/{n_chunks} done -- "
            f"{n_processed}/{n_total} images, {n_annotations} annotations"
        )

    print(f"Inference complete: {n_processed} images, {n_annotations} annotations")

    # Build final COCO dict
    coco_json = {
        "info": {
            "description": "Predicted wood anatomical feature annotations",
            "version": "1.0",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": str(model_path),
            "confidence_threshold": conf,
            "iou_nms_threshold": iou_nms,
            "image_size": imgsz,
        },
        "licenses": [],
        "categories": categories,
        "images": images_list,
        "annotations": annotations_list,
    }

    return coco_json


def print_summary(coco_json: dict, class_names: List[str]):
    """Print a summary of the generated COCO JSON."""
    n_images = len(coco_json["images"])
    n_anns = len(coco_json["annotations"])

    # Per-category counts
    cat_counts = {}
    for ann in coco_json["annotations"]:
        cid = ann["category_id"]
        cat_counts[cid] = cat_counts.get(cid, 0) + 1

    # Per-split counts
    split_counts = {}
    for img in coco_json["images"]:
        sp = img.get("_split", "unknown")
        split_counts[sp] = split_counts.get(sp, 0) + 1

    # Images with zero annotations
    image_ids_with_anns = {ann["image_id"] for ann in coco_json["annotations"]}
    n_empty = sum(1 for img in coco_json["images"] if img["id"] not in image_ids_with_anns)

    print()
    print("=" * 60)
    print("COCO JSON Summary")
    print("=" * 60)
    print(f"Total images      : {n_images}")
    print(f"Total annotations : {n_anns}")
    print(f"Images with 0 pred: {n_empty}")
    print()
    print("Per split:")
    for sp in ("train", "val", "test", "unknown"):
        if sp in split_counts:
            print(f"  {sp:>8}: {split_counts[sp]} images")
    print()
    print("Per category:")
    for i, name in enumerate(class_names):
        cid = i + 1
        count = cat_counts.get(cid, 0)
        print(f"  {cid:2d} ({name:>8}): {count:6d} annotations")
    print()

    # Average annotations per image
    if n_images > 0:
        print(f"Avg annotations/image: {n_anns / n_images:.1f}")

    # Confidence statistics
    if n_anns > 0:
        scores = [ann["score"] for ann in coco_json["annotations"]]
        print(f"Confidence range    : [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"Confidence mean     : {sum(scores) / len(scores):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO JSON from YOLO model predictions over the full dataset"
    )
    parser.add_argument("--model", required=True,
                        help="Path to .pt model weights")
    parser.add_argument("--dataset-root", default="data/raw_selected",
                        help="Dataset root directory (where images/ lives)")
    parser.add_argument("--data-yaml", default="data/raw_selected/data.yaml",
                        help="Path to data.yaml for class names")
    parser.add_argument("--split-dir", default=None,
                        help="Directory containing train.csv, val.csv, test.csv. "
                             "If omitted, all images under --dataset-root are used.")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou-nms", type=float, default=0.7,
                        help="NMS IoU threshold")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device (use -1 for CPU)")
    parser.add_argument("--output", type=str, default="predicted_instances.json",
                        help="Output COCO JSON path")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Images per inference chunk (lower = less RAM, default 50)")
    parser.add_argument("--strip-meta", action="store_true",
                        help="Remove _species and _split metadata from image entries")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    split_dir = Path(args.split_dir) if args.split_dir is not None else None
    data_yaml = Path(args.data_yaml)
    output_path = Path(args.output)

    # Load class names
    class_names = load_class_names(data_yaml)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Collect image records
    if split_dir is not None:
        records = collect_all_records(dataset_root, split_dir)
        if not records:
            print("ERROR: No image records found. Check --split-dir and --dataset-root.")
            sys.exit(1)
        print(f"Collected {len(records)} unique images across all splits")
    else:
        records = collect_records_from_directory(dataset_root)
        if not records:
            print(f"ERROR: No image files found under {dataset_root}.")
            sys.exit(1)
        print(f"Collected {len(records)} images from {dataset_root} (no split CSVs)")

    # Run inference and build COCO JSON
    device = args.device if args.device >= 0 else "cpu"
    coco_json = run_inference_to_coco(
        model_path=args.model,
        records=records,
        class_names=class_names,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_nms=args.iou_nms,
        device=device,
        chunk_size=args.chunk_size,
    )

    # Optionally strip internal metadata
    if args.strip_meta:
        for img in coco_json["images"]:
            img.pop("_species", None)
            img.pop("_split", None)

    # Print summary
    print_summary(coco_json, class_names)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_path}  ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
