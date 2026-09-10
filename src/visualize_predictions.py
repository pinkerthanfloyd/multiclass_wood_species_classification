"""
visualize_predictions_limited.py

Versión simplificada:
- SOLO hace predicciones para las primeras 10 imágenes
- SIEMPRE guarda las imágenes generadas
- No hace sampling ni filtrado complejo
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import json

# ---------------------------------------------------------------------------
# Import shared helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_yolo_seg_baseline_csvs import (
    IMAGE_REF_COLUMNS,
    LABEL_REF_COLUMNS,
    SPECIES_COLUMNS,
    TASK_NAME_COLUMNS,
    first_existing_column,
    parse_task_name,
    resolve_image_path,
    resolve_label_path,
    load_class_names,
    get_yolo_class,
    parse_yolo_seg_label_file,
    result_to_pred_instances,
    greedy_match_instances,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
CLASS_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (255, 180, 100),
    (180, 100, 255),
    (100, 180, 255),
    (200, 200, 100),
    (200, 100, 200),
    (100, 200, 180),
    (0, 255, 200),
    (50, 200, 50),
]


def parse_species_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def load_test_records(test_csv: Path, dataset_root: Path) -> List[dict]:
    df = pd.read_csv(test_csv)

    columns = list(df.columns)

    image_col = first_existing_column(columns, IMAGE_REF_COLUMNS)
    label_col = first_existing_column(columns, LABEL_REF_COLUMNS)
    species_col = first_existing_column(columns, SPECIES_COLUMNS)
    task_name_col = first_existing_column(columns, TASK_NAME_COLUMNS)

    if image_col is None:
        raise ValueError("No image column found")

    records = []

    for _, row in df.iterrows():
        image_path = resolve_image_path(str(row[image_col]), dataset_root)

        label_ref = row[label_col] if label_col else None
        label_path = resolve_label_path(label_ref, image_path, dataset_root)

        task_name = str(row[task_name_col]).strip() if task_name_col else ""
        _, species_from_task = parse_task_name(task_name)

        species_name = str(row[species_col]).strip() if species_col else ""

        if not species_name or species_name.lower() == "nan":
            species_name = (
                species_from_task
                or parse_species_from_filename(image_path.name)
            )

        records.append({
            "image_path": str(image_path),
            "label_path": str(label_path),
            "species": species_name,
            "image_name": image_path.name,
        })

    return records


def get_class_color(class_id: int):
    if 0 <= class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    return (200, 200, 200)


def draw_instance_overlay(
    canvas: np.ndarray,
    points: np.ndarray,
    color,
    alpha: float = 0.3,
    border_thickness: int = 2,
    label: str = "",
):
    pts = np.round(points).astype(np.int32)

    overlay = canvas.copy()

    cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(
        overlay,
        alpha,
        canvas,
        1 - alpha,
        0,
        canvas,
    )

    cv2.polylines(
        canvas,
        [pts],
        isClosed=True,
        color=color,
        thickness=border_thickness,
    )

    if label:
        x = int(pts[:, 0].mean())
        y = int(pts[:, 1].mean())

        cv2.putText(
            canvas,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def render_side_by_side(
    image,
    gt_instances,
    pred_instances,
    class_names,
):
    left = image.copy()
    right = image.copy()

    # ---------------- GT ----------------
    for gt in gt_instances:
        cls_id = gt["class_id"]

        cls_name = (
            class_names[cls_id]
            if cls_id < len(class_names)
            else str(cls_id)
        )

        draw_instance_overlay(
            left,
            gt["points"],
            get_class_color(cls_id),
            label=f"GT {cls_name}",
        )

    # ---------------- PRED ----------------
    for pred in pred_instances:
        cls_id = pred["class_id"]

        cls_name = (
            class_names[cls_id]
            if cls_id < len(class_names)
            else str(cls_id)
        )

        conf = pred.get("confidence", 0)

        draw_instance_overlay(
            right,
            pred["points"],
            get_class_color(cls_id),
            label=f"{cls_name} {conf:.2f}",
        )

    h, w = left.shape[:2]

    header_h = 30

    left_header = np.zeros((header_h, w, 3), dtype=np.uint8)
    right_header = np.zeros((header_h, w, 3), dtype=np.uint8)

    cv2.putText(
        left_header,
        "GROUND TRUTH",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        right_header,
        "PREDICTION",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

    left_panel = np.vstack([left_header, left])
    right_panel = np.vstack([right_header, right])

    separator = np.ones((left_panel.shape[0], 5, 3), dtype=np.uint8) * 120

    combined = np.hstack([left_panel, separator, right_panel])

    return combined


def main():

    # -----------------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------------
    model_path = "models/sem_loss_stratified/best.pt"

    test_csv = "data/raw_selected/split1_species_aware/test.csv"

    dataset_root = Path("data/32_images/images")

    data_yaml = "data/raw_selected/data.yaml"

    save_dir = Path("analysis/visual_review/sample_review/32_images")

    imgsz = 1024

    conf = 0.25

    device = "cpu"

    max_images = 10
    # -----------------------------------------------------------------------

    save_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(Path(data_yaml))

    records = load_test_records(Path(test_csv), dataset_root)

    # ============================================================
    # SOLO 10 IMÁGENES
    # ============================================================
    records = records[:max_images]

    print(f"Using only {len(records)} images")

    YOLO = get_yolo_class()

    model = YOLO(model_path)

    image_paths = [r["image_path"] for r in records]

    print("Running predictions...")

    results = model.predict(
        source=image_paths,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        max_det=300,
        retina_masks=True,
        device=device,
        verbose=False,
        stream=True,
    )

    # ============================================================
    # SIEMPRE GUARDAR
    # ============================================================
    for idx, (record, result) in enumerate(zip(records, results)):

        print(f"[{idx+1}/{len(records)}] {record['image_name']}")

        pred_instances = result_to_pred_instances(result)

        output = {
            "image_name": record["image_name"],
            "image_path": record["image_path"],
            "predictions": [],
        }

        for pred in pred_instances:

            output["predictions"].append({
                "class_id": int(pred["class_id"]),
                "confidence": float(pred.get("confidence", 0.0)),
                "polygon": pred["points"].tolist(),
            })

        stem = Path(record["image_name"]).stem

        out_path = save_dir / f"{stem}_predictions.json"

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved: {out_path}")

        print(f"[{idx+1}/{len(records)}] {record['image_name']}")

        image = cv2.imread(record["image_path"])

        if image is None:
            print("Could not load image")
            continue

        h, w = image.shape[:2]

        gt_instances = parse_yolo_seg_label_file(
            Path(record["label_path"]),
            w,
            h,
        )

        pred_instances = result_to_pred_instances(result)

        canvas = render_side_by_side(
            image=image,
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            class_names=class_names,
        )

        stem = Path(record["image_name"]).stem

        out_path = save_dir / f"{idx+1:02d}_{stem}_viz.jpg"

        cv2.imwrite(
            str(out_path),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        print(f"Saved: {out_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()