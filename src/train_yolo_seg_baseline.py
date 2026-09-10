from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
# from ultralytics import YOLO


# ============================================================
# SIMPLE CONFIGURATION
# Edit these constants and run the file. No argparse on purpose.
# ============================================================

# Dataset in Ultralytics YOLO SEGMENTATION format
# dataset_root/
#   images/
#     train/
#     val/
#     test/
#   labels/
#     train/
#     val/
#     test/
DATASET_ROOT = Path("data/raw_selected")
DATASET_YAML_PATH = DATASET_ROOT / "dataset.yaml"
SPLIT = "split1"

# Class names must follow the class indices used in the YOLO .txt labels.
CLASS_NAMES = [
    # Example:
    # "79",
    # "81",
    # "82",
    # "83",
    # "V1",
]

# Optional CSV to recover species for each image.
# Expected columns: image_name,species
# image_name may be either the file name (e.g. image_001.jpg)
# or the relative path inside images/test (e.g. test/image_001.jpg).
IMAGE_TO_SPECIES_CSV = None
IMAGE_NAME_COLUMN = "image_name"
SPECIES_COLUMN = "species"

# Small pretrained segmentation checkpoint for a baseline.
# If your environment uses a different official checkpoint, replace it here.
BASE_MODEL_WEIGHTS = "yolo11n-seg.pt"

# Output folder for all experiments.
OUTPUT_ROOT = Path("runs_wood_feature_baseline")

# Each experiment is one training run.
# This is a simple way to compare different batch sizes and image sizes.
EXPERIMENTS = [
    {"name": "baseline_bs4_img1024", "batch": 4, "imgsz": 1024},
    {"name": "baseline_bs8_img1024", "batch": 8, "imgsz": 1024},
]

# Training settings
EPOCHS = 100
PATIENCE = 25
WORKERS = 4
DEVICE = 0            # use "cpu" if needed
SEED = 42
PRETRAINED = True
OPTIMIZER = "auto"
COS_LR = True
CLOSE_MOSAIC = 10
AMP = True
CACHE = False
EXIST_OK = True

# Evaluation settings
TEST_CONFIDENCE = 0.25
TEST_NMS_IOU = 0.7
MATCH_IOU = 0.50      # IoU threshold used to match predicted instances with GT instances
MAX_DET = 300
RETINA_MASKS = True


# ============================================================
# HELPERS
# ============================================================

def ensure_dataset_yaml() -> Path:
    """Create a simple Ultralytics dataset YAML from the split folders."""
    if not CLASS_NAMES:
        raise ValueError("CLASS_NAMES is empty. Fill it before training.")

    yaml_data = {
        "path": str(DATASET_ROOT.resolve()),
        "train": f"images/{SPLIT}/train",
        "val": f"images/{SPLIT}/train",
        "test": f"images/{SPLIT}/train",
        "names": CLASS_NAMES,
    }
    DATASET_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    return DATASET_YAML_PATH


def validate_dataset_layout() -> None:
    expected = [
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "images" / "test",
        DATASET_ROOT / "labels" / "train",
        DATASET_ROOT / "labels" / "val",
        DATASET_ROOT / "labels" / "test",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset folders are missing. Expected YOLO segmentation structure. Missing:\n"
            + "\n".join(missing)
        )


def list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def corresponding_label_path(image_path: Path) -> Path:
    relative = image_path.relative_to(DATASET_ROOT / "images")
    return (DATASET_ROOT / "labels" / relative).with_suffix(".txt")


def load_species_map() -> Dict[str, str]:
    if IMAGE_TO_SPECIES_CSV is None:
        return {}
    csv_path = Path(IMAGE_TO_SPECIES_CSV)
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if IMAGE_NAME_COLUMN not in df.columns or SPECIES_COLUMN not in df.columns:
        raise ValueError(
            f"Species CSV must contain columns '{IMAGE_NAME_COLUMN}' and '{SPECIES_COLUMN}'."
        )

    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row[IMAGE_NAME_COLUMN])] = str(row[SPECIES_COLUMN])
    return mapping


def resolve_species(image_path: Path, species_map: Dict[str, str]) -> str:
    basename = image_path.name
    rel = str(image_path.relative_to(DATASET_ROOT / "images"))
    stem = image_path.stem
    if basename in species_map:
        return species_map[basename]
    if rel in species_map:
        return species_map[rel]
    if stem in species_map:
        return species_map[stem]
    return "unknown"


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.round(points).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def parse_yolo_seg_label_file(label_path: Path, image_width: int, image_height: int) -> List[dict]:
    instances = []
    if not label_path.exists():
        return instances

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue

        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue

        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * image_width
            y = coords[i + 1] * image_height
            points.append([x, y])
        points = np.array(points, dtype=np.float32)
        if len(points) < 3:
            continue

        instances.append(
            {
                "class_id": class_id,
                "points": points,
                "mask": polygon_to_mask(points, image_height, image_width),
            }
        )
    return instances


def result_to_pred_instances(result) -> List[dict]:
    instances = []
    if result.boxes is None or len(result.boxes) == 0:
        return instances
    if result.masks is None or result.masks.xy is None:
        return instances

    height, width = result.orig_shape
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy().astype(float)
    polygons = result.masks.xy

    for class_id, confidence, points in zip(class_ids, confidences, polygons):
        points = np.asarray(points, dtype=np.float32)
        if len(points) < 3:
            continue
        instances.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "points": points,
                "mask": polygon_to_mask(points, height, width),
            }
        )
    return instances


def greedy_match_instances(gt_instances: List[dict], pred_instances: List[dict], match_iou: float) -> Tuple[List[tuple], List[int], List[int]]:
    candidates = []
    for gt_idx, gt in enumerate(gt_instances):
        for pred_idx, pred in enumerate(pred_instances):
            iou = mask_iou(gt["mask"], pred["mask"])
            if iou >= match_iou:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    unmatched_gt = [i for i in range(len(gt_instances)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_instances)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def build_instance_confusion_matrix(gt_instances: List[dict], pred_instances: List[dict], num_classes: int, match_iou: float) -> np.ndarray:
    """
    Rows: ground truth class
    Cols: predicted class
    Last row/col: background

    - matched GT/pred increments [gt_class, pred_class]
    - unmatched GT increments [gt_class, background]
    - unmatched prediction increments [background, pred_class]
    """
    background_index = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    matches, unmatched_gt, unmatched_pred = greedy_match_instances(gt_instances, pred_instances, match_iou)

    for gt_idx, pred_idx, _ in matches:
        gt_class = gt_instances[gt_idx]["class_id"]
        pred_class = pred_instances[pred_idx]["class_id"]
        matrix[gt_class, pred_class] += 1

    for gt_idx in unmatched_gt:
        gt_class = gt_instances[gt_idx]["class_id"]
        matrix[gt_class, background_index] += 1

    for pred_idx in unmatched_pred:
        pred_class = pred_instances[pred_idx]["class_id"]
        matrix[background_index, pred_class] += 1

    return matrix


def confusion_matrix_to_dataframe(matrix: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    labels = class_names + ["background"]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def confusion_matrix_to_long_records(matrix: np.ndarray, class_names: List[str], level_name: str, level_value: str) -> List[dict]:
    labels = class_names + ["background"]
    records = []
    for row_idx, gt_label in enumerate(labels):
        for col_idx, pred_label in enumerate(labels):
            count = int(matrix[row_idx, col_idx])
            if count == 0:
                continue
            records.append(
                {
                    level_name: level_value,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "count": count,
                }
            )
    return records


def per_class_metrics_from_confusion(matrix: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    rows = []
    for class_idx, class_name in enumerate(class_names):
        tp = int(matrix[class_idx, class_idx])
        fn = int(matrix[class_idx, :].sum() - tp)
        fp = int(matrix[:, class_idx].sum() - tp)
        support = int(matrix[class_idx, :].sum())
        predicted = int(matrix[:, class_idx].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class_name": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def global_metrics_from_confusion(matrix: np.ndarray, class_names: List[str]) -> dict:
    tp_total = int(np.trace(matrix[:-1, :-1]))
    total_gt = int(matrix[:-1, :].sum())
    total_pred = int(matrix[:, :-1].sum())

    fn_total = total_gt - tp_total
    fp_total = total_pred - tp_total

    micro_precision = tp_total / total_pred if total_pred > 0 else 0.0
    micro_recall = tp_total / total_gt if total_gt > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    per_class = per_class_metrics_from_confusion(matrix, class_names)
    valid = per_class[per_class["support"] > 0].copy()
    macro_precision = float(valid["precision"].mean()) if not valid.empty else 0.0
    macro_recall = float(valid["recall"].mean()) if not valid.empty else 0.0
    macro_f1 = float(valid["f1"].mean()) if not valid.empty else 0.0

    return {
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "num_classes": len(class_names),
        "total_gt_instances": total_gt,
        "total_pred_instances": total_pred,
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def evaluate_on_test_split(best_model_path: Path, experiment_dir: Path, imgsz: int) -> dict:
    model = YOLO(str(best_model_path))
    species_map = load_species_map()
    test_images = list_image_files(DATASET_ROOT / "images" / "test")
    num_classes = len(CLASS_NAMES)

    global_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    species_matrices = defaultdict(lambda: np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64))

    image_metrics_rows = []
    image_confusion_records = []

    if not test_images:
        raise RuntimeError("No test images found.")

    results = model.predict(
        source=[str(p) for p in test_images],
        imgsz=imgsz,
        conf=TEST_CONFIDENCE,
        iou=TEST_NMS_IOU,
        max_det=MAX_DET,
        retina_masks=RETINA_MASKS,
        device=DEVICE,
        verbose=False,
        stream=True,
    )

    for result in results:
        image_path = Path(result.path)
        species = resolve_species(image_path, species_map)
        image_height, image_width = result.orig_shape
        label_path = corresponding_label_path(image_path)

        gt_instances = parse_yolo_seg_label_file(label_path, image_width, image_height)
        pred_instances = result_to_pred_instances(result)

        image_matrix = build_instance_confusion_matrix(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            num_classes=num_classes,
            match_iou=MATCH_IOU,
        )

        global_matrix += image_matrix
        species_matrices[species] += image_matrix

        metrics = global_metrics_from_confusion(image_matrix, CLASS_NAMES)
        metrics["image_name"] = image_path.name
        metrics["species"] = species
        image_metrics_rows.append(metrics)

        image_confusion_records.extend(
            confusion_matrix_to_long_records(
                image_matrix,
                CLASS_NAMES,
                level_name="image_name",
                level_value=image_path.name,
            )
        )

    # Save global confusion matrix and class metrics
    global_confusion_df = confusion_matrix_to_dataframe(global_matrix, CLASS_NAMES)
    global_confusion_df.to_csv(experiment_dir / "custom_global_confusion_matrix.csv")

    global_class_metrics_df = per_class_metrics_from_confusion(global_matrix, CLASS_NAMES)
    save_dataframe(global_class_metrics_df, experiment_dir / "custom_global_class_metrics.csv")

    global_metrics = global_metrics_from_confusion(global_matrix, CLASS_NAMES)
    with open(experiment_dir / "custom_global_metrics.json", "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)

    # Save per-image summaries and per-image confusion records
    save_dataframe(pd.DataFrame(image_metrics_rows), experiment_dir / "custom_image_metrics.csv")
    save_dataframe(pd.DataFrame(image_confusion_records), experiment_dir / "custom_image_confusion_long.csv")

    # Save per-species confusion matrices and metrics
    species_summary_rows = []
    species_confusion_records = []
    for species, matrix in species_matrices.items():
        metrics = global_metrics_from_confusion(matrix, CLASS_NAMES)
        metrics["species"] = species
        species_summary_rows.append(metrics)

        species_confusion_records.extend(
            confusion_matrix_to_long_records(
                matrix,
                CLASS_NAMES,
                level_name="species",
                level_value=species,
            )
        )

        species_confusion_df = confusion_matrix_to_dataframe(matrix, CLASS_NAMES)
        safe_species = species.replace("/", "_").replace("\\", "_").replace(" ", "_")
        species_confusion_df.to_csv(experiment_dir / "species_confusion_matrices" / f"{safe_species}.csv")

    save_dataframe(pd.DataFrame(species_summary_rows), experiment_dir / "custom_species_metrics.csv")
    save_dataframe(pd.DataFrame(species_confusion_records), experiment_dir / "custom_species_confusion_long.csv")

    return global_metrics


def run_training_experiment(experiment: dict, dataset_yaml: Path) -> dict:
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]

    model = YOLO(BASE_MODEL_WEIGHTS)
    train_results = model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        batch=batch_size,
        imgsz=imgsz,
        device=DEVICE,
        workers=WORKERS,
        pretrained=PRETRAINED,
        optimizer=OPTIMIZER,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        amp=AMP,
        cache=CACHE,
        seed=SEED,
        patience=PATIENCE,
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=EXIST_OK,
        plots=True,
        save=True,
        verbose=True,
    )

    experiment_dir = Path(train_results.save_dir)
    best_model_path = experiment_dir / "weights" / "best.pt"

    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch_size,
        conf=0.001,
        iou=TEST_NMS_IOU,
        device=DEVICE,
        plots=True,
        save_json=True,
        project=str(experiment_dir),
        name="test_eval",
        verbose=True,
    )

    # Save built-in confusion matrix when available
    try:
        built_in_confusion_df = val_results.confusion_matrix.to_df()
        built_in_confusion_df.to_csv(experiment_dir / "ultralytics_confusion_matrix.csv", index=False)
    except Exception:
        pass

    # Custom evaluation grouped by image / species / global
    custom_global_metrics = evaluate_on_test_split(best_model_path, experiment_dir, imgsz)

    summary = {
        "experiment": run_name,
        "batch": batch_size,
        "imgsz": imgsz,
        "best_model_path": str(best_model_path),
        "ultralytics_box_map50": float(val_results.box.map50),
        "ultralytics_box_map50_95": float(val_results.box.map),
        "ultralytics_seg_map50": float(val_results.seg.map50),
        "ultralytics_seg_map50_95": float(val_results.seg.map),
        "custom_micro_precision": float(custom_global_metrics["micro_precision"]),
        "custom_micro_recall": float(custom_global_metrics["micro_recall"]),
        "custom_micro_f1": float(custom_global_metrics["micro_f1"]),
        "custom_macro_precision": float(custom_global_metrics["macro_precision"]),
        "custom_macro_recall": float(custom_global_metrics["macro_recall"]),
        "custom_macro_f1": float(custom_global_metrics["macro_f1"]),
        "custom_total_gt_instances": int(custom_global_metrics["total_gt_instances"]),
        "custom_total_pred_instances": int(custom_global_metrics["total_pred_instances"]),
    }

    with open(experiment_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    validate_dataset_layout()
    dataset_yaml = ensure_dataset_yaml()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_training_experiment(experiment, dataset_yaml)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("Finished. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
