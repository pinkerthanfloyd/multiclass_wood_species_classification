from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

# ============================================================
# SIMPLE CONFIGURATION
# ============================================================
DATASET_ROOT = Path("data/raw_selected")
DATA_YAML_PATH = DATASET_ROOT / "data.yaml"
SPLIT_FOLDER = DATASET_ROOT / "split1"

TRAIN_CSV = SPLIT_FOLDER / "train.csv"
VAL_CSV = SPLIT_FOLDER / "val.csv"
TEST_CSV = SPLIT_FOLDER / "test.csv"

# If the CSVs do not exist, the script can fall back to the txt files named in data.yaml.
USE_SPLIT_CSVS_IF_AVAILABLE = True

# Training checkpoint and outputs
BASE_MODEL_WEIGHTS = "yolo11n-seg.pt"
OUTPUT_ROOT = Path("runs_wood_feature_baseline") / SPLIT_FOLDER.name

EXPERIMENTS = [
    {"name": "baseline_bs4_img1024", "batch": 4, "imgsz": 1024},
    {"name": "baseline_bs8_img1024", "batch": 8, "imgsz": 1024},
]

# Baseline training settings
EPOCHS = 250
WORKERS = 4
DEVICE = 0  # use "cpu" if needed
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
MATCH_IOU = 0.50
MAX_DET = 300
RETINA_MASKS = True

# Candidate column names inside split CSVs
IMAGE_REF_COLUMNS = [
    "image_path", "path", "filepath", "file_path", "relative_path",
    "image", "file", "filename", "image_name", "name"
]
LABEL_REF_COLUMNS = ["label_path", "annotation_path", "label_file", "label"]
SPECIES_COLUMNS = ["species_name", "species", "species_tag", "species_id", "class_species"]
TASK_NAME_COLUMNS = ["task_name"]

def get_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics is not installed in this environment. Install it with `pip install ultralytics` before running training."
        ) from e
    return YOLO


# ============================================================
# HELPERS
# ============================================================
def load_class_names(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        return [str(v) for v in names]
    if isinstance(names, dict):
        ordered = []
        for key in sorted(names, key=lambda x: int(x)):
            ordered.append(str(names[key]))
        return ordered
    raise ValueError(f"Could not read class names from {yaml_path}")


def parse_task_name(task_name: str) -> tuple[str, str]:
    task_name = (task_name or "").strip()
    parts = task_name.split()
    if parts and parts[0].isdigit():
        return parts[0], " ".join(parts[1:]).strip()
    return "", task_name


def first_existing_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def deduplicate_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    output = []
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            output.append(path)
    return output


def resolve_dataset_path(ref: str, dataset_root: Path, kind: str) -> Optional[Path]:
    if ref is None:
        return None

    # Normalize Windows-style backslashes to forward slashes.
    # YOLO index txt files often mix separators (e.g. 'data/images/train\0001-1.jpg')
    # which on POSIX systems makes Path treat the backslash as part of the filename
    # and silently breaks every path candidate built below.
    ref = str(ref).strip().replace("\\", "/")
    if not ref or ref.lower() == "nan":
        return None

    raw = Path(ref)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    # direct relative to dataset root
    candidates.append(dataset_root / raw)

    parts = list(raw.parts)

    # allow refs that start with "data/..."
    if parts and parts[0] == "data":
        candidates.append(dataset_root / Path(*parts[1:]))

    # allow refs that already contain images or labels somewhere in the path
    if "images" in parts:
        idx = parts.index("images")
        candidates.append(dataset_root / Path(*parts[idx:]))
    if "labels" in parts:
        idx = parts.index("labels")
        candidates.append(dataset_root / Path(*parts[idx:]))

    # common aggregate layout after removing train/val/test subfolders
    if len(parts) == 1:
        if kind == "image":
            if (dataset_root / "images").exists():
                candidates.extend((dataset_root / "images").rglob(parts[0]))
        if kind == "label":
            if (dataset_root / "labels").exists():
                candidates.extend((dataset_root / "labels").rglob(parts[0]))

    for path in deduplicate_paths(candidates):
        if path.exists():
            return path.resolve()
    return None


def resolve_image_path(image_ref: str, dataset_root: Path) -> Path:
    path = resolve_dataset_path(image_ref, dataset_root, kind="image")
    if path is None:
        raise FileNotFoundError(f"Could not resolve image path from reference: {image_ref}")
    return path


def derive_label_from_image(image_path: Path, dataset_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    # Same normalization for safety: image_path may have arrived with a literal '\'.
    image_str = str(image_path).replace("\\", "/")

    # images/... -> labels/...
    if "/images/" in image_str or "\\images\\" in image_str:
        replaced = image_str.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        candidates.append(Path(replaced).with_suffix(".txt"))

    # relative to dataset_root/images
    try:
        rel = image_path.relative_to(dataset_root / "images")
        candidates.append((dataset_root / "labels" / rel).with_suffix(".txt"))
    except Exception:
        pass

    # if images are already directly under dataset_root/images/*.jpg and labels/*.txt
    candidates.append((dataset_root / "labels" / image_path.with_suffix(".txt").name))

    # basename fallback search
    basename = image_path.with_suffix(".txt").name
    if (dataset_root / "labels").exists():
        candidates.extend((dataset_root / "labels").rglob(basename))

    for path in deduplicate_paths(candidates):
        if path.exists():
            return path.resolve()
    return None


def resolve_label_path(label_ref: Optional[str], image_path: Path, dataset_root: Path) -> Path:
    if label_ref is not None and str(label_ref).strip() and str(label_ref).lower() != "nan":
        path = resolve_dataset_path(str(label_ref), dataset_root, kind="label")
        if path is not None:
            return path

    derived = derive_label_from_image(image_path, dataset_root)
    if derived is None:
        raise FileNotFoundError(f"Could not resolve label path for image: {image_path}")
    return derived


def read_split_txt(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        return []
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_records_from_split_csvs(dataset_root: Path) -> List[dict]:
    csv_files = {
        "train": TRAIN_CSV,
        "val": VAL_CSV,
        "test": TEST_CSV,
    }
    if not any(path.exists() for path in csv_files.values()):
        return []

    records: List[dict] = []
    for split_name, csv_path in csv_files.items():
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        columns = list(df.columns)
        image_col = first_existing_column(columns, IMAGE_REF_COLUMNS)
        label_col = first_existing_column(columns, LABEL_REF_COLUMNS)
        species_col = first_existing_column(columns, SPECIES_COLUMNS)
        task_name_col = first_existing_column(columns, TASK_NAME_COLUMNS)

        if image_col is None:
            raise ValueError(f"{csv_path} does not contain an image reference column. Checked: {IMAGE_REF_COLUMNS}")

        for _, row in df.iterrows():
            image_path = resolve_image_path(str(row[image_col]), dataset_root)
            label_ref = row[label_col] if label_col else None
            label_path = resolve_label_path(label_ref, image_path, dataset_root)

            task_name = str(row[task_name_col]).strip() if task_name_col else ""
            _, species_from_task = parse_task_name(task_name)
            species_name = str(row[species_col]).strip() if species_col else ""
            if not species_name or species_name.lower() == "nan":
                species_name = species_from_task or "unknown"

            records.append(
                {
                    "split": split_name,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "species_name": species_name,
                }
            )
    return records


def collect_records_from_yaml_txts(dataset_root: Path, yaml_path: Path) -> List[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    split_refs = {
        "train": data.get("train"),
        "val": data.get("val"),
        "test": data.get("test"),
    }

    records: List[dict] = []
    for split_name, split_ref in split_refs.items():
        if not split_ref:
            continue

        txt_path = resolve_dataset_path(str(split_ref), dataset_root, kind="image")
        if txt_path is None:
            txt_path = dataset_root / str(split_ref)
        if not txt_path.exists():
            continue

        for image_ref in read_split_txt(txt_path):
            image_path = resolve_image_path(image_ref, dataset_root)
            label_path = resolve_label_path(None, image_path, dataset_root)
            records.append(
                {
                    "split": split_name,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "species_name": "unknown",
                }
            )
    return records


def prepare_dataset_yaml_and_txts(records: List[dict], class_names: List[str]) -> tuple[Path, Dict[str, List[dict]]]:
    by_split = {"train": [], "val": [], "test": []}
    for rec in records:
        if rec["split"] in by_split:
            by_split[rec["split"]].append(rec)

    missing_splits = [name for name in ["train", "val", "test"] if not by_split[name]]
    if missing_splits:
        raise RuntimeError(f"Missing split records for: {missing_splits}")

    SPLIT_FOLDER.mkdir(parents=True, exist_ok=True)

    txt_paths = {}
    for split_name, split_records in by_split.items():
        txt_path = SPLIT_FOLDER / f"{split_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in split_records:
                f.write(str(Path(rec["image_path"]).resolve()) + "\n")
        txt_paths[split_name] = txt_path

    dataset_yaml_path = SPLIT_FOLDER / "dataset_for_training.yaml"
    yaml_data = {
        "path": str(DATASET_ROOT.resolve()),
        "train": str(txt_paths["train"].resolve()),
        "val": str(txt_paths["val"].resolve()),
        "test": str(txt_paths["test"].resolve()),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    return dataset_yaml_path, by_split


# ============================================================
# CUSTOM EVALUATION HELPERS
# ============================================================
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

        try:
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue

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


def greedy_match_instances(gt_instances: List[dict], pred_instances: List[dict], match_iou: float):
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
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

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


def evaluate_on_test_split(best_model_path: Path, experiment_dir: Path, imgsz: int, class_names: List[str], test_records: List[dict]) -> dict:
    YOLO = get_yolo_class()
    model = YOLO(str(best_model_path))
    num_classes = len(class_names)

    image_to_label = {str(Path(rec["image_path"]).resolve()): str(Path(rec["label_path"]).resolve()) for rec in test_records}
    image_to_species = {str(Path(rec["image_path"]).resolve()): rec.get("species_name", "unknown") for rec in test_records}
    test_images = [Path(rec["image_path"]).resolve() for rec in test_records]

    global_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    species_matrices = defaultdict(lambda: np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64))

    image_metrics_rows = []
    image_confusion_records = []

    if not test_images:
        raise RuntimeError("No test images found in the provided split records.")

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
        image_path = Path(result.path).resolve()
        species = image_to_species.get(str(image_path), "unknown")
        image_height, image_width = result.orig_shape
        label_path = Path(image_to_label[str(image_path)])

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

        metrics = global_metrics_from_confusion(image_matrix, class_names)
        metrics["image_name"] = image_path.name
        metrics["species"] = species
        image_metrics_rows.append(metrics)

        image_confusion_records.extend(
            confusion_matrix_to_long_records(
                image_matrix,
                class_names,
                level_name="image_name",
                level_value=image_path.name,
            )
        )

    global_confusion_df = confusion_matrix_to_dataframe(global_matrix, class_names)
    global_confusion_df.to_csv(experiment_dir / "custom_global_confusion_matrix.csv")

    global_class_metrics_df = per_class_metrics_from_confusion(global_matrix, class_names)
    save_dataframe(global_class_metrics_df, experiment_dir / "custom_global_class_metrics.csv")

    global_metrics = global_metrics_from_confusion(global_matrix, class_names)
    with open(experiment_dir / "custom_global_metrics.json", "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)

    save_dataframe(pd.DataFrame(image_metrics_rows), experiment_dir / "custom_image_metrics.csv")
    save_dataframe(pd.DataFrame(image_confusion_records), experiment_dir / "custom_image_confusion_long.csv")

    species_summary_rows = []
    species_confusion_records = []
    for species, matrix in species_matrices.items():
        metrics = global_metrics_from_confusion(matrix, class_names)
        metrics["species"] = species
        species_summary_rows.append(metrics)

        species_confusion_records.extend(
            confusion_matrix_to_long_records(
                matrix,
                class_names,
                level_name="species",
                level_value=species,
            )
        )

        species_confusion_df = confusion_matrix_to_dataframe(matrix, class_names)
        safe_species = re.sub(r"[^0-9A-Za-z_\-]+", "_", species)
        species_confusion_df.to_csv(experiment_dir / "species_confusion_matrices" / f"{safe_species}.csv")

    save_dataframe(pd.DataFrame(species_summary_rows), experiment_dir / "custom_species_metrics.csv")
    save_dataframe(pd.DataFrame(species_confusion_records), experiment_dir / "custom_species_confusion_long.csv")

    return global_metrics


# ============================================================
# TRAINING
# ============================================================
def run_training_experiment(experiment: dict, dataset_yaml_path: Path, class_names: List[str], by_split: Dict[str, List[dict]]) -> dict:
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]

    YOLO = get_yolo_class()
    model = YOLO(BASE_MODEL_WEIGHTS)
    train_results = model.train(
        data=str(dataset_yaml_path),
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
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=EXIST_OK,
        plots=True,
        save=True,
        verbose=True,
    )

    experiment_dir = Path(train_results.save_dir)
    best_model_path = experiment_dir / "weights" / "best.pt"

    YOLO = get_yolo_class()
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(dataset_yaml_path),
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

    try:
        built_in_confusion_df = val_results.confusion_matrix.to_df()
        built_in_confusion_df.to_csv(experiment_dir / "ultralytics_confusion_matrix.csv", index=False)
    except Exception:
        pass

    custom_global_metrics = evaluate_on_test_split(best_model_path, experiment_dir, imgsz, class_names, by_split["test"])

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
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML_PATH}")

    class_names = load_class_names(DATA_YAML_PATH)

    records = []
    if USE_SPLIT_CSVS_IF_AVAILABLE:
        records = collect_records_from_split_csvs(DATASET_ROOT)
    if not records:
        records = collect_records_from_yaml_txts(DATASET_ROOT, DATA_YAML_PATH)

    if not records:
        raise RuntimeError("No split records found. Check split CSVs or the txt files referenced in data.yaml.")

    dataset_yaml_path, by_split = prepare_dataset_yaml_and_txts(records, class_names)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_training_experiment(experiment, dataset_yaml_path, class_names, by_split)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("Finished. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
