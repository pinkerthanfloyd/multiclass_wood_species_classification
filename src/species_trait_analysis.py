"""
species_trait_analysis.py

Two views of how a trained YOLOv8/v11 segmentation model performs on a per-species,
per-trait basis:

  1. Per-species confusion matrices: for each tree species (extracted from the
     filename prefix, e.g. '0003-1.jpg' -> species '0003'), a 14x14 + background
     confusion matrix over the YOLO classes plus the per-class TP/FP/FN/P/R/F1
     metrics. Tells you which traits are detected well in which species.

  2. Per-trait species rankings: for each trait/class, a ranked list of species
     sorted by F1 (with precision and recall), so you can answer 'in which
     species is trait X best detected?'.

Reuses the matching and confusion-matrix helpers from train_yolo_seg_baseline_csvs.py.
Run it after a successful training run; point MODEL_WEIGHTS at best.pt.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Pull in helpers from the training script (matching by IoU, confusion matrix
# building, per-class metrics) so we don't duplicate logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from train_yolo_seg_baseline_csvs import (
        parse_yolo_seg_label_file,
        result_to_pred_instances,
        build_instance_confusion_matrix,
        confusion_matrix_to_dataframe,
        per_class_metrics_from_confusion,
        load_class_names,
        get_yolo_class,
        resolve_dataset_path,
        derive_label_from_image,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import helpers from train_yolo_seg_baseline_csvs.py. "
        "Place this script next to it in the project root."
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================
# Paths -- all relative to the directory you run the script from.
DATASET_ROOT  = Path("data/raw_selected")                       # where images/ and labels/ live
MODEL_WEIGHTS = Path("runs_wood_feature_baseline/split2/baseline_bs4_img1024/weights/best.pt")
EVAL_CSV      = Path("data/raw_selected/split2/test.csv")       # may be val.csv or any CSV with image_path/label_path
DATA_YAML     = Path("data/raw_selected/data.yaml")
OUTPUT_DIR    = Path("analysis/species_trait_breakdown")

# Inference settings (match what was used to evaluate at training time).
# NOTE on memory: retina_masks=True upscales masks to the original image size
# during postprocess and is the single biggest memory consumer. If the GPU is
# crowded (other processes using most VRAM), set RETINA_MASKS = False — the
# IoU-based matching used here only needs the masks to be coarser-than-GT,
# Ultralytics resizes them anyway before comparison. The mAP/IoU difference
# in practice is small.
IMGSZ = 1024
CONF = 0.25
NMS_IOU = 0.7
MATCH_IOU = 0.50
MAX_DET = 300
RETINA_MASKS = False     # was True; switch back if you have ample VRAM
DEVICE = 0               # use "cpu" if GPU is unusable, or another id if multiple GPUs

# Try to keep memory fragmentation low. Set BEFORE importing torch downstream.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Filtering / display
MIN_IMAGES_PER_SPECIES = 3        # individual species CSV/heatmap only if at least this many images
MIN_GT_PER_TRAIT_FOR_RANKING = 3  # in trait ranking, ignore (species, trait) cells with fewer GT instances
HEATMAP_TOP_SPECIES = 12          # how many species (most images) get a heatmap PNG; rest only get CSV

# Visualization
SAVE_HEATMAPS = True


# ============================================================
# HELPERS
# ============================================================
def parse_species_from_filename(file_name: str) -> str:
    """Same logic the build_relative_vectors uses: prefix before first '-' or '_'."""
    s = str(file_name).replace("\\", "/").split("/")[-1]
    stem = Path(s).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def detect_path_columns(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    image_candidates = ["image_path", "path", "filepath", "file_path", "image", "filename", "image_name"]
    label_candidates = ["label_path", "annotation_path", "label_file", "label"]
    image_col = next((c for c in image_candidates if c in df.columns), None)
    if image_col is None:
        raise ValueError(f"No image-path column in CSV. Looked for: {image_candidates}")
    label_col = next((c for c in label_candidates if c in df.columns), None)
    return image_col, label_col


def derive_label_from_image(image_path: Path) -> Path:
    """If no label column in the CSV, derive label from image path the YOLO way."""
    s = str(image_path).replace("\\", "/")
    if "/images/" in s:
        candidate = Path(s.replace("/images/", "/labels/")).with_suffix(".txt")
        if candidate.exists():
            return candidate
    candidate = image_path.with_suffix(".txt")
    return candidate


def safe_filename_token(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "unknown"


def collect_records(csv_path: Path) -> List[dict]:
    """
    Read the eval CSV and resolve every (image, label) reference using the same
    multi-candidate resolver that the training script uses. This handles CSVs
    written with relative paths like '../data/raw_selected/images/0003-1.jpg'
    even when run from a different working directory, as well as Windows-style
    backslashes mixed with forward slashes.
    """
    df = pd.read_csv(csv_path)
    image_col, label_col = detect_path_columns(df)
    records = []
    n_unresolved = 0
    for _, row in df.iterrows():
        image_ref = str(row[image_col])
        image_path = resolve_dataset_path(image_ref, DATASET_ROOT, kind="image")
        if image_path is None:
            n_unresolved += 1
            if n_unresolved <= 5:
                print(f"[warn] image not found, skipping: {image_ref}")
            continue

        label_path = None
        if label_col and isinstance(row[label_col], str) and str(row[label_col]).strip():
            label_path = resolve_dataset_path(str(row[label_col]), DATASET_ROOT, kind="label")
        if label_path is None:
            label_path = derive_label_from_image(image_path, DATASET_ROOT)
        if label_path is None or not Path(label_path).exists():
            n_unresolved += 1
            if n_unresolved <= 5:
                print(f"[warn] label not found for {image_path.name}, skipping")
            continue

        records.append({
            "image_path": Path(image_path),
            "label_path": Path(label_path),
            "species": parse_species_from_filename(image_path.name),
        })

    if n_unresolved > 5:
        print(f"[warn] ...and {n_unresolved - 5} more unresolved entries (suppressed).")
    print(f"[info] resolved {len(records)} / {len(df)} entries from {csv_path.name}")
    return records


# ============================================================
# CORE EVALUATION
# ============================================================
def evaluate_per_image(records: List[dict], class_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Run the model over all images. Build a confusion matrix per (species).
    Returns { species_id: np.ndarray of shape (C+1, C+1) }.
    """
    YOLO = get_yolo_class()
    model = YOLO(str(MODEL_WEIGHTS))
    num_classes = len(class_names)

    species_matrices: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    )
    species_image_counts: Dict[str, int] = defaultdict(int)
    image_to_species = {str(r["image_path"]): r["species"] for r in records}
    image_to_label = {str(r["image_path"]): r["label_path"] for r in records}

    for r in records:
        species_image_counts[r["species"]] += 1

    print(f"Running inference on {len(records)} images across "
          f"{len(species_image_counts)} species...")

    results = model.predict(
        source=[str(r["image_path"]) for r in records],
        imgsz=IMGSZ,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        retina_masks=RETINA_MASKS,
        device=DEVICE,
        verbose=False,
        stream=True,
    )

    for result in results:
        image_path_key = str(Path(result.path).resolve())
        species = image_to_species.get(image_path_key)
        if species is None:
            # Resolution mismatch fallback: parse from result.path directly
            species = parse_species_from_filename(Path(result.path).name)
        label_path = image_to_label.get(image_path_key)
        if label_path is None:
            label_path = derive_label_from_image(Path(result.path))

        height, width = result.orig_shape
        gt_instances = parse_yolo_seg_label_file(Path(label_path), width, height)
        pred_instances = result_to_pred_instances(result)

        image_matrix = build_instance_confusion_matrix(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            num_classes=num_classes,
            match_iou=MATCH_IOU,
        )
        species_matrices[species] += image_matrix

    return dict(species_matrices), dict(species_image_counts)


# ============================================================
# OUTPUT BUILDERS
# ============================================================
def build_long_metrics(
    species_matrices: Dict[str, np.ndarray],
    species_image_counts: Dict[str, int],
    class_names: List[str],
) -> pd.DataFrame:
    """Long-format table with one row per (species, trait): TP/FP/FN/P/R/F1 + supports."""
    rows = []
    for species, matrix in species_matrices.items():
        per_class = per_class_metrics_from_confusion(matrix, class_names)
        for _, c in per_class.iterrows():
            rows.append({
                "species": species,
                "trait": c["class_name"],
                "n_images_in_species": int(species_image_counts.get(species, 0)),
                "gt_instances": int(c["support"]),
                "pred_instances": int(c["predicted"]),
                "tp": int(c["tp"]),
                "fp": int(c["fp"]),
                "fn": int(c["fn"]),
                "precision": round(float(c["precision"]), 6),
                "recall": round(float(c["recall"]), 6),
                "f1": round(float(c["f1"]), 6),
            })
    return pd.DataFrame(rows)


def write_per_species_outputs(
    species_matrices: Dict[str, np.ndarray],
    species_image_counts: Dict[str, int],
    class_names: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Long-format combined confusion matrix
    long_records = []
    labels = class_names + ["background"]
    for species, matrix in species_matrices.items():
        for i, gt_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                count = int(matrix[i, j])
                if count == 0:
                    continue
                long_records.append({
                    "species": species,
                    "n_images_in_species": int(species_image_counts.get(species, 0)),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "count": count,
                })
    pd.DataFrame(long_records).to_csv(out_dir / "all_species_confusion_long.csv", index=False)

    # Individual confusion matrices for species with enough images
    per_species_dir = out_dir / "per_species"
    per_species_dir.mkdir(exist_ok=True)
    qualifying = sorted(
        [(sp, n) for sp, n in species_image_counts.items()
         if n >= MIN_IMAGES_PER_SPECIES],
        key=lambda kv: -kv[1],
    )
    print(f"Writing per-species CSVs for {len(qualifying)} species "
          f"(>= {MIN_IMAGES_PER_SPECIES} images each)...")

    for species, _ in qualifying:
        matrix = species_matrices[species]
        token = safe_filename_token(species)
        df_cm = confusion_matrix_to_dataframe(matrix, class_names)
        df_cm.to_csv(per_species_dir / f"{token}_confusion_matrix.csv")

        per_class = per_class_metrics_from_confusion(matrix, class_names)
        per_class.to_csv(per_species_dir / f"{token}_class_metrics.csv", index=False)

    # Heatmaps for the top species
    if SAVE_HEATMAPS:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[warn] matplotlib not available; skipping heatmaps.")
            return
        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        for species, _ in qualifying[:HEATMAP_TOP_SPECIES]:
            matrix = species_matrices[species]
            token = safe_filename_token(species)
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.where(row_sums > 0, matrix / np.maximum(row_sums, 1), 0)
            fig, ax = plt.subplots(figsize=(7.5, 6))
            im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            tick_labels = class_names + ["background"]
            ax.set_xticks(range(len(tick_labels)))
            ax.set_yticks(range(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground truth")
            ax.set_title(f"Species {species}  (n={species_image_counts[species]} images)")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    count = int(matrix[i, j])
                    if count > 0:
                        color = "white" if normalized[i, j] > 0.5 else "black"
                        ax.text(j, i, str(count), ha="center", va="center",
                                fontsize=6, color=color)
            plt.colorbar(im, ax=ax, label="Row-normalized rate")
            plt.tight_layout()
            plt.savefig(heatmap_dir / f"{token}_confusion_heatmap.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


def write_trait_rankings(long_metrics: pd.DataFrame, out_dir: Path) -> None:
    """For each trait, save a CSV of species ranked by F1 (filtered by min support)."""
    trait_dir = out_dir / "per_trait"
    trait_dir.mkdir(parents=True, exist_ok=True)

    rankings = []
    for trait, group in long_metrics.groupby("trait"):
        eligible = group[group["gt_instances"] >= MIN_GT_PER_TRAIT_FOR_RANKING].copy()
        if eligible.empty:
            continue
        ranked = eligible.sort_values(by=["f1", "recall", "gt_instances"], ascending=False)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked.to_csv(trait_dir / f"{safe_filename_token(trait)}_species_ranking.csv", index=False)
        # Also collect top-3 for the summary
        for _, row in ranked.head(3).iterrows():
            rankings.append({
                "trait": trait,
                "rank": int(row["rank"]),
                "species": row["species"],
                "n_images_in_species": int(row["n_images_in_species"]),
                "gt_instances": int(row["gt_instances"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            })
    if rankings:
        pd.DataFrame(rankings).to_csv(out_dir / "top3_species_per_trait.csv", index=False)


def write_wide_pivot(long_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Wide pivot tables: rows = traits, cols = species. Values = F1 / recall / support."""
    qualifying = long_metrics[long_metrics["gt_instances"] >= MIN_GT_PER_TRAIT_FOR_RANKING]
    for metric in ("f1", "recall", "precision", "gt_instances"):
        if qualifying.empty:
            continue
        pivot = qualifying.pivot_table(
            index="trait", columns="species", values=metric, aggfunc="first"
        )
        pivot.to_csv(out_dir / f"trait_x_species_{metric}.csv")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_WEIGHTS.resolve()}")
    if not EVAL_CSV.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {EVAL_CSV.resolve()}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML.resolve()}")

    class_names = load_class_names(DATA_YAML)
    print(f"Loaded {len(class_names)} class names from {DATA_YAML}")

    records = collect_records(EVAL_CSV)
    if not records:
        raise RuntimeError("No images resolved from the eval CSV.")
    print(f"Resolved {len(records)} (image, label) pairs.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    species_matrices, species_image_counts = evaluate_per_image(records, class_names)

    long_metrics = build_long_metrics(species_matrices, species_image_counts, class_names)
    long_metrics.to_csv(OUTPUT_DIR / "per_species_per_trait_metrics.csv", index=False)

    write_per_species_outputs(species_matrices, species_image_counts, class_names, OUTPUT_DIR)
    write_trait_rankings(long_metrics, OUTPUT_DIR)
    write_wide_pivot(long_metrics, OUTPUT_DIR)

    # Top-line summary
    summary = {
        "model_weights": str(MODEL_WEIGHTS),
        "eval_csv": str(EVAL_CSV),
        "match_iou": MATCH_IOU,
        "conf": CONF,
        "n_images": int(sum(species_image_counts.values())),
        "n_species": int(len(species_image_counts)),
        "species_with_min_images": int(sum(1 for n in species_image_counts.values()
                                           if n >= MIN_IMAGES_PER_SPECIES)),
        "min_images_per_species": MIN_IMAGES_PER_SPECIES,
        "min_gt_per_trait_for_ranking": MIN_GT_PER_TRAIT_FOR_RANKING,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Outputs written under {OUTPUT_DIR}/:")
    print(f"  per_species_per_trait_metrics.csv  -- long-format master table")
    print(f"  all_species_confusion_long.csv     -- combined confusion long format")
    print(f"  per_species/                       -- per-species CSVs")
    print(f"  per_trait/                         -- per-trait species rankings")
    print(f"  trait_x_species_{{f1,recall,precision,gt_instances}}.csv -- wide pivots")
    print(f"  heatmaps/                          -- top-{HEATMAP_TOP_SPECIES} species heatmaps")
    print(f"  summary.json")


if __name__ == "__main__":
    main()
