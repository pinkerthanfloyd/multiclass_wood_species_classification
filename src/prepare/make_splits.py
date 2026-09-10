"""
Build train/val/test splits from relative_feature_vectors.csv preserving the
vector representation per image/label pair.

Strategy: hybrid coverage-first + iterative multilabel stratification.

1. Detect rare classes (total presence below RARE_CLASS_THRESHOLD).
2. Seed the test split with a minimum number of images per rare class.
3. Fill the rest of the test split via iterative multilabel stratification
   (skmultilearn). Falls back to a greedy stratified approach if the package
   is not installed.
4. Repeat for val on the remaining images.
5. The leftover goes to train.
6. Write three CSVs that mirror the columns of the input vectors CSV plus a
   `split` column. A coverage report is dumped as JSON for inspection.

Run after build_relative_vectors_from_yolo_dataset.py.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from skmultilearn.model_selection import iterative_train_test_split
    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_CSV = Path("outputs/relative_feature_vectors.csv")
OUTPUT_DIR = Path("data/raw_selected/split1")
TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
COVERAGE_REPORT_JSON = OUTPUT_DIR / "split_coverage_report.json"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Coverage-first quotas
MIN_TEST_PER_CLASS = 3
MIN_VAL_PER_CLASS = 2

# Only classes with global presence below this threshold trigger seeding.
# Classes above this number are reliably handled by iterative stratification.
RARE_CLASS_THRESHOLD = 15

# If True, the per-class quota gets clipped to ceil(class_total * fraction)
# whenever the class has too few images to satisfy the static minimum.
ADAPTIVE_MIN = True

# If True, raises if any class ends up below MIN_TEST_PER_CLASS in test.
# If False, only logs a warning and continues.
STRICT_MODE = False

SEED = 42


# ============================================================
# HELPERS
# ============================================================
def detect_presence_columns(columns: Sequence[str]) -> Tuple[List[str], str]:
    """
    Return (columns, kind) describing how to build the binary presence matrix.

    Priority:
      1. presence_feat_*  -> already binary (preferred)
      2. count_feat_*     -> presence = (count > 0)
      3. feat_*           -> fallback: presence = (score > 0). Less reliable: a
                             score may legitimately be 0 even when the trait is
                             present once if all components round to 0, but in
                             practice this matches presence well enough to seed
                             splits when the previous columns are not available.
    """
    presence_cols = [c for c in columns if c.startswith("presence_feat_")]
    if presence_cols:
        return presence_cols, "presence"

    count_cols = [c for c in columns if c.startswith("count_feat_")]
    if count_cols:
        return count_cols, "count"

    skip_prefixes = (
        "count_feat_", "local_feat_", "species_feat_", "global_feat_",
        "presence_feat_", "area_feat_", "score_feat_",
    )
    score_cols = [
        c for c in columns
        if c.startswith("feat_") and not any(c.startswith(p) for p in skip_prefixes)
    ]
    if score_cols:
        print(
            "[warn] Falling back to 'feat_*' score columns for presence detection. "
            "Re-run build_relative_vectors_from_yolo_dataset.py with "
            "WRITE_PRESENCE_COLUMNS=True for a cleaner signal."
        )
        return score_cols, "score"

    raise ValueError(
        "Could not find any of: presence_feat_*, count_feat_*, feat_*. "
        "Re-run build_relative_vectors_from_yolo_dataset.py."
    )


def build_presence_matrix(df: pd.DataFrame, columns: Sequence[str], kind: str) -> np.ndarray:
    """Binary presence matrix Y[image, class] = 1 if the trait is present."""
    arr = df[columns].fillna(0).to_numpy()
    if kind == "presence":
        return (arr != 0).astype(np.int8)
    return (arr > 0).astype(np.int8)


def adaptive_quota(static_quota: int, class_total: int, fraction: float) -> int:
    """Return how many images of one class we want in this split."""
    if class_total <= 0:
        return 0
    if not ADAPTIVE_MIN:
        return min(static_quota, class_total - 1) if class_total > 1 else 0
    floor_estimate = max(1, int(math.ceil(class_total * fraction)))
    desired = min(static_quota, floor_estimate)
    # Always leave at least one image of the class out of the seed for train/val coverage.
    return max(0, min(desired, class_total - 1))


def density_score(row_idx: int, Y: np.ndarray, current_class_loads: np.ndarray) -> Tuple[int, int]:
    """
    Score used to pick which images to seed first when several candidates exist.

    Prefer images that:
      1. Cover MORE rare classes still under quota (reduces total picks).
      2. Have FEWER common classes (avoids unbalancing majority classes in test).
    Returns a tuple where lower is better.
    """
    classes_in_image = np.where(Y[row_idx] == 1)[0]
    rare_covered = int(np.sum(current_class_loads[classes_in_image] > 0))
    total_classes = int(len(classes_in_image))
    return (-rare_covered, total_classes)


def seed_split_for_classes(
    Y: np.ndarray,
    class_totals: np.ndarray,
    quotas: Dict[int, int],
    available: Set[int],
    rng: random.Random,
) -> Set[int]:
    """
    Greedy seeding: walk classes from rarest to commonest, top up each one until
    its quota is met using images from `available`.
    """
    seeded: Set[int] = set()
    class_loads = np.zeros(Y.shape[1], dtype=np.int32)

    rare_order = sorted(
        [c for c, q in quotas.items() if q > 0],
        key=lambda c: (class_totals[c], rng.random()),
    )

    for cls in rare_order:
        already = sum(1 for i in seeded if Y[i, cls] == 1)
        need = quotas[cls] - already
        if need <= 0:
            continue

        candidates = [i for i in available if Y[i, cls] == 1 and i not in seeded]
        if not candidates:
            continue

        remaining_quotas = np.array([
            max(0, quotas.get(c, 0) - sum(1 for i in seeded if Y[i, c] == 1))
            for c in range(Y.shape[1])
        ], dtype=np.int32)

        candidates.sort(key=lambda i: density_score(i, Y, remaining_quotas) + (rng.random(),))

        for i in candidates[:need]:
            seeded.add(i)

    return seeded


def iterative_extract(
    Y: np.ndarray,
    available: Set[int],
    target_count: int,
) -> Set[int]:
    """Extract approximately `target_count` indices via iterative multilabel stratification."""
    if target_count <= 0 or not available:
        return set()

    available_list = sorted(available)
    if target_count >= len(available_list):
        return set(available_list)

    fraction = target_count / len(available_list)
    fraction = min(max(fraction, 1e-4), 1 - 1e-4)

    if HAS_SKMULTILEARN:
        X = np.asarray(available_list).reshape(-1, 1)
        y = Y[available_list]
        try:
            _, _, X_extra, _ = iterative_train_test_split(X, y, test_size=fraction)
            return set(int(v) for v in X_extra.flatten().tolist())
        except Exception as exc:
            print(f"[warn] iterative_train_test_split failed ({exc}); falling back to greedy.")

    return greedy_stratified_extract(Y, available_list, target_count)


def greedy_stratified_extract(Y: np.ndarray, available_list: List[int], target_count: int) -> Set[int]:
    """
    Fallback when skmultilearn is unavailable. Approximates iterative stratification
    by picking, at each step, the image whose label vector pulls the running label
    proportions closest to the target proportions.
    """
    Y_pool = Y[available_list]
    n_classes = Y.shape[1]
    target_proportions = Y_pool.sum(axis=0) / max(1, len(available_list))

    chosen: List[int] = []
    chosen_loads = np.zeros(n_classes, dtype=np.float64)
    pool = list(range(len(available_list)))

    for step in range(target_count):
        best_idx = None
        best_score = math.inf
        denom = step + 1
        for j in pool:
            new_loads = chosen_loads + Y_pool[j]
            new_proportions = new_loads / denom
            score = float(np.sum((new_proportions - target_proportions) ** 2))
            if score < best_score:
                best_score = score
                best_idx = j
        if best_idx is None:
            break
        chosen.append(best_idx)
        chosen_loads += Y_pool[best_idx]
        pool.remove(best_idx)

    return set(available_list[j] for j in chosen)


# ============================================================
# MAIN PIPELINE
# ============================================================
def compute_quotas(
    class_totals: np.ndarray,
    static_min: int,
    fraction: float,
) -> Dict[int, int]:
    quotas: Dict[int, int] = {}
    for class_id, total in enumerate(class_totals):
        total = int(total)
        if total < RARE_CLASS_THRESHOLD:
            quotas[class_id] = adaptive_quota(static_min, total, fraction)
        else:
            quotas[class_id] = 0  # iterative split will cover it
    return quotas


def coverage_per_split(
    Y: np.ndarray,
    indices_by_split: Dict[str, Sequence[int]],
    feature_columns: Sequence[str],
) -> Dict[str, dict]:
    report = {}
    for split_name, indices in indices_by_split.items():
        if not indices:
            split_report = {"images": 0, "classes_present": 0, "per_class": {}}
        else:
            sub = Y[list(indices)]
            presence = sub.sum(axis=0).astype(int).tolist()
            instances = sub.sum(axis=0).astype(int).tolist()
            per_class = {
                feature_columns[c]: {
                    "images_with_class": int(presence[c]),
                    "instances_or_more": int(instances[c]),
                }
                for c in range(Y.shape[1])
            }
            split_report = {
                "images": int(len(indices)),
                "classes_present": int(np.sum(np.array(presence) > 0)),
                "per_class": per_class,
            }
        report[split_name] = split_report
    return report


def species_distribution(
    df: pd.DataFrame,
    indices_by_split: Dict[str, Sequence[int]],
) -> Dict[str, Dict[str, int]]:
    if "species_id" not in df.columns:
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for split_name, indices in indices_by_split.items():
        sub = df.iloc[list(indices)] if len(indices) else df.iloc[[]]
        out[split_name] = Counter(sub["species_id"].astype(str).tolist())
    return out


def assert_strict(report: Dict[str, dict], feature_columns: Sequence[str], min_test: int) -> None:
    if not STRICT_MODE:
        return
    failures = []
    test_per_class = report["test"]["per_class"]
    for col in feature_columns:
        present = test_per_class.get(col, {}).get("images_with_class", 0)
        if present < min_test:
            failures.append((col, present))
    if failures:
        details = ", ".join(f"{col}={present}" for col, present in failures)
        raise RuntimeError(
            f"STRICT_MODE: classes below MIN_TEST_PER_CLASS={min_test} in test split: {details}"
        )


def run() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    presence_columns, presence_kind = detect_presence_columns(df.columns)
    Y = build_presence_matrix(df, presence_columns, presence_kind)
    n_images, n_classes = Y.shape

    # Strip the prefix so we display short, stable class labels regardless of source.
    prefix_strip = {
        "presence": "presence_",
        "count": "count_",
        "score": "",
    }[presence_kind]
    feature_columns_short = [c[len(prefix_strip):] if prefix_strip else c for c in presence_columns]
    class_totals = Y.sum(axis=0)
    print(f"[info] presence representation: {presence_kind} (using {len(presence_columns)} columns)")

    print(f"Loaded {n_images} images and {n_classes} classes from {INPUT_CSV}.")
    print("Class totals (images containing the class):")
    for col, total in zip(feature_columns_short, class_totals):
        flag = "  rare" if total < RARE_CLASS_THRESHOLD else ""
        print(f"  {col:>16s} : {int(total):4d}{flag}")

    rng = random.Random(SEED)
    np.random.seed(SEED)

    all_indices: Set[int] = set(range(n_images))

    # ---- TEST ----
    test_quotas = compute_quotas(class_totals, MIN_TEST_PER_CLASS, TEST_FRACTION)
    test_seeded = seed_split_for_classes(Y, class_totals, test_quotas, all_indices, rng)
    test_target = int(round(TEST_FRACTION * n_images))
    test_remaining_target = max(0, test_target - len(test_seeded))
    test_pool = all_indices - test_seeded
    test_extra = iterative_extract(Y, test_pool, test_remaining_target)
    test_idx = test_seeded | test_extra

    # ---- VAL ----
    val_pool = all_indices - test_idx
    val_class_totals = Y[list(val_pool)].sum(axis=0) if val_pool else np.zeros(n_classes)
    val_quotas = compute_quotas(val_class_totals, MIN_VAL_PER_CLASS, VAL_FRACTION)
    val_seeded = seed_split_for_classes(Y, val_class_totals, val_quotas, val_pool, rng)
    val_target = int(round(VAL_FRACTION * n_images))
    val_remaining_target = max(0, val_target - len(val_seeded))
    val_extra = iterative_extract(Y, val_pool - val_seeded, val_remaining_target)
    val_idx = val_seeded | val_extra

    # ---- TRAIN ----
    train_idx = all_indices - test_idx - val_idx

    # Sanity check
    assert not (test_idx & val_idx), "Overlap between test and val"
    assert not (test_idx & train_idx), "Overlap between test and train"
    assert not (val_idx & train_idx), "Overlap between val and train"
    assert (test_idx | val_idx | train_idx) == all_indices, "Some images are unassigned"

    indices_by_split = {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }

    # ---- WRITE CSVs (preserving every column from input + 'split') ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_with_split = df.copy()
    df_with_split["split"] = "train"
    df_with_split.loc[indices_by_split["val"], "split"] = "val"
    df_with_split.loc[indices_by_split["test"], "split"] = "test"

    df_with_split.iloc[indices_by_split["train"]].to_csv(TRAIN_CSV, index=False)
    df_with_split.iloc[indices_by_split["val"]].to_csv(VAL_CSV, index=False)
    df_with_split.iloc[indices_by_split["test"]].to_csv(TEST_CSV, index=False)

    # ---- REPORT ----
    coverage = coverage_per_split(Y, indices_by_split, feature_columns_short)
    species_report = species_distribution(df, indices_by_split)

    rare_classes_report = {
        feature_columns_short[c]: {
            "total_images": int(class_totals[c]),
            "test_quota_used": int(test_quotas.get(c, 0)),
            "val_quota_used": int(val_quotas.get(c, 0)),
        }
        for c in range(n_classes)
        if class_totals[c] < RARE_CLASS_THRESHOLD
    }

    summary = {
        "input_csv": str(INPUT_CSV),
        "n_images": int(n_images),
        "n_classes": int(n_classes),
        "fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": TEST_FRACTION},
        "split_sizes": {k: len(v) for k, v in indices_by_split.items()},
        "rare_class_threshold": RARE_CLASS_THRESHOLD,
        "min_test_per_class": MIN_TEST_PER_CLASS,
        "min_val_per_class": MIN_VAL_PER_CLASS,
        "adaptive_min": ADAPTIVE_MIN,
        "iterative_stratifier": "skmultilearn" if HAS_SKMULTILEARN else "greedy_fallback",
        "seed": SEED,
        "rare_classes": rare_classes_report,
        "coverage": coverage,
        "species_distribution": {k: dict(v) for k, v in species_report.items()},
    }

    with open(COVERAGE_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- LOG ----
    print()
    print("Split sizes:")
    for k, v in indices_by_split.items():
        print(f"  {k:>5s} : {len(v):4d}")

    print()
    print("Per-class images in each split (rare classes flagged):")
    header = f"{'class':>16s}  total   train   val   test"
    print(header)
    for c, col in enumerate(feature_columns_short):
        total = int(class_totals[c])
        train_count = int(coverage["train"]["per_class"][col]["images_with_class"])
        val_count = int(coverage["val"]["per_class"][col]["images_with_class"])
        test_count = int(coverage["test"]["per_class"][col]["images_with_class"])
        flag = "  <-- rare" if total < RARE_CLASS_THRESHOLD else ""
        print(f"{col:>16s} {total:6d}  {train_count:6d} {val_count:5d} {test_count:5d}{flag}")

    print()
    print(f"Train CSV : {TRAIN_CSV}")
    print(f"Val CSV   : {VAL_CSV}")
    print(f"Test CSV  : {TEST_CSV}")
    print(f"Report    : {COVERAGE_REPORT_JSON}")

    assert_strict(coverage, feature_columns_short, MIN_TEST_PER_CLASS)


if __name__ == "__main__":
    run()
