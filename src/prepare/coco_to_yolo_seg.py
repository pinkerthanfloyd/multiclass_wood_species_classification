"""
coco_to_yolo_seg.py

Deterministically regenerate the YOLO segmentation .txt files from a COCO JSON.
Treat the JSON as the single source of truth -- after running this script, the
.txt directory contains the curated dataset and Ultralytics consumes the same
annotations that the rest of the pipeline (feature vectors, splits, sem_loss
priors) is built on.

Each .txt file gets one line per annotation:

    class_id x1 y1 x2 y2 ... xn yn

where class_id is the YOLO 0-indexed class (COCO category_id - 1) and the
coordinates are normalized to [0, 1] with COORD_DECIMALS digits.

For COCO annotations with multiple polygon parts (rare; doesn't happen in
this dataset) each part is written as its own line, sharing class_id. This
matches Ultralytics behaviour of treating each line as one instance.

Multi-polygon (MultiPart) handling is configurable below.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

# Optional RLE support. pycocotools is the canonical decoder and cv2 lets us
# extract polygons from the decoded mask. If either is missing we fall back to
# skipping RLE annotations (the user will see the count of skipped instances
# in the report and can install the deps).
try:
    import numpy as np
    import cv2
    from pycocotools import mask as coco_mask
    HAS_RLE_SUPPORT = True
except ImportError:
    HAS_RLE_SUPPORT = False

# When an RLE annotation decodes to multiple disconnected regions, the
# behaviour follows MULTIPART_POLICY: split / skip / concat (see below).
RLE_CONTOUR_EPSILON = 1.0   # pixel tolerance for cv2.approxPolyDP (0 disables)
RLE_MIN_AREA_PX = 4         # tiny contours (often noise) are dropped


# -----------------------------
# Configuration
# -----------------------------
INSTANCES_JSON = Path("data/raw_selected/instances_selected.json")
LABELS_DIR     = Path("data/raw_selected/labels")      # where the .txt files will be written
COORD_DECIMALS = 6                                     # 6 -> sub-pixel for 1500x1500; raise to 7-8 for huge images

# What to do when a single annotation has multiple polygon parts:
#   "split"  -> write each part as a separate YOLO line (one instance becomes N).
#   "skip"   -> drop the whole annotation (safest if multi-part means real artifact).
#   "concat" -> concatenate all parts into a single line (Ultralytics will treat it as one
#               polygon traversed with bridges; works visually but counts as one instance).
MULTIPART_POLICY = "split"

# Drop any image entry whose file_name is not found under this directory. Set to
# None to skip the existence check entirely (useful when the .jpg-s live elsewhere
# or when you just want to regenerate labels regardless).
REQUIRE_IMAGES_AT = Path("data/raw_selected/images")   # or None

# Whether to also create empty .txt files for images with zero annotations.
# Ultralytics tolerates missing label files (treats them as no GT) but creates
# warnings; emitting empty files keeps the dataset directory tidy.
WRITE_EMPTY_TXT = True

# Sanity: refuse to overwrite the labels dir if it contains files and the count
# would change drastically (more than this fraction). Helps avoid wiping a good
# label set by accident.
OVERWRITE_GUARD_FRACTION = 0.50


# -----------------------------
# Helpers
# -----------------------------
def load_coco(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_path.resolve()}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_polygon(coords: Iterable[float], image_width: int, image_height: int,
                   decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    parts = []
    coords = list(coords)
    if not coords or len(coords) % 2 != 0:
        return ""
    for i in range(0, len(coords), 2):
        x = float(coords[i]) / image_width
        y = float(coords[i + 1]) / image_height
        # Clamp to [0, 1] -- annotations occasionally have points sitting one
        # pixel outside the image due to rounding in the annotation tool.
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        parts.append(fmt.format(x))
        parts.append(fmt.format(y))
    return " ".join(parts)


def rle_to_polygons(segmentation, image_height: int, image_width: int) -> List[list]:
    """
    Decode an RLE segmentation to a binary mask, then extract polygon contours
    via OpenCV. Returns a list of flat [x1,y1,x2,y2,...] coordinate lists, one
    per connected region.

    Handles both compressed (counts as bytes/str) and uncompressed (counts as
    list) RLE flavours. Falls back to [] if pycocotools / cv2 are not installed.
    """
    if not HAS_RLE_SUPPORT:
        return []

    rle = dict(segmentation)
    if isinstance(rle.get("counts"), list):
        # Uncompressed RLE -- frPyObjects converts to compressed form.
        rle = coco_mask.frPyObjects(rle, image_height, image_width)
    elif isinstance(rle.get("counts"), str):
        # Compressed RLE stored as a Python str; pycocotools wants bytes.
        rle = {"counts": rle["counts"].encode("ascii"), "size": rle["size"]}

    binary_mask = coco_mask.decode(rle).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[list] = []
    for contour in contours:
        if cv2.contourArea(contour) < RLE_MIN_AREA_PX:
            continue
        if RLE_CONTOUR_EPSILON > 0:
            contour = cv2.approxPolyDP(contour, RLE_CONTOUR_EPSILON, closed=True)
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue
        flat = []
        for x, y in pts:
            flat.append(float(x))
            flat.append(float(y))
        polygons.append(flat)
    return polygons


def expand_polygons(segmentation, policy: str,
                    image_height: int = 0, image_width: int = 0) -> List[list]:
    """Return a list of polygon-coordinate lists in PIXEL space.

    COCO segmentation can be:
      - list of polygons: [[x1,y1,...], [x1,y1,...], ...]
      - RLE dict: {"counts": ..., "size": [...]}
    """
    if isinstance(segmentation, dict):
        polygons = rle_to_polygons(segmentation, image_height, image_width)
        if not polygons:
            return []
        if policy == "skip" and len(polygons) > 1:
            return []
        if policy == "concat":
            flat = []
            for part in polygons:
                flat.extend(part)
            return [flat]
        return polygons
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if all(isinstance(part, (int, float)) for part in segmentation):
        return [list(segmentation)]
    if policy == "skip" and len(segmentation) > 1:
        return []
    if policy == "concat":
        flat = []
        for part in segmentation:
            flat.extend(part)
        return [flat]
    return [list(p) for p in segmentation]


def safe_overwrite_check(labels_dir: Path, new_count: int) -> None:
    if not labels_dir.exists():
        return
    existing = list(labels_dir.glob("*.txt"))
    if not existing:
        return
    old = len(existing)
    delta = abs(new_count - old) / float(old)
    if delta > OVERWRITE_GUARD_FRACTION:
        raise RuntimeError(
            f"Refusing to overwrite {labels_dir}: would change file count from "
            f"{old} to {new_count} ({delta:.0%} change). If this is intended, "
            f"raise OVERWRITE_GUARD_FRACTION or delete the directory first."
        )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    coco = load_coco(INSTANCES_JSON)
    n_images = len(coco["images"])
    n_anns = len(coco["annotations"])
    n_cats = len(coco["categories"])

    # Build category_id (1-indexed) -> yolo class_id (0-indexed)
    coco_to_class = {int(cat["id"]): int(cat["id"]) - 1 for cat in coco["categories"]}
    cat_names = {int(cat["id"]) - 1: str(cat["name"]) for cat in coco["categories"]}

    # Index annotations by image_id for O(N) lookup
    anns_by_image: dict = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    # Optional image-exists filter
    images_to_process = []
    skipped_no_image = 0
    for img in coco["images"]:
        if REQUIRE_IMAGES_AT is not None:
            candidate = REQUIRE_IMAGES_AT / str(img["file_name"])
            if not candidate.exists():
                skipped_no_image += 1
                continue
        images_to_process.append(img)

    expected_txt_count = sum(1 for img in images_to_process
                             if WRITE_EMPTY_TXT or anns_by_image.get(int(img["id"])))
    safe_overwrite_check(LABELS_DIR, expected_txt_count)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Stats
    written = 0
    empty = 0
    multipart_split = 0
    multipart_skipped = 0
    rle_skipped = 0
    unknown_cat = Counter()
    per_class_counts = Counter()

    for img in images_to_process:
        file_name = str(img["file_name"])
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)
        if width <= 0 or height <= 0:
            print(f"[warn] skipping {file_name}: missing image dimensions")
            continue

        anns = anns_by_image.get(int(img["id"]), [])
        lines: List[str] = []

        for ann in anns:
            coco_cid = int(ann["category_id"])
            class_id = coco_to_class.get(coco_cid)
            if class_id is None:
                unknown_cat[coco_cid] += 1
                continue

            segmentation = ann.get("segmentation")
            is_rle = isinstance(segmentation, dict)

            polygons = expand_polygons(segmentation, MULTIPART_POLICY,
                                       image_height=height, image_width=width)
            if not polygons:
                if is_rle and not HAS_RLE_SUPPORT:
                    rle_skipped += 1
                elif is_rle:
                    # RLE decoded to no polygons (e.g. mask was empty)
                    rle_skipped += 1
                elif (isinstance(segmentation, list) and len(segmentation) > 1
                      and MULTIPART_POLICY == "skip"):
                    multipart_skipped += 1
                continue

            if len(polygons) > 1 and MULTIPART_POLICY == "split":
                multipart_split += (len(polygons) - 1)

            for poly in polygons:
                line_coords = format_polygon(poly, width, height, COORD_DECIMALS)
                if not line_coords:
                    continue
                lines.append(f"{class_id} {line_coords}")
                per_class_counts[class_id] += 1

        if not lines and not WRITE_EMPTY_TXT:
            continue

        txt_name = Path(file_name).with_suffix(".txt").name
        out_path = LABELS_DIR / txt_name
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        if not lines:
            empty += 1

    # Report
    print(f"Source JSON       : {INSTANCES_JSON}")
    print(f"Output labels dir : {LABELS_DIR}")
    print(f"Coordinate format : {COORD_DECIMALS} decimals (normalized)")
    print(f"Multipart policy  : {MULTIPART_POLICY}")
    print(f"RLE support       : {'enabled (pycocotools + cv2)' if HAS_RLE_SUPPORT else 'DISABLED (RLE annotations will be skipped)'}")
    print()
    print(f"Images in JSON      : {n_images}")
    print(f"Annotations in JSON : {n_anns}")
    print(f"Categories          : {n_cats}")
    if REQUIRE_IMAGES_AT is not None:
        print(f"Skipped (no image)  : {skipped_no_image}")
    print(f"TXT files written   : {written}")
    print(f"  of which empty    : {empty}")
    if multipart_split:
        print(f"Multipart split into extra lines: {multipart_split}")
    if multipart_skipped:
        print(f"Multipart annotations skipped (policy=skip): {multipart_skipped}")
    if rle_skipped:
        print(f"RLE annotations skipped (unsupported)      : {rle_skipped}")
    if unknown_cat:
        print(f"Unknown category_ids: {dict(unknown_cat)}")
    print()
    print("Per-class line counts in regenerated .txt files:")
    for class_id in sorted(per_class_counts):
        print(f"  class_id={class_id:2d}  ({cat_names.get(class_id, '?'):>8}): "
              f"{per_class_counts[class_id]:6d}")


if __name__ == "__main__":
    main()
