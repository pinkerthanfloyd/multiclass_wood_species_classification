"""
build_relative_vectors_from_coco_json.py

Same output as build_relative_vectors_from_yolo_dataset.py, but reading the
annotations from a COCO-format JSON instead of YOLO .txt label files.

Why a separate script:
  - The COCO JSON is a single source of truth (image dims + polygon + area
    + category_id all in one place). No path resolution headaches, no
    Windows backslash issues, no missing-label silent failures.
  - The JSON already carries 'area' in pixel^2 per instance, so the per-class
    area is simply sum(area) / (width * height). No need to re-run shoelace.
  - The JSON's 'images' list is the authoritative subset to score (e.g. an
    'instances_selected.json' tells you which images count). We iterate it
    directly instead of scanning a folder.

The output CSV is byte-compatible with make_splits.py:
  base columns       : image_path, label_path, image_name, label_name,
                       species_id, species_name, species_reliability,
                       total_instances
  feat_*             : continuous score per class
  presence_feat_*    : binary 0/1 per class
  area_feat_*        : sum of normalized polygon areas per class, in [0,1]
  count_feat_*       : raw instance count per class (when WRITE_COMPONENT_COLUMNS)
  local_feat_*       : count_in_image / total_annotated_instances
  species_feat_*     : count_in_image / total_count_for_class_in_species
  global_feat_*      : count_in_image / total_count_for_class_in_dataset

Note that COCO category ids are 1-indexed; YOLO class ids are 0-indexed.
We map COCO id -> YOLO id by subtracting 1, so feat columns end up indexed
the same way as data.yaml expects.
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
INSTANCES_JSON = Path("data/raw_selected/instances_selected.json")  # COCO file
IMAGES_DIR     = Path("data/raw_selected/images")                   # where the file_name-s actually live
LABELS_DIR     = Path("data/raw_selected/labels")                   # used to populate the label_path column for downstream training

OUTPUT_CSV         = Path("outputs/relative_feature_vectors.csv")
OUTPUT_SCHEMA_JSON = Path("outputs/relative_feature_vectors.schema.json")

# Score formula (identical to the YOLO-based script).
LOCAL_WEIGHT = 0.50
SPECIES_WEIGHT = 0.20
GLOBAL_WEIGHT = 0.30
MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT = 5

# Output columns
WRITE_COMPONENT_COLUMNS = True   # count_*, local_*, species_*, global_*
WRITE_PRESENCE_COLUMNS  = True   # presence_feat_*
WRITE_AREA_COLUMNS      = True   # area_feat_*

# Hard guard: if more than this fraction of images end up with zero instances,
# abort. With a COCO file this is much less likely than with the YOLO script,
# but a corrupt JSON or a category-id mismatch can still produce empties.
ALLOW_EMPTY_FRACTION = 0.10


# -----------------------------
# Helpers
# -----------------------------
def safe_feature_name(name):
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "unknown"
    return f"feat_{text}"


def normalize_path_str(path_value):
    if path_value is None:
        return ""
    return str(path_value).replace("\\", "/")


def parse_species_id_from_filename(file_name):
    normalized = normalize_path_str(file_name).split("/")[-1]
    stem = Path(normalized).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def load_coco(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_path.resolve()}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_mapping(categories):
    """
    Build the mapping COCO category_id (1-indexed) -> YOLO class_id (0-indexed).

    Returns:
        class_ids: sorted list of 0-indexed YOLO ids actually present
        class_names: {class_id: human-readable name}
        coco_id_to_class_id: {coco_category_id: class_id}
    """
    coco_id_to_class_id = {}
    class_names = {}
    for cat in categories:
        coco_id = int(cat["id"])
        class_id = coco_id - 1  # 1-indexed -> 0-indexed
        coco_id_to_class_id[coco_id] = class_id
        class_names[class_id] = str(cat["name"])
    return sorted(class_names.keys()), class_names, coco_id_to_class_id


# -----------------------------
# Dataset reading
# -----------------------------
def collect_rows():
    coco = load_coco(INSTANCES_JSON)
    class_ids, class_names, coco_to_class = build_class_mapping(coco["categories"])
    feature_columns = {cid: safe_feature_name(class_names[cid]) for cid in class_ids}

    # Group annotations by image_id for O(N) lookup.
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    # Warn about category ids in annotations that are not in the categories list.
    known_coco_ids = set(coco_to_class.keys())
    unknown = {int(a["category_id"]) for a in coco["annotations"]
               if int(a["category_id"]) not in known_coco_ids}
    if unknown:
        print(f"[warn] annotations reference unknown category_ids that will be skipped: {sorted(unknown)}")

    rows = []
    for img in coco["images"]:
        file_name = str(img["file_name"])
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)
        denom = float(width * height) if width > 0 and height > 0 else 0.0

        anns = anns_by_image.get(int(img["id"]), [])
        counts = Counter()
        areas = defaultdict(float)
        for ann in anns:
            coco_cid = int(ann["category_id"])
            class_id = coco_to_class.get(coco_cid)
            if class_id is None:
                continue
            counts[class_id] += 1
            if denom > 0:
                ann_area = float(ann.get("area", 0.0))
                areas[class_id] += ann_area / denom
            # if denom is 0 we silently skip the area; counts still apply.

        total_instances = int(sum(counts.values()))
        species_id = parse_species_id_from_filename(file_name)

        # Compose paths the rest of the pipeline expects.
        image_path = os.path.normpath(str(IMAGES_DIR / file_name))
        label_basename = Path(file_name).with_suffix(".txt").name
        label_path = os.path.normpath(str(LABELS_DIR / label_basename))

        rows.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "image_name": file_name,
                "label_name": label_basename,
                "species_id": species_id,
                "species_name": species_id,
                "total_instances": total_instances,
                "trait_counts": {cid: int(counts.get(cid, 0)) for cid in class_ids},
                "trait_areas":  {cid: float(areas.get(cid, 0.0)) for cid in class_ids},
            }
        )

    return rows, class_names, class_ids, feature_columns


# -----------------------------
# Statistics for normalization (identical to the YOLO-based script)
# -----------------------------
def compute_dataset_statistics(rows, class_ids):
    global_trait_totals = {cid: 0 for cid in class_ids}
    species_trait_totals = defaultdict(int)
    species_image_counts = Counter()
    global_trait_presence = {cid: 0 for cid in class_ids}
    global_trait_areas = {cid: 0.0 for cid in class_ids}

    for row in rows:
        species = row["species_id"]
        species_image_counts[species] += 1
        for cid in class_ids:
            count = int(row["trait_counts"].get(cid, 0))
            area = float(row.get("trait_areas", {}).get(cid, 0.0))
            global_trait_totals[cid] += count
            species_trait_totals[(species, cid)] += count
            if count > 0:
                global_trait_presence[cid] += 1
            global_trait_areas[cid] += area

    return (
        global_trait_totals,
        species_trait_totals,
        species_image_counts,
        global_trait_presence,
        global_trait_areas,
    )


def species_reliability(species_id, species_image_counts):
    n = species_image_counts.get(species_id, 0)
    if MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT <= 0:
        return 1.0
    return min(1.0, n / float(MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT))


# -----------------------------
# Vector building
# -----------------------------
def build_output_rows(rows, class_names, class_ids, feature_columns,
                      global_trait_totals, species_trait_totals, species_image_counts):
    output_rows = []

    for row in rows:
        species = row["species_id"]
        total_instances = row["total_instances"]

        reliability = species_reliability(species, species_image_counts)
        effective_species_weight = SPECIES_WEIGHT * reliability
        effective_global_weight = GLOBAL_WEIGHT + SPECIES_WEIGHT * (1.0 - reliability)

        output_row = {
            "image_path": row["image_path"],
            "label_path": row["label_path"],
            "image_name": row["image_name"],
            "label_name": row["label_name"],
            "species_id": species,
            "species_name": species,
            "species_reliability": round(reliability, 6),
            "total_instances": total_instances,
        }

        for cid in class_ids:
            feature_name = feature_columns[cid]
            count = int(row["trait_counts"].get(cid, 0))
            area = float(row.get("trait_areas", {}).get(cid, 0.0))
            presence = 1 if count > 0 else 0

            if count <= 0:
                local_component = 0.0
                species_component = 0.0
                global_component = 0.0
                score = 0.0
            else:
                global_total = int(global_trait_totals.get(cid, 0))
                species_total = int(species_trait_totals.get((species, cid), 0))

                local_component = count / float(total_instances) if total_instances > 0 else 0.0
                species_component = count / float(species_total) if species_total > 0 else 0.0
                global_component = count / float(global_total) if global_total > 0 else 0.0

                score = (
                    LOCAL_WEIGHT * local_component
                    + effective_species_weight * species_component
                    + effective_global_weight * global_component
                )
                score = min(max(score, 0.0), 1.0)

            output_row[feature_name] = round(score, 6)

            if WRITE_PRESENCE_COLUMNS:
                output_row[f"presence_{feature_name}"] = presence
            if WRITE_AREA_COLUMNS:
                output_row[f"area_{feature_name}"] = round(area, 6)

            if WRITE_COMPONENT_COLUMNS:
                output_row[f"count_{feature_name}"] = count
                output_row[f"local_{feature_name}"] = round(local_component, 6)
                output_row[f"species_{feature_name}"] = round(species_component, 6)
                output_row[f"global_{feature_name}"] = round(global_component, 6)

        output_rows.append(output_row)

    return output_rows


# -----------------------------
# Writing output
# -----------------------------
def write_output_csv(output_rows, class_ids, feature_columns):
    base_columns = [
        "image_path",
        "label_path",
        "image_name",
        "label_name",
        "species_id",
        "species_name",
        "species_reliability",
        "total_instances",
    ]
    vector_columns = [feature_columns[cid] for cid in class_ids]

    fieldnames = list(base_columns) + list(vector_columns)
    if WRITE_PRESENCE_COLUMNS:
        for cid in class_ids:
            fieldnames.append(f"presence_{feature_columns[cid]}")
    if WRITE_AREA_COLUMNS:
        for cid in class_ids:
            fieldnames.append(f"area_{feature_columns[cid]}")
    if WRITE_COMPONENT_COLUMNS:
        for cid in class_ids:
            feature_name = feature_columns[cid]
            fieldnames.extend([
                f"count_{feature_name}",
                f"local_{feature_name}",
                f"species_{feature_name}",
                f"global_{feature_name}",
            ])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_schema(class_names, class_ids, feature_columns,
                 global_trait_totals, species_image_counts):
    schema = {
        "source": "coco-json",
        "instances_json": str(INSTANCES_JSON),
        "class_id_to_name": {str(cid): class_names[cid] for cid in class_ids},
        "feature_columns": [feature_columns[cid] for cid in class_ids],
        "n_features": len(class_ids),
        "value_range": [0.0, 1.0],
        "species_source": "species_id is the prefix of file_name before the first '-' or '_'",
        "vector_meaning": (
            "Continuous trait-intensity vector built from per-instance counts in the COCO JSON. "
            "Each score reflects trait weight inside the image, inside the same species, and in "
            "the full dataset, blended with weights LOCAL/SPECIES/GLOBAL."
        ),
        "formula": {
            "score": ("LOCAL_WEIGHT * local_component "
                      "+ effective_species_weight * species_component "
                      "+ effective_global_weight * global_component"),
            "local_component": "count_in_image_for_trait / total_annotated_instances_in_image",
            "species_component": "count_in_image_for_trait / total_count_for_trait_in_same_species",
            "global_component": "count_in_image_for_trait / total_count_for_trait_in_full_dataset",
            "effective_species_weight": "SPECIES_WEIGHT * species_reliability",
            "effective_global_weight": "GLOBAL_WEIGHT + SPECIES_WEIGHT * (1 - species_reliability)",
            "species_reliability": "min(1, number_of_images_for_species / MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT)",
            "presence": "1 if any annotation of the trait exists in the image, else 0",
            "area": "sum_over_polygons(annotation_area_px / (image_width * image_height))",
        },
        "weights": {
            "LOCAL_WEIGHT": LOCAL_WEIGHT,
            "SPECIES_WEIGHT": SPECIES_WEIGHT,
            "GLOBAL_WEIGHT": GLOBAL_WEIGHT,
            "MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT": MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT,
        },
        "global_trait_totals": {str(cid): int(global_trait_totals[cid]) for cid in class_ids},
        "species_image_counts": dict(species_image_counts),
    }
    OUTPUT_SCHEMA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCHEMA_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------
def main():
    if abs((LOCAL_WEIGHT + SPECIES_WEIGHT + GLOBAL_WEIGHT) - 1.0) > 1e-9:
        raise ValueError("LOCAL_WEIGHT + SPECIES_WEIGHT + GLOBAL_WEIGHT must add up to 1.0")

    rows, class_names, class_ids, feature_columns = collect_rows()
    if not rows:
        raise ValueError("No images in the COCO JSON")

    n_total = len(rows)
    n_empty = sum(1 for r in rows if r["total_instances"] == 0)
    empty_fraction = n_empty / float(n_total) if n_total else 0.0
    if empty_fraction > ALLOW_EMPTY_FRACTION:
        sample = [r["image_name"] for r in rows if r["total_instances"] == 0][:5]
        raise RuntimeError(
            f"{n_empty}/{n_total} images ({empty_fraction:.1%}) have zero annotations, "
            f"above ALLOW_EMPTY_FRACTION={ALLOW_EMPTY_FRACTION:.1%}. "
            f"Examples: {sample}. Check that 'category_id' values in annotations match "
            f"the 'id' values in 'categories'."
        )

    (
        global_trait_totals,
        species_trait_totals,
        species_image_counts,
        global_trait_presence,
        global_trait_areas,
    ) = compute_dataset_statistics(rows, class_ids)

    output_rows = build_output_rows(
        rows,
        class_names,
        class_ids,
        feature_columns,
        global_trait_totals,
        species_trait_totals,
        species_image_counts,
    )

    write_output_csv(output_rows, class_ids, feature_columns)
    write_schema(class_names, class_ids, feature_columns, global_trait_totals, species_image_counts)

    print(f"Source JSON       : {INSTANCES_JSON}")
    print(f"Images processed  : {n_total}")
    print(f"Empty images      : {n_empty} ({empty_fraction:.2%})")
    print(f"Total annotations : {sum(global_trait_totals.values())}")
    print(f"Feature dimension : {len(class_ids)}")
    print(f"Distinct species  : {len(species_image_counts)}")
    print(f"CSV written to    : {OUTPUT_CSV}")
    print(f"Schema written to : {OUTPUT_SCHEMA_JSON}")


if __name__ == "__main__":
    main()
