import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# -----------------------------
# Configuration
# -----------------------------
DATASET_ROOT = "data/raw_selected"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")

# Optional. If you have a txt listing image routes, set it here.
# Leave as None to scan IMAGES_DIR recursively.
IMAGE_INDEX_TXT = None

OUTPUT_CSV = os.path.join("outputs/relative_feature_vectors.csv")
OUTPUT_SCHEMA_JSON = os.path.join("outputs/relative_feature_vectors.schema.json")

# Final score for one image and one trait:
# score = LOCAL_WEIGHT * local_component
#       + effective_species_weight * species_component
#       + effective_global_weight * global_component
#
# local_component   = count of trait in the image / total annotated instances in the image
# species_component = count of trait in the image / total count of that trait in the same species
# global_component  = count of trait in the image / total count of that trait in the whole dataset
#
# Species contribution is softened when a species has very few images.
LOCAL_WEIGHT = 0.50
SPECIES_WEIGHT = 0.20
GLOBAL_WEIGHT = 0.30
MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT = 5

# Keep True if you want the CSV to include raw counts and components for inspection.
WRITE_COMPONENT_COLUMNS = True
# Always write presence_feat_* (binary) and area_feat_* (sum of normalized polygon areas).
# These are the columns downstream consumers (split generation, balanced sampling,
# class-aware metrics) typically rely on.
WRITE_PRESENCE_COLUMNS = True
WRITE_AREA_COLUMNS = True

# Hard guard: if more than this fraction of images end up with zero instances, abort.
# A high empty rate almost always means label files are not being found and is a
# silent bug we want to fail loudly on.
ALLOW_EMPTY_FRACTION = 0.05

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Make sure the output directory exists before writing.
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)


def normalize_path_str(path_value):
    """Convert any backslashes to forward slashes for consistent processing.

    YOLO index txt files written on Windows often mix separators
    (e.g. 'data/images/train\\0001-1.jpg'). On POSIX systems the backslash is
    not a separator, which breaks os.path.basename / Path.stem and silently
    makes label resolution fail.
    """
    if path_value is None:
        return ""
    return str(path_value).replace("\\", "/")


# -----------------------------
# Helpers
# -----------------------------
def safe_feature_name(name):
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    return f"feat_{text}"


def load_class_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        return {int(i): str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    raise ValueError("Could not read class names from data.yaml")


def parse_species_id_from_filename(file_name):
    # Normalize Windows-style backslashes before extracting the basename.
    normalized = normalize_path_str(file_name).split("/")[-1]
    stem = Path(normalized).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def read_index_paths(txt_path):
    if not txt_path or not os.path.exists(txt_path):
        return []

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    resolved = []
    for line in raw_lines:
        candidates = []

        if os.path.isabs(line):
            candidates.append(line)
        else:
            candidates.append(os.path.normpath(os.path.join(DATASET_ROOT, line)))
            candidates.append(os.path.normpath(os.path.join(IMAGES_DIR, line)))

            normalized = line.replace("\\", "/")
            if "/images/" in normalized:
                tail = normalized.split("/images/", 1)[1]
                candidates.append(os.path.normpath(os.path.join(IMAGES_DIR, tail)))

            candidates.append(os.path.normpath(os.path.join(IMAGES_DIR, os.path.basename(line))))

        chosen = None
        for candidate in candidates:
            if os.path.exists(candidate):
                chosen = candidate
                break

        if chosen is None:
            chosen = candidates[0]
        resolved.append(chosen)

    return resolved


def scan_images(images_dir):
    image_paths = []
    for path in Path(images_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(path))
    return sorted(image_paths)


def build_label_index(labels_dir):
    by_stem = defaultdict(list)
    all_paths = []
    for path in Path(labels_dir).rglob("*.txt"):
        all_paths.append(path)
        by_stem[path.stem].append(path)
    return by_stem, all_paths


def resolve_label_path(image_path, images_dir, labels_dir, label_index):
    image_path_obj = Path(image_path)

    # 1) Preserve the relative path below IMAGES_DIR.
    try:
        rel = image_path_obj.relative_to(Path(images_dir))
        candidate = Path(labels_dir) / rel.with_suffix(".txt")
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    # 2) Replace a visible /images/ segment with /labels/.
    normalized = str(image_path_obj).replace("\\", "/")
    if "/images/" in normalized:
        candidate = normalized.replace("/images/", "/labels/")
        candidate = os.path.splitext(candidate)[0] + ".txt"
        if os.path.exists(candidate):
            return os.path.normpath(candidate)

    # 3) Try aggregate labels directory with the same basename.
    candidate = Path(labels_dir) / f"{image_path_obj.stem}.txt"
    if candidate.exists():
        return str(candidate)

    # 4) Fallback to stem index.
    matches = label_index.get(image_path_obj.stem, [])
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        return str(sorted(matches)[0])

    return ""


def _polygon_area(coords):
    """Shoelace area for a flat list of normalized [x1,y1,x2,y2,...]. Result in [0,1]."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def parse_traits_in_label_file(label_path):
    """Parse a YOLO segmentation label file and return per-class counts and area sums.

    Each line in the file has the form: class_id x1 y1 x2 y2 ... xn yn (normalized).
    The returned area is the sum of normalized polygon areas for that class, so values
    are roughly in [0, 1] (or above 1 if multiple polygons of the same class overlap
    or together cover more than one image-equivalent of area, which is rare).
    """
    counts = Counter()
    areas = defaultdict(float)
    if not label_path or not os.path.exists(label_path):
        return counts, areas

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            counts[class_id] += 1
            areas[class_id] += _polygon_area(coords)
    return counts, areas


def count_traits_in_label_file(label_path):
    """Backward-compatible wrapper that returns only counts."""
    counts, _ = parse_traits_in_label_file(label_path)
    return counts


# -----------------------------
# Dataset reading
# -----------------------------
def collect_rows():
    class_names = load_class_names(YAML_PATH)
    class_ids = sorted(class_names.keys())
    feature_columns = {class_id: safe_feature_name(class_names[class_id]) for class_id in class_ids}

    if IMAGE_INDEX_TXT:
        image_paths = read_index_paths(IMAGE_INDEX_TXT)
    else:
        image_paths = scan_images(IMAGES_DIR)

    label_index, _ = build_label_index(LABELS_DIR)

    rows = []
    for image_path in image_paths:
        normalized_image_path = normalize_path_str(image_path)
        label_path = resolve_label_path(normalized_image_path, IMAGES_DIR, LABELS_DIR, label_index)
        normalized_label_path = normalize_path_str(label_path) if label_path else ""

        counts, areas = parse_traits_in_label_file(normalized_label_path)
        total_instances = int(sum(counts.values()))
        image_name = normalize_path_str(image_path).split("/")[-1]
        species_id = parse_species_id_from_filename(image_name)

        rows.append(
            {
                "image_path": os.path.normpath(normalized_image_path),
                "label_path": os.path.normpath(normalized_label_path) if normalized_label_path else "",
                "image_name": image_name,
                "label_name": (normalized_label_path.split("/")[-1] if normalized_label_path else ""),
                "species_id": species_id,
                "species_name": species_id,
                "total_instances": total_instances,
                "trait_counts": {class_id: int(counts.get(class_id, 0)) for class_id in class_ids},
                "trait_areas": {class_id: float(areas.get(class_id, 0.0)) for class_id in class_ids},
            }
        )

    return rows, class_names, class_ids, feature_columns


# -----------------------------
# Statistics for normalization
# -----------------------------
def compute_dataset_statistics(rows, class_ids):
    global_trait_totals = {class_id: 0 for class_id in class_ids}
    species_trait_totals = defaultdict(int)
    species_image_counts = Counter()
    global_trait_presence = {class_id: 0 for class_id in class_ids}
    global_trait_areas = {class_id: 0.0 for class_id in class_ids}

    for row in rows:
        species = row["species_id"]
        species_image_counts[species] += 1

        for class_id in class_ids:
            count = int(row["trait_counts"].get(class_id, 0))
            area = float(row.get("trait_areas", {}).get(class_id, 0.0))
            global_trait_totals[class_id] += count
            species_trait_totals[(species, class_id)] += count
            if count > 0:
                global_trait_presence[class_id] += 1
            global_trait_areas[class_id] += area

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
def build_output_rows(rows, class_names, class_ids, feature_columns, global_trait_totals, species_trait_totals, species_image_counts):
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

        for class_id in class_ids:
            feature_name = feature_columns[class_id]
            count = int(row["trait_counts"].get(class_id, 0))
            area = float(row.get("trait_areas", {}).get(class_id, 0.0))
            presence = 1 if count > 0 else 0

            if count <= 0:
                local_component = 0.0
                species_component = 0.0
                global_component = 0.0
                score = 0.0
            else:
                global_total = int(global_trait_totals.get(class_id, 0))
                species_total = int(species_trait_totals.get((species, class_id), 0))

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
    vector_columns = [feature_columns[class_id] for class_id in class_ids]

    fieldnames = list(base_columns)
    fieldnames.extend(vector_columns)

    if WRITE_PRESENCE_COLUMNS:
        for class_id in class_ids:
            fieldnames.append(f"presence_{feature_columns[class_id]}")

    if WRITE_AREA_COLUMNS:
        for class_id in class_ids:
            fieldnames.append(f"area_{feature_columns[class_id]}")

    if WRITE_COMPONENT_COLUMNS:
        for class_id in class_ids:
            feature_name = feature_columns[class_id]
            fieldnames.extend(
                [
                    f"count_{feature_name}",
                    f"local_{feature_name}",
                    f"species_{feature_name}",
                    f"global_{feature_name}",
                ]
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def write_schema(class_names, class_ids, feature_columns, global_trait_totals, species_image_counts):
    schema = {
        "class_id_to_name": {str(class_id): class_names[class_id] for class_id in class_ids},
        "feature_columns": [feature_columns[class_id] for class_id in class_ids],
        "n_features": len(class_ids),
        "value_range": [0.0, 1.0],
        "species_source": "species_id is extracted from the filename prefix before the first '-' or '_'",
        "vector_meaning": "Continuous trait-intensity vector built from YOLO segmentation instance counts. Each score reflects trait weight inside the image, inside the same species, and in the full dataset.",
        "formula": {
            "score": "LOCAL_WEIGHT * local_component + effective_species_weight * species_component + effective_global_weight * global_component",
            "local_component": "count_in_image_for_trait / total_annotated_instances_in_image",
            "species_component": "count_in_image_for_trait / total_count_for_trait_in_same_species",
            "global_component": "count_in_image_for_trait / total_count_for_trait_in_full_dataset",
            "effective_species_weight": "SPECIES_WEIGHT * species_reliability",
            "effective_global_weight": "GLOBAL_WEIGHT + SPECIES_WEIGHT * (1 - species_reliability)",
            "species_reliability": "min(1, number_of_images_for_species / MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT)",
        },
        "weights": {
            "LOCAL_WEIGHT": LOCAL_WEIGHT,
            "SPECIES_WEIGHT": SPECIES_WEIGHT,
            "GLOBAL_WEIGHT": GLOBAL_WEIGHT,
            "MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT": MIN_IMAGES_FOR_FULL_SPECIES_WEIGHT,
        },
        "global_trait_totals": {str(class_id): int(global_trait_totals[class_id]) for class_id in class_ids},
        "species_image_counts": dict(species_image_counts),
    }

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
        raise ValueError("No images found to build vectors")

    # Loud diagnostic: catch the silent "labels not found" failure mode early.
    n_empty = sum(1 for row in rows if row["total_instances"] == 0)
    n_total = len(rows)
    empty_fraction = n_empty / float(n_total) if n_total else 0.0
    if empty_fraction > ALLOW_EMPTY_FRACTION:
        sample = [r["image_path"] for r in rows if r["total_instances"] == 0][:5]
        raise RuntimeError(
            f"{n_empty}/{n_total} images ({empty_fraction:.1%}) ended with zero "
            f"instances, above ALLOW_EMPTY_FRACTION={ALLOW_EMPTY_FRACTION:.1%}. "
            "This usually means the label files are not being resolved correctly. "
            f"Check DATASET_ROOT={DATASET_ROOT}, IMAGES_DIR={IMAGES_DIR}, "
            f"LABELS_DIR={LABELS_DIR}. Examples of empty images: {sample}"
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

    print(f"Images processed: {len(output_rows)}")
    print(f"Feature dimension: {len(class_ids)}")
    print(f"Images with at least one instance: {n_total - n_empty}/{n_total}")
    print(f"CSV written to: {OUTPUT_CSV}")
    print(f"Schema written to: {OUTPUT_SCHEMA_JSON}")


if __name__ == "__main__":
    main()
