"""
compute_features.py
-------------------
Recorre las carpetas `images/` y `labels/` (pares de archivos con el mismo
nombre base) y, para cada anotacion tipo YOLO-seg, calcula su area
aproximada en pixeles usando la formula de Shoelace sobre el poligono.

Si el area cae POR DEBAJO del umbral correspondiente al tipo
(FEATURE_THRESHOLD), la anotacion se considera anomalia (demasiado
pequena) y se vuelca a un CSV con:
    - archivo de origen
    - id y nombre del tipo (segun data.yaml)
    - indice de la linea en el .txt (label_idx) -> permite borrarla con
      precision en clean_labels.py
    - coordenadas aproximadas (centroide del poligono)
    - area en pixeles

Formato YOLO-seg: cada linea -> "<class_id> x1 y1 x2 y2 ... xn yn"
con coordenadas normalizadas en [0, 1].

Uso:
    python compute_features.py
"""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
import yaml


# ---- Parametros configurables -------------------------------------------

IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
DATA_YAML = Path("data.yaml")
OUTPUT_CSV = Path("flagged_features.csv")

FEATURE_THRESHOLD = defaultdict(lambda: 20, {i: 20 for i in range(14)})

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# -------------------------------------------------------------------------


def load_class_names(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def polygon_area(points):
    """Area de un poligono cerrado (formula de Shoelace)."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_centroid(points):
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return cx, cy


def find_image(base: str):
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def main():
    class_names = load_class_names(DATA_YAML)
    rows = []

    for label_file in sorted(LABELS_DIR.glob("*.txt")):
        base = label_file.stem
        img_path = find_image(base)
        if img_path is None:
            print(f"[WARN] sin imagen para {label_file.name}, salto")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        with open(label_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 1 + 6:
                    continue
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
                points = [
                    (vals[i] * W, vals[i + 1] * H)
                    for i in range(0, len(vals) - 1, 2)
                ]
                area = polygon_area(points)
                # Anomalia = rasgo POR DEBAJO del umbral (demasiado pequeno)
                if area < FEATURE_THRESHOLD[cls]:
                    cx, cy = polygon_centroid(points)
                    rows.append(
                        {
                            "file": base,
                            "type_id": cls,
                            "type_name": class_names.get(cls, str(cls)),
                            "label_idx": idx,
                            "x_center_px": round(cx, 2),
                            "y_center_px": round(cy, 2),
                            "area_px": round(area, 2),
                        }
                    )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "type_id", "type_name", "label_idx",
                "x_center_px", "y_center_px", "area_px",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {len(rows)} anomalias detectadas -> {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
