"""
compute_proximity.py
--------------------
Recorre las carpetas `images/` y `labels/` (pares con el mismo nombre base)
y, para cada anotacion YOLO-seg, calcula su contorno y centroide en pixeles.
Para cada rasgo busca sus vecinos del MISMO tipo en la misma imagen,
ordenados por distancia minima entre bordes (vertice-a-vertice como
aproximacion). Si esa distancia (px) es menor que el umbral del tipo
(PROXIMITY_THRESHOLD), el par se guarda en un CSV con:
    - archivo de origen
    - id y nombre del tipo (codificado segun data.yaml)
    - indice de la linea en el .txt (idx_a, idx_b) -> sirve para recuperar
      luego el contorno completo
    - coordenadas (centroides) de ambos rasgos
    - distancia minima entre bordes en pixeles

Optimizacion pedida: una vez ordenados los vecinos por esa distancia
ascendente, se rompe el bucle en cuanto aparece el primero que supera el
umbral.

Filtro por percentil P10: al menos uno de los dos rasgos del par tiene
que estar en el 10% mas pequeno de su clase (area < AREA_P10_THRESHOLD
para esa clase). Si ninguno cumple, el par se descarta y no se evalua.
Para las clases no especificadas en el diccionario el umbral es 99999.

Deteccion de duplicados: para cada par candidato (que ya pasa el filtro
P10) se calcula el porcentaje de interseccion relativo al rasgo mas
grande ( interseccion / max(area_a, area_b) * 100 ). Si supera
DUPLICATE_OVERLAP_PCT, se considera anotacion repetida: NO se anade el
par a `proximity_pairs.csv`; en su lugar se descarta automaticamente la
anotacion con MENOS vertices y se registra en `auto_duplicates.csv`.

Uso:
    python compute_proximity.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import yaml


# ---- Parametros configurables -------------------------------------------

IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
DATA_YAML = Path("data.yaml")
OUTPUT_CSV = Path("proximity_pairs.csv")
DUPLICATES_CSV = Path("auto_duplicates.csv")

PROXIMITY_THRESHOLD = defaultdict(lambda: 20, {i: 20 for i in range(14)})

# Umbral de area por clase = percentil 10 (10% mas pequenos) calculado
# previamente sobre el dataset. Un par solo se considera si AL MENOS uno
# de sus dos rasgos tiene area < AREA_P10_THRESHOLD[clase].
# Las clases no listadas usan 99999 (= practicamente "todo cuenta como
# pequeno"), de modo que no se filtran por tamano.
AREA_P10_THRESHOLD = defaultdict(lambda: 99999, {
    1: 1073.2,    # '56'
    2: 1169.4,    # '58'
    3: 1630.6,    # '79'
    4: 5710.0,    # '81'
    6: 5999.8,    # '83'
    8: 37492.9,   # '85'
    12: 527.0,    # V1
})

# Si la interseccion relativa al rasgo mayor supera este % se considera
# anotacion duplicada y se descarta la de menos vertices.
DUPLICATE_OVERLAP_PCT = 95.0

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# -------------------------------------------------------------------------


def load_class_names(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def find_image(base: str):
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def polygon_centroid(points):
    n = len(points)
    return (sum(p[0] for p in points) / n,
            sum(p[1] for p in points) / n)


def polygon_area(points: np.ndarray) -> float:
    """Area aproximada (formula de Shoelace) en pixeles."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def min_edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Aproximacion a la minima distancia entre los bordes de dos
    poligonos: minima distancia vertice-a-vertice. Es exacta cuando
    los vertices son densos (caso tipico de YOLO-seg)."""
    diff = a[:, None, :] - b[None, :, :]          # (Na, Nb, 2)
    return float(np.sqrt((diff ** 2).sum(-1)).min())


def overlap_pct_vs_bigger(a: np.ndarray, b: np.ndarray) -> float:
    """Porcentaje de area del rasgo mayor ocupado por el menor:
        interseccion / max(area_a, area_b) * 100
    Si los poligonos son invalidos (auto-interseccion etc.) se reparan
    con .buffer(0). Devuelve 0 si alguna area es 0."""
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    bigger = max(pa.area, pb.area)
    if bigger <= 0:
        return 0.0
    return float(pa.intersection(pb).area / bigger * 100.0)


def main():
    class_names = load_class_names(DATA_YAML)
    pairs = []
    duplicates = []

    for label_file in sorted(LABELS_DIR.glob("*.txt")):
        base = label_file.stem
        img_path = find_image(base)
        if img_path is None:
            print(f"[WARN] sin imagen para {label_file.name}, salto")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        # 1) Cargamos TODOS los rasgos del archivo (sin filtro de area)
        feats = []
        with open(label_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 1 + 4:  # clase + al menos 2 puntos
                    continue
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
                pts = [
                    (vals[i] * W, vals[i + 1] * H)
                    for i in range(0, len(vals) - 1, 2)
                ]
                if len(pts) < 2:
                    continue
                cx, cy = polygon_centroid(pts)
                pts_np = np.asarray(pts, dtype=np.float32)
                feats.append(
                    {
                        "idx": idx,
                        "type": cls,
                        "x": cx,
                        "y": cy,
                        "pts": pts_np,
                        "area": polygon_area(pts_np),
                    }
                )

        # 2) Para cada rasgo, vecinos del MISMO tipo ordenados por la
        #    distancia minima entre sus bordes (vertice-a-vertice).
        seen = set()
        discarded = set()  # indices descartados por duplicado
        for i, f1 in enumerate(feats):
            if i in discarded:
                continue
            thr = PROXIMITY_THRESHOLD[f1["type"]]
            cands = []
            for j, f2 in enumerate(feats):
                if (j == i or j in discarded
                        or f2["type"] != f1["type"]):
                    continue
                d = min_edge_distance(f1["pts"], f2["pts"])
                cands.append((d, j))
            cands.sort(key=lambda t: t[0])

            for d, j in cands:
                if i in discarded:
                    break
                if j in discarded:
                    continue
                if d > thr:
                    break  # los siguientes solo seran mayores
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                f2 = feats[j]

                # Filtro P10: al menos uno tiene que estar entre el 10%
                # mas pequeno de su clase (ambos son del mismo tipo).
                area_thr = AREA_P10_THRESHOLD[f1["type"]]
                if f1["area"] >= area_thr and f2["area"] >= area_thr:
                    continue

                # Interseccion relativa al rasgo mas grande
                pct = overlap_pct_vs_bigger(f1["pts"], f2["pts"])

                if pct > DUPLICATE_OVERLAP_PCT:
                    # Anotacion repetida: descartamos la de MENOS vertices
                    n_i, n_j = len(f1["pts"]), len(f2["pts"])
                    kill = i if n_i < n_j else j
                    keep = j if kill == i else i
                    discarded.add(kill)
                    fk, fp = feats[kill], feats[keep]
                    duplicates.append(
                        {
                            "file": base,
                            "type_id": fk["type"],
                            "type_name": class_names.get(fk["type"], str(fk["type"])),
                            "label_idx": fk["idx"],
                            "x_px": round(fk["x"], 2),
                            "y_px": round(fk["y"], 2),
                            "kept_label_idx": fp["idx"],
                            "n_vertices_discarded": len(fk["pts"]),
                            "n_vertices_kept": len(fp["pts"]),
                            "intersection_pct": round(pct, 2),
                        }
                    )
                    # No se anade el par; el bucle externo lo skipea si
                    # `i` quedo descartado, o seguimos con el siguiente j.
                    continue

                pairs.append(
                    {
                        "file": base,
                        "type_id": f1["type"],
                        "type_name": class_names.get(f1["type"], str(f1["type"])),
                        "idx_a": f1["idx"],
                        "x_a_px": round(f1["x"], 2),
                        "y_a_px": round(f1["y"], 2),
                        "idx_b": f2["idx"],
                        "x_b_px": round(f2["x"], 2),
                        "y_b_px": round(f2["y"], 2),
                        "distance_px": round(d, 2),
                        "intersection_pct": round(pct, 2),
                    }
                )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "type_id", "type_name",
                "idx_a", "x_a_px", "y_a_px",
                "idx_b", "x_b_px", "y_b_px",
                "distance_px", "intersection_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)

    with open(DUPLICATES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "type_id", "type_name",
                "label_idx", "x_px", "y_px",
                "kept_label_idx",
                "n_vertices_discarded", "n_vertices_kept",
                "intersection_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(duplicates)

    print(f"[OK] {len(pairs)} pares cercanos -> {OUTPUT_CSV.resolve()}")
    print(f"[OK] {len(duplicates)} duplicados descartados -> {DUPLICATES_CSV.resolve()}")


if __name__ == "__main__":
    main()
