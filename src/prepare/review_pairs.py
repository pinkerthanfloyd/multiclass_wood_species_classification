"""
review_pairs.py
---------------
Recorre `proximity_pairs.csv`. Para cada par muestra una ventana emergente
con TRES paneles dispuestos de izquierda a derecha y la misma region
recortada en cada uno:

    [ A en rojo ]   [ A + B  (verde A, rojo B) ]   [ B en rojo ]

El usuario decide con una tecla:

    a -> eliminar A
    b -> eliminar B
    d -> eliminar ambos
    m -> juntar
    s -> mantener por separado (no se guarda nada)
    p -> volver al par anterior y deshacer su decision (se eliminan
         de las listas las filas que se hubieran anadido)
    q -> salir antes de terminar

NO modifica las anotaciones; solo registra la decision (en modo "append")
en uno de estos dos CSV:

    eliminar_rasgos.csv  -> rasgos individuales que el usuario marco para borrar
    juntar_rasgos.csv    -> pares que el usuario marco para fusionar

Uso:
    python review_pairs.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np


IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
PAIRS_CSV = Path("proximity_pairs.csv")
ELIMINAR_CSV = Path("eliminar_rasgos.csv")
JUNTAR_CSV = Path("juntar_rasgos.csv")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
MARGIN = 80                     # px de contexto alrededor del par recortado
WINDOW = "Pair Review"

# Colores BGR
RED_FILL, RED_EDGE = (0, 0, 200), (0, 0, 255)
GREEN_FILL, GREEN_EDGE = (0, 200, 0), (0, 255, 0)


def find_image(base):
    for ext in IMG_EXTS:
        p = IMAGES_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None


def get_polygon(label_file, idx, W, H):
    """Devuelve el contorno (Nx2 int) de la linea `idx` del .txt."""
    with open(label_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                vals = list(map(float, line.split()[1:]))
                return np.array(
                    [(vals[k] * W, vals[k + 1] * H)
                     for k in range(0, len(vals) - 1, 2)],
                    dtype=np.int32,
                )
    return None


def overlay(img, polys, fills, edges):
    """Pinta `polys` con relleno semitransparente y bordes solidos."""
    filled = img.copy()
    for p, c in zip(polys, fills):
        cv2.fillPoly(filled, [p], c)
    out = cv2.addWeighted(img, 0.55, filled, 0.45, 0)
    for p, c in zip(polys, edges):
        cv2.polylines(out, [p], True, c, 2)
    return out


def crop_box(shape, poly_a, poly_b):
    H, W = shape[:2]
    xs = np.concatenate([poly_a[:, 0], poly_b[:, 0]])
    ys = np.concatenate([poly_a[:, 1], poly_b[:, 1]])
    x0 = max(int(xs.min()) - MARGIN, 0)
    x1 = min(int(xs.max()) + MARGIN, W)
    y0 = max(int(ys.min()) - MARGIN, 0)
    y1 = min(int(ys.max()) + MARGIN, H)
    return x0, y0, x1, y1


def build_view(img, poly_a, poly_b):
    """Triple panel: A en rojo | A+B | B en rojo, con la misma region."""
    x0, y0, x1, y1 = crop_box(img.shape, poly_a, poly_b)

    only_a = overlay(img, [poly_a], [RED_FILL], [RED_EDGE])
    both = overlay(img, [poly_a, poly_b],
                   [GREEN_FILL, RED_FILL], [GREEN_EDGE, RED_EDGE])
    only_b = overlay(img, [poly_b], [RED_FILL], [RED_EDGE])

    panels = [only_a[y0:y1, x0:x1].copy(),
              both[y0:y1, x0:x1].copy(),
              only_b[y0:y1, x0:x1].copy()]

    for panel, label in zip(panels, ["A", "A + B", "B"]):
        cv2.putText(panel, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    sep = np.full((panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
    return np.hstack([panels[0], sep, panels[1], sep, panels[2]])


def feature_row(row, side):
    s = side.lower()
    return {
        "file": row["file"],
        "type_id": row["type_id"],
        "type_name": row["type_name"],
        "label_idx": row[f"idx_{s}"],
        "x_px": row[f"x_{s}_px"],
        "y_px": row[f"y_{s}_px"],
    }


def append_csv(path, rows, fieldnames):
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    with open(PAIRS_CSV, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    eliminar, juntar = [], []
    # Historial de decisiones para "p": (idx_par, n_filas_a_eliminar, n_filas_a_juntar)
    history = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(pairs):
        row = pairs[i]
        img_path = find_image(row["file"])
        label_file = LABELS_DIR / f"{row['file']}.txt"
        if img_path is None or not label_file.exists():
            print(f"[SKIP] {row['file']}: faltan archivos")
            i += 1
            continue

        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        poly_a = get_polygon(label_file, int(row["idx_a"]), W, H)
        poly_b = get_polygon(label_file, int(row["idx_b"]), W, H)
        if poly_a is None or poly_b is None:
            i += 1
            continue

        view = build_view(img, poly_a, poly_b)

        title = (f"[{i + 1}/{len(pairs)}] {row['file']} ({row['type_name']})  "
                 f"d={row['distance_px']}  a/b/d/m/s, p atras, q sale")
        cv2.setWindowTitle(WINDOW, title)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(0) & 0xFF
        c = chr(key).lower() if 32 <= key < 127 else ""

        if c == "q":
            print("[INFO] revision interrumpida")
            break

        if c == "p":
            # Deshacer la decision anterior y volver a su par
            if not history:
                print("[INFO] no hay decision previa para deshacer")
                continue
            prev_i, n_e, n_j = history.pop()
            for _ in range(n_e):
                eliminar.pop()
            for _ in range(n_j):
                juntar.pop()
            i = prev_i
            continue

        n_e = n_j = 0
        if c in ("a", "d"):
            eliminar.append(feature_row(row, "A"))
            n_e += 1
        if c in ("b", "d"):
            eliminar.append(feature_row(row, "B"))
            n_e += 1
        if c == "m":
            juntar.append(row)
            n_j += 1
        # 's' (o cualquier otra tecla) -> no se guarda nada

        history.append((i, n_e, n_j))
        i += 1

    cv2.destroyAllWindows()

    append_csv(
        ELIMINAR_CSV, eliminar,
        fieldnames=["file", "type_id", "type_name", "label_idx", "x_px", "y_px"],
    )
    append_csv(
        JUNTAR_CSV, juntar,
        fieldnames=list(pairs[0].keys()) if pairs else [],
    )

    print(f"[OK] {len(eliminar)} -> {ELIMINAR_CSV} | "
          f"{len(juntar)} -> {JUNTAR_CSV}")


if __name__ == "__main__":
    main()
