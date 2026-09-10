"""species_annotation_proximity.py
=================================

Análisis exploratorio de PROXIMIDAD entre especies a partir de las anotaciones
predichas (o GT en formato COCO) por el primer modelo de segmentación con
pérdida semántica (sem_loss). Construye una noción de distancia entre
especies justificable como cota superior de lo que un clasificador RF/SVM
entrenado sobre estas anotaciones puede llegar a discriminar y, en
consecuencia, de la calidad esperable de sus predicciones top-k.

Alineación con los criterios discutidos:

  * Imágenes de MACROSCOPÍA de madera: cada anotación es un rasgo anatómico
    (vasos, radios, parénquima, etc.). La firma de una especie no es la
    posición absoluta de los rasgos sino su MEZCLA y sus RELACIONES.
  * Invariancia a la posición: no usamos ni un solo coordinate xy crudo.
    Los descriptores geométricos por anotación son invariantes a
    traslación (sphericity, elongación PCA, aspect ratio del bbox,
    area-fraction). Los descriptores relacionales son co-ocurrencia
    Jaccard entre clases en la misma imagen, independientes del lugar
    donde caen los polígonos.
  * Presencia / correlación: bloque B1 = presencia + área-fracción por
    clase (replica los priors del sem_loss); bloque B3 = Jaccard de
    co-ocurrencia clase-clase a nivel imagen.
  * Tamaño / distribución / morfología: bloque B2 = media y desviación
    por clase de log-área, aspect-ratio, sphericity y elongación.
  * Granularidad: se produce CSV por anotación, por imagen y por especie.
    La distancia entre especies se calcula sobre el par (imagen,
    anotación) — agregando primero por imagen y después por especie — en
    lugar de sólo por anotación, porque la co-ocurrencia y la mezcla
    sólo emergen al respetar la imagen como contexto. Ver discusión al
    pie de este docstring.

Entradas:

  --coco-json PATH         COCO con images + annotations + categories
                           (p. ej. sem_loss_stratified).
  --especies-json PATH     Mapping id->Latin name (opcional, para legibilidad).

Salidas (en --output-dir):

  per_annotation_features.csv  Una fila por anotación, con descriptores
                               geométricos invariantes a traslación.
  per_image_features.csv       Una fila por imagen con la firma agregada
                               (presencia + área-fracción + morfología media
                               + clases presentes).
  species_signatures.csv       Una fila por especie con la firma combinada
                               por bloques B1+B2+B3.
  species_cooccurrence/        Matriz Jaccard C×C por especie, en npz.
  species_distance.csv         Matriz S×S de distancia coseno entre especies
                               (combinación ponderada de bloques).
  species_separability.csv     Por par de especies: distancia, intra_a,
                               intra_b, contraste = inter - max(intra_a,
                               intra_b). Pares de contraste bajo = especies
                               difíciles de separar para un RF/SVM.
  species_topk_neighbors.csv   Para cada especie, sus K vecinos más
                               cercanos con nombre latino. Aproximación
                               directa a las TOP-K predicciones
                               esperables de un clasificador entrenado
                               con la misma representación.
  species_distance.png         Heatmap (si --plot y matplotlib disponible).

Justificación:

  * Distancia COSENO sobre vectores L1-normalizados por bloque: ignora la
    magnitud absoluta del vector (sesgada por anotación selectiva /
    parcial) y mide sólo la FORMA de la mezcla — que es la firma
    distintiva de la especie. Para presencia/área es lo natural; para
    co-ocurrencia Jaccard también.
  * Pesos por bloque (--weight-*): el operador puede priorizar B1
    (replicar priors del sem_loss), B2 (morfología) o B3 (relaciones).
    Por defecto se reparte uniforme entre los tres bloques.

Lectura del par (imagen, anotación) frente a anotación únicamente
--------------------------------------------------------------------
Una anotación aislada lleva sólo descriptores intrínsecos: clase, área,
forma. Pierde el contexto de qué OTROS rasgos aparecen en la misma
imagen — exactamente la información que un RF/SVM aprovecha cuando se
entrena sobre vectores por imagen. Si se usa la anotación como unidad,
todas las anotaciones de clase 'V1' acaban en una sola nube en el
espacio de descriptores, y la diferenciación entre dos especies con la
misma clase mayoritaria queda como ruido alrededor de la media de la
clase. Con el par (imagen, anotación) preservamos la pertenencia y
podemos:

  - calcular Jaccard de co-ocurrencia entre clases por imagen,
  - estimar la proporción relativa de área de cada clase dentro de la
    imagen (independiente del tamaño absoluto del corte),
  - estimar la morfología media por clase EN UNA imagen y propagarla a
    la especie.

Lo recomendado es trabajar siempre con el par (imagen, anotación) — que
es lo que este script hace internamente — y exponer la vista
'anotación únicamente' sólo como CSV diagnóstico (per_annotation_features.csv).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utilidades
# ============================================================
def parse_species_id_from_filename(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/").split("/")[-1]).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def load_especies_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Acepta claves "0030" o "30"; normaliza eliminando ceros a la izquierda.
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).lstrip("0") or "0"
        out[key] = str(v)
        out[str(k)] = str(v)  # también la clave original
    return out


# ============================================================
# Geometría de polígonos (sólo numpy, sin shapely / scipy)
# ============================================================
def _flatten_segmentation(segmentation) -> Optional[np.ndarray]:
    """Devuelve array (N, 2) en píxeles del polígono más grande del segmento.

    Acepta la convención COCO: lista de listas o lista plana. Para
    multipolígonos se queda con la componente de mayor área (Shoelace).
    """
    if segmentation is None:
        return None
    polys: List[np.ndarray] = []
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        for seg in segmentation:
            if len(seg) >= 6 and len(seg) % 2 == 0:
                polys.append(np.asarray(seg, dtype=np.float64).reshape(-1, 2))
    elif isinstance(segmentation, list) and segmentation:
        seg = segmentation
        if len(seg) >= 6 and len(seg) % 2 == 0:
            polys.append(np.asarray(seg, dtype=np.float64).reshape(-1, 2))
    if not polys:
        return None
    # quedarse con la componente de mayor área
    areas = [abs(_shoelace(p)) for p in polys]
    return polys[int(np.argmax(areas))]


def _shoelace(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _perimeter(pts: np.ndarray) -> float:
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def _pca_elongation(pts: np.ndarray) -> float:
    """Elongación = sqrt(lambda_max / lambda_min) sobre los vértices.

    Invariante a traslación (centramos) y a rotación. 1.0 = isotrópico.
    """
    c = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(c, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 1.0
    w, _ = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-12)
    return float(np.sqrt(w.max() / w.min()))


def annotation_geometry(ann: dict, im_w: int, im_h: int) -> Optional[dict]:
    """Vector de descriptores invariantes a traslación de UNA anotación.

    Devuelve None si la geometría es degenerada.
    """
    pts = _flatten_segmentation(ann.get("segmentation"))
    if pts is None or len(pts) < 3:
        return None
    area_px = abs(_shoelace(pts))
    if area_px <= 0:
        return None
    img_area = float(im_w) * float(im_h) if im_w and im_h else 0.0
    perim_px = _perimeter(pts)
    # Sphericity / isoperimetric quotient en 2D = 4πA / P²; 1 = círculo.
    sphericity = (4.0 * np.pi * area_px) / (perim_px ** 2) if perim_px > 0 else 0.0
    sphericity = float(np.clip(sphericity, 0.0, 1.0))
    # bbox del polígono (no usamos el bbox COCO porque no siempre está consistente)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bw = max(float(xmax - xmin), 1e-6)
    bh = max(float(ymax - ymin), 1e-6)
    aspect = bw / bh
    log_aspect = float(np.log(aspect))
    bbox_extent = float(area_px / (bw * bh))  # solidez relativa al bbox
    elong = _pca_elongation(pts)
    log_elong = float(np.log(max(elong, 1e-6)))
    area_frac = area_px / img_area if img_area > 0 else 0.0
    log_area = float(np.log(max(area_frac, 1e-12)))
    return {
        "area_frac": float(area_frac),
        "log_area": log_area,
        "sphericity": sphericity,
        "log_aspect": log_aspect,
        "log_elong": log_elong,
        "bbox_extent": bbox_extent,
    }


SHAPE_KEYS = ("log_area", "sphericity", "log_aspect", "log_elong", "bbox_extent")


# ============================================================
# B4: proximidad espacial inter-clase con umbral adaptativo por imagen
# ============================================================
def annotation_centroid_and_bbox(ann: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pts = _flatten_segmentation(ann.get("segmentation"))
    if pts is None or len(pts) < 3:
        return None
    cx, cy = pts.mean(axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return np.array([cx, cy], dtype=np.float64), np.array([xmin, ymin, xmax, ymax], dtype=np.float64)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, a[2] - a[0]); ah = max(0.0, a[3] - a[1])
    bw = max(0.0, b[2] - b[0]); bh = max(0.0, b[3] - b[1])
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fracción del área de a que cae dentro del bbox de b (asimétrica)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aw = max(0.0, a[2] - a[0]); ah = max(0.0, a[3] - a[1])
    area_a = aw * ah
    return float(inter / area_a) if area_a > 0 else 0.0


def per_image_spatial_relations(
    coco: dict, coco_ids: List[int], q_close: float = 0.25
) -> pd.DataFrame:
    """Por imagen, calcula para cada par ordenado (a,b) de clases:
      rel_dist[a,b]    = mediana de d(centroide_a, centroide_b) / mediana_d_imagen
      close_frac[a,b]  = fracción de pares (a,b) con d_norm <= Q_close de la imagen
      iou[a,b]         = IoU medio de bboxes a vs b
      contain_a_b[a,b] = fracción media del bbox a contenido en bbox b

    El umbral 'cerca' es Q_close-cuantil de las distancias par a par EN ESA
    imagen — adaptativo, no global. La normalización por la mediana de la
    imagen elimina el sesgo del aumento y del recorte: "cerca" es relativo
    al par típico de la imagen, no a un número absoluto.
    """
    images = {int(im["id"]): im for im in coco["images"]}
    # Acumulador por imagen
    L = len(coco_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(coco_ids)}
    rows = []
    # Agrupar anotaciones por imagen
    from collections import defaultdict
    by_img: Dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[int(ann["image_id"])].append(ann)

    for iid, anns in by_img.items():
        im = images.get(iid)
        if im is None:
            continue
        W = float(im.get("width", 0))
        H = float(im.get("height", 0))
        diag = float(np.hypot(W, H)) if W > 0 and H > 0 else 1.0
        prepared = []
        for ann in anns:
            geom = annotation_centroid_and_bbox(ann)
            if geom is None:
                continue
            centroid, bbox = geom
            prepared.append((cid_to_idx.get(int(ann["category_id"])), centroid, bbox))
        prepared = [p for p in prepared if p[0] is not None]
        n = len(prepared)
        if n < 2:
            continue
        cents = np.vstack([p[1] for p in prepared])
        # Matriz de distancias entre centroides, normalizada por la diagonal
        D = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)) / diag
        iu = np.triu_indices(n, k=1)
        if iu[0].size == 0:
            continue
        d_vec = D[iu]
        d_med = float(np.median(d_vec)) if d_vec.size > 0 else 0.0
        if d_med <= 0:
            d_med = 1e-12
        close_thr = float(np.quantile(d_vec, q_close))
        rel = D / d_med  # adaptativo a la imagen

        # Agrupar índices por clase
        by_cls: Dict[int, list] = defaultdict(list)
        for k, p in enumerate(prepared):
            by_cls[p[0]].append(k)

        row = {"image_id": iid, "species_id": parse_species_id_from_filename(im["file_name"])}
        for a in range(L):
            for b in range(L):
                ia = by_cls.get(a, []); ib = by_cls.get(b, [])
                if not ia or not ib:
                    row[f"reldist_{a}_{b}"] = 0.0
                    row[f"closefrac_{a}_{b}"] = 0.0
                    row[f"iou_{a}_{b}"] = 0.0
                    row[f"contain_{a}_{b}"] = 0.0
                    continue
                # pares cross-class (incluyendo a==b para distancias intra-clase)
                d_block = D[np.ix_(ia, ib)]
                rel_block = rel[np.ix_(ia, ib)]
                if a == b and len(ia) > 1:
                    mask = ~np.eye(len(ia), dtype=bool)
                    d_vals = d_block[mask]
                    rel_vals = rel_block[mask]
                else:
                    d_vals = d_block.flatten()
                    rel_vals = rel_block.flatten()
                if d_vals.size == 0:
                    row[f"reldist_{a}_{b}"] = 0.0
                    row[f"closefrac_{a}_{b}"] = 0.0
                else:
                    row[f"reldist_{a}_{b}"] = float(np.median(rel_vals))
                    row[f"closefrac_{a}_{b}"] = float((d_vals <= close_thr).mean())
                # IoU y contención asimétrica entre bboxes
                iou_acc = 0.0; cont_acc = 0.0; count = 0
                for ka in ia:
                    for kb in ib:
                        if ka == kb:
                            continue
                        ba = prepared[ka][2]; bb = prepared[kb][2]
                        iou_acc += _bbox_iou(ba, bb)
                        cont_acc += _bbox_containment(ba, bb)
                        count += 1
                row[f"iou_{a}_{b}"] = float(iou_acc / count) if count > 0 else 0.0
                row[f"contain_{a}_{b}"] = float(cont_acc / count) if count > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def species_spatial_block(spatial_df: pd.DataFrame, L: int) -> Dict[str, np.ndarray]:
    """Promedia por especie las matrices LxL de B4 y devuelve un vector aplanado."""
    cols_reldist  = [f"reldist_{a}_{b}"  for a in range(L) for b in range(L)]
    cols_close    = [f"closefrac_{a}_{b}" for a in range(L) for b in range(L)]
    cols_iou      = [f"iou_{a}_{b}"      for a in range(L) for b in range(L)]
    cols_contain  = [f"contain_{a}_{b}"  for a in range(L) for b in range(L)]
    out: Dict[str, np.ndarray] = {}
    for sp, g in spatial_df.groupby("species_id"):
        v = np.concatenate([
            g[cols_reldist].mean(axis=0).to_numpy(dtype=np.float64),
            g[cols_close].mean(axis=0).to_numpy(dtype=np.float64),
            g[cols_iou].mean(axis=0).to_numpy(dtype=np.float64),
            g[cols_contain].mean(axis=0).to_numpy(dtype=np.float64),
        ])
        out[sp] = v
    return out


# ============================================================
# Carga COCO + features por anotación
# ============================================================
def load_coco(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_table(categories: List[dict]) -> Tuple[List[int], Dict[int, str]]:
    cats = sorted(categories, key=lambda c: int(c["id"]))
    coco_ids = [int(c["id"]) for c in cats]
    names = {int(c["id"]): str(c["name"]) for c in cats}
    return coco_ids, names


def per_annotation_features(coco: dict) -> Tuple[pd.DataFrame, Dict[int, dict]]:
    images = {int(im["id"]): im for im in coco["images"]}
    rows = []
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        im = images.get(iid)
        if im is None:
            continue
        geom = annotation_geometry(ann, int(im.get("width", 0)), int(im.get("height", 0)))
        if geom is None:
            continue
        rows.append({
            "annotation_id": int(ann.get("id", -1)),
            "image_id": iid,
            "image_name": str(im["file_name"]),
            "species_id": parse_species_id_from_filename(im["file_name"]),
            "category_id": int(ann["category_id"]),
            **geom,
        })
    return pd.DataFrame(rows), images


# ============================================================
# Agregación por imagen y por especie
# ============================================================
def per_image_features(
    ann_df: pd.DataFrame, coco_ids: List[int]
) -> pd.DataFrame:
    """Una fila por imagen con presencia, área-fracción y media de morfología por clase."""
    if ann_df.empty:
        return pd.DataFrame()
    img_groups = ann_df.groupby(["image_id", "image_name", "species_id"], sort=False)

    out_rows = []
    for (iid, iname, sp), g in img_groups:
        row = {
            "image_id": iid,
            "image_name": iname,
            "species_id": sp,
            "n_annotations": int(len(g)),
        }
        # presencia binaria + área-fracción acumulada + media de descriptores por clase
        for cid in coco_ids:
            sub = g[g["category_id"] == cid]
            present = int(len(sub) > 0)
            area_frac = float(sub["area_frac"].sum()) if present else 0.0
            row[f"presence_c{cid}"] = present
            row[f"area_c{cid}"] = area_frac
            for k in SHAPE_KEYS:
                row[f"{k}_c{cid}"] = float(sub[k].mean()) if present else 0.0
                row[f"{k}_std_c{cid}"] = float(sub[k].std(ddof=0)) if present and len(sub) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _l1_normalize(v: np.ndarray) -> np.ndarray:
    s = float(np.abs(v).sum())
    return v / s if s > 0 else v


def species_blocks(
    img_df: pd.DataFrame, coco_ids: List[int]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Bloques B1 (presencia+área), B2 (morfología media+std) y B3 (Jaccard de co-oc).

    Devuelve el DataFrame de firmas y los diccionarios bruto por especie.
    """
    presence_cols = [f"presence_c{cid}" for cid in coco_ids]
    area_cols = [f"area_c{cid}" for cid in coco_ids]
    shape_mean_cols = [f"{k}_c{cid}" for cid in coco_ids for k in SHAPE_KEYS]
    shape_std_cols = [f"{k}_std_c{cid}" for cid in coco_ids for k in SHAPE_KEYS]

    species_rows = []
    pres_by_sp: Dict[str, np.ndarray] = {}
    area_by_sp: Dict[str, np.ndarray] = {}
    shape_by_sp: Dict[str, np.ndarray] = {}
    cooc_by_sp: Dict[str, np.ndarray] = {}

    for sp, g in img_df.groupby("species_id"):
        n_img = int(len(g))
        # B1: frecuencia de presencia y área-fracción media por clase
        pres = g[presence_cols].mean(axis=0).to_numpy(dtype=np.float64)
        area = g[area_cols].mean(axis=0).to_numpy(dtype=np.float64)
        # área media normalizada por la suma total (forma de la mezcla)
        area_dist = _l1_normalize(area.copy())
        # B2: morfología media y dispersión (ponderada por presencia)
        shape_mean = np.zeros(len(coco_ids) * len(SHAPE_KEYS), dtype=np.float64)
        shape_std = np.zeros_like(shape_mean)
        for j, cid in enumerate(coco_ids):
            present_mask = g[f"presence_c{cid}"].to_numpy() > 0
            if present_mask.any():
                for k_idx, k in enumerate(SHAPE_KEYS):
                    col_mean = g.loc[present_mask, f"{k}_c{cid}"].mean()
                    col_std = g.loc[present_mask, f"{k}_std_c{cid}"].mean()
                    shape_mean[j * len(SHAPE_KEYS) + k_idx] = float(col_mean)
                    shape_std[j * len(SHAPE_KEYS) + k_idx] = float(col_std)
        # B3: co-ocurrencia Jaccard a nivel imagen (P(a∧b) / P(a∨b))
        L = len(coco_ids)
        P = g[presence_cols].to_numpy(dtype=np.float64)  # (n_img, L)
        inter = P.T @ P  # (L, L)
        any_mat = np.zeros((L, L), dtype=np.float64)
        for a in range(L):
            for b in range(L):
                union = float(((P[:, a] + P[:, b]) > 0).sum())
                any_mat[a, b] = union
        jacc = np.where(any_mat > 0, inter / np.maximum(any_mat, 1.0), 0.0)
        # Vector relacional = triangular superior (incluye diagonal=presencia
        # marginal: P(a∧a)/P(a∨a)=1 si a aparece, 0 si no — útil)
        iu = np.triu_indices(L, k=0)
        rel_vec = jacc[iu]

        species_rows.append({
            "species_id": sp,
            "n_images": n_img,
            **{f"B1_presence_c{cid}": float(pres[i]) for i, cid in enumerate(coco_ids)},
            **{f"B1_areafrac_c{cid}": float(area_dist[i]) for i, cid in enumerate(coco_ids)},
            **{f"B2_{k}_c{cid}": float(shape_mean[j * len(SHAPE_KEYS) + ki])
               for j, cid in enumerate(coco_ids) for ki, k in enumerate(SHAPE_KEYS)},
            **{f"B2_{k}_std_c{cid}": float(shape_std[j * len(SHAPE_KEYS) + ki])
               for j, cid in enumerate(coco_ids) for ki, k in enumerate(SHAPE_KEYS)},
            **{f"B3_jacc_{a}_{b}": float(jacc[a, b])
               for a in range(L) for b in range(a, L)},
        })
        pres_by_sp[sp] = pres
        area_by_sp[sp] = area_dist
        shape_by_sp[sp] = np.concatenate([shape_mean, shape_std])
        cooc_by_sp[sp] = rel_vec

    return pd.DataFrame(species_rows), pres_by_sp, area_by_sp, shape_by_sp, cooc_by_sp


# ============================================================
# Distancia entre especies por bloques
# ============================================================
def _cos_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    nA = np.linalg.norm(A, axis=1, keepdims=True)
    nB = np.linalg.norm(B, axis=1, keepdims=True)
    nA[nA == 0] = 1.0
    nB[nB == 0] = 1.0
    sim = (A / nA) @ (B / nB).T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def build_species_distance(
    species_ids: List[str],
    pres: Dict[str, np.ndarray],
    area: Dict[str, np.ndarray],
    shape: Dict[str, np.ndarray],
    cooc: Dict[str, np.ndarray],
    spatial: Optional[Dict[str, np.ndarray]],
    w_b1: float, w_b2: float, w_b3: float, w_b4: float,
) -> np.ndarray:
    P = np.vstack([pres[s] for s in species_ids])     # (S, L)
    A = np.vstack([area[s] for s in species_ids])     # (S, L)
    H = np.vstack([shape[s] for s in species_ids])    # (S, L*2K)
    C = np.vstack([cooc[s] for s in species_ids])     # (S, L(L+1)/2)

    # B1 = concat(presence_distribution, area_distribution) — ya están en [0,1].
    B1 = np.hstack([P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12),
                    A])  # area ya l1-normalizado
    # B2 = morfología: z-score por columna (escalas heterogéneas) → coseno
    B2 = _zscore_cols(H)
    # B3 = co-ocurrencia Jaccard, ya en [0,1]
    B3 = C

    d_b1 = _cos_dist(B1, B1)
    d_b2 = _cos_dist(B2, B2)
    d_b3 = _cos_dist(B3, B3)
    w_sum = w_b1 + w_b2 + w_b3
    d_total = (w_b1 * d_b1 + w_b2 * d_b2 + w_b3 * d_b3)
    if spatial is not None and w_b4 > 0:
        S4 = np.vstack([spatial[s] for s in species_ids])
        # rel_dist está en escala relativa (no [0,1]); z-score por columna.
        S4 = _zscore_cols(S4)
        d_b4 = _cos_dist(S4, S4)
        d_total = d_total + w_b4 * d_b4
        w_sum = w_sum + w_b4
    return d_total / max(w_sum, 1e-12)


def intra_species_dispersion(img_df: pd.DataFrame, coco_ids: List[int]) -> Dict[str, float]:
    """Distancia coseno media intra-especie sobre el vector (presencia+area+shape) por imagen.

    Usa z-score GLOBAL (sobre todas las imágenes del dataset) para que la escala
    sea comparable con la distancia inter-especie del bloque combinado. Sin esta
    normalización compartida los intra salen artificialmente inflados cuando la
    especie tiene muy pocas imágenes.
    """
    presence_cols = [f"presence_c{cid}" for cid in coco_ids]
    area_cols = [f"area_c{cid}" for cid in coco_ids]
    shape_cols = [f"{k}_c{cid}" for cid in coco_ids for k in SHAPE_KEYS]
    Xp = img_df[presence_cols].to_numpy(dtype=np.float64)
    Xa = img_df[area_cols].to_numpy(dtype=np.float64)
    Xs = img_df[shape_cols].to_numpy(dtype=np.float64)
    Xs_z = _zscore_cols(Xs)  # z-score sobre TODAS las imágenes
    species = img_df["species_id"].to_numpy()
    out: Dict[str, float] = {}
    for sp in np.unique(species):
        mask = species == sp
        n = int(mask.sum())
        if n < 2:
            out[sp] = 0.0
            continue
        X_sp = np.hstack([Xp[mask], Xa[mask], Xs_z[mask]])
        d = _cos_dist(X_sp, X_sp)
        iu = np.triu_indices(n, k=1)
        out[sp] = float(d[iu].mean())
    return out


# ============================================================
# Salidas
# ============================================================
def write_separability(
    sp_ids: List[str], dist: np.ndarray, intra: Dict[str, float],
    latin: Dict[str, str], out: Path
) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(sp_ids):
        for j in range(i + 1, len(sp_ids)):
            b = sp_ids[j]
            ia = intra.get(a, 0.0)
            ib = intra.get(b, 0.0)
            rows.append({
                "species_a": a,
                "species_a_name": latin.get(a, ""),
                "species_b": b,
                "species_b_name": latin.get(b, ""),
                "inter_distance": float(dist[i, j]),
                "intra_a": ia,
                "intra_b": ib,
                "contrast": float(dist[i, j] - max(ia, ib)),
            })
    df = pd.DataFrame(rows).sort_values("contrast")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def write_topk(
    sp_ids: List[str], dist: np.ndarray, latin: Dict[str, str],
    k: int, out: Path
) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(sp_ids):
        order = np.argsort(dist[i])
        rank = 0
        for j in order:
            if j == i:
                continue
            rank += 1
            rows.append({
                "species_id": a,
                "species_name": latin.get(a, ""),
                "rank": rank,
                "neighbor_id": sp_ids[j],
                "neighbor_name": latin.get(sp_ids[j], ""),
                "distance": float(dist[i, j]),
            })
            if rank >= k:
                break
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def plot_heatmap(sp_ids: List[str], dist: np.ndarray, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib no disponible; salto heatmap.")
        return
    n = len(sp_ids)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.18), max(5, n * 0.18)))
    im = ax.imshow(dist, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sp_ids, rotation=90, fontsize=5)
    ax.set_yticklabels(sp_ids, fontsize=5)
    plt.colorbar(im, ax=ax, label="distancia coseno combinada")
    ax.set_title("Distancia entre especies (B1+B2+B3)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coco-json", type=Path, required=True,
                    help="COCO con annotations predichas o GT.")
    ap.add_argument("--especies-json", type=Path, default=None,
                    help="Mapping id->Latin name (opcional).")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/species_proximity"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--w-b1", type=float, default=1.0, help="Peso del bloque presencia+área.")
    ap.add_argument("--w-b2", type=float, default=1.0, help="Peso del bloque morfología.")
    ap.add_argument("--w-b3", type=float, default=1.0, help="Peso del bloque co-ocurrencia.")
    ap.add_argument("--w-b4", type=float, default=1.0,
                    help="Peso del bloque proximidad/solapamiento espacial (rel_dist + close_frac + bbox IoU + contención asimétrica). "
                         "Pon 0 para desactivarlo.")
    ap.add_argument("--q-close", type=float, default=0.25,
                    help="Cuantil per-imagen para definir 'cerca' en close_frac. "
                         "Por defecto Q1 de las distancias intra-imagen — adaptativo, no global.")
    ap.add_argument("--min-images", type=int, default=2,
                    help="Especies con menos imágenes se omiten del análisis de distancia.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latin = load_especies_map(args.especies_json)

    print(f"[1/5] Leyendo {args.coco_json} ...")
    coco = load_coco(args.coco_json)
    coco_ids, cat_names = build_class_table(coco["categories"])
    print(f"      {len(coco['images'])} imágenes, {len(coco['annotations'])} anotaciones, "
          f"{len(coco_ids)} clases")

    print("[2/5] Calculando features por anotación ...")
    ann_df, _ = per_annotation_features(coco)
    if ann_df.empty:
        raise SystemExit("No se pudieron derivar features (anotaciones degeneradas?).")
    ann_df.to_csv(args.output_dir / "per_annotation_features.csv",
                  index=False, encoding="utf-8-sig")
    print(f"      {len(ann_df)} anotaciones válidas")

    print("[3/5] Agregando por imagen ...")
    img_df = per_image_features(ann_df, coco_ids)
    img_df.to_csv(args.output_dir / "per_image_features.csv",
                  index=False, encoding="utf-8-sig")
    print(f"      {len(img_df)} imágenes")

    print("[4/5] Construyendo firmas por especie y bloques B1/B2/B3 ...")
    # filtrar especies con muy pocas imágenes
    counts = img_df["species_id"].value_counts()
    keep_sp = set(counts[counts >= args.min_images].index.tolist())
    img_df_keep = img_df[img_df["species_id"].isin(keep_sp)].copy()
    print(f"      especies retenidas (>= {args.min_images} imágenes): {len(keep_sp)}")

    sig_df, pres, area, shape, cooc = species_blocks(img_df_keep, coco_ids)
    sig_df.insert(1, "species_name", sig_df["species_id"].map(latin).fillna(""))
    sig_df.to_csv(args.output_dir / "species_signatures.csv",
                  index=False, encoding="utf-8-sig")

    spatial_sp: Optional[Dict[str, np.ndarray]] = None
    if args.w_b4 > 0:
        print("[4b/5] Calculando bloque B4 (proximidad/solapamiento espacial)...")
        spatial_df = per_image_spatial_relations(coco, coco_ids, q_close=args.q_close)
        spatial_df = spatial_df[spatial_df["species_id"].isin(keep_sp)].copy()
        spatial_df.to_csv(args.output_dir / "per_image_spatial_relations.csv",
                          index=False, encoding="utf-8-sig")
        spatial_sp = species_spatial_block(spatial_df, len(coco_ids))

    print("[5/5] Calculando distancias y vecinos top-k ...")
    species_ids = sig_df["species_id"].tolist()
    dist = build_species_distance(species_ids, pres, area, shape, cooc, spatial_sp,
                                  args.w_b1, args.w_b2, args.w_b3, args.w_b4)
    pd.DataFrame(dist, index=species_ids, columns=species_ids).to_csv(
        args.output_dir / "species_distance.csv", encoding="utf-8-sig"
    )
    np.savez(args.output_dir / "species_distance.npz",
             distance=dist.astype(np.float32),
             species_ids=np.array(species_ids))

    intra = intra_species_dispersion(img_df_keep, coco_ids)
    sep_df = write_separability(species_ids, dist, intra, latin,
                                args.output_dir / "species_separability.csv")
    topk_df = write_topk(species_ids, dist, latin, args.topk,
                         args.output_dir / "species_topk_neighbors.csv")

    if args.plot:
        plot_heatmap(species_ids, dist, args.output_dir / "species_distance.png")

    # resumen en stdout
    print()
    print("Pares con MENOR contraste (más confundibles bajo esta representación):")
    print(sep_df.head(10).to_string(index=False))
    print()
    print("Pares con MAYOR contraste (más separables):")
    print(sep_df.tail(10).to_string(index=False))
    print()
    print(f"Todo en: {args.output_dir}")


if __name__ == "__main__":
    main()
