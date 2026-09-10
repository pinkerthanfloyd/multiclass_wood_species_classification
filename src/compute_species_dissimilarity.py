"""compute_species_dissimilarity.py

Construye matrices de DISIMILARIDAD entre imágenes y entre especies
a partir de las anotaciones de rasgos (GT YOLO o predicciones COCO).

Permite responder:

  - Cuán distintas son DOS imágenes en el espacio de rasgos.
  - Cuán distintas son DOS especies entre sí (distancia inter-especie
    media).
  - Cuán DISPERSA es una especie internamente (varianza intra-especie).

Bajo el paradigma de anotación selectiva (sólo se marcan los rasgos
distintivos por especie) estas matrices son la métrica natural de
calidad de las anotaciones: si dos imágenes de la misma especie son
muy disímiles, las anotaciones son inconsistentes; si dos imágenes de
especies distintas son indistinguibles, las anotaciones no logran
separar esas especies.

Entradas (mutuamente excluyentes):

  --yolo-dataset DIR        Carpeta con data.yaml + images/labels (formato
                            YOLO seg). El species_id se extrae del prefijo
                            del nombre del archivo (antes del primer '-'
                            o '_').
  --coco-json PATH          COCO JSON (p. ej. predicted_instances.json
                            generado por el segmentador). Las categorías
                            del COCO definen el espacio de rasgos.

Salidas (en --output-dir, default outputs/dissimilarity):

  per_image_features.csv        Una fila por imagen.
  image_pairwise_distance.npz   Matriz N×N float32 + arrays con imagen y
                                especie por fila.
  species_dissimilarity.csv     Matriz S×S. Celda (a, b) = distancia
                                inter-especie media; la diagonal es la
                                distancia intra-especie media.
  species_intra_variance.csv    Por especie: n_imagenes, intra_mean,
                                intra_std, intra_max.
  species_separability.csv      Para cada par (a, b): inter_mean,
                                intra_mean_a/b, contraste =
                                inter_mean - max(intra_a, intra_b).
                                Contraste alto = par bien separado.
  species_dissimilarity.png     Heatmap (si --plot y matplotlib).

Métricas (--metric):

  cosine          1 - cos_sim del vector areafrac.  [recomendado]
  l1              Distancia Manhattan del vector areafrac.
  jensenshannon   Distancia JS del vector presencia normalizado.

Vector base (--vector):

  area            Distribución de área por clase normalizada por imagen
                  (continuo). Recomendado.
  presence        Vector binario de presencia por clase, normalizado.

Notas sobre anotación parcial:
  Bajo este paradigma los CONTEOS y ÁREAS absolutos son una cota inferior,
  no la verdad. Cosine sobre area-fraction normalizada ignora la magnitud
  total (sesgada por anotación parcial) y mide solo la FORMA de la mezcla
  de rasgos — que es justamente la firma distintiva de la especie.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ============================================================
# Helpers comunes
# ============================================================
def parse_species_id(fname: str) -> str:
    stem = Path(fname).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def load_class_names_from_yaml(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
    raise ValueError("Could not read class names from data.yaml")


def polygon_area_norm(coords: List[float]) -> float:
    """Shoelace en coordenadas normalizadas [0,1]. Devuelve área en [0, ~1]."""
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


# ============================================================
# Extracción de features por imagen (YOLO labels)
# ============================================================
def _resolve_label(img_path: Path) -> Optional[Path]:
    candidates = [
        img_path.with_suffix(".txt"),
        img_path.parent.parent / "labels" / (img_path.stem + ".txt"),
        img_path.parent / "labels" / (img_path.stem + ".txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def features_from_yolo_dataset(root: Path, class_names: List[str]) -> pd.DataFrame:
    nc = len(class_names)
    rows = []
    for img_path in sorted(root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        counts = np.zeros(nc, dtype=np.int64)
        areas = np.zeros(nc, dtype=np.float64)
        label_path = _resolve_label(img_path)
        if label_path is not None:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    try:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        continue
                    if 0 <= cls_id < nc:
                        counts[cls_id] += 1
                        areas[cls_id] += polygon_area_norm(coords)
        row = {"image": img_path.name, "species_id": parse_species_id(img_path.name)}
        for i, name in enumerate(class_names):
            row[f"count_{name}"] = int(counts[i])
            row[f"area_{name}"] = float(areas[i])
        rows.append(row)
    return pd.DataFrame(rows)


def features_from_coco(coco_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_names = [str(c["name"]) for c in cats]
    catid_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    nc = len(class_names)
    imgs = {im["id"]: im for im in coco["images"]}
    counts = {iid: np.zeros(nc, dtype=np.int64) for iid in imgs}
    areas = {iid: np.zeros(nc, dtype=np.float64) for iid in imgs}

    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in imgs:
            continue
        cidx = catid_to_idx.get(ann["category_id"])
        if cidx is None:
            continue
        counts[iid][cidx] += 1
        im = imgs[iid]
        denom = float(im["width"] * im["height"])
        if denom > 0:
            # Área del COCO está en píxeles; la pasamos a fracción de imagen.
            areas[iid][cidx] += float(ann.get("area", 0.0)) / denom

    rows = []
    for iid, im in imgs.items():
        row = {"image": im["file_name"], "species_id": parse_species_id(im["file_name"])}
        for i, name in enumerate(class_names):
            row[f"count_{name}"] = int(counts[iid][i])
            row[f"area_{name}"] = float(areas[iid][i])
        rows.append(row)
    return pd.DataFrame(rows), class_names


# ============================================================
# Vectores y distancias
# ============================================================
def build_vectors(
    df: pd.DataFrame, class_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    area_cols = [f"area_{n}" for n in class_names]
    count_cols = [f"count_{n}" for n in class_names]
    area = df[area_cols].to_numpy(dtype=np.float64)
    counts = df[count_cols].to_numpy(dtype=np.float64)
    # Distribución de áreas (fila normalizada)
    row_sum = area.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    area_dist = area / row_sum
    # Distribución de presencia binaria normalizada
    presence = (counts > 0).astype(np.float64)
    pres_sum = presence.sum(axis=1, keepdims=True)
    pres_sum[pres_sum == 0] = 1.0
    presence_dist = presence / pres_sum
    return area_dist, presence_dist


def pairwise_distance(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Xn = X / norm
        sim = Xn @ Xn.T
        sim = np.clip(sim, -1.0, 1.0)
        return (1.0 - sim).astype(np.float64)
    if metric == "l1":
        # O(N^2 * D). Para N de cientos a unos miles es manejable.
        return np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2).astype(np.float64)
    if metric == "jensenshannon":
        eps = 1e-12
        P = X + eps
        P = P / P.sum(axis=1, keepdims=True)
        N = P.shape[0]
        D = np.zeros((N, N), dtype=np.float64)
        logP = np.log(P)
        for i in range(N):
            M = 0.5 * (P[i:i + 1] + P)
            logM = np.log(M)
            kl_pm = (P[i:i + 1] * (logP[i:i + 1] - logM)).sum(axis=1)
            kl_qm = (P * (logP - logM)).sum(axis=1)
            js = 0.5 * kl_pm + 0.5 * kl_qm
            D[i] = np.sqrt(np.maximum(js, 0.0))
        return D
    raise ValueError(f"Unknown metric: {metric}")


# ============================================================
# Agregación por especie
# ============================================================
def aggregate_by_species(
    dists: np.ndarray, species_ids: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    species_arr = np.array(species_ids)
    uniq = sorted(set(species_ids))
    idx_by_sp = {sp: np.where(species_arr == sp)[0] for sp in uniq}

    S = len(uniq)
    M = np.zeros((S, S), dtype=np.float64)
    intra_rows = []
    for a, sp_a in enumerate(uniq):
        idx_a = idx_by_sp[sp_a]
        n_a = len(idx_a)
        sub = dists[np.ix_(idx_a, idx_a)]
        if n_a >= 2:
            iu = np.triu_indices(n_a, k=1)
            intra_pairs = sub[iu]
            intra_mean = float(intra_pairs.mean())
            intra_std = float(intra_pairs.std())
            intra_max = float(intra_pairs.max())
        else:
            intra_mean = intra_std = intra_max = 0.0
        intra_rows.append({
            "species_id": sp_a,
            "n_images": int(n_a),
            "intra_mean": intra_mean,
            "intra_std": intra_std,
            "intra_max": intra_max,
        })
        M[a, a] = intra_mean
        for b in range(a + 1, S):
            sp_b = uniq[b]
            idx_b = idx_by_sp[sp_b]
            block = dists[np.ix_(idx_a, idx_b)]
            inter_mean = float(block.mean())
            M[a, b] = M[b, a] = inter_mean

    sp_mat = pd.DataFrame(M, index=uniq, columns=uniq)
    intra_df = pd.DataFrame(intra_rows)
    return sp_mat, intra_df


def species_separability(
    sp_mat: pd.DataFrame, intra_df: pd.DataFrame
) -> pd.DataFrame:
    intra = dict(zip(intra_df["species_id"], intra_df["intra_mean"]))
    rows = []
    species = list(sp_mat.index)
    for i, a in enumerate(species):
        for b in species[i + 1:]:
            inter = float(sp_mat.loc[a, b])
            ia = intra.get(a, 0.0)
            ib = intra.get(b, 0.0)
            contrast = inter - max(ia, ib)
            rows.append({
                "species_a": a,
                "species_b": b,
                "inter_mean": inter,
                "intra_mean_a": ia,
                "intra_mean_b": ib,
                "contrast": contrast,
            })
    return pd.DataFrame(rows).sort_values("contrast")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yolo-dataset", type=Path, default=None,
                    help="Dataset root with data.yaml + images/labels (YOLO seg).")
    ap.add_argument("--data-yaml", type=Path, default=None,
                    help="Path to data.yaml if it is not under --yolo-dataset/data.yaml.")
    ap.add_argument("--coco-json", type=Path, default=None,
                    help="COCO instances JSON (alternative to --yolo-dataset).")
    ap.add_argument("--metric", choices=["cosine", "l1", "jensenshannon"], default="cosine")
    ap.add_argument("--vector", choices=["area", "presence"], default="area")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/dissimilarity"))
    ap.add_argument("--plot", action="store_true",
                    help="Save species_dissimilarity.png heatmap (needs matplotlib).")
    args = ap.parse_args()

    if (args.yolo_dataset is None) == (args.coco_json is None):
        ap.error("Provide exactly one of --yolo-dataset or --coco-json")

    if args.yolo_dataset is not None:
        yaml_path = args.data_yaml or (args.yolo_dataset / "data.yaml")
        if not yaml_path.exists():
            ap.error(f"data.yaml not found at {yaml_path}")
        class_names = load_class_names_from_yaml(yaml_path)
        print(f"Reading YOLO dataset at {args.yolo_dataset}, classes={class_names}")
        df = features_from_yolo_dataset(args.yolo_dataset, class_names)
    else:
        print(f"Reading COCO JSON at {args.coco_json}")
        df, class_names = features_from_coco(args.coco_json)

    if df.empty:
        sys.exit("No images found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "per_image_features.csv", index=False, encoding="utf-8-sig")
    print(f"Built features for {len(df)} images, {len(class_names)} classes.")

    area_dist, pres_dist = build_vectors(df, class_names)
    X = area_dist if args.vector == "area" else pres_dist
    if args.metric == "jensenshannon" and args.vector == "area":
        # Para JS necesitamos distribución; area_dist ya lo es.
        pass

    print(f"Computing pairwise distance (metric={args.metric}, vector={args.vector})...")
    dists = pairwise_distance(X, args.metric)
    np.savez(
        args.output_dir / "image_pairwise_distance.npz",
        distance=dists.astype(np.float32),
        image=df["image"].to_numpy(),
        species=df["species_id"].to_numpy(),
        metric=args.metric,
        vector=args.vector,
    )
    print(f"Image pairwise matrix saved: {dists.shape}")

    sp_mat, intra_df = aggregate_by_species(dists, df["species_id"].tolist())
    sp_mat.to_csv(args.output_dir / "species_dissimilarity.csv",
                  encoding="utf-8-sig")
    intra_df.to_csv(args.output_dir / "species_intra_variance.csv",
                    index=False, encoding="utf-8-sig")

    sep = species_separability(sp_mat, intra_df)
    sep.to_csv(args.output_dir / "species_separability.csv",
               index=False, encoding="utf-8-sig")

    print()
    print(f"Species dissimilarity matrix: {sp_mat.shape}")
    print()
    print("Lowest contrast pairs (more easily confused under this annotation):")
    print(sep.head(10).to_string(index=False))
    print()
    print("Highest contrast pairs (most cleanly separable):")
    print(sep.tail(10).to_string(index=False))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            n = len(sp_mat)
            fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
            im = ax.imshow(sp_mat.values, cmap="viridis")
            ax.set_xticks(range(n))
            ax.set_xticklabels(sp_mat.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(n))
            ax.set_yticklabels(sp_mat.index, fontsize=6)
            plt.colorbar(im, ax=ax, label=f"{args.metric} distance")
            ax.set_title(f"Species dissimilarity ({args.metric}, {args.vector})")
            plt.tight_layout()
            plt.savefig(args.output_dir / "species_dissimilarity.png", dpi=150)
            plt.close(fig)
            print(f"Heatmap saved: {args.output_dir / 'species_dissimilarity.png'}")
        except ImportError:
            print("matplotlib not available; skipping heatmap.")

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
