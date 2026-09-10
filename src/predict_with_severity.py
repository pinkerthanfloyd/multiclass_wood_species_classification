"""predict_with_severity.py

Pipeline de inferencia en DOS ETAPAS + análisis de severidad de error.

Flujo:

  1. SEGMENTADOR (best.pt YOLO seg) -> COCO JSON con rasgos predichos en
     cada imagen.
  2. CLASIFICADOR DE ESPECIES (pipeline joblib, salida de
     species_classifier.py) -> especie predicha a partir del vector de
     features por imagen extraído del COCO JSON.
  3. SEVERIDAD DEL ERROR: para cada predicción incorrecta se mira la
     matriz de disimilaridad entre especies (salida de
     compute_species_dissimilarity.py) y se computa:
         severity = dissim(true, predicted) / max_dissim
     Confundir una especie con otra muy parecida tiene severidad ~0.
     Confundirla con una muy distinta tiene severidad ~1.

  Este es el reporte que de hecho mide lo que el nuevo paradigma de
  anotación busca (que las anotaciones DIFERENCIEN especies) — la
  accuracy plana no.

Argumentos:

  --segmentation-model PATH    best.pt del segmentador YOLO seg.
  --classifier PATH            joblib del clasificador (RF o SVM).
  --dissimilarity-csv PATH     species_dissimilarity.csv
                                (compute_species_dissimilarity.py).
  --images-dir DIR             Carpeta con las imágenes de entrada.
                                La verdadera especie se infiere del
                                prefijo del nombre del archivo, salvo
                                que se pase --labels-csv.
  --data-yaml PATH             data.yaml del segmentador.
  --species-map PATH           JSON dict species_id -> nombre humano
                                (mismo mapping que usa el clasificador).
  --labels-csv PATH            CSV opcional con columnas
                                'image,true_species' para sobreescribir
                                la inferencia del nombre.
  --output-dir DIR             Directorio de salida.

Salidas (en --output-dir):

  predicted_instances.json     COCO JSON del segmentador.
  per_image_features.csv       Vectores que entran al clasificador.
  predictions_with_severity.csv
       columnas: image, true_species, predicted_species,
                 classifier_top_prob, correct, dissim, severity,
                 segmentator_conf_mean, segmentator_n_inst
  severity_summary.json        Accuracy plana, error severity-weighted,
                                top-N peores predicciones.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ============================================================
# Helpers
# ============================================================
def parse_species_id(fname: str) -> str:
    stem = Path(fname).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def load_class_names(yaml_path: Path) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
    raise ValueError("Could not read class names from data.yaml")


def get_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. pip install ultralytics")
    return YOLO


def polygon_area_px(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_bbox_coco(points: np.ndarray) -> List[float]:
    if len(points) < 3:
        return [0.0, 0.0, 0.0, 0.0]
    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


# ============================================================
# Segmentación: COCO predicho
# ============================================================
def run_segmentation(
    model_path: Path,
    images: List[Path],
    class_names: List[str],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
) -> dict:
    YOLO = get_yolo_class()
    model = YOLO(str(model_path))

    coco = {
        "categories": [
            {"id": i + 1, "name": n, "supercategory": "wood_feature"}
            for i, n in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }
    next_ann_id = 1

    for img_id, img_path in enumerate(images, start=1):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=300,
            retina_masks=False,
            device=device,
            verbose=False,
        )
        r = results[0]
        h, w = (None, None)
        if hasattr(r, "orig_shape") and r.orig_shape is not None:
            h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
        else:
            try:
                import cv2
                im = cv2.imread(str(img_path))
                if im is not None:
                    h, w = im.shape[:2]
            except Exception:
                pass
        if h is None or w is None:
            print(f"[skip] could not read shape for {img_path}")
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": int(w),
            "height": int(h),
        })

        if r.masks is None or r.boxes is None:
            continue
        polys = r.masks.xy  # lista de arrays (N, 2) en píxeles
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for poly, cls_id, score in zip(polys, cls_ids, confs):
            pts = np.asarray(poly, dtype=np.float32)
            if pts.ndim != 2 or len(pts) < 3:
                continue
            area = polygon_area_px(pts)
            bbox = polygon_bbox_coco(pts)
            seg_flat = pts.reshape(-1).tolist()
            coco["annotations"].append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": int(cls_id) + 1,
                "segmentation": [seg_flat],
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "score": float(score),
            })
            next_ann_id += 1
    return coco


# ============================================================
# Extracción de features (idéntica a species_classifier.py)
# ============================================================
def extract_classifier_features(
    coco: dict, id_to_species: Optional[Dict[str, str]]
) -> pd.DataFrame:
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    rows: Dict[int, dict] = {}

    for img in coco["images"]:
        sp_id = parse_species_id(img["file_name"])
        sp_name = id_to_species.get(sp_id, sp_id) if id_to_species else sp_id
        img_area = float(img["width"] * img["height"])
        row = {
            "image_id": img["id"],
            "file_name": img["file_name"],
            "species_id": sp_id,
            "species_name": sp_name,
            "img_width": int(img["width"]),
            "img_height": int(img["height"]),
            "img_area": img_area,
            "total_annotations": 0,
            "total_annotated_area": 0.0,
            "total_density_%": 0.0,
        }
        for cat in categories.values():
            prefix = f"Clase_{cat}"
            row[f"{prefix}_count"] = 0
            row[f"{prefix}_area_sum"] = 0.0
            row[f"{prefix}_area_mean"] = 0.0
            row[f"{prefix}_area_std"] = 0.0
            row[f"{prefix}_density_%"] = 0.0
        rows[img["id"]] = row

    areas_per: Dict[Tuple[int, str], List[float]] = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in rows:
            continue
        cat = categories[ann["category_id"]]
        prefix = f"Clase_{cat}"
        area = float(ann.get("area", 0.0))
        img_area = rows[iid]["img_area"]
        rows[iid]["total_annotations"] += 1
        rows[iid]["total_annotated_area"] += area
        rows[iid][f"{prefix}_count"] += 1
        rows[iid][f"{prefix}_area_sum"] += area
        if img_area > 0:
            rows[iid]["total_density_%"] += (area / img_area) * 100.0
            rows[iid][f"{prefix}_density_%"] += (area / img_area) * 100.0
        areas_per.setdefault((iid, cat), []).append(area)

    for (iid, cat), areas in areas_per.items():
        prefix = f"Clase_{cat}"
        arr = np.array(areas, dtype=float)
        rows[iid][f"{prefix}_area_mean"] = float(arr.mean())
        rows[iid][f"{prefix}_area_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return pd.DataFrame(rows.values())


def align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                expected = list(step.feature_names_in_)
                break
    if expected is None and hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    if expected is None:
        drop = [c for c in ("image_id", "file_name", "species_id", "species_name")
                if c in df.columns]
        return df.drop(columns=drop)
    return df.reindex(columns=expected, fill_value=0)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segmentation-model", required=True, type=Path)
    ap.add_argument("--classifier", required=True, type=Path)
    ap.add_argument("--dissimilarity-csv", required=True, type=Path,
                    help="species_dissimilarity.csv from compute_species_dissimilarity.py")
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--data-yaml", required=True, type=Path)
    ap.add_argument("--species-map", default=None, type=Path,
                    help="JSON species_id -> human-readable name "
                         "(must match the mapping species_classifier.py used).")
    ap.add_argument("--labels-csv", default=None, type=Path,
                    help="Optional CSV with columns image, true_species "
                         "(overrides filename-based inference).")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/severity"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--coco-cache", type=Path, default=None,
                    help="If passed and exists, skip segmentation and load this "
                         "COCO JSON. If passed but missing, write segmentation "
                         "output here for reuse.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Imágenes
    images = sorted(
        [p for p in args.images_dir.rglob("*")
         if p.is_file() and p.suffix.lower() in IMG_EXTS]
    )
    if not images:
        sys.exit(f"No images found in {args.images_dir}")
    print(f"Found {len(images)} images.")

    # ── 2. Species map
    id_to_species: Optional[Dict[str, str]] = None
    if args.species_map:
        if not args.species_map.exists():
            print(f"WARNING: --species-map {args.species_map} not found; using raw IDs.")
        else:
            with open(args.species_map, "r", encoding="utf-8") as f:
                id_to_species = json.load(f)

    # ── 3. Segmentación (con cache opcional)
    cache_path = args.coco_cache
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached COCO predictions: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
    else:
        print("Running segmentation...")
        class_names = load_class_names(args.data_yaml)
        coco = run_segmentation(
            args.segmentation_model, images, class_names,
            imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device,
        )
        out_coco = cache_path if cache_path is not None else (
            args.output_dir / "predicted_instances.json"
        )
        out_coco.parent.mkdir(parents=True, exist_ok=True)
        with open(out_coco, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"Wrote {len(coco['annotations'])} annotations over "
              f"{len(coco['images'])} images -> {out_coco}")

    # Asegura una copia en output_dir incluso si se usó cache externo.
    target_coco = args.output_dir / "predicted_instances.json"
    if not target_coco.exists():
        with open(target_coco, "w", encoding="utf-8") as f:
            json.dump(coco, f)

    # ── 4. Confianza media + nº instancias por imagen (para reporte).
    conf_by_image: Dict[int, List[float]] = {im["id"]: [] for im in coco["images"]}
    n_by_image: Dict[int, int] = {im["id"]: 0 for im in coco["images"]}
    for ann in coco["annotations"]:
        conf_by_image[ann["image_id"]].append(float(ann.get("score", 0.0)))
        n_by_image[ann["image_id"]] += 1

    # ── 5. Features para el clasificador
    print("Extracting classifier features...")
    df = extract_classifier_features(coco, id_to_species)
    df.to_csv(args.output_dir / "per_image_features.csv",
              index=False, encoding="utf-8-sig")

    # ── 6. Clasificador
    print(f"Loading classifier: {args.classifier}")
    model = joblib.load(args.classifier)
    X = align_features(df, model)
    print(f"Classifier input: {X.shape[1]} features, {X.shape[0]} images.")
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X).max(axis=1)
    except Exception:
        probs = np.full(len(preds), np.nan)

    # ── 7. True species (override opcional)
    true_species = df["species_name"].to_numpy()
    if args.labels_csv and args.labels_csv.exists():
        lbl = pd.read_csv(args.labels_csv)
        lookup = dict(zip(lbl["image"], lbl["true_species"]))
        true_species = np.array([
            lookup.get(fn, df.loc[df["file_name"] == fn, "species_name"].iloc[0])
            for fn in df["file_name"]
        ])

    # ── 8. Matriz de disimilaridad
    sp_mat = pd.read_csv(args.dissimilarity_csv, index_col=0)
    # Forzamos índice y columnas a string (los CSV los pueden parsear como int).
    sp_mat.index = sp_mat.index.astype(str)
    sp_mat.columns = sp_mat.columns.astype(str)
    name_to_id = {v: k for k, v in id_to_species.items()} if id_to_species else {}

    def species_key(val) -> Optional[str]:
        if val is None:
            return None
        s = str(val)
        if s in sp_mat.index:
            return s
        if s in name_to_id and name_to_id[s] in sp_mat.index:
            return name_to_id[s]
        return None

    max_dissim = float(np.nanmax(sp_mat.values))
    if max_dissim <= 0:
        print("WARNING: dissimilarity matrix max is zero. Severity will be undefined.")
        max_dissim = 1.0
    else:
        print(f"Max dissimilarity in matrix: {max_dissim:.4f}")

    out_rows = []
    for i, fn in enumerate(df["file_name"]):
        ts, ps = true_species[i], preds[i]
        ts_k = species_key(ts)
        ps_k = species_key(ps)
        correct = int(str(ts) == str(ps))
        if correct:
            dissim = 0.0
        elif ts_k is None or ps_k is None:
            dissim = np.nan
        else:
            try:
                dissim = float(sp_mat.loc[ts_k, ps_k])
            except KeyError:
                dissim = np.nan
        if correct:
            severity = 0.0
        elif np.isnan(dissim):
            severity = np.nan
        else:
            severity = float(dissim) / max_dissim
        iid = int(df["image_id"].iloc[i])
        confs = conf_by_image.get(iid, [])
        out_rows.append({
            "image": fn,
            "true_species": ts,
            "predicted_species": ps,
            "classifier_top_prob": float(probs[i]) if not np.isnan(probs[i]) else np.nan,
            "correct": correct,
            "dissim": dissim,
            "severity": severity,
            "segmentator_conf_mean": float(np.mean(confs)) if confs else 0.0,
            "segmentator_n_inst": int(n_by_image.get(iid, 0)),
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_dir / "predictions_with_severity.csv",
                  index=False, encoding="utf-8-sig")

    # ── 9. Resumen
    n = len(out_df)
    n_correct = int(out_df["correct"].sum())
    acc = n_correct / n if n else 0.0
    sev = out_df["severity"].dropna()
    sev_weighted_err = float(sev.mean()) if len(sev) else 0.0
    n_with_dissim = int(out_df["dissim"].notna().sum())
    worst = out_df.dropna(subset=["severity"]).sort_values(
        "severity", ascending=False
    ).head(20)

    summary = {
        "n_images": n,
        "n_correct": n_correct,
        "accuracy": acc,
        "severity_weighted_error": sev_weighted_err,
        "max_dissim_used": max_dissim,
        "predictions_with_known_dissim": n_with_dissim,
        "dissimilarity_csv": str(args.dissimilarity_csv),
        "segmentation_model": str(args.segmentation_model),
        "classifier": str(args.classifier),
        "worst_predictions_top20": worst[[
            "image", "true_species", "predicted_species", "severity"
        ]].to_dict("records"),
    }
    with open(args.output_dir / "severity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Total images:                {n}")
    print(f"Plain accuracy:              {acc:.4f}  ({n_correct}/{n})")
    print(f"Severity-weighted error:     {sev_weighted_err:.4f}")
    print(f"  (errors among very dissimilar species count more)")
    print("=" * 60)
    print()
    print("Top 10 worst predictions:")
    print(worst.head(10)[[
        "image", "true_species", "predicted_species", "severity"
    ]].to_string(index=False))
    print()
    print(f"All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
