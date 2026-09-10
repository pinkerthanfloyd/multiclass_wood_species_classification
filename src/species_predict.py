"""species_predict_fixed.py

Versión corregida de species_predict.py para generar el top-K (k=5 por
defecto) con rank_true / prob_true sobre un lote de evaluación, lista para
la tabla de resultados de la memoria.

IMPORTANTE: esto es INFERENCIA, no entrenamiento. Necesita que ya existan:
  - models/species_classifier/species_model_rf.joblib
  - models/species_classifier/species_model_svm.joblib
    (salida de species_classifier.py — no se tocan ni se reentrenan aquí)
  - inferences/especies.json (mapa "0034" -> "Anacardium L.", etc.)
  - Un COCO JSON REAL de anotaciones PREDICHAS por el segmentador para el
    lote a evaluar. OJO: revisa que el archivo sea de verdad un COCO JSON
    (debe tener claves "images", "annotations", "categories") y no un CSV
    con extensión .json por error — este script lo valida y aborta si no.
  - Un CSV de etiquetas verdaderas para ese lote (ver TRUE_LABELS_CANDIDATES
    más abajo) con columnas image/file_name y true_species.

Qué cambia respecto al species_predict.py original (marcado inline con
"# FIX" / "# NUEVO"):

1. FIX — extract_features ya no deriva species_id de file_name.split("-")[0]
   a ciegas. Antes, un lote cuyos archivos se llaman "1.jpg", "2.jpg" (sin el
   prefijo "id_especie-") producía sp_id = "1.jpg" completo y, al no
   encontrarlo en el mapa de especies, un species_name tipo
   "Especie_Desconocida_1.jpg" que parece una especie real y no lo es. Ahora
   se exige que el nombre matchee ^\\d{3,5}-; si no matchea, species_id y
   species_name quedan en blanco (NaN) en vez de un texto engañoso. Esta
   columna es solo informativa — la evaluación real siempre usa el CSV de
   etiquetas verdaderas, nunca este campo.

2. FIX — se valida que el JSON de entrada tenga forma de COCO (claves
   images/annotations/categories) antes de intentar usarlo. Si alguien le
   pasa por error un CSV renombrado a .json, el script falla con un mensaje
   claro en vez de producir columnas basura silenciosamente.

3. NUEVO — columna `true_in_catalog_<modelo>`: indica si la especie
   verdadera de esa fila pertenece al catálogo de clases con el que se
   entrenó ese modelo. Sin esto, una especie que el modelo JAMÁS pudo
   predecir (porque no estaba en el set de entrenamiento) se cuenta como un
   error más, mezclando "el modelo se equivocó" con "acertar era imposible".
   El resumen separa ambos casos.

4. NUEVO — hit-rate@1, @3 y @5 explícitos (antes solo se imprimía top-1 y
   top-TOP_K), calculados directamente sobre rank_true, y volcados también
   al JSON de resumen.

5. NUEVO — al final se imprime (y se guarda en OUTPUT_DIR/resultados_topk.tex)
   un bloque LaTeX booktabs con la tabla de hit-rate@1/3/5 por modelo, listo
   para pegar en la sección de resultados.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================
# RUTAS — editar aquí si la estructura del proyecto cambia
# ============================================================
COCO_JSON_PATH = Path("inferences/32_images.json")        # COCO real del segmentador, NO un CSV
SPECIES_MAP_PATH = Path("inferences/especies.json")
MODEL_RF_PATH = Path("models/species_classifier/species_model_rf.joblib")
MODEL_SVM_PATH = Path("models/species_classifier/species_model_svm.joblib")
OUTPUT_DIR = Path("inferences/32_images_pred")

TOP_K = 5
HITRATE_KS = (1, 3, 5)          # niveles de k a reportar (deben ser <= TOP_K)
ALLOW_MISSING_MODEL = False

TRUE_LABELS_CANDIDATES = [
    Path("inferences/32_images_labels.csv"),
    Path("inferences/32_images_true.csv"),
    Path("inferences/labels_32_images.csv"),
    OUTPUT_DIR / "true_labels.csv",
]

# Patrón que deben cumplir los nombres de archivo para poder derivar un
# species_id "de cortesía" a partir del nombre (convención id_especie-indice).
FILENAME_ID_RE = re.compile(r"^(\d{3,5})-")


# ============================================================
# UTILIDADES DE CARGA
# ============================================================
def resolve_coco_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in sorted(path.glob("*.json")):
            return candidate
        raise FileNotFoundError(f"No JSON found inside directory: {path}")
    with_ext = path.with_suffix(".json")
    if with_ext.is_file():
        return with_ext
    raise FileNotFoundError(
        f"COCO JSON not found at {path} (also tried {with_ext})"
    )


def validate_coco(coco_data: dict, source: Path) -> None:
    """# FIX: aborta con un mensaje claro si el JSON no es un COCO real.

    El bug que motivó esto: un CSV de predicciones guardado por error con
    extensión .json (columnas file_name,species_id,pred_rf,...) se cargaba
    como si fuera COCO y fallaba más adelante de forma confusa, o peor,
    producía filas vacías sin avisar.
    """
    required = {"images", "annotations", "categories"}
    missing = required - set(coco_data.keys())
    if missing:
        sys.exit(
            f"ERROR: {source} no tiene forma de COCO JSON (faltan claves "
            f"{sorted(missing)}). ¿Es en realidad un CSV con extensión .json? "
            f"Revisa el archivo antes de continuar."
        )


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(
    coco_data: dict,
    id_to_species: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    categories: Dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco_data["categories"]
    }

    def _sp_id_from_filename(file_name: str) -> Optional[str]:
        # FIX: antes era file_name.split("-")[0] sin validar formato.
        m = FILENAME_ID_RE.match(file_name)
        return m.group(1) if m else None

    def _name(sp_id: Optional[str]) -> Optional[str]:
        if sp_id is None:
            return None
        if id_to_species:
            return id_to_species.get(sp_id, f"Especie_Desconocida_{sp_id}")
        return sp_id

    image_rows: Dict[int, dict] = {}
    for img in coco_data["images"]:
        file_name = img["file_name"]
        sp_id = _sp_id_from_filename(file_name)
        img_area = img["width"] * img["height"]

        row: dict = {
            "image_id": img["id"],
            "file_name": file_name,
            "species_id_from_filename": sp_id,          # NUEVO nombre, ver nota abajo
            "species_name_from_filename": _name(sp_id),  # renombrado para no confundir con la verdad
            "img_area": img_area,
            "total_annotations": 0,
            "total_annotated_area": 0.0,
            "total_density_%": 0.0,
        }
        for cat_name in categories.values():
            prefix = f"Clase_{cat_name}"
            row[f"{prefix}_count"] = 0
            row[f"{prefix}_area_sum"] = 0.0
            row[f"{prefix}_area_mean"] = 0.0
            row[f"{prefix}_area_std"] = 0.0
            row[f"{prefix}_density_%"] = 0.0
        image_rows[img["id"]] = row

    areas_by_image_class: Dict[Tuple[int, str], List[float]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_rows:
            continue
        cat_name = categories[ann["category_id"]]
        prefix = f"Clase_{cat_name}"
        area = float(ann.get("area", 0.0))
        img_area = image_rows[img_id]["img_area"]

        image_rows[img_id]["total_annotations"] += 1
        image_rows[img_id]["total_annotated_area"] += area
        image_rows[img_id][f"{prefix}_count"] += 1
        image_rows[img_id][f"{prefix}_area_sum"] += area
        if img_area > 0:
            image_rows[img_id]["total_density_%"] += (area / img_area) * 100.0
            image_rows[img_id][f"{prefix}_density_%"] += (area / img_area) * 100.0
        areas_by_image_class.setdefault((img_id, cat_name), []).append(area)

    for (img_id, cat_name), areas in areas_by_image_class.items():
        prefix = f"Clase_{cat_name}"
        arr = np.array(areas, dtype=float)
        image_rows[img_id][f"{prefix}_area_mean"] = float(arr.mean())
        image_rows[img_id][f"{prefix}_area_std"] = (
            float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        )

    return pd.DataFrame(image_rows.values())


# ============================================================
# ALINEAMIENTO DE FEATURES SEGÚN EL MODELO
# ============================================================
_METADATA_COLS = {
    "image_id", "file_name", "species_id_from_filename",
    "species_name_from_filename", "img_area",
}


def _norm_species_name(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ")
    s = " ".join(s.split())
    return s


def _get_model_feature_names(model) -> Optional[List[str]]:
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def align_features(df: pd.DataFrame, model, model_name: str) -> pd.DataFrame:
    expected = _get_model_feature_names(model)
    if expected is None:
        drop = [c for c in _METADATA_COLS if c in df.columns]
        X = df.drop(columns=drop)
        print(f"  [{model_name}] No feature_names_in_ — dropping metadata "
              f"({X.shape[1]} features).")
        return X

    present = set(df.columns)
    missing = [c for c in expected if c not in present]
    extra = [c for c in present if c not in set(expected) and c not in _METADATA_COLS]
    if missing:
        print(f"  [{model_name}] Adding {len(missing)} zero-filled columns "
              f"missing from inference (e.g. {missing[:5]}{'...' if len(missing) > 5 else ''}).")
    if extra:
        print(f"  [{model_name}] Ignoring {len(extra)} columns not in model.")
    return df.reindex(columns=expected, fill_value=0)


# ============================================================
# TOP-K
# ============================================================
def get_model_classes(model) -> Optional[np.ndarray]:
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    return None


def predict_topk(
    model, X: pd.DataFrame, k: int = TOP_K
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    preds = model.predict(X)
    try:
        proba = model.predict_proba(X)
    except (AttributeError, NotImplementedError):
        return preds, None, None, None

    classes = get_model_classes(model)
    if classes is None:
        return preds, None, None, np.asarray(proba)

    n_classes = proba.shape[1]
    k_eff = min(k, n_classes)
    topk_idx = np.argsort(-proba, axis=1)[:, :k_eff]
    topk_classes = classes[topk_idx]
    topk_probs = np.take_along_axis(proba, topk_idx, axis=1)
    return preds, topk_classes, topk_probs, np.asarray(proba)


# ============================================================
# CARGA DE ETIQUETAS VERDADERAS (OPCIONAL)
# ============================================================
def load_true_labels() -> Tuple[Optional[Path], Optional[Dict[str, str]]]:
    found = next((p for p in TRUE_LABELS_CANDIDATES if p.exists()), None)
    if found is None:
        return None, None
    suffix = found.suffix.lower()
    if suffix == ".json":
        with open(found, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return found, {str(k): str(v) for k, v in data.items()}
        print(f"WARNING: {found} JSON format not recognised (expected dict).")
        return found, None
    try:
        df_true = pd.read_csv(found, encoding="utf-8-sig")
    except Exception as exc:
        print(f"WARNING: could not read {found}: {exc}")
        return found, None
    img_col = next(
        (c for c in ("image", "file_name", "filename", "image_file") if c in df_true.columns),
        None,
    )
    sp_col = next(
        (c for c in ("true_species", "species", "species_name", "real_species")
         if c in df_true.columns),
        None,
    )
    if img_col is None or sp_col is None:
        print(f"WARNING: {found} has no recognised image/true_species columns "
              f"(found: {list(df_true.columns)}).")
        return found, None
    mapping = dict(zip(df_true[img_col].astype(str), df_true[sp_col].astype(str)))
    return found, mapping


# ============================================================
# LATEX
# ============================================================
def build_latex_table(per_model: Dict[str, dict]) -> str:
    """# NUEVO: genera un bloque LaTeX booktabs con hit-rate@1/3/5 por modelo."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Modelo} & \textbf{$n$ evaluado} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} \\",
        r"\midrule",
    ]
    for name, m in per_model.items():
        n = m["n_eval"]
        h1 = m["hitrate_at_1"] * 100
        h3 = m["hitrate_at_3"] * 100
        h5 = m["hitrate_at_5"] * 100
        lines.append(f"{name} & {n} & {h1:.1f}\\,\\% & {h3:.1f}\\,\\% & {h5:.1f}\\,\\% \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Hit-rate top-$k$ por modelo sobre el lote de evaluaci\'on "
        r"(imágenes no usadas en el entrenamiento).}",
        r"\label{tab:topk_resultados}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    # ── 1. Resolver COCO ────────────────────────────────────────
    try:
        coco_path = resolve_coco_path(COCO_JSON_PATH)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Loading COCO JSON: {coco_path}")
    with open(coco_path, "r", encoding="utf-8-sig") as f:
        coco_data = json.load(f)
    validate_coco(coco_data, coco_path)  # FIX
    print(f"  {len(coco_data['images'])} images, "
          f"{len(coco_data['annotations'])} annotations, "
          f"{len(coco_data['categories'])} categories.")

    # ── 2. Mapa de especies ─────────────────────────────────────
    id_to_species: Optional[Dict[str, str]] = None
    if SPECIES_MAP_PATH.exists():
        with open(SPECIES_MAP_PATH, "r", encoding="utf-8") as f:
            id_to_species = json.load(f)
        print(f"Species map loaded: {len(id_to_species)} entries.")
    else:
        print(f"WARNING: species map not found ({SPECIES_MAP_PATH}); using raw IDs.")

    # ── 3. Cargar modelos ───────────────────────────────────────
    models: Dict[str, object] = {}
    for name, path in [("RF", MODEL_RF_PATH), ("SVM", MODEL_SVM_PATH)]:
        if path.exists():
            models[name] = joblib.load(path)
            print(f"Loaded {name}: {path}")
        else:
            msg = f"{name} model not found at {path}"
            if ALLOW_MISSING_MODEL:
                print(f"WARNING: {msg} — skipping.")
            else:
                sys.exit(f"ERROR: {msg}. Set ALLOW_MISSING_MODEL=True to skip.")
    if not models:
        sys.exit("ERROR: No models available. Nothing to predict.")

    # ── 4. Features ─────────────────────────────────────────────
    print("Extracting per-image features...")
    df = extract_features(coco_data, id_to_species=id_to_species)
    if df.empty:
        sys.exit("ERROR: No images produced features.")
    print(f"  Feature matrix: {df.shape[0]} images x {df.shape[1]} columns.")

    # ── 5. Predicciones top-K por modelo ────────────────────────
    results = df[["file_name", "species_id_from_filename", "species_name_from_filename"]].copy()
    topk_by_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    classes_by_model: Dict[str, Optional[np.ndarray]] = {}

    for name, model in models.items():
        print(f"Running {name} top-{TOP_K}...")
        X = align_features(df, model, name)
        preds, topk_classes, topk_probs, proba_full = predict_topk(model, X, k=TOP_K)
        col = name.lower()
        results[f"pred_{col}"] = preds
        classes_by_model[name] = get_model_classes(model)
        if topk_classes is None:
            print(f"  [{name}] predict_proba not available; only top-1 stored.")
            continue
        topk_by_model[name] = (topk_classes, topk_probs, proba_full)
        results[f"top1_prob_{col}"] = np.round(topk_probs[:, 0], 4)
        for i in range(topk_classes.shape[1]):
            results[f"top{i + 1}_{col}"] = topk_classes[:, i]
            results[f"top{i + 1}_prob_{col}"] = np.round(topk_probs[:, i], 4)

    # ── 6. Acuerdo entre modelos en top-1 ───────────────────────
    if "pred_rf" in results.columns and "pred_svm" in results.columns:
        results["agree_top1"] = (results["pred_rf"] == results["pred_svm"]).astype(int)

    # ── 7. Etiquetas verdaderas (opcional) ──────────────────────
    true_path, true_map = load_true_labels()
    has_eval = False
    if true_path is None:
        print("\nNo true-labels file found. Searched at:")
        for p in TRUE_LABELS_CANDIDATES:
            print(f"  - {p}")
        print("Top-K evaluation skipped.")
    elif true_map is None:
        print(f"\nLabels file detected at {true_path} but could not be parsed. "
              f"Top-K evaluation skipped.")
    else:
        print(f"\nUsing true labels from: {true_path}")
        results["true_species"] = results["file_name"].astype(str).map(true_map)
        n_with_label = int(results["true_species"].notna().sum())
        print(f"Matched true labels for {n_with_label}/{len(results)} images.")
        if n_with_label == 0:
            print("WARNING: no file_name matched the labels file.")
        else:
            has_eval = True

    summary: dict = {
        "n_images": int(len(results)),
        "coco_json": str(coco_path),
        "models": list(models.keys()),
        "top_k": TOP_K,
        "evaluation": bool(has_eval),
    }

    if has_eval:
        per_model_summary: Dict[str, dict] = {}
        for name, (topk_classes, topk_probs, proba_full) in topk_by_model.items():
            col = name.lower()
            classes_all = classes_by_model[name]
            class_to_col = (
                {_norm_species_name(cls): i for i, cls in enumerate(classes_all)}
                if classes_all is not None else None
            )
            catalog_norm = set(class_to_col.keys()) if class_to_col else set()
            topk_classes_norm = np.array(
                [[_norm_species_name(c) for c in row] for row in topk_classes],
                dtype=object,
            )

            ranks: List[Optional[int]] = []
            in_topk: List[Optional[int]] = []
            prob_true: List[Optional[float]] = []
            true_in_catalog: List[Optional[int]] = []  # NUEVO

            for i, true_sp in enumerate(results["true_species"].values):
                if pd.isna(true_sp):
                    ranks.append(np.nan)
                    in_topk.append(np.nan)
                    prob_true.append(np.nan)
                    true_in_catalog.append(np.nan)
                    continue
                true_sp_norm = _norm_species_name(true_sp)
                true_in_catalog.append(int(true_sp_norm in catalog_norm))  # NUEVO

                p_true = np.nan
                if proba_full is not None and class_to_col is not None and true_sp_norm in class_to_col:
                    p_true = float(proba_full[i, class_to_col[true_sp_norm]])

                row_classes = topk_classes_norm[i]
                match = np.where(row_classes == true_sp_norm)[0]
                if len(match) > 0:
                    ranks.append(int(match[0]) + 1)
                    in_topk.append(1)
                else:
                    ranks.append(np.nan)
                    in_topk.append(0)
                prob_true.append(p_true)

            results[f"in_top{TOP_K}_{col}"] = in_topk
            results[f"rank_true_{col}"] = ranks
            results[f"prob_true_{col}"] = np.round(prob_true, 4)
            results[f"true_in_catalog_{col}"] = true_in_catalog  # NUEVO

            ranks_arr = np.array(ranks, dtype=float)
            in_arr = np.array(in_topk, dtype=float)
            cat_arr = np.array(true_in_catalog, dtype=float)
            valid = ~np.isnan(in_arr)
            n_eval = int(valid.sum())
            n_out_of_catalog = int(np.nansum(cat_arr[valid] == 0))  # NUEVO
            if n_eval > 0:
                print(f"\n  {name}: n_eval={n_eval}  "
                      f"fuera_de_catalogo={n_out_of_catalog}")
                model_summary = {
                    "n_eval": n_eval,
                    "n_out_of_catalog": n_out_of_catalog,
                    "mean_prob_true": float(np.nanmean(np.array(prob_true)[valid])),
                }
                for k in HITRATE_KS:  # NUEVO: hit-rate@1/3/5 explícito
                    if k > TOP_K:
                        continue
                    hr = float(np.nansum(ranks_arr[valid] <= k) / n_eval)
                    model_summary[f"hitrate_at_{k}"] = hr
                    print(f"    hit-rate@{k} = {hr*100:.1f}%")
                per_model_summary[name] = model_summary

        summary["per_model"] = per_model_summary
        summary["true_labels_source"] = str(true_path)

        # ── NUEVO: bloque LaTeX listo para pegar ─────────────────
        if per_model_summary:
            latex = build_latex_table(per_model_summary)
            print("\n" + "=" * 60)
            print("LaTeX (copiar a la sección de resultados):")
            print("=" * 60)
            print(latex)
            (OUTPUT_DIR / "resultados_topk.tex").parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_DIR / "resultados_topk.tex", "w", encoding="utf-8") as f:
                f.write(latex + "\n")

    # ── 8. Escribir salida ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "species_predictions.csv"
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")

    with open(OUTPUT_DIR / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUTPUT_DIR / 'prediction_summary.json'}")
    if has_eval:
        print(f"Saved: {OUTPUT_DIR / 'resultados_topk.tex'}")


if __name__ == "__main__":
    main()