import json
import joblib
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC


ID_SPECIES_PATH = "models/inferences/especies.json"
INSTANCES_PATH = "models/inferences/sem_loss_stratified"

OUTPUT_DIR = "species_inferences"
OUTPUT_FEATURES = os.path.join(OUTPUT_DIR, "features_por_imagen.csv")
OUTPUT_PREDICTIONS = os.path.join(OUTPUT_DIR, "predictions.csv")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "classification_report.txt")
OUTPUT_MODEL_RF = "models/species_classifier/species_model_rf.joblib"
OUTPUT_MODEL_SVM = "models/species_classifier/species_model_svm.joblib"
WRONG_PRED_DIR = os.path.join(OUTPUT_DIR, "wrong_predictions")

TARGET_IDS = [
    '0009', '0013', '0041', '0045', '0071', '0085', '0095', '0096',
    '0114', '0157', '0158', '0181', '0195', '0207', '0239', '0243',
    '0313', '0380', '0425', '0003', '0006', '0007', '0678', '0030',
    '0033', '0034', '0035', '0679', '0066', '0070', '0072', '0073',
    '0080', '0081', '0091', '0092', '0093', '0115', '0129', '0130',
    '0147', '0214', '0236', '0238', '0245', '0244', '0256', '0257',
    '0258', '0259', '0269', '0270', '0293', '0294', '0297', '0295',
    '0296', '0301', '0302', '0323', '0326', '0324', '0325', '0331',
    '0354', '0353', '0367', '0366', '0371', '0372', '0375', '0318',
    '0396', '0399', '0418', '0426', '0430', '0431', '0461', '0479',
    '0480', '0478', '0530', '0541', '0542', '0552', '0550', '0553',
    '0554', '0606', '0607', '0608', '0613', '0621', '0625', '0648'
]


# ── Create output directories ─────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WRONG_PRED_DIR, exist_ok=True)
os.makedirs("models/species_classifier", exist_ok=True)


# ── Helper: tee print (print + write to file) ─────────────────
report_lines = []

def tee_print(*args, **kwargs):
    """Print to stdout and capture the line for the report file."""
    import io
    buf = io.StringIO()
    print(*args, file=buf, **kwargs)
    text = buf.getvalue()
    sys.stdout.write(text)
    sys.stdout.flush()
    report_lines.append(text)


# ── Carga de datos ──────────────────────────────────────────────

with open(ID_SPECIES_PATH, "r", encoding="utf-8") as f:
    id_to_species = json.load(f)

with open(INSTANCES_PATH, "r", encoding="utf-8") as f:
    coco_data = json.load(f)


def get_species_name(sp_id):
    return id_to_species.get(sp_id, f"Especie_Desconocida_{sp_id}")


categories = {
    cat["id"]: cat["name"]
    for cat in coco_data["categories"]
}


# ── Extracción de features ──────────────────────────────────────

image_rows = {}

for img in tqdm(coco_data["images"], desc="Procesando imágenes"):
    file_name = img["file_name"]
    sp_id = file_name.split("-")[0]

    if sp_id not in TARGET_IDS:
        continue

    img_area = img["width"] * img["height"]

    row = {
        "image_id": img["id"],
        "file_name": file_name,
        "species_id": sp_id,
        "species_name": get_species_name(sp_id),
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


areas_by_image_class = {}

for ann in tqdm(coco_data["annotations"], desc="Procesando anotaciones"):
    img_id = ann["image_id"]

    if img_id not in image_rows:
        continue

    cat_name = categories[ann["category_id"]]
    prefix = f"Clase_{cat_name}"

    area = float(ann.get("area", 0.0))
    img_area = image_rows[img_id]["img_area"]

    image_rows[img_id]["total_annotations"] += 1
    image_rows[img_id]["total_annotated_area"] += area
    image_rows[img_id]["total_density_%"] += (area / img_area) * 100.0

    image_rows[img_id][f"{prefix}_count"] += 1
    image_rows[img_id][f"{prefix}_area_sum"] += area
    image_rows[img_id][f"{prefix}_density_%"] += (area / img_area) * 100.0

    key = (img_id, cat_name)
    areas_by_image_class.setdefault(key, []).append(area)


for (img_id, cat_name), areas in tqdm(
    areas_by_image_class.items(),
    desc="Calculando métricas"
):
    prefix = f"Clase_{cat_name}"
    arr = np.array(areas, dtype=float)

    image_rows[img_id][f"{prefix}_area_mean"] = arr.mean()
    image_rows[img_id][f"{prefix}_area_std"] = arr.std(ddof=1) if len(arr) > 1 else 0.0


df = pd.DataFrame(image_rows.values())
df.to_csv(OUTPUT_FEATURES, index=False, encoding="utf-8-sig")

tee_print(f"Dataset generado: {OUTPUT_FEATURES}")
tee_print(df.shape)
tee_print(df["species_name"].value_counts().to_string())


# ── Preparación de X e y ────────────────────────────────────────

drop_cols = [
    "image_id",
    "file_name",
    "species_id",
    "species_name",
    "img_area",
]

X = df.drop(columns=drop_cols)
y = df["species_name"]


# ── Definición de modelos ───────────────────────────────────────

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

svm_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])


# ── Evaluación mediante CV ──────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
}

tee_print("\n===== CROSS VALIDATION — RANDOM FOREST =====")
rf_cv = cross_validate(
    rf_pipeline, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)
tee_print(f"Accuracy por fold: {rf_cv['test_accuracy']}")
tee_print(f"Accuracy media:    {rf_cv['test_accuracy'].mean():.4f} ± {rf_cv['test_accuracy'].std():.4f}")
tee_print(f"F1 macro por fold: {rf_cv['test_f1_macro']}")
tee_print(f"F1 macro media:    {rf_cv['test_f1_macro'].mean():.4f} ± {rf_cv['test_f1_macro'].std():.4f}")

tee_print("\n===== CROSS VALIDATION — SVM =====")
svm_cv = cross_validate(
    svm_pipeline, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)
tee_print(f"Accuracy por fold: {svm_cv['test_accuracy']}")
tee_print(f"Accuracy media:    {svm_cv['test_accuracy'].mean():.4f} ± {svm_cv['test_accuracy'].std():.4f}")
tee_print(f"F1 macro por fold: {svm_cv['test_f1_macro']}")
tee_print(f"F1 macro media:    {svm_cv['test_f1_macro'].mean():.4f} ± {svm_cv['test_f1_macro'].std():.4f}")


# ── Selección y entrenamiento final ─────────────────────────────

rf_mean = rf_cv["test_f1_macro"].mean()
svm_mean = svm_cv["test_f1_macro"].mean()

best_name = "Random Forest"
best_pipeline = rf_pipeline
best_output = OUTPUT_MODEL_RF

tee_print(f"\n>> Mejor modelo por F1 macro: {best_name} ({max(rf_mean, svm_mean):.4f})")

best_name = "SVM"
best_pipeline = svm_pipeline
best_output = OUTPUT_MODEL_SVM

tee_print(f"\n>> Mejor modelo por F1 macro: {best_name} ({max(rf_mean, svm_mean):.4f})")




# Refit sobre todos los datos
best_pipeline.fit(X, y)
joblib.dump(best_pipeline, best_output)
tee_print(f"Modelo guardado en: {best_output}")


# ── Out-of-fold predictions + CSV ───────────────────────────────

y_pred_cv = cross_val_predict(best_pipeline, X, y, cv=cv, n_jobs=-1)

tee_print(f"\n===== CLASSIFICATION REPORT ({best_name}, out-of-fold) =====")
report_text = classification_report(y, y_pred_cv, zero_division=0)
tee_print(report_text)

# Build predictions DataFrame
predictions_df = pd.DataFrame({
    "image_file": df["file_name"].values,
    "predicted_species": y_pred_cv,
    "real_species": y.values,
})
predictions_df["correct"] = (predictions_df["predicted_species"] == predictions_df["real_species"]).astype(int)

predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False, encoding="utf-8-sig")
tee_print(f"\nPredicciones guardadas en: {OUTPUT_PREDICTIONS}")
tee_print(f"Total imágenes: {len(predictions_df)}")
tee_print(f"Correctas: {predictions_df['correct'].sum()}")
tee_print(f"Incorrectas: {(~predictions_df['correct'].astype(bool)).sum()}")


# ── Save full report to file ────────────────────────────────────

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("".join(report_lines))

print(f"Reporte completo guardado en: {OUTPUT_REPORT}")


# ── Generate comparison images for wrong predictions ────────────
# We import the visualization helpers from visualize_predictions.
# This requires the YOLO model + label files to be available.

wrong_df = predictions_df[predictions_df["correct"] == 0]

if len(wrong_df) == 0:
    print("\nNo hay predicciones incorrectas — no se generan imágenes de comparación.")
else:
    print(f"\nGenerando {len(wrong_df)} imágenes de comparación para predicciones incorrectas...")

    try:
        import gc
        import cv2
        import torch

        # Import helpers from the visualize_predictions module
        from train_yolo_seg_baseline_csvs import (
            load_class_names,
            get_yolo_class,
            parse_yolo_seg_label_file,
            result_to_pred_instances,
        )
        from visualize_predictions import (
            render_side_by_side,
        )

        # ── Config — adjust these paths to match your setup ──
        MODEL_PATH = "models/sem_loss_stratified/best.pt"
        DATA_YAML = "data/raw_selected/data.yaml"
        DATASET_ROOT = Path("data/raw_selected")
        IMGSZ = 1024
        CONF = 0.25
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        class_names = load_class_names(Path(DATA_YAML))
        YOLO = get_yolo_class()
        model = YOLO(MODEL_PATH)

        # Build a lookup: file_name -> image info from COCO data
        # so we can find the corresponding label file
        coco_image_lookup = {}
        for img_info in coco_data["images"]:
            coco_image_lookup[img_info["file_name"]] = img_info

        # We also need to find the YOLO label path for each image.
        # Convention: labels are in a parallel "labels" folder next to "images",
        # with .txt extension.
        def find_image_and_label(file_name):
            """Try to locate the image file and its YOLO label."""
            # Search common locations
            candidates = [
                DATASET_ROOT / "images" / file_name,
                DATASET_ROOT / file_name,
            ]
            image_path = None
            for c in candidates:
                if c.exists():
                    image_path = c
                    break

            if image_path is None:
                # Try recursive search
                found = list(DATASET_ROOT.rglob(file_name))
                if found:
                    image_path = found[0]

            if image_path is None:
                return None, None

            # Label path: parallel labels/ folder, .txt extension
            label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
            if not label_path.exists():
                # Try same folder
                label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                label_path = None

            return image_path, label_path

        generated = 0
        for idx, row in tqdm(wrong_df.iterrows(), total=len(wrong_df),
                             desc="Generando comparaciones"):
            file_name = row["image_file"]
            pred_species = row["predicted_species"]
            real_species = row["real_species"]

            image_path, label_path = find_image_and_label(file_name)

            if image_path is None:
                print(f"  [SKIP] No se encontró imagen: {file_name}")
                continue

            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  [SKIP] No se pudo cargar: {image_path}")
                continue

            h, w = image.shape[:2]

            # Parse GT annotations
            gt_instances = []
            if label_path and label_path.exists():
                gt_instances = parse_yolo_seg_label_file(label_path, w, h)

            # Run YOLO prediction
            results = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF,
                iou=0.7,
                max_det=300,
                retina_masks=False,
                device=DEVICE,
                verbose=False,
            )
            pred_instances = result_to_pred_instances(results[0])

            # Render side-by-side comparison
            canvas = render_side_by_side(
                image=image,
                gt_instances=gt_instances,
                pred_instances=pred_instances,
                class_names=class_names,
            )

            # Add a header with species info
            header_h = 40
            header = np.zeros((header_h, canvas.shape[1], 3), dtype=np.uint8)
            text = f"REAL: {real_species}  |  PREDICTED: {pred_species}"
            cv2.putText(header, text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            canvas = np.vstack([header, canvas])

            # Save
            stem = Path(file_name).stem
            out_path = os.path.join(WRONG_PRED_DIR, f"{stem}_wrong.jpg")
            cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            generated += 1

            # Free memory
            del results, pred_instances, image, canvas, gt_instances
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{generated} imágenes de comparación guardadas en: {WRONG_PRED_DIR}")

    except ImportError as e:
        print(f"\n[WARNING] No se pudieron generar imágenes de comparación.")
        print(f"  Falta dependencia: {e}")
        print(f"  Asegúrate de que visualize_predictions.py y train_yolo_seg_baseline_csvs.py")
        print(f"  estén en el mismo directorio o en el PYTHONPATH.")
    except Exception as e:
        print(f"\n[ERROR] Error generando imágenes de comparación: {e}")
        import traceback
        traceback.print_exc()


print(f"\n{'='*60}")
print(f"Todos los resultados guardados en: {OUTPUT_DIR}/")
print(f"  - {OUTPUT_FEATURES}")
print(f"  - {OUTPUT_PREDICTIONS}")
print(f"  - {OUTPUT_REPORT}")
print(f"  - {WRONG_PRED_DIR}/ (imágenes de predicciones incorrectas)")
print(f"{'='*60}")
