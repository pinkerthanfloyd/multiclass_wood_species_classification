"""train_yolo_seg_partial_anno.py

YOLO11 segmentation training adapted to PARTIAL annotations.

Premisa nueva: en este dataset las anotaciones NO son exhaustivas. El
anotador marcó únicamente los rasgos que distinguen una especie de otra,
no todos los rasgos visibles. Bajo la pérdida estándar de YOLO esto
penaliza al modelo cuando predice correctamente rasgos visibles pero no
anotados, sesgándolo hacia la sub-predicción y produciendo la firma de
matriz que se observa en sem_diag_bs8_img1024 (filas y columnas
"background" muy infladas).

Cambios respecto a train_yolo_seg_sem_loss_species_diag.py:

  1. Pérdida per-imagen ASIMÉTRICA: ahora penaliza únicamente la
     SUB-predicción frente al GT. Sobre-predecir no se penaliza, porque
     en este paradigma una predicción extra puede ser un rasgo real
     que el anotador no marcó (no quiso ser exhaustivo).

  2. Penalización de EXCLUSIVIDAD species-level (sem_spexcl)
     DESACTIVADA. Su supuesto ("pares de clases no observados
     conjuntamente en GT están prohibidos") es justo lo contrario a lo
     que sucede en este dataset: pueden faltar simplemente porque el
     anotador no los consideró distintivos.

  3. DIAG_EMPHASIS se aplica con su PROPIO peso, fuera de la rampa de
     lambda_sem (LAMBDA_SEM_MAX = 0.10). En el script original su peso
     efectivo era LAMBDA_SEM_MAX * DIAG_EMPHASIS_WEIGHT ≈ 0.015, muy por
     debajo de la pérdida base. Aquí entra con DIAG_OWN_WEIGHT = 0.20.

  4. Se baja el peso del término `cls` de Ultralytics (default 0.5 -> 0.3).
     Ese es el término que genera el gradiente "este anchor es fondo" y
     es la fuente directa del castigo por predecir rasgos no anotados.
     Reducirlo es lo más cercano a un "ignore mask" sin parchear las
     internals de Ultralytics.

  5. La rampa de lambda_sem (warmup/ramp) se conserva tal cual para los
     términos per_image y per_species: arrancar suave + subir es seguro.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_yolo_seg_baseline_csvs import (
    DATASET_ROOT,
    DATA_YAML_PATH,
    SPLIT_FOLDER,
    EPOCHS,
    WORKERS,
    DEVICE,
    SEED,
    PRETRAINED,
    OPTIMIZER,
    COS_LR,
    CLOSE_MOSAIC,
    AMP,
    CACHE,
    EXIST_OK,
    BASE_MODEL_WEIGHTS,
    TEST_NMS_IOU,
    collect_records_from_split_csvs,
    collect_records_from_yaml_txts,
    evaluate_on_test_split,
    get_yolo_class,
    load_class_names,
    prepare_dataset_yaml_and_txts,
)

from train_yolo_seg_sem_loss_species_diag import (
    SemLossWrapper,
    RELATIVE_VECTORS_CSV,
    LAMBDA_SEM_MAX,
    WARMUP_END_EPOCH,
    RAMP_END_EPOCH,
    PER_IMAGE_WEIGHT,
    PER_SPECIES_WEIGHT,
    COUNT_VS_AREA,
    DIAG_EMPHASIS_WEIGHT,
    load_species_priors,
    load_diagnostic_weights,
    save_diagnostic_weights,
    DIAG_CACHE_NAME,
)


# ============================================================
# OUTPUT LOCATION
# ============================================================
OUTPUT_ROOT = Path("runs_wood_feature_partial_anno") / SPLIT_FOLDER.name


# ============================================================
# CONFIG ESPECÍFICO DE ESTE PARADIGMA
# ============================================================
# Peso de la pérdida `cls` de Ultralytics. Default = 0.5. Lo bajamos porque
# es el término que penaliza al modelo por predecir cualquier rasgo en
# regiones no anotadas — exactamente lo que NO queremos castigar bajo
# anotación selectiva.
CLS_WEIGHT_OVERRIDE = 0.3

# Peso propio (fuera del cap lambda_sem) del término diagnóstico, para que
# tenga un peso efectivo real (~0.20 vs ~0.015 del script original).
DIAG_OWN_WEIGHT = 0.20

# Si >0, se penaliza también la sobre-predicción que excede gt_count + slack.
# Mantener en 0.0 mientras no se observe runaway. Se sube si en algún
# experimento el modelo empieza a predecir basura por todas partes.
OVERPRED_SLACK = 0.0
OVERPRED_PENALTY = 0.10


EXPERIMENTS = [
    # Recomendado para la primera corrida: diag activo, cls bajo, exclusividad off.
    {"name": "partial_anno_diag_bs8_img1024",   "batch": 8, "imgsz": 1024, "diag_emphasis": True},
    # Ablación: sin diag para aislar el efecto del cls override + pérdida asimétrica.
    {"name": "partial_anno_nodiag_bs8_img1024", "batch": 8, "imgsz": 1024, "diag_emphasis": False},
]


# ============================================================
# WRAPPER DE PÉRDIDA: PARTIAL-ANNOTATION
# ============================================================
class PartialAnnoSemLossWrapper(SemLossWrapper):
    """Subclase de SemLossWrapper especializada para anotación parcial.

    - Sustituye la pérdida per-imagen por una versión ASIMÉTRICA
      (sólo penaliza pred < gt).
    - Desactiva incondicionalmente la pérdida de exclusividad species-level
      (no asume "no observado = prohibido").
    - El término diagnóstico se aplica con un peso PROPIO, FUERA de la
      rampa de lambda_sem.
    """

    def _per_image_distribution_loss(self, pred_count, gt_count, pred_area, gt_area):
        """Asimétrica: el GT bajo anotación parcial es una COTA INFERIOR del
        número/área real de rasgos en la imagen. Penalizamos sólo el
        déficit (gt - pred)+; el exceso (pred - gt)+ se ignora porque
        puede ser una predicción correcta de un rasgo no anotado.
        """
        count_short = F.relu(gt_count - pred_count)
        area_short = F.relu(gt_area - pred_area)
        loss = count_short.mean() + COUNT_VS_AREA * area_short.mean()
        if OVERPRED_SLACK > 0.0:
            count_over = F.relu(pred_count - (gt_count + OVERPRED_SLACK))
            loss = loss + OVERPRED_PENALTY * count_over.mean()
        return loss

    def _species_exclusivity_loss(self, expected_mask_logit, batch):
        return torch.zeros(
            (),
            device=expected_mask_logit.device,
            dtype=expected_mask_logit.dtype,
        )

    def __call__(self, preds, batch):
        base_loss, base_items = self.base(preds, batch)

        lam = self.lambda_sem()
        sem_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        diag_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

        compute_sem = lam > 0.0
        compute_diag = self.diag_emphasis and bool(self.diagnostic_weights) and bool(self.species_priors)

        if compute_sem or compute_diag:
            try:
                _device_type = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.amp.autocast(device_type=_device_type, enabled=False):
                    pred_scores, pred_masks, proto = self._parse_preds(preds)
                    pred_scores = pred_scores.float()
                    pred_masks = pred_masks.float()
                    proto = proto.float()
                    device = pred_scores.device

                    B, _A, _ = pred_scores.shape
                    _, M, mh, mw = proto.shape
                    total_pixels = float(mh * mw)

                    score_sig = pred_scores.sigmoid()
                    pred_count = score_sig.sum(dim=1)
                    weighted_coeffs = score_sig.transpose(1, 2) @ pred_masks
                    proto_flat = proto.view(B, M, -1)
                    expected_mask_logit = weighted_coeffs @ proto_flat
                    pred_area = expected_mask_logit.sigmoid().sum(dim=2) / total_pixels

                    gt_count, gt_area = self._gt_count_and_area(batch, B, device, mh, mw)

                    if compute_sem:
                        per_im = self._per_image_distribution_loss(
                            pred_count, gt_count, pred_area, gt_area
                        )
                        per_sp = self._per_species_prior_loss(
                            batch, pred_count, pred_area, device
                        )
                        sem = (
                            PER_IMAGE_WEIGHT * per_im
                            + PER_SPECIES_WEIGHT * per_sp
                        )
                        sem_loss = sem.to(base_loss.dtype)

                    if compute_diag:
                        diag = self._diagnostic_emphasis_loss(
                            batch, pred_count, pred_area, device
                        )
                        diag_loss = diag.to(base_loss.dtype)
            except Exception as exc:
                if self._last_failure_epoch != self._epoch:
                    self._last_failure_epoch = self._epoch
                    print(f"[partial_anno] WARNING (epoch {self._epoch}): "
                          f"sem/diag computation failed: "
                          f"{type(exc).__name__}: {exc}")

        total_loss = base_loss + lam * sem_loss + DIAG_OWN_WEIGHT * diag_loss

        self._last_sem_loss_value = float(sem_loss.detach().item())
        self._last_lambda_value = float(lam)
        self._last_exclusivity_value = 0.0
        self._last_diag_value = float(diag_loss.detach().item())
        return total_loss, base_items


# ============================================================
# CALLBACKS
# ============================================================
def make_partial_anno_callbacks(
    class_names: List[str],
    species_priors,
    diagnostic_weights,
    diag_emphasis: bool,
):
    state: Dict[str, object] = {"wrapper": None, "csv_path": None}

    def _build_wrapper(criterion, reg_max):
        return PartialAnnoSemLossWrapper(
            base_criterion=criterion,
            class_names=class_names,
            species_priors=species_priors,
            species_overlap_priors=None,
            diagnostic_weights=diagnostic_weights,
            iaway_bias=False,
            diag_emphasis=diag_emphasis,
            nc=len(class_names),
            reg_max=reg_max,
        )

    def on_train_start(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None and hasattr(trainer.model, "init_criterion"):
            criterion = trainer.model.init_criterion()
            trainer.model.criterion = criterion
        if criterion is None:
            print("[partial_anno] WARNING: no criterion at on_train_start.")
            return
        if isinstance(criterion, PartialAnnoSemLossWrapper):
            state["wrapper"] = criterion
            return

        reg_max = getattr(criterion, "reg_max", 16)
        wrapper = _build_wrapper(criterion, reg_max)
        trainer.model.criterion = wrapper
        state["wrapper"] = wrapper

        try:
            ema_obj = getattr(trainer, "ema", None)
            ema_model = getattr(ema_obj, "ema", None) if ema_obj is not None else None
            if ema_model is not None:
                ema_criterion = getattr(ema_model, "criterion", None)
                if ema_criterion is None and hasattr(ema_model, "init_criterion"):
                    ema_criterion = ema_model.init_criterion()
                    ema_model.criterion = ema_criterion
                if ema_criterion is not None and not isinstance(ema_criterion, PartialAnnoSemLossWrapper):
                    ema_wrapper = _build_wrapper(
                        ema_criterion, getattr(ema_criterion, "reg_max", reg_max)
                    )
                    ema_wrapper._accumulate_overlaps = False
                    ema_model.criterion = ema_wrapper
        except Exception as exc:
            print(f"[partial_anno] WARNING: could not wrap EMA criterion: {exc}")

        try:
            save_dir = Path(getattr(trainer, "save_dir", "."))
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "sem_loss_log.csv"
            if not csv_path.exists():
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("epoch,batch,lambda,sem_loss,diagnostic\n")
            state["csv_path"] = csv_path
        except Exception as exc:
            print(f"[partial_anno] WARNING: log setup failed: {exc}")

        print(f"[partial_anno] Wrapped criterion (diag_emphasis={diag_emphasis}, "
              f"cls={CLS_WEIGHT_OVERRIDE}, diag_own_weight={DIAG_OWN_WEIGHT}, "
              f"overpred_slack={OVERPRED_SLACK}).")

    def on_train_epoch_start(trainer):
        w = state["wrapper"]
        if isinstance(w, PartialAnnoSemLossWrapper):
            w.set_epoch(int(trainer.epoch))

    def on_train_batch_end(trainer):
        w = state["wrapper"]
        csv_path = state["csv_path"]
        if not isinstance(w, PartialAnnoSemLossWrapper) or csv_path is None:
            return
        try:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{int(getattr(trainer, 'epoch', -1))},"
                    f"{int(getattr(trainer, '_i', -1))},"
                    f"{w._last_lambda_value:.6f},"
                    f"{w._last_sem_loss_value:.6f},"
                    f"{w._last_diag_value:.6f}\n"
                )
        except Exception:
            pass

    return on_train_start, on_train_epoch_start, on_train_batch_end


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_experiment(
    experiment, dataset_yaml_path, class_names, by_split,
    species_priors, diagnostic_weights,
) -> dict:
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]
    diag_emphasis = bool(experiment.get("diag_emphasis", False))

    expected_dir = OUTPUT_ROOT / run_name
    summary_path = expected_dir / "experiment_summary.json"

    if summary_path.exists():
        print(f"[skip] {run_name}: experiment_summary.json already present.")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    YOLO = get_yolo_class()
    last_path = expected_dir / "weights" / "last.pt"
    resuming = last_path.exists()
    model = YOLO(str(last_path)) if resuming else YOLO(BASE_MODEL_WEIGHTS)
    if resuming:
        print(f"[resume] {run_name}: resuming from {last_path}")

    cb_start, cb_epoch, cb_batch = make_partial_anno_callbacks(
        class_names=class_names,
        species_priors=species_priors,
        diagnostic_weights=diagnostic_weights,
        diag_emphasis=diag_emphasis,
    )
    model.add_callback("on_train_start", cb_start)
    model.add_callback("on_train_epoch_start", cb_epoch)
    model.add_callback("on_train_batch_end", cb_batch)

    try:
        if resuming:
            train_results = model.train(resume=True)
        else:
            train_results = model.train(
                data=str(dataset_yaml_path),
                epochs=EPOCHS,
                batch=batch_size,
                imgsz=imgsz,
                device=DEVICE,
                workers=WORKERS,
                pretrained=PRETRAINED,
                optimizer=OPTIMIZER,
                cos_lr=COS_LR,
                close_mosaic=CLOSE_MOSAIC,
                amp=AMP,
                cache=CACHE,
                seed=SEED,
                project=str(OUTPUT_ROOT),
                name=run_name,
                exist_ok=EXIST_OK,
                cls=CLS_WEIGHT_OVERRIDE,
                plots=True,
                save=True,
                verbose=True,
            )
        experiment_dir = Path(train_results.save_dir)
    except Exception as exc:
        print(f"[resume] {run_name}: train(resume=True) raised "
              f"{type(exc).__name__}: {exc}. Assuming training complete.")
        experiment_dir = expected_dir

    best_model_path = experiment_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        best_model_path = experiment_dir / "weights" / "last.pt"

    YOLO = get_yolo_class()
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(dataset_yaml_path),
        split="test",
        imgsz=imgsz,
        batch=batch_size,
        conf=0.001,
        iou=TEST_NMS_IOU,
        device=DEVICE,
        plots=True,
        save_json=True,
        project=str(experiment_dir),
        name="test_eval",
        exist_ok=True,
        retina_masks=True,
        verbose=True,
    )

    try:
        df_cm = val_results.confusion_matrix.to_df()
        df_cm.to_csv(experiment_dir / "ultralytics_confusion_matrix.csv", index=False)
    except Exception:
        pass

    custom = evaluate_on_test_split(
        best_model_path, experiment_dir, imgsz, class_names, by_split["test"]
    )

    summary = {
        "experiment": run_name,
        "batch": batch_size,
        "imgsz": imgsz,
        "diag_emphasis": diag_emphasis,
        "cls_weight_override": CLS_WEIGHT_OVERRIDE,
        "diag_own_weight": DIAG_OWN_WEIGHT,
        "overpred_slack": OVERPRED_SLACK,
        "best_model_path": str(best_model_path),
        "ultralytics_box_map50": float(val_results.box.map50),
        "ultralytics_box_map50_95": float(val_results.box.map),
        "ultralytics_seg_map50": float(val_results.seg.map50),
        "ultralytics_seg_map50_95": float(val_results.seg.map),
        "custom_micro_precision": float(custom["micro_precision"]),
        "custom_micro_recall": float(custom["micro_recall"]),
        "custom_micro_f1": float(custom["micro_f1"]),
        "custom_macro_precision": float(custom["macro_precision"]),
        "custom_macro_recall": float(custom["macro_recall"]),
        "custom_macro_f1": float(custom["macro_f1"]),
        "custom_total_gt_instances": int(custom["total_gt_instances"]),
        "custom_total_pred_instances": int(custom["total_pred_instances"]),
    }

    tmp = experiment_dir / "experiment_summary.json.tmp"
    final = experiment_dir / "experiment_summary.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(final)
    return summary


def main() -> None:
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML_PATH}")

    class_names = load_class_names(DATA_YAML_PATH)
    records = collect_records_from_split_csvs(DATASET_ROOT)
    if not records:
        records = collect_records_from_yaml_txts(DATASET_ROOT, DATA_YAML_PATH)
    if not records:
        raise RuntimeError("No split records found.")

    dataset_yaml_path, by_split = prepare_dataset_yaml_and_txts(records, class_names)
    species_priors = load_species_priors(RELATIVE_VECTORS_CSV, class_names)
    diagnostic_weights = load_diagnostic_weights(RELATIVE_VECTORS_CSV, class_names)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if diagnostic_weights:
        save_diagnostic_weights(diagnostic_weights, class_names, OUTPUT_ROOT / DIAG_CACHE_NAME)

    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_experiment(
            experiment, dataset_yaml_path, class_names, by_split,
            species_priors, diagnostic_weights,
        )
        all_summaries.append(summary)

    pd.DataFrame(all_summaries).to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print("Done. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
