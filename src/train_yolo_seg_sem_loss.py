"""Training script for YOLO11 segmentation with a semantic / co-occurrence loss (sem_loss).

NOTE on Ultralytics 8.4.45: v8SegmentationLoss returns a 5-component loss_items
tensor (box, seg, cls, dfl, semseg). The validator uses an EMA copy of the
model whose criterion is the unwrapped one. Therefore the wrapper DOES NOT
alter the size of loss_items - sem_loss only contributes to the total scalar
loss for the backward pass and is logged separately.
"""

from __future__ import annotations

import json
import traceback
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

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
    TEST_CONFIDENCE,
    TEST_NMS_IOU,
    MAX_DET,
    RETINA_MASKS,
    collect_records_from_split_csvs,
    collect_records_from_yaml_txts,
    evaluate_on_test_split,
    get_yolo_class,
    load_class_names,
    prepare_dataset_yaml_and_txts,
)

OUTPUT_ROOT = Path("runs_wood_feature_sem_loss") / SPLIT_FOLDER.name

RELATIVE_VECTORS_CSV = Path("outputs/relative_feature_vectors.csv")

LAMBDA_SEM_MAX = 0.10
WARMUP_END_EPOCH = 30
RAMP_END_EPOCH = 80

PER_IMAGE_WEIGHT = 0.7
PER_SPECIES_WEIGHT = 0.3
COUNT_VS_AREA = 0.5

IAWA_PRESENCE_PENALTY = 0.5
IAWA_CONTAINMENT_PENALTY = 0.5

CLASS_NAME_VESSEL = "V1"
CLASS_NAME_RAY = "radio"
CLASS_NAMES_VESSEL_CONTENT = ("56", "58")

EXPERIMENTS = [
    {"name": "sem_cooc_bs4_img1024", "batch": 4, "imgsz": 1024, "iaway_bias": False},
    {"name": "sem_cooc_bs8_img1024", "batch": 8, "imgsz": 1024, "iaway_bias": False},
    {"name": "sem_cooc_iaway_bs4_img1024", "batch": 4, "imgsz": 1024, "iaway_bias": True},
    {"name": "sem_cooc_iaway_bs8_img1024", "batch": 8, "imgsz": 1024, "iaway_bias": True},
]


def parse_species_id_from_filename(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/").split("/")[-1]).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def safe_feature_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not text:
        text = "unknown"
    return f"feat_{text}"


def load_species_priors(csv_path: Path, class_names: List[str]) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    if not csv_path.exists():
        print(f"[sem_loss] WARNING: {csv_path} not found. Per-species prior disabled.")
        return None

    df = pd.read_csv(csv_path)
    feature_columns = [safe_feature_name(name) for name in class_names]
    presence_columns = [f"presence_{col}" for col in feature_columns]
    area_columns = [f"area_{col}" for col in feature_columns]

    use_presence = all(c in df.columns for c in presence_columns)
    use_area = all(c in df.columns for c in area_columns)
    if not use_presence and not use_area:
        print(f"[sem_loss] WARNING: {csv_path} does not expose presence/area columns. Per-species prior disabled.")
        return None

    if "species_id" not in df.columns:
        print(f"[sem_loss] WARNING: {csv_path} missing species_id. Per-species prior disabled.")
        return None

    priors: Dict[str, Dict[str, np.ndarray]] = {}
    for species_id, group in df.groupby("species_id"):
        species_id = str(species_id)
        entry: Dict[str, np.ndarray] = {}
        if use_presence:
            entry["presence"] = group[presence_columns].mean().to_numpy(dtype=np.float32)
        if use_area:
            entry["area"] = group[area_columns].mean().to_numpy(dtype=np.float32)
        priors[species_id] = entry

    print(f"[sem_loss] Loaded species priors for {len(priors)} species (presence={use_presence}, area={use_area}).")
    return priors


def build_iaway_class_indices(class_names: List[str]) -> Dict[str, object]:
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    return {
        "vessel_idx": name_to_idx.get(CLASS_NAME_VESSEL),
        "ray_idx": name_to_idx.get(CLASS_NAME_RAY),
        "vessel_content_idx": [name_to_idx[n] for n in CLASS_NAMES_VESSEL_CONTENT if n in name_to_idx],
    }


class SemLossWrapper(torch.nn.Module):
    """Decorates the native v8SegmentationLoss with a sem_loss term.

    Returns (total_loss, loss_items) like the original criterion. loss_items
    is returned UNCHANGED in shape - sem_loss only contributes to total_loss.
    """

    def __init__(self, base_criterion, class_names, species_priors, iaway_bias, nc, reg_max):
        super().__init__()
        self.base = base_criterion
        self.class_names = class_names
        self.nc = nc
        self.reg_max = reg_max
        self.iaway_bias = iaway_bias
        self.iaway_idx = build_iaway_class_indices(class_names) if iaway_bias else {}
        self.species_priors = species_priors or {}
        self._epoch = 0
        self._max_epochs = EPOCHS
        self._last_sem_loss_value = 0.0
        self._last_lambda_value = 0.0
        self._last_per_image_value = 0.0
        self._last_per_species_value = 0.0
        self._last_iaway_value = 0.0
        self._last_failure_epoch = -1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def lambda_sem(self) -> float:
        if self._epoch < WARMUP_END_EPOCH:
            return 0.0
        if self._epoch >= RAMP_END_EPOCH:
            return float(LAMBDA_SEM_MAX)
        progress = (self._epoch - WARMUP_END_EPOCH) / max(1, RAMP_END_EPOCH - WARMUP_END_EPOCH)
        return float(LAMBDA_SEM_MAX) * float(progress)

    def __call__(self, preds, batch):
        base_loss, base_items = self.base(preds, batch)
        lam = self.lambda_sem()
        sem_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        if lam > 0.0:
            try:
                sem_loss = self._compute_sem_loss(preds, batch).to(base_loss.dtype)
            except Exception as exc:
                # Print the full traceback only once per epoch so the log is
                # readable but we still see what is failing.
                if self._last_failure_epoch != self._epoch:
                    self._last_failure_epoch = self._epoch
                    print(
                        f"[sem_loss] WARNING (epoch {self._epoch}): sem_loss computation failed: {exc}"
                    )
                    traceback.print_exc()

        total_loss = base_loss + lam * sem_loss
        self._last_sem_loss_value = float(sem_loss.detach().item())
        self._last_lambda_value = float(lam)
        return total_loss, base_items

    def _parse_preds(self, preds):
        # Ultralytics 8.4.45 passes preds as a dict where the head outputs are
        # already split out: preds["scores"] is (B, nc, A) class logits and
        # preds["mask_coefficient"] is (B, M, A) mask coefficients. preds["feats"]
        # contains the BACKBONE feature maps (channels 64/128/256), NOT the
        # head outputs, so it cannot be reshaped with no = reg_max*4 + nc.
        # Older Ultralytics versions used a tuple (feats, mask_coefficient, proto)
        # where feats DID contain the head outputs - we still support that.
        if isinstance(preds, dict):
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()        # (B, A, nc)
            pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()  # (B, A, M)
            proto = preds["proto"]
            if isinstance(proto, tuple) and len(proto) == 2:
                proto = proto[0]  # discard pred_semseg
        else:
            feats, pred_masks_raw, proto = preds if len(preds) == 3 else preds[1]
            no = self.reg_max * 4 + self.nc
            feat_concat = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], dim=2)
            _pred_distri, pred_scores = feat_concat.split((self.reg_max * 4, self.nc), dim=1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()       # (B, A, nc)
            pred_masks = pred_masks_raw.permute(0, 2, 1).contiguous()      # (B, A, M)

        return pred_scores, pred_masks, proto

    def _gt_count_and_area(self, batch, batch_size, device, mask_h, mask_w):
        gt_count = torch.zeros(batch_size, self.nc, device=device, dtype=torch.float32)
        gt_area = torch.zeros(batch_size, self.nc, device=device, dtype=torch.float32)
        cls_t = batch.get("cls")
        bidx_t = batch.get("batch_idx")
        masks_t = batch.get("masks")
        if cls_t is None or bidx_t is None or cls_t.numel() == 0:
            return gt_count, gt_area
        cls_long = cls_t.long().flatten()
        bidx_long = bidx_t.long().flatten()
        gt_count.index_put_(
            (bidx_long, cls_long),
            torch.ones_like(cls_long, dtype=torch.float32),
            accumulate=True,
        )
        if masks_t is not None and masks_t.numel() > 0:
            total_pixels = float(mask_h * mask_w)
            if masks_t.dim() == 3:
                masks_resized = self._resize_masks(masks_t, mask_h, mask_w).long()
                for b in range(batch_size):
                    targets_in_b = (bidx_long == b).nonzero(as_tuple=True)[0]
                    if targets_in_b.numel() == 0:
                        continue
                    image_mask = masks_resized[b]
                    for local_idx, t_idx in enumerate(targets_in_b):
                        c = cls_long[t_idx].item()
                        area_pixels = float((image_mask == (local_idx + 1)).sum().item())
                        gt_area[b, c] += area_pixels / total_pixels
            elif masks_t.dim() == 4:
                masks_resized = self._resize_masks(masks_t, mask_h, mask_w)
                for b in range(batch_size):
                    targets_in_b = (bidx_long == b).nonzero(as_tuple=True)[0]
                    if targets_in_b.numel() == 0:
                        continue
                    for local_idx, t_idx in enumerate(targets_in_b):
                        c = cls_long[t_idx].item()
                        if local_idx < masks_resized.shape[1]:
                            area_pixels = float(masks_resized[b, local_idx].sum().item())
                            gt_area[b, c] += area_pixels / total_pixels
        return gt_count, gt_area

    @staticmethod
    def _resize_masks(masks, mask_h, mask_w):
        if masks.dim() == 3 and masks.shape[-2:] == (mask_h, mask_w):
            return masks
        if masks.dim() == 4 and masks.shape[-2:] == (mask_h, mask_w):
            return masks
        if masks.dim() == 3:
            return F.interpolate(masks.unsqueeze(1).float(), size=(mask_h, mask_w), mode="nearest").squeeze(1)
        if masks.dim() == 4:
            return F.interpolate(masks.float(), size=(mask_h, mask_w), mode="nearest")
        raise ValueError(f"Unexpected masks shape: {masks.shape}")

    def _compute_sem_loss(self, preds, batch):
        # Force float32 and disable autocast inside this block. With AMP the
        # raw model outputs are float16 and the matmuls / KL operations were
        # mixing fp16 with the fp32 GT tensors, throwing dtype errors that the
        # outer try/except was silently swallowing into a zero loss.
        # Use the new torch.amp API; auto-detect device type for CPU/CUDA.
        _device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=_device_type, enabled=False):
            pred_scores, pred_masks, proto = self._parse_preds(preds)
            pred_scores = pred_scores.float()
            pred_masks = pred_masks.float()
            proto = proto.float()

            device = pred_scores.device
            B, A, _ = pred_scores.shape
            _, M, mask_h, mask_w = proto.shape
            total_pixels = float(mask_h * mask_w)

            score_sig = pred_scores.sigmoid()
            pred_count = score_sig.sum(dim=1)

            weighted_coeffs = score_sig.transpose(1, 2) @ pred_masks
            proto_flat = proto.view(B, M, -1)
            expected_mask_logit = weighted_coeffs @ proto_flat
            pred_area = expected_mask_logit.sigmoid().sum(dim=2) / total_pixels

            gt_count, gt_area = self._gt_count_and_area(batch, B, device, mask_h, mask_w)

            per_image_loss = self._per_image_distribution_loss(pred_count, gt_count, pred_area, gt_area)
            per_species_loss = self._per_species_prior_loss(batch, pred_count, pred_area, device)
            iaway_loss = self._iaway_bias_loss(pred_count) if self.iaway_bias else torch.zeros((), device=device)

            sem = (
                PER_IMAGE_WEIGHT * per_image_loss
                + PER_SPECIES_WEIGHT * per_species_loss
                + iaway_loss
            )

            self._last_per_image_value = float(per_image_loss.detach().item()) if torch.is_tensor(per_image_loss) else float(per_image_loss)
            self._last_per_species_value = float(per_species_loss.detach().item()) if torch.is_tensor(per_species_loss) else float(per_species_loss)
            self._last_iaway_value = float(iaway_loss.detach().item()) if torch.is_tensor(iaway_loss) else float(iaway_loss)

            return sem

    @staticmethod
    def _to_distribution(t, eps=1e-8):
        s = t.sum(dim=-1, keepdim=True)
        safe = torch.where(s > 0, s, torch.ones_like(s))
        out = t / safe
        out = out + eps
        out = out / out.sum(dim=-1, keepdim=True)
        return out

    @classmethod
    def _kl(cls, target, pred):
        target_dist = cls._to_distribution(target)
        pred_dist = cls._to_distribution(pred)
        return (target_dist * (target_dist.log() - pred_dist.log())).sum(dim=-1).mean()

    def _per_image_distribution_loss(self, pred_count, gt_count, pred_area, gt_area):
        kl_count = self._kl(gt_count, pred_count)
        l1_area = F.l1_loss(pred_area, gt_area)
        return kl_count + COUNT_VS_AREA * l1_area

    def _per_species_prior_loss(self, batch, pred_count, pred_area, device):
        if not self.species_priors:
            return torch.zeros((), device=device)
        im_files = batch.get("im_file") or []
        if not im_files:
            return torch.zeros((), device=device)

        species_ids = [parse_species_id_from_filename(f) for f in im_files]
        loss = torch.zeros((), device=device)
        n_terms = 0
        pred_presence = torch.tanh(pred_count)

        for b, sp in enumerate(species_ids):
            entry = self.species_priors.get(sp)
            if entry is None:
                continue
            presence = entry.get("presence")
            if presence is not None:
                target = torch.tensor(presence, device=device, dtype=pred_presence.dtype)
                loss = loss + F.l1_loss(pred_presence[b], target)
                n_terms += 1
            area = entry.get("area")
            if area is not None:
                target = torch.tensor(area, device=device, dtype=pred_area.dtype)
                loss = loss + F.l1_loss(pred_area[b], target)
                n_terms += 1

        if n_terms == 0:
            return torch.zeros((), device=device)
        return loss / float(n_terms)

    def _iaway_bias_loss(self, pred_count):
        device = pred_count.device
        loss = torch.zeros((), device=device)
        pred_presence = torch.tanh(pred_count)

        vessel_idx = self.iaway_idx.get("vessel_idx")
        ray_idx = self.iaway_idx.get("ray_idx")
        vessel_content_idx = self.iaway_idx.get("vessel_content_idx") or []

        if vessel_idx is not None:
            loss = loss + IAWA_PRESENCE_PENALTY * F.l1_loss(
                pred_presence[:, vessel_idx],
                torch.ones(pred_presence.shape[0], device=device),
            )
        if ray_idx is not None:
            loss = loss + IAWA_PRESENCE_PENALTY * F.l1_loss(
                pred_presence[:, ray_idx],
                torch.ones(pred_presence.shape[0], device=device),
            )

        if vessel_idx is not None and vessel_content_idx:
            v_presence = pred_presence[:, vessel_idx]
            for c_idx in vessel_content_idx:
                c_presence = pred_presence[:, c_idx]
                loss = loss + IAWA_CONTAINMENT_PENALTY * F.relu(c_presence - v_presence).mean()

        return loss


def make_sem_loss_callbacks(class_names, species_priors, iaway_bias):
    state: Dict[str, object] = {"wrapper": None, "csv_path": None}

    def on_train_start(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None and hasattr(trainer.model, "init_criterion"):
            criterion = trainer.model.init_criterion()
            trainer.model.criterion = criterion
        if criterion is None:
            print("[sem_loss] WARNING: model has no criterion at on_train_start. Sem_loss disabled.")
            return

        if isinstance(criterion, SemLossWrapper):
            state["wrapper"] = criterion
            return

        nc = len(class_names)
        reg_max = getattr(criterion, "reg_max", 16)

        wrapper = SemLossWrapper(
            base_criterion=criterion,
            class_names=class_names,
            species_priors=species_priors,
            iaway_bias=iaway_bias,
            nc=nc,
            reg_max=reg_max,
        )
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
                if ema_criterion is not None and not isinstance(ema_criterion, SemLossWrapper):
                    ema_wrapper = SemLossWrapper(
                        base_criterion=ema_criterion,
                        class_names=class_names,
                        species_priors=species_priors,
                        iaway_bias=iaway_bias,
                        nc=nc,
                        reg_max=getattr(ema_criterion, "reg_max", reg_max),
                    )
                    ema_model.criterion = ema_wrapper
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not wrap EMA criterion: {exc}")

        try:
            save_dir = Path(getattr(trainer, "save_dir", "."))
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "sem_loss_log.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,batch,lambda,sem_loss,per_image,per_species,iaway\n")
            state["csv_path"] = csv_path
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not create sem_loss_log.csv: {exc}")

        print(f"[sem_loss] Wrapped criterion with sem_loss (iaway_bias={iaway_bias}).")

    def on_train_epoch_start(trainer):
        wrapper = state["wrapper"]
        if isinstance(wrapper, SemLossWrapper):
            wrapper.set_epoch(int(trainer.epoch))

    def on_train_batch_end(trainer):
        wrapper = state["wrapper"]
        csv_path = state["csv_path"]
        if not isinstance(wrapper, SemLossWrapper) or csv_path is None:
            return
        try:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{int(getattr(trainer, 'epoch', -1))},"
                    f"{int(getattr(trainer, '_i', -1))},"
                    f"{wrapper._last_lambda_value:.6f},"
                    f"{wrapper._last_sem_loss_value:.6f},"
                    f"{wrapper._last_per_image_value:.6f},"
                    f"{wrapper._last_per_species_value:.6f},"
                    f"{wrapper._last_iaway_value:.6f}\n"
                )
        except Exception:
            pass

    return on_train_start, on_train_epoch_start, on_train_batch_end


def run_training_experiment(experiment, dataset_yaml_path, class_names, by_split, species_priors):
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]
    iaway_bias = bool(experiment.get("iaway_bias", False))

    YOLO = get_yolo_class()
    model = YOLO(BASE_MODEL_WEIGHTS)

    on_train_start_cb, on_train_epoch_start_cb, on_train_batch_end_cb = make_sem_loss_callbacks(
        class_names=class_names,
        species_priors=species_priors,
        iaway_bias=iaway_bias,
    )
    model.add_callback("on_train_start", on_train_start_cb)
    model.add_callback("on_train_epoch_start", on_train_epoch_start_cb)
    model.add_callback("on_train_batch_end", on_train_batch_end_cb)

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
        plots=True,
        save=True,
        verbose=True,
    )

    experiment_dir = Path(train_results.save_dir)
    best_model_path = experiment_dir / "weights" / "best.pt"

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
        verbose=True,
    )

    try:
        built_in_confusion_df = val_results.confusion_matrix.to_df()
        built_in_confusion_df.to_csv(experiment_dir / "ultralytics_confusion_matrix.csv", index=False)
    except Exception:
        pass

    custom_global_metrics = evaluate_on_test_split(
        best_model_path, experiment_dir, imgsz, class_names, by_split["test"]
    )

    summary = {
        "experiment": run_name,
        "batch": batch_size,
        "imgsz": imgsz,
        "iaway_bias": iaway_bias,
        "best_model_path": str(best_model_path),
        "ultralytics_box_map50": float(val_results.box.map50),
        "ultralytics_box_map50_95": float(val_results.box.map),
        "ultralytics_seg_map50": float(val_results.seg.map50),
        "ultralytics_seg_map50_95": float(val_results.seg.map),
        "custom_micro_precision": float(custom_global_metrics["micro_precision"]),
        "custom_micro_recall": float(custom_global_metrics["micro_recall"]),
        "custom_micro_f1": float(custom_global_metrics["micro_f1"]),
        "custom_macro_precision": float(custom_global_metrics["macro_precision"]),
        "custom_macro_recall": float(custom_global_metrics["macro_recall"]),
        "custom_macro_f1": float(custom_global_metrics["macro_f1"]),
        "custom_total_gt_instances": int(custom_global_metrics["total_gt_instances"]),
        "custom_total_pred_instances": int(custom_global_metrics["total_pred_instances"]),
    }

    with open(experiment_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML_PATH}")

    class_names = load_class_names(DATA_YAML_PATH)

    records = collect_records_from_split_csvs(DATASET_ROOT)
    if not records:
        records = collect_records_from_yaml_txts(DATASET_ROOT, DATA_YAML_PATH)
    if not records:
        raise RuntimeError("No split records found. Check split CSVs or the txt files referenced in data.yaml.")

    dataset_yaml_path, by_split = prepare_dataset_yaml_and_txts(records, class_names)
    species_priors = load_species_priors(RELATIVE_VECTORS_CSV, class_names)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_training_experiment(
            experiment, dataset_yaml_path, class_names, by_split, species_priors
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("Finished. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
