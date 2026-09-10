"""YOLO11 segmentation training with a species-aware semantic loss + diagnostic
(distinctive-morphology) emphasis.

This script extends train_yolo_seg_sem_loss_species_overlap.py. It keeps every
prior feature:

  - per-image (count KL + area L1) and per-species (presence/area prior) terms,
  - species-level pixel-exclusivity penalty whose allowed-overlap matrix is
    DISCOVERED from the GT during the warmup+ramp window (not assumed),
  - AMP-safe sem_loss (float32 block), dict/tuple-aware pred parsing,
    EMA-criterion wrapping, retina_masks=True at val, per-batch CSV logging.

NEW — diagnostic emphasis (flag `diag_emphasis`):
  Some (species, feature) pairs are visually obvious; others have an UNUSUAL
  morphology for that species and deserve extra attention so the model does not
  default to the more common features. We quantify "unusual morphology" as the
  distinctiveness of a feature's AREA in a species relative to that feature's
  area distribution across all species (a z-score). The resulting per-(species,
  feature) weight W >= 1 (1 = ordinary, up to 1+DIAG_BOOST = very distinctive)
  scales an ADDITIVE emphasis term that pushes presence/area of those distinctive
  combos toward their species prior. With DIAG_EMPHASIS_WEIGHT = 0 (or
  diag_emphasis=False) the script reduces exactly to the previous behaviour.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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

# ============================================================
# OUTPUT LOCATION
# ============================================================
OUTPUT_ROOT = Path("runs_wood_feature_sem_loss_diag") / SPLIT_FOLDER.name

# ============================================================
# SEM_LOSS CONFIGURATION
# ============================================================
RELATIVE_VECTORS_CSV = Path("outputs/relative_feature_vectors.csv")

LAMBDA_SEM_MAX = 0.10
WARMUP_END_EPOCH = 30
RAMP_END_EPOCH = 80

PER_IMAGE_WEIGHT = 0.7
PER_SPECIES_WEIGHT = 0.3
COUNT_VS_AREA = 0.5

# Species-level pixel-exclusivity term (GT-discovered allowed overlaps).
EXCLUSIVITY_PENALTY = 0.10
OVERLAP_GRID_RES = 128
OVERLAP_MIN_OBSERVATIONS = 3   # higher than 1 to filter spurious augmentation overlaps

# Diagnostic / distinctive-morphology emphasis.
DIAG_EMPHASIS_WEIGHT = 0.15    # weight of the emphasis term inside sem_loss
DIAG_BOOST = 2.0               # max extra multiplier for the most distinctive (species, feat)
DIAG_Z_CLAMP = 3.0             # z-score saturation point
DIAG_PRESENCE_GATE = 0.10      # only boost features actually present in the species
DIAG_USE_RARITY = False        # multiply boost by the feature's global rarity (idf)

OVERLAP_CACHE_NAME = "species_overlap_priors.json"
DIAG_CACHE_NAME = "diagnostic_weights.json"

# ============================================================
# EXPERIMENTS (factorial: exclusivity x diagnostic)
# ============================================================
EXPERIMENTS = [
    {"name": "sem_base_bs8_img1024",        "batch": 8, "imgsz": 1024, "iaway_bias": False, "diag_emphasis": False},
    {"name": "sem_spexcl_bs8_img1024",      "batch": 8, "imgsz": 1024, "iaway_bias": True,  "diag_emphasis": False},
    {"name": "sem_diag_bs8_img1024",        "batch": 8, "imgsz": 1024, "iaway_bias": False, "diag_emphasis": True},
    {"name": "sem_spexcl_diag_bs8_img1024", "batch": 8, "imgsz": 1024, "iaway_bias": True,  "diag_emphasis": True},
]


# ============================================================
# UTILITIES
# ============================================================
def parse_species_id_from_filename(file_name: str) -> str:
    stem = Path(str(file_name).replace("\\", "/").split("/")[-1]).stem
    token = re.split(r"[-_]", stem, maxsplit=1)[0].strip()
    return token or "unknown"


def safe_feature_name(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not text:
        text = "unknown"
    return f"feat_{text}"


# ============================================================
# SPECIES PRIORS (presence / area)
# ============================================================
def load_species_priors(
    csv_path: Path, class_names: List[str]
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
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
        print(f"[sem_loss] WARNING: {csv_path} has no presence/area columns. Per-species prior disabled.")
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

    print(f"[sem_loss] Loaded species priors for {len(priors)} species "
          f"(presence={use_presence}, area={use_area}).")
    return priors


# ============================================================
# DIAGNOSTIC (DISTINCTIVE-MORPHOLOGY) WEIGHTS
# ============================================================
def load_diagnostic_weights(
    csv_path: Path, class_names: List[str]
) -> Optional[Dict[str, np.ndarray]]:
    """Per-(species, feature) weight >= 1 reflecting how distinctive the
    feature's morphology (area) is for that species.

    For each feature f, the distribution of per-species mean areas is built over
    the species where f is present. A species whose mean area for f deviates
    strongly from that distribution (high |z|) gets a boost; ordinary species
    get weight 1. Returns {species_id: np.ndarray(nc,)} or None.
    """
    if not csv_path.exists():
        print(f"[sem_loss] WARNING: {csv_path} not found. Diagnostic emphasis disabled.")
        return None

    df = pd.read_csv(csv_path)
    feature_columns = [safe_feature_name(name) for name in class_names]
    area_columns = [f"area_{c}" for c in feature_columns]
    presence_columns = [f"presence_{c}" for c in feature_columns]

    if "species_id" not in df.columns or not all(c in df.columns for c in area_columns):
        print(f"[sem_loss] WARNING: {csv_path} lacks species_id/area columns. Diagnostic emphasis disabled.")
        return None

    nc = len(class_names)
    sp_area = df.groupby("species_id")[area_columns].mean()
    species_ids = [str(s) for s in sp_area.index.tolist()]
    area = sp_area.to_numpy(dtype=np.float64)  # (S, nc)

    if all(c in df.columns for c in presence_columns):
        pres = df.groupby("species_id")[presence_columns].mean().to_numpy(dtype=np.float64)
    else:
        pres = (area > 0).astype(np.float64)

    # Per-feature global stats over species where the feature is present.
    glob_mean = np.zeros(nc)
    glob_std = np.ones(nc)
    for f in range(nc):
        vals = area[pres[:, f] >= DIAG_PRESENCE_GATE, f]
        if vals.size >= 2:
            glob_mean[f] = float(vals.mean())
            glob_std[f] = float(vals.std()) + 1e-8
        elif vals.size == 1:
            glob_mean[f] = float(vals[0])
            glob_std[f] = 1e-8

    idf = np.ones(nc)
    if DIAG_USE_RARITY:
        S = area.shape[0]
        present_counts = (pres >= DIAG_PRESENCE_GATE).sum(axis=0).astype(np.float64)
        idf = np.log((S + 1.0) / (present_counts + 1.0)) + 1.0
        idf = idf / max(idf.max(), 1e-8)

    weights: Dict[str, np.ndarray] = {}
    for si, sp in enumerate(species_ids):
        w = np.ones(nc, dtype=np.float32)
        for f in range(nc):
            if pres[si, f] < DIAG_PRESENCE_GATE:
                continue
            z = abs(area[si, f] - glob_mean[f]) / glob_std[f]
            z = min(z, DIAG_Z_CLAMP) / DIAG_Z_CLAMP        # -> [0, 1]
            boost = DIAG_BOOST * z
            if DIAG_USE_RARITY:
                boost *= float(idf[f])
            w[f] = 1.0 + boost
        weights[sp] = w

    n_boosted = sum(int((w > 1.0).any()) for w in weights.values())
    print(f"[sem_loss] Loaded diagnostic weights for {len(weights)} species "
          f"({n_boosted} with at least one boosted feature).")
    return weights


def save_diagnostic_weights(weights: Dict[str, np.ndarray], class_names: List[str], path: Path) -> None:
    payload = {
        "class_names": class_names,
        "boost": DIAG_BOOST,
        "z_clamp": DIAG_Z_CLAMP,
        "presence_gate": DIAG_PRESENCE_GATE,
        "use_rarity": DIAG_USE_RARITY,
        "species": {sp: w.tolist() for sp, w in weights.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# SPECIES-LEVEL ALLOWED-OVERLAP MASK (startup seed only)
# ============================================================
def rasterize_yolo_label(label_path: Path, nc: int, grid_res: int) -> np.ndarray:
    masks = np.zeros((nc, grid_res, grid_res), dtype=np.uint8)
    if not HAS_CV2 or not label_path.exists():
        return masks.astype(bool)
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return masks.astype(bool)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id < 0 or class_id >= nc:
            continue
        try:
            coords = np.array(parts[1:], dtype=np.float32)
        except ValueError:
            continue
        if coords.size < 6 or coords.size % 2 != 0:
            continue
        poly_px = (coords.reshape(-1, 2) * grid_res).astype(np.int32)
        poly_px = np.clip(poly_px, 0, grid_res - 1)
        layer = np.zeros((grid_res, grid_res), dtype=np.uint8)
        cv2.fillPoly(layer, [poly_px], 1)
        masks[class_id] = np.maximum(masks[class_id], layer)
    return masks.astype(bool)


def compute_species_overlap_priors(
    train_records: List[dict],
    class_names: List[str],
    grid_res: int = OVERLAP_GRID_RES,
    min_observations: int = OVERLAP_MIN_OBSERVATIONS,
) -> Dict[str, np.ndarray]:
    nc = len(class_names)
    if not HAS_CV2:
        print("[sem_loss] WARNING: cv2 not available; overlap seed disabled.")
        return {}
    counts: Dict[str, np.ndarray] = {}
    for rec in train_records:
        image_field = rec.get("image_path") or rec.get("image_name") or ""
        species_id = str(parse_species_id_from_filename(image_field))
        label_path = rec.get("label_path")
        if not label_path:
            continue
        layers = rasterize_yolo_label(Path(label_path), nc, grid_res)
        if not layers.any():
            continue
        layers_i = layers.astype(np.int8)
        pair = (np.einsum("cij,dij->cd", layers_i, layers_i) > 0)
        counts.setdefault(species_id, np.zeros((nc, nc), dtype=np.int32))
        counts[species_id] += pair.astype(np.int32)

    allowed: Dict[str, np.ndarray] = {}
    for sp, cnt in counts.items():
        m = (cnt >= min_observations).astype(np.int8)
        np.fill_diagonal(m, 1)
        m = ((m | m.T) > 0).astype(np.int8)
        allowed[sp] = m
    print(f"[sem_loss] Built overlap SEED for {len(allowed)} species (for contrast only).")
    return allowed


# ============================================================
# SEM_LOSS WRAPPER
# ============================================================
class SemLossWrapper(torch.nn.Module):
    def __init__(
        self,
        base_criterion,
        class_names: List[str],
        species_priors: Optional[Dict[str, Dict[str, np.ndarray]]],
        species_overlap_priors: Optional[Dict[str, np.ndarray]],
        diagnostic_weights: Optional[Dict[str, np.ndarray]],
        iaway_bias: bool,
        diag_emphasis: bool,
        nc: int,
        reg_max: int,
    ) -> None:
        super().__init__()
        self.base = base_criterion
        self.class_names = class_names
        self.nc = nc
        self.reg_max = reg_max
        self.iaway_bias = iaway_bias
        self.diag_emphasis = diag_emphasis
        self.species_priors = species_priors or {}
        self.diagnostic_weights = diagnostic_weights or {}
        self._epoch = 0
        self._max_epochs = EPOCHS

        self._last_sem_loss_value = 0.0
        self._last_lambda_value = 0.0
        self._last_exclusivity_value = 0.0
        self._last_diag_value = 0.0
        self._last_failure_epoch = -1

        # GT-discovered allowed-overlap counts (TRAIN wrapper only).
        self._overlap_counts: Dict[str, np.ndarray] = {}
        self._accumulate_overlaps = True
        self._seed_overlap_priors = species_overlap_priors or {}

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

        if (self.iaway_bias and self._accumulate_overlaps and self._epoch < RAMP_END_EPOCH):
            try:
                with torch.no_grad():
                    self._accumulate_gt_overlaps(batch)
            except Exception:
                pass

        lam = self.lambda_sem()
        sem_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        if lam > 0.0:
            try:
                sem_loss = self._compute_sem_loss(preds, batch).to(base_loss.dtype)
            except Exception as exc:
                if self._last_failure_epoch != self._epoch:
                    self._last_failure_epoch = self._epoch
                    print(f"[sem_loss] WARNING (epoch {self._epoch}): "
                          f"sem_loss failed: {type(exc).__name__}: {exc}")
                    traceback.print_exc()

        total_loss = base_loss + lam * sem_loss
        self._last_sem_loss_value = float(sem_loss.detach().item())
        self._last_lambda_value = float(lam)
        return total_loss, base_items

    # ------------------------------------------------------------------
    # Pred parsing
    # ------------------------------------------------------------------
    def _parse_preds(self, preds) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(preds, dict):
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
            proto = preds["proto"]
            if isinstance(proto, tuple) and len(proto) == 2:
                proto = proto[0]
        else:
            feats, pred_masks_raw, proto = preds if len(preds) == 3 else preds[1]
            no = self.reg_max * 4 + self.nc
            feat_concat = torch.cat(
                [xi.view(feats[0].shape[0], no, -1) for xi in feats], dim=2
            )
            _pred_distri, pred_scores = feat_concat.split((self.reg_max * 4, self.nc), dim=1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            pred_masks = pred_masks_raw.permute(0, 2, 1).contiguous()
        return pred_scores, pred_masks, proto

    def _gt_count_and_area(
        self, batch: dict, batch_size: int, device: torch.device,
        mask_h: int, mask_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def _resize_masks(masks: torch.Tensor, mask_h: int, mask_w: int) -> torch.Tensor:
        if masks.dim() == 3 and masks.shape[-2:] == (mask_h, mask_w):
            return masks
        if masks.dim() == 4 and masks.shape[-2:] == (mask_h, mask_w):
            return masks
        if masks.dim() == 3:
            return F.interpolate(masks.unsqueeze(1).float(), size=(mask_h, mask_w), mode="nearest").squeeze(1)
        if masks.dim() == 4:
            return F.interpolate(masks.float(), size=(mask_h, mask_w), mode="nearest")
        raise ValueError(f"Unexpected masks shape: {masks.shape}")

    # ------------------------------------------------------------------
    # GT-driven overlap discovery
    # ------------------------------------------------------------------
    def _accumulate_gt_overlaps(self, batch: dict, grid: int = OVERLAP_GRID_RES) -> None:
        cls_t = batch.get("cls")
        bidx_t = batch.get("batch_idx")
        masks_t = batch.get("masks")
        im_files = batch.get("im_file") or []
        if (cls_t is None or bidx_t is None or masks_t is None
                or masks_t.numel() == 0 or not im_files):
            return

        cls_long = cls_t.long().flatten()
        bidx_long = bidx_t.long().flatten()
        nc = self.nc
        masks_resized = self._resize_masks(masks_t, grid, grid)

        for b in range(len(im_files)):
            targets_in_b = (bidx_long == b).nonzero(as_tuple=True)[0]
            if targets_in_b.numel() == 0:
                continue
            per_class = torch.zeros(nc, grid, grid, dtype=torch.bool, device=masks_resized.device)
            if masks_t.dim() == 3:
                image_mask = masks_resized[b].long()
                for local_idx, t_idx in enumerate(targets_in_b):
                    c = int(cls_long[t_idx].item())
                    if 0 <= c < nc:
                        per_class[c] |= (image_mask == (local_idx + 1))
            else:
                for local_idx, t_idx in enumerate(targets_in_b):
                    c = int(cls_long[t_idx].item())
                    if 0 <= c < nc and local_idx < masks_resized.shape[1]:
                        per_class[c] |= (masks_resized[b, local_idx] > 0.5)
            flat = per_class.view(nc, -1).float()
            pair = (flat @ flat.t() > 0).cpu().numpy()
            sp = parse_species_id_from_filename(im_files[b])
            acc = self._overlap_counts.setdefault(sp, np.zeros((nc, nc), dtype=np.int32))
            acc += pair.astype(np.int32)

    def _current_allowed(self, species_id: str) -> Optional[np.ndarray]:
        counts = self._overlap_counts.get(species_id)
        if counts is None:
            return None
        m = (counts >= OVERLAP_MIN_OBSERVATIONS).astype(np.int8)
        np.fill_diagonal(m, 1)
        m = ((m | m.T) > 0).astype(np.int8)
        return m

    def dump_learned_overlaps(self, path) -> None:
        out = {"min_observations": OVERLAP_MIN_OBSERVATIONS, "species": {}}
        for sp, counts in self._overlap_counts.items():
            allowed = (counts >= OVERLAP_MIN_OBSERVATIONS).astype(int)
            np.fill_diagonal(allowed, 1)
            entry = {"allowed": allowed.tolist(), "counts": counts.tolist()}
            seed = self._seed_overlap_priors.get(sp)
            if seed is not None:
                seed = np.asarray(seed)
                if seed.shape == allowed.shape:
                    entry["unanticipated_vs_seed"] = (
                        ((allowed == 1) & (seed == 0)).astype(int).tolist()
                    )
            out["species"][sp] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    def dump_overlap_counts(self, path) -> None:
        """Persist the RAW running overlap counts so a spot resume can restore
        the discovery state at any point of the warmup+ramp window."""
        payload = {"counts": {sp: c.tolist() for sp, c in self._overlap_counts.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_overlap_counts(self, path) -> None:
        """Restore running overlap counts dumped by dump_overlap_counts()."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._overlap_counts = {
                str(sp): np.asarray(c, dtype=np.int32)
                for sp, c in payload.get("counts", {}).items()
            }
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not load overlap counts: {exc}")

    # ------------------------------------------------------------------
    # Sem_loss
    # ------------------------------------------------------------------
    def _compute_sem_loss(self, preds, batch) -> torch.Tensor:
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

            if self.iaway_bias:
                exclusivity_loss = self._species_exclusivity_loss(expected_mask_logit, batch)
            else:
                exclusivity_loss = torch.zeros((), device=device)

            if self.diag_emphasis:
                diag_loss = self._diagnostic_emphasis_loss(batch, pred_count, pred_area, device)
            else:
                diag_loss = torch.zeros((), device=device)

            sem = (
                PER_IMAGE_WEIGHT * per_image_loss
                + PER_SPECIES_WEIGHT * per_species_loss
                + EXCLUSIVITY_PENALTY * exclusivity_loss
                + DIAG_EMPHASIS_WEIGHT * diag_loss
            )
            self._last_exclusivity_value = (
                float(exclusivity_loss.detach().item()) if torch.is_tensor(exclusivity_loss) else float(exclusivity_loss)
            )
            self._last_diag_value = (
                float(diag_loss.detach().item()) if torch.is_tensor(diag_loss) else float(diag_loss)
            )
            return sem

    # ------------------------------------------------------------------
    # Term helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_distribution(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = t.sum(dim=-1, keepdim=True)
        safe = torch.where(s > 0, s, torch.ones_like(s))
        out = t / safe
        out = (out + eps)
        out = out / out.sum(dim=-1, keepdim=True)
        return out

    @classmethod
    def _kl(cls, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        target_dist = cls._to_distribution(target)
        pred_dist = cls._to_distribution(pred)
        return (target_dist * (target_dist.log() - pred_dist.log())).sum(dim=-1).mean()

    def _per_image_distribution_loss(
        self, pred_count, gt_count, pred_area, gt_area
    ) -> torch.Tensor:
        kl_count = self._kl(gt_count, pred_count)
        l1_area = F.l1_loss(pred_area, gt_area)
        return kl_count + COUNT_VS_AREA * l1_area

    def _per_species_prior_loss(self, batch, pred_count, pred_area, device) -> torch.Tensor:
        if not self.species_priors:
            return torch.zeros((), device=device)
        im_files = batch.get("im_file") or []
        if not im_files:
            return torch.zeros((), device=device)

        species_ids = [parse_species_id_from_filename(f) for f in im_files]
        pred_presence = torch.tanh(pred_count)
        loss = torch.zeros((), device=device)
        n_terms = 0
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

    def _diagnostic_emphasis_loss(self, batch, pred_count, pred_area, device) -> torch.Tensor:
        """Additive term that up-weights distinctive (species, feature) pairs.

        Uses ONLY the boost part (W - 1) of the diagnostic weights, so it is a
        pure add-on to per_species_loss: it pushes presence/area of the
        distinctive features toward their species prior, weighted by how
        unusual their morphology is. Ordinary features (W == 1) contribute 0.
        """
        if not self.diagnostic_weights or not self.species_priors:
            return torch.zeros((), device=device)
        im_files = batch.get("im_file") or []
        if not im_files:
            return torch.zeros((), device=device)

        species_ids = [parse_species_id_from_filename(f) for f in im_files]
        pred_presence = torch.tanh(pred_count)
        loss = torch.zeros((), device=device)
        n_terms = 0
        for b, sp in enumerate(species_ids):
            w = self.diagnostic_weights.get(sp)
            prior = self.species_priors.get(sp)
            if w is None or prior is None:
                continue
            extra = np.clip(w - 1.0, 0.0, None)
            if float(extra.sum()) <= 0.0:
                continue
            extra_t = torch.tensor(extra, device=device, dtype=pred_presence.dtype)
            term = torch.zeros((), device=device)
            presence = prior.get("presence")
            if presence is not None:
                tgt = torch.tensor(presence, device=device, dtype=pred_presence.dtype)
                term = term + (extra_t * (pred_presence[b] - tgt).abs()).sum()
            area = prior.get("area")
            if area is not None:
                tgt = torch.tensor(area, device=device, dtype=pred_area.dtype)
                term = term + COUNT_VS_AREA * (extra_t * (pred_area[b] - tgt).abs()).sum()
            loss = loss + term / extra_t.sum().clamp_min(1.0)
            n_terms += 1
        if n_terms == 0:
            return torch.zeros((), device=device)
        return loss / float(n_terms)

    def _species_exclusivity_loss(self, expected_mask_logit, batch) -> torch.Tensor:
        device = expected_mask_logit.device
        B, nc, P = expected_mask_logit.shape
        im_files = batch.get("im_file") or []
        if not im_files:
            return torch.zeros((), device=device)

        probs = expected_mask_logit.sigmoid()
        diag_mask = (1.0 - torch.eye(nc, device=device, dtype=probs.dtype))

        forbidden = torch.zeros(B, nc, nc, device=device, dtype=probs.dtype)
        any_seen = False
        for b, fpath in enumerate(im_files):
            species_id = parse_species_id_from_filename(fpath)
            allowed = self._current_allowed(species_id)
            if allowed is None:
                continue
            allowed_t = torch.tensor(allowed, device=device, dtype=probs.dtype)
            forb = (1.0 - allowed_t) * diag_mask
            forb = torch.triu(forb, diagonal=1)
            forbidden[b] = forb
            any_seen = True
        if not any_seen:
            return torch.zeros((), device=device)

        gram = torch.bmm(probs, probs.transpose(1, 2))
        overlap = (forbidden * gram).sum(dim=(1, 2)) / float(P)
        forb_count = forbidden.sum(dim=(1, 2)).clamp_min(1.0)
        overlap = overlap / forb_count
        return overlap.mean()


# ============================================================
# CALLBACK REGISTRATION
# ============================================================
def make_sem_loss_callbacks(
    class_names: List[str],
    species_priors,
    species_overlap_priors,
    diagnostic_weights,
    iaway_bias: bool,
    diag_emphasis: bool,
):
    state: Dict[str, object] = {"wrapper": None, "csv_path": None}

    def _build_wrapper(criterion, reg_max):
        return SemLossWrapper(
            base_criterion=criterion,
            class_names=class_names,
            species_priors=species_priors,
            species_overlap_priors=species_overlap_priors,
            diagnostic_weights=diagnostic_weights,
            iaway_bias=iaway_bias,
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
            print("[sem_loss] WARNING: no criterion at on_train_start. Sem_loss disabled.")
            return
        if isinstance(criterion, SemLossWrapper):
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
                if ema_criterion is not None and not isinstance(ema_criterion, SemLossWrapper):
                    ema_wrapper = _build_wrapper(ema_criterion, getattr(ema_criterion, "reg_max", reg_max))
                    ema_wrapper._accumulate_overlaps = False  # never learn the rule from val GT
                    ema_model.criterion = ema_wrapper
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not wrap EMA criterion: {exc}")

        try:
            save_dir = Path(getattr(trainer, "save_dir", "."))
            save_dir.mkdir(parents=True, exist_ok=True)

            # Restore the GT-discovered overlap counts on resume so the
            # exclusivity rule survives a spot interruption. Without this, a
            # resume past RAMP_END_EPOCH would start with an empty matrix and
            # the penalty would be silently disabled.
            running_overlap_path = save_dir / "overlap_counts_running.json"
            state["running_overlap_path"] = running_overlap_path
            if iaway_bias and running_overlap_path.exists():
                wrapper.load_overlap_counts(running_overlap_path)
                print(f"[sem_loss] Restored overlap counts for "
                      f"{len(wrapper._overlap_counts)} species (resume).")

            # Append to sem_loss_log.csv if it already exists (resume); else
            # create it with a header.
            csv_path = save_dir / "sem_loss_log.csv"
            if not csv_path.exists():
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("epoch,batch,lambda,sem_loss,exclusivity,diagnostic\n")
            state["csv_path"] = csv_path
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not set up sem_loss logging/restore: {exc}")

        print(f"[sem_loss] Wrapped criterion (iaway_bias={iaway_bias}, "
              f"diag_emphasis={diag_emphasis}).")

    def on_train_epoch_start(trainer):
        wrapper = state["wrapper"]
        if isinstance(wrapper, SemLossWrapper):
            wrapper.set_epoch(int(trainer.epoch))
            # Persist the running overlap counts every epoch while we are still
            # discovering, so a resume at ANY epoch can restore them.
            if wrapper._accumulate_overlaps and int(trainer.epoch) < RAMP_END_EPOCH:
                rp = state.get("running_overlap_path")
                if rp is not None:
                    try:
                        wrapper.dump_overlap_counts(rp)
                    except Exception:
                        pass
            if int(trainer.epoch) == RAMP_END_EPOCH:
                try:
                    save_dir = Path(getattr(trainer, "save_dir", "."))
                    wrapper.dump_learned_overlaps(save_dir / "learned_overlap_priors.json")
                    wrapper.dump_overlap_counts(save_dir / "overlap_counts_running.json")
                    print(f"[sem_loss] Dumped GT-discovered overlap matrix at epoch {RAMP_END_EPOCH}.")
                except Exception as exc:
                    print(f"[sem_loss] WARNING: could not dump learned overlaps: {exc}")

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
                    f"{wrapper._last_exclusivity_value:.6f},"
                    f"{wrapper._last_diag_value:.6f}\n"
                )
        except Exception:
            pass

    return on_train_start, on_train_epoch_start, on_train_batch_end


# ============================================================
# TRAINING
# ============================================================
def run_training_experiment(
    experiment, dataset_yaml_path, class_names, by_split,
    species_priors, species_overlap_priors, diagnostic_weights,
) -> dict:
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]
    iaway_bias = bool(experiment.get("iaway_bias", False))
    diag_emphasis = bool(experiment.get("diag_emphasis", False))

    expected_dir = OUTPUT_ROOT / run_name
    summary_path = expected_dir / "experiment_summary.json"

    # 1) Idempotencia entre experimentos: si este ya terminó del todo (existe el
    #    summary final, que se escribe atómicamente al final de la función),
    #    saltarlo y devolver el resumen guardado. Tras una interrupción spot
    #    esto evita reentrenar los experimentos ya completados.
    if summary_path.exists():
        print(f"[skip] {run_name}: experiment_summary.json already present.")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    YOLO = get_yolo_class()
    last_path = expected_dir / "weights" / "last.pt"
    resuming = last_path.exists()

    # 2) Reanudación: si Ultralytics dejó un last.pt para este experimento,
    #    cargarlo y pasar resume=True. Ultralytics restaura época, optimizador
    #    y scheduler; el wrapper se reconstruye en on_train_start y lambda_sem()
    #    se recalcula a partir de trainer.epoch (no se reinicia el warmup).
    model = YOLO(str(last_path)) if resuming else YOLO(BASE_MODEL_WEIGHTS)
    if resuming:
        print(f"[resume] {run_name}: resuming from {last_path}")

    cbs = make_sem_loss_callbacks(
        class_names=class_names,
        species_priors=species_priors,
        species_overlap_priors=species_overlap_priors,
        diagnostic_weights=diagnostic_weights,
        iaway_bias=iaway_bias,
        diag_emphasis=diag_emphasis,
    )
    on_train_start_cb, on_train_epoch_start_cb, on_train_batch_end_cb = cbs
    model.add_callback("on_train_start", on_train_start_cb)
    model.add_callback("on_train_epoch_start", on_train_epoch_start_cb)
    model.add_callback("on_train_batch_end", on_train_batch_end_cb)

    try:
        if resuming:
            # En resume Ultralytics lee los kwargs del checkpoint; no se vuelven
            # a pasar para garantizar continuidad exacta.
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
                plots=True,
                save=True,
                verbose=True,
            )
        experiment_dir = Path(train_results.save_dir)
    except Exception as exc:
        # Caso típico: resume=True sobre un checkpoint cuya última época ya era
        # la final -> Ultralytics puede lanzar. Continuamos a la evaluación
        # apuntando al directorio que ya conocemos.
        print(f"[resume] {run_name}: train(resume=True) raised "
              f"{type(exc).__name__}: {exc}. Assuming training is complete.")
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
        "diag_emphasis": diag_emphasis,
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

    tmp_path = experiment_dir / "experiment_summary.json.tmp"
    final_path = experiment_dir / "experiment_summary.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp_path.replace(final_path)

    return summary


# ============================================================
# MAIN
# ============================================================
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

    species_overlap_priors = compute_species_overlap_priors(
        by_split["train"], class_names,
        grid_res=OVERLAP_GRID_RES,
        min_observations=OVERLAP_MIN_OBSERVATIONS,
    )
    save_overlap_priors_path = OUTPUT_ROOT / OVERLAP_CACHE_NAME
    with open(save_overlap_priors_path, "w", encoding="utf-8") as f:
        json.dump(
            {"species": {sp: m.tolist() for sp, m in species_overlap_priors.items()}},
            f, indent=2, ensure_ascii=False,
        )
    if diagnostic_weights:
        save_diagnostic_weights(diagnostic_weights, class_names, OUTPUT_ROOT / DIAG_CACHE_NAME)

    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_training_experiment(
            experiment, dataset_yaml_path, class_names, by_split,
            species_priors, species_overlap_priors, diagnostic_weights,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("Finished. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
