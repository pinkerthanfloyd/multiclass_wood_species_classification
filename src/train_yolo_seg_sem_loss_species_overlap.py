"""Training script for YOLO11 segmentation with a species-aware semantic loss.

Changes vs the previous train_yolo_seg_sem_loss.py:

1. REMOVED the "presence(V1) ≈ 1" and "presence(radio) ≈ 1" hard rules. Real
   annotations are not exhaustive in this dataset — only categorisation-
   relevant features were labelled — so forcing universal presence was
   incorrect. Per-species presence/absence is now handled exclusively by the
   per_species_loss term, which is data-driven via the priors CSV.

2. REPLACED the hardcoded "56/58 contained in V1" containment penalty by a
   per-species pixel-level exclusivity penalty. For each species, an (nc, nc)
   binary mask is built from the actual training annotations indicating which
   pairs of classes have ever been observed to overlap in any image of that
   species. Pairs that have NEVER been observed to overlap incur a penalty
   when the model co-activates them at the same pixel; pairs that DO co-occur
   in the data (V1↔56, V1↔58, 56↔58, …) are tolerated.

3. The `iaway_bias` flag is repurposed: now it switches ON the species-level
   exclusivity rule.

4. `model.val(...)` passes `retina_masks=True` so the test-time confusion
   matrix and overlays use full-resolution masks.

5. `_parse_preds` handles both Ultralytics 8.4.45+ dict-style preds and older
   tuple-style preds.

6. `_compute_sem_loss` runs under `autocast(enabled=False)` with explicit
   float32 casts, to avoid silent fp16/fp32 dtype errors under AMP.

7. The exclusivity term uses a memory-efficient bmm (Gram matrix between
   classes) instead of a 4-axis einsum, to avoid OOM with imgsz=1024.

8. Wrapper does NOT extend loss_items (preserves validator's fixed-size
   accumulator). sem_loss values are logged per-batch to sem_loss_log.csv.

Generation of the per-species overlap mask is automatic at startup from the
YOLO labels of the TRAIN split and cached as JSON next to the run directory.
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
OUTPUT_ROOT = Path("runs_wood_feature_sem_loss") / SPLIT_FOLDER.name

# ============================================================
# SEM_LOSS CONFIGURATION
# ============================================================
RELATIVE_VECTORS_CSV = Path("outputs/relative_feature_vectors.csv")

# Lambda schedule.
LAMBDA_SEM_MAX = 0.10
WARMUP_END_EPOCH = 30
RAMP_END_EPOCH = 80

# Internal weights inside sem_loss.
PER_IMAGE_WEIGHT = 0.7
PER_SPECIES_WEIGHT = 0.3
COUNT_VS_AREA = 0.5

# Species-level pixel-exclusivity term.
EXCLUSIVITY_PENALTY = 0.10
OVERLAP_GRID_RES = 128
OVERLAP_MIN_OBSERVATIONS = 1

OVERLAP_CACHE_NAME = "species_overlap_priors.json"

# ============================================================
# EXPERIMENTS
# ============================================================
EXPERIMENTS = [
    {"name": "sem_cooc_bs4_img1024",        "batch": 4, "imgsz": 1024, "iaway_bias": False},
    {"name": "sem_cooc_bs8_img1024",        "batch": 8, "imgsz": 1024, "iaway_bias": False},
    {"name": "sem_cooc_spexcl_bs4_img1024", "batch": 4, "imgsz": 1024, "iaway_bias": True},
    {"name": "sem_cooc_spexcl_bs8_img1024", "batch": 8, "imgsz": 1024, "iaway_bias": True},
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

    print(f"[sem_loss] Loaded species priors for {len(priors)} species "
          f"(presence={use_presence}, area={use_area}).")
    return priors


# ============================================================
# SPECIES-LEVEL ALLOWED-OVERLAP MASK
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
        print("[sem_loss] WARNING: cv2 not available; species overlap priors disabled.")
        return {}

    counts: Dict[str, np.ndarray] = {}
    n_skipped = 0
    for rec in train_records:
        image_field = rec.get("image_path") or rec.get("image_name") or ""
        species_id = str(parse_species_id_from_filename(image_field))
        label_path = rec.get("label_path")
        if not label_path:
            n_skipped += 1
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

    if n_skipped:
        print(f"[sem_loss] WARNING: {n_skipped} train records had no label_path.")
    print(f"[sem_loss] Built species overlap priors for {len(allowed)} species "
          f"(grid={grid_res}, min_obs={min_observations}).")
    return allowed


def save_overlap_priors(
    priors: Dict[str, np.ndarray], class_names: List[str], path: Path
) -> None:
    payload = {
        "class_names": class_names,
        "grid_res": OVERLAP_GRID_RES,
        "min_observations": OVERLAP_MIN_OBSERVATIONS,
        "species": {sp: mat.tolist() for sp, mat in priors.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
        iaway_bias: bool,
        nc: int,
        reg_max: int,
    ) -> None:
        super().__init__()
        self.base = base_criterion
        self.class_names = class_names
        self.nc = nc
        self.reg_max = reg_max
        self.iaway_bias = iaway_bias
        self.species_priors = species_priors or {}
        self.species_overlap_priors = species_overlap_priors or {}
        self._epoch = 0
        self._max_epochs = EPOCHS
        self._last_sem_loss_value = 0.0
        self._last_lambda_value = 0.0
        self._last_exclusivity_value = 0.0
        self._last_failure_epoch = -1
        # Allowed-overlap matrix is DISCOVERED from GT during the warmup+ramp
        # window instead of being assumed. self._overlap_counts[species] is an
        # (nc, nc) int32 counting in how many images of that species a class
        # pair was observed overlapping in the GT. Only the TRAIN wrapper
        # accumulates; the EMA/validation wrapper has _accumulate_overlaps=False
        # so val GT never leaks into the rule.
        self._overlap_counts: Dict[str, np.ndarray] = {}
        self._accumulate_overlaps = True
        # The startup precompute (if provided) is kept ONLY to contrast against
        # what GT reveals during ramp. It is never used to drive the penalty.
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

        # Discover allowed overlaps from GT during the warmup+ramp window. This
        # runs even while lambda==0 (warmup), so by the time the penalty turns
        # on (WARMUP_END_EPOCH) the matrix already reflects observed GT, and it
        # keeps refining until RAMP_END_EPOCH, after which it is frozen.
        if (
            self.iaway_bias
            and self._accumulate_overlaps
            and self._epoch < RAMP_END_EPOCH
        ):
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
                    print(
                        f"[sem_loss] WARNING (epoch {self._epoch}): "
                        f"sem_loss computation failed: {type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

        total_loss = base_loss + lam * sem_loss

        # IMPORTANT: do NOT extend base_items. The validator's accumulator has
        # a fixed size (5 components). sem_loss reaches backward via
        # total_loss; visibility is preserved via the per-batch CSV logger.
        self._last_sem_loss_value = float(sem_loss.detach().item())
        self._last_lambda_value = float(lam)
        return total_loss, base_items

    # ------------------------------------------------------------------
    # Pred parsing — handles both Ultralytics 8.4.45+ dict and older tuple.
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
    # GT-driven overlap discovery (replaces the static startup prior)
    # ------------------------------------------------------------------
    def _accumulate_gt_overlaps(self, batch: dict, grid: int = OVERLAP_GRID_RES) -> None:
        """Update the running per-species overlap counts from this batch's GT.

        For each image, builds a per-class boolean grid at `grid` resolution
        from the GT masks, computes which class pairs share at least one cell,
        and accumulates the result (per species) into self._overlap_counts.
        Runs under no_grad; the small Gram matmul is the only GPU work and the
        accumulator lives on CPU as numpy.
        """
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
            if masks_t.dim() == 3:  # overlap_mask=True: pixel value = 1-indexed local target
                image_mask = masks_resized[b].long()
                for local_idx, t_idx in enumerate(targets_in_b):
                    c = int(cls_long[t_idx].item())
                    if 0 <= c < nc:
                        per_class[c] |= (image_mask == (local_idx + 1))
            else:                   # overlap_mask=False: (B, T, h, w)
                for local_idx, t_idx in enumerate(targets_in_b):
                    c = int(cls_long[t_idx].item())
                    if 0 <= c < nc and local_idx < masks_resized.shape[1]:
                        per_class[c] |= (masks_resized[b, local_idx] > 0.5)
            flat = per_class.view(nc, -1).float()
            pair = (flat @ flat.t() > 0).cpu().numpy()      # (nc, nc) bool
            sp = parse_species_id_from_filename(im_files[b])
            acc = self._overlap_counts.setdefault(sp, np.zeros((nc, nc), dtype=np.int32))
            acc += pair.astype(np.int32)

    def _current_allowed(self, species_id: str) -> Optional[np.ndarray]:
        """Binary (nc, nc) allowed-overlap mask discovered so far for a species,
        or None if no GT has been observed yet for it (=> no penalty applied)."""
        counts = self._overlap_counts.get(species_id)
        if counts is None:
            return None
        m = (counts >= OVERLAP_MIN_OBSERVATIONS).astype(np.int8)
        np.fill_diagonal(m, 1)
        m = ((m | m.T) > 0).astype(np.int8)
        return m

    def dump_learned_overlaps(self, path) -> None:
        """Persist the GT-discovered matrices plus a contrast against the
        startup precompute (the pairs allowed by GT but missed by the seed =
        the 'unanticipated combinations')."""
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

    # ------------------------------------------------------------------
    # Sem_loss
    # ------------------------------------------------------------------
    def _compute_sem_loss(self, preds, batch) -> torch.Tensor:
        # Force float32 and disable autocast inside this block. Under AMP the
        # raw model outputs are fp16 and mixing them with fp32 GT tensors was
        # raising dtype errors that the outer try/except silently swallowed.
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

            weighted_coeffs = score_sig.transpose(1, 2) @ pred_masks      # (B, nc, M)
            proto_flat = proto.view(B, M, -1)                              # (B, M, P)
            expected_mask_logit = weighted_coeffs @ proto_flat             # (B, nc, P)
            pred_area = expected_mask_logit.sigmoid().sum(dim=2) / total_pixels

            gt_count, gt_area = self._gt_count_and_area(batch, B, device, mask_h, mask_w)

            per_image_loss = self._per_image_distribution_loss(pred_count, gt_count, pred_area, gt_area)
            per_species_loss = self._per_species_prior_loss(batch, pred_count, pred_area, device)

            # The exclusivity term is driven by the GT-discovered matrix
            # (self._overlap_counts). It returns 0 for any species not yet
            # observed, so it is safe to call from the first ramp batch on.
            if self.iaway_bias:
                exclusivity_loss = self._species_exclusivity_loss(expected_mask_logit, batch)
            else:
                exclusivity_loss = torch.zeros((), device=device)

            sem = (
                PER_IMAGE_WEIGHT * per_image_loss
                + PER_SPECIES_WEIGHT * per_species_loss
                + EXCLUSIVITY_PENALTY * exclusivity_loss
            )
            self._last_exclusivity_value = (
                float(exclusivity_loss.detach().item())
                if torch.is_tensor(exclusivity_loss) else float(exclusivity_loss)
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
        self, pred_count: torch.Tensor, gt_count: torch.Tensor,
        pred_area: torch.Tensor, gt_area: torch.Tensor,
    ) -> torch.Tensor:
        kl_count = self._kl(gt_count, pred_count)
        l1_area = F.l1_loss(pred_area, gt_area)
        return kl_count + COUNT_VS_AREA * l1_area

    def _per_species_prior_loss(
        self, batch: dict, pred_count: torch.Tensor,
        pred_area: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
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

    def _species_exclusivity_loss(
        self, expected_mask_logit: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        device = expected_mask_logit.device
        B, nc, P = expected_mask_logit.shape
        im_files = batch.get("im_file") or []
        if not im_files:
            return torch.zeros((), device=device)

        probs = expected_mask_logit.sigmoid()                          # (B, nc, P)
        diag_mask = (1.0 - torch.eye(nc, device=device, dtype=probs.dtype))

        forbidden = torch.zeros(B, nc, nc, device=device, dtype=probs.dtype)
        any_seen = False
        for b, fpath in enumerate(im_files):
            species_id = parse_species_id_from_filename(fpath)
            allowed = self._current_allowed(species_id)  # GT-discovered, not assumed
            if allowed is None:
                continue
            allowed_t = torch.tensor(allowed, device=device, dtype=probs.dtype)
            forb = (1.0 - allowed_t) * diag_mask
            forb = torch.triu(forb, diagonal=1)
            forbidden[b] = forb
            any_seen = True
        if not any_seen:
            return torch.zeros((), device=device)

        # Gram-matrix formulation: avoids the (B, nc, nc, P) intermediate that
        # the equivalent einsum("bcp,bdp,bcd->b", ...) would materialise.
        gram = torch.bmm(probs, probs.transpose(1, 2))                  # (B, nc, nc)
        overlap = (forbidden * gram).sum(dim=(1, 2)) / float(P)         # (B,)
        forb_count = forbidden.sum(dim=(1, 2)).clamp_min(1.0)
        overlap = overlap / forb_count
        return overlap.mean()


# ============================================================
# CALLBACK REGISTRATION
# ============================================================
def make_sem_loss_callbacks(
    class_names: List[str],
    species_priors: Optional[Dict[str, Dict[str, np.ndarray]]],
    species_overlap_priors: Optional[Dict[str, np.ndarray]],
    iaway_bias: bool,
):
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
            species_overlap_priors=species_overlap_priors,
            iaway_bias=iaway_bias,
            nc=nc,
            reg_max=reg_max,
        )
        trainer.model.criterion = wrapper
        state["wrapper"] = wrapper

        # Wrap the EMA criterion too, so trainer-side validations see the same
        # term composition. Both wrappers preserve loss_items shape.
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
                        species_overlap_priors=species_overlap_priors,
                        iaway_bias=iaway_bias,
                        nc=nc,
                        reg_max=getattr(ema_criterion, "reg_max", reg_max),
                    )
                    ema_wrapper._accumulate_overlaps = False  # never learn the rule from val GT
                    ema_model.criterion = ema_wrapper
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not wrap EMA criterion: {exc}")

        try:
            save_dir = Path(getattr(trainer, "save_dir", "."))
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "sem_loss_log.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,batch,lambda,sem_loss,exclusivity\n")
            state["csv_path"] = csv_path
        except Exception as exc:
            print(f"[sem_loss] WARNING: could not create sem_loss_log.csv: {exc}")

        n_overlap = len(species_overlap_priors or {})
        print(f"[sem_loss] Wrapped criterion (iaway_bias={iaway_bias}, "
              f"species_overlap_priors={n_overlap}).")

    def on_train_epoch_start(trainer):
        wrapper = state["wrapper"]
        if isinstance(wrapper, SemLossWrapper):
            wrapper.set_epoch(int(trainer.epoch))
            # When the discovery window closes, freeze + dump the learned
            # matrix (and its contrast against the startup seed) for audit.
            if int(trainer.epoch) == RAMP_END_EPOCH:
                try:
                    save_dir = Path(getattr(trainer, "save_dir", "."))
                    wrapper.dump_learned_overlaps(save_dir / "learned_overlap_priors.json")
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
                    f"{wrapper._last_exclusivity_value:.6f}\n"
                )
        except Exception:
            pass

    return on_train_start, on_train_epoch_start, on_train_batch_end


# ============================================================
# TRAINING
# ============================================================
def run_training_experiment(
    experiment: dict,
    dataset_yaml_path: Path,
    class_names: List[str],
    by_split: Dict[str, List[dict]],
    species_priors: Optional[Dict[str, Dict[str, np.ndarray]]],
    species_overlap_priors: Optional[Dict[str, np.ndarray]],
) -> dict:
    run_name = experiment["name"]
    batch_size = experiment["batch"]
    imgsz = experiment["imgsz"]
    iaway_bias = bool(experiment.get("iaway_bias", False))

    YOLO = get_yolo_class()
    model = YOLO(BASE_MODEL_WEIGHTS)

    on_train_start_cb, on_train_epoch_start_cb, on_train_batch_end_cb = make_sem_loss_callbacks(
        class_names=class_names,
        species_priors=species_priors,
        species_overlap_priors=species_overlap_priors,
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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    species_overlap_priors = compute_species_overlap_priors(
        by_split["train"], class_names,
        grid_res=OVERLAP_GRID_RES,
        min_observations=OVERLAP_MIN_OBSERVATIONS,
    )
    save_overlap_priors(species_overlap_priors, class_names, OUTPUT_ROOT / OVERLAP_CACHE_NAME)

    all_summaries = []
    for experiment in EXPERIMENTS:
        summary = run_training_experiment(
            experiment, dataset_yaml_path, class_names, by_split,
            species_priors, species_overlap_priors,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_ROOT / "all_experiments_summary.csv", index=False)
    with open(OUTPUT_ROOT / "all_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("Finished. Summary saved to:", OUTPUT_ROOT / "all_experiments_summary.csv")


if __name__ == "__main__":
    main()
