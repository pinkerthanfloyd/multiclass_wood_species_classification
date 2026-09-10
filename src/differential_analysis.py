"""differential_analysis.py
=============================

Análisis diferencial entre los outputs de species_annotation_proximity.py
calculados sobre dos conjuntos de anotaciones — típicamente GT y predichas
por el segmentador con sem_loss — para localizar dónde el modelo distorsiona
el mapa anatómico inter-especies.

Funcionamiento
--------------
1. Carga las dos matrices de distancia inter-especies (`species_distance.csv`).
2. Restringe a la intersección de especies presentes en ambas (con al menos
   MIN_IMAGES en cada lado si los CSV de firmas están disponibles).
3. Calcula:
     - Mantel test (correlación de Pearson y Spearman sobre el triángulo
       superior; p-value por permutación de filas/columnas).
     - Matriz delta D_pred − D_gt (positivo = pred separa MÁS que GT).
     - Pares con mayor |delta| en distancia absoluta y en rank de vecino.
     - "Movimiento" por especie: cuánto cambia su distancia media al resto.
     - Procrustes alignment de proyecciones MDS-2D — overlay de los dos
       mapas alineados.
4. Si se proveen las firmas y/o relaciones espaciales, repite la
   comparación bloque por bloque (B1/B2/B3/B4) para identificar dónde
   vive la distorsión.

Inputs
------
DISTANCE_GT, DISTANCE_PRED        species_distance.csv (obligatorios).
SIGNATURES_GT, SIGNATURES_PRED    species_signatures.csv (opcional, habilita
                                  comparación por bloque B1/B2/B3).
SPATIAL_GT, SPATIAL_PRED          per_image_spatial_relations.csv (opcional,
                                  añade B4).
ESPECIES_JSON                     mapping id->nombre latino para etiquetas.

Outputs
-------
mantel.json                       correlaciones global y por bloque, con p-value
delta_matrix.csv                  D_pred − D_gt, reindexada a especies comunes
delta_heatmap.png                 heatmap de delta reordenado por jerarquía de GT
top_changed_pairs.csv             pares ordenados por |delta|
top_rank_flips.csv                pares donde el rango de vecino cambia más
species_movement.csv              movimiento promedio y máximo por especie
procrustes_overlay.png            MDS 2D alineado de los dos espacios
per_block_correlation.csv         Mantel por bloque (si firmas disponibles)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIG
# ============================================================
DISTANCE_GT      = Path("outputs/species_proximity_gt/species_distance.csv")
DISTANCE_PRED    = Path("outputs/species_proximity/species_distance.csv")
SIGNATURES_GT    = Path("outputs/species_proximity_gt/species_signatures.csv")
SIGNATURES_PRED  = Path("outputs/species_proximity/species_signatures.csv")
SPATIAL_GT       = Path("outputs/species_proximity_gt/per_image_spatial_relations.csv")
SPATIAL_PRED     = Path("outputs/species_proximity/per_image_spatial_relations.csv")
ESPECIES_JSON    = Path("inferences/especies.json")
OUTPUT_DIR       = Path("outputs/diff_analysis")

MIN_IMAGES = 2
TOP_PAIRS = 30
MANTEL_PERMUTATIONS = 1000
SEED = 42


# ============================================================
# Helpers
# ============================================================
def load_especies_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        ks = str(k)
        out[ks] = str(v); out[ks.lstrip("0") or "0"] = str(v)
    return out


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = [str(i).zfill(4) if str(i).isdigit() else str(i) for i in df.index]
    df.columns = [str(c).zfill(4) if str(c).isdigit() else str(c) for c in df.columns]
    return df


def common_species(D_gt: pd.DataFrame, D_pr: pd.DataFrame) -> List[str]:
    common = sorted(set(D_gt.index) & set(D_pr.index))
    return common


def _condensed(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def mantel_test(D1: np.ndarray, D2: np.ndarray, n_perm: int,
                rng: np.random.Generator) -> Dict[str, float]:
    v1 = _condensed(D1)
    v2 = _condensed(D2)
    r_pearson, _ = pearsonr(v1, v2)
    r_spearman, _ = spearmanr(v1, v2)
    # p-value por permutación (Mantel clásico: permuta filas+columnas)
    n = D1.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        r, _ = pearsonr(v1, _condensed(D2p))
        if abs(r) >= abs(r_pearson):
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return {
        "pearson": float(r_pearson),
        "spearman": float(r_spearman),
        "p_value": float(p_val),
        "n_species": int(n),
        "n_permutations": int(n_perm),
    }


def hierarchical_order(D: np.ndarray) -> np.ndarray:
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average",
                optimal_ordering=True)
    return leaves_list(Z)


def classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    D = (D + D.T) / 2.0
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))


def procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Alinea B sobre A (rotación + escala + traslación) y devuelve B', error."""
    A0 = A - A.mean(axis=0); B0 = B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(B0.T @ A0)
    R = U @ Vt
    s = np.trace(B0.T @ A0 @ R.T) / np.trace(B0.T @ B0)
    Bp = s * B0 @ R + A.mean(axis=0)
    err = float(np.linalg.norm(A - Bp))
    return Bp, err


# ============================================================
# Bloques desde firmas (opcional)
# ============================================================
def block_distances_from_signatures(sig_path: Path, species: List[str]
                                    ) -> Optional[Dict[str, np.ndarray]]:
    if not sig_path.exists():
        return None
    sig = pd.read_csv(sig_path)
    sig["species_id"] = sig["species_id"].astype(str).str.zfill(4)
    sig = sig[sig["species_id"].isin(species)]
    sig = sig.set_index("species_id").reindex(species).reset_index()
    pres = sorted([c for c in sig.columns if c.startswith("B1_presence_")])
    area = sorted([c for c in sig.columns if c.startswith("B1_areafrac_")])
    shape_mean = sorted([c for c in sig.columns if c.startswith("B2_") and "_std_" not in c])
    shape_std = sorted([c for c in sig.columns if c.startswith("B2_") and "_std_" in c])
    cooc = sorted([c for c in sig.columns if c.startswith("B3_jacc_")])
    if not (pres and area and cooc):
        return None

    def _cos(X):
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
        return 1 - np.clip((X / nrm) @ (X / nrm).T, -1, 1)

    def _z(X):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
        return (X - mu) / sd

    P = sig[pres].to_numpy(float)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    A = sig[area].to_numpy(float)
    H = np.hstack([sig[shape_mean].to_numpy(float), sig[shape_std].to_numpy(float)])
    return {
        "B1": _cos(np.hstack([P, A])),
        "B2": _cos(_z(H)),
        "B3": _cos(sig[cooc].to_numpy(float)),
    }


def block_distance_b4(spatial_path: Path, species: List[str]) -> Optional[np.ndarray]:
    if not spatial_path.exists():
        return None
    df = pd.read_csv(spatial_path)
    df["species_id"] = df["species_id"].astype(str).str.zfill(4)
    df = df[df["species_id"].isin(species)]
    cols = [c for c in df.columns
            if c.startswith(("reldist_", "closefrac_", "iou_", "contain_"))]
    agg = df.groupby("species_id")[cols].mean().reindex(species).fillna(0.0)
    X = agg.to_numpy(float)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1
    X = (X - mu) / sd
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return 1 - np.clip((X / nrm) @ (X / nrm).T, -1, 1)


# ============================================================
# Outputs específicos
# ============================================================
def write_top_changed_pairs(species: List[str], delta: np.ndarray,
                            latin: Dict[str, str], out: Path, top: int) -> None:
    n = len(species)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "species_a": species[i],
                "name_a": latin.get(species[i], ""),
                "species_b": species[j],
                "name_b": latin.get(species[j], ""),
                "delta": float(delta[i, j]),
            })
    df = pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False)
    df.head(top).to_csv(out, index=False, encoding="utf-8-sig")


def write_top_rank_flips(species: List[str], D_gt: np.ndarray, D_pr: np.ndarray,
                         latin: Dict[str, str], out: Path, top: int) -> None:
    n = len(species)
    rows = []
    for i in range(n):
        rk_gt = np.argsort(D_gt[i]); rk_pr = np.argsort(D_pr[i])
        pos_gt = {idx: k for k, idx in enumerate(rk_gt)}
        pos_pr = {idx: k for k, idx in enumerate(rk_pr)}
        for j in range(n):
            if i == j: continue
            d_rank = pos_pr[j] - pos_gt[j]
            rows.append({
                "species_a": species[i],
                "name_a": latin.get(species[i], ""),
                "species_b": species[j],
                "name_b": latin.get(species[j], ""),
                "rank_gt": int(pos_gt[j]),
                "rank_pred": int(pos_pr[j]),
                "rank_delta": int(d_rank),
            })
    df = pd.DataFrame(rows).sort_values("rank_delta", key=lambda s: s.abs(), ascending=False)
    df.head(top).to_csv(out, index=False, encoding="utf-8-sig")


def write_species_movement(species: List[str], D_gt: np.ndarray, D_pr: np.ndarray,
                           latin: Dict[str, str], out: Path) -> None:
    delta = D_pr - D_gt
    rows = []
    for i, sp in enumerate(species):
        row = delta[i].copy(); row[i] = np.nan
        rows.append({
            "species_id": sp,
            "species_name": latin.get(sp, ""),
            "mean_delta": float(np.nanmean(row)),
            "abs_mean_delta": float(np.nanmean(np.abs(row))),
            "max_abs_delta": float(np.nanmax(np.abs(row))),
        })
    df = pd.DataFrame(rows).sort_values("abs_mean_delta", ascending=False)
    df.to_csv(out, index=False, encoding="utf-8-sig")


def plot_delta_heatmap(species: List[str], delta: np.ndarray, order: np.ndarray,
                       latin: Dict[str, str], out: Path) -> None:
    n = len(species)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.16), max(8, n * 0.16)))
    Dr = delta[np.ix_(order, order)]
    vmax = float(np.nanmax(np.abs(Dr)))
    im = ax.imshow(Dr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    labels = [f"{species[i]} {latin.get(species[i], '').split()[0][:16]}" for i in order]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.025,
                 label="D_pred − D_gt  (+ pred separa más  / − pred colapsa)")
    ax.set_title("Delta inter-especie: predicho menos GT (reordenado por GT)")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def plot_procrustes(species: List[str], A: np.ndarray, Bp: np.ndarray,
                    latin: Dict[str, str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, sp in enumerate(species):
        ax.plot([A[i, 0], Bp[i, 0]], [A[i, 1], Bp[i, 1]],
                color="gray", alpha=0.4, linewidth=0.8)
    ax.scatter(A[:, 0], A[:, 1], s=45, c="#1f77b4", label="GT", edgecolors="black", linewidths=0.4)
    ax.scatter(Bp[:, 0], Bp[:, 1], s=45, c="#d62728", label="pred (alineado)",
               edgecolors="black", linewidths=0.4, marker="^")
    for i, sp in enumerate(species):
        ax.annotate(sp, (A[i, 0], A[i, 1]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.legend(loc="best")
    ax.set_title("Procrustes: MDS 2D de GT vs. pred — líneas muestran desplazamiento")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latin = load_especies_map(ESPECIES_JSON)
    rng = np.random.default_rng(SEED)

    print("[1] Cargando matrices...")
    D_gt = load_matrix(DISTANCE_GT)
    D_pr = load_matrix(DISTANCE_PRED)
    species = common_species(D_gt, D_pr)
    print(f"    especies comunes: {len(species)}")
    D_gt = D_gt.loc[species, species].to_numpy(float)
    D_pr = D_pr.loc[species, species].to_numpy(float)

    print("[2] Mantel test sobre la matriz combinada...")
    mantel = {"combined": mantel_test(D_gt, D_pr, MANTEL_PERMUTATIONS, rng)}
    print(f"    Pearson={mantel['combined']['pearson']:+.3f}  "
          f"Spearman={mantel['combined']['spearman']:+.3f}  "
          f"p={mantel['combined']['p_value']:.4f}")

    print("[3] Mantel test por bloque (si firmas/spatial disponibles)...")
    bd_gt = block_distances_from_signatures(SIGNATURES_GT, species)
    bd_pr = block_distances_from_signatures(SIGNATURES_PRED, species)
    if bd_gt and bd_pr:
        for k in ("B1", "B2", "B3"):
            if k in bd_gt and k in bd_pr:
                mantel[k] = mantel_test(bd_gt[k], bd_pr[k], MANTEL_PERMUTATIONS, rng)
                print(f"    {k}: r={mantel[k]['pearson']:+.3f}  p={mantel[k]['p_value']:.4f}")
    b4_gt = block_distance_b4(SPATIAL_GT, species)
    b4_pr = block_distance_b4(SPATIAL_PRED, species)
    if b4_gt is not None and b4_pr is not None:
        mantel["B4"] = mantel_test(b4_gt, b4_pr, MANTEL_PERMUTATIONS, rng)
        print(f"    B4: r={mantel['B4']['pearson']:+.3f}  p={mantel['B4']['p_value']:.4f}")

    (OUTPUT_DIR / "mantel.json").write_text(json.dumps(mantel, indent=2), encoding="utf-8")
    pd.DataFrame([{"block": k, **v} for k, v in mantel.items()]).to_csv(
        OUTPUT_DIR / "per_block_correlation.csv", index=False, encoding="utf-8-sig")

    print("[4] Matriz delta y outputs derivados...")
    delta = D_pr - D_gt
    pd.DataFrame(delta, index=species, columns=species).to_csv(
        OUTPUT_DIR / "delta_matrix.csv", encoding="utf-8-sig")
    order = hierarchical_order(D_gt)
    plot_delta_heatmap(species, delta, order, latin,
                       OUTPUT_DIR / "delta_heatmap.png")
    write_top_changed_pairs(species, delta, latin,
                            OUTPUT_DIR / "top_changed_pairs.csv", TOP_PAIRS)
    write_top_rank_flips(species, D_gt, D_pr, latin,
                         OUTPUT_DIR / "top_rank_flips.csv", TOP_PAIRS)
    write_species_movement(species, D_gt, D_pr, latin,
                           OUTPUT_DIR / "species_movement.csv")

    print("[5] Procrustes sobre MDS 2D...")
    A = classical_mds(D_gt, 2)
    B = classical_mds(D_pr, 2)
    Bp, err = procrustes(A, B)
    print(f"    error Procrustes = {err:.4f}")
    plot_procrustes(species, A, Bp, latin,
                    OUTPUT_DIR / "procrustes_overlay.png")

    print(f"\nTodo en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
