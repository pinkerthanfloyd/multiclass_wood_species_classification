"""visualize_species_proximity.py
=================================

Vistas complementarias sobre la matriz de proximidad entre especies generada
por species_annotation_proximity.py. No produce nada nuevo del análisis: sólo
lee los CSV de firmas y de relaciones espaciales y los reordena, proyecta y
descompone para que la lectura sea directa.

Vistas:

  1. Heatmap combinado (B1+B2+B3+B4) con dendrograma jerárquico al margen y
     filas/columnas reordenadas según ese dendrograma.
  2. Cuatro heatmaps por bloque (B1, B2, B3, B4) con el MISMO orden de filas
     que el combinado, para identificar dónde vive la señal de cada par.
  3. Proyección MDS 2D de la matriz combinada, coloreando por género
     (primera palabra del nombre latino) — referencia visual de cuánto se
     aproxima la firma anatómica a la taxonomía.
  4. Barras agrupadas por bloque para los k pares con MENOR contraste —
     responde "¿por qué se parecen estas dos especies?" descomponiendo la
     distancia en sus cuatro componentes.
  5. Grafo de top-k vecinos por especie, con aristas inversamente
     proporcionales a la distancia combinada y colores por género.

Uso típico:

    python visualize_species_proximity.py \\
        --signatures outputs/species_proximity/species_signatures.csv \\
        --spatial    outputs/species_proximity/per_image_spatial_relations.csv \\
        --especies   data/especies.json \\
        --output-dir outputs/species_proximity/figures \\
        --topk 3 --confusable 12

Notas de diseño:

  * Las matrices por bloque se recomponen aquí en vez de leerse de disco:
    species_signatures.csv ya trae B1/B2/B3 desglosados por columnas, y B4
    se reconstruye agregando per_image_spatial_relations.csv por especie.
    Esto deja la visualización desacoplada del script de análisis.
  * El dendrograma usa Ward sobre la matriz combinada con `optimal_ordering`
    activado para que los clusters salgan bien dispuestos.
  * MDS clásico (eigen-decomposición del double-centering): suficiente para
    visualización y sin dependencia de sklearn. Si tienes sklearn instalado
    y quieres MDS no-métrico o t-SNE/UMAP, los argumentos --backend lo
    permiten.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaves_list
from scipy.spatial.distance import squareform


# ============================================================
# Helpers
# ============================================================
def load_especies_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
        out[str(k).lstrip("0") or "0"] = str(v)
    return out


def genus_of(name: str) -> str:
    if not name:
        return "?"
    return re.split(r"\s+", name.strip())[0]


def _l1(v: np.ndarray) -> np.ndarray:
    s = float(np.abs(v).sum())
    return v / s if s > 0 else v


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def cos_dist(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    if B is None:
        B = A
    nA = np.linalg.norm(A, axis=1, keepdims=True)
    nB = np.linalg.norm(B, axis=1, keepdims=True)
    nA[nA == 0] = 1.0
    nB[nB == 0] = 1.0
    return 1.0 - np.clip((A / nA) @ (B / nB).T, -1.0, 1.0)


# ============================================================
# Reconstrucción de bloques desde los CSV
# ============================================================
def build_blocks(
    sig_path: Path, spatial_path: Optional[Path], min_species: int = 2,
) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
    sig = pd.read_csv(sig_path)
    # primera columna numérica de imágenes para filtrar mínimo
    sig = sig[sig["n_images"] >= min_species].reset_index(drop=True)
    sp_ids = sig["species_id"].astype(str).tolist()
    sp_names = sig["species_name"].fillna("").astype(str).tolist()

    pres_cols = sorted([c for c in sig.columns if c.startswith("B1_presence_")])
    area_cols = sorted([c for c in sig.columns if c.startswith("B1_areafrac_")])
    shape_mean_cols = sorted([c for c in sig.columns
                              if c.startswith("B2_") and "_std_" not in c])
    shape_std_cols = sorted([c for c in sig.columns
                             if c.startswith("B2_") and "_std_" in c])
    cooc_cols = sorted([c for c in sig.columns if c.startswith("B3_jacc_")])

    if not pres_cols or not area_cols or not cooc_cols:
        raise ValueError("species_signatures.csv no expone columnas B1/B2/B3 esperadas.")

    P = sig[pres_cols].to_numpy(dtype=np.float64)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    A = sig[area_cols].to_numpy(dtype=np.float64)
    B1 = np.hstack([P, A])
    H = np.hstack([sig[shape_mean_cols].to_numpy(dtype=np.float64),
                   sig[shape_std_cols].to_numpy(dtype=np.float64)])
    B2 = _zscore_cols(H)
    B3 = sig[cooc_cols].to_numpy(dtype=np.float64)

    blocks: Dict[str, np.ndarray] = {"B1": B1, "B2": B2, "B3": B3}

    if spatial_path is not None and spatial_path.exists():
        sp_df = pd.read_csv(spatial_path)
        sp_df = sp_df[sp_df["species_id"].astype(str).isin(sp_ids)]
        agg_cols = [c for c in sp_df.columns
                    if c.startswith(("reldist_", "closefrac_", "iou_", "contain_"))]
        agg = sp_df.groupby(sp_df["species_id"].astype(str))[agg_cols].mean()
        agg = agg.reindex(sp_ids).fillna(0.0)
        S4 = agg.to_numpy(dtype=np.float64)
        blocks["B4"] = _zscore_cols(S4)

    return sp_ids, sp_names, blocks


def block_distances(blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: cos_dist(v) for k, v in blocks.items()}


def combined_distance(D_blocks: Dict[str, np.ndarray],
                      weights: Dict[str, float]) -> np.ndarray:
    total = None
    wsum = 0.0
    for k, w in weights.items():
        if k not in D_blocks or w <= 0:
            continue
        total = D_blocks[k] * w if total is None else total + D_blocks[k] * w
        wsum += w
    if total is None or wsum == 0:
        raise ValueError("No hay bloques con peso > 0")
    return total / wsum


# ============================================================
# Visualizaciones
# ============================================================
def _to_condensed(D: np.ndarray) -> np.ndarray:
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def hierarchical_order(D: np.ndarray, method: str = "average") -> Tuple[np.ndarray, np.ndarray]:
    Z = linkage(_to_condensed(D), method=method, optimal_ordering=True)
    order = leaves_list(Z)
    return Z, order


def classical_mds(D: np.ndarray, k: int = 2) -> np.ndarray:
    D = (D + D.T) / 2.0
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    coords = V[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))
    return coords


def genus_palette(genera: List[str]) -> Dict[str, str]:
    uniq = sorted(set(g for g in genera if g and g != "?"))
    cmap = get_cmap("tab20", max(20, len(uniq)))
    return {g: to_hex(cmap(i % cmap.N)) for i, g in enumerate(uniq)}


def plot_combined_heatmap(
    sp_ids: List[str], sp_names: List[str], D: np.ndarray,
    order: np.ndarray, out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(sp_ids) * 0.18),
                                    max(7, len(sp_ids) * 0.18)))
    Dr = D[np.ix_(order, order)]
    labels = [f"{sp_ids[i]} {sp_names[i][:24]}" for i in order]
    im = ax.imshow(Dr, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(im, ax=ax, fraction=0.025, label="distancia coseno combinada")
    ax.set_title("Distancia entre especies — reordenada por jerarquía")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def plot_block_heatmaps(
    sp_ids: List[str], D_blocks: Dict[str, np.ndarray],
    order: np.ndarray, out: Path,
) -> None:
    keys = [k for k in ("B1", "B2", "B3", "B4") if k in D_blocks]
    n = len(keys)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axs = [axs]
    for ax, k in zip(axs, keys):
        D = D_blocks[k][np.ix_(order, order)]
        im = ax.imshow(D, cmap="viridis_r", aspect="auto")
        ax.set_title(k)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Distancia por bloque — mismo orden que el combinado")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def plot_mds(
    sp_ids: List[str], sp_names: List[str], D: np.ndarray, out: Path,
) -> None:
    coords = classical_mds(D, k=2)
    genera = [genus_of(n) for n in sp_names]
    palette = genus_palette(genera)
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, sp in enumerate(sp_ids):
        color = palette.get(genera[i], "#888888")
        ax.scatter(coords[i, 0], coords[i, 1], s=60, c=color,
                   edgecolors="black", linewidths=0.5)
        label = sp_names[i].split()[1] if len(sp_names[i].split()) > 1 else sp
        ax.annotate(label[:14], (coords[i, 0], coords[i, 1]),
                    fontsize=6, alpha=0.75, xytext=(3, 3),
                    textcoords="offset points")
    handles = [Patch(facecolor=palette[g], edgecolor="black", label=g)
               for g in sorted(palette.keys())]
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, ncol=1, frameon=False, title="Género")
    ax.set_title("MDS 2D de la distancia combinada — color por género")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusable_decomposition(
    sp_ids: List[str], sp_names: List[str],
    D_blocks: Dict[str, np.ndarray], D_combined: np.ndarray,
    top_n: int, out: Path,
) -> None:
    """Para los top_n pares MÁS cercanos en la matriz combinada (los que el
    RF/SVM va a confundir), descompone la distancia en sus 4 bloques."""
    pairs = []
    for i in range(len(sp_ids)):
        for j in range(i + 1, len(sp_ids)):
            pairs.append((D_combined[i, j], i, j))
    pairs.sort(key=lambda x: x[0])
    chosen = pairs[:top_n]

    keys = [k for k in ("B1", "B2", "B3", "B4") if k in D_blocks]
    vals = np.zeros((len(chosen), len(keys)), dtype=np.float64)
    labels = []
    for r, (_, i, j) in enumerate(chosen):
        for c, k in enumerate(keys):
            vals[r, c] = D_blocks[k][i, j]
        ni = (sp_names[i] or sp_ids[i]).split()
        nj = (sp_names[j] or sp_ids[j]).split()
        a_lab = " ".join(ni[:2]) if ni else sp_ids[i]
        b_lab = " ".join(nj[:2]) if nj else sp_ids[j]
        labels.append(f"{a_lab}  ↔  {b_lab}")

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(chosen) + 2)))
    width = 0.8 / len(keys)
    y = np.arange(len(chosen))
    cmap = get_cmap("tab10")
    for c, k in enumerate(keys):
        ax.barh(y + c * width - 0.4 + width / 2, vals[:, c],
                height=width, label=k, color=cmap(c))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("distancia coseno (por bloque)")
    ax.set_title(f"Top-{top_n} pares más confundibles — descomposición por bloque")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def plot_topk_graph(
    sp_ids: List[str], sp_names: List[str],
    D: np.ndarray, k: int, out: Path,
) -> None:
    """Grafo dirigido (por especie, sus k-vecinos más cercanos) sobre la
    proyección MDS. Aristas con grosor inversamente proporcional a la
    distancia y color por género del origen."""
    coords = classical_mds(D, k=2)
    genera = [genus_of(n) for n in sp_names]
    palette = genus_palette(genera)
    fig, ax = plt.subplots(figsize=(11, 8))
    # aristas
    for i in range(len(sp_ids)):
        order = np.argsort(D[i])
        rank = 0
        for j in order:
            if j == i:
                continue
            rank += 1
            w = max(0.2, 1.5 - D[i, j])
            ax.plot([coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=palette.get(genera[i], "#888888"),
                    alpha=0.25, linewidth=w * 0.9, zorder=1)
            if rank >= k:
                break
    # nodos
    for i, sp in enumerate(sp_ids):
        ax.scatter(coords[i, 0], coords[i, 1], s=70,
                   c=palette.get(genera[i], "#888888"),
                   edgecolors="black", linewidths=0.5, zorder=2)
        ax.annotate(sp_ids[i], (coords[i, 0], coords[i, 1]),
                    fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"Grafo top-{k} vecinos por especie (MDS 2D, color = género)")
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def plot_dendrogram(
    sp_ids: List[str], sp_names: List[str], Z: np.ndarray, out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.12 * len(sp_ids)), 6))
    labels = [f"{sp_ids[i]} {sp_names[i].split()[0] if sp_names[i] else ''}"
              for i in range(len(sp_ids))]
    dendrogram(Z, labels=labels, leaf_font_size=6, color_threshold=0.7 * Z[:, 2].max(), ax=ax)
    ax.set_title("Dendrograma jerárquico (Ward) — distancia combinada")
    ax.set_ylabel("distancia")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signatures", type=Path, required=True,
                    help="species_signatures.csv generado por species_annotation_proximity.py")
    ap.add_argument("--spatial", type=Path, default=None,
                    help="per_image_spatial_relations.csv (opcional, habilita B4).")
    ap.add_argument("--especies", type=Path, default=None,
                    help="Mapping id->Latin name (opcional, refuerza el coloreado por género).")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/species_proximity/figures"))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--confusable", type=int, default=12,
                    help="Cuántos pares más cercanos descomponer por bloque.")
    ap.add_argument("--min-species", type=int, default=2,
                    help="Mínimo de imágenes por especie a incluir.")
    ap.add_argument("--w-b1", type=float, default=1.0)
    ap.add_argument("--w-b2", type=float, default=1.0)
    ap.add_argument("--w-b3", type=float, default=1.0)
    ap.add_argument("--w-b4", type=float, default=1.0)
    ap.add_argument("--linkage", default="average",
                    choices=["average", "ward", "complete", "single"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latin = load_especies_map(args.especies)

    print("[1] Cargando firmas y reconstruyendo bloques...")
    sp_ids, sp_names_in, blocks = build_blocks(
        args.signatures, args.spatial, min_species=args.min_species
    )
    # completar nombre latino desde --especies si faltó en el CSV
    sp_names = [n if n else latin.get(sp_ids[i], "") for i, n in enumerate(sp_names_in)]
    print(f"     S={len(sp_ids)}  bloques={list(blocks.keys())}")

    print("[2] Calculando distancias por bloque y combinada...")
    D_blocks = block_distances(blocks)
    weights = {"B1": args.w_b1, "B2": args.w_b2, "B3": args.w_b3, "B4": args.w_b4}
    D_total = combined_distance(D_blocks, weights)
    pd.DataFrame(D_total, index=sp_ids, columns=sp_ids).to_csv(
        args.output_dir / "combined_distance_used_for_plots.csv",
        encoding="utf-8-sig",
    )

    print("[3] Reordenando por jerarquía...")
    Z, order = hierarchical_order(D_total, method=args.linkage)

    print("[4] Dibujando vistas...")
    plot_combined_heatmap(sp_ids, sp_names, D_total, order,
                          args.output_dir / "combined_heatmap.png")
    plot_block_heatmaps(sp_ids, D_blocks, order,
                        args.output_dir / "block_heatmaps.png")
    plot_mds(sp_ids, sp_names, D_total,
             args.output_dir / "mds_scatter.png")
    plot_confusable_decomposition(sp_ids, sp_names, D_blocks, D_total,
                                  args.confusable,
                                  args.output_dir / "confusable_decomposition.png")
    plot_topk_graph(sp_ids, sp_names, D_total, args.topk,
                    args.output_dir / "topk_graph.png")
    plot_dendrogram(sp_ids, sp_names, Z,
                    args.output_dir / "dendrogram.png")

    print(f"Figuras escritas en {args.output_dir}")


if __name__ == "__main__":
    main()
