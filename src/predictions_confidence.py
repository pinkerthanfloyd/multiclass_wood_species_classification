"""predictions_confidence.py
=============================

Completa el segundo modelo (RF/SVM) con métricas de CONFIANZA y EVALUACIÓN
basadas en la matriz de distancias inter-especies.

Funcionamiento
--------------
Lee un CSV de predicciones top-K producido por species_predict.py con
columnas como pred_<m>, prob_<m>, topk_<m>, probk_<m> para cada modelo
m ∈ {rf, svm}. Para cada fila calcula:

  CONFIANZA (no requiere especie verdadera)
  -----------------------------------------
  spread_<m>_<src>        Σ p_i p_j d(p_i, p_j) / Σ p_i p_j   sobre el top-K.
                          Bajo = top-K concentrado en un vecindario coherente.
                          Alto = el modelo duda entre especies anatómicamente
                          dispares — predicción sospechosa.
  spread_z_<m>_<src>      z-score del spread respecto al spread esperable
                          tomando K especies al azar del catálogo. Positivo
                          si el top-K es MÁS coherente que el azar.
  coherence_<m>_<src>     1 - spread_<m>_<src> / spread_random_mean. Lectura
                          intuitiva: 1 = perfectamente coherente, 0 = como
                          una predicción aleatoria, < 0 = peor que aleatoria.

  EVALUACIÓN (requiere species_id verdadera en la fila)
  -----------------------------------------------------
  d_to_true_<m>_<src>     E_{s ~ p_model}[d(s_true, s)] = Σ p_i d(s_true, p_i).
                          Distancia esperada del top-K a la verdad bajo la
                          distribución del modelo.
  score_calibrated_<m>_<src>  z-score de la d_to_true respecto a la
                              distribución de distancias d(s_true, random).
                              Positivo = el modelo predijo MÁS cerca que el
                              azar; ≥1 = predicción claramente útil aunque
                              haya errado en rank-1.

Sufijo <src> = "gt" o "pred" según la matriz utilizada — el script computa
las dos para que se puedan contrastar lado a lado.

Inputs
------
PREDICTIONS_CSV       CSV de salida de species_predict.py.
DISTANCE_MATRIX_GT    species_distance.csv calculada sobre las anotaciones
                      GT (con species_annotation_proximity.py).
DISTANCE_MATRIX_PRED  species_distance.csv calculada sobre las anotaciones
                      predichas por el segmentador.
ESPECIES_JSON         mapping id->Latin name; necesario porque el
                      clasificador predice nombres latinos y la matriz
                      está indexada por species_id (4 dígitos).
TRUE_LABEL_COL        nombre de la columna con la especie verdadera (si
                      existe). Acepta tanto species_id (4 dígitos) como
                      nombre latino.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
PREDICTIONS_CSV = Path("inferences/32_images_pred.csv")
DISTANCE_MATRIX_GT   = Path("outputs/species_proximity_gt/species_distance.csv")
DISTANCE_MATRIX_PRED = Path("outputs/species_proximity_pred/species_distance.csv")
ESPECIES_JSON = Path("inferences/especies.json")
OUTPUT_CSV    = Path("inferences/32_images_pred/predictions_scored.csv")

MODELS = ("rf", "svm")        # busca topk_<m> y probk_<m>
TOP_K = 5                      # cuántas opciones del top-K considerar
N_RANDOM = 1000                # muestras para baselines aleatorios
TRUE_LABEL_COL = "true_species"  # opcional; salta evaluación si no existe
SEED = 42


# ============================================================
# Helpers
# ============================================================
def load_especies_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Devuelve (id->name, name->id)."""
    if not path.exists():
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id_to_name: Dict[str, str] = {}
    name_to_id: Dict[str, str] = {}
    for k, v in raw.items():
        kstr = str(k)
        id_to_name[kstr] = str(v)
        id_to_name[kstr.lstrip("0") or "0"] = str(v)
        name_to_id[str(v)] = kstr
    return id_to_name, name_to_id


def to_species_id(value, name_to_id: Dict[str, str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # ya parece un id de 4 dígitos
    if re.fullmatch(r"\d{3,5}", s):
        return s.zfill(4)
    # nombre latino → id
    return name_to_id.get(s)


def parse_list_cell(cell) -> Optional[list]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return None
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            # fallback: "a;b;c" o "a,b,c"
            parts = re.split(r"[;,|]", cell.strip("[]()"))
            return [p.strip().strip("'\"") for p in parts if p.strip()]
    return None


def extract_topk_for_model(row, model: str, k: int,
                           name_to_id: Dict[str, str]) -> Tuple[List[str], List[float]]:
    """Devuelve listas de species_id y probs, recortadas a k."""
    topk_raw = parse_list_cell(row.get(f"topk_{model}"))
    probk_raw = parse_list_cell(row.get(f"probk_{model}"))
    if topk_raw is None:
        single = row.get(f"pred_{model}")
        prob = row.get(f"prob_{model}", 1.0)
        topk_raw = [single] if single is not None else []
        probk_raw = [prob]
    ids = []
    probs = []
    for s, p in zip(topk_raw or [], probk_raw or []):
        sid = to_species_id(s, name_to_id)
        if sid is None:
            continue
        try:
            ids.append(sid)
            probs.append(float(p))
        except (TypeError, ValueError):
            continue
        if len(ids) >= k:
            break
    return ids, probs


def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = [str(i).zfill(4) if str(i).isdigit() else str(i) for i in df.index]
    df.columns = [str(c).zfill(4) if str(c).isdigit() else str(c) for c in df.columns]
    return df


# ============================================================
# Métricas
# ============================================================
def spread_topk(topk: List[str], probs: List[float], D: pd.DataFrame) -> float:
    if len(topk) < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(len(topk)):
        if topk[i] not in D.index:
            continue
        for j in range(i + 1, len(topk)):
            if topk[j] not in D.columns:
                continue
            d = float(D.loc[topk[i], topk[j]])
            w = probs[i] * probs[j]
            num += w * d
            den += w
    return float(num / den) if den > 0 else 0.0


def random_spread_baseline(D: pd.DataFrame, k: int, n_samples: int,
                           rng: np.random.Generator) -> Tuple[float, float]:
    species = D.index.tolist()
    n = len(species)
    if n < 2 or k < 2:
        return 0.0, 0.0
    M = D.to_numpy()
    spreads = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        idx = rng.choice(n, size=min(k, n), replace=False)
        sub = M[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        spreads[s] = sub[iu].mean()
    return float(spreads.mean()), float(spreads.std() + 1e-12)


def expected_distance_to_true(topk: List[str], probs: List[float],
                              true_id: Optional[str], D: pd.DataFrame) -> float:
    if true_id is None or true_id not in D.index:
        return float("nan")
    total_p = sum(probs)
    if total_p <= 0:
        return float("nan")
    e = 0.0
    used = 0.0
    for s, p in zip(topk, probs):
        if s in D.columns:
            e += p * float(D.loc[true_id, s])
            used += p
    return float(e / used) if used > 0 else float("nan")


def random_dist_baseline(D: pd.DataFrame, true_id: Optional[str],
                         n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    if true_id is None or true_id not in D.index:
        return float("nan"), float("nan")
    others = [s for s in D.index if s != true_id]
    if not others:
        return float("nan"), float("nan")
    sampled = rng.choice(others, size=n_samples, replace=True)
    dists = D.loc[true_id, sampled].to_numpy(dtype=np.float64)
    return float(dists.mean()), float(dists.std() + 1e-12)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    id_to_name, name_to_id = load_especies_map(ESPECIES_JSON)

    print(f"[1] Leyendo {PREDICTIONS_CSV} ...")
    pred = pd.read_csv(PREDICTIONS_CSV)
    print(f"    {len(pred)} filas, columnas={list(pred.columns)[:12]}...")

    matrices: Dict[str, pd.DataFrame] = {}
    if DISTANCE_MATRIX_GT.exists():
        matrices["gt"] = load_distance_matrix(DISTANCE_MATRIX_GT)
        print(f"    matriz GT cargada: {matrices['gt'].shape}")
    if DISTANCE_MATRIX_PRED.exists():
        matrices["pred"] = load_distance_matrix(DISTANCE_MATRIX_PRED)
        print(f"    matriz pred cargada: {matrices['pred'].shape}")
    if not matrices:
        raise SystemExit("Ninguna matriz de distancias disponible.")

    # Baselines aleatorios por matriz (depende sólo de la matriz y de K)
    random_spread = {
        src: random_spread_baseline(D, TOP_K, N_RANDOM, rng)
        for src, D in matrices.items()
    }
    print(f"[2] Baseline aleatorio del spread:")
    for src, (mu, sd) in random_spread.items():
        print(f"    {src}: mu={mu:.4f}  sd={sd:.4f}")

    out = pred.copy()
    has_truth = TRUE_LABEL_COL in pred.columns
    if has_truth:
        print(f"[3] Columna verdad detectada: {TRUE_LABEL_COL}")

    for model in MODELS:
        if not any(c.startswith(f"topk_{model}") or c == f"pred_{model}"
                   for c in pred.columns):
            print(f"    [skip] modelo '{model}' no aparece en el CSV")
            continue
        for src, D in matrices.items():
            mu_sp, sd_sp = random_spread[src]
            spread_col = f"spread_{model}_{src}"
            spreadz_col = f"spread_z_{model}_{src}"
            coh_col = f"coherence_{model}_{src}"
            dtt_col = f"d_to_true_{model}_{src}"
            scal_col = f"score_calibrated_{model}_{src}"
            out[spread_col] = np.nan
            out[spreadz_col] = np.nan
            out[coh_col] = np.nan
            if has_truth:
                out[dtt_col] = np.nan
                out[scal_col] = np.nan

            for idx, row in pred.iterrows():
                topk, probs = extract_topk_for_model(row, model, TOP_K, name_to_id)
                if not topk:
                    continue
                # confianza
                sp = spread_topk(topk, probs, D)
                out.at[idx, spread_col] = sp
                out.at[idx, spreadz_col] = (mu_sp - sp) / sd_sp  # positivo = mejor que azar
                out.at[idx, coh_col] = 1.0 - (sp / mu_sp) if mu_sp > 0 else np.nan
                # evaluación
                if has_truth:
                    true_id = to_species_id(row.get(TRUE_LABEL_COL), name_to_id)
                    e = expected_distance_to_true(topk, probs, true_id, D)
                    mu_d, sd_d = random_dist_baseline(D, true_id, N_RANDOM, rng)
                    out.at[idx, dtt_col] = e
                    if not np.isnan(e) and not np.isnan(mu_d):
                        out.at[idx, scal_col] = (mu_d - e) / sd_d

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[4] Escrito {OUTPUT_CSV} con {len(out.columns)} columnas")

    # resumen rápido por modelo y matriz
    print("\nResumen — medias sobre el lote:")
    for model in MODELS:
        for src in matrices:
            cols = {
                "spread": f"spread_{model}_{src}",
                "spread_z": f"spread_z_{model}_{src}",
                "coherence": f"coherence_{model}_{src}",
            }
            if has_truth:
                cols["score_cal"] = f"score_calibrated_{model}_{src}"
            present = {k: c for k, c in cols.items() if c in out.columns}
            if not present:
                continue
            line = f"  {model}/{src}: " + "  ".join(
                f"{k}={out[c].mean():+.3f}" for k, c in present.items()
            )
            print(line)


if __name__ == "__main__":
    main()
