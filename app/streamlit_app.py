"""app.py — Demo local Streamlit
================================
Levanta una demo local que:
  1. Sube una imagen de macroscopía de madera.
  2. La pasa por el segmentador YOLO (sem_loss best.pt) para predecir
     anotaciones (polígonos + clase).
  3. Construye el vector de features POR IMAGEN con el mismo esquema que
     species_classifier.py (Clase_<name>_count / _area_sum / _area_mean /
     _area_std / _density_%, más total_annotations / total_annotated_area
     / total_density_%).
  4. Pasa el vector por RF y/o SVM ya entrenados y devuelve top-K con
     probabilidades.
  5. Calcula, para cada modelo, el spread_topk de confianza usando las
     matrices de distancias inter-especies — la del GT y la del conjunto
     de anotaciones predichas — y los muestra lado a lado.
Cómo ejecutarla
---------------
    pip install streamlit ultralytics joblib pandas numpy pillow
    streamlit run app.py
Rutas de configuración: edita la sección CONFIG. La app cachea el modelo
de segmentación y los clasificadores entre subidas.
"""
from __future__ import annotations
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import joblib
# ============================================================
# CONFIG
# ============================================================
YOLO_WEIGHTS  = Path("models/sem_loss_stratified/best.pt")
DATA_YAML     = Path("data/raw_selected/data.yaml")
MODEL_RF      = Path("models/species_classifier/species_model_rf.joblib")
MODEL_SVM     = Path("models/species_classifier/species_model_svm.joblib")
DIST_GT       = Path("outputs/species_proximity_gt/species_distance.csv")
DIST_PRED     = Path("outputs/species_proximity/species_distance.csv")
ESPECIES_JSON = Path("inferences/especies.json")
IMGSZ = 1024
CONF = 0.25
NMS_IOU = 0.7
MAX_DET = 300
TOP_K = 5
N_RANDOM_BASELINE = 500
# ============================================================
# Carga cacheada
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo de segmentación...")
def load_yolo(weights: Path):
    from ultralytics import YOLO
    return YOLO(str(weights))
@st.cache_resource(show_spinner="Cargando clasificadores...")
def load_classifiers() -> Dict[str, object]:
    out = {}
    if MODEL_RF.exists():
        out["rf"] = joblib.load(MODEL_RF)
    if MODEL_SVM.exists():
        out["svm"] = joblib.load(MODEL_SVM)
    return out
@st.cache_data(show_spinner=False)
def load_class_names(yaml_path: Path) -> List[str]:
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return [str(n) for n in names]
    return [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
@st.cache_data(show_spinner=False)
def load_especies_map(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not path.exists():
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id_to_name, name_to_id = {}, {}
    for k, v in raw.items():
        ks = str(k); name = str(v)
        id_to_name[ks] = name
        id_to_name[ks.lstrip("0") or "0"] = name
        name_to_id[name] = ks
    return id_to_name, name_to_id
@st.cache_data(show_spinner=False)
def load_distance_matrix(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = [str(i).zfill(4) if str(i).isdigit() else str(i) for i in df.index]
    df.columns = [str(c).zfill(4) if str(c).isdigit() else str(c) for c in df.columns]
    return df
# ============================================================
# Feature extraction (mismo esquema que species_classifier.py)
# ============================================================
def build_feature_row(result, class_names: List[str]) -> pd.DataFrame:
    """Construye una sola fila (DataFrame 1xN) con el esquema del classifier."""
    H, W = result.orig_shape
    img_area = float(H * W)
    row = {
        "total_annotations": 0,
        "total_annotated_area": 0.0,
        "total_density_%": 0.0,
    }
    for name in class_names:
        prefix = f"Clase_{name}"
        row[f"{prefix}_count"] = 0
        row[f"{prefix}_area_sum"] = 0.0
        row[f"{prefix}_area_mean"] = 0.0
        row[f"{prefix}_area_std"] = 0.0
        row[f"{prefix}_density_%"] = 0.0
    areas_by_class: Dict[str, List[float]] = {n: [] for n in class_names}
    if result.masks is not None and result.boxes is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        mH, mW = masks.shape[1], masks.shape[2]
        scale = float((H * W) / (mH * mW))
        for k, c in enumerate(classes):
            if not (0 <= c < len(class_names)):
                continue
            mask_area = float(masks[k].sum()) * scale
            name = class_names[c]
            prefix = f"Clase_{name}"
            row[f"{prefix}_count"] += 1
            row[f"{prefix}_area_sum"] += mask_area
            row[f"{prefix}_density_%"] += (mask_area / img_area) * 100.0
            row["total_annotations"] += 1
            row["total_annotated_area"] += mask_area
            row["total_density_%"] += (mask_area / img_area) * 100.0
            areas_by_class[name].append(mask_area)
    for name, arr in areas_by_class.items():
        if not arr:
            continue
        a = np.asarray(arr, dtype=float)
        row[f"Clase_{name}_area_mean"] = float(a.mean())
        row[f"Clase_{name}_area_std"]  = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return pd.DataFrame([row])
def align_features_to_pipeline(row: pd.DataFrame, pipe) -> pd.DataFrame:
    expected = getattr(pipe, "feature_names_in_", None)
    if expected is None:
        first = pipe.steps[0][1]
        expected = getattr(first, "feature_names_in_", None)
    if expected is None:
        return row
    expected = list(expected)
    aligned = pd.DataFrame(columns=expected)
    for c in expected:
        aligned[c] = row[c] if c in row.columns else 0.0
    return aligned
# ============================================================
# Top-K y confianza
# ============================================================
def topk_from_pipeline(pipe, X: pd.DataFrame, k: int) -> Tuple[List[str], List[float]]:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
    else:
        scores = pipe.decision_function(X)[0]
        proba = np.exp(scores - scores.max()); proba /= proba.sum()
    classes = list(pipe.classes_)
    order = np.argsort(proba)[::-1][:k]
    return [classes[i] for i in order], [float(proba[i]) for i in order]
def to_species_id(name_or_id: str, name_to_id: Dict[str, str]) -> Optional[str]:
    s = str(name_or_id).strip()
    if re.fullmatch(r"\d{3,5}", s):
        return s.zfill(4)
    return name_to_id.get(s)
def spread_topk(ids: List[str], probs: List[float], D: pd.DataFrame) -> float:
    if D is None or len(ids) < 2:
        return float("nan")
    num = 0.0; den = 0.0
    for i in range(len(ids)):
        if ids[i] not in D.index: continue
        for j in range(i + 1, len(ids)):
            if ids[j] not in D.columns: continue
            d = float(D.loc[ids[i], ids[j]])
            w = probs[i] * probs[j]
            num += w * d; den += w
    return float(num / den) if den > 0 else float("nan")
@st.cache_data(show_spinner=False)
def random_spread_baseline(_D_signature: str, D_path: Path, k: int,
                           n_samples: int) -> Tuple[float, float]:
    D = load_distance_matrix(D_path)
    if D is None:
        return float("nan"), float("nan")
    species = D.index.tolist()
    n = len(species)
    if n < 2 or k < 2:
        return 0.0, 0.0
    M = D.to_numpy()
    rng = np.random.default_rng(42)
    out = np.empty(n_samples)
    for s in range(n_samples):
        idx = rng.choice(n, size=min(k, n), replace=False)
        sub = M[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        out[s] = sub[iu].mean()
    return float(out.mean()), float(out.std() + 1e-12)
# ============================================================
# Overlay de anotaciones — paleta gruvbox/tierra + leyenda numerada
# ============================================================
# Paleta gruvbox saturada — contraste alto contra madera, pero dentro
# de la gama tierra/vino/oliva de la memoria
PALETTE = [
    (214,  93,  14),   # naranja fuerte    (gruvbox orange dark)
    (204,  36,  29),   # rojo terracota    (gruvbox red)
    (177,  98, 134),   # púrpura           (gruvbox purple)
    ( 69, 133, 136),   # petróleo          (gruvbox blue)
    (152, 151,  26),   # oliva             (gruvbox yellow-green)
    (104, 157, 106),   # verde bosque      (gruvbox aqua)
    (215, 153,  33),   # mostaza           (gruvbox yellow)
    (157,   0,   6),   # rojo vino oscuro
    (143,  63, 113),   # púrpura oscuro    (gruvbox purple dark)
    (254, 128,  25),   # naranja brillante (gruvbox orange bright)
    (250, 189,  47),   # amarillo cálido   (gruvbox yellow bright)
    (142, 192, 124),   # verde claro       (gruvbox green bright)
    (131, 165, 152),   # aqua desaturado
    (211, 134, 155),   # rosa vino
]
def _fit_font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
def draw_overlay(image: Image.Image, result, class_names: List[str]) -> Image.Image:
    """Overlay estilo memoria: relleno translúcido, borde nítido y leyenda
    (idx clase — n instancias) arriba a la derecha."""
    out = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if result.masks is None or result.boxes is None:
        return out.convert("RGB")
    classes = result.boxes.cls.cpu().numpy().astype(int)
    polys = result.masks.xy
    counts: Dict[int, int] = {}
    for c, poly in zip(classes, polys):
        if poly is None or len(poly) < 3:
            continue
        counts[int(c)] = counts.get(int(c), 0) + 1
        color_rgb = PALETTE[int(c) % len(PALETTE)]
        fill = color_rgb + (140,)
        outline = color_rgb + (255,)
        pts = [tuple(p) for p in poly]
        draw.polygon(pts, fill=fill, outline=None)
        draw.line(pts + [pts[0]], fill=outline, width=4)
    composed = Image.alpha_composite(out, overlay)
    if counts:
        W, H = composed.size
        base = max(14, int(min(W, H) * 0.020))
        font = _fit_font(base)
        pad = int(base * 0.6)
        swatch = int(base * 1.1)
        gap = int(base * 0.35)
        entries = sorted(counts.items(), key=lambda kv: kv[0])
        labels = [
            f"{class_names[c] if 0 <= c < len(class_names) else c}  ({n})"
            for c, n in entries
        ]
        tw = max(draw.textlength(lbl, font=font) for lbl in labels)
        row_h = base + gap
        box_w = int(swatch + pad + tw + pad * 2)
        box_h = int(pad * 2 + row_h * len(labels))
        x0 = W - box_w - int(base * 0.8)
        y0 = int(base * 0.8)
        legend = Image.new("RGBA", composed.size, (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(legend)
        ldraw.rounded_rectangle(
            [x0, y0, x0 + box_w, y0 + box_h],
            radius=int(base * 0.5),
            fill=(241, 237, 223, 235),
            outline=(60, 58, 44, 220),
            width=2,
        )
        for i, ((c, _n), lbl) in enumerate(zip(entries, labels)):
            cy = y0 + pad + i * row_h + base // 2
            sx0 = x0 + pad
            sy0 = cy - swatch // 2
            color_rgb = PALETTE[int(c) % len(PALETTE)]
            ldraw.rectangle(
                [sx0, sy0, sx0 + swatch, sy0 + swatch],
                fill=color_rgb + (255,),
                outline=(60, 58, 44, 220),
                width=1,
            )
            ldraw.text(
                (sx0 + swatch + pad, cy - base // 2 - 1),
                lbl, fill=(60, 58, 44, 255), font=font,
            )
        composed = Image.alpha_composite(composed, legend)
    return composed.convert("RGB")
# ============================================================
# UI — CSS gruvbox/tierra + PT Serif
# ============================================================
HIDE_CHROME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=PT+Serif:ital,wght@0,400;0,700;1,400&display=swap');
:root {
    --bg:          #f1eddf;
    --bg-panel:    #e9e2ce;
    --fg:          #3c3a2c;
    --fg-soft:     #6b6857;
    --accent-org:  #cc844a;
    --accent-grn:  #7a8471;
    --accent-aqua: #689d6a;
    --border:      #b8ac8f;
}
#MainMenu {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
.stAlert {display: none;}
[data-testid="stStatusWidget"] {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
[data-testid="stHeader"] { background: var(--bg) !important; }
.stApp, [data-testid="stAppViewContainer"], .main, .block-container {
    background: var(--bg) !important;
    color: var(--fg) !important;
    font-family: 'PT Serif', 'Georgia', serif !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
.stApp h1 { font-size: 2.15rem !important; }
.stApp h1:first-of-type {
    border-bottom: 3px solid var(--accent-org);
    padding-bottom: 0.35rem;
    display: inline-block;
}
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg-soft) !important;
    font-style: italic;
}
[data-testid="stMetricLabel"] p { color: var(--fg-soft) !important; }
[data-testid="stDataFrame"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px;
}
[data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
.stButton > button,
.stDownloadButton > button,
[data-testid="stFileUploader"] button,
[data-testid="baseButton-secondary"] {
    background: var(--accent-grn) !important;
    color: #f1eddf !important;
    border: 1px solid var(--fg) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    min-height: 2.6rem !important;
    font-family: 'PT Serif', 'Georgia', serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    white-space: nowrap !important;
    line-height: 1.2 !important;
}
.stButton > button *,
.stDownloadButton > button *,
[data-testid="stFileUploader"] button * {
    color: #f1eddf !important;
    font-family: 'PT Serif', 'Georgia', serif !important;
}
.stButton > button:hover,
[data-testid="stFileUploader"] button:hover {
    background: var(--accent-aqua) !important;
    color: #f1eddf !important;
}
[role="radiogroup"] label {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
[role="radiogroup"] label p { color: var(--fg) !important; }
[data-baseweb="select"] > div {
    background: var(--bg-panel) !important;
    border-color: var(--border) !important;
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
[data-baseweb="select"] * {
    font-family: 'PT Serif', 'Georgia', serif !important;
    color: var(--fg) !important;
}
hr, [data-testid="stDivider"] { border-color: var(--border) !important; }
</style>
"""
def render_stats_panel(feat_row: pd.DataFrame, class_names: List[str]) -> None:
    """Estadísticas compactas: totales arriba, selectbox por rasgo abajo."""
    st.markdown("**Estadísticas**")
    total_ann = int(feat_row["total_annotations"].iloc[0])
    total_area = float(feat_row["total_annotated_area"].iloc[0])
    total_dens = float(feat_row["total_density_%"].iloc[0])
    st.metric("Anotaciones", f"{total_ann}")
    st.metric("Densidad total", f"{total_dens:.2f}%")
    st.metric("Área anotada", f"{total_area:,.0f} px²")
    st.divider()
    counts = {n: int(feat_row[f"Clase_{n}_count"].iloc[0]) for n in class_names}
    ordered = sorted(class_names, key=lambda n: -counts[n])
    trait = st.selectbox("Rasgo", ordered, key="trait_select",
                         format_func=lambda n: f"{n}  (n={counts[n]})")
    prefix = f"Clase_{trait}"
    c = int(feat_row[f"{prefix}_count"].iloc[0])
    a_sum = float(feat_row[f"{prefix}_area_sum"].iloc[0])
    a_mean = float(feat_row[f"{prefix}_area_mean"].iloc[0])
    a_std = float(feat_row[f"{prefix}_area_std"].iloc[0])
    dens = float(feat_row[f"{prefix}_density_%"].iloc[0])
    st.metric("Cuenta", f"{c}")
    st.metric("Área total", f"{a_sum:,.0f} px²")
    st.metric("Área media", f"{a_mean:,.0f} px²")
    st.metric("Área std", f"{a_std:,.0f} px²")
    st.metric("Densidad", f"{dens:.2f}%")
def render_topk_panel(title: str, ids: List[str], names: List[str],
                      probs: List[float], D_gt, D_pr,
                      base_gt, base_pr) -> None:
    st.markdown(f"#### {title}")
    sp_gt = spread_topk(ids, probs, D_gt)
    sp_pr = spread_topk(ids, probs, D_pr)
    mu_gt, sd_gt = base_gt
    mu_pr, sd_pr = base_pr
    z_gt = (mu_gt - sp_gt) / sd_gt if sd_gt > 0 and not np.isnan(sp_gt) else float("nan")
    z_pr = (mu_pr - sp_pr) / sd_pr if sd_pr > 0 and not np.isnan(sp_pr) else float("nan")
    coh_gt = 1.0 - sp_gt / mu_gt if mu_gt > 0 and not np.isnan(sp_gt) else float("nan")
    coh_pr = 1.0 - sp_pr / mu_pr if mu_pr > 0 and not np.isnan(sp_pr) else float("nan")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Confianza · matriz GT**")
        st.metric("spread_topk", f"{sp_gt:.3f}" if not np.isnan(sp_gt) else "n/a")
        st.metric("z vs aleatorio", f"{z_gt:+.2f}" if not np.isnan(z_gt) else "n/a")
        st.metric("coherence", f"{coh_gt:+.2f}" if not np.isnan(coh_gt) else "n/a")
    with cols[1]:
        st.markdown("**Confianza · matriz predicha**")
        st.metric("spread_topk", f"{sp_pr:.3f}" if not np.isnan(sp_pr) else "n/a")
        st.metric("z vs aleatorio", f"{z_pr:+.2f}" if not np.isnan(z_pr) else "n/a")
        st.metric("coherence", f"{coh_pr:+.2f}" if not np.isnan(coh_pr) else "n/a")
    df = pd.DataFrame({
        "rank": range(1, len(ids) + 1),
        "species_id": ids,
        "species_name": names,
        "prob": probs,
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
def main() -> None:
    st.set_page_config(page_title="Wood species demo", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(HIDE_CHROME_CSS, unsafe_allow_html=True)
    st.title("Identificación especies maderas")
    id_to_name, name_to_id = load_especies_map(ESPECIES_JSON)
    class_names = load_class_names(DATA_YAML)
    classifiers = load_classifiers()
    if not classifiers:
        st.stop()
    D_gt = load_distance_matrix(DIST_GT)
    D_pr = load_distance_matrix(DIST_PRED)
    base_gt = random_spread_baseline("gt", DIST_GT, TOP_K, N_RANDOM_BASELINE) \
        if D_gt is not None else (float("nan"), float("nan"))
    base_pr = random_spread_baseline("pred", DIST_PRED, TOP_K, N_RANDOM_BASELINE) \
        if D_pr is not None else (float("nan"), float("nan"))
    upload = st.file_uploader("Imagen de macroscopía",
                              type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                              label_visibility="collapsed")
    if upload is None:
        st.stop()
    image = Image.open(io.BytesIO(upload.read())).convert("RGB")
    yolo = load_yolo(YOLO_WEIGHTS)
    results = yolo.predict(source=np.asarray(image),
                           imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                           max_det=MAX_DET, verbose=False)
    result = results[0]
    feat_row = build_feature_row(result, class_names)
    overlay = draw_overlay(image, result, class_names)
    col_orig, col_ann, col_stats = st.columns([4, 4, 2])
    with col_orig:
        st.markdown("**Original**")
        st.image(image, use_container_width=True)
    with col_ann:
        st.markdown("**Anotada**")
        st.image(overlay, use_container_width=True)
    with col_stats:
        render_stats_panel(feat_row, class_names)
    st.divider()
    st.markdown("### Predicciones")
    model_choice = st.radio("Modelo",
                            options=list(classifiers.keys()),
                            horizontal=True, index=0,
                            label_visibility="collapsed")
    pipe = classifiers[model_choice]
    X = align_features_to_pipeline(feat_row, pipe)
    top_names, top_probs = topk_from_pipeline(pipe, X, TOP_K)
    top_ids = [to_species_id(n, name_to_id) or "?" for n in top_names]
    top_latin = [id_to_name.get(i, n) for i, n in zip(top_ids, top_names)]
    render_topk_panel(f"Top-{TOP_K} — modelo {model_choice.upper()}",
                      top_ids, top_latin, top_probs,
                      D_gt, D_pr, base_gt, base_pr)
if __name__ == "__main__":
    main()