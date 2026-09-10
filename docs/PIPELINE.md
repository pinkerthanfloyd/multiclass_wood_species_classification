# Pipeline: qué hace cada script y en qué orden

Todos los scripts se lanzan **desde la raíz del repositorio**. Los que llevan rutas en
constantes de cabecera lo indican abajo; edítalas o coloca los datos donde las esperan.

## Fichero canónico por etapa

| Etapa | Usa esto | Existen también |
|---|---|---|
| COCO → YOLO-seg | `src/prepare/coco_to_yolo_seg.py` | — |
| Limpieza de polígonos | `src/prepare/filter_small_polygons.py` | — |
| Splits | `src/prepare/make_splits.py` | — |
| Priors por especie | `src/prepare/build_relative_vectors_from_yolo_dataset.py` | `..._from_coco_json.py` (misma salida desde COCO) |
| Entrenamiento del segmentador | `src/train_yolo_seg_sem_loss_species_diag.py` | `_overlap.py`, `train_yolo_seg_sem_loss.py`, `train_yolo_seg_partial_anno.py` |
| Configuración base del segmentador | `src/train_yolo_seg_baseline_csvs.py` | `train_yolo_seg_baseline.py` (histórico, **no medir con él**) |
| COCO de predicciones | `src/generate_prediction_coco_json.py` | — |
| Clasificador de especie | `src/species_classifier.py` | — |
| Inferencia y top-k | `src/species_predict.py` | `predict_with_severity.py` (añade severidad) |
| Proximidad entre especies | `src/species_annotation_proximity.py` | `compute_species_dissimilarity.py`, `src/prepare/compute_proximity.py` (versiones previas, más simples) |
| Comparación GT vs. predicho | `src/differential_analysis.py` | — |
| Figuras de proximidad | `src/visualize_species_proximity.py` | `_noargs.py` (mismas figuras con rutas fijas) |

La variante **diag** del segmentador es la definitiva: con `diag_emphasis=False` colapsa
exactamente a la variante **overlap**, que se conserva porque es la que quedó publicada como
línea base de la pérdida semántica.

---

## 1. Preparación del dataset

```bash
python src/prepare/coco_to_yolo_seg.py
python src/prepare/filter_small_polygons.py
python src/prepare/make_splits.py
python src/prepare/build_relative_vectors_from_yolo_dataset.py
```

Salida esperada: estructura YOLO (`images/`, `labels/`), los CSV `train.csv` / `val.csv` /
`test.csv` dentro de la carpeta de split, y `outputs/relative_feature_vectors.csv` con los
priors por especie que consume la pérdida semántica.

`src/prepare/renamer.py` es auxiliar: renombra imágenes añadiendo el nombre latino a partir
del id de especie del prefijo. Corre en *dry-run* salvo que pases `--apply`.

## 2. Entrenamiento del segmentador

```bash
python src/train_yolo_seg_sem_loss_species_diag.py
```

**No es autocontenido.** Importa de `train_yolo_seg_baseline_csvs.py` la configuración
(`DATASET_ROOT`, `DATA_YAML_PATH`, `SPLIT_FOLDER`, `EPOCHS`, `PRETRAINED`) y la evaluación
personalizada (`evaluate_on_test_split`). Necesita, además del dataset: `data.yaml`, la
carpeta de splits y `outputs/relative_feature_vectors.csv`.

La lista `EXPERIMENTS` está en la cabecera del script (sobre las líneas 110-120): define
`batch`, `imgsz`, si se activa la exclusividad espacial por especie (`iaway_bias`) y, en la
variante diag, el énfasis diagnóstico (`diag_emphasis`). Comenta las que no quieras lanzar.

Parámetros de la pérdida semántica en las ejecuciones reportadas:

| Parámetro | Valor | Qué hace |
|---|---|---|
| `T_warm` | 30 épocas | La pérdida se calcula pero no contribuye al gradiente |
| `T_ramp` | 80 épocas | Rampa lineal de λ hasta el máximo |
| `lambda_max` | 0,10 | Peso final de la pérdida semántica |
| `imgsz` | 1024 | Resolución de las variantes reportadas |
| `retina_masks` | `True` en validación | Máscaras a resolución nativa |
| `K_min` | 3 | Observaciones mínimas por especie para usar la matriz de exclusividad |

Salidas: `runs_wood_feature_sem_loss_diag/<split>/` con `best.pt`, CSVs de progreso,
`sem_loss_log.csv` por lote y el JSON de la matriz de solapamientos descubierta.

### Avisos operativos

- **La matriz de exclusividad por especie se recalcula durante el warmup y se cachea en un
  JSON.** Si cambias de split, bórralo o seguirá reflejando el dataset anterior.
- **AMP está activo.** La pérdida semántica se calcula bajo `autocast=False` con casts
  explícitos a `float32`. No lo toques sin entender por qué está así.
- **El checkpointing es reanudable**: si matas el proceso, al relanzar detecta `last.pt` y
  continúa con optimizador, scheduler, RNG y buffers de la pérdida semántica intactos.
- **No mezcles `best.pt` entre las variantes overlap y diag**: los buffers cacheados no son
  compatibles.
- **OOM con `imgsz=1024` y `batch=8`**: baja a `batch=4`. La exclusividad usa `bmm`
  precisamente para evitarlo, pero con GPUs pequeñas sigue siendo apretado.
- **Versión de Ultralytics**: el parser de predicciones tolera `dict` (8.4.45+) y tupla
  (anteriores), pero versiones muy nuevas pueden romperlo.

## 3. COCO de anotaciones predichas

```bash
python src/generate_prediction_coco_json.py \
    --model runs_wood_feature_sem_loss_diag/split1/<exp>/weights/best.pt \
    --dataset-root data/raw_selected \
    --data-yaml data/raw_selected/data.yaml \
    --output predicted_instances.json
```

Sin `--split-dir` recorre recursivamente todas las imágenes bajo `--dataset-root`. Con
`--split-dir` usa `train.csv` + `val.csv` + `test.csv`. El JSON resultante replica el esquema
de `instances_selected.json` y añade un campo `score` por anotación.

Procesa en bloques (`--chunk-size`, 50 por defecto) y vacía la caché de GPU entre bloques;
`retina_masks` está desactivado aquí a propósito, porque las máscaras a resolución de modelo
bastan para extraer polígonos y consumen ~4× menos VRAM.

## 4. Clasificador de especie

```bash
python src/species_classifier.py
```

Consume, en rutas relativas fijadas en cabecera:

- `models/inferences/especies.json` — mapa id → nombre latino
- `models/inferences/sem_loss_stratified` — COCO con las anotaciones predichas
- `TARGET_IDS` — la lista de especies consideradas, también en cabecera

Produce `species_inferences/` (features por imagen, predicciones *out-of-fold*, reporte y
galería de fallos) y los dos `.joblib` en `models/species_classifier/`.

> **Bug conocido**: el bloque de selección de modelo sobrescribe la elección con SVM de forma
> incondicional. Ver `results/RESULTS.md` §2 antes de usar el modelo guardado.

## 5. Evaluación top-k

```bash
python src/species_predict.py
```

Es **inferencia**, no entrenamiento: carga los `.joblib` ya existentes. Necesita un COCO real
de anotaciones predichas para el lote a evaluar (valida que lo sea y aborta con mensaje claro
si le pasas un CSV renombrado a `.json`) y un CSV de etiquetas verdaderas con columnas
`image` y `true_species`.

Produce `species_predictions.csv` con top-k, `rank_true`, `prob_true` y
`true_in_catalog_<modelo>` —que separa "el modelo falló" de "acertar era imposible"—, un
`prediction_summary.json` con hit-rate@1/@3/@5 y un bloque LaTeX listo para la memoria.

`src/predict_with_severity.py` hace lo mismo ponderando cada error por la disimilaridad entre
la especie real y la predicha.

## 6. Análisis de proximidad

```bash
python src/species_annotation_proximity.py --coco-json <coco> \
       --especies-json inferences/especies.json --output-dir outputs/species_proximity --plot
python src/differential_analysis.py
python src/visualize_species_proximity.py
```

Ejecuta el primero dos veces, una sobre el COCO del *ground truth* y otra sobre el de
predicciones, a directorios distintos; `differential_analysis.py` compara ambos (Mantel,
matriz delta, Procrustes sobre MDS 2D) y espera encontrarlos en `outputs/species_proximity_gt`
y `outputs/species_proximity`.

Pesos por bloque con `--w-b1` … `--w-b4` (uniforme por defecto; `--w-b4 0` desactiva el
bloque espacial). `--min-images` descarta especies con muy pocas imágenes: por debajo de 2 la
dispersión intra-especie no se puede estimar.

## 7. Demo

```bash
streamlit run app/streamlit_app.py
```

Sube una imagen, la pasa por el segmentador, construye el vector de features con el mismo
esquema que el clasificador, y muestra el top-5 con las métricas de confianza calculadas
sobre las dos matrices de distancia (GT y predicha) lado a lado. Las rutas a pesos y matrices
están en la sección CONFIG del fichero.
