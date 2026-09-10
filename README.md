# Identificación multiclase de especies de madera por macroscopía

Pipeline reproducible de **dos etapas** que estima la especie de una muestra de madera a
partir de una imagen de macroscopía:

1. **Segmentador de rasgos anatómicos** (YOLO11n-seg). Sobre cada imagen produce una lista
   de instancias anotadas —vasos, radios, parénquima, contenidos vesiculares…— sobre un
   catálogo de 14 clases de rasgo. Se entrena con una **pérdida semántica** (`sem_loss`)
   añadida a la pérdida nativa de Ultralytics: términos de distribución por imagen, *priors*
   por especie, exclusividad espacial entre clases descubierta del *ground truth* y un
   término de énfasis diagnóstico por rareza morfológica.
2. **Clasificador de especie** (Random Forest / SVM-RBF). Toma un vector tabular por imagen
   derivado de esas anotaciones (conteo, área suma/media/desviación y densidad porcentual
   por clase, más agregados) y devuelve un **top-k** de especies con probabilidad.

Alrededor del pipeline hay un bloque de análisis: distancias inter-especie sobre las firmas
de anotación (presencia, área-fracción, morfología, co-ocurrencia y proximidad espacial),
comparación diferencial entre el mapa derivado del GT y el derivado de las predicciones, y
métricas de confianza y de severidad de error.

> Trabajo de Fin de Grado. El código se publica bajo licencia MIT; **el dataset de imágenes
> no se distribuye** (ver [`docs/DATA_CARD.md`](docs/DATA_CARD.md)).

---

## Qué hay y qué no hay en este repositorio

| Incluido | No incluido |
|---|---|
| Todo el código del pipeline (`src/`) y la demo (`app/`) | Las imágenes de macroscopía y sus etiquetas |
| Configuración de entorno y de las ejecuciones reportadas | Pesos entrenados (`*.pt`, `*.joblib`) |
| Resultados ligeros y reproducibles (`results/`) | Directorios de ejecución completos (`runs_*/`) |
| Documentación de datos, modelo y despliegue (`docs/`) | La memoria del TFG |

Los artefactos excluidos se piden a la persona responsable del proyecto; su procedencia y
condiciones de uso están en [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Estructura

```
.
├── configs/          Configuración de las ejecuciones reportadas (data.yaml + parámetros)
├── src/              Código del pipeline; los módulos conservan su nombre original
│   └── prepare/      CLIs autónomos de preparación del dataset
├── app/              Demo local en Streamlit (imagen → anotaciones → top-k)
├── data/            (vacío en git) Dónde se espera el dataset — ver data/README.md
├── results/          Resultados publicados + RESULTS.md con la trazabilidad de cada cifra
├── docs/             Instalación, pipeline, tarjeta de datos, tarjeta de modelo, despliegue
├── environment.yml   Entorno conda (Python 3.10, PyTorch CUDA 12.1, Ultralytics)
└── pyproject.toml    Instalación editable: pone src/ en el path
```

`src/` es **plano a propósito**. Los módulos se importan entre sí por su nombre
(`from train_yolo_seg_baseline_csvs import ...`), tal y como se escribieron y se ejecutaron
para producir los resultados publicados. Instalar el paquete en modo editable pone `src/` en
el `PYTHONPATH` y todos esos imports siguen resolviendo sin tocar una línea de código.

## Instalación

```bash
conda env create -f environment.yml     # o: bash setup_env.sh
conda activate maderas
pip install -e .                        # deja src/ importable desde cualquier sitio
```

Detalle y resolución de problemas en [`docs/INSTALACION.md`](docs/INSTALACION.md). Si el
entorno queda inconsistente, `LIMPIAR_Y_REINSTALAR.sh` lo borra y lo recrea desde cero.

## Ejecución del pipeline

Todos los scripts se lanzan **desde la raíz del repositorio** y llevan sus rutas en las
constantes de cabecera (ver [`docs/PIPELINE.md`](docs/PIPELINE.md), que documenta paso a
paso qué espera cada uno y en qué orden).

```bash
python src/prepare/coco_to_yolo_seg.py          # 1. COCO → formato YOLO-seg
python src/prepare/make_splits.py               # 2. splits train/val/test por CSV
python src/prepare/build_relative_vectors_from_yolo_dataset.py   # 3. priors por especie
python src/train_yolo_seg_sem_loss_species_diag.py               # 4. segmentador
python src/generate_prediction_coco_json.py --model <best.pt>    # 5. COCO de predicciones
python src/species_classifier.py                                 # 6. clasificador RF/SVM
python src/species_annotation_proximity.py --coco-json <coco> --plot   # 7. proximidad
python src/species_predict.py                                    # 8. evaluación top-k
streamlit run app/streamlit_app.py                               # demo
```

## Resultados

Cifras de las ejecuciones publicadas en `results/`. **Lee
[`results/RESULTS.md`](results/RESULTS.md) antes de citar cualquiera de ellas**: explica de
qué conjunto sale cada número y por qué dos de ellos no son comparables entre sí.

| Etapa | Métrica | Valor |
|---|---|---|
| Segmentador (baseline bs4, imgsz 1024, 100 épocas, split1) | mAP50 caja / máscara | 0,429 / 0,332 |
| Segmentador | mAP50-95 caja / máscara | 0,282 / 0,172 |
| Segmentador | F1 macro (14 rasgos, conf. 0,391) | 0,44 |
| Clasificador, CV 5-fold sobre el dataset anotado (n=1061, 96 especies) | Accuracy RF / SVM | 0,965 ± 0,006 / 0,944 ± 0,011 |
| Clasificador, CV 5-fold | F1 macro RF / SVM | 0,965 ± 0,008 / 0,936 ± 0,012 |
| Clasificador, **lote externo** (n=25 evaluables) | Top-1 | 0,28 |
| Mapa inter-especie GT vs. predicho (95 especies) | Mantel *r* Pearson | 0,928 (p ≈ 0,001) |

La diferencia entre el 0,965 de validación cruzada y el 0,28 del lote externo **no es ruido**:
mide cuánto de la validación cruzada se apoya en imágenes que el segmentador ya había visto
durante su entrenamiento. Está explicada en `results/RESULTS.md` §3.

## Cómo leer estos números

Cinco advertencias que condicionan cualquier lectura de los resultados; el desarrollo está en
[`results/RESULTS.md`](results/RESULTS.md) y en [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

1. **Las 14 clases del segmentador son rasgos anatómicos, no especies.** El F1 de 0,44 es del
   segmentador sobre rasgos; no es la precisión del sistema identificando especies.
2. **Anotación selectiva.** El experto anota sólo los rasgos que distinguen una especie de
   otra, no todos los visibles. Un rasgo detectado correctamente pero no anotado cuenta como
   falso positivo: mAP es una **cota inferior**, no una medida limpia de calidad.
3. **Validación cruzada optimista.** Las features salen de predicciones del segmentador sobre
   imágenes con las que se entrenó, y varias imágenes proceden del mismo espécimen físico.
4. **El lote externo es pequeño** (26 imágenes, 25 evaluables): intervalo de confianza al 95 %
   de ±18 puntos aproximadamente.
5. **`spread`, `coherence` y `severity` son construcciones propias**, no métricas estándar;
   se publican como diagnóstico exploratorio con su baseline aleatorio.

## Estado conocido del código

Cosas pendientes, documentadas para que nadie las descubra por sorpresa:

- `src/species_classifier.py` **no selecciona el mejor modelo**: calcula el F1 macro de RF y
  SVM y a continuación sobrescribe la selección con SVM de forma incondicional. En la
  ejecución publicada eso guardó la SVM (F1 0,936) etiquetada con el F1 del RF (0,965). Ver
  `results/classifier/classification_report.txt`, líneas 113-116.
- Las rutas y los hiperparámetros viven en constantes de cabecera, no en `configs/`. Los
  ficheros de `configs/` **documentan** las ejecuciones reportadas; todavía no los lee nadie.
- `src/train_yolo_seg_baseline.py` construye un `dataset.yaml` con `train`, `val` y `test`
  apuntando los tres al directorio de entrenamiento. Además tiene comentado su propio
  `import YOLO`. Es un script de arranque histórico: los resultados publicados provienen de
  `train_yolo_seg_baseline_csvs.py`, que sí respeta los splits. No uses el primero para
  medir nada.
- Quedan variantes redundantes del mismo paso (`visualize_species_proximity.py` y su versión
  `_noargs`, `compute_proximity.py` frente a `compute_species_dissimilarity.py`). El fichero
  canónico de cada etapa está marcado en `docs/PIPELINE.md`.

## Licencia y cita

Código bajo [licencia MIT](LICENSE). El dataset **no** está cubierto por esa licencia. Si
usas este trabajo, cita según [`CITATION.cff`](CITATION.cff).
