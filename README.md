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
> no está disponible para distribución** (ver [`docs/DATA_CARD.md`](docs/DATA_CARD.md) o consultar escribiendo un correo a ig.diaz@alumnos.upm.es).


Los artefactos excluidos se piden a la persona responsable del proyecto; su procedencia y
condiciones de uso están en [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Estructura de contenidos

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

## Instalación

```bash
conda env create -f environment.yml     # o: bash setup_env.sh
conda activate maderas
pip install -e .                        # deja src/ importable desde cualquier sitio
```

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

## Variantes del segmentador

Los cinco scripts de entrenamiento son una misma línea evolutiva: cada uno parte del
anterior y cambia una idea. Todos comparten configuración, *splits* y evaluación, que viven
en `train_yolo_seg_baseline_csvs.py`.

| Script | Qué añade respecto al anterior |
|---|---|
| `train_yolo_seg_baseline_csvs.py` | Línea base: pérdida nativa de Ultralytics. Aporta además la configuración y la evaluación propia (matching IoU ≥ 0,50, desglose por imagen y por especie) que usan todos los demás. |
| `train_yolo_seg_sem_loss.py` | Introduce la **pérdida semántica**: un término por imagen (KL sobre conteos + L1 sobre áreas) y otro por especie (*priors* del CSV de vectores relativos), con *warmup* y rampa de λ. Incluye reglas de anatomía cableadas a mano (presencia obligatoria de `V1` y `radio`, contención de `56`/`58` dentro de `V1`). |
| `..._sem_loss_species_overlap.py` | **Elimina esas reglas cableadas** —eran incorrectas bajo anotación selectiva— y las sustituye por una **exclusividad espacial por especie descubierta de los datos**: para cada especie se construye del *ground truth* qué pares de clases llegan a solaparse alguna vez, y sólo se penaliza la co-activación de los pares nunca observados. |
| `..._sem_loss_species_diag.py` | Añade el **énfasis diagnóstico**: pondera cada par (especie, rasgo) por lo atípica que es el área de ese rasgo en esa especie frente al resto de especies, de modo que el modelo no se acomode en los rasgos comunes. Con `diag_emphasis=False` colapsa exactamente a la variante anterior. **Es la variante canónica de los resultados publicados.** |
| `train_yolo_seg_partial_anno.py` | Reformulación posterior, **experimental**, que lleva el paradigma de anotación parcial hasta el final: la pérdida por imagen pasa a ser **asimétrica** (sólo castiga la sub-predicción, porque una detección extra puede ser un rasgo real sin anotar), desactiva la exclusividad —su supuesto se invierte cuando la ausencia de un par sólo significa que el anotador no lo consideró distintivo— y rebaja el peso del término `cls`, que es el que empuja al modelo a llamar «fondo» a lo no anotado. |

En resumen: **`sem_loss`** añade la firma anatómica, **`overlap`** deja de imponerla a mano y
la aprende del GT, **`diag`** insiste en lo que distingue a cada especie, y **`partial_anno`**
deja de penalizar aquello que el experto simplemente no anotó. El detalle operativo de cada
uno está en [`docs/PIPELINE.md`](docs/PIPELINE.md).

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
