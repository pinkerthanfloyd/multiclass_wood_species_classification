# Resultados publicados y su trazabilidad

Cada cifra de este repositorio sale de un fichero concreto producido por un script concreto.
Esta tabla es el índice; las secciones siguientes explican cómo debe leerse cada bloque.

| Resultado | Fichero en `results/` | Script que lo produce |
|---|---|---|
| Curvas F1/PR del segmentador, matriz de confusión, curva de entrenamiento | `segmentation/` | Ultralytics, dentro de `src/train_yolo_seg_baseline_csvs.py` |
| Configuración exacta de esa ejecución | `segmentation/args.yaml` | Ultralytics |
| Evolución de la pérdida semántica y de λ(t) | `segmentation/sem_loss_log_evolucion.png` | `src/plot_sem_loss.py` |
| Distancias inter-especie sobre anotaciones **predichas** | `proximity/` | `src/species_annotation_proximity.py` |
| Distancias inter-especie sobre anotaciones **GT** | `proximity_gt/` | `src/species_annotation_proximity.py` |
| Comparación entre ambos mapas (Mantel, delta, Procrustes) | `diff_analysis/` | `src/differential_analysis.py` |
| Validación cruzada y predicciones *out-of-fold* del clasificador | `classifier/` | `src/species_classifier.py` |
| Lote externo de evaluación | `inference_32/` | `src/species_predict.py` |

No se publican aquí los ficheros pesados que estos scripts también generan
(`per_annotation_features.csv`, `per_image_spatial_relations.csv`, el COCO de predicciones
completo, los directorios `runs_*`). Se regeneran ejecutando el pipeline.

---

## 1. Segmentador de rasgos anatómicos

Ejecución publicada: `yolo11n-seg.pt`, `imgsz=1024`, `batch=4`, 100 épocas, `seed=42`,
`cos_lr`, `close_mosaic=10`, AMP, split1. Config completa en `segmentation/args.yaml`.

| Métrica | Caja | Máscara |
|---|---|---|
| Precisión | 0,619 | 0,552 |
| Recall | 0,443 | 0,374 |
| mAP50 | 0,429 | 0,332 |
| mAP50-95 | 0,282 | 0,172 |

(última fila de `segmentation/results.csv`, época 100)

**Cómo se lee.** Tres cosas antes de citar estos números:

- **Las 14 clases son rasgos anatómicos**, identificados por su código de catálogo
  (`8`, `56`, `58`, `79`, `81`, `82`, `83`, `84`, `85`, `86`, `89`, `10`, `V1`, `radio`), no
  especies. La curva `BoxF1_curve.png` que resume "all classes 0.44 at 0.391" es el F1 macro
  del segmentador sobre esos 14 rasgos al umbral de confianza que lo maximiza. Quien lea 0,44
  como "el sistema acierta la especie el 44 % de las veces" está leyendo mal.
- **El umbral 0,391 es el óptimo de esa misma curva.** Si se usa como punto de operación en
  producción, hay que congelarlo con el conjunto de validación y volver a medir en test; tal
  como está, el punto de operación se ha ajustado sobre el conjunto que se reporta.
- **La dispersión entre clases es enorme.** El mejor rasgo ronda F1 0,82 y al menos uno
  (`84`) se queda plano en cero. Un F1 por clase sin su `support` al lado no significa nada:
  una clase con tres instancias anotadas produce un número que no es una medida. La columna
  de soportes está en `custom_global_class_metrics.csv` dentro del directorio de ejecución.
- **Anotación selectiva.** El experto anota únicamente los rasgos que discriminan; los rasgos
  presentes pero no anotados penalizan la métrica cuando el modelo los detecta. mAP y
  precisión son por tanto cotas inferiores. El F1 propio con *matching* IoU ≥ 0,50 que calcula
  `train_yolo_seg_baseline_csvs.py` es el indicador interno más fiable.

**Pendiente.** La comparación entre las cuatro configuraciones de la pérdida semántica
(baseline / spexcl / diag / spexcl+diag) medida *aguas abajo* en hit-rate top-k no está en este
repositorio. Hasta que exista esa tabla, la afirmación "la pérdida semántica mejora las
anotaciones aunque no se vea en mAP" es una hipótesis de trabajo, no un resultado.

---

## 2. Clasificador de especie — validación cruzada

Conjunto: 1061 imágenes, 96 especies, entre 4 y 23 imágenes por especie (mediana 11).
Validación cruzada estratificada de 5 pliegues.

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Random Forest | 0,9651 ± 0,0064 | 0,9646 ± 0,0083 |
| SVM-RBF | 0,9444 ± 0,0105 | 0,9364 ± 0,0122 |

Predicciones *out-of-fold* del modelo guardado: 1002 aciertos de 1061 (94,4 %),
`classifier/predictions.csv`.

**Cómo se lee.** Estas cifras son **optimistas por construcción**, por dos motivos
independientes y acumulativos:

1. **Fuga entre etapas.** El vector de features de cada imagen procede de las anotaciones que
   predijo el segmentador, y el segmentador se entrenó con parte de esas mismas imágenes.
   Sobre ellas las anotaciones son mejores de lo que serán nunca sobre madera nueva, así que
   el clasificador recibe una entrada de calidad irrepetible.
2. **Fuga entre pliegues.** La estratificación es por imagen, no por espécimen. Varias
   imágenes son cortes distintos de la misma pieza física (`0003-1`, `0003-2`, …). Si dos
   cortes de la misma tabla caen en pliegues distintos, el acierto mide parecido físico, no
   capacidad de generalizar. El arreglo es agrupar por espécimen (`GroupKFold`).

**Aviso sobre el modelo guardado.** El log registra literalmente:

```
>> Mejor modelo por F1 macro: Random Forest (0.9646)
>> Mejor modelo por F1 macro: SVM (0.9646)
Modelo guardado en: models/species_classifier/species_model_svm.joblib
```

El Random Forest ganó (0,9646 frente a 0,9364) pero el script sobrescribe la selección con la
SVM sin condición y le adjunta el F1 del RF. El modelo desplegado es el peor de los dos y la
etiqueta del número no corresponde al modelo. Hasta que se corrija ese bloque, no debe
afirmarse que el sistema "selecciona el mejor modelo por F1 macro".

---

## 3. Lote externo de evaluación — la cifra honesta

26 imágenes que no forman parte del conjunto de entrenamiento (`inference_32/`). De ellas,
**25 son evaluables**: la especie verdadera de `18.jpg` (*Pterygota bequaertii*, id 0540) no
pertenece al catálogo de 96 especies, así que acertarla era imposible y no cuenta como error
del modelo.

| | |
|---|---|
| Imágenes del lote | 26 |
| Evaluables (especie dentro del catálogo) | 25 |
| Aciertos top-1 | 7 → **28,0 %** |

**Cómo se lee.** Esta es la estimación más cercana al uso real, y es la que hay que citar
junto —nunca en lugar— del 0,965 de la sección 2. La caída de 0,96 a 0,28 es la medida
empírica de las dos fugas descritas arriba: es lo que cuesta pasar de imágenes que el
segmentador conocía a imágenes que no.

Tres cautelas sobre el 28 %:

- **n = 25.** El intervalo de confianza al 95 % para una proporción de 0,28 con esa muestra es
  aproximadamente [0,12; 0,49]. Es una cifra indicativa, no un punto.
- **Sólo mide top-1.** El proyecto está diseñado para entregar un top-5 al experto: el
  hit-rate@3 y @5 sobre este mismo lote los calcula `src/species_predict.py` y son la métrica
  de cabecera pendiente de publicar aquí.
- **El CSV `32_images_pred.csv` conserva un artefacto conocido**: la columna `species_name`
  muestra valores tipo `Especie_Desconocida_1.jpg` porque el lote no sigue la convención
  `<id>-<n>.jpg` de la que el script derivaba la especie. Es informativa y no se usa en la
  evaluación —la verdad sale siempre de `32_images_labels.csv`—, pero no debe leerse como
  una predicción. `src/species_predict.py` ya deja esa columna vacía en lugar de inventarla.

---

## 4. Mapa de proximidad entre especies

`proximity/` y `proximity_gt/` contienen la misma familia de artefactos calculada sobre dos
conjuntos de anotaciones: las predichas por el segmentador y las del *ground truth*. La
distancia combina cuatro bloques: B1 presencia y área-fracción, B2 morfología (área
logarítmica, esfericidad, aspecto, elongación), B3 co-ocurrencia Jaccard entre clases y B4
proximidad y solapamiento espacial con umbral adaptativo por imagen.

Correlación de Mantel entre ambos mapas, 95 especies comunes, 1000 permutaciones
(`diff_analysis/mantel.json`):

| Bloque | Pearson *r* | Spearman *ρ* | p |
|---|---|---|---|
| Combinado | 0,928 | 0,922 | 0,001 |
| B1 presencia + área | 0,906 | 0,910 | 0,001 |
| B2 morfología | 0,849 | 0,837 | 0,001 |
| B3 co-ocurrencia | 0,917 | 0,911 | 0,001 |
| B4 espacial | 0,901 | 0,895 | 0,001 |

**Cómo se lee.** El mapa que el segmentador induce reproduce el del GT con una correlación
alta y significativa, y el bloque morfológico es el que más se distorsiona —esperable: la
forma del polígono predicho es más frágil que su mera presencia. Esto justifica usar el mapa
derivado de predicciones como diagnóstico cuando no hay GT disponible.

**Cautela de circularidad.** Para *evaluar* errores del clasificador (severidad, distancia a
la verdad) hay que usar la matriz **GT**. Usar la matriz derivada de las predicciones del
mismo modelo que se está juzgando es medirlo con su propia vara. La matriz de predicciones
sirve para diagnóstico y para el caso de uso sin GT, no para puntuar.

El p-valor mínimo alcanzable con 1000 permutaciones es 1/1001 ≈ 0,000999: "p = 0,001"
significa "por debajo de la resolución del test", no un valor medido.

---

## 5. Métricas propias

`spread`, `spread_z`, `coherence` (en `src/predictions_confidence.py`) y `severity` (en
`src/predict_with_severity.py`) **no son métricas estándar**. Se definen en el docstring de
cada script y se acompañan siempre de su baseline aleatorio: el spread esperado tomando k
especies al azar del catálogo, y la distancia esperada a la verdad bajo muestreo aleatorio.
Sin ese baseline los valores absolutos no son interpretables. Preséntalas como diagnóstico
exploratorio.

---

## 6. Reproducibilidad

Las semillas están fijadas (`seed=42` en el segmentador y en los pipelines de scikit-learn),
pero AMP y las rutinas no deterministas de cuDNN hacen que una repetición no dé bit a bit lo
mismo. Para reproducir hace falta, además del código: el dataset, `data.yaml`, los CSV de
split, `outputs/relative_feature_vectors.csv` (priors por especie de la pérdida semántica) y
las versiones declaradas en `environment.yml` — en particular `ultralytics >= 8.4.45`, cuyo
formato de salida asume el envoltorio de la pérdida semántica.
