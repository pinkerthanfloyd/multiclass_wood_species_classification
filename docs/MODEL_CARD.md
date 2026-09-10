# Tarjeta de modelo

## Sistema

Dos modelos encadenados. **No evalúes el segundo sin declarar de dónde vino la entrada.**

| | Etapa 1 | Etapa 2 |
|---|---|---|
| Tarea | Segmentación de instancias de rasgos anatómicos | Clasificación de especie |
| Arquitectura | YOLO11n-seg (Ultralytics) + pérdida semántica propia | Random Forest 500 árboles / SVM-RBF (C=10), con imputación por mediana y estandarizado en la SVM |
| Entrada | Imagen de macroscopía, 1024×1024 | Vector tabular por imagen: conteo, área suma/media/desviación y densidad % por cada una de las 14 clases, más agregados |
| Salida | Polígonos + clase + confianza | Distribución sobre 96 especies → top-k |
| Entrenamiento | 100-250 épocas, batch 4-8, AMP, `seed=42` | Validación cruzada estratificada de 5 pliegues, `class_weight="balanced"` |

## Uso previsto

**Apoyo al criterio experto**, no sustitución. La salida útil del sistema es una lista corta
de candidatos que un anatomista de la madera arbitra: por eso la métrica de cabecera es el
hit-rate top-k y no la exactitud top-1.

## Fuera de alcance

- Certificación legal de especie, peritaje o cualquier uso con consecuencia jurídica o
  aduanera (p. ej. verificación CITES). El sistema no está validado para eso.
- Especies fuera de las 96 del catálogo: no puede acertarlas, y devolverá igualmente una lista
  con probabilidades. La ausencia de un mecanismo de rechazo ("ninguna de las conocidas") es
  una limitación importante para uso en campo.
- Imágenes con preparación, aumento o modalidad distintas de las del corpus de entrenamiento.

## Rendimiento

Cifras completas y su contexto en [`../results/RESULTS.md`](../results/RESULTS.md).

| Métrica | Valor | Sobre qué |
|---|---|---|
| mAP50 caja / máscara | 0,429 / 0,332 | Segmentador, split1 de test |
| F1 macro de rasgos | 0,44 @ conf 0,391 | Segmentador, 14 clases |
| Accuracy CV (RF / SVM) | 0,965 / 0,944 | Clasificador, imágenes del propio corpus |
| **Top-1 en lote externo** | **0,28** (n=25) | Clasificador, imágenes no vistas |

La cifra que describe el comportamiento esperable con material nuevo es la última. Las de
validación cruzada están infladas por dos fugas documentadas (features generadas por un
segmentador que vio esas imágenes; varias imágenes por espécimen repartidas entre pliegues).

## Cómo NO leer los resultados

1. El F1 de 0,44 es de **rasgos anatómicos**, no de especies.
2. mAP está sesgada a la baja por la anotación selectiva: es cota inferior.
3. El umbral de confianza 0,391 se eligió maximizando la propia curva reportada.
4. Las clases de rasgo con soporte pequeño producen F1 que no son medidas; mira siempre el
   `support`.
5. `spread`, `coherence` y `severity` son construcciones propias con baseline aleatorio, no
   métricas estándar comparables con la literatura.

## Factores que degradan el rendimiento

Especies con pocas imágenes; cortes con preparación deficiente; rasgos pequeños y dispersos
(las clases con F1 cercano a cero); y cualquier cambio de escala o iluminación respecto al
corpus. El sistema no detecta ninguna de estas condiciones por sí mismo.

## Consideraciones éticas y de impacto

El caso de uso motivador —control del comercio de maderas y apoyo a la identificación de
especies protegidas— es sensible: **un falso negativo puede facilitar comercio ilegal y un
falso positivo puede bloquear comercio legítimo**. Con un top-1 del 28 % sobre material nuevo,
el sistema no es apto para decidir por sí solo, y presentarlo como tal sería incorrecto. Su
valor está en reducir el espacio de búsqueda del experto.

## Mantenimiento

Al ampliar el catálogo de especies hay que actualizar `TARGET_IDS`, reentrenar ambas etapas y
recalcular los priors por especie y la matriz de exclusividad cacheada. Los resultados
publicados dejan de ser válidos en cuanto cambie el catálogo.
