# configs/

| Fichero | Estado |
|---|---|
| `data.yaml` | **Ejecutable.** Es el mapa de las 14 clases de rasgo que consume Ultralytics. Cópialo a `data/raw_selected/data.yaml` junto al dataset. |
| `segmenter_semloss.yaml` | **Documentación.** Recoge los hiperparámetros de las ejecuciones publicadas. |
| `classifier.yaml` | **Documentación.** Recoge features, esquema de validación, hiperparámetros y el catálogo de 96 especies. |

Los dos últimos **no los lee ningún script todavía**: la configuración sigue viviendo en las
constantes de cabecera de cada fichero de `src/`. Están aquí para que cualquiera pueda
auditar con qué parámetros se produjeron los resultados de `results/`, y como punto de
partida del refactor pendiente. Si cambias un valor en un script, actualiza también estos
ficheros o dejarán de describir la realidad.

Aviso sobre `data.yaml`: los nombres de clase son **códigos del catálogo de rasgos**
anatómicos, no nombres legibles. La clase de índice 11 se llama `'10'` y en las leyendas de
las figuras se lee como "1.0". Ver `docs/DATA_CARD.md`.
