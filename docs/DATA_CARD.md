# Tarjeta de datos

> **Pendiente antes de publicar**: los campos marcados con `<…>` requieren confirmación de la
> entidad propietaria del corpus. No hagas público este fichero con los marcadores sin
> rellenar.

## Resumen

| | |
|---|---|
| Dominio | Macroscopía de madera (superficie transversal) |
| Unidad | Imagen de una muestra; varias imágenes por espécimen físico |
| Tamaño del corpus anotado | 1061 imágenes con anotaciones utilizables |
| Especies | 96 en el catálogo del clasificador (`TARGET_IDS`) |
| Imágenes por especie | 4 mínimo, 11 mediana, 23 máximo |
| Clases de anotación | 14 rasgos anatómicos |
| Formato de anotación | COCO *instance segmentation* (polígonos), exportado desde CVAT |
| Convención de nombre | `<id_especie de 4 dígitos>-<índice>.jpg`, p. ej. `0003-1.jpg` |
| Propiedad | `<entidad propietaria>` — **no se redistribuye en este repositorio** |
| Licencia de los datos | `<condiciones de uso acordadas>` — distinta de la licencia MIT del código |

## Qué NO está en este repositorio

Ni las imágenes, ni las etiquetas YOLO, ni los COCO completos, ni los pesos entrenados. El
`.gitignore` los bloquea explícitamente. El plan de infraestructura del proyecto
(`Guia_Despliegue_Infraestructura_Cloud.docx`) recoge la razón: el dataset es propiedad
privada de una entidad externa, lo que impone almacenamiento cifrado, control de acceso
estricto y prohibición de infraestructura compartida o comunitaria.

Consecuencia práctica para quien contribuya: **una imagen subida por error a un repositorio
público no se retira con `git rm`**. Los objetos siguen accesibles por su SHA en GitHub y en
cualquier fork. Revisa `git status` antes de cada `git add`.

## Las 14 clases de rasgo

Los nombres de clase de `configs/data.yaml` son **códigos del catálogo de rasgos**, no
nombres legibles ni especies:

```
0:'8'  1:'56'  2:'58'  3:'79'  4:'81'  5:'82'  6:'83'
7:'84' 8:'85'  9:'86'  10:'89' 11:'10' 12:'V1' 13:'radio'
```

`V1` corresponde a vasos y `radio` a radios; el resto son códigos numéricos del catálogo de
referencia. `<añadir aquí la traducción código → rasgo anatómico>`: sin esa tabla, ninguna
figura por clase es interpretable fuera del proyecto. Ojo con la clase `11:'10'`, que se lee
como "1.0" en las leyendas de las figuras.

## Paradigma de anotación: selectiva, no exhaustiva

Esta es la característica que condiciona todo el diseño y todas las métricas. El experto
**no anota todos los rasgos visibles en la imagen**: anota los que distinguen esa especie de
las demás. En consecuencia:

- La ausencia de una anotación **no** significa ausencia del rasgo.
- Un rasgo detectado correctamente por el modelo pero no anotado cuenta como falso positivo.
- mAP, precisión y recall calculados de la forma estándar están sesgados a la baja, y no de
  forma uniforme entre clases: los rasgos que casi nunca son diagnósticos salen peor.

Por eso el proyecto añade una pérdida semántica que compara distribuciones agregadas en lugar
de exigir exhaustividad, y por eso reporta métricas top-k en vez de una única respuesta.

## Sesgos y limitaciones conocidos

- **Desequilibrio entre especies**: entre 4 y 23 imágenes por especie. Las especies con menos
  de ~8 imágenes producen métricas por clase que no son estimaciones fiables.
- **Varias imágenes por espécimen físico.** Si se reparten entre pliegues de validación
  cruzada, el resultado mide parecido entre cortes de la misma pieza, no generalización.
  Agrupa por espécimen.
- **Heterogeneidad de captura**: iluminación, escala y preparación de la superficie varían
  entre lotes. No hay normalización de escala física (píxeles por milímetro) en el pipeline:
  las áreas se expresan como fracción del área de imagen, lo que absorbe parte del problema
  pero no la diferencia real de aumento.
- **Cobertura del catálogo**: una imagen de una especie fuera de las 96 no puede clasificarse
  correctamente por construcción. El pipeline lo marca con `true_in_catalog_<modelo>`; sepáralo
  siempre de los errores reales.
- **Consistencia entre anotadores**: no hay medida de acuerdo inter-anotador. Cuando una
  especie muestra dispersión intra-especie comparable a su distancia a las vecinas, no se
  puede distinguir si es variabilidad biológica real o criterio de anotación inconsistente.

## Procedencia y trazabilidad

- Anotación en CVAT; exportación COCO; conversión a formato YOLO-seg con
  `src/prepare/coco_to_yolo_seg.py`.
- Limpieza por percentil de área con `src/prepare/filter_small_polygons.py`.
- Revisión de duplicados y de anotaciones sospechosas con `src/prepare/review_pairs.py`.
- El mapa id → nombre latino vive en `especies.json` (copia publicada en
  `results/inference_32/especies.json`). Los nombres siguen la nomenclatura del catálogo de
  referencia con autoría taxonómica; versiona ese fichero si cambias la fuente.
