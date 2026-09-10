# data/

Este directorio está **vacío en el repositorio a propósito**. El corpus de imágenes de
macroscopía y sus anotaciones son propiedad de una entidad externa y no se redistribuyen
(ver [`../docs/DATA_CARD.md`](../docs/DATA_CARD.md)).

Estructura que esperan los scripts cuando el dataset está colocado:

```
data/
└── raw_selected/
    ├── data.yaml                  copia de ../configs/data.yaml
    ├── images/
    │   ├── train/  val/  test/
    ├── labels/
    │   ├── train/  val/  test/    etiquetas YOLO-seg (.txt)
    ├── annotations/
    │   └── instances_selected.json   COCO del ground truth
    └── split1/
        ├── train.csv  val.csv  test.csv
```

Y, fuera de `data/`, en la raíz del proyecto:

```
outputs/relative_feature_vectors.csv     priors por especie (pérdida semántica)
models/inferences/especies.json          mapa id → nombre latino
models/sem_loss_stratified/best.pt       pesos del segmentador
models/species_classifier/*.joblib       clasificadores entrenados
```

Ninguna de estas rutas se versiona: todas están en `.gitignore`.
