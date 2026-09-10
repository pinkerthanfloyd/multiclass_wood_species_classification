# Documentación

| Fichero | Contenido |
|---|---|
| [`INSTALACION.md`](INSTALACION.md) | Creación del entorno conda, verificación de CUDA, problemas frecuentes |
| [`PIPELINE.md`](PIPELINE.md) | Qué hace cada script, en qué orden y qué espera encontrar |
| [`DATA_CARD.md`](DATA_CARD.md) | Origen, tamaño, sesgos y condiciones de uso del corpus |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Uso previsto, límites, rendimiento y consideraciones éticas |
| `Guia_Despliegue_Infraestructura_Cloud.docx` | Plan de infraestructura GPU en cloud, dimensionamiento y coste |

**Antes de hacer público el repositorio**: la guía de despliegue contiene nombres de bucket y
rangos de red internos (`gs://maderas-results`, `maderas-dataset`, subred `10.0.0.0/24`). No
son credenciales, pero valora sustituirlos por marcadores genéricos. Y completa los campos
`<…>` de `DATA_CARD.md`.
