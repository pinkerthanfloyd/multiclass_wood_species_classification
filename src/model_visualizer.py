"""visualize_yolo_torchview.py
Renderiza el segmentador YOLO como grafo colapsado por profundidad.
"""
from pathlib import Path
from ultralytics import YOLO
from torchview import draw_graph

YOLO_WEIGHTS = Path("models/sem_loss_stratified/best.pt")
IMGSZ = 1024

model = YOLO(str(YOLO_WEIGHTS)).model
model.eval()

# depth controla el nivel de colapso: 1 = solo bloques de alto nivel,
# 2 = un nivel de detalle, 3 = mas granular. Prueba 2 primero.
graph = draw_graph(
    model,
    input_size=(1, 3, IMGSZ, IMGSZ),
    depth=2,
    expand_nested=False,
    graph_name="YOLO11_seg",
    save_graph=True,
    directory=str(YOLO_WEIGHTS.parent),
    filename="arch_torchview",
)
print(f"Grafo guardado en {YOLO_WEIGHTS.parent}/arch_torchview.png")