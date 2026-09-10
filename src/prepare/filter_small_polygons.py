"""
Script para eliminar polígonos pequeños (cristalizaciones y espuma) del dataset YOLO segmentation.

Calcula el área de cada polígono usando la fórmula de Shoelace y elimina aquellos
que no superan un umbral de área especificado.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple
import sys

def calculate_polygon_area(coords: List[float]) -> float:
    """
    Calcula el área de un polígono usando la fórmula de Shoelace.

    Args:
        coords: Lista de coordenadas normalizadas [x1, y1, x2, y2, ..., xn, yn]

    Returns:
        Área del polígono normalizado (entre 0 y 1)
    """
    if len(coords) < 6:  # Menos de 3 puntos
        return 0.0

    # Convertir a lista de tuplas (x, y)
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    # Fórmula de Shoelace
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    area = abs(area) / 2.0
    return area


def parse_annotation_line(line: str) -> Tuple[int, List[float], bool]:
    """
    Parsea una línea de anotación YOLO segmentation.

    Args:
        line: Línea del archivo de anotación

    Returns:
        Tupla (class_id, coordenadas, es_válido)
    """
    line = line.strip()
    if not line:
        return -1, [], False

    try:
        parts = line.split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]

        # Validar que tenemos pares de coordenadas válidas
        if len(coords) % 2 != 0 or len(coords) < 6:
            return class_id, coords, False

        return class_id, coords, True
    except (ValueError, IndexError):
        return -1, [], False


def filter_annotation_file(
    filepath: str,
    area_threshold: float = 0.003,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Filtra polígonos pequeños de un archivo de anotación YOLO.

    Args:
        filepath: Ruta del archivo de anotación
        area_threshold: Área mínima normalizada (0.001 = 0.1% de la imagen)
        verbose: Si mostrar información de progreso

    Returns:
        Tupla (total_inicial, total_eliminados)
    """
    # Leer archivo original
    with open(filepath, 'r') as f:
        lines = f.readlines()

    total_initial = len(lines)
    filtered_lines = []
    removed_count = 0
    removed_by_class = {}

    for line_idx, line in enumerate(lines):
        class_id, coords, is_valid = parse_annotation_line(line)

        if not is_valid:
            # Lineas inválidas o vacías las mantenemos (por si acaso)
            if line.strip():
                filtered_lines.append(line)
            continue

        # Calcular área del polígono
        area = calculate_polygon_area(coords)

        # Decidir si mantener el polígono
        if area >= area_threshold:
            filtered_lines.append(line)
        else:
            removed_count += 1
            if class_id not in removed_by_class:
                removed_by_class[class_id] = 0
            removed_by_class[class_id] += 1

            if verbose:
                print(f"  Eliminado: Clase {class_id}, Área: {area:.6f} "
                      f"(umbral: {area_threshold})")

    # Escribir archivo filtrado
    with open(filepath, 'w') as f:
        f.writelines(filtered_lines)

    if verbose:
        print(f"  Archivo: {os.path.basename(filepath)}")
        print(f"    - Total inicial: {total_initial}")
        print(f"    - Eliminados: {removed_count}")
        print(f"    - Mantenidos: {total_initial - removed_count}")
        if removed_by_class:
            print(f"    - Por clase: {removed_by_class}")

    return total_initial, removed_count


def main():
    """
    Script principal. Busca todos los archivos .txt en la carpeta
    y aplica el filtro de área.
    """
    # Configuración
    PROJECT_DIR = Path(__file__).parent
    AREA_THRESHOLD = 0.003  # Ajusta este valor según necesites

    print(f"\n{'='*70}")
    print(f"FILTRADO DE POLÍGONOS PEQUEÑOS - Dataset YOLO Segmentation")
    print(f"{'='*70}")
    print(f"📁 Directorio: {PROJECT_DIR}")
    print(f"📏 Umbral de área: {AREA_THRESHOLD} (área normalizada)")
    print(f"{'='*70}\n")

    # Encontrar archivos de anotación
    txt_files = sorted(glob.glob(str(PROJECT_DIR / "*.txt")))

    # Excluir el archivo train.txt si existe (es una lista de rutas)
    txt_files = [f for f in txt_files if not f.endswith('train.txt')]

    if not txt_files:
        print("❌ No se encontraron archivos de anotación (.txt)")
        return

    print(f"🔍 Se encontraron {len(txt_files)} archivo(s) de anotación\n")

    total_polygons = 0
    total_removed = 0

    # Procesar cada archivo
    for txt_file in txt_files:
        print(f"Procesando: {os.path.basename(txt_file)}")
        initial, removed = filter_annotation_file(txt_file, AREA_THRESHOLD, verbose=True)
        total_polygons += initial
        total_removed += removed
        print()

    # Resumen final
    print(f"{'='*70}")
    print(f"RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"✓ Total de polígonos procesados: {total_polygons}")
    print(f"✗ Total de polígonos eliminados: {total_removed}")
    print(f"✓ Total de polígonos mantenidos: {total_polygons - total_removed}")
    print(f"📊 Porcentaje eliminado: {100*total_removed/total_polygons:.2f}%")
    print(f"{'='*70}\n")

    if total_removed > 0:
        print("💡 Sugerencias:")
        print(f"   - Si eliminaste demasiado, reduce el umbral (ej: {AREA_THRESHOLD*0.5:.6f})")
        print(f"   - Si eliminaste muy poco, aumenta el umbral (ej: {AREA_THRESHOLD*1.5:.6f})")
        print("   - Edita la variable AREA_THRESHOLD en este script para ajustarlo.\n")


if __name__ == "__main__":
    main()
