#!/bin/bash

echo "======================================"
echo "  LIMPIEZA COMPLETA DEL ENTORNO"
echo "======================================"
echo ""

# 1. Inicializar conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Desactivar cualquier entorno activo
echo "✓ Desactivando entornos activos..."
conda deactivate 2>/dev/null || true
sleep 1

# 3. ELIMINAR completamente el entorno
echo "✓ Eliminando entorno 'maderas' completamente..."
rm -rf "$(conda info --base)/envs/maderas" 2>/dev/null || true
sleep 2

# 4. Limpiar caché de conda
echo "✓ Limpiando caché de conda..."
conda clean --all -y
conda clean --packages -y
sleep 2

# 5. CREAR NUEVO ENTORNO LIMPIO
echo ""
echo "✓ Creando nuevo entorno 'maderas'..."
echo ""

# Usar la ruta completa por seguridad
conda env create --name maderas --file environment.yml

# 6. Activar
echo ""
echo "✓ Activando entorno..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maderas

# 7. Verificar instalación
echo ""
echo "======================================"
echo "  VERIFICACIÓN"
echo "======================================"
echo ""

python --version
echo ""

echo "PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" || echo "  ⚠ Error al importar torch"

echo ""
echo "OpenCV:"
python -c "import cv2; print(f'  Version: {cv2.__version__}')" || echo "  ⚠ Error al importar cv2"

echo ""
echo "Ultralytics:"
python -c "import ultralytics; print(f'  Version: {ultralytics.__version__}')" || echo "  ⚠ Error al importar ultralytics"

echo ""
echo "======================================"
echo "✅ ¡Listo! Entorno limpio y funcional"
echo "======================================"
