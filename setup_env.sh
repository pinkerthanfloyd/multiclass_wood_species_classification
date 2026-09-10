#!/bin/bash
set -e

echo "======================================"
echo "  Configurar entorno conda: maderas"
echo "======================================"
echo ""

# 1. Inicializar conda
echo "✓ Inicializando conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Limpiar entorno anterior si existe
echo "✓ Buscando entorno anterior..."
if conda env list | grep -q "^maderas "; then
    echo "  → Eliminando entorno 'maderas' anterior..."
    conda remove --name maderas --all -y
    sleep 3
fi

# 3. Crear nuevo entorno
echo "✓ Creando nuevo entorno 'maderas'..."
conda env create --name maderas --file environment.yml

# 4. Activar entorno
echo "✓ Activando entorno..."
conda activate maderas

# 5. Verificar instalación
echo ""
echo "======================================"
echo "  Verificación de instalación"
echo "======================================"
echo ""
echo "Python versión:"
python --version

echo ""
echo "PyTorch instalado:"
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "OpenCV instalado:"
python -c "import cv2; print(f'Version: {cv2.__version__}')"

echo ""
echo "Paquetes instalados (primeros 20):"
pip list | head -20

echo ""
echo "======================================"
echo "✅ ¡Entorno listo para usar!"
echo "======================================"
echo ""
echo "Próximas veces, activa con:"
echo '  source "$(conda info --base)/etc/profile.d/conda.sh"'
echo "  conda activate maderas"
echo ""
