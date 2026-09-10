# Guía de Instalación del Entorno Conda - Proyecto Maderas

## 🔍 Diagnóstico del Problema

El error ocurría porque:
1. **Conflicto de CUDA**: El environment.yml original usaba CUDA 11.8, pero el servidor tiene CUDA 12.x
2. **Conflicto conda/pip**: OpenCV fue instalado por conda, pero pip intentaba desinstalarlo
3. **Paquetes incompatibles**: Las versiones de triton y setuptools causaban conflictos

## ✅ Solución: environment.yml

Se han realizado los siguientes cambios:

| Problema | Solución |
|----------|----------|
| CUDA 11.8 → CUDA 12.1 | Detecta automáticamente tu CUDA |
| OpenCV pip conflictivo | Usa opencv de conda-forge (más estable) |
| Orden de canales | pytorch → conda-forge → nvidia (orden correcto) |
| Dependencias pip | Solo paquetes que NO están en conda |

## 🚀 Instalación Paso a Paso

### Opción 1: Script Automático (RECOMENDADO)

```bash
# En tu servidor remoto:
cd ~
chmod +x setup_env.sh
./setup_env.sh
```

### Opción 2: Instalación Manual

```bash
# 1. Inicializar conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Limpiar entorno anterior si existe
conda env remove --name maderas -y --force-pkgs-dirs

# 3. Esperar un poco
sleep 3

# 4. Crear entorno limpio
conda env create --name maderas --file environment.yml

# 5. Activar
conda activate maderas

# 6. Verificar
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🔧 Si Aún Hay Problemas

### Problema: "CondaError: Run 'conda init' before 'conda activate'"
**Solución:**
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maderas
```

### Problema: "pip failed" con opencv-python
**Solución:**
```bash
# Usa conda para todo, no pip
conda install -c conda-forge opencv -y
```

### Problema: CUDA no disponible en PyTorch
**Solución:**
```bash
# Verifica tu CUDA
nvidia-smi

# Si tienes CUDA 13.x, cambia en environment.yml:
# pytorch-cuda=12.1  →  pytorch-cuda=13.2
```

### Problema: "ConnectionError" descargando paquetes
**Solución:**
```bash
# Limpia caché de conda
conda clean --all -y

# Reintentar
conda env create --name maderas --file environment.yml --force
```

## 📋 Activación Futura

Cada vez que quieras usar el entorno:

```bash
# Opción 1: Desde cualquier ubicación
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maderas

# Opción 2: Si la opción 1 no funciona, usa la ruta completa
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maderas

# Desactivar
conda deactivate
```

## ✨ Extras

### Ver qué se instaló
```bash
conda list
pip list
```

### Entrenar tu modelo
```bash
conda activate maderas
python train_yolo_seg_loss.py
```

### Actualizar paquete específico
```bash
conda activate maderas
conda update ultralytics -y
# o
pip install --upgrade ultralytics
```

## 📞 Problemas Restantes

Si después de seguir esta guía aún tienes problemas, ejecuta esto para diagnosticar:

```bash
echo "=== Sistema ==="
uname -a

echo "=== CUDA ==="
nvidia-smi

echo "=== Conda ==="
conda --version
conda list

echo "=== Python ==="
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

**Archivos incluidos:**
- `environment.yml` - Archivo de configuración corregido
- `setup_env.sh` - Script de instalación automática
- `INSTALACION_GUIA.md` - Esta guía
