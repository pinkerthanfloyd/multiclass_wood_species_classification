"""rename_by_species.py
========================
Recorre un directorio de imágenes y las renombra añadiendo el nombre de la
especie según el mapeo id -> nombre latino en `species.json`.

Se asume que cada imagen tiene como prefijo el id de especie, p.ej.
`0257_foto1.jpg` -> `0257_Entandrophragma_utile_foto1.jpg`.
Si tus nombres de archivo no llevan el id como prefijo, ajusta
`extract_species_id()`.

Uso
---
    python rename_by_species.py --dir ./imagenes --json species.json
    python rename_by_species.py --dir ./imagenes --json species.json --apply
    python rename_by_species.py --dir ./imagenes --json species.json --apply -r

Por defecto corre en modo *dry-run* (solo imprime qué haría). Pasa `--apply`
para renombrar de verdad. Usa `-r/--recursive` para bajar a subcarpetas.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_species_map(path: Path) -> Dict[str, str]:
    """Carga id -> nombre, aceptando tanto '0257' como '257' como clave."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        ks = str(k)
        out[ks] = str(v)
        out[ks.zfill(4)] = str(v)
        out[ks.lstrip("0") or "0"] = str(v)
    return out


def slugify(name: str) -> str:
    """'Entandrophragma utile' -> 'Entandrophragma_utile' (sin acentos/raros)."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[\s]+", "_", name)


def extract_species_id(stem: str) -> Optional[str]:
    """Extrae el id de especie del inicio del nombre de archivo (dígitos)."""
    m = re.match(r"^(\d+)", stem)
    return m.group(1) if m else None


def already_tagged(stem: str, slug: str) -> bool:
    """Evita volver a añadir el nombre si el archivo ya fue renombrado antes."""
    return slug.lower() in stem.lower()


def unique_target(path: Path) -> Path:
    """Si el destino ya existe, añade _1, _2... para no pisar archivos."""
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def rename_directory(img_dir: Path, species_map: Dict[str, str],
                      recursive: bool, apply: bool) -> None:
    pattern = img_dir.rglob("*") if recursive else img_dir.glob("*")
    n_ok = n_skip = n_missing = 0

    for path in sorted(pattern):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        species_id = extract_species_id(path.stem)
        if species_id is None:
            print(f"[SKIP] sin id reconocible: {path.name}")
            n_skip += 1
            continue

        name = species_map.get(species_id) or species_map.get(species_id.zfill(4))
        if not name:
            print(f"[MISS] id={species_id} no está en species.json: {path.name}")
            n_missing += 1
            continue

        slug = slugify(name)
        if already_tagged(path.stem, slug):
            n_skip += 1
            continue

        # conserva el resto del nombre original (todo lo que sigue al id)
        rest = path.stem[len(species_id):].lstrip("_- ")
        new_stem = f"{species_id}_{slug}" + (f"_{rest}" if rest else "")
        target = unique_target(path.with_name(new_stem + path.suffix))

        print(f"[{'RENAME' if apply else 'DRY-RUN'}] {path.name} -> {target.name}")
        if apply:
            path.rename(target)
        n_ok += 1

    print(f"\nTotal: {n_ok} renombrados/{'aplicados' if apply else 'previstos'}, "
          f"{n_skip} omitidos, {n_missing} sin especie en el JSON.")
    if not apply:
        print("(modo dry-run: no se ha movido ningún archivo — usa --apply para ejecutar)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path, help="Directorio de imágenes")
    ap.add_argument("--json", required=True, type=Path, help="species.json (id -> nombre)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Incluir subcarpetas")
    ap.add_argument("--apply", action="store_true", help="Renombrar de verdad (si no, dry-run)")
    args = ap.parse_args()

    species_map = load_species_map(args.json)
    rename_directory(args.dir, species_map, args.recursive, args.apply)


if __name__ == "__main__":
    main()