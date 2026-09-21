#!/usr/bin/env python3
"""Importación del dataset de fotografías al proyecto.

Copia la carpeta original a `datasets/raw/` normalizando de paso las
variaciones de nombre que trae la captura real, para que el resto de la
cadena pueda asumir una estructura única:

    "1. Frontal "   -> "1.Frontal"
    "6. Poca luz"   -> "6.Poca luz"
    "Descripción 85"-> "Descripción"

Los nombres de archivo no se tocan: son la referencia con la que se anotará y
con la que se construye la verdad de campo.

Uso:
    python training/import_dataset.py --source ~/Downloads/"Cilindros Electrametal"
    python training/import_dataset.py --source ... --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
DOC_SUFFIXES = {".docx", ".doc", ".txt"}

# Condiciones de captura canónicas, en el orden del protocolo.
CANONICAL_CONDITIONS = (
    "1.Frontal", "2.Diagonal", "3.Cerca", "4.Lejos", "5.Luz normal", "6.Poca luz",
)
DESCRIPTION_DIR = "Descripción"


def _fold(text: str) -> str:
    """Minúsculas sin acentos ni signos, para comparar nombres de carpeta."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


_CONDITION_LOOKUP = {_fold(name): name for name in CANONICAL_CONDITIONS}


def canonical_condition(name: str) -> str | None:
    """Traduce el nombre de una subcarpeta a su forma canónica."""
    folded = _fold(name)
    if folded in _CONDITION_LOOKUP:
        return _CONDITION_LOOKUP[folded]
    # "Descripción 85", "Descripcion " y variantes.
    if folded.startswith("descripcion"):
        return DESCRIPTION_DIR
    # "1 Frontal", "1_frontal": se resuelve por el número inicial.
    match = re.match(r"^(\d)", folded)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(CANONICAL_CONDITIONS):
            return CANONICAL_CONDITIONS[index]
    return None


_CYLINDER_RE = re.compile(r"cilindro[_\s-]*(\d+)", re.IGNORECASE)


def canonical_cylinder(name: str) -> str | None:
    """Normaliza 'Cilindro 7' o 'cilindro_007' a 'Cilindro_007'."""
    match = _CYLINDER_RE.search(name)
    if not match:
        return None
    return f"Cilindro_{int(match.group(1)):03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra lo que haría sin copiar nada.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe archivos ya importados.")
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.exists():
        print(f"ERROR: no existe {source}")
        return 1

    copied = skipped = 0
    unknown_dirs: list[str] = []
    cylinders: set[str] = set()
    renamed: list[tuple[str, str]] = []

    for cylinder_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        cylinder = canonical_cylinder(cylinder_dir.name)
        if cylinder is None:
            unknown_dirs.append(cylinder_dir.name)
            continue
        cylinders.add(cylinder)
        if cylinder != cylinder_dir.name:
            renamed.append((cylinder_dir.name, cylinder))

        for sub in sorted(d for d in cylinder_dir.iterdir() if d.is_dir()):
            condition = canonical_condition(sub.name)
            if condition is None:
                unknown_dirs.append(f"{cylinder_dir.name}/{sub.name}")
                continue
            if condition != sub.name:
                renamed.append((f"{cylinder_dir.name}/{sub.name}",
                                f"{cylinder}/{condition}"))

            for item in sorted(sub.rglob("*")):
                if not item.is_file() or item.name.startswith("."):
                    continue
                suffix = item.suffix.lower()
                if suffix not in IMAGE_SUFFIXES and suffix not in DOC_SUFFIXES:
                    continue

                target = args.dest / cylinder / condition / item.name
                if target.exists() and not args.overwrite:
                    skipped += 1
                    continue
                if not args.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target)
                copied += 1

    print("=" * 62)
    print("IMPORTACIÓN DEL DATASET" + ("  (simulación)" if args.dry_run else ""))
    print("=" * 62)
    print(f"  Origen ............. {source}")
    print(f"  Destino ............ {args.dest}")
    print(f"  Cilindros .......... {len(cylinders)}")
    print(f"  Archivos copiados .. {copied}")
    print(f"  Ya existían ........ {skipped}")

    if renamed:
        print(f"\n  Carpetas normalizadas ({len(renamed)}):")
        for original, canonical in renamed[:12]:
            print(f"    {original}  ->  {canonical}")
        if len(renamed) > 12:
            print(f"    ... y {len(renamed) - 12} más")

    if unknown_dirs:
        print(f"\n  AVISO: carpetas no reconocidas ({len(unknown_dirs)}):")
        for name in unknown_dirs[:12]:
            print(f"    {name}")
        print("    No se han importado. Revise si contienen datos necesarios.")

    print("=" * 62)
    if not args.dry_run:
        print("\nSiguiente paso:")
        print(f"  python training/audit_dataset.py --root {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
