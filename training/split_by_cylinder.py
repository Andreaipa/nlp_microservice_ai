#!/usr/bin/env python3
"""Separación train/val/test por IDENTIDAD DE CILINDRO.

Por qué no vale un reparto aleatorio de imágenes: si dos fotos del mismo
cilindro caen una en entrenamiento y otra en test, el modelo habrá visto ese
número de serie concreto durante el entrenamiento y la métrica de test medirá
memorización, no generalización. El resultado sería un mAP alto y un fracaso
en planta el primer día que aparezca un cilindro nuevo.

Aquí el reparto se hace siempre a nivel de cilindro: todas las fotos de un
cilindro van íntegras al mismo subconjunto.

Además, el reparto no es puramente aleatorio: se equilibra el número de
imágenes por subconjunto asignando primero los cilindros con más fotos, porque
con un inventario pequeño un reparto al azar puede dejar el test con muy pocas
imágenes y hacer la métrica inestable.

Uso:
    python training/split_by_cylinder.py --root datasets/raw --out datasets/processed
    python training/split_by_cylinder.py --root datasets/raw --ratios 0.7 0.15 0.15
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def index_cylinders(root: Path) -> dict[str, list[Path]]:
    """Mapa cilindro -> lista de imágenes."""
    index: dict[str, list[Path]] = {}
    for cylinder_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        images = [
            path for path in sorted(cylinder_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if images:
            index[cylinder_dir.name] = images
    return index


def assign(
    index: dict[str, list[Path]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[str]]:
    """Reparte cilindros entre subconjuntos respetando las proporciones.

    Se asignan primero los cilindros con más imágenes, cada uno al subconjunto
    que esté más por debajo de su cuota. Así las proporciones se cumplen en
    número de IMÁGENES, que es lo que determina la estabilidad de la métrica,
    y no sólo en número de cilindros.
    """
    total_images = sum(len(v) for v in index.values())
    targets = {split: ratio * total_images for split, ratio in zip(SPLITS, ratios, strict=True)}

    order = sorted(index, key=lambda name: (-len(index[name]), name))
    # Un barajado con semilla fija sobre los grupos de igual tamaño evita que
    # el orden alfabético introduzca sesgo, manteniendo la reproducibilidad.
    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda name: -len(index[name]))

    assigned: dict[str, list[str]] = {split: [] for split in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)

    for name in order:
        size = len(index[name])
        # Subconjunto con mayor déficit relativo respecto a su objetivo.
        split = max(
            SPLITS,
            key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9)
            if targets[s] > 0 else float("-inf"),
        )
        assigned[split].append(name)
        counts[split] += size

    return assigned


def write_split(
    index: dict[str, list[Path]],
    assigned: dict[str, list[str]],
    out_dir: Path,
    root: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"root": str(root), "splits": {}}

    for split in SPLITS:
        names = sorted(assigned[split])
        paths: list[str] = []
        for name in names:
            paths.extend(str(p.relative_to(root)) for p in index[name])

        listing = out_dir / f"{split}.txt"
        listing.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")

        manifest["splits"][split] = {
            "cylinders": names,
            "cylinder_count": len(names),
            "image_count": len(paths),
            "listing": str(listing),
        }

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return manifest


def verify_no_leakage(manifest: dict) -> list[str]:
    """Comprueba que ningún cilindro aparece en más de un subconjunto."""
    problems: list[str] = []
    seen: Counter = Counter()
    for split in SPLITS:
        for name in manifest["splits"][split]["cylinders"]:
            seen[name] += 1
    for name, count in seen.items():
        if count > 1:
            problems.append(f"El cilindro {name} aparece en {count} subconjuntos.")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--out", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: no existe {args.root}")
        return 1

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"ERROR: las proporciones deben sumar 1.0 (suman {sum(args.ratios)})")
        return 1

    index = index_cylinders(args.root)
    if not index:
        print(f"ERROR: no se encontró ninguna imagen bajo {args.root}.")
        print("       Capture primero el dataset (ver docs/PROTOCOLO_CAPTURA.md).")
        return 1

    if len(index) < 3:
        print(f"ERROR: sólo hay {len(index)} cilindros con imágenes. Se necesitan al "
              "menos 3 para formar tres subconjuntos disjuntos, y bastantes más "
              "para que la métrica sea significativa.")
        return 1

    assigned = assign(index, tuple(args.ratios), args.seed)
    manifest = write_split(index, assigned, args.out, args.root)

    problems = verify_no_leakage(manifest)
    if problems:
        for problem in problems:
            print(f"ERROR DE FUGA: {problem}")
        return 1

    print("=" * 62)
    print("SEPARACIÓN POR IDENTIDAD DE CILINDRO")
    print("=" * 62)
    total_images = sum(manifest["splits"][s]["image_count"] for s in SPLITS)
    for split in SPLITS:
        info = manifest["splits"][split]
        share = 100.0 * info["image_count"] / max(total_images, 1)
        print(f"  {split:<6} {info['cylinder_count']:>4} cilindros  "
              f"{info['image_count']:>5} imágenes  ({share:5.1f}%)")
    print("=" * 62)
    print("Sin solapamiento de cilindros entre subconjuntos.")
    print(f"Manifiesto: {args.out / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
