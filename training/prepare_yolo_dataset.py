#!/usr/bin/env python3
"""Ensamblado del dataset en la estructura que espera ultralytics.

Ultralytics localiza las etiquetas sustituyendo `/images/` por `/labels/` en
la ruta de cada imagen, de modo que la organización por cilindro y condición
del dataset original no le sirve tal cual. Este script construye la estructura
que necesita sin duplicar 1,5 GB: las imágenes se enlazan simbólicamente.

    datasets/yolo/
        images/{train,val,test}/Cilindro_007__3.Cerca__IMG_1234.jpg
        labels/{train,val,test}/Cilindro_007__3.Cerca__IMG_1234.txt
        data.yaml

El nombre aplanado conserva cilindro y condición, así que sigue siendo posible
saber de dónde viene cada muestra al analizar los fallos.

El reparto se toma del manifiesto de `split_by_cylinder.py`, que separa por
identidad de cilindro. No se recalcula aquí para que no puedan divergir.

Uso:
    python training/prepare_yolo_dataset.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision.base import DetectionClass  # noqa: E402

SPLITS = ("train", "val", "test")


def rebalance_validation(
    manifest: dict, root: Path, labels: Path, min_val: int
) -> dict[str, list[Path]]:
    """Reparte las imágenes ANOTADAS entre subconjuntos.

    El reparto de `split_by_cylinder.py` equilibra todas las fotografías, pero
    sólo una fracción llegó a anotarse, y esa fracción puede quedar muy
    desigual: en la primera pasada la validación se quedó con tres imágenes,
    insuficientes para decidir una parada temprana.

    Aquí se trasladan cilindros COMPLETOS de entrenamiento a validación hasta
    alcanzar un mínimo razonable. Se mueven cilindros enteros, nunca
    fotografías sueltas, para no romper la separación por identidad, que es lo
    que hace que la evaluación mida generalización.

    El conjunto de test no se toca: es la referencia de la evaluación final.
    """
    by_cylinder: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split in SPLITS:
        for cylinder in manifest["splits"][split]["cylinders"]:
            cylinder_dir = root / cylinder
            if not cylinder_dir.exists():
                continue
            for image in sorted(cylinder_dir.rglob("*")):
                if not image.is_file() or image.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".webp"
                }:
                    continue
                relative = image.relative_to(root)
                if (labels / relative.with_suffix(".txt")).exists():
                    by_cylinder[split][cylinder].append(relative)

    assignment: dict[str, list[Path]] = {
        split: [img for imgs in by_cylinder[split].values() for img in imgs]
        for split in SPLITS
    }

    moved: list[str] = []
    if len(assignment["val"]) < min_val:
        # Se eligen los cilindros con menos imágenes anotadas, para alcanzar el
        # mínimo vaciando lo menos posible el entrenamiento.
        candidates = sorted(
            by_cylinder["train"].items(), key=lambda item: len(item[1])
        )
        for cylinder, images in candidates:
            if len(assignment["val"]) >= min_val:
                break
            if not images:
                continue
            assignment["val"].extend(images)
            for image in images:
                assignment["train"].remove(image)
            moved.append(cylinder)

    if moved:
        print(f"  Trasladados a validación {len(moved)} cilindros completos "
              f"para alcanzar el mínimo de {min_val}: {', '.join(moved)}")
    return assignment


def flatten_name(relative: Path) -> str:
    """Cilindro_007/3.Cerca/IMG.jpg -> Cilindro_007__3.Cerca__IMG.jpg"""
    return "__".join(relative.parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels"))
    parser.add_argument("--split-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--out", type=Path, default=Path("datasets/yolo"))
    parser.add_argument("--copy", action="store_true",
                        help="Copia las imágenes en lugar de enlazarlas.")
    parser.add_argument("--min-val", type=int, default=10,
                        help="Mínimo de imágenes de validación. Si el reparto "
                             "original deja menos, se trasladan cilindros "
                             "completos desde train.")
    args = parser.parse_args()

    manifest_path = args.split_dir / "split_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: falta {manifest_path}. Ejecute antes split_by_cylinder.py")
        return 1
    if not args.labels.exists():
        print(f"ERROR: no existe {args.labels}. Ejecute antes preannotate.py "
              "y revise las cajas propuestas.")
        return 1

    root = args.root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment = rebalance_validation(
        manifest, args.root, args.labels, args.min_val
    )
    linked = dict.fromkeys(SPLITS, 0)
    unlabelled = dict.fromkeys(SPLITS, 0)

    for split in SPLITS:
        entries = assignment.get(split, [])
        if not entries:
            continue

        image_dir = args.out / "images" / split
        label_dir = args.out / "labels" / split
        for directory in (image_dir, label_dir):
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)

        for relative in entries:
            source = root / relative
            label_source = args.labels / relative.with_suffix(".txt")
            if not source.exists():
                continue

            # Una imagen sin etiqueta revisada no entra: el detector
            # interpretaría la ausencia como "aquí no hay nada que detectar" y
            # aprendería justo lo contrario de lo que se busca.
            if not label_source.exists():
                unlabelled[split] += 1
                continue

            flat = flatten_name(relative)
            target = image_dir / flat
            if args.copy:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source)
            shutil.copy2(label_source, label_dir / Path(flat).with_suffix(".txt"))
            linked[split] += 1

    data_yaml = args.out / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(DetectionClass.ALL))
    data_yaml.write_text(
        f"# Generado por training/prepare_yolo_dataset.py\n"
        f"# El reparto proviene de split_by_cylinder.py: ningún cilindro\n"
        f"# aparece en más de un subconjunto.\n"
        f"path: {args.out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )

    print("=" * 62)
    print("DATASET YOLO")
    print("=" * 62)
    for split in SPLITS:
        print(f"  {split:<6} {linked[split]:>4} imágenes etiquetadas"
              + (f"   ({unlabelled[split]} sin etiqueta, omitidas)"
                 if unlabelled[split] else ""))
    print("=" * 62)
    print(f"  {data_yaml}")

    total = sum(linked.values())
    if total == 0:
        print("\n  No hay ninguna imagen etiquetada: no se puede entrenar todavía.")
        return 1
    if linked["train"] < 100:
        print(f"\n  AVISO: sólo {linked['train']} imágenes de entrenamiento. "
              "El modelo resultante será poco fiable.")

    print(f"\n  Entrenar:  python training/train_detector.py --data {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
