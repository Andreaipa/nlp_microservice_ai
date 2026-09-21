#!/usr/bin/env python3
"""Pseudo-etiquetado: ampliar las anotaciones con el detector ya entrenado.

La pre-anotación por OCR sólo cubre las fotografías en las que el troquelado
es legible, que en este dataset es una minoría: el resto se queda sin
anotación y no puede usarse para entrenar.

Este script cierra ese círculo. Un detector entrenado con las pocas imágenes
disponibles ya ha aprendido qué aspecto tiene el hombro de un cilindro, que es
una forma muy consistente, y puede localizarlo también donde el texto no se
lee. Sus detecciones se convierten en anotaciones nuevas, se revisan, y con
ellas se reentrena. Cada vuelta amplía la cobertura.

Dos salvaguardas, porque un error aquí se amplifica en la siguiente vuelta:

* sólo se acepta una detección por encima de un umbral de confianza alto;
* nunca se sobrescribe una anotación existente, ni las derivadas del OCR
  (que están ancladas a la verdad de campo) ni las revisadas a mano.

Uso:
    python training/pseudo_label.py --weights models/runs/detector/weights/best.pt
    python training/pseudo_label.py --weights ... --conf 0.7 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.preprocessing import image_io  # noqa: E402
from app.vision.base import DetectionClass  # noqa: E402

CLASS_INDEX = {name: index for index, name in enumerate(DetectionClass.ALL)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Umbral deliberadamente alto: en pseudo-etiquetado es preferible añadir pocas
# muestras correctas que muchas dudosas, porque el error se realimenta.
DEFAULT_CONFIDENCE = 0.70


def to_yolo(box, width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    return (
        ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
        (x2 - x1) / width, (y2 - y1) / height,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels"))
    parser.add_argument("--out", type=Path, default=Path("datasets/labels_pseudo"),
                        help="Directorio separado, para poder revisarlas antes "
                             "de mezclarlas con las anotaciones buenas.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--only-split", type=Path, default=None,
                        help="Limitar a un listado de imágenes, p. ej. train.txt. "
                             "Pseudo-etiquetar el test invalidaría la evaluación.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: no existe {args.weights}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta ultralytics. Instale: pip install -r requirements/train.txt")
        return 1

    if args.only_split and args.only_split.exists():
        candidates = [
            args.root / line.strip()
            for line in args.only_split.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if not args.only_split:
            print("AVISO: sin --only-split se recorre todo el dataset. Conviene")
            print("       limitarlo a train.txt: pseudo-etiquetar las imágenes de")
            print("       test contamina la evaluación.")
        candidates = [
            p for p in sorted(args.root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]

    pending = [
        path for path in candidates
        if path.exists()
        and not (args.labels / path.relative_to(args.root).with_suffix(".txt")).exists()
    ]
    if not pending:
        print("No hay imágenes sin anotar: nada que pseudo-etiquetar.")
        return 0

    model = YOLO(str(args.weights))
    settings = get_settings()

    added = 0
    per_condition: Counter = Counter()
    per_class: Counter = Counter()

    print(f"Pseudo-etiquetando {len(pending)} imágenes sin anotación "
          f"(confianza mínima {args.conf})...")

    for index, path in enumerate(pending, start=1):
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image, _ = image_io.limit_size(image, settings.max_image_side)
        height, width = image.shape[:2]

        results = model.predict(source=image, conf=args.conf,
                                imgsz=settings.detector_input_size, verbose=False)

        lines: list[str] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                label = names.get(int(box.cls.item()), "")
                if label not in CLASS_INDEX:
                    continue
                coords = [float(v) for v in box.xyxy[0].tolist()]
                lines.append(
                    f"{CLASS_INDEX[label]} "
                    + " ".join(f"{v:.6f}" for v in to_yolo(coords, width, height))
                )
                per_class[label] += 1

        if not lines:
            continue

        relative = path.relative_to(args.root)
        condition = relative.parts[1] if len(relative.parts) > 2 else "?"
        per_condition[condition] += 1
        added += 1

        if not args.dry_run:
            target = args.out / relative.with_suffix(".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if index % 50 == 0:
            print(f"  {index}/{len(pending)}  nuevas: {added}")

    print("\n" + "=" * 62)
    print("PSEUDO-ETIQUETADO" + ("  (simulación)" if args.dry_run else ""))
    print("=" * 62)
    print(f"  Imágenes sin anotar ....... {len(pending)}")
    print(f"  Con detección aceptada .... {added} ({added / len(pending):.1%})")
    if per_class:
        print("\n  Cajas por clase:")
        for label, count in per_class.most_common():
            print(f"    {label:<16} {count}")
    if per_condition:
        print("\n  Por condición:")
        for condition, count in sorted(per_condition.items()):
            print(f"    {condition:<20} {count}")
    print("=" * 62)
    print(f"\n  Escritas en {args.out}, separadas de las anotaciones fiables.")
    print("  Revíselas antes de mezclarlas: un error aquí se realimenta en la")
    print("  siguiente vuelta de entrenamiento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
