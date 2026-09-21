#!/usr/bin/env python3
"""Extrae un dataset de reconocimiento de texto a partir de lo ya anotado.

Por qué existe: la evaluación de extremo a extremo dejó claro que el límite
del sistema no es localizar el troquelado —el detector lo encuentra con
confianza ~0.83— sino LEERLO. El CER medio es 0.537: más de la mitad de los
caracteres salen mal. Ningún ajuste de umbrales arregla eso; hace falta un
reconocedor que conozca este tipo de texto.

Lo bueno es que el dataset para entrenarlo ya existe sin pedir una sola
fotografía nueva:

* `training/preannotate.py` localizó la línea del serial en 326 imágenes;
* la ficha de cada cilindro dice qué pone exactamente en esa línea.

Recorte + texto correcto es justo lo que necesita un reconocedor.

Y el problema es más pequeño de lo que parece: los seriales de los 100
cilindros usan sólo 21 caracteres distintos, y en la práctica son los diez
dígitos más la 'K'; el resto aparece una o dos veces. Reconocer un alfabeto
así está mucho más cerca de un problema cerrado que el OCR general para el que
están entrenados los modelos genéricos.

Salida en el formato habitual de los reconocedores (PaddleOCR, RapidOCR,
TrOCR): una carpeta de recortes y un fichero `ruta<TAB>texto`.

Uso:
    python training/build_ocr_dataset.py
    python training/build_ocr_dataset.py --margin 0.25 --min-width 60
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preprocessing import image_io  # noqa: E402
from app.vision.base import DetectionClass  # noqa: E402

SERIAL_CLASS_INDEX = list(DetectionClass.ALL).index(DetectionClass.SERIAL_TEXT)

# Un recorte demasiado pequeño no contiene información recuperable: entrenar
# con él sólo añade ruido.
MIN_CROP_WIDTH = 48
MIN_CROP_HEIGHT = 12


def load_ground_truth(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["imagen"]: row["numero_serie"].strip().upper()
            for row in csv.DictReader(handle)
        }


def read_serial_box(label_path: Path) -> tuple[float, float, float, float] | None:
    """Devuelve la caja de la clase serial_text en formato YOLO normalizado."""
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != SERIAL_CLASS_INDEX:
            continue
        return tuple(float(v) for v in parts[1:])  # type: ignore[return-value]
    return None


def to_pixels(box, width: int, height: int, margin: float):
    """Convierte de YOLO normalizado a píxeles, con un margen relativo."""
    cx, cy, bw, bh = box
    half_w = bw * width / 2 * (1 + margin)
    half_h = bh * height / 2 * (1 + margin)
    return (
        max(0.0, cx * width - half_w), max(0.0, cy * height - half_h),
        min(float(width), cx * width + half_w),
        min(float(height), cy * height + half_h),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/labels"))
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("datasets/ground_truth.csv"))
    parser.add_argument("--out", type=Path, default=Path("datasets/ocr"))
    parser.add_argument("--split-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--margin", type=float, default=0.20,
                        help="Margen alrededor de la caja del serial.")
    parser.add_argument("--min-width", type=int, default=MIN_CROP_WIDTH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"ERROR: no existe {args.labels}. Ejecute antes preannotate.py")
        return 1

    truth = load_ground_truth(args.ground_truth)

    # El reparto se hereda del split por cilindro: un recorte del mismo
    # cilindro no puede caer en entrenamiento y evaluación a la vez.
    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        listing = args.split_dir / f"{split}.txt"
        if listing.exists():
            for line in listing.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    split_of[Path(line.strip()).parts[0]] = split

    rng = random.Random(args.seed)
    written: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    skipped = Counter()
    sizes: list[tuple[int, int]] = []

    for label_path in sorted(args.labels.rglob("*.txt")):
        relative = label_path.relative_to(args.labels).with_suffix(".jpg")
        text = truth.get(str(relative))
        if not text:
            skipped["sin verdad de campo"] += 1
            continue

        box = read_serial_box(label_path)
        if box is None:
            skipped["sin caja de serial"] += 1
            continue

        image_path = args.root / relative
        if not image_path.exists():
            skipped["imagen ausente"] += 1
            continue

        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            skipped["imagen ilegible"] += 1
            continue

        image, _ = image_io.limit_size(image, 2048)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = to_pixels(box, width, height, args.margin)
        crop = image[int(y1):int(y2), int(x1):int(x2)]

        if crop.size == 0 or crop.shape[1] < args.min_width or crop.shape[0] < MIN_CROP_HEIGHT:
            skipped["recorte demasiado pequeño"] += 1
            continue

        cylinder = relative.parts[0]
        split = split_of.get(cylinder) or rng.choice(["train", "train", "val"])

        name = "__".join(relative.with_suffix(".jpg").parts)
        target = args.out / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), crop)

        written[split].append((f"{split}/{name}", text))
        sizes.append((crop.shape[1], crop.shape[0]))

    for split, rows in written.items():
        if not rows:
            continue
        listing = args.out / f"{split}_label.txt"
        listing.write_text(
            "\n".join(f"{path}\t{text}" for path, text in rows) + "\n",
            encoding="utf-8",
        )

    alphabet = sorted({c for rows in written.values() for _, t in rows for c in t})
    (args.out / "alphabet.txt").write_text("\n".join(alphabet) + "\n", encoding="utf-8")

    total = sum(len(v) for v in written.values())
    print("=" * 62)
    print("DATASET DE RECONOCIMIENTO")
    print("=" * 62)
    for split, rows in written.items():
        print(f"  {split:<6} {len(rows):>4} recortes")
    print(f"  {'total':<6} {total:>4}")

    if sizes:
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        print(f"\n  Tamaño mediano del recorte: "
              f"{int(np.median(widths))} x {int(np.median(heights))} px")

    print(f"\n  Alfabeto ({len(alphabet)} caracteres): {''.join(alphabet)}")
    if skipped:
        print("\n  Descartados:")
        for reason, count in skipped.most_common():
            print(f"    {reason:<28} {count}")

    print("=" * 62)
    print(f"\n  Recortes en {args.out}, etiquetas en <split>_label.txt")
    print("\n  Con esto se puede afinar el reconocedor. Conviene combinarlo con")
    print("  recortes sintéticos de troquelado para cubrir los caracteres poco")
    print("  frecuentes del alfabeto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
