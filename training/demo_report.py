#!/usr/bin/env python3
"""Genera evidencia visual del sistema funcionando sobre fotografías reales.

Produce un montaje con las regiones detectadas, el texto reconocido y el
resultado de la identificación, más un JSON con las respuestas completas.
Sirve para documentar el comportamiento del sistema y para anexar ejemplos
reales a un informe.

Las imágenes se toman del conjunto de TEST, es decir, de cilindros que el
modelo no vio durante el entrenamiento: lo contrario sería enseñar resultados
sobre datos memorizados.

Uso:
    python training/demo_report.py --count 6
    python training/demo_report.py --count 6 --condition 3.Cerca
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import build_container  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.parsing.serial import normalize_serial_charset  # noqa: E402
from app.preprocessing import image_io  # noqa: E402

PANEL_WIDTH = 520
COLOURS = {
    "marking_area": (60, 60, 230),
    "serial_text": (60, 200, 60),
    "cylinder": (200, 160, 40),
}


def annotate(image: np.ndarray, result, expected: str) -> np.ndarray:
    """Dibuja las detecciones y un pie con el resultado."""
    canvas = image.copy()
    for detection in result.detections:
        colour = COLOURS.get(detection.label, (200, 200, 200))
        box = detection.box
        cv2.rectangle(canvas, (int(box.x1), int(box.y1)),
                      (int(box.x2), int(box.y2)), colour, 4)
        cv2.putText(canvas, f"{detection.label} {detection.confidence:.2f}",
                    (int(box.x1), max(20, int(box.y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    scale = PANEL_WIDTH / canvas.shape[1]
    canvas = cv2.resize(canvas, (PANEL_WIDTH, int(canvas.shape[0] * scale)))

    read = result.data.numero_serie.value
    correct = bool(read) and normalize_serial_charset(read) == expected
    footer = np.full((120, PANEL_WIDTH, 3), 245, dtype=np.uint8)

    lines = [
        (f"esperado: {expected}", (40, 40, 40)),
        (f"leido:    {read or '-'}", (20, 140, 20) if correct else (20, 20, 200)),
        (f"{result.match.status.value} · {result.processing_time_ms} ms"
         f" · {'confirmar' if result.requires_manual_confirmation else 'automatico'}",
         (90, 90, 90)),
    ]
    for index, (text, colour) in enumerate(lines):
        cv2.putText(footer, text, (10, 32 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)

    return np.vstack([canvas, footer])


async def run(paths: list[tuple[Path, str]], out_dir: Path) -> list[dict]:
    container = build_container(get_settings())
    await container.backend.startup()
    panels: list[np.ndarray] = []
    payloads: list[dict] = []

    try:
        for index, (path, expected) in enumerate(paths, start=1):
            result = await container.pipeline.analyze(
                path.read_bytes(), request_id=f"demo-{index}"
            )
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            image, _ = image_io.limit_size(image, get_settings().max_image_side)

            panels.append(annotate(image, result, expected))
            payloads.append({
                "imagen": str(path),
                "esperado": expected,
                "respuesta": json.loads(result.model_dump_json()),
            })
            print(f"  {index}/{len(paths)}  {path.parent.parent.name}/"
                  f"{path.parent.name}  leido={result.data.numero_serie.value}")
    finally:
        await container.backend.shutdown()

    if panels:
        height = max(p.shape[0] for p in panels)
        padded = [
            np.pad(p, ((0, height - p.shape[0]), (0, 10), (0, 0)),
                   constant_values=255)
            for p in panels
        ]
        rows = [np.hstack(padded[i:i + 3]) for i in range(0, len(padded), 3)]
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)),
                       constant_values=255) for r in rows]
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "demo.jpg"), np.vstack(rows))

    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("datasets/ground_truth.csv"))
    parser.add_argument("--split", type=Path,
                        default=Path("datasets/processed/test.txt"))
    parser.add_argument("--out", type=Path, default=Path("models/demo"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--condition", default=None,
                        help="Filtra por condición, p. ej. 3.Cerca")
    args = parser.parse_args()

    truth = {
        row["imagen"]: normalize_serial_charset(row["numero_serie"])
        for row in csv.DictReader(args.ground_truth.open(encoding="utf-8"))
    }
    listing = [
        line.strip() for line in args.split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.condition:
        listing = [r for r in listing if args.condition in r]

    selected = [
        (args.images / rel, truth[rel])
        for rel in listing[: args.count] if rel in truth
    ]
    if not selected:
        print("ERROR: no hay imágenes que cumplan el filtro.")
        return 1

    print(f"Generando evidencia sobre {len(selected)} imágenes de test...")
    payloads = asyncio.run(run(selected, args.out))

    (args.out / "demo.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Montaje: {args.out / 'demo.jpg'}")
    print(f"  Respuestas completas: {args.out / 'demo.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
