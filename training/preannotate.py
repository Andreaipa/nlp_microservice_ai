#!/usr/bin/env python3
"""Pre-anotación automática guiada por la verdad de campo.

Anotar 599 fotografías a mano es el cuello de botella para entrenar el
detector. Este script reduce el trabajo a revisar propuestas en lugar de
dibujar cajas desde cero.

La idea que lo hace fiable: **ya sabemos qué número de serie lleva cada
fotografía**, porque está en la ficha del cilindro. Así que no hay que
adivinar dónde está el troquelado; basta ejecutar el OCR sobre la imagen y
quedarse con la región cuyo texto se parece al serial que debe aparecer. Eso
convierte un problema de detección no supervisada en una simple verificación.

De cada acierto se derivan dos cajas:

* `serial_text`   - la línea reconocida;
* `marking_area`  - su entorno expandido, que es lo que el servicio recorta
                    para leer todos los campos.

La clase `cylinder` NO se pre-anota: delimitar el envase completo no se puede
deducir del texto y requiere criterio humano.

Lo que el script produce son PROPUESTAS. Deben revisarse en Label Studio o
CVAT antes de entrenar: una caja mal puesta enseña al modelo a mirar donde no
debe.

Uso:
    python training/preannotate.py --root datasets/raw --ground-truth datasets/ground_truth.csv
    python training/preannotate.py --limit 20          # prueba rápida
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing.serial import (  # noqa: E402
    normalize_serial_charset,
    weighted_edit_distance,
)
from app.preprocessing import enhance, image_io  # noqa: E402
from app.vision.base import DetectionClass  # noqa: E402

CLASS_INDEX = {name: index for index, name in enumerate(DetectionClass.ALL)}

# Criterio de aceptación de una línea como el serial buscado.
#
# La primera versión sólo exigía distancia de edición <= 3, y al revisar las
# cajas se vio que en las tomas frontales producía falsos positivos: sobre una
# imagen de 9 MP el OCR devuelve muchos fragmentos, y con un serial de nueve
# caracteres una distancia de 3 la alcanza cualquier cosa. Una caja mal puesta
# es peor que ninguna, porque enseña al detector a mirar donde no debe.
#
# Ahora se exige además que el texto tenga una longitud comparable a la del
# serial y que el OCR lo haya leído con cierta seguridad. El criterio es
# deliberadamente estricto: se prefiere anotar menos y bien.
MAX_DISTANCE_RATIO = 0.25          # como fracción de la longitud del serial
MAX_SERIAL_DISTANCE = 3.0          # tope absoluto
MIN_LENGTH_RATIO = 0.7             # longitud mínima del texto frente al serial
MIN_OCR_CONFIDENCE = 0.50

# Un fragmento contenido literalmente en el serial esperado se acepta aunque
# sea más corto. Leer "206055" dentro de "19S206055" no ocurre por azar: son
# seis caracteres exactos y en el mismo orden. Es frecuente que el OCR pierda
# los primeros dígitos del troquelado, más desgastados, y descartar esas
# lecturas dejaba fuera anotaciones perfectamente válidas.
MIN_SUBSTRING_LENGTH = 5

# Filtros geométricos. Al revisar las cajas propuestas se encontró que sobre
# el cuerpo del cilindro, cubierto de pintura descascarada, PaddleOCR llega a
# devolver cadenas parecidas al serial con 0.96 de confianza sobre lo que sólo
# es textura. Dos restricciones eliminan esos casos sin perder los buenos:
#
#   * el troquelado se lee en horizontal, así que su caja es más ancha que
#     alta; las alucinaciones sobre textura salían verticales;
#   * está estampado en el hombro, en la parte superior del envase, nunca en
#     la mitad inferior del cuerpo.
MIN_ASPECT_RATIO = 1.5
# Fracción de la altura de la IMAGEN COMPLETA por debajo de la cual no se
# acepta un hallazgo. Ojo: la búsqueda se hace sobre una banda superior, así
# que la posición hay que convertirla antes de comparar. Calcularla respecto
# a la banda hacía el filtro mucho más estricto de lo previsto y descartaba
# lecturas correctas en las tomas cercanas, donde el cilindro llena el
# encuadre y el troquelado no queda tan arriba.
MAX_VERTICAL_POSITION = 0.60

# Acotar la búsqueda antes de llamar al OCR. Sobre las fotografías completas
# de 9 MP cada imagen costaba más de un minuto, lo que hacía inviable recorrer
# las 599. Dos recortes lo reducen a una fracción sin perder cobertura:
#
#   * sólo la franja superior, porque el troquelado está en el hombro y el
#     filtro de posición ya descartaba cualquier hallazgo por debajo;
# La resolución NO se reduce. Se probó a trabajar a 1400 px y la cobertura
# cayó a la mitad: el troquelado es texto de bajo contraste y pierde
# legibilidad enseguida. La velocidad se consigue repartiendo las imágenes
# entre varios procesos, no recortando píxeles.
SEARCH_HEIGHT_FRACTION = 0.60
SEARCH_MAX_SIDE = 2048

# Escalas a las que se busca. En las tomas lejanas y frontales el troquelado
# mide unos pocos píxeles de alto y el detector de texto sencillamente no lo
# ve; ampliarlo le da una oportunidad. Antes esto era impensable por coste,
# pero con RapidOCR una pasada cuesta ~130 ms en vez de ~1,2 s, así que
# explorar varias escalas sale a cuenta.
SEARCH_SCALES = (1.0, 2.0)

# Cuánto se expande la caja del serial para abarcar el resto del troquelado.
# El bloque estampado ocupa varias líneas bajo el número de serie.
MARKING_PAD_X = 0.35
MARKING_PAD_ABOVE = 0.8
MARKING_PAD_BELOW = 2.5


@dataclass
class Outcome:
    annotated: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    per_condition: dict[str, list[int]] = field(default_factory=dict)


def load_ground_truth(path: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            truth[row["imagen"]] = normalize_serial_charset(row["numero_serie"])
    return truth


def locate_serial(
    image: np.ndarray, expected: str, ocr, variants: int
) -> tuple[tuple[float, float, float, float], float, str] | None:
    """Busca en la imagen la línea que mejor coincide con el serial esperado.

    Las coordenadas devueltas corresponden siempre a `image`, aunque la
    búsqueda se haga sobre una versión recortada y reducida.
    """
    full_height, full_width = image.shape[:2]
    original_band = image[: int(full_height * SEARCH_HEIGHT_FRACTION)]
    original_band, band_scale = image_io.limit_size(original_band, SEARCH_MAX_SIDE)

    best: tuple[tuple[float, float, float, float], float, str] | None = None
    threshold = min(MAX_SERIAL_DISTANCE, max(1.0, len(expected) * MAX_DISTANCE_RATIO))
    best_distance = threshold
    min_length = max(4, int(len(expected) * MIN_LENGTH_RATIO))

    for scale in SEARCH_SCALES:
        if scale == 1.0:
            band = original_band
        else:
            band = cv2.resize(original_band, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        height = band.shape[0]

        for _, prepared in enhance.build_ocr_variants(band, variants):
            for line in ocr.read(prepared):
                if line.box is None or line.confidence < MIN_OCR_CONFIDENCE:
                    continue

                box_height = max(line.box.y2 - line.box.y1, 1.0)
                if (line.box.x2 - line.box.x1) / box_height < MIN_ASPECT_RATIO:
                    continue
                centre_y = (line.box.y1 + line.box.y2) / 2.0
                scale = height / max(prepared.shape[0], 1)
                # Posición dentro de la banda, y de ahí a la imagen completa.
                position_in_band = (centre_y * scale) / height
                if position_in_band * SEARCH_HEIGHT_FRACTION > MAX_VERTICAL_POSITION:
                    continue

                candidate = normalize_serial_charset(line.text)
                if not candidate:
                    continue

                is_substring = (
                    len(candidate) >= MIN_SUBSTRING_LENGTH and candidate in expected
                )
                if len(candidate) < min_length and not is_substring:
                    continue

                distance = weighted_edit_distance(expected, candidate)
                # El serial puede venir acompañado de ruido en la misma línea, o
                # aparecer recortado. Ambos casos son localizaciones válidas.
                if expected in candidate or is_substring:
                    distance = min(distance, 0.5)

                if distance < best_distance:
                    # Del realce a la banda, y de la banda a la imagen original.
                    # Del realce a la banda ampliada, de ahí a la banda
                    # original, y de ahí a la imagen de entrada.
                    scale_x = (band.shape[1] / max(prepared.shape[1], 1)
                               / scale / band_scale)
                    scale_y = (band.shape[0] / max(prepared.shape[0], 1)
                               / scale / band_scale)
                    best = (
                        (line.box.x1 * scale_x, line.box.y1 * scale_y,
                         line.box.x2 * scale_x, line.box.y2 * scale_y),
                        line.confidence,
                        line.text,
                    )
                    best_distance = distance
    return best


def to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple:
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    return (
        ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
        (x2 - x1) / width, (y2 - y1) / height,
    )


def expand_marking(box, width: int, height: int):
    """Expande la caja del serial hasta cubrir el bloque troquelado."""
    x1, y1, x2, y2 = box
    line_height = max(y2 - y1, 1.0)
    span = x2 - x1
    return (
        max(0.0, x1 - span * MARKING_PAD_X),
        max(0.0, y1 - line_height * MARKING_PAD_ABOVE),
        min(float(width), x2 + span * MARKING_PAD_X),
        min(float(height), y2 + line_height * MARKING_PAD_BELOW),
    )


# --- Ejecución en paralelo -------------------------------------------------
# Cada imagen tarda unos diez segundos en CPU, así que recorrer las 599 en
# serie lleva más de hora y media. El trabajo es independiente por imagen, de
# modo que se reparte entre varios procesos. Cada uno carga su propio motor
# OCR (unos 1,5 GB), y de ahí que el número de procesos por omisión sea
# conservador y no el total de núcleos.
_WORKER: dict = {}


def _init_worker(variants: int, root: str, out: str) -> None:
    from app.core.config import get_settings
    from app.ocr.registry import build_ocr_engine

    settings = get_settings()
    _WORKER["ocr"] = build_ocr_engine(settings)
    _WORKER["variants"] = variants
    _WORKER["max_side"] = settings.max_image_side
    _WORKER["root"] = Path(root)
    _WORKER["out"] = Path(out)


def _process_one(task: tuple[str, str]) -> tuple[str, str, bool]:
    """Localiza el serial de una imagen y escribe su etiqueta si lo encuentra."""
    relative, expected = task
    parts = Path(relative).parts
    condition = parts[1] if len(parts) > 2 else "?"

    path = _WORKER["root"] / relative
    if not path.exists():
        return relative, condition, False

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return relative, condition, False

    image, _ = image_io.limit_size(image, _WORKER["max_side"])
    height, width = image.shape[:2]

    found = locate_serial(image, expected, _WORKER["ocr"], _WORKER["variants"])
    if found is None:
        return relative, condition, False

    box, _confidence, _text = found
    marking = expand_marking(box, width, height)

    label_path = _WORKER["out"] / Path(relative).with_suffix(".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        f"{CLASS_INDEX[DetectionClass.MARKING_AREA]} "
        + " ".join(f"{v:.6f}" for v in to_yolo(marking, width, height)) + "\n"
        + f"{CLASS_INDEX[DetectionClass.SERIAL_TEXT]} "
        + " ".join(f"{v:.6f}" for v in to_yolo(box, width, height)) + "\n",
        encoding="utf-8",
    )
    return relative, condition, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--ground-truth", type=Path, default=Path("datasets/ground_truth.csv"))
    parser.add_argument("--out", type=Path, default=Path("datasets/labels"))
    parser.add_argument("--limit", type=int, default=0, help="0 = todas.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Salta las primeras N imágenes; útil para muestrear.")
    parser.add_argument("--variants", type=int, default=3,
                        help="Variantes de realce por imagen. Con 2 la cobertura "
                             "cae a la mitad: 3 es el mínimo razonable.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Procesos en paralelo. Cada uno carga su propio "
                             "motor OCR (~1,5 GB).")
    parser.add_argument("--json", type=Path, default=Path("models/preannotation_report.json"))
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"ERROR: falta {args.ground_truth}. Ejecute antes build_catalog.py")
        return 1

    truth = load_ground_truth(args.ground_truth)
    images = sorted(truth)[args.offset:]
    if args.limit:
        images = images[: args.limit]

    outcome = Outcome()
    tasks = [(relative, truth[relative]) for relative in images]
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    print(f"Pre-anotando {len(tasks)} imágenes con {workers} procesos...")

    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(args.variants, str(args.root), str(args.out)),
    ) as pool:
        for index, (relative, condition, ok) in enumerate(
            pool.map(_process_one, tasks, chunksize=1), start=1
        ):
            outcome.per_condition.setdefault(condition, [0, 0])
            outcome.per_condition[condition][1] += 1
            if ok:
                outcome.annotated.append(relative)
                outcome.per_condition[condition][0] += 1
            else:
                outcome.missed.append(relative)

            if index % 25 == 0:
                rate = len(outcome.annotated) / index
                print(f"  {index}/{len(tasks)}  localizados: "
                      f"{len(outcome.annotated)} ({rate:.0%})")

    total = len(outcome.annotated) + len(outcome.missed)
    print("\n" + "=" * 66)
    print("PRE-ANOTACIÓN")
    print("=" * 66)
    print(f"  Imágenes procesadas ........ {total}")
    print(f"  Con serial localizado ...... {len(outcome.annotated)} "
          f"({len(outcome.annotated) / max(total, 1):.1%})")
    print(f"  Sin localizar .............. {len(outcome.missed)}")
    print("\n  Por condición de captura:")
    for condition, (hits, seen) in sorted(outcome.per_condition.items()):
        print(f"    {condition:<20} {hits:>3}/{seen:<3} ({hits / max(seen, 1):.0%})")
    print("=" * 66)
    print("\n  Las cajas son PROPUESTAS. Revíselas antes de entrenar:")
    print("  la clase 'cylinder' no se pre-anota y hay que añadirla a mano.")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "total": total,
        "annotated": len(outcome.annotated),
        "missed": outcome.missed,
        "per_condition": {k: {"hits": v[0], "seen": v[1]}
                          for k, v in outcome.per_condition.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Informe: {args.json}")
    print(f"  Etiquetas: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
