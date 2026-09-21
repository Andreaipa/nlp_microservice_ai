#!/usr/bin/env python3
"""Evaluación de extremo a extremo del sistema completo.

Ésta es la métrica que decide si el proyecto sirve. No importa el mAP del
detector ni la precisión aislada del OCR: importa qué porcentaje de
fotografías permite identificar correctamente el cilindro, que es lo que hará
un operario con el móvil en la mano.

Se reportan cuatro cantidades, y las cuatro importan:

* aciertos          - identificó y acertó. Es el objetivo.
* errores silenciosos - identificó y se equivocó SIN pedir confirmación. Es el
                      fallo grave: registra un movimiento en el cilindro
                      equivocado y corrompe la trazabilidad. Debe tender a cero
                      aunque cueste aciertos.
* derivados         - pidió confirmación manual. Es un coste operativo
                      aceptable, no un fallo.
* no identificados  - no leyó nada utilizable.

Requiere un fichero de verdad de campo que asocie cada imagen a su serial:

    ground_truth.csv
        imagen,numero_serie
        Cilindro_002/5.Luz normal/IMG_001.jpg,19S206055

Uso:
    python training/evaluate_pipeline.py --images datasets/raw \
        --ground-truth datasets/ground_truth.csv --split datasets/processed/test.txt
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import build_container  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.parsing.serial import normalize_serial_charset, weighted_edit_distance  # noqa: E402


def load_ground_truth(path: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image = (row.get("imagen") or row.get("image") or "").strip()
            serial = (row.get("numero_serie") or row.get("serial") or "").strip()
            if image and serial:
                truth[image] = normalize_serial_charset(serial)
    return truth


def character_error_rate(expected: str, actual: str) -> float:
    """CER clásico: ediciones necesarias entre longitud de referencia."""
    if not expected:
        return 0.0 if not actual else 1.0
    # Aquí se usa distancia sin ponderar: para medir al OCR, una confusión
    # sigue siendo un error de un carácter.
    previous = list(range(len(actual) + 1))
    for i, char_e in enumerate(expected, start=1):
        current = [i]
        for j, char_a in enumerate(actual, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_e != char_a)))
        previous = current
    return previous[-1] / len(expected)


async def evaluate(
    images_root: Path,
    truth: dict[str, str],
    listing: list[str],
    settings: Settings,
) -> dict:
    container = build_container(settings)
    await container.backend.startup()
    pipeline = container.pipeline

    outcomes: Counter = Counter()
    latencies: list[float] = []
    cers: list[float] = []
    silent_errors: list[dict] = []
    per_condition: dict[str, Counter] = {}

    try:
        for index, relative in enumerate(listing, start=1):
            expected = truth.get(relative)
            if expected is None:
                outcomes["sin_verdad_de_campo"] += 1
                continue

            path = images_root / relative
            if not path.exists():
                outcomes["fichero_ausente"] += 1
                continue

            # La condición de captura es la carpeta intermedia; permite saber
            # en qué escenario falla el sistema.
            parts = Path(relative).parts
            condition = parts[1] if len(parts) > 2 else "(sin condición)"
            per_condition.setdefault(condition, Counter())

            started = time.perf_counter()
            result = await pipeline.analyze(path.read_bytes(), request_id=f"eval-{index}")
            latencies.append((time.perf_counter() - started) * 1000)

            read = result.data.numero_serie.value
            needs_review = result.requires_manual_confirmation

            if read:
                cers.append(character_error_rate(expected, normalize_serial_charset(read)))

            if read and normalize_serial_charset(read) == expected:
                outcome = "derivado_correcto" if needs_review else "acierto"
            elif read and needs_review:
                outcome = "derivado_incorrecto"
            elif read:
                outcome = "error_silencioso"
                silent_errors.append({
                    "imagen": relative, "esperado": expected, "leido": read,
                    "confianza": result.data.numero_serie.confidence,
                    "distancia": weighted_edit_distance(expected, normalize_serial_charset(read)),
                })
            else:
                outcome = "derivado_sin_lectura" if needs_review else "no_identificado"

            outcomes[outcome] += 1
            per_condition[condition][outcome] += 1

            if index % 25 == 0:
                print(f"  procesadas {index}/{len(listing)}...")
    finally:
        await container.backend.shutdown()

    evaluated = sum(
        outcomes[k] for k in
        ("acierto", "derivado_correcto", "derivado_incorrecto",
         "error_silencioso", "derivado_sin_lectura", "no_identificado")
    ) or 1

    return {
        "imagenes_evaluadas": evaluated,
        "resultados": dict(outcomes),
        "tasa_acierto_automatico": outcomes["acierto"] / evaluated,
        "tasa_acierto_total": (outcomes["acierto"] + outcomes["derivado_correcto"]) / evaluated,
        "tasa_error_silencioso": outcomes["error_silencioso"] / evaluated,
        "tasa_confirmacion_manual": (
            outcomes["derivado_correcto"] + outcomes["derivado_incorrecto"]
            + outcomes["derivado_sin_lectura"]
        ) / evaluated,
        "cer_medio": statistics.fmean(cers) if cers else None,
        "latencia_ms": {
            "media": round(statistics.fmean(latencies), 1) if latencies else None,
            "mediana": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
            if len(latencies) >= 20 else None,
        },
        "por_condicion": {k: dict(v) for k, v in per_condition.items()},
        "errores_silenciosos": silent_errors[:50],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--ground-truth", type=Path, default=Path("datasets/ground_truth.csv"))
    parser.add_argument("--split", type=Path, default=Path("datasets/processed/test.txt"),
                        help="Listado de imágenes a evaluar (por defecto, el test).")
    parser.add_argument("--json", type=Path, default=Path("models/pipeline_metrics.json"))
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"ERROR: falta la verdad de campo en {args.ground_truth}.")
        print("       Formato esperado: imagen,numero_serie")
        return 1

    truth = load_ground_truth(args.ground_truth)
    if not truth:
        print("ERROR: el fichero de verdad de campo está vacío.")
        return 1

    if args.split.exists():
        listing = [
            line.strip()
            for line in args.split.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        print(f"AVISO: no existe {args.split}; se evalúan todas las imágenes con verdad de campo.")
        listing = sorted(truth)

    if not listing:
        print("ERROR: no hay imágenes que evaluar.")
        return 1

    print(f"Evaluando {len(listing)} imágenes...")
    settings = get_settings()
    report = asyncio.run(evaluate(args.images, truth, listing, settings))

    print("\n" + "=" * 66)
    print("EVALUACIÓN DE EXTREMO A EXTREMO")
    print("=" * 66)
    print(f"  Imágenes evaluadas ............. {report['imagenes_evaluadas']}")
    print(f"  Identificación automática ...... {report['tasa_acierto_automatico']:.1%}")
    print(f"  Identificación correcta total .. {report['tasa_acierto_total']:.1%}")
    print(f"  ERRORES SILENCIOSOS ............ {report['tasa_error_silencioso']:.1%}")
    print(f"  Requiere confirmación manual ... {report['tasa_confirmacion_manual']:.1%}")
    if report["cer_medio"] is not None:
        print(f"  CER medio del serial ........... {report['cer_medio']:.3f}")
    latency = report["latencia_ms"]
    print(f"  Latencia mediana / p95 ......... {latency['mediana']} ms / {latency['p95']} ms")

    if report["por_condicion"]:
        print("\n  Desglose por condición de captura:")
        for condition, counts in sorted(report["por_condicion"].items()):
            total = sum(counts.values()) or 1
            hits = counts.get("acierto", 0) + counts.get("derivado_correcto", 0)
            print(f"    {condition:<20} {hits}/{total} correctos ({hits/total:.0%})")

    if report["errores_silenciosos"]:
        print(f"\n  Errores silenciosos ({len(report['errores_silenciosos'])} primeros):")
        for error in report["errores_silenciosos"][:10]:
            print(f"    {error['imagen']}: esperado {error['esperado']}, "
                  f"leído {error['leido']} (conf {error['confianza']})")
        print("\n  Un error silencioso registra el movimiento en el cilindro")
        print("  equivocado. Si esta tasa no es casi nula, suba el umbral de")
        print("  aceptación automática aunque baje la tasa de acierto.")
    print("=" * 66)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme guardado en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
