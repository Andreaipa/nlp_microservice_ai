#!/usr/bin/env python3
"""Calibración empírica de los umbrales de confianza.

El servicio arranca con `AI_THRESHOLDS_CALIBRATED=false` y marca TODAS las
respuestas para confirmación manual. Es deliberado: elegir "0.9" porque suena
razonable no tiene ninguna base, y un umbral mal puesto produce errores
silenciosos, que es el peor fallo posible en un sistema de trazabilidad.

Este script recorre el conjunto de test, recoge la confianza y el acierto de
cada lectura, y busca el umbral más bajo que mantiene la tasa de error
silencioso por debajo del máximo tolerado. Se elige el más bajo porque cada
punto de umbral de más se paga en trabajo manual del operario.

Uso:
    python training/calibrate_thresholds.py --images datasets/raw \
        --ground-truth datasets/ground_truth.csv --max-silent-error-rate 0.01
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import build_container  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.parsing.serial import normalize_serial_charset  # noqa: E402
from app.schemas.analyze import MatchStatus  # noqa: E402
from training.evaluate_pipeline import load_ground_truth  # noqa: E402


async def collect_observations(
    images_root: Path, truth: dict[str, str], listing: list[str]
) -> list[dict]:
    """Ejecuta el pipeline y anota confianza frente a acierto real."""
    settings = get_settings()
    container = build_container(settings)
    await container.backend.startup()

    observations: list[dict] = []
    try:
        for index, relative in enumerate(listing, start=1):
            expected = truth.get(relative)
            path = images_root / relative
            if expected is None or not path.exists():
                continue

            result = await container.pipeline.analyze(
                path.read_bytes(), request_id=f"cal-{index}"
            )
            read = result.data.numero_serie.value
            observations.append({
                "imagen": relative,
                "esperado": expected,
                "leido": read,
                "confianza": result.data.numero_serie.confidence or 0.0,
                "correcto": bool(read) and normalize_serial_charset(read) == expected,
                "en_catalogo": result.match.status is MatchStatus.MATCHED,
            })
            if index % 25 == 0:
                print(f"  procesadas {index}/{len(listing)}...")
    finally:
        await container.backend.shutdown()
    return observations


def sweep(observations: list[dict], step: float = 0.01) -> list[dict]:
    """Calcula el comportamiento del sistema a cada umbral posible."""
    total = len(observations) or 1
    rows: list[dict] = []

    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        auto = [o for o in observations if o["leido"] and o["confianza"] >= threshold]
        auto_correct = [o for o in auto if o["correcto"]]
        silent_errors = [o for o in auto if not o["correcto"]]

        rows.append({
            "umbral": round(threshold, 3),
            "aceptados_automaticamente": len(auto),
            "aciertos_automaticos": len(auto_correct),
            "errores_silenciosos": len(silent_errors),
            "tasa_error_silencioso": len(silent_errors) / total,
            "tasa_automatizacion": len(auto) / total,
            "precision_automatica": len(auto_correct) / len(auto) if auto else 1.0,
            "tasa_confirmacion_manual": 1.0 - len(auto) / total,
        })
        threshold += step
    return rows


def recommend(rows: list[dict], max_silent_error_rate: float) -> dict | None:
    """Umbral mínimo que respeta el techo de error silencioso."""
    admissible = [r for r in rows if r["tasa_error_silencioso"] <= max_silent_error_rate]
    if not admissible:
        return None
    # Entre los admisibles, el que más automatiza (equivale al más bajo).
    return max(admissible, key=lambda r: (r["tasa_automatizacion"], -r["umbral"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--ground-truth", type=Path, default=Path("datasets/ground_truth.csv"))
    parser.add_argument("--split", type=Path, default=Path("datasets/processed/test.txt"))
    parser.add_argument("--max-silent-error-rate", type=float, default=0.01,
                        help="Tasa máxima tolerada de identificaciones erróneas "
                             "aceptadas sin confirmación (por defecto 1%%).")
    parser.add_argument("--json", type=Path, default=Path("models/threshold_calibration.json"))
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"ERROR: falta la verdad de campo en {args.ground_truth}.")
        return 1

    truth = load_ground_truth(args.ground_truth)
    listing = (
        [line.strip()
         for line in args.split.read_text(encoding="utf-8").splitlines()
         if line.strip()]
        if args.split.exists() else sorted(truth)
    )
    if not listing:
        print("ERROR: no hay imágenes que evaluar.")
        return 1

    print(f"Calibrando sobre {len(listing)} imágenes...")
    observations = asyncio.run(collect_observations(args.images, truth, listing))
    if not observations:
        print("ERROR: no se obtuvo ninguna observación.")
        return 1

    rows = sweep(observations)
    best = recommend(rows, args.max_silent_error_rate)

    print("\n" + "=" * 72)
    print("CALIBRACIÓN DE UMBRALES")
    print("=" * 72)
    print(f"{'umbral':>8} {'automatiz.':>12} {'precisión':>11} "
          f"{'err.silenc.':>12} {'manual':>9}")
    for row in rows:
        if abs(row["umbral"] * 100 % 5) < 1e-6:  # cada 0.05, para que quepa
            print(f"{row['umbral']:>8.2f} {row['tasa_automatizacion']:>11.1%} "
                  f"{row['precision_automatica']:>10.1%} "
                  f"{row['tasa_error_silencioso']:>11.1%} "
                  f"{row['tasa_confirmacion_manual']:>8.1%}")
    print("=" * 72)

    if best is None:
        print(f"\nNingún umbral mantiene el error silencioso por debajo de "
              f"{args.max_silent_error_rate:.1%}.")
        print("El sistema NO debe aceptar identificaciones automáticamente todavía.")
        print("Mantenga AI_THRESHOLDS_CALIBRATED=false y mejore el modelo o la captura.")
        exit_code = 1
    else:
        print(f"\nUmbral recomendado: {best['umbral']:.2f}")
        print(f"  Automatiza ................ {best['tasa_automatizacion']:.1%}")
        print(f"  Precisión automática ...... {best['precision_automatica']:.1%}")
        print(f"  Error silencioso .......... {best['tasa_error_silencioso']:.1%}")
        print(f"  Confirmación manual ....... {best['tasa_confirmacion_manual']:.1%}")
        print("\nConfiguración a aplicar en .env:")
        print(f"  AI_SERIAL_AUTO_ACCEPT_CONFIDENCE={best['umbral']:.2f}")
        print(f"  AI_SERIAL_REVIEW_CONFIDENCE={max(0.0, best['umbral'] - 0.30):.2f}")
        print("  AI_THRESHOLDS_CALIBRATED=true")
        exit_code = 0

    payload = {
        "observaciones": len(observations),
        "max_silent_error_rate": args.max_silent_error_rate,
        "recomendado": best,
        "curva": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCurva completa guardada en {args.json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
