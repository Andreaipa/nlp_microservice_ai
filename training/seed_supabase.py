#!/usr/bin/env python3
"""Registra en la tabla `cilindros` de Supabase los cilindros del dataset.

Por qué hace falta: el inventario de la aplicación móvil y el catálogo del
servicio de IA nacieron por separado. La IA conoce los 99 cilindros
fotografiados del dataset; la aplicación sólo tiene los que un operario haya
dado de alta a mano. Resultado: cuando la IA reconoce un cilindro del dataset,
la aplicación avisa de que "no está registrado en el inventario" y no deja
darle salida.

Este script cierra ese hueco dando de alta esos cilindros con el mismo
formato que usa el formulario de registro de la aplicación.

Es SEGURO por defecto:

* sin --apply sólo muestra lo que haría; no escribe nada;
* nunca duplica: omite los seriales que ya existen en la tabla;
* no da de alta los seriales repetidos en origen (K5738166 figura en dos
  cilindros distintos): una base de datos con el mismo serial dos veces rompe
  la trazabilidad, y decidir cuál es el bueno le corresponde a una persona.

Uso:
    # Ver qué se registraría (no escribe nada)
    python training/seed_supabase.py

    # Registrar de verdad, a nombre del propietario indicado
    SUPABASE_KEY=... python training/seed_supabase.py --apply --cliente-id 12

La clave se lee de la variable de entorno SUPABASE_KEY. Con la clave pública
(anon) la inserción sólo funcionará si las políticas RLS lo permiten; si no,
hace falta una clave con permisos de escritura, que nunca debe guardarse en el
repositorio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing.serial import normalize_serial_charset  # noqa: E402

# Códigos del catálogo -> nombres que usa el formulario de la aplicación.
GAS_APP = {
    "O2": "OXÍGENO",
    "ACET": "ACETILENO",
    "AR": "ARGÓN",
    "N2": "NITRÓGENO",
    "CO2": "DIÓXIDO DE CARBONO",
    "MIX": "MEZCLA",
}

# La aplicación asigna esta periodicidad de prueba hidrostática por defecto.
GASES_5_ANIOS = {"DIÓXIDO DE CARBONO", "MEZCLA"}


def to_app_payload(record: dict[str, Any], cliente_id: int | None) -> dict[str, Any]:
    """Convierte un registro del catálogo al formato del formulario de la app."""
    gas = GAS_APP.get(str(record.get("tipo_gas") or ""), "")

    if record.get("ph_periodicidad_anios"):
        ph = f"CADA {record['ph_periodicidad_anios']} AÑOS PH"
    elif record.get("ph"):
        year, month = str(record["ph"]).split("-")
        ph = f"{month}/{year}"
    else:
        ph = "CADA 5 AÑOS PH" if gas in GASES_5_ANIOS else "CADA 10 AÑOS PH"

    fecha = record.get("fecha_fabricacion")
    anio = f"{fecha[5:7]}/{fecha[:4]}" if fecha else record.get("anio_fabricacion")

    litros = record.get("litros")
    return {
        "numero_serie": normalize_serial_charset(str(record["numero_serie"])),
        "cliente_id": cliente_id,
        "tipo_gas": gas or None,
        "marca": record.get("marca"),
        "anio_fabricacion": str(anio) if anio else None,
        "unidad_medida": record.get("unidad_medida"),
        "peso": record.get("peso"),
        "ancho_diametro": record.get("ancho_diametro"),
        "largo_altura": record.get("largo_altura"),
        # Como en el formulario: el acetileno se registra por contenido.
        "litros": litros if gas != "ACETILENO" else None,
        "contenido": litros if gas == "ACETILENO" else None,
        "color": str(record["color"]).upper() if record.get("color") else None,
        "ph": ph,
        "procedencia": record.get("procedencia"),
    }


def existing_serials(client: httpx.Client) -> set[str]:
    response = client.get("/cilindros", params={"select": "numero_serie"})
    response.raise_for_status()
    return {normalize_serial_charset(str(r["numero_serie"])) for r in response.json()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=Path("datasets/catalog.json"))
    parser.add_argument("--url", default=os.environ.get("SUPABASE_URL"),
                        help="URL del proyecto Supabase (o variable SUPABASE_URL).")
    parser.add_argument("--apply", action="store_true",
                        help="Escribe de verdad. Sin esto sólo se simula.")
    parser.add_argument("--cliente-id", type=int, default=None,
                        help="Propietario a nombre del que se registran. "
                             "Obligatorio con --apply, como en el formulario.")
    args = parser.parse_args()

    key = os.environ.get("SUPABASE_KEY")
    if not args.url or not key:
        print("ERROR: indique --url (o SUPABASE_URL) y la variable SUPABASE_KEY.")
        return 1
    if args.apply and args.cliente_id is None:
        print("ERROR: --apply exige --cliente-id: el formulario de la aplicación no")
        print("       permite registrar un cilindro sin propietario.")
        return 1

    records = json.loads(args.catalog.read_text(encoding="utf-8"))["cylinders"]
    counts = Counter(normalize_serial_charset(str(r["numero_serie"])) for r in records)
    duplicated = {serial for serial, n in counts.items() if n > 1}

    client = httpx.Client(
        base_url=args.url.rstrip("/") + "/rest/v1",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=15,
    )
    try:
        already = existing_serials(client)
    except httpx.HTTPError as exc:
        print(f"ERROR: no se pudo leer la tabla cilindros: {exc}")
        return 1

    to_insert: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        serial = normalize_serial_charset(str(record["numero_serie"]))
        if serial in already or serial in duplicated or serial in seen:
            continue
        seen.add(serial)
        to_insert.append(to_app_payload(record, args.cliente_id))

    print("=" * 64)
    print("ALTA DEL DATASET EN SUPABASE" + ("" if args.apply else "  (simulación)"))
    print("=" * 64)
    print(f"  En el catálogo del dataset ...... {len(counts)} seriales")
    print(f"  Ya registrados en la app ........ {len(already & set(counts))}")
    print(f"  Omitidos por duplicados ......... {len(duplicated)}"
          + (f"  ({', '.join(sorted(duplicated))})" if duplicated else ""))
    print(f"  Por registrar ................... {len(to_insert)}")

    if to_insert:
        print("\n  Ejemplo del primer registro:")
        for key_name, value in to_insert[0].items():
            print(f"    {key_name:<16} {value}")

    if not args.apply:
        print("\n  No se ha escrito nada. Para registrar de verdad:")
        print("    SUPABASE_KEY=... python training/seed_supabase.py --apply --cliente-id <id>")
        return 0

    if not to_insert:
        print("\n  Nada que registrar.")
        return 0

    response = client.post(
        "/cilindros", json=to_insert, headers={"Prefer": "return=minimal"}
    )
    if response.status_code >= 400:
        print(f"\n  ERROR {response.status_code}: {response.text[:300]}")
        if response.status_code in (401, 403):
            print("  La clave no tiene permiso de escritura (políticas RLS).")
        return 1

    print(f"\n  Registrados {len(to_insert)} cilindros.")
    print("  El servicio de IA los tomará en su próxima sincronización (≤ 60 s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
