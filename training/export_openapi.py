#!/usr/bin/env python3
"""Exporta el esquema OpenAPI a un fichero.

Permite versionar el contrato, publicarlo en la documentación y generar
clientes sin necesidad de levantar el servicio.

Uso:
    python training/export_openapi.py --out docs/openapi.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/openapi.json"))
    args = parser.parse_args()

    from app.main import app

    schema = app.openapi()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    paths = len(schema.get("paths", {}))
    schemas = len(schema.get("components", {}).get("schemas", {}))
    print(f"OpenAPI {schema['openapi']} exportado a {args.out}")
    print(f"  rutas: {paths} · modelos: {schemas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
