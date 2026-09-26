#!/usr/bin/env python3
"""Construcción del catálogo de inventario y de la verdad de campo.

Lee los `Descripción.docx` de cada cilindro, normaliza sus campos y produce:

* `datasets/catalog.json`     - maestro de los cilindros conocidos, que el
                                servicio usa para resolver los seriales leídos;
* `datasets/ground_truth.csv` - imagen -> número de serie, necesario para
                                evaluar el sistema y calibrar los umbrales.

Las fichas están escritas a mano y traen las inconsistencias propias de eso:
unidades mezcladas, códigos de país en dos formatos, "SN" como marcador de
dato ausente y algún valor fuera de rango físico. El script normaliza lo que
puede e informa del resto en lugar de silenciarlo.

Uso:
    python training/build_catalog.py --root datasets/raw
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing import normalizers as norm  # noqa: E402
from app.parsing.fields import FIELD_LABELS  # noqa: E402
from app.parsing.records import normalize_record  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}



def docx_text(path: Path) -> str:
    """Extrae el texto plano de un .docx sin dependencias externas."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)


def parse_description(text: str) -> dict[str, str]:
    """Convierte el texto de la ficha en pares campo -> valor crudo."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = norm.clean_text(label)
        value = value.strip()
        if not label or not value:
            continue

        best: tuple[str, int] | None = None
        for field_name, aliases in FIELD_LABELS.items():
            for alias in aliases:
                if (label == alias or label.endswith(alias) or alias in label) and (
                    best is None or len(alias) > best[1]
                ):
                    best = (field_name, len(alias))
        if best and best[0] not in raw:
            raw[best[0]] = value
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--catalog", type=Path, default=Path("datasets/catalog.json"))
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("datasets/ground_truth.csv"))
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: no existe {args.root}")
        return 1

    records: list[dict] = []
    issues: list[str] = []
    rows: list[tuple[str, str]] = []
    by_serial: dict[str, list[str]] = {}
    without_description: list[str] = []

    for cylinder_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        docs = sorted(cylinder_dir.glob("Descripción/*.docx"))
        if not docs:
            without_description.append(cylinder_dir.name)
            continue

        try:
            raw = parse_description(docx_text(docs[0]))
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            issues.append(f"{cylinder_dir.name}: ficha ilegible ({exc.__class__.__name__}).")
            continue

        record, record_issues = normalize_record(cylinder_dir.name, raw)
        issues.extend(record_issues)
        if not record:
            continue

        record["cilindro"] = cylinder_dir.name
        records.append(record)
        by_serial.setdefault(record["numero_serie"], []).append(cylinder_dir.name)

        for image in sorted(cylinder_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((str(image.relative_to(args.root)), record["numero_serie"]))

    duplicates = {s: c for s, c in by_serial.items() if len(c) > 1}

    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(json.dumps({
        "version": "2026-09-20",
        "source": str(args.root),
        "count": len(records),
        "cylinders": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with args.ground_truth.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["imagen", "numero_serie"])
        writer.writerows(rows)

    filled = {}
    for record in records:
        for key in record:
            if key != "cilindro":
                filled[key] = filled.get(key, 0) + 1

    print("=" * 66)
    print("CATÁLOGO Y VERDAD DE CAMPO")
    print("=" * 66)
    print(f"  Cilindros en el catálogo ....... {len(records)}")
    print(f"  Seriales distintos ............. {len(by_serial)}")
    print(f"  Imágenes con verdad de campo ... {len(rows)}")
    print("\n  Cobertura por campo:")
    for key, count in sorted(filled.items(), key=lambda kv: -kv[1]):
        print(f"    {key:<26} {count:>3}/{len(records)}  ({count / len(records):.0%})")

    if duplicates:
        print("\n  SERIALES DUPLICADOS:")
        for serial, cylinders in duplicates.items():
            print(f"    {serial} -> {', '.join(cylinders)}")
        print("    Un serial repetido deja de ser identificador único: el")
        print("    sistema no podrá distinguir esos cilindros entre sí.")

    if without_description:
        print(f"\n  Cilindros sin ficha ({len(without_description)}): "
              f"{', '.join(without_description[:10])}")

    if issues:
        print(f"\n  Incidencias de normalización ({len(issues)}):")
        for issue in issues[:25]:
            print(f"    - {issue}")
        if len(issues) > 25:
            print(f"    ... y {len(issues) - 25} más")

    print("=" * 66)
    print(f"  {args.catalog}")
    print(f"  {args.ground_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
