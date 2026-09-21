#!/usr/bin/env python3
"""Auditoría del dataset de fotografías antes de anotar o entrenar.

Un dataset no está listo por el hecho de existir. Este script responde a las
preguntas que condicionan todas las decisiones posteriores: cuántos cilindros
distintos hay, cuántas fotos por cilindro, si están corruptas o duplicadas, y
si las condiciones de captura cubren lo que la aplicación móvil encontrará en
planta.

Estructura esperada (la que ya usa la carpeta de Drive del proyecto):

    datasets/raw/
        Cilindro_002/
            1.Frontal/*.jpg
            2.Diagonal/*.jpg
            ...
            Descripción/Descripción.docx
        Cilindro_003/
            ...

Uso:
    python training/audit_dataset.py --root datasets/raw
    python training/audit_dataset.py --root datasets/raw --json informe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preprocessing.image_io import assess_quality  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

# Mínimos por debajo de los cuales entrenar no es razonable. Provienen de la
# aritmética del propio proyecto: con separación por identidad de cilindro,
# hacen falta suficientes cilindros distintos para que validación y test
# midan generalización y no memorización.
MIN_CYLINDERS_FOR_SPLIT = 10
MIN_IMAGES_PER_CYLINDER = 5
MIN_TOTAL_IMAGES = 300


@dataclass
class ImageRecord:
    path: str
    cylinder: str
    condition: str
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    digest: str = ""
    corrupt: bool = False


@dataclass
class AuditReport:
    root: str
    total_files: int = 0
    total_images: int = 0
    cylinders: int = 0
    corrupt: list[str] = field(default_factory=list)
    duplicates: list[list[str]] = field(default_factory=list)
    images_per_cylinder: dict[str, int] = field(default_factory=dict)
    images_per_condition: dict[str, int] = field(default_factory=dict)
    empty_cylinders: list[str] = field(default_factory=list)
    resolutions: dict[str, int] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    low_quality_images: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ready_to_train: bool = False


def digest_of(path: Path) -> str:
    """Hash del contenido, para detectar duplicados exactos."""
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image(path: Path, cylinder: str, condition: str) -> ImageRecord:
    record = ImageRecord(path=str(path), cylinder=cylinder, condition=condition)
    try:
        record.digest = digest_of(path)
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        record.corrupt = True
        return record

    if image is None:
        record.corrupt = True
        return record

    quality = assess_quality(image)
    record.width = quality.width
    record.height = quality.height
    record.megapixels = round(quality.width * quality.height / 1e6, 2)
    record.blur_score = quality.blur_score
    record.brightness = quality.brightness
    record.contrast = quality.contrast
    return record


def collect(root: Path) -> tuple[list[ImageRecord], list[str], int]:
    """Recorre la estructura y devuelve registros, cilindros vacíos y total."""
    records: list[ImageRecord] = []
    empty: list[str] = []
    total_files = 0

    cylinder_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for cylinder_dir in cylinder_dirs:
        found = 0
        for path in sorted(cylinder_dir.rglob("*")):
            if not path.is_file():
                continue
            total_files += 1
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # La condición es la carpeta inmediatamente bajo el cilindro.
            try:
                relative = path.relative_to(cylinder_dir)
                condition = relative.parts[0] if len(relative.parts) > 1 else "(raíz)"
            except ValueError:
                condition = "(desconocida)"
            records.append(inspect_image(path, cylinder_dir.name, condition))
            found += 1
        if found == 0:
            empty.append(cylinder_dir.name)

    return records, empty, total_files


def build_report(root: Path) -> AuditReport:
    report = AuditReport(root=str(root))

    if not root.exists():
        report.blockers.append(f"El directorio {root} no existe.")
        return report

    records, empty, total_files = collect(root)
    report.total_files = total_files
    report.total_images = len([r for r in records if not r.corrupt])
    report.empty_cylinders = empty

    valid = [r for r in records if not r.corrupt]
    report.corrupt = [r.path for r in records if r.corrupt]

    per_cylinder = Counter(r.cylinder for r in valid)
    report.images_per_cylinder = dict(sorted(per_cylinder.items()))
    report.cylinders = len(set(per_cylinder) | set(empty))
    report.images_per_condition = dict(sorted(Counter(r.condition for r in valid).items()))
    report.resolutions = dict(
        sorted(Counter(f"{r.width}x{r.height}" for r in valid).items(),
               key=lambda item: -item[1])[:10]
    )

    by_digest: dict[str, list[str]] = defaultdict(list)
    for record in valid:
        if record.digest:
            by_digest[record.digest].append(record.path)
    report.duplicates = [paths for paths in by_digest.values() if len(paths) > 1]

    if valid:
        report.quality = {
            "blur_mediana": round(float(np.median([r.blur_score for r in valid])), 2),
            "brillo_medio": round(float(np.mean([r.brightness for r in valid])), 2),
            "contraste_medio": round(float(np.mean([r.contrast for r in valid])), 2),
            "megapixeles_mediana": round(float(np.median([r.megapixels for r in valid])), 2),
        }
        report.low_quality_images = [
            r.path for r in valid
            if r.blur_score < 60 or r.contrast < 18 or not (25 <= r.brightness <= 235)
        ]

    _evaluate_readiness(report, per_cylinder)
    return report


def _evaluate_readiness(report: AuditReport, per_cylinder: Counter) -> None:
    """Traduce los conteos en un veredicto accionable."""
    if report.total_images == 0:
        report.blockers.append(
            "No hay ninguna fotografía. La captura del dataset es el primer "
            "paso pendiente: ver docs/PROTOCOLO_CAPTURA.md."
        )

    cylinders_with_images = len(per_cylinder)
    if 0 < cylinders_with_images < MIN_CYLINDERS_FOR_SPLIT:
        report.blockers.append(
            f"Sólo {cylinders_with_images} cilindros con fotografías. Con menos "
            f"de {MIN_CYLINDERS_FOR_SPLIT} no es posible separar train/val/test "
            "por identidad de cilindro, y sin esa separación la evaluación no "
            "mide generalización."
        )

    if 0 < report.total_images < MIN_TOTAL_IMAGES:
        report.warnings.append(
            f"{report.total_images} imágenes es poco para entrenar un detector "
            f"con garantías; el objetivo razonable son {MIN_TOTAL_IMAGES} o más."
        )

    scarce = [name for name, count in per_cylinder.items() if count < MIN_IMAGES_PER_CYLINDER]
    if scarce:
        report.warnings.append(
            f"{len(scarce)} cilindros tienen menos de {MIN_IMAGES_PER_CYLINDER} "
            f"fotografías: {', '.join(sorted(scarce)[:10])}"
        )

    if report.empty_cylinders:
        report.warnings.append(
            f"Carpetas de cilindro sin ninguna imagen: "
            f"{', '.join(report.empty_cylinders[:10])}"
        )

    if report.duplicates:
        report.warnings.append(
            f"{len(report.duplicates)} grupos de imágenes duplicadas. Duplicar "
            "fotos infla las métricas sin aportar información."
        )

    if report.corrupt:
        report.warnings.append(f"{len(report.corrupt)} archivos no se pudieron decodificar.")

    if report.low_quality_images:
        share = 100.0 * len(report.low_quality_images) / max(report.total_images, 1)
        report.warnings.append(
            f"{len(report.low_quality_images)} imágenes ({share:.1f}%) con nitidez, "
            "brillo o contraste bajos."
        )

    conditions = len(report.images_per_condition)
    if 0 < conditions < 3:
        report.warnings.append(
            f"Sólo {conditions} condiciones de captura distintas. El conjunto de "
            "test debe incluir ángulo, distancia e iluminación variados para "
            "predecir el comportamiento real en planta."
        )

    report.ready_to_train = not report.blockers


def print_report(report: AuditReport) -> None:
    line = "=" * 68
    print(line)
    print(f"AUDITORÍA DEL DATASET  ·  {report.root}")
    print(line)
    print(f"  Archivos recorridos .... {report.total_files}")
    print(f"  Imágenes válidas ....... {report.total_images}")
    print(f"  Cilindros .............. {report.cylinders}")
    print(f"  Corruptas .............. {len(report.corrupt)}")
    print(f"  Grupos duplicados ...... {len(report.duplicates)}")

    if report.images_per_cylinder:
        print("\n  Imágenes por cilindro:")
        for name, count in list(report.images_per_cylinder.items())[:20]:
            print(f"    {name:<24} {count}")
        if len(report.images_per_cylinder) > 20:
            print(f"    ... y {len(report.images_per_cylinder) - 20} más")

    if report.images_per_condition:
        print("\n  Imágenes por condición de captura:")
        for name, count in report.images_per_condition.items():
            print(f"    {name:<24} {count}")

    if report.resolutions:
        print("\n  Resoluciones más frecuentes:")
        for name, count in report.resolutions.items():
            print(f"    {name:<24} {count}")

    if report.quality:
        print("\n  Calidad:")
        for name, value in report.quality.items():
            print(f"    {name:<24} {value}")

    if report.blockers:
        print("\n  BLOQUEANTES:")
        for item in report.blockers:
            print(f"    - {item}")

    if report.warnings:
        print("\n  Avisos:")
        for item in report.warnings:
            print(f"    - {item}")

    print()
    print(f"  Listo para entrenar: {'SÍ' if report.ready_to_train else 'NO'}")
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("datasets/raw"),
                        help="Directorio raíz con una carpeta por cilindro.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Ruta donde volcar el informe en JSON.")
    args = parser.parse_args()

    report = build_report(args.root)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Informe JSON escrito en {args.json}")

    return 0 if report.ready_to_train else 1


if __name__ == "__main__":
    raise SystemExit(main())
