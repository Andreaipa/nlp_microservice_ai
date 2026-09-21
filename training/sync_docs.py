#!/usr/bin/env python3
"""Sincroniza los documentos de `docs/` con el sitio Starlight.

Los documentos viven en `docs/` como Markdown corriente, legible en el
repositorio y en GitHub sin necesidad de compilar nada. El sitio los publica
añadiéndoles el frontmatter que Starlight necesita y reescribiendo los
enlaces entre documentos a las rutas del sitio.

Mantener una sola fuente evita que las dos copias se desvíen, que es lo que
acaba pasando cuando se duplican a mano.

Uso:
    python training/sync_docs.py
    python training/sync_docs.py --check     # sólo comprueba, no escribe
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Documento de origen -> destino en el sitio, título y descripción.
MAPPING: list[tuple[str, str, str, str]] = [
    ("docs/PUESTA_EN_MARCHA.md", "guias/puesta-en-marcha.md",
     "Puesta en marcha",
     "Arrancar el servicio, integrarlo y reproducir los resultados."),
    ("docs/ARQUITECTURA.md", "sistema/arquitectura.md",
     "Arquitectura",
     "Pipeline, decisiones de diseño y por qué se tomaron."),
    ("docs/RENDIMIENTO.md", "sistema/rendimiento.md",
     "Rendimiento y tecnología",
     "Dónde se va el tiempo y cómo ajustar el servicio a cada equipo."),
    ("docs/THRESHOLDS.md", "sistema/umbrales.md",
     "Umbrales de confianza",
     "Cómo se calibran y por qué no se eligen a ojo."),
    ("docs/INTEGRACION_BACKEND.md", "sistema/integracion.md",
     "Integración con el backend",
     "Contrato, errores y reparto de responsabilidades."),
    ("docs/DATASET.md", "datos/dataset.md",
     "El dataset",
     "Contenido real, hallazgos e inconsistencias del inventario."),
    ("docs/PROTOCOLO_CAPTURA.md", "datos/protocolo-captura.md",
     "Protocolo de captura",
     "Cómo fotografiar los cilindros para que el sistema funcione."),
    ("docs/ANOTACION.md", "datos/anotacion.md",
     "Estrategia de anotación",
     "Clases, criterios y pre-anotación automática."),
    ("docs/RECONOCEDOR.md", "datos/reconocedor.md",
     "Reconocedor especializado",
     "Un experimento que no funcionó, y por qué."),
    ("docs/DIAGNOSTICO.md", "referencia/diagnostico.md",
     "Diagnóstico inicial",
     "Qué se encontró al empezar y qué implicaba."),
]

SITE_DOCS = Path("docs-site/src/content/docs")


def convert(text: str) -> str:
    """Quita el H1 y reescribe los enlaces entre documentos."""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    body = "\n".join(lines)

    slugs = {
        Path(source).name: "/" + destination.replace(".md", "")
        for source, destination, _, _ in MAPPING
    }
    for name, slug in slugs.items():
        body = body.replace(f"]({name})", f"]({slug}/)")
        body = body.replace(f"](docs/{name})", f"]({slug}/)")
    return body


def render(source: Path, title: str, description: str) -> str:
    frontmatter = (
        "---\n"
        f"title: {title}\n"
        f"description: {description}\n"
        "---\n\n"
    )
    return frontmatter + convert(source.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Devuelve error si el sitio está desactualizado.")
    args = parser.parse_args()

    outdated: list[str] = []
    written = 0

    for source_name, destination_name, title, description in MAPPING:
        source = Path(source_name)
        if not source.exists():
            print(f"  AVISO: falta {source_name}")
            continue

        target = SITE_DOCS / destination_name
        content = render(source, title, description)

        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            if current != content:
                outdated.append(destination_name)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written += 1

    if args.check:
        if outdated:
            print("El sitio está desactualizado respecto a docs/:")
            for name in outdated:
                print(f"  - {name}")
            print("\nEjecute: python training/sync_docs.py")
            return 1
        print("El sitio está sincronizado con docs/.")
        return 0

    print(f"Sincronizados {written} documentos en {SITE_DOCS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
