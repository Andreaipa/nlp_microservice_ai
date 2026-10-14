"""Geometría de las líneas reconocidas por el OCR.

El motor OCR devuelve cajas sueltas. Dos problemas observados con fotos
reales obligan a mirar su geometría y no sólo su texto:

* Lecturas imposibles. Sobre pintura descascarada el OCR llegó a "leer" una
  cadena parecida a un serial, con 0.96 de confianza, en una caja vertical
  donde no había texto. El troquelado se lee en horizontal.
* Seriales partidos. La serie K57... lleva un espacio en medio del número
  ("K5738 060"), y el OCR lo devuelve como dos cajas. Por separado ninguna
  identifica el cilindro; juntas, sí.
"""
from __future__ import annotations

from app.schemas.common import BoundingBox, OcrLine

# Relación ancho/alto mínima de una línea de texto troquelado. Se asume que la
# foto llega en la orientación en que se tomó (envase vertical, troquelado
# horizontal). Si se admitieran fotos giradas 90 grados habría que corregir la
# orientación antes del OCR, no relajar este filtro.
MIN_TEXT_ASPECT_RATIO = 1.2

# Para unir dos cajas en una línea: sus centros verticales no pueden distar más
# de esta fracción de la altura del texto, y el hueco horizontal no puede
# superar este múltiplo de esa altura (el espacio de "K5738 060" es de menos
# de un carácter; el que separa el serial de "UT" o de otra marca, mayor).
_SAME_LINE_TOLERANCE = 0.6
_MAX_GAP_IN_HEIGHTS = 2.0
_MAX_PIECES = 4


def is_plausible_text_box(line: OcrLine) -> bool:
    """False si la caja es más alta que ancha: no es texto troquelado."""
    if line.box is None:
        return True
    height = max(line.box.y2 - line.box.y1, 1.0)
    return (line.box.x2 - line.box.x1) / height >= MIN_TEXT_ASPECT_RATIO


def join_split_lines(lines: list[OcrLine]) -> list[OcrLine]:
    """Une cajas contiguas de una misma línea de texto, de izquierda a derecha.

    Devuelve sólo las uniones (dos o más cajas); las líneas originales siguen
    valiendo por sí mismas. El texto unido conserva el espacio, de modo que el
    serial aparece tanto junto ("K5738060") como por partes.
    """
    boxed = sorted((line for line in lines if line.box is not None),
                   key=lambda line: line.box.x1)  # type: ignore[union-attr]
    joined: list[OcrLine] = []
    used: set[int] = set()

    for start, first in enumerate(boxed):
        if start in used:
            continue
        group = [first]
        current = first.box
        assert current is not None
        while len(group) < _MAX_PIECES:
            height = current.y2 - current.y1
            center = (current.y1 + current.y2) / 2
            best: tuple[float, int] | None = None
            for index, other in enumerate(boxed):
                box = other.box
                if index in used or other in group or box is None:
                    continue
                other_height = box.y2 - box.y1
                scale = max(height, other_height, 1.0)
                gap = box.x1 - current.x2
                if gap < -0.5 * scale or gap > _MAX_GAP_IN_HEIGHTS * scale:
                    continue
                if abs((box.y1 + box.y2) / 2 - center) > _SAME_LINE_TOLERANCE * scale:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, index)
            if best is None:
                break
            used.add(best[1])
            nxt = boxed[best[1]]
            group.append(nxt)
            assert nxt.box is not None
            current = BoundingBox(
                x1=min(current.x1, nxt.box.x1), y1=min(current.y1, nxt.box.y1),
                x2=max(current.x2, nxt.box.x2), y2=max(current.y2, nxt.box.y2),
            )
        if len(group) > 1:
            used.add(start)
            joined.append(OcrLine(
                text=" ".join(line.text for line in group),
                confidence=min(line.confidence for line in group),
                box=current,
                source_region=first.source_region,
            ))
    return joined
