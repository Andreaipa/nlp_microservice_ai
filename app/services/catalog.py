"""Catálogo local del inventario de cilindros.

Por qué existe: el inventario de la empresa es CERRADO y pequeño (100
cilindros). Eso cambia la naturaleza del problema. No hay que reconocer una
cadena arbitraria, sino decidir cuál de 100 seriales conocidos aparece en la
foto.

La consecuencia práctica es importante: un OCR que se equivoque en uno o dos
caracteres sigue permitiendo identificar el cilindro correcto, porque la
distancia ponderada al serial real será mucho menor que a cualquier otro del
inventario. Sin esta etapa, un solo carácter mal leído produciría un
identificador inexistente.

Salvaguarda: si dos candidatos quedan a distancia similar, el resultado se
marca como ambiguo y se deriva a confirmación manual. Elegir el más cercano
por un margen mínimo es exactamente el tipo de suposición que provoca que un
movimiento se registre en el cilindro equivocado.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.parsing.serial import (
    DEFAULT_PATTERN,
    SerialPattern,
    normalize_serial_charset,
    weighted_edit_distance,
)
from app.schemas.analyze import CatalogCandidate, MatchStatus, SerialMatch

logger = get_logger(__name__)


# Longitud por debajo de la cual un serial no sirve como identificador fiable.
#
# El umbral está en 7 por un fallo observado al medir sobre el conjunto de
# test, no por prudencia abstracta. En una fotografía del cilindro con serial
# "21S062189" el OCR leyó el fragmento "20640", que resulta ser el serial
# COMPLETO del Cilindro_015. El sistema lo habría dado por identificado y el
# movimiento habría quedado registrado en el cilindro equivocado, sin que
# nada lo delatara: el peor fallo posible en un sistema de trazabilidad.
#
# El inventario tiene 12 seriales de seis caracteres o menos. Todos ellos
# quedan marcados para confirmación manual: es el precio de no equivocarse.
# Si la empresa vuelve a marcar esos cilindros con un código más largo, el
# umbral deja de afectarles.
MIN_RELIABLE_SERIAL_LENGTH = 7

# Longitud mínima para aceptar que un texto leído es un FRAGMENTO de un serial
# del inventario. El OCR pierde con frecuencia los primeros caracteres del
# troquelado, que son los más desgastados, y devuelve algo como "206055" donde
# el serial real es "19S206055". La distancia de edición rechaza ese caso (son
# tres inserciones), pero seis caracteres consecutivos y en orden no coinciden
# por azar.
#
# La salvaguarda es la unicidad: si el fragmento aparece en más de un serial
# del inventario, no identifica nada y el resultado se declara ambiguo.
MIN_FRAGMENT_LENGTH = 6


@dataclass(frozen=True)
class CatalogEntry:
    """Registro maestro de un cilindro conocido.

    `cilindros` es una lista porque el inventario real contiene un serial
    repetido en dos envases distintos. Un identificador duplicado deja de
    identificar, y el sistema tiene que decirlo en lugar de elegir uno.
    """

    numero_serie: str
    attributes: dict[str, Any]
    cilindros: tuple[str, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return len(self.cilindros) > 1

    @property
    def is_too_short(self) -> bool:
        return len(self.numero_serie) < MIN_RELIABLE_SERIAL_LENGTH


class CylinderCatalog:
    """Índice en memoria del inventario conocido."""

    def __init__(
        self,
        entries: list[CatalogEntry] | None = None,
        *,
        max_distance: float = 2.0,
        ambiguity_margin: float = 1.0,
    ) -> None:
        self._entries: dict[str, CatalogEntry] = {}
        self._max_distance = max_distance
        self._ambiguity_margin = ambiguity_margin
        self._pattern: SerialPattern = DEFAULT_PATTERN

        for entry in entries or []:
            self._entries[entry.numero_serie] = entry
        self._refresh_pattern()

    # -- construcción --------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        max_distance: float = 2.0,
        ambiguity_margin: float = 1.0,
    ) -> CylinderCatalog:
        """Carga el catálogo desde JSON. Si no existe, devuelve uno vacío.

        Un catálogo ausente no impide arrancar: el servicio seguirá devolviendo
        lo que lea, sin resolución contra inventario, y lo indicará con el
        estado `no_catalog`.
        """
        if not path.exists():
            logger.warning("catálogo no encontrado", extra={"path": str(path)})
            return cls(max_distance=max_distance, ambiguity_margin=ambiguity_margin)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.exception("catálogo ilegible", extra={"path": str(path)})
            return cls(max_distance=max_distance, ambiguity_margin=ambiguity_margin)

        raw_entries = payload.get("cylinders", payload if isinstance(payload, list) else [])

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in raw_entries:
            serial = normalize_serial_charset(str(record.get("numero_serie", "")))
            if not serial:
                continue
            grouped.setdefault(serial, []).append(record)

        entries: list[CatalogEntry] = []
        for serial, records in grouped.items():
            cylinders = tuple(
                str(r["cilindro"]) for r in records if r.get("cilindro")
            )
            # Con un serial repetido no se puede saber a cuál de los dos
            # envases pertenece una lectura, así que no se mezclan sus datos:
            # se conservan los del primero y la entrada queda marcada como
            # ambigua.
            attributes = {
                k: v for k, v in records[0].items()
                if k not in ("numero_serie", "cilindro")
            }
            entries.append(CatalogEntry(
                numero_serie=serial, attributes=attributes, cilindros=cylinders,
            ))

        duplicated = [e.numero_serie for e in entries if len(e.cilindros) > 1]
        if duplicated:
            logger.warning(
                "el catálogo contiene seriales repetidos",
                extra={"seriales": duplicated},
            )

        catalog = cls(entries, max_distance=max_distance, ambiguity_margin=ambiguity_margin)
        logger.info(
            "catálogo cargado",
            extra={"path": str(path), "cilindros": len(catalog),
                   "patron": catalog.pattern.mask or "sin máscara"},
        )
        return catalog

    def _refresh_pattern(self) -> None:
        if self._entries:
            self._pattern = SerialPattern.infer_from_catalog(list(self._entries))

    # -- consulta ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return not self._entries

    @property
    def pattern(self) -> SerialPattern:
        """Patrón de serial derivado de los datos, no escrito a mano."""
        return self._pattern

    def get(self, serial: str) -> CatalogEntry | None:
        return self._entries.get(normalize_serial_charset(serial))

    def integrity_warnings(self, serial: str) -> list[str]:
        """Problemas del maestro que afectan a este serial.

        No son fallos de lectura: son defectos del inventario que impiden
        identificar con certeza aunque el OCR haya acertado.
        """
        entry = self.get(serial)
        if entry is None:
            return []

        warnings: list[str] = []
        if entry.is_ambiguous:
            warnings.append(
                f"El serial {entry.numero_serie} está registrado en "
                f"{len(entry.cilindros)} cilindros ({', '.join(entry.cilindros)}): "
                "no permite distinguirlos."
            )
        if entry.is_too_short:
            warnings.append(
                f"El serial {entry.numero_serie} tiene sólo "
                f"{len(entry.numero_serie)} caracteres; es demasiado corto para "
                "identificar el cilindro de forma fiable."
            )
        return warnings

    def resolve(self, candidate: str) -> SerialMatch:
        """Resuelve un serial leído contra el inventario conocido."""
        cleaned = normalize_serial_charset(candidate)
        if not cleaned:
            return SerialMatch(status=MatchStatus.NOT_ATTEMPTED)

        if self.is_empty:
            # Sin inventario cargado no se puede confirmar ni descartar.
            return SerialMatch(status=MatchStatus.NO_CATALOG, resolved_serial=cleaned)

        if cleaned in self._entries:
            return SerialMatch(
                status=MatchStatus.MATCHED,
                resolved_serial=cleaned,
                candidates=[CatalogCandidate(numero_serie=cleaned, distance=0, score=1.0)],
            )

        # Emparejamiento por fragmento: el texto leído está contenido en un
        # serial del inventario, o lo contiene. Sólo vale si identifica a uno
        # solo.
        if len(cleaned) >= MIN_FRAGMENT_LENGTH:
            containing = [
                serial for serial in self._entries
                if cleaned in serial or serial in cleaned
            ]
            if len(containing) == 1:
                serial = containing[0]
                return SerialMatch(
                    status=MatchStatus.MATCHED,
                    resolved_serial=serial,
                    candidates=[CatalogCandidate(
                        numero_serie=serial,
                        distance=abs(len(serial) - len(cleaned)),
                        # Proporción de solapamiento: el fragmento puede ser
                        # más corto que el serial (el OCR perdió caracteres)
                        # o más largo (arrastró ruido alrededor). En ambos
                        # casos la puntuación debe quedar entre 0 y 1.
                        score=round(
                            min(len(cleaned), len(serial))
                            / max(len(cleaned), len(serial), 1),
                            4,
                        ),
                    )],
                )
            if len(containing) > 1:
                return SerialMatch(
                    status=MatchStatus.AMBIGUOUS,
                    resolved_serial=None,
                    candidates=[
                        CatalogCandidate(numero_serie=s_, distance=0, score=0.5)
                        for s_ in containing[:5]
                    ],
                )

        # Se puntúa el inventario entero. El umbral decide si hay
        # identificación, pero los candidatos se devuelven igualmente: cuando
        # la lectura no es concluyente, ofrecer al operario los tres o cuatro
        # cilindros más parecidos es mucho más útil que un "no encontrado",
        # porque elegir de una lista corta es inmediato y teclear nueve
        # caracteres no lo es.
        scored = sorted(
            ((weighted_edit_distance(cleaned, serial), serial)
             for serial in self._entries),
            key=lambda item: (item[0], item[1]),
        )
        if not scored:
            return SerialMatch(status=MatchStatus.NOT_FOUND, resolved_serial=None)

        within_threshold = [item for item in scored if item[0] <= self._max_distance]
        reference_length = max(len(cleaned), 1)
        candidates = [
            CatalogCandidate(
                numero_serie=serial,
                distance=int(round(distance)),
                score=round(max(0.0, 1.0 - distance / reference_length), 4),
            )
            for distance, serial in scored[:5]
        ]

        if not within_threshold:
            # Nada lo bastante cerca para afirmar nada, pero los candidatos
            # siguen siendo la mejor pista disponible para quien confirme.
            return SerialMatch(
                status=MatchStatus.NOT_FOUND,
                resolved_serial=None,
                candidates=candidates,
            )

        best_distance = within_threshold[0][0]
        runner_up = within_threshold[1][0] if len(within_threshold) > 1 else None

        # Si el segundo candidato está prácticamente igual de cerca, no hay
        # base para elegir: es ambiguo y lo decide una persona.
        if runner_up is not None and (runner_up - best_distance) < self._ambiguity_margin:
            return SerialMatch(
                status=MatchStatus.AMBIGUOUS, resolved_serial=None, candidates=candidates
            )

        return SerialMatch(
            status=MatchStatus.MATCHED,
            resolved_serial=within_threshold[0][1],
            candidates=candidates,
        )
