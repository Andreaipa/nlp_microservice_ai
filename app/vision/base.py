"""Interfaz de detección de regiones.

El servicio depende de esta abstracción, no de un modelo concreto. Cambiar
YOLO por otra arquitectura, o sustituir los pesos, no obliga a tocar el
pipeline ni la API.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas.common import Detection


class DetectionClass:
    """Clases del detector.

    Se mantienen al mínimo a propósito. Cada clase adicional multiplica el
    trabajo de anotación y el riesgo de confusión entre categorías visualmente
    parecidas, mientras que el OCR más el parser ya separan los campos dentro
    de una misma región. La regla aplicada: una clase sólo existe si implica un
    recorte distinto o un tratamiento distinto aguas abajo.
    """

    CYLINDER = "cylinder"          # cuerpo del envase: encuadre y descarte de fondo
    MARKING_AREA = "marking_area"  # collarín troquelado: contiene todos los campos
    SERIAL_TEXT = "serial_text"    # línea concreta del número de serie

    ALL: tuple[str, ...] = (CYLINDER, MARKING_AREA, SERIAL_TEXT)

    # Regiones de las que tiene sentido extraer texto.
    TEXT_BEARING: tuple[str, ...] = (SERIAL_TEXT, MARKING_AREA)


@runtime_checkable
class Detector(Protocol):
    """Contrato de cualquier backend de detección."""

    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Devuelve regiones en coordenadas de píxel de `image`."""
        ...

    def is_ready(self) -> bool:
        """True si el backend puede atender inferencias."""
        ...

    def describe(self) -> str:
        """Identificador legible del modelo en uso, para la respuesta y los logs."""
        ...
