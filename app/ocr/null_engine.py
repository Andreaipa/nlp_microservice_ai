"""Motor OCR inerte.

Se usa en tests y cuando PaddleOCR no está disponible en el entorno. Permite
ejercitar el resto del pipeline sin cargar el modelo de reconocimiento.
"""
from __future__ import annotations

import numpy as np

from app.schemas.common import OcrLine


class NullOcrEngine:
    name = "null"

    def __init__(self, canned: list[OcrLine] | None = None) -> None:
        self._canned = canned or []

    def is_ready(self) -> bool:
        return True

    def describe(self) -> str:
        return "null-ocr (sin reconocimiento)"

    def read(self, image: np.ndarray) -> list[OcrLine]:
        return list(self._canned)
