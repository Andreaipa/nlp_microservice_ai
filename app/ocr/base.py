"""Interfaz del motor OCR."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.schemas.common import OcrLine


@runtime_checkable
class OcrEngine(Protocol):
    name: str

    def read(self, image: np.ndarray) -> list[OcrLine]:
        """Reconoce texto en una imagen ya preprocesada."""
        ...

    def is_ready(self) -> bool:
        ...

    def describe(self) -> str:
        ...
