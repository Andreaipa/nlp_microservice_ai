"""Tipos compartidos del contrato público."""
from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class FieldSource(str, Enum):
    """Procedencia de un dato. Permite auditar de dónde salió cada valor."""

    AI = "ai"               # leído de la imagen por detección + OCR
    CATALOG = "catalog"     # resuelto contra el catálogo local de cilindros
    BACKEND = "backend"     # entregado por el backend principal
    MANUAL = "manual"       # confirmado o corregido por un operador
    UNKNOWN = "unknown"     # no disponible en ninguna fuente


class TracedValue(BaseModel, Generic[T]):
    """Valor con trazabilidad de origen y confianza.

    `confidence` es None cuando la fuente es determinista (backend, catálogo o
    confirmación manual): en esos casos hablar de confianza no tiene sentido.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {"value": "19S206055", "confidence": 0.94, "source": "ai",
                 "raw": "SERIE 19S206055"},
                {"value": "CO2", "confidence": None, "source": "catalog", "raw": None},
                {"value": None, "confidence": None, "source": "unknown", "raw": None},
            ]
        },
    )

    value: T | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: FieldSource = FieldSource.UNKNOWN
    raw: str | None = Field(
        default=None,
        description="Texto OCR crudo del que se derivó el valor, antes de normalizar.",
    )

    @classmethod
    def unknown(cls) -> TracedValue[T]:
        return cls(value=None, confidence=None, source=FieldSource.UNKNOWN)

    @classmethod
    def from_ai(cls, value: T, confidence: float, raw: str | None = None) -> TracedValue[T]:
        return cls(value=value, confidence=confidence, source=FieldSource.AI, raw=raw)

    @classmethod
    def from_backend(cls, value: T) -> TracedValue[T]:
        return cls(value=value, confidence=None, source=FieldSource.BACKEND)

    @classmethod
    def from_catalog(cls, value: T) -> TracedValue[T]:
        return cls(value=value, confidence=None, source=FieldSource.CATALOG)

    @property
    def is_present(self) -> bool:
        return self.value is not None


class BoundingBox(BaseModel):
    """Caja en píxeles sobre la imagen recibida, tras el reescalado interno."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """Región detectada por el modelo de visión."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "label": "marking_area",
            "confidence": 0.83,
            "box": {"x1": 376.0, "y1": 790.0, "x2": 866.0, "y2": 1085.0},
        }
    })

    label: str = Field(description="cylinder | marking_area | serial_text")
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class OcrLine(BaseModel):
    """Una línea reconocida por el motor OCR."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox | None = None
    source_region: str | None = Field(
        default=None, description="Etiqueta de la detección de la que provino el recorte."
    )


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)
