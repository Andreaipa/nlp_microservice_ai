"""Contrato de request/response del endpoint de análisis."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import Detection, OcrLine
from app.schemas.cylinder import CylinderData


class MatchStatus(str, Enum):
    """Resultado de resolver el serial leído contra el inventario conocido."""

    MATCHED = "matched"           # un único candidato claro
    AMBIGUOUS = "ambiguous"       # varios candidatos a distancia similar
    NOT_FOUND = "not_found"       # ningún candidato dentro del umbral
    NO_CATALOG = "no_catalog"     # no hay catálogo cargado: sin resolución
    NOT_ATTEMPTED = "not_attempted"  # no se leyó serial alguno


class ReviewReason(str, Enum):
    """Motivo por el que un análisis requiere confirmación humana."""

    NO_SERIAL_DETECTED = "no_serial_detected"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    SERIAL_FORMAT_INVALID = "serial_format_invalid"
    AMBIGUOUS_CATALOG_MATCH = "ambiguous_catalog_match"
    SERIAL_NOT_IN_CATALOG = "serial_not_in_catalog"
    DUPLICATE_SERIAL_IN_CATALOG = "duplicate_serial_in_catalog"
    SERIAL_TOO_SHORT = "serial_too_short"
    NO_DETECTOR_MODEL = "no_detector_model"
    THRESHOLDS_NOT_CALIBRATED = "thresholds_not_calibrated"
    POOR_IMAGE_QUALITY = "poor_image_quality"


class CatalogCandidate(BaseModel):
    """Candidato del inventario compatible con el texto leído."""

    numero_serie: str
    distance: int = Field(description="Distancia de edición contra el texto OCR.")
    score: float = Field(ge=0.0, le=1.0)


class SerialMatch(BaseModel):
    status: MatchStatus = MatchStatus.NOT_ATTEMPTED
    resolved_serial: str | None = None
    candidates: list[CatalogCandidate] = Field(default_factory=list)


class ImageQuality(BaseModel):
    """Señales objetivas de calidad, útiles para depurar capturas malas."""

    width: int
    height: int
    blur_score: float = Field(description="Varianza del Laplaciano; menor = más borroso.")
    brightness: float = Field(description="Luminancia media 0-255.")
    contrast: float = Field(description="Desviación estándar de luminancia.")
    is_acceptable: bool


class BackendMatch(BaseModel):
    found: bool = False
    queried: bool = False
    error: str | None = None


class AnalyzeResponse(BaseModel):
    """Respuesta del análisis de una fotografía de cilindro.

    El microservicio declara lo que observó; la decisión de negocio es del
    backend principal.
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "summary": "Identificación con confirmación requerida",
                "description": (
                    "Caso habitual: el serial se lee y se resuelve contra el "
                    "inventario, pero la confianza no alcanza el umbral de "
                    "aceptación automática. El backend debe pedir confirmación."
                ),
                "value": {
                    "success": True,
                    "request_id": "8f3a1c9d2b7e4a10",
                    "service_version": "1.0.0",
                    "model_version":
                        "detector:onnx:detector.onnx|ocr:rapid:onnxruntime|parser:1.0.0",
                    "processing_time_ms": 667,
                    "requires_manual_confirmation": True,
                    "review_reasons": ["low_ocr_confidence"],
                    "data": {
                        "numero_serie": {"value": "21S062189", "confidence": 0.61,
                                         "source": "ai", "raw": "21S062189"},
                        "marca": {"value": "JP", "confidence": None, "source": "catalog"},
                        "tipo_gas": {"value": "CO2", "confidence": None, "source": "catalog"},
                        "peso": {"value": 48.0, "confidence": None, "source": "catalog"},
                        "m3": {"value": None, "confidence": None, "source": "unknown"},
                    },
                    "match": {
                        "status": "matched",
                        "resolved_serial": "21S062189",
                        "candidates": [
                            {"numero_serie": "21S062189", "distance": 0, "score": 1.0}
                        ],
                    },
                    "backend_match": {"found": False, "queried": False, "error": None},
                    "detections": [
                        {"label": "marking_area", "confidence": 0.83,
                         "box": {"x1": 376.0, "y1": 790.0, "x2": 866.0, "y2": 1085.0}}
                    ],
                    "warnings": [],
                },
            },
            {
                "summary": "No se pudo leer el número de serie",
                "description": (
                    "La fotografía no permite leer el troquelado: suele ocurrir "
                    "en tomas lejanas o frontales. La aplicación debe pedir una "
                    "foto más cercana al collarín."
                ),
                "value": {
                    "success": True,
                    "request_id": "1b77c0de55aa4f21",
                    "service_version": "1.0.0",
                    "model_version":
                        "detector:onnx:detector.onnx|ocr:rapid:onnxruntime|parser:1.0.0",
                    "processing_time_ms": 700,
                    "requires_manual_confirmation": True,
                    "review_reasons": ["no_serial_detected", "poor_image_quality"],
                    "data": {
                        "numero_serie": {"value": None, "confidence": None,
                                         "source": "unknown"}
                    },
                    "match": {"status": "not_attempted", "resolved_serial": None,
                              "candidates": []},
                    "backend_match": {"found": False, "queried": False},
                    "detections": [],
                    "warnings": ["El OCR no reconoció texto en la imagen."],
                },
            },
            {
                "summary": "Serial duplicado en el inventario",
                "description": (
                    "La lectura es correcta pero el serial está registrado en "
                    "dos cilindros distintos, así que no identifica. Ocurre con "
                    "K5738166 en el inventario actual."
                ),
                "value": {
                    "success": True,
                    "request_id": "c0ffee1234567890",
                    "requires_manual_confirmation": True,
                    "review_reasons": ["duplicate_serial_in_catalog"],
                    "data": {
                        "numero_serie": {"value": "K5738166", "confidence": 0.91,
                                         "source": "ai"}
                    },
                    "match": {"status": "matched", "resolved_serial": "K5738166"},
                    "warnings": [
                        "El serial K5738166 está registrado en 2 cilindros "
                        "(Cilindro_051, Cilindro_053): no permite distinguirlos."
                    ],
                },
            },
        ]
    })

    success: bool = True
    request_id: str
    service_version: str
    model_version: str
    processing_time_ms: int

    requires_manual_confirmation: bool = False
    review_reasons: list[ReviewReason] = Field(default_factory=list)

    data: CylinderData = Field(default_factory=CylinderData)
    match: SerialMatch = Field(default_factory=SerialMatch)
    backend_match: BackendMatch = Field(default_factory=BackendMatch)

    detections: list[Detection] = Field(default_factory=list)
    ocr_lines: list[OcrLine] = Field(default_factory=list)
    image_quality: ImageQuality | None = None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class ComponentStatus(BaseModel):
    name: str
    ready: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    components: list[ComponentStatus]


class VersionResponse(BaseModel):
    service: str
    service_version: str
    model_version: str
    detector_backend: str
    ocr_backend: str
    catalog_size: int
    thresholds_calibrated: bool
