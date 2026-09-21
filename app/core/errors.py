"""Errores de dominio del microservicio.

Cada error lleva un `code` estable que el backend principal puede usar para
tomar decisiones sin parsear mensajes en castellano.
"""
from __future__ import annotations

from typing import Any


class AIServiceError(Exception):
    """Base de todos los errores controlados del servicio."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidImageError(AIServiceError):
    code = "invalid_image"
    http_status = 422


class ImageTooLargeError(AIServiceError):
    code = "image_too_large"
    http_status = 413


class UnsupportedMediaTypeError(AIServiceError):
    code = "unsupported_media_type"
    http_status = 415


class ModelNotAvailableError(AIServiceError):
    code = "model_not_available"
    http_status = 503


class BackendUnavailableError(AIServiceError):
    """El backend principal no respondió.

    No es fatal: la IA devuelve lo que detectó y marca el enriquecimiento como
    no disponible.
    """

    code = "backend_unavailable"
    http_status = 502
