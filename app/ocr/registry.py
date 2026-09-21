"""Selección del motor OCR según configuración."""
from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.ocr.base import OcrEngine
from app.ocr.null_engine import NullOcrEngine

logger = get_logger(__name__)


def build_ocr_engine(settings: Settings) -> OcrEngine:
    """Construye el motor OCR configurado.

    El valor por omisión es 'rapid': sobre un recorte real del troquelado
    resultó diez veces más rápido que PaddleOCR (127 ms frente a 1224 ms) y
    leyó mejor el número de serie, además de ahorrar la dependencia de
    paddlepaddle. 'paddle' se mantiene como alternativa porque reconoce más
    texto secundario en algunas fotografías.
    """
    if settings.ocr_backend == "null":
        return NullOcrEngine()

    if settings.ocr_backend == "rapid":
        from app.ocr.rapid_engine import RapidOcrEngine

        engine: OcrEngine = RapidOcrEngine(
            min_confidence=settings.ocr_min_confidence,
            threads=settings.ocr_threads,
        )
    else:
        from app.ocr.paddle_engine import PaddleOcrEngine

        engine = PaddleOcrEngine(
            lang=settings.ocr_lang,
            use_gpu=settings.ocr_use_gpu,
            min_confidence=settings.ocr_min_confidence,
        )

    logger.info("motor ocr configurado", extra={"backend": settings.ocr_backend})
    return engine
