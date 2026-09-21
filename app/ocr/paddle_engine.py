"""Adaptador de PaddleOCR.

PaddleOCR cambió su API entre la serie 2.x (`ocr(img, cls=True)`) y la 3.x
(`predict(img)`, con resultados en forma de diccionario). El adaptador detecta
cuál está instalada en tiempo de ejecución, de modo que actualizar la
dependencia no rompe el servicio.

El modelo se carga una sola vez y de forma perezosa: la primera inferencia
paga la inicialización, no el arranque del contenedor.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.schemas.common import BoundingBox, OcrLine

logger = get_logger(__name__)


def _polygon_to_box(polygon) -> BoundingBox | None:
    try:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if points.size == 0:
        return None
    return BoundingBox(
        x1=float(points[:, 0].min()), y1=float(points[:, 1].min()),
        x2=float(points[:, 0].max()), y2=float(points[:, 1].max()),
    )


class PaddleOcrEngine:
    name = "paddle"

    def __init__(self, *, lang: str = "en", use_gpu: bool = False,
                 min_confidence: float = 0.3) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._min_confidence = min_confidence
        self._engine: Any = None
        self._api: str | None = None
        self._lock = threading.Lock()
        self._failed = False

    # -- carga perezosa ------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._engine is not None or self._failed:
            return
        with self._lock:
            if self._engine is not None or self._failed:
                return
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                logger.warning("paddleocr no está instalado; el OCR quedará inactivo")
                self._failed = True
                return

            # La firma del constructor también cambió entre versiones. Se
            # intenta la más reciente y se retrocede si no la acepta.
            for kwargs, api in (
                ({"lang": self._lang, "use_textline_orientation": True}, "v3"),
                ({"lang": self._lang, "use_angle_cls": True, "show_log": False}, "v2"),
                ({"lang": self._lang}, "minimal"),
            ):
                try:
                    self._engine = PaddleOCR(**kwargs)
                    self._api = api
                    logger.info("paddleocr inicializado", extra={"api": api, "lang": self._lang})
                    return
                except (TypeError, ValueError):
                    continue
                except Exception:
                    logger.exception("fallo al inicializar paddleocr")
                    self._failed = True
                    return
            self._failed = True
            logger.error("no se encontró una firma compatible de PaddleOCR")

    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._engine is not None

    def describe(self) -> str:
        if self._failed:
            return "paddle (no disponible)"
        return f"paddle:{self._api or 'lazy'}:{self._lang}"

    # -- inferencia ----------------------------------------------------------
    def read(self, image: np.ndarray) -> list[OcrLine]:
        self._ensure_loaded()
        if self._engine is None or image.size == 0:
            return []

        # PaddleOCR espera 3 canales; las variantes de realce son en gris.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        try:
            raw = self._invoke(image)
        except Exception:
            logger.exception("error durante el reconocimiento OCR")
            return []

        return [line for line in self._parse(raw) if line.confidence >= self._min_confidence]

    def _invoke(self, image: np.ndarray):
        engine = self._engine
        if hasattr(engine, "predict"):
            try:
                return engine.predict(image)
            except TypeError:
                pass
        try:
            return engine.ocr(image, cls=True)
        except TypeError:
            return engine.ocr(image)

    def _parse(self, raw) -> list[OcrLine]:
        """Normaliza las distintas formas de salida a OcrLine."""
        if not raw:
            return []

        lines: list[OcrLine] = []

        # Formato 3.x: lista de dicts con rec_texts / rec_scores / rec_polys.
        first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        if isinstance(first, dict):
            for page in raw:
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                for index, text in enumerate(texts):
                    score = float(scores[index]) if index < len(scores) else 0.0
                    box = _polygon_to_box(polys[index]) if index < len(polys) else None
                    if text and text.strip():
                        lines.append(OcrLine(text=text.strip(), confidence=score, box=box))
            return lines

        # Formato 2.x: [[ [poly, (texto, score)], ... ]]
        pages = raw if isinstance(raw[0], list) else [raw]
        for page in pages:
            if not page:
                continue
            for item in page:
                try:
                    polygon, payload = item[0], item[1]
                    text, score = payload[0], float(payload[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if text and text.strip():
                    lines.append(
                        OcrLine(text=text.strip(), confidence=score,
                                box=_polygon_to_box(polygon))
                    )
        return lines
