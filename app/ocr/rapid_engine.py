"""Adaptador de RapidOCR.

RapidOCR ejecuta los mismos modelos PP-OCR que PaddleOCR, pero sobre ONNX
Runtime en lugar del motor de PaddlePaddle. Medido sobre un recorte real del
troquelado de este proyecto:

    RapidOCR    127 ms   ->  "19s206055"  (correcto)
    PaddleOCR  1224 ms   ->  "198206055"  (confunde S con 8)

Diez veces más rápido y, en esa muestra, más acertado. Además evita la
dependencia de `paddlepaddle`, que son unos 600 MB en la imagen, y reutiliza
onnxruntime, que el servicio ya necesita para el detector.

Esto importa especialmente porque el despliegue de prueba es un equipo Windows
de gama media sin GPU: ahí la diferencia entre 127 ms y 1,2 s por pasada
decide si el servicio es usable o no.
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


class RapidOcrEngine:
    name = "rapid"

    def __init__(self, *, min_confidence: float = 0.3, threads: int = 0) -> None:
        self._min_confidence = min_confidence
        self._threads = threads
        self._engine: Any = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure_loaded(self) -> None:
        if self._engine is not None or self._failed:
            return
        with self._lock:
            if self._engine is not None or self._failed:
                return
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                logger.warning("rapidocr-onnxruntime no está instalado")
                self._failed = True
                return
            try:
                # Limitar los hilos evita que el OCR acapare la máquina cuando
                # el servicio atiende varias peticiones a la vez.
                kwargs = {}
                if self._threads > 0:
                    kwargs = {
                        "intra_op_num_threads": self._threads,
                        "inter_op_num_threads": self._threads,
                    }
                self._engine = RapidOCR(**kwargs) if kwargs else RapidOCR()
                logger.info("rapidocr inicializado", extra={"hilos": self._threads or "auto"})
            except Exception:
                logger.exception("fallo al inicializar rapidocr")
                self._failed = True

    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._engine is not None

    def describe(self) -> str:
        if self._failed:
            return "rapid (no disponible)"
        return "rapid:onnxruntime"

    def read(self, image: np.ndarray) -> list[OcrLine]:
        self._ensure_loaded()
        if self._engine is None or image.size == 0:
            return []

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        try:
            result, _ = self._engine(image)
        except Exception:
            logger.exception("error durante el reconocimiento OCR")
            return []

        lines: list[OcrLine] = []
        for item in result or []:
            try:
                polygon, text, score = item[0], item[1], float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if not text or not text.strip() or score < self._min_confidence:
                continue
            lines.append(OcrLine(text=text.strip(), confidence=score,
                                 box=_polygon_to_box(polygon)))
        return lines
