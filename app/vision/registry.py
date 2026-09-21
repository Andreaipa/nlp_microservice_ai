"""Selección del backend de detección según configuración.

Política de degradación: si el backend solicitado no tiene modelo disponible,
el servicio no falla al arrancar. Cae al detector heurístico y lo declara en
`/version` y en cada respuesta afectada, para que quede explícito que está
operando sin modelo entrenado.
"""
from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.vision.base import Detector
from app.vision.heuristic_detector import HeuristicDetector

logger = get_logger(__name__)


def build_detector(settings: Settings) -> Detector:
    backend = settings.detector_backend

    if backend == "heuristic":
        logger.info("detector: heuristico por configuracion")
        return HeuristicDetector(max_detections=settings.detector_max_detections)

    try:
        detector: Detector
        if backend == "onnx":
            from app.vision.onnx_detector import OnnxDetector

            detector = OnnxDetector(
                settings.detector_model_path,
                input_size=settings.detector_input_size,
                conf_threshold=settings.detector_conf_threshold,
                iou_threshold=settings.detector_iou_threshold,
                max_detections=settings.detector_max_detections,
            )
        else:
            from app.vision.ultralytics_detector import UltralyticsDetector

            detector = UltralyticsDetector(
                settings.detector_model_path,
                conf_threshold=settings.detector_conf_threshold,
                iou_threshold=settings.detector_iou_threshold,
                input_size=settings.detector_input_size,
                max_detections=settings.detector_max_detections,
            )
    except Exception:
        logger.exception("no se pudo construir el detector %s; se usa el heuristico", backend)
        return HeuristicDetector(max_detections=settings.detector_max_detections)

    if not detector.is_ready():
        logger.warning(
            "backend %s sin modelo disponible; se degrada a heuristico", backend,
            extra={"path": str(settings.detector_model_path)},
        )
        return HeuristicDetector(max_detections=settings.detector_max_detections)

    return detector
