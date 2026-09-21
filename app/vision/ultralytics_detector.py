"""Detector sobre pesos .pt vía ultralytics.

Pensado para desarrollo y depuración, no para el contenedor de producción:
arrastra PyTorch. Permite probar un modelo recién entrenado sin exportarlo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import ModelNotAvailableError
from app.core.logging import get_logger
from app.schemas.common import BoundingBox, Detection

logger = get_logger(__name__)


class UltralyticsDetector:
    name = "ultralytics"

    def __init__(
        self,
        model_path: Path,
        *,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 960,
        max_detections: int = 50,
    ) -> None:
        self._model_path = Path(model_path)
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._input_size = input_size
        self._max_detections = max_detections
        self._model: Any = None
        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning("pesos no encontrados", extra={"path": str(self._model_path)})
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelNotAvailableError(
                "ultralytics no está instalado. Use el backend 'onnx' o instale "
                "requirements/train.txt."
            ) from exc
        self._model = YOLO(str(self._model_path))
        logger.info("detector ultralytics cargado", extra={"path": str(self._model_path)})

    def is_ready(self) -> bool:
        return self._model is not None

    def describe(self) -> str:
        if not self.is_ready():
            return f"ultralytics (no cargado: {self._model_path.name})"
        return f"ultralytics:{self._model_path.name}"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.is_ready():
            raise ModelNotAvailableError("El detector ultralytics no tiene pesos cargados.")

        results = self._model.predict(
            source=image, conf=self._conf, iou=self._iou,
            imgsz=self._input_size, verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes[: self._max_detections]:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_index = int(box.cls.item())
                detections.append(
                    Detection(
                        label=names.get(class_index, f"class_{class_index}"),
                        confidence=float(round(box.conf.item(), 4)),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections
