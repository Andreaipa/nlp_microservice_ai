"""Detector YOLO sobre ONNX Runtime.

Es el backend previsto para producción: el contenedor de inferencia no
necesita PyTorch ni ultralytics, sólo onnxruntime, lo que reduce la imagen en
más de 1 GB y elimina la dependencia de la pila de entrenamiento.

Asume un modelo exportado con `training/export_onnx.py`, es decir salida
YOLO en formato (1, 4 + num_classes, num_anchors).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.errors import ModelNotAvailableError
from app.core.logging import get_logger
from app.schemas.common import BoundingBox, Detection
from app.vision.base import DetectionClass

logger = get_logger(__name__)


class OnnxDetector:
    name = "onnx"

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: int = 960,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 50,
    ) -> None:
        self._model_path = Path(model_path)
        self._input_size = input_size
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._max_detections = max_detections
        # La sesión se crea al cargar el modelo; onnxruntime se importa
        # dentro de _load para no exigirlo si el backend no se usa.
        self._session: Any = None
        self._input_name: str | None = None
        self._classes: list[str] = list(DetectionClass.ALL)
        self._load()

    # -- carga ---------------------------------------------------------------
    def _load(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "modelo de deteccion no encontrado",
                extra={"path": str(self._model_path)},
            )
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailableError(
                "onnxruntime no está instalado en este entorno."
            ) from exc

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        # Si hay GPU disponible se usa; el servicio no la exige.
        for gpu_provider in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
            if gpu_provider in available:
                providers.insert(0, gpu_provider)
                break

        self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name

        # El tamaño de entrada lo manda el modelo, no la configuración. Un
        # modelo exportado a 768 px y un servicio configurado a 960 producen
        # un error de dimensiones en cada inferencia; como el pipeline degrada
        # al detector heurístico cuando la detección falla, el servicio sigue
        # respondiendo y el desajuste pasa desapercibido salvo que se lean los
        # logs. Tomarlo del propio modelo elimina esa clase de error.
        shape = list(model_input.shape or [])
        if len(shape) == 4 and all(isinstance(v, int) and v > 0 for v in shape[2:]):
            declared = int(shape[2])
            if declared != self._input_size:
                logger.info(
                    "el modelo declara otro tamaño de entrada; se usa el suyo",
                    extra={"configurado": self._input_size, "modelo": declared},
                )
                self._input_size = declared

        # Los nombres de clase viajan junto al modelo para que el orden de
        # índices no dependa de la configuración del servicio.
        meta = self._model_path.with_suffix(".classes.json")
        if meta.exists():
            try:
                self._classes = json.loads(meta.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("no se pudo leer el mapa de clases; se usa el orden por defecto")

        logger.info(
            "detector onnx cargado",
            extra={"path": str(self._model_path), "providers": providers,
                   "classes": self._classes},
        )

    def is_ready(self) -> bool:
        return self._session is not None

    def describe(self) -> str:
        if not self.is_ready():
            return f"onnx (no cargado: {self._model_path.name})"
        return f"onnx:{self._model_path.name}"

    # -- inferencia ----------------------------------------------------------
    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Redimensiona conservando la relación de aspecto y rellena con gris."""
        height, width = image.shape[:2]
        scale = min(self._input_size / height, self._input_size / width)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x = (self._input_size - new_w) // 2
        pad_y = (self._input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.is_ready():
            raise ModelNotAvailableError(
                "El detector ONNX no tiene un modelo cargado.",
                details={"path": str(self._model_path)},
            )

        canvas, scale, pad_x, pad_y = self._letterbox(image)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self._session.run(None, {self._input_name: blob})
        boxes, scores, class_ids = self._decode(outputs[0])
        if len(boxes) == 0:
            return []

        keep = self._nms(boxes, scores)
        height, width = image.shape[:2]

        detections: list[Detection] = []
        for index in keep[: self._max_detections]:
            cx, cy, bw, bh = boxes[index]
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale

            class_index = int(class_ids[index])
            label = (
                self._classes[class_index]
                if 0 <= class_index < len(self._classes)
                else f"class_{class_index}"
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=float(round(scores[index], 4)),
                    box=BoundingBox(
                        x1=float(max(0.0, x1)), y1=float(max(0.0, y1)),
                        x2=float(min(width, x2)), y2=float(min(height, y2)),
                    ),
                )
            )
        return detections

    def _decode(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convierte la salida cruda en cajas, puntuaciones e índices de clase."""
        prediction = raw[0] if raw.ndim == 3 else raw
        # Salida YOLO: (4 + num_classes, anchors). Se transpone a (anchors, ...).
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        boxes = prediction[:, :4]
        class_scores = prediction[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        mask = scores >= self._conf
        return boxes[mask], scores[mask], class_ids[mask]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Supresión de no máximos sobre cajas en formato centro-ancho-alto."""
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 2]
        xyxy[:, 3] = boxes[:, 3]

        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), scores.tolist(),
            score_threshold=self._conf, nms_threshold=self._iou,
        )
        if len(indices) == 0:
            return []
        return np.asarray(indices).flatten().tolist()
