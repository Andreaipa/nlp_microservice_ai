"""Detector sin modelo entrenado, basado en morfología de OpenCV.

Razón de existir: el dataset de fotografías todavía no está capturado, de modo
que aún no hay pesos de YOLO. Sin este componente el microservicio no podría
ejecutarse de extremo a extremo y ni el backend ni la aplicación móvil podrían
integrarse hasta que el modelo estuviera listo.

Lo que hace es localizar aglomeraciones de bordes con forma de línea de texto,
que en una foto encuadrada al collarín corresponden al troquelado. No
distingue clases: etiqueta todo como MARKING_AREA y deja que el OCR y el
parser decidan.

Limitaciones asumidas: es sensible al fondo y no sabe qué es un cilindro. Su
precisión será claramente inferior a la de un YOLO entrenado. Sirve para
desbloquear la integración y como red de seguridad si el modelo no carga, no
como solución definitiva.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger
from app.schemas.common import BoundingBox, Detection
from app.vision.base import DetectionClass

logger = get_logger(__name__)

MIN_REGION_AREA_RATIO = 0.0008
MAX_REGION_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 1.4      # una línea de texto es más ancha que alta
MAX_ASPECT_RATIO = 40.0


class HeuristicDetector:
    name = "heuristic"

    def __init__(self, max_detections: int = 10) -> None:
        self._max_detections = max_detections

    def is_ready(self) -> bool:
        return True

    def describe(self) -> str:
        return "heuristic-opencv-morphology (sin modelo entrenado)"

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        total_area = float(height * width)
        if total_area <= 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        # El gradiente morfológico responde al relieve del troquelado.
        grad = cv2.morphologyEx(
            gray, cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Unir caracteres contiguos en una línea: kernel ancho y bajo.
        connect = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 60), 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0:
                continue
            area_ratio = (w * h) / total_area
            aspect = w / float(h)
            if not (MIN_REGION_AREA_RATIO <= area_ratio <= MAX_REGION_AREA_RATIO):
                continue
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                continue

            # Densidad de borde dentro de la caja: descarta zonas planas.
            region = closed[y:y + h, x:x + w]
            fill = float(region.mean()) / 255.0
            if fill < 0.10:
                continue

            # Puntuación heurística. No es una probabilidad calibrada y se
            # limita a 0.5 para que nunca supere por sí sola el umbral de
            # aceptación automática de un serial.
            score = min(0.50, 0.20 + fill * 0.4 + min(area_ratio * 8, 0.1))
            candidates.append(
                Detection(
                    label=DetectionClass.MARKING_AREA,
                    confidence=round(score, 4),
                    box=BoundingBox(x1=float(x), y1=float(y),
                                    x2=float(x + w), y2=float(y + h)),
                )
            )

        merged = self._merge_into_lines(candidates)
        merged.sort(key=lambda d: d.box.area, reverse=True)
        selected = merged[: self._max_detections]
        logger.debug(
            "deteccion heuristica",
            extra={"fragmentos": len(candidates), "regiones": len(selected)},
        )
        return selected

    @staticmethod
    def _merge_into_lines(detections: list[Detection]) -> list[Detection]:
        """Une fragmentos que pertenecen a la misma línea de texto.

        La morfología tiende a partir una línea troquelada en varios trozos
        cuando hay separación entre grupos de caracteres. Entregar esos trozos
        por separado al OCR es lo peor que puede pasar: cada recorte corta el
        número de serie por la mitad y ninguna lectura resulta utilizable.

        Se fusionan las cajas que comparten banda vertical y están próximas en
        horizontal, que es lo que caracteriza a una misma línea.
        """
        if not detections:
            return []

        remaining = sorted(detections, key=lambda d: (d.box.y1, d.box.x1))
        lines: list[list[Detection]] = []

        for detection in remaining:
            placed = False
            for line in lines:
                reference = line[0].box
                height = max(reference.height, detection.box.height, 1.0)
                # El criterio se mide contra la caja MENOR: un fragmento
                # pequeño (una letra suelta que la morfología no llegó a unir)
                # queda dentro de la banda de la línea y debe absorberse.
                span = max(min(reference.height, detection.box.height), 1.0)

                overlap = min(reference.y2, detection.box.y2) - max(reference.y1, detection.box.y1)
                if overlap < 0.45 * span:
                    continue

                # Proximidad horizontal: hasta tres alturas de hueco, que es lo
                # que puede separar dos grupos de caracteres de una misma línea.
                line_x1 = min(d.box.x1 for d in line)
                line_x2 = max(d.box.x2 for d in line)
                gap = max(detection.box.x1 - line_x2, line_x1 - detection.box.x2)
                if gap > 3.0 * height:
                    continue

                line.append(detection)
                placed = True
                break

            if not placed:
                lines.append([detection])

        fused: list[Detection] = []
        for line in lines:
            fused.append(
                Detection(
                    label=line[0].label,
                    confidence=round(max(d.confidence for d in line), 4),
                    box=BoundingBox(
                        x1=min(d.box.x1 for d in line),
                        y1=min(d.box.y1 for d in line),
                        x2=max(d.box.x2 for d in line),
                        y2=max(d.box.y2 for d in line),
                    ),
                )
            )
        return fused
