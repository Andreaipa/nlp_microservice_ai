"""Pruebas del detector heurístico de respaldo."""
from __future__ import annotations

import numpy as np

from app.vision.base import DetectionClass
from app.vision.heuristic_detector import HeuristicDetector


def test_finds_the_two_stamped_lines(stamped_image: np.ndarray):
    detections = HeuristicDetector().detect(stamped_image)
    assert len(detections) == 2
    assert all(d.label == DetectionClass.MARKING_AREA for d in detections)


def test_fragments_are_merged_into_whole_lines(stamped_image: np.ndarray):
    """Una línea partida deja el número de serie ilegible al recortarlo."""
    detections = HeuristicDetector().detect(stamped_image)
    width = stamped_image.shape[1]
    # Cada línea detectada debe abarcar buena parte del ancho del texto, no un
    # fragmento suelto de unos pocos caracteres.
    assert all(d.box.width > 0.4 * width for d in detections)


def test_confidence_never_reaches_auto_accept_level(stamped_image: np.ndarray):
    """Sin modelo entrenado, la puntuación no debe parecer una probabilidad."""
    detections = HeuristicDetector().detect(stamped_image)
    assert all(d.confidence <= 0.5 for d in detections)


def test_blank_image_yields_nothing():
    blank = np.full((400, 400, 3), 128, dtype=np.uint8)
    assert HeuristicDetector().detect(blank) == []


def test_detector_is_always_ready():
    detector = HeuristicDetector()
    assert detector.is_ready() is True
    assert "sin modelo entrenado" in detector.describe()
