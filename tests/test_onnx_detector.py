"""Pruebas del detector ONNX contra el modelo realmente entrenado."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.vision.onnx_detector import OnnxDetector

MODEL = Path(__file__).resolve().parents[1] / "models" / "detector.onnx"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="no hay modelo exportado; ejecute training/export_onnx.py",
)


def test_input_size_comes_from_the_model_not_the_config():
    """Regresión: un desajuste de tamaño rompía cada inferencia en silencio.

    El modelo se exportó a 768 px y el servicio venía configurado a 960. ONNX
    Runtime rechazaba la entrada, el pipeline degradaba al detector heurístico
    y el servicio seguía respondiendo, de modo que el fallo sólo se veía
    leyendo los logs o comparando métricas.
    """
    detector = OnnxDetector(MODEL, input_size=960)
    assert detector.is_ready()
    # Debe haber adoptado el tamaño que declara el modelo.
    assert detector._input_size != 960


def test_detects_regions_on_a_blank_image_without_crashing():
    detector = OnnxDetector(MODEL, conf_threshold=0.25)
    blank = np.full((2048, 1153, 3), 120, dtype=np.uint8)
    assert isinstance(detector.detect(blank), list)


def test_boxes_stay_inside_the_image():
    detector = OnnxDetector(MODEL, conf_threshold=0.05)
    image = np.random.default_rng(0).integers(
        0, 255, (1400, 800, 3), dtype=np.uint8
    )
    for detection in detector.detect(image):
        assert 0 <= detection.box.x1 <= detection.box.x2 <= image.shape[1]
        assert 0 <= detection.box.y1 <= detection.box.y2 <= image.shape[0]
        assert 0.0 <= detection.confidence <= 1.0


def test_class_names_are_loaded_from_the_sidecar_file():
    detector = OnnxDetector(MODEL)
    assert "marking_area" in detector._classes


def test_missing_model_degrades_instead_of_raising(tmp_path: Path):
    detector = OnnxDetector(tmp_path / "no-existe.onnx")
    assert detector.is_ready() is False
    assert "no cargado" in detector.describe()
