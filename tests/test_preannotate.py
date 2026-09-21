"""Pruebas de la localización del serial en la pre-anotación."""
from __future__ import annotations

import numpy as np
import pytest

import training.preannotate as pa
from app.schemas.common import BoundingBox, OcrLine


class FakeOcr:
    """OCR de prueba: devuelve las mismas líneas para cualquier entrada."""

    def __init__(self, lines: list[OcrLine]) -> None:
        self._lines = lines

    def read(self, image):
        return list(self._lines)


def image(width: int = 1000, height: int = 2000) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def line(text: str, *, y: float, height: float = 40.0,
         width: float = 300.0, confidence: float = 0.9) -> OcrLine:
    return OcrLine(
        text=text, confidence=confidence,
        box=BoundingBox(x1=100.0, y1=y, x2=100.0 + width, y2=y + height),
    )


def test_serial_low_in_the_band_is_still_accepted():
    """Regresión: en las tomas cercanas el troquelado no queda muy arriba.

    La búsqueda se hace sobre la banda superior de la imagen. La posición del
    hallazgo hay que convertirla a coordenadas de la imagen completa antes de
    aplicarle el límite; compararla contra la banda descartaba lecturas
    correctas y hundió la cobertura de '3.Cerca' al 4%.
    """
    # Banda = 60% de 2000 = 1200 px. Un hallazgo a y=950 está al 79% de la
    # banda, pero sólo al 47% de la imagen: debe aceptarse.
    ocr = FakeOcr([line("19S206055", y=950.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is not None


def test_serial_near_the_top_is_accepted():
    ocr = FakeOcr([line("19S206055", y=200.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is not None


def test_vertical_text_is_rejected():
    """Una caja más alta que ancha no es una línea de troquelado."""
    ocr = FakeOcr([line("19S206055", y=400.0, height=200.0, width=45.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is None


def test_low_confidence_reading_is_rejected():
    ocr = FakeOcr([line("19S206055", y=400.0, confidence=0.2)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is None


def test_unrelated_text_is_rejected():
    ocr = FakeOcr([line("CONTENIDO NETO", y=400.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is None


def test_partial_reading_contained_in_the_serial_is_accepted():
    """El OCR suele perder los primeros caracteres, más desgastados."""
    ocr = FakeOcr([line("206055", y=400.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is not None


def test_too_short_fragment_is_rejected():
    ocr = FakeOcr([line("055", y=400.0)])
    assert pa.locate_serial(image(), "19S206055", ocr, 1) is None


def test_located_box_is_in_original_image_coordinates():
    """Las cajas se anotan sobre la imagen original, no sobre la banda."""
    canvas = image(1000, 2000)
    ocr = FakeOcr([line("19S206055", y=500.0)])
    box, _confidence, _text = pa.locate_serial(canvas, "19S206055", ocr, 1)
    x1, y1, x2, y2 = box
    assert 0 <= y1 < y2 <= canvas.shape[0]
    assert 0 <= x1 < x2 <= canvas.shape[1]


@pytest.mark.parametrize("expected,read", [
    ("19S206055", "19S206O55"),   # 0 leído como O
    ("21S491074", "21S49IO74"),   # 1 como I, 0 como O
])
def test_ocr_confusions_still_match_the_expected_serial(expected: str, read: str):
    ocr = FakeOcr([line(read, y=400.0)])
    assert pa.locate_serial(image(), expected, ocr, 1) is not None


def test_marking_area_extends_below_the_serial():
    """El bloque troquelado continúa bajo el número de serie."""
    box = (100.0, 500.0, 400.0, 540.0)
    marking = pa.expand_marking(box, 1000, 2000)
    assert marking[1] < box[1], "debe abarcar algo por encima"
    assert marking[3] > box[3], "y bastante por debajo"
    assert (marking[3] - box[3]) > (box[1] - marking[1])
