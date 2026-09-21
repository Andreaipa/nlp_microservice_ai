"""Pruebas de la selección del motor OCR."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.config import Settings
from app.ocr.null_engine import NullOcrEngine
from app.ocr.registry import build_ocr_engine
from app.schemas.common import OcrLine


def make_settings(backend: str) -> Settings:
    return Settings(_env_file=None, ocr_backend=backend)


def test_null_backend_is_inert():
    engine = build_ocr_engine(make_settings("null"))
    assert isinstance(engine, NullOcrEngine)
    assert engine.read(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_rapid_is_the_default_backend():
    """RapidOCR es 10x más rápido que paddle en CPU: ver docs/RENDIMIENTO.md."""
    assert Settings(_env_file=None).ocr_backend == "rapid"


@pytest.mark.parametrize("backend,expected", [("rapid", "rapid"), ("paddle", "paddle")])
def test_registry_builds_the_requested_backend(backend: str, expected: str):
    engine = build_ocr_engine(make_settings(backend))
    assert engine.name == expected


def test_engines_share_the_same_contract():
    """El pipeline depende de la interfaz, no del motor concreto."""
    for backend in ("null", "rapid", "paddle"):
        engine = build_ocr_engine(make_settings(backend))
        assert callable(engine.read)
        assert callable(engine.is_ready)
        assert isinstance(engine.describe(), str)


def test_null_engine_can_replay_canned_lines():
    canned = [OcrLine(text="19S206055", confidence=0.9)]
    engine = NullOcrEngine(canned)
    assert engine.read(np.zeros((5, 5, 3), dtype=np.uint8))[0].text == "19S206055"


def test_unavailable_engine_degrades_instead_of_raising():
    """Un motor sin librería instalada no debe tumbar el arranque."""
    from app.ocr.rapid_engine import RapidOcrEngine

    engine = RapidOcrEngine()
    engine._failed = True
    assert engine.is_ready() is False
    assert engine.read(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    assert "no disponible" in engine.describe()
