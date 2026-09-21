"""Pruebas de lectura y corrección del número de serie."""
from __future__ import annotations

import pytest

from app.parsing.serial import (
    MIN_SAMPLES_FOR_MASK,
    SerialPattern,
    apply_pattern_corrections,
    extract_serial_candidates,
    normalize_serial_charset,
    weighted_edit_distance,
)

REAL_SERIAL = "19S206055"


@pytest.mark.parametrize("raw,expected", [
    ("N° DE SERIE: 19S206055", "19S206055"),
    ("  19s206055 ", "19S206055"),
    ("SERIE 19-S20.6055", "19S206055"),
    ("", ""),
])
def test_normalize_charset(raw, expected):
    assert normalize_serial_charset(raw) == expected


def test_extracts_serial_from_labelled_line():
    candidates = extract_serial_candidates([("N° DE SERIE: 19S206055", 0.95)])
    assert candidates[0].text == REAL_SERIAL


def test_extracts_serial_embedded_in_noise():
    candidates = extract_serial_candidates([("JP5 19S206055 CO2", 0.8)])
    assert REAL_SERIAL in {c.text for c in candidates}


def test_no_correction_without_mask():
    """Sin patrón posicional no se corrige: cualquier cambio sería una suposición."""
    pattern = SerialPattern(mask=None)
    corrected, notes = apply_pattern_corrections("195206055", pattern)
    assert corrected == "195206055"
    assert notes == []


def test_correction_follows_expected_position_type():
    """Con máscara, un dígito en posición de letra se corrige a letra."""
    pattern = SerialPattern(mask="DDADDDDDD", min_length=9, max_length=9)
    corrected, notes = apply_pattern_corrections("195206055", pattern)
    assert corrected == REAL_SERIAL
    assert len(notes) == 1


def test_correction_does_not_touch_valid_characters():
    pattern = SerialPattern(mask="DDADDDDDD", min_length=9, max_length=9)
    corrected, notes = apply_pattern_corrections(REAL_SERIAL, pattern)
    assert corrected == REAL_SERIAL
    assert notes == []


def test_mask_requires_enough_samples():
    """Una sola muestra no basta para imponer una estructura posicional."""
    assert SerialPattern.infer_from_catalog([REAL_SERIAL]).mask is None

    many = [REAL_SERIAL] + [f"2{i}S20605{i}" for i in range(MIN_SAMPLES_FOR_MASK)]
    assert SerialPattern.infer_from_catalog(many).mask == "DDADDDDDD"


def test_mask_is_not_inferred_when_lengths_vary():
    assert SerialPattern.infer_from_catalog(
        ["19S206055", "20S2060", "21S20605512"] * 4
    ).mask is None


def test_confusable_substitution_is_cheaper():
    """'S'->'5' es un error de lectura esperable; 'S'->'X' no lo es."""
    confusable = weighted_edit_distance(REAL_SERIAL, "195206055")
    unrelated = weighted_edit_distance(REAL_SERIAL, "19X206055")
    assert confusable < unrelated
    assert confusable == pytest.approx(0.5)


def test_identical_serials_have_zero_distance():
    assert weighted_edit_distance(REAL_SERIAL, REAL_SERIAL) == 0.0
