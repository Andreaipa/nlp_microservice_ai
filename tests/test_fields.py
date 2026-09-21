"""Pruebas del parser de campos sobre el contenido real del cilindro 002."""
from __future__ import annotations

import pytest

from app.parsing.fields import parse_fields
from app.schemas.common import FieldSource

LABELLED = [
    ("N° DE SERIE: 19S206055", 0.95),
    ("MARCA: JP5", 0.93),
    ("AÑO DE FABRICACIÓN: 09/2019", 0.91),
    ("TIPO DE GAS: DIÓXIDO DE CARBONO CO2", 0.90),
    ("UNIDAD DE MEDIDA: KG", 0.92),
    ("PESO: 45.2 KG", 0.94),
    ("ANCHO DIÁMETRO: 22 CM", 0.89),
    ("LARGO ALTURA: 1.21 CM", 0.88),
    ("LITROS: 40.4 L", 0.93),
    ("COLOR: GRIS", 0.90),
    ("PH: 02/2025", 0.87),
    ("PROCEDENCIA: CHN", 0.86),
]

# Cómo se lee realmente el collarín: sin rótulos y con el año abreviado.
STAMPED = [
    ("JP5 19S206055", 0.72),
    ("FAB 09/19  CO2  45.2KG", 0.66),
    ("40.4L CHN", 0.70),
    ("PH 02/25", 0.61),
]


def test_labelled_document_is_parsed_completely():
    result = parse_fields(LABELLED)
    data = result.data
    assert data.marca.value == "JP5"
    assert data.anio_fabricacion.value == 2019
    assert data.fecha_fabricacion.value == "2019-09"
    assert data.tipo_gas.value == "CO2"
    assert data.unidad_medida.value == "KG"
    assert data.peso.value == pytest.approx(45.2)
    assert data.litros.value == pytest.approx(40.4)
    assert data.ancho_diametro.value == pytest.approx(22.0)
    assert data.color.value == "gris"
    assert data.procedencia.value == "CHN"


def test_ph_is_a_hydrostatic_test_date_not_acidity():
    result = parse_fields(LABELLED)
    assert result.data.ph.value == "2025-02"


def test_impossible_height_is_corrected_with_a_warning():
    result = parse_fields(LABELLED)
    assert result.data.largo_altura.value == pytest.approx(121.0)
    assert any("largo_altura" in w for w in result.warnings)


def test_stamped_text_without_labels_is_parsed():
    result = parse_fields(STAMPED)
    data = result.data
    assert data.marca.value == "JP5"
    assert data.tipo_gas.value == "CO2"
    assert data.fecha_fabricacion.value == "2019-09"
    assert data.ph.value == "2025-02"
    assert data.peso.value == pytest.approx(45.2)
    assert data.litros.value == pytest.approx(40.4)


def test_unlabelled_values_carry_lower_confidence():
    """Inferir por forma es menos fiable que leer un rótulo explícito."""
    labelled = parse_fields(LABELLED).data
    stamped = parse_fields(STAMPED).data
    assert stamped.marca.confidence < labelled.marca.confidence


def test_absent_fields_stay_unknown_instead_of_being_invented():
    result = parse_fields(STAMPED)
    assert result.data.m3.value is None
    assert result.data.m3.source is FieldSource.UNKNOWN
    assert result.data.ancho_diametro.value is None


def test_empty_input_produces_empty_data():
    result = parse_fields([])
    assert result.data.marca.value is None
    assert result.warnings == []


def test_field_markers_are_not_mistaken_for_a_brand():
    """'FAB' precede a la fecha de fabricación; no es el fabricante."""
    result = parse_fields([("FAB 09/19 CO2 45.2KG", 0.8)])
    assert result.data.marca.value != "FAB"


def test_country_code_is_not_mistaken_for_a_brand():
    result = parse_fields([("40.4L CHN", 0.8)])
    assert result.data.marca.value != "CHN"
    assert result.data.procedencia.value == "CHN"


# --- Falsos positivos observados sobre fotografías reales -------------------

def test_preposition_de_is_not_read_as_germany():
    """'DIÓXIDO DE CARBONO' no es procedencia Alemania.

    Ocurrió sobre una foto real: el OCR leyó "DIO XIDO DE CARSN" y el "DE" se
    tomó como el código ISO de Alemania.
    """
    result = parse_fields([("DIO XIDO DE CARSN", 0.8)])
    assert result.data.procedencia.value is None


def test_explicit_label_still_accepts_two_letter_codes():
    """Con la etiqueta delante no hay ambigüedad: 35 fichas usan 'CN'."""
    result = parse_fields([("PROCEDENCIA: CN", 0.8)])
    assert result.data.procedencia.value == "CHN"


def test_low_confidence_text_does_not_become_a_brand():
    """Una pegatina mal leída no debe convertirse en la marca del cilindro."""
    result = parse_fields([("Linca pgp", 0.45)])
    assert result.data.marca.value is None


def test_confident_text_still_yields_the_brand():
    result = parse_fields([("JP5 19S206055", 0.72)])
    assert result.data.marca.value == "JP5"


def test_measurements_do_not_need_high_confidence():
    """'45.2 KG' se valida por su forma: no hace falta exigirle confianza."""
    result = parse_fields([("45.2KG 40.4L", 0.35)])
    assert result.data.peso.value == pytest.approx(45.2)
    assert result.data.litros.value == pytest.approx(40.4)
