"""Pruebas de normalización de campos."""
from __future__ import annotations

import pytest

from app.parsing import normalizers as norm


@pytest.mark.parametrize("raw,expected", [
    ("DIÓXIDO DE CARBONO CO2", "CO2"),
    ("CO2", "CO2"),
    ("C02", "CO2"),                 # cero en lugar de O, error típico de OCR
    ("OXIGENO MEDICINAL", "O2"),
    ("ARGON", "AR"),
    ("AIRE COMPRIMIDO", "AIR"),     # no debe confundirse con ARGON
    ("basura", None),
])
def test_normalize_gas(raw, expected):
    assert norm.normalize_gas(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("GRIS", "gris"), ("PLOMO", "gris"), ("Verde", "verde"), ("zzz", None),
])
def test_normalize_color(raw, expected):
    assert norm.normalize_color(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("09/2019", "2019-09"),
    ("02/2025", "2025-02"),
    ("2019/09", "2019-09"),
    ("02/25", "2025-02"),           # año abreviado, habitual en troquelado
    ("09-19", "2019-09"),
    ("sin fecha", None),
])
def test_normalize_month_year(raw, expected):
    assert norm.normalize_month_year(raw) == expected


def test_normalize_year_from_month_year():
    assert norm.normalize_year("AÑO DE FABRICACIÓN: 09/2019") == 2019


@pytest.mark.parametrize("raw,expected", [
    ("45.2 KG", 45.2), ("45,2 KG", 45.2), ("100 LB", 45.36), ("", None),
])
def test_normalize_weight(raw, expected):
    result = norm.normalize_weight_kg(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, abs=0.05)


def test_height_out_of_range_is_converted_and_reported():
    """'1.21 CM' es imposible para un cilindro de 40 L: son 1.21 m."""
    value, warning = norm.normalize_length_cm("1.21 CM", expected_range=norm.HEIGHT_RANGE_CM)
    assert value == pytest.approx(121.0)
    assert warning is not None and "metros" in warning


def test_diameter_in_range_is_untouched():
    value, warning = norm.normalize_length_cm("22 CM", expected_range=norm.DIAMETER_RANGE_CM)
    assert value == pytest.approx(22.0)
    assert warning is None


def test_implausible_length_is_rejected_rather_than_guessed():
    value, warning = norm.normalize_length_cm("9999 CM", expected_range=norm.HEIGHT_RANGE_CM)
    assert value is None
    assert warning is not None


def test_country_only_accepts_known_codes():
    assert norm.normalize_country("PROCEDENCIA: CHN") == "CHN"
    assert norm.normalize_country("XYZ") is None


# --- Casos surgidos del inventario real (100 fichas) ------------------------

@pytest.mark.parametrize("raw,expected", [
    ("80% ARGON 20% DIOXIDO DE CARBONO", "MIX"),   # mezcla, no argón puro
    ("MEZCLA ARGON CO2", "MIX"),
    ("OXIGENO", "O2"),
    ("NITRÓGENO", "N2"),
    ("ACETILENO", "ACET"),
    ("ARGÓN", "AR"),
])
def test_gas_variants_from_real_inventory(raw, expected):
    assert norm.normalize_gas(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("CHN", "CHN"), ("CN", "CHN"),          # los dos códigos conviven
    ("BRAZIL", "BRA"), ("BRASIL", "BRA"),
    ("ARGENTINA", "ARG"), ("USA", "USA"),
])
def test_country_variants_from_real_inventory(raw, expected):
    assert norm.normalize_country(raw) == expected


@pytest.mark.parametrize("marker", ["SN", "S/N", "N/A", "-", ""])
def test_absent_markers_are_not_values(marker):
    """'SN' significa 'sin dato', no es un valor de marca ni de país."""
    assert norm.is_absent(marker) is True
    assert norm.normalize_country(marker) is None
    assert norm.normalize_gas(marker) is None


def test_country_code_does_not_match_inside_another_word():
    """'CO' no debe reconocerse como Colombia dentro de 'COLOR'."""
    assert norm.normalize_country("COLOR") is None


@pytest.mark.parametrize("raw,expected", [
    ("MAARON", "marron"),          # errata presente en una ficha
    ("MARRÓN", "marron"),
    ("MARRON/GRIS", "marron/gris"),  # bicolor: información real, se conserva
    ("VERDE", "verde"),
])
def test_color_variants_from_real_inventory(raw, expected):
    assert norm.normalize_color(raw) == expected


@pytest.mark.parametrize("raw,date,period", [
    ("02/2025", "2025-02", None),
    ("CADA 10 AÑOS", None, 10),      # periodicidad, no fecha
    ("CADA 5 AÑOS", None, 5),
    ("SN", None, None),
    ("", None, None),
])
def test_hydrostatic_test_accepts_date_or_period(raw, date, period):
    assert norm.normalize_hydrostatic_test(raw) == (date, period)


def test_period_does_not_become_a_date():
    """De 'cada 10 años' no se deduce cuándo fue la última prueba."""
    date, period = norm.normalize_hydrostatic_test("CADA 10 AÑOS")
    assert date is None
    assert period == 10
