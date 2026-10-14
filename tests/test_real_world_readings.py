"""Casos de las pruebas con teléfono sobre cilindros reales (serie K57...).

Cada prueba fija un fallo observado: lo que leyó el OCR y lo que el servicio
respondía antes de corregirlo.
"""
from __future__ import annotations

import pytest

from app.ocr.layout import is_plausible_text_box, join_split_lines
from app.parsing.serial import (
    allowed_errors,
    extract_serial_candidates,
    fuzzy_substring_distance,
    is_non_serial_marking,
)
from app.schemas.analyze import MatchStatus
from app.schemas.common import BoundingBox, OcrLine
from app.services.catalog import CatalogEntry, CylinderCatalog

INVENTORY = [
    "K5738060", "K5738144", "K5738166", "K5738003", "K5732188", "K5732198",
    "K5732103", "19S206055", "804", "20640", "26K010018", "26K011018",
]


@pytest.fixture
def catalog() -> CylinderCatalog:
    return CylinderCatalog([CatalogEntry(numero_serie=s, attributes={}) for s in INVENTORY])


def line(text: str, x1: float, x2: float, *, y1: float = 100, y2: float = 140,
         confidence: float = 0.9) -> OcrLine:
    return OcrLine(text=text, confidence=confidence, box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))


# -- tolerancia proporcional --------------------------------------------------

def test_short_serials_require_an_exact_reading():
    assert allowed_errors("804") == 0
    assert allowed_errors("20640") == 0
    assert allowed_errors("K5738060") == 2


def test_three_digit_reading_is_not_taken_for_a_short_serial(catalog):
    # Antes: el OCR leyó "300" y el servicio dio por bueno el cilindro "804".
    assert catalog.resolve("300").status is not MatchStatus.MATCHED
    assert catalog.search(["06144", "300"]).status is not MatchStatus.MATCHED


def test_unrelated_reading_offers_no_unrelated_alternatives(catalog):
    # Antes se ofrecían 20640, 804... para una lectura como "200".
    assert catalog.search(["AL", "200", "D0300"]).candidates == []


# -- marcas que no son el serial ---------------------------------------------

@pytest.mark.parametrize(
    "text", ["PW200", "0.5LPW200", "PH300", "BH300BAR", "2021/05", "09/19", "UN1006", "ONU1072"]
)
def test_pressure_and_date_markings_are_not_serials(text):
    assert is_non_serial_marking(text)
    assert extract_serial_candidates([(text, 1.0)]) == []


@pytest.mark.parametrize("text", ["K5738060", "19S206055", "K5738 060"])
def test_real_serials_are_not_mistaken_for_markings(text):
    assert not is_non_serial_marking(text)


# -- búsqueda del serial dentro de lo leído ----------------------------------

def test_fuzzy_substring_ignores_noise_around_the_serial():
    assert fuzzy_substring_distance("K5738060", "2K5738060") == 0
    assert fuzzy_substring_distance("K5738060", "JDK573306001") == 1


@pytest.mark.parametrize("texts", [["K5733060"], ["2K5738060", "0.5LPW200"], ["JD K573306001"]])
def test_serial_is_found_inside_noisy_readings(catalog, texts):
    match = catalog.search(texts)
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == "K5738060"


def test_marking_read_with_more_confidence_does_not_hide_the_serial(catalog):
    # El OCR lee "PW200" al 100% y el serial con menos confianza: debe ganar
    # el que existe en el inventario.
    assert catalog.search(["PW200", "K5738144"]).resolved_serial == "K5738144"


def test_short_serial_only_counts_as_a_whole_number(catalog):
    assert catalog.search(["804"]).resolved_serial == "804"
    assert catalog.search(["18043"]).status is not MatchStatus.MATCHED


def test_two_serials_one_character_apart_need_a_clear_reading(catalog):
    # "26K010018" y "26K011018" sólo difieren en un carácter. Una lectura
    # limpia identifica; una dudosa justo en ese carácter, no: se ofrecen los
    # dos y decide el operario.
    assert catalog.search(["26K010018"]).resolved_serial == "26K010018"
    for doubtful in (["26K01O018"], ["26K01X018"]):
        match = catalog.search(doubtful)
        assert match.status is MatchStatus.AMBIGUOUS
        assert {"26K010018", "26K011018"} <= {c.numero_serie for c in match.candidates}


# -- seriales partidos en dos cajas --------------------------------------------

def test_split_serial_is_joined_back():
    # "K5738 060": el espacio troquelado hace que el OCR devuelva dos cajas.
    joined = join_split_lines([line("060", 300, 380), line("K5738", 100, 280)])
    assert [j.text for j in joined] == ["K5738 060"]


def test_lines_at_different_heights_are_not_joined():
    assert join_split_lines([line("K5738", 100, 280), line("PW200", 290, 400, y1=200, y2=240)]) == []


def test_distant_boxes_on_the_same_line_are_not_joined():
    assert join_split_lines([line("K5738060", 100, 380), line("UT", 700, 760)]) == []


def test_split_serial_resolves_once_joined(catalog):
    pieces = [line("K5738", 100, 280), line("144", 300, 380)]
    texts = [p.text for p in pieces] + [j.text for j in join_split_lines(pieces)]
    assert catalog.search(texts).resolved_serial == "K5738144"


def test_vertical_box_is_not_plausible_text():
    assert not is_plausible_text_box(line("K5738060", 100, 140, y1=0, y2=300))


def test_configured_label_codes_are_ignored():
    from app.core.config import Settings

    settings = Settings(_env_file=None, ignored_readings="1892847, 12-34")
    assert settings.ignored_reading_set == {"1892847", "1234"}
