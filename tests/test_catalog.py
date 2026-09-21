"""Pruebas de resolución contra el inventario conocido."""
from __future__ import annotations

import pytest

from app.schemas.analyze import MatchStatus
from app.services.catalog import CatalogEntry, CylinderCatalog

REAL = "19S206055"


def build_catalog(serials: list[str], **kwargs) -> CylinderCatalog:
    entries = [CatalogEntry(numero_serie=s, attributes={}) for s in serials]
    return CylinderCatalog(entries, **kwargs)


def test_exact_match():
    catalog = build_catalog([REAL])
    match = catalog.resolve(REAL)
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == REAL


def test_ocr_confusions_still_resolve_to_the_right_cylinder():
    """Es el motivo de ser del catálogo: rescatar lecturas imperfectas."""
    catalog = build_catalog([REAL])
    for degraded in ("195206055", "I9S206055", "19S2060S5", "19s2O6O55"):
        match = catalog.resolve(degraded)
        assert match.status is MatchStatus.MATCHED, degraded
        assert match.resolved_serial == REAL, degraded


def test_unknown_serial_is_not_forced_into_a_match():
    catalog = build_catalog([REAL])
    match = catalog.resolve("99Z999999")
    assert match.status is MatchStatus.NOT_FOUND
    assert match.resolved_serial is None


def test_two_close_candidates_are_reported_as_ambiguous():
    """Ante duda real no se elige: se deriva a una persona."""
    catalog = build_catalog(["19S206055", "19S206054"])
    match = catalog.resolve("19S20605X")
    assert match.status is MatchStatus.AMBIGUOUS
    assert match.resolved_serial is None
    assert len(match.candidates) >= 2


def test_empty_catalog_reports_no_catalog():
    catalog = CylinderCatalog([])
    match = catalog.resolve(REAL)
    assert match.status is MatchStatus.NO_CATALOG
    assert match.resolved_serial == REAL


def test_attributes_are_available_for_enrichment():
    catalog = CylinderCatalog(
        [CatalogEntry(numero_serie=REAL, attributes={"marca": "JP5", "litros": 40.4})]
    )
    entry = catalog.get(REAL)
    assert entry is not None
    assert entry.attributes["marca"] == "JP5"


# --- Defectos del inventario real ------------------------------------------

def test_duplicate_serial_is_flagged_not_silently_collapsed():
    """El inventario real repite K5738166 en dos cilindros distintos."""
    catalog = CylinderCatalog([
        CatalogEntry(numero_serie="K5738166", attributes={},
                     cilindros=("Cilindro_051", "Cilindro_053")),
    ])
    entry = catalog.get("K5738166")
    assert entry.is_ambiguous is True

    warnings = catalog.integrity_warnings("K5738166")
    assert warnings and "Cilindro_051" in warnings[0]


def test_short_serial_is_flagged_as_unreliable():
    """El cilindro 081 lleva el serial '804': tres dígitos no identifican."""
    catalog = CylinderCatalog([
        CatalogEntry(numero_serie="804", attributes={}, cilindros=("Cilindro_081",)),
    ])
    assert catalog.get("804").is_too_short is True
    assert catalog.integrity_warnings("804")


def test_normal_serial_has_no_integrity_warnings():
    catalog = build_catalog([REAL])
    assert catalog.integrity_warnings(REAL) == []


# --- Emparejamiento por fragmento -------------------------------------------

def test_partial_reading_resolves_when_it_identifies_one_cylinder():
    """El OCR pierde a menudo los primeros caracteres, más desgastados."""
    catalog = build_catalog([REAL, "K5738028"])
    match = catalog.resolve("206055")
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == REAL


def test_fragment_present_in_several_serials_is_ambiguous():
    """Si el fragmento no distingue, no se elige: decide una persona."""
    catalog = build_catalog(["19S206055", "20S206055"])
    match = catalog.resolve("206055")
    assert match.status is MatchStatus.AMBIGUOUS
    assert match.resolved_serial is None
    assert len(match.candidates) >= 2


def test_too_short_fragment_does_not_match():
    catalog = build_catalog([REAL])
    assert catalog.resolve("055").status is MatchStatus.NOT_FOUND


def test_reading_with_surrounding_noise_still_resolves():
    """'SERIE19S206055X': el serial va acompañado de ruido en la misma línea."""
    catalog = build_catalog([REAL])
    match = catalog.resolve("SERIE19S206055X")
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == REAL


@pytest.mark.parametrize("probe", ["206055", "SERIE19S206055X", REAL])
def test_fragment_score_stays_within_range(probe: str):
    """Regresión: un texto más largo que el serial daba score > 1."""
    catalog = build_catalog([REAL])
    match = catalog.resolve(probe)
    for candidate in match.candidates:
        assert 0.0 <= candidate.score <= 1.0
