"""Comprobaciones sobre el catálogo real del inventario.

Estas pruebas cubren el maestro que el servicio usa en producción, con sus
defectos incluidos. Si alguien regenera el catálogo y alguna propiedad deja de
cumplirse, conviene enterarse aquí y no en planta.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.analyze import MatchStatus
from app.services.catalog import CylinderCatalog

CATALOG_PATH = Path(__file__).resolve().parents[1] / "datasets" / "catalog.json"

pytestmark = pytest.mark.skipif(
    not CATALOG_PATH.exists(),
    reason="el catálogo real no está presente; ejecute training/build_catalog.py",
)


@pytest.fixture(scope="module")
def catalog() -> CylinderCatalog:
    return CylinderCatalog.from_file(CATALOG_PATH)


def test_catalog_covers_the_whole_inventory():
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert payload["count"] == 100, "el inventario tiene 100 cilindros"


def test_serial_pattern_is_not_over_constrained(catalog: CylinderCatalog):
    """Los seriales reales no comparten estructura: no debe imponerse máscara.

    Hay doce patrones distintos y longitudes de 3 a 10 caracteres. Fijar una
    máscara haría que el corrector cambiara caracteres correctos.
    """
    assert catalog.pattern.mask is None


def test_known_serial_resolves_exactly(catalog: CylinderCatalog):
    match = catalog.resolve("19S206055")
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == "19S206055"


@pytest.mark.parametrize("degraded,expected", [
    ("195206055", "19S206055"),   # S leído como 5
    ("21S49IO74", "21S491074"),   # 1 como I, 0 como O
    ("K573BO28", "K5738028"),     # 8 como B, 0 como O
])
def test_ocr_confusions_resolve_against_the_real_inventory(
    catalog: CylinderCatalog, degraded: str, expected: str
):
    match = catalog.resolve(degraded)
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == expected


def test_duplicate_serial_is_known_and_flagged(catalog: CylinderCatalog):
    """K5738166 está registrado en dos cilindros distintos del inventario."""
    entry = catalog.get("K5738166")
    assert entry is not None
    assert entry.is_ambiguous
    assert set(entry.cilindros) == {"Cilindro_051", "Cilindro_053"}
    assert catalog.integrity_warnings("K5738166")


def test_short_serial_is_known_and_flagged(catalog: CylinderCatalog):
    """El cilindro 081 lleva el serial '804', de sólo tres caracteres."""
    entry = catalog.get("804")
    assert entry is not None
    assert entry.is_too_short
    assert catalog.integrity_warnings("804")


def test_unknown_serial_is_rejected(catalog: CylinderCatalog):
    assert catalog.resolve("ZZ9999999").status is MatchStatus.NOT_FOUND


def test_catalog_fields_are_normalized(catalog: CylinderCatalog):
    """Todo valor almacenado debe haber pasado por los normalizadores."""
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for record in payload["cylinders"]:
        if "procedencia" in record:
            assert len(record["procedencia"]) == 3, record["numero_serie"]
        if "ph" in record:
            assert len(record["ph"]) == 7 and record["ph"][4] == "-"
        if "largo_altura" in record:
            # Alturas siempre en centímetros y dentro de un rango físico.
            assert 30.0 <= record["largo_altura"] <= 200.0, record["numero_serie"]
        if "tipo_gas" in record:
            assert record["tipo_gas"].isupper()


# --- Seriales cortos: riesgo de identificación cruzada ---------------------

SHORT_SERIALS_IN_INVENTORY = ["20640", "804", "001087", "P30176", "42705"]


@pytest.mark.parametrize("serial", SHORT_SERIALS_IN_INVENTORY)
def test_short_serials_are_flagged_for_manual_confirmation(
    catalog: CylinderCatalog, serial: str
):
    """Un serial corto puede coincidir con una lectura parcial de otro.

    Caso observado al medir sobre el conjunto de test: en una fotografía del
    cilindro '21S062189' el OCR leyó el fragmento '20640', que es el serial
    COMPLETO del Cilindro_015. Sin esta salvaguarda el movimiento se habría
    registrado en el cilindro equivocado y nada lo habría delatado.
    """
    entry = catalog.get(serial)
    assert entry is not None, f"{serial} debería estar en el inventario"
    assert entry.is_too_short is True
    assert catalog.integrity_warnings(serial)


def test_long_serials_are_not_flagged(catalog: CylinderCatalog):
    for serial in ("19S206055", "21S062189", "K5738028"):
        entry = catalog.get(serial)
        assert entry is not None and entry.is_too_short is False
        assert catalog.integrity_warnings(serial) == []
