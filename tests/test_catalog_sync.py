"""Pruebas de la sincronización del catálogo con Supabase.

Se simula la API REST de Supabase con un transporte de httpx, de modo que las
pruebas no dependen de la red ni de la base de datos real.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.schemas.analyze import MatchStatus
from app.services.catalog import CylinderCatalog, entries_from_records
from app.services.catalog_sync import CatalogSync, supabase_row_to_record

STATIC = [
    {"numero_serie": "19S206055", "marca": "JP5", "tipo_gas": "CO2", "cilindro": "Cilindro_002"},
    {"numero_serie": "K5738028", "marca": "JD", "tipo_gas": "O2", "cilindro": "Cilindro_050"},
]

# Filas tal como las guarda la aplicación móvil en la tabla `cilindros`.
ROW_NEW = {
    "id": 7, "numero_serie": "74054821", "marca": "norris",
    "anio_fabricacion": "2021", "tipo_gas": "DIÓXIDO DE CARBONO",
    "unidad_medida": "Kg", "peso": 48, "ancho_diametro": 22, "largo_altura": 1.3,
    "litros": 40, "contenido": None, "color": "gris",
    "ph": "CADA 5 AÑOS PH", "procedencia": "CN",
}
ROW_OVERRIDE = {
    "id": 8, "numero_serie": "19s206055", "marca": "JP5 corregida",
    "anio_fabricacion": None, "tipo_gas": "DIÓXIDO DE CARBONO",
    "unidad_medida": None, "peso": None, "ancho_diametro": None,
    "largo_altura": None, "litros": None, "contenido": None,
    "color": None, "ph": None, "procedencia": None,
}


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "supabase_url": "https://demo.supabase.co",
        "supabase_key": "anon-key",
    }
    values.update(overrides)
    return Settings(**values)


def supabase_returning(rows: list[dict] | None = None, *, status: int = 200,
                       seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"message": "error"})
        return httpx.Response(200, json=rows or [])
    return httpx.MockTransport(handler)


def catalog_from(records: list[dict]) -> CylinderCatalog:
    return CylinderCatalog(entries_from_records(records))


# --- Conversión de filas -----------------------------------------------------

def test_row_is_normalized_like_the_dataset_records():
    """La app guarda 'DIÓXIDO DE CARBONO' y 'CADA 5 AÑOS PH'; el catálogo no."""
    record = supabase_row_to_record(ROW_NEW)
    assert record is not None
    assert record["numero_serie"] == "74054821"
    assert record["tipo_gas"] == "CO2"
    assert record["marca"] == "NORRIS"
    assert record["ph_periodicidad_anios"] == 5
    assert record["procedencia"] == "CHN"
    # 1.3 como altura es imposible en centímetros: son metros.
    assert record["largo_altura"] == pytest.approx(130.0)
    assert record["cilindro"] == "cilindros#7"


def test_acetylene_content_is_used_when_there_are_no_litres():
    row = dict(ROW_NEW, litros=None, contenido=6, tipo_gas="ACETILENO")
    record = supabase_row_to_record(row)
    assert record["litros"] == pytest.approx(6.0)


def test_row_without_serial_is_discarded():
    assert supabase_row_to_record(dict(ROW_NEW, numero_serie="  ")) is None


# --- Sincronización ----------------------------------------------------------

@pytest.mark.asyncio
async def test_disabled_without_credentials_keeps_the_static_catalog():
    catalog = catalog_from(STATIC)
    sync = CatalogSync(Settings(_env_file=None), catalog, STATIC)
    await sync.startup()

    assert sync.enabled is False
    assert len(catalog) == 2
    assert "estático" in sync.status.describe()


@pytest.mark.asyncio
async def test_a_cylinder_registered_in_the_app_becomes_recognizable():
    """Es la razón de ser de la sincronización.

    Un cilindro que el operario acaba de registrar en la aplicación no está en
    el catálogo estático. Tras sincronizar, una lectura OCR con un carácter
    confundido debe resolverse a él.
    """
    catalog = catalog_from(STATIC)
    assert catalog.resolve("74054821").status is MatchStatus.NOT_FOUND

    sync = CatalogSync(settings(), catalog, STATIC,
                       transport=supabase_returning([ROW_NEW]))
    await sync.startup()

    match = catalog.resolve("7405482I")  # 1 leído como I
    assert match.status is MatchStatus.MATCHED
    assert match.resolved_serial == "74054821"
    await sync.shutdown()


@pytest.mark.asyncio
async def test_supabase_wins_over_the_static_catalog_for_the_same_serial():
    catalog = catalog_from(STATIC)
    sync = CatalogSync(settings(), catalog, STATIC,
                       transport=supabase_returning([ROW_OVERRIDE]))
    await sync.startup()

    entry = catalog.get("19S206055")
    assert entry.attributes["marca"] == "JP5 CORREGIDA"
    assert entry.cilindros == ("cilindros#8",)
    # Un serial que está en los dos sitios no cuenta como duplicado.
    assert entry.is_ambiguous is False
    await sync.shutdown()


@pytest.mark.asyncio
async def test_static_only_cylinders_are_kept():
    """Los cilindros del dataset aún no registrados en la app no se pierden."""
    catalog = catalog_from(STATIC)
    sync = CatalogSync(settings(), catalog, STATIC,
                       transport=supabase_returning([ROW_NEW]))
    await sync.startup()

    assert catalog.get("K5738028") is not None
    assert len(catalog) == 3
    assert sync.status.supabase_count == 1
    await sync.shutdown()


@pytest.mark.asyncio
async def test_a_failure_keeps_the_last_good_catalog():
    """Nunca se vacía el catálogo por un fallo de red."""
    catalog = catalog_from(STATIC)
    sync = CatalogSync(settings(), catalog, STATIC,
                       transport=supabase_returning(status=503))
    await sync.startup()

    assert len(catalog) == 2
    assert sync.status.last_error == "HTTP 503"
    assert sync.status.last_success_at is None
    assert "HTTP 503" in sync.status.describe()
    await sync.shutdown()


@pytest.mark.asyncio
async def test_request_reads_only_the_needed_columns_with_the_key():
    """No se piden datos de clientes, y la clave viaja en las cabeceras."""
    seen: list[httpx.Request] = []
    catalog = catalog_from(STATIC)
    sync = CatalogSync(settings(), catalog, STATIC,
                       transport=supabase_returning([], seen=seen))
    await sync.startup()

    request = seen[0]
    assert request.url.path == "/rest/v1/cilindros"
    columns = request.url.params["select"]
    assert "cliente" not in columns and "*" not in columns
    assert request.headers["apikey"] == "anon-key"
    assert request.headers["authorization"] == "Bearer anon-key"
    await sync.shutdown()


@pytest.mark.asyncio
async def test_large_inventories_are_paginated():
    """PostgREST limita las filas por respuesta: no se debe truncar en silencio."""
    pages = {
        0: [dict(ROW_NEW, id=i, numero_serie=f"A{i:07d}") for i in range(1000)],
        1000: [dict(ROW_NEW, id=1000, numero_serie="B0000001")],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].split("-")[0])
        return httpx.Response(200, json=pages.get(start, []))

    catalog = catalog_from([])
    sync = CatalogSync(settings(), catalog, [], transport=httpx.MockTransport(handler))
    await sync.startup()

    assert sync.status.supabase_count == 1001
    assert catalog.get("B0000001") is not None
    await sync.shutdown()


# --- CORS --------------------------------------------------------------------

def test_cors_is_open_outside_production_by_default():
    assert Settings(_env_file=None, environment="dev").cors_origins == ["*"]


def test_cors_is_closed_in_production_unless_configured():
    assert Settings(_env_file=None, environment="prod").cors_origins == []


def test_cors_origins_for_the_mobile_app():
    configured = Settings(
        _env_file=None, environment="prod",
        cors_allow_origins="capacitor://localhost, http://localhost ,http://localhost:8100",
    )
    assert configured.cors_origins == [
        "capacitor://localhost", "http://localhost", "http://localhost:8100",
    ]


def test_version_reports_the_catalog_source(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.deps import build_container
    from app.main import app

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"cylinders": STATIC}), encoding="utf-8")
    app.state.container = build_container(Settings(
        _env_file=None, ocr_backend="null", catalog_path=catalog_path,
        audit_enabled=False,
    ))
    with TestClient(app) as client:
        body = client.get("/api/v1/version").json()
    assert body["catalog_source"] == "estatico"
    assert body["catalog_size"] == 2
