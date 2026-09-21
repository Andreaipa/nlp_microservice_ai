"""Pruebas del contrato HTTP del microservicio."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import build_container
from app.core.config import Settings
from app.main import app


@pytest.fixture
def client(settings: Settings):
    app.state.container = build_container(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_available(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_version_reports_the_active_components(client):
    response = client.get("/api/v1/version")
    assert response.status_code == 200

    body = response.json()
    assert body["catalog_size"] >= 1
    # Sin modelo entrenado el servicio debe declararlo, no disimularlo.
    assert "heuristic" in body["detector_backend"]
    assert body["thresholds_calibrated"] is False


def test_readiness_fails_while_the_detector_is_untrained(client):
    response = client.get("/api/v1/ready")
    assert response.status_code == 503

    body = response.json()
    assert body["ready"] is False
    detector = next(c for c in body["components"] if c["name"] == "detector")
    assert detector["ready"] is False


def test_analyze_returns_a_complete_contract(client, stamped_jpeg: bytes):
    response = client.post(
        "/api/v1/cylinders/analyze",
        files={"image": ("cilindro.jpg", stamped_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["success"] is True
    assert body["request_id"]
    assert body["processing_time_ms"] >= 0

    # Los once campos pedidos por el contrato deben estar siempre presentes,
    # aunque su valor sea nulo.
    for field in ("numero_serie", "marca", "anio_fabricacion", "tipo_gas", "m3",
                  "peso", "ancho_diametro", "largo_altura", "litros", "color", "ph"):
        assert field in body["data"], field
        assert set(body["data"][field]) >= {"value", "confidence", "source"}


def test_analyze_requires_review_while_uncalibrated(client, stamped_jpeg: bytes):
    """Sin umbrales calibrados nada se da por bueno automáticamente."""
    response = client.post(
        "/api/v1/cylinders/analyze",
        files={"image": ("cilindro.jpg", stamped_jpeg, "image/jpeg")},
    )
    body = response.json()
    assert body["requires_manual_confirmation"] is True
    assert "thresholds_not_calibrated" in body["review_reasons"]


def test_analyze_rejects_non_image_content_type(client):
    response = client.post(
        "/api/v1/cylinders/analyze",
        files={"image": ("notas.txt", b"no soy una imagen", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_analyze_rejects_undecodable_bytes(client):
    response = client.post(
        "/api/v1/cylinders/analyze",
        files={"image": ("roto.jpg", b"\x00\x01\x02 basura", "image/jpeg")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_image"


def test_missing_file_is_a_validation_error(client):
    response = client.post("/api/v1/cylinders/analyze")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_request_id_is_echoed_back(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "traza-123"})
    assert response.headers["X-Request-ID"] == "traza-123"


def test_openapi_schema_is_generated(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/cylinders/analyze" in response.json()["paths"]
