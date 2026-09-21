"""Construcción y acceso a las dependencias del servicio.

Los componentes pesados (detector, OCR, catálogo) se crean una sola vez al
arrancar y se reutilizan en todas las peticiones. Cargarlos por request
multiplicaría la latencia por un factor de decenas.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability.audit import AuditLog
from app.ocr.base import OcrEngine
from app.ocr.registry import build_ocr_engine
from app.services.backend_client import BackendClient
from app.services.catalog import CylinderCatalog
from app.services.pipeline import AnalysisPipeline
from app.vision.base import Detector
from app.vision.registry import build_detector

logger = get_logger(__name__)


@dataclass
class ServiceContainer:
    settings: Settings
    detector: Detector
    ocr_engine: OcrEngine
    catalog: CylinderCatalog
    backend: BackendClient
    audit: AuditLog
    pipeline: AnalysisPipeline


def build_container(settings: Settings | None = None) -> ServiceContainer:
    settings = settings or get_settings()

    detector = build_detector(settings)
    ocr_engine = build_ocr_engine(settings)
    catalog = CylinderCatalog.from_file(
        settings.catalog_path,
        max_distance=float(settings.catalog_fuzzy_max_distance),
        ambiguity_margin=float(settings.catalog_fuzzy_margin),
    )
    backend = BackendClient(settings)
    audit = AuditLog(
        settings.audit_dir,
        enabled=settings.audit_enabled,
        store_images=settings.audit_store_images_on_review,
        retention_days=settings.audit_retention_days,
    )
    pipeline = AnalysisPipeline(
        settings=settings, detector=detector, ocr_engine=ocr_engine,
        catalog=catalog, backend=backend, audit=audit,
    )

    logger.info(
        "componentes inicializados",
        extra={"detector": detector.describe(), "ocr": ocr_engine.describe(),
               "catalogo": len(catalog)},
    )
    return ServiceContainer(
        settings=settings, detector=detector, ocr_engine=ocr_engine,
        catalog=catalog, backend=backend, audit=audit, pipeline=pipeline,
    )


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_pipeline(request: Request) -> AnalysisPipeline:
    return request.app.state.container.pipeline
