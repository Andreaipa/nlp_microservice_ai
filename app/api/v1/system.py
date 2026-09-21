"""Endpoints operativos: salud, disponibilidad y versión."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import ServiceContainer, get_container
from app.schemas.analyze import (
    ComponentStatus,
    HealthResponse,
    ReadinessResponse,
    VersionResponse,
)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Vivacidad del proceso")
async def health(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    """Responde mientras el proceso esté vivo. No comprueba dependencias."""
    return HealthResponse(
        status="ok",
        service=container.settings.app_name,
        environment=container.settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Disponibilidad real de los componentes",
)
async def ready(
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> ReadinessResponse:
    """Comprueba si el servicio puede atender análisis con garantías.

    El detector heurístico cuenta como no listo: el servicio responde, pero sin
    modelo entrenado, y un orquestador debe poder distinguir ese estado.
    """
    detector_ready = container.detector.name != "heuristic"
    components = [
        ComponentStatus(
            name="detector",
            ready=detector_ready,
            detail=container.detector.describe(),
        ),
        ComponentStatus(
            name="ocr",
            ready=container.ocr_engine.is_ready(),
            detail=container.ocr_engine.describe(),
        ),
        ComponentStatus(
            name="catalog",
            ready=not container.catalog.is_empty,
            detail=f"{len(container.catalog)} cilindros cargados",
        ),
        ComponentStatus(
            name="thresholds",
            ready=container.settings.thresholds_calibrated,
            detail="umbrales calibrados" if container.settings.thresholds_calibrated
            else "umbrales sin calibrar: todo resultado se marca para revisión",
        ),
    ]

    is_ready = all(component.ready for component in components)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(ready=is_ready, components=components)


@router.get("/version", response_model=VersionResponse, summary="Versiones en uso")
async def version(container: ServiceContainer = Depends(get_container)) -> VersionResponse:
    settings = container.settings
    return VersionResponse(
        service=settings.app_name,
        service_version=settings.service_version,
        model_version=container.pipeline.model_version,
        detector_backend=container.detector.describe(),
        ocr_backend=container.ocr_engine.describe(),
        catalog_size=len(container.catalog),
        thresholds_calibrated=settings.thresholds_calibrated,
    )
