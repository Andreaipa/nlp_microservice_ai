"""Endpoint de análisis de fotografías de cilindros."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import ServiceContainer, get_container
from app.core.context import get_request_id
from app.core.errors import UnsupportedMediaTypeError
from app.core.logging import get_logger
from app.schemas.analyze import AnalyzeResponse
from app.schemas.common import ErrorPayload

logger = get_logger(__name__)
router = APIRouter(prefix="/cylinders", tags=["cylinders"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analiza una fotografía de cilindro",
    description="""
Recibe una fotografía y devuelve la información que pudo identificarse en
ella, con su nivel de confianza y el origen de cada dato.

El microservicio no toma decisiones de negocio: informa de lo que observó.
Cuando la lectura no es concluyente marca `requires_manual_confirmation` y
detalla el motivo en `review_reasons`, en lugar de devolver un valor dudoso
como si fuera cierto.

### La fotografía importa más que cualquier parámetro

| Encuadre | Acierto medido |
|---|---|
| Cerca del troquelado | **60%** |
| De frente, lejos o en penumbra | 13-20% |

La aplicación cliente debería guiar al operario a encuadrar **el collarín**
de cerca. Es la diferencia entre que el sistema sirva y que no.

### Motivos de revisión

| `review_reasons` | Qué mostrar al operario |
|---|---|
| `no_serial_detected` | "No se distingue el número. Acérquese al collarín." |
| `low_ocr_confidence` | "Lectura poco clara. Confirme el número." |
| `ambiguous_catalog_match` | Ofrecer los candidatos de `match.candidates` |
| `serial_not_in_catalog` | "Ese cilindro no está en el inventario." |
| `duplicate_serial_in_catalog` | "Ese número está en más de un cilindro." |
| `serial_too_short` | "El número grabado es demasiado corto." |
| `poor_image_quality` | "Foto movida u oscura. Repítala." |
| `no_detector_model` | Estado del sistema, no del usuario |
| `thresholds_not_calibrated` | Estado del sistema, no del usuario |

### Cabeceras

Envíe `X-Request-ID` para poder rastrear una incidencia de punta a punta: el
servicio lo propaga a sus registros y lo devuelve en la respuesta.
""",
    responses={
        413: {"model": ErrorPayload, "description": "Imagen demasiado grande"},
        415: {"model": ErrorPayload, "description": "Tipo de archivo no admitido"},
        422: {"model": ErrorPayload, "description": "La imagen no se pudo decodificar"},
    },
)
async def analyze_cylinder(
    image: UploadFile = File(..., description="Fotografía del cilindro (JPEG, PNG o WEBP)."),
    container: ServiceContainer = Depends(get_container),
) -> AnalyzeResponse:
    settings = container.settings

    content_type = (image.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Tipo de archivo no admitido: {content_type}.",
            details={"allowed": list(settings.allowed_content_types)},
        )

    payload = await image.read()
    logger.info(
        "analisis solicitado",
        extra={"archivo": image.filename, "content_type": content_type,
               "bytes": len(payload)},
    )
    return await container.pipeline.analyze(payload, request_id=get_request_id())
