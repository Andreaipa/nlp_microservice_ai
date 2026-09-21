"""Punto de entrada del microservicio de IA para cilindros industriales.

Responsabilidad del servicio: dada una fotografía, decir qué información pudo
identificar y con cuánta confianza. La interpretación de negocio (qué hacer
con ese cilindro, qué movimiento registrar) corresponde al backend principal.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import build_container
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.context import get_request_id, new_request_id, set_request_id
from app.core.errors import AIServiceError
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "iniciando servicio",
        extra={"version": settings.service_version, "entorno": settings.environment},
    )
    # Un contenedor ya inyectado (por ejemplo desde las pruebas) tiene
    # prioridad: reconstruirlo aquí descartaría la configuración de prueba.
    container = getattr(app.state, "container", None) or build_container(settings)
    await container.backend.startup()
    app.state.container = container

    if container.detector.name == "heuristic":
        logger.warning(
            "sin modelo de deteccion entrenado: el servicio opera en modo degradado"
        )
    if not settings.thresholds_calibrated:
        logger.warning(
            "umbrales sin calibrar: toda respuesta se marcara para confirmacion manual"
        )

    try:
        yield
    finally:
        await container.backend.shutdown()
        logger.info("servicio detenido")


API_DESCRIPTION = """
Microservicio de visión artificial para la identificación de cilindros
industriales a partir de una fotografía del troquelado de su collarín.

## Qué hace y qué no

Responde a una sola pregunta: **¿qué información puedo identificar en esta
imagen?**. No decide nada de negocio. El backend principal es quien determina
qué movimiento registrar con esa información.

```
imagen → detección de regiones → recorte → realce → OCR
       → normalización → resolución contra el inventario → JSON
```

## Cómo interpretar la respuesta

El campo que gobierna la decisión es **`requires_manual_confirmation`**.
Cuando es `true`, el backend no debe registrar nada automáticamente:
`review_reasons` explica por qué y `match.candidates` ofrece los cilindros
más parecidos para que el operario elija.

Cada dato llega como `{value, confidence, source}`. El campo `source` indica
de dónde salió:

| source | significado |
|---|---|
| `ai` | leído de la fotografía |
| `catalog` | completado desde el inventario local |
| `backend` | aportado por el backend principal |
| `manual` | confirmado por una persona |
| `unknown` | no se pudo determinar |

Un campo con `value: null` y `source: "unknown"` significa que no se pudo
determinar. **El servicio nunca rellena huecos por verosimilitud.**

## Estado actual del reconocimiento

Medido sobre 90 fotografías de cilindros no vistos en el entrenamiento:
**22% de identificación correcta en general y 60% en tomas cercanas al
collarín**, con **0% de errores silenciosos** y una latencia mediana de
**667 ms**.

La calibración empírica de umbrales concluyó que no existe un punto de corte
que permita aceptar lecturas automáticamente con menos del 1% de error, de
modo que el servicio **siempre pide confirmación**. Es un asistente de
registro, no un lector autónomo.
"""

TAGS_METADATA = [
    {
        "name": "cylinders",
        "description": "Análisis de fotografías de cilindros.",
    },
    {
        "name": "system",
        "description": (
            "Estado del servicio. `/health` para vivacidad, `/ready` para "
            "disponibilidad real de los componentes y `/version` para saber "
            "qué modelos están en uso."
        ),
    },
]

app = FastAPI(
    title="AI Cylinder Service",
    version=settings.service_version,
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    contact={"name": "Documentación del proyecto", "url": "https://localhost:4321"},
    license_info={"name": "Uso interno"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# El consumidor previsto es el backend principal, no un navegador. CORS queda
# abierto sólo fuera de producción, para poder probar desde herramientas web.
if settings.environment != "prod":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Asigna un identificador a cada petición y lo propaga en logs y respuesta."""
    incoming = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming or new_request_id()
    set_request_id(request_id)

    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(AIServiceError)
async def handle_service_error(request: Request, exc: AIServiceError) -> JSONResponse:
    logger.warning("error controlado", extra={"code": exc.code, "detalle": exc.message})
    return JSONResponse(
        status_code=exc.http_status,
        content={"success": False, "request_id": get_request_id(), "error": exc.to_payload()},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "request_id": get_request_id(),
            "error": {
                "code": "validation_error",
                "message": "La petición no cumple el contrato esperado.",
                "details": {"errors": exc.errors()},
            },
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("error no controlado")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "request_id": get_request_id(),
            "error": {
                "code": "internal_error",
                "message": "Error interno del servicio.",
                "details": {},
            },
        },
    )


app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.service_version,
        "docs": "/docs",
        "analyze": f"{settings.api_prefix}/cylinders/analyze",
    }
