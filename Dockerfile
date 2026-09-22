# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Imagen de INFERENCIA. No incluye la pila de entrenamiento (PyTorch,
# ultralytics): el modelo llega ya exportado a ONNX, lo que ahorra más de 1 GB
# y acorta el arranque.
#
# Construcción en dos etapas para que las dependencias compiladas no dejen
# herramientas de compilación en la imagen final.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/ ./requirements/

# ARG OCR selecciona el motor:
#   rapid  -> RapidOCR sobre ONNX Runtime (por omisión, ~700 MB de imagen)
#   paddle -> PaddleOCR (~1,6 GB; sólo si se necesita su mayor cobertura)
#   none   -> sin OCR, para pruebas de contrato
ARG OCR=rapid
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && case "$OCR" in \
         paddle) /opt/venv/bin/pip install -r requirements/ocr-paddle.txt ;; \
         none)   /opt/venv/bin/pip install -r requirements/base.txt ;; \
         *)      /opt/venv/bin/pip install -r requirements/ocr.txt \
                 && /opt/venv/bin/pip install --no-deps -r requirements/ocr-engine.txt ;; \
       esac

# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Sólo se instala la variante headless de OpenCV (ver requirements/ocr.txt),
# así que no hace falta libGL. libglib sí la usan algunas rutas de OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AI_ENVIRONMENT=prod \
    AI_LOG_JSON=true \
    AI_OCR_BACKEND=rapid

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser datasets/catalog.example.json ./datasets/catalog.example.json

# Los modelos y el catálogo real se montan como volumen: sustituir un modelo
# no debe obligar a reconstruir la imagen.
RUN mkdir -p /app/models /app/datasets/audit && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# /health comprueba que el proceso responde. La disponibilidad real de los
# componentes se consulta en /ready, que un orquestador puede usar como
# readinessProbe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
