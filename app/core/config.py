"""Configuración del microservicio.

Toda la configuración se resuelve desde variables de entorno (prefijo AI_) o
desde un archivo .env. No debe existir ningún valor operativo hardcodeado fuera
de este módulo.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Servicio -----------------------------------------------------------
    app_name: str = "ai-cylinder-service"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    debug: bool = False
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000

    # Versión del contrato de la API y versión del artefacto de modelo.
    # model_version identifica el conjunto (detector + ocr + parser) y viaja en
    # cada respuesta para poder auditar qué produjo un resultado.
    service_version: str = "1.0.0"
    model_version: str = "detector-none+ocr-paddle+parser-1.0.0"

    # --- Detección ----------------------------------------------------------
    # 'onnx'      -> modelo YOLO exportado a ONNX (producción)
    # 'ultralytics' -> pesos .pt vía ultralytics (desarrollo/depuración)
    # 'heuristic' -> detector OpenCV sin modelo entrenado (fallback operativo)
    detector_backend: Literal["onnx", "ultralytics", "heuristic"] = "heuristic"
    detector_model_path: Path = BASE_DIR / "models" / "detector.onnx"
    detector_input_size: int = 960
    detector_conf_threshold: float = 0.25
    detector_iou_threshold: float = 0.45
    detector_max_detections: int = 50
    # Márgenes al recortar una región detectada, como fracción de su tamaño.
    # El inferior es mayor a propósito: ver app/preprocessing/image_io.crop.
    detector_crop_padding: float = 0.15
    detector_crop_padding_bottom: float = 0.90

    # --- OCR ----------------------------------------------------------------
    # 'rapid'  -> RapidOCR sobre ONNX Runtime: ~10x más rápido que paddle en
    #             CPU y sin la dependencia de paddlepaddle. Es el recomendado
    #             para el despliegue previsto (equipos sin GPU).
    # 'paddle' -> PaddleOCR: reconoce más texto secundario, a cambio de
    #             latencia y de ~600 MB de dependencias.
    ocr_backend: Literal["rapid", "paddle", "null"] = "rapid"
    ocr_lang: str = "en"
    ocr_use_gpu: bool = False
    ocr_min_confidence: float = 0.30
    # Hilos del motor OCR. 0 deja decidir a la librería; limitarlo evita que
    # una sola petición acapare todos los núcleos del servidor.
    ocr_threads: int = 0

    # --- Reconocedor especializado del número de serie ----------------------
    # Modelo propio entrenado sobre recortes de troquelado. El serial es el
    # identificador del activo y un OCR genérico lo lee con un CER de 0.537 en
    # este dominio; un modelo con el alfabeto cerrado del inventario hace mejor
    # ese trabajo concreto. Si no está disponible, el pipeline usa el OCR
    # general: es una mejora, no un requisito.
    serial_recognizer_path: Path | None = None
    serial_recognizer_enabled: bool = True
    # Variantes de realce que se prueban por recorte. Cada una es una pasada
    # completa de OCR, así que el valor gobierna directamente la latencia: en
    # CPU, 5 variantes sobre 3 regiones superan los 8 s por imagen. Con 3 se
    # conserva la mayor parte del beneficio del consenso a un coste razonable.
    ocr_preprocess_variants: int = 3

    # --- Umbrales de decisión ----------------------------------------------
    # IMPORTANTE: estos valores son provisionales. Deben recalibrarse con
    # training/calibrate_thresholds.py sobre el conjunto de test real antes de
    # considerarlos definitivos. Ver docs/THRESHOLDS.md.
    serial_auto_accept_confidence: float = 0.90
    serial_review_confidence: float = 0.60
    catalog_fuzzy_max_distance: int = 2
    catalog_fuzzy_margin: int = 1
    thresholds_calibrated: bool = False

    # --- Catálogo de cilindros ---------------------------------------------
    # Inventario cerrado: restringir la salida del OCR al catálogo conocido es
    # lo que convierte un OCR imperfecto en una identificación fiable.
    catalog_path: Path = BASE_DIR / "datasets" / "catalog.json"
    catalog_enabled: bool = True

    # --- Backend principal --------------------------------------------------
    backend_base_url: str | None = None
    backend_api_key: str | None = None
    backend_timeout_seconds: float = 5.0
    backend_enabled: bool = False

    # --- Entrada ------------------------------------------------------------
    max_upload_bytes: int = 15 * 1024 * 1024
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    )
    max_image_side: int = 2048

    # --- Observabilidad -----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    audit_enabled: bool = True
    audit_dir: Path = BASE_DIR / "datasets" / "audit"
    # Guardar la imagen sólo cuando el caso requiere revisión manual, para no
    # acumular datos innecesarios.
    audit_store_images_on_review: bool = True
    audit_retention_days: int = 30

    @field_validator("detector_conf_threshold", "detector_iou_threshold",
                     "ocr_min_confidence", "serial_auto_accept_confidence",
                     "serial_review_confidence")
    @classmethod
    def _check_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("debe estar entre 0.0 y 1.0")
        return v

    @property
    def base_dir(self) -> Path:
        return BASE_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
