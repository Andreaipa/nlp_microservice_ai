"""Realce de imagen para OCR sobre caracteres estampados en metal.

Contexto del problema: en los cilindros de Electrametal el número de serie no
está impreso en una etiqueta, está TROQUELADO en el collarín metálico. Eso
significa que:

* el carácter y el fondo son del mismo color y material: no hay contraste
  cromático, sólo sombras producidas por el relieve;
* la legibilidad depende casi por completo del ángulo de la luz;
* el collarín es curvo, así que el texto sigue un arco y sufre escorzo;
* suele haber óxido, pintura, golpes y suciedad encima.

Un motor OCR genérico está entrenado sobre texto impreso y rinde mal aquí. Las
transformaciones de este módulo buscan convertir relieve en contraste.

Cada función devuelve una imagen en escala de grises lista para OCR. El
pipeline prueba varias variantes y se queda con la de mayor confianza: cuál
gana depende de la iluminación de cada foto, y no es predecible de antemano.
"""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# Alto objetivo del recorte antes de OCR. Los motores de reconocimiento
# trabajan mejor con texto de 32-64 px de alto; subir de ahí no aporta.
TARGET_TEXT_HEIGHT = 64
MAX_UPSCALE = 4.0


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale_for_ocr(image: np.ndarray, target_height: int = TARGET_TEXT_HEIGHT) -> np.ndarray:
    """Amplía recortes pequeños hasta una altura util para el reconocedor."""
    height = image.shape[0]
    if height <= 0:
        return image
    factor = min(MAX_UPSCALE, max(1.0, target_height / float(height)))
    if factor <= 1.01:
        return image
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def variant_clahe(image: np.ndarray) -> np.ndarray:
    """Ecualización adaptativa: recupera detalle en sombras y reflejos.

    Es la primera opción para fotos con poca luz, que en este dataset es una de
    las dos condiciones de captura previstas.
    """
    gray = to_gray(image)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def variant_relief(image: np.ndarray) -> np.ndarray:
    """Realza el relieve del troquelado mediante gradiente morfológico.

    El gradiente responde a los bordes del hundimiento del carácter, que es la
    única señal disponible cuando carácter y fondo comparten color.
    """
    gray = to_gray(image)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Normalizar y volver a polaridad "texto oscuro sobre fondo claro", que es
    # lo que esperan la mayoría de reconocedores.
    # Los stubs de OpenCV declaran `dst` obligatorio, aunque la función admite
    # None para que asigne la salida.
    normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-overload]
    return cv2.bitwise_not(normalized.astype(np.uint8))


def variant_tophat(image: np.ndarray) -> np.ndarray:
    """Black-hat: aísla depresiones oscuras sobre fondo metálico claro.

    Funciona bien con iluminación rasante, donde el hueco del carácter queda en
    sombra.
    """
    gray = to_gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    normalized = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore[call-overload]
    return cv2.bitwise_not(normalized.astype(np.uint8))


def variant_sharpen(image: np.ndarray) -> np.ndarray:
    """Máscara de enfoque suave. Útil cuando la foto está algo desenfocada."""
    gray = to_gray(image)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def variant_identity(image: np.ndarray) -> np.ndarray:
    """Sin realce. Sirve de referencia: a veces el original ya es el mejor."""
    return to_gray(image)


# Orden deliberado: de la transformación más general a la más agresiva. El
# pipeline las aplica en secuencia hasta alcanzar confianza suficiente.
OCR_VARIANTS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("clahe", variant_clahe),
    ("relief", variant_relief),
    ("sharpen", variant_sharpen),
    ("blackhat", variant_tophat),
    ("identity", variant_identity),
)


def build_ocr_variants(image: np.ndarray, limit: int) -> list[tuple[str, np.ndarray]]:
    """Genera hasta `limit` versiones realzadas y escaladas de un recorte."""
    if image.size == 0:
        return []
    variants: list[tuple[str, np.ndarray]] = []
    for name, fn in OCR_VARIANTS[: max(1, limit)]:
        try:
            processed = upscale_for_ocr(fn(image))
        except cv2.error:  # pragma: no cover - recorte degenerado
            continue
        variants.append((name, processed))
    return variants


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Corrige la inclinación del texto usando el rectángulo de área mínima.

    Sólo corrige rotaciones suaves. La curvatura del collarín es un problema
    distinto y lo trata `unwrap_arc`.
    """
    gray = to_gray(image)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def unwrap_arc(image: np.ndarray, curvature: float = 0.0) -> np.ndarray:
    """Aplana texto dispuesto sobre una superficie cilíndrica.

    `curvature` es la flecha del arco como fracción del alto del recorte. Con 0
    la función no toca la imagen. El valor adecuado depende de la distancia de
    captura y del diámetro del collarín; debe estimarse sobre imágenes reales
    antes de activarlo (ver docs/PIPELINE.md).
    """
    if curvature <= 0.0:
        return image

    height, width = image.shape[:2]
    sag = curvature * height
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = np.zeros((height, width), dtype=np.float32)

    xs = np.arange(width, dtype=np.float32)
    # Parábola centrada: aproxima el arco del collarín en el rango visible.
    offsets = sag * (1.0 - ((xs - width / 2.0) / (width / 2.0)) ** 2)
    for row in range(height):
        map_y[row] = row + offsets

    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
