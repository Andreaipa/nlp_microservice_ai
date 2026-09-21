"""Carga y validación de imágenes de entrada."""
from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import ImageTooLargeError, InvalidImageError
from app.schemas.analyze import ImageQuality

# Umbrales de calidad. Son heurísticos y sirven para avisar, no para rechazar:
# una foto mediocre todavía puede producir una lectura correcta.
BLUR_MIN = 60.0
BRIGHTNESS_MIN = 25.0
BRIGHTNESS_MAX = 235.0
CONTRAST_MIN = 18.0


def decode_image(payload: bytes, *, max_bytes: int) -> np.ndarray:
    """Decodifica bytes a BGR. Lanza InvalidImageError si no es una imagen."""
    if not payload:
        raise InvalidImageError("El archivo recibido está vacío.")
    if len(payload) > max_bytes:
        raise ImageTooLargeError(
            f"La imagen supera el máximo permitido de {max_bytes} bytes.",
            details={"size": len(payload), "max_bytes": max_bytes},
        )

    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(
            "No se pudo decodificar la imagen. Formatos aceptados: JPEG, PNG, WEBP."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("Se esperaba una imagen en color de 3 canales.")
    return image


def limit_size(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Reduce la imagen si excede `max_side`.

    Devuelve la imagen y la escala aplicada, necesaria para reproyectar las
    cajas al sistema de coordenadas de la imagen original.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0

    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def assess_quality(image: np.ndarray) -> ImageQuality:
    """Mide nitidez, brillo y contraste de la imagen completa."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    acceptable = (
        blur >= BLUR_MIN
        and BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX
        and contrast >= CONTRAST_MIN
    )
    height, width = gray.shape[:2]
    return ImageQuality(
        width=width,
        height=height,
        blur_score=round(blur, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        is_acceptable=acceptable,
    )


def crop(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    padding: float = 0.06,
    padding_bottom: float | None = None,
) -> np.ndarray:
    """Recorta una caja con márgenes relativos, sin salirse de la imagen.

    `padding_bottom` permite un margen inferior mayor que el resto. Existe por
    una razón concreta del dominio: el troquelado está siempre por DEBAJO de
    la válvula y el asa, que es la parte del cilindro que un detector aprende
    a reconocer primero por ser la más llamativa. Si la caja se desvía hacia
    arriba, un margen inferior generoso sigue capturando el texto, mientras
    que uno simétrico lo dejaría fuera y el OCR no leería nada.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box

    box_height = y2 - y1
    pad_x = (x2 - x1) * padding
    pad_y = box_height * padding
    pad_y_bottom = box_height * (padding if padding_bottom is None else padding_bottom)

    xa = int(max(0, round(x1 - pad_x)))
    ya = int(max(0, round(y1 - pad_y)))
    xb = int(min(width, round(x2 + pad_x)))
    yb = int(min(height, round(y2 + pad_y_bottom)))

    if xb <= xa or yb <= ya:
        return image[0:0, 0:0]
    return image[ya:yb, xa:xb]
