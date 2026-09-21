"""Reconocedor especializado del número de serie.

Se ocupa sólo del serial, no del resto del texto del cilindro. La razón es que
el serial es el identificador del activo: leerlo mal no es un campo
incompleto, es un movimiento registrado en el cilindro equivocado. Merece un
modelo entrenado específicamente para él.

Ventaja sobre un OCR genérico en este dominio: el alfabeto está cerrado en 21
caracteres, el estilo es siempre troquelado sobre metal y la salida es una
única línea. Un modelo pequeño entrenado en esas condiciones no tiene que
resolver el problema general de leer cualquier texto.

Corre sobre ONNX Runtime, igual que el detector, así que el contenedor de
inferencia no necesita PyTorch. El modelo ocupa unos pocos megabytes.

Si no hay modelo entrenado el componente queda inactivo y el pipeline sigue
funcionando con el OCR general: es una mejora, no un requisito.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger
from app.ocr.charset import ALPHABET, decode

logger = get_logger(__name__)

IMG_HEIGHT = 32
IMG_WIDTH = 192


def preprocess(image: np.ndarray) -> np.ndarray:
    """Normaliza un recorte igual que en el entrenamiento.

    Cualquier diferencia con el preprocesado de entrenamiento degrada el
    reconocimiento sin dar ninguna señal de error, así que esta función y la
    de `training/train_recognizer.py` deben mantenerse idénticas.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    scale = IMG_HEIGHT / max(height, 1)
    new_width = max(8, min(IMG_WIDTH, int(round(width * scale))))
    resized = cv2.resize(image, (new_width, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMG_HEIGHT, IMG_WIDTH), int(resized.mean()), dtype=np.uint8)
    canvas[:, :new_width] = resized

    array = canvas.astype(np.float32)
    array = (array - array.mean()) / (array.std() + 1e-6)
    return array


class SerialRecognizer:
    name = "crnn"

    def __init__(self, model_path: Path | None, *, threads: int = 0) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._threads = threads
        self._session: Any = None
        self._input_name: str | None = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure_loaded(self) -> None:
        if self._session is not None or self._failed:
            return
        with self._lock:
            if self._session is not None or self._failed:
                return
            if not self._model_path or not self._model_path.exists():
                self._failed = True
                return
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                if self._threads > 0:
                    options.intra_op_num_threads = self._threads
                    options.inter_op_num_threads = self._threads
                self._session = ort.InferenceSession(
                    str(self._model_path), options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception:
                logger.exception("no se pudo cargar el reconocedor de seriales")
                self._failed = True
                return

            # El alfabeto viaja junto a los pesos: si no coincide con el del
            # servicio, los índices que emite la red apuntan a otros
            # caracteres y el resultado sería basura silenciosa.
            alphabet_file = self._model_path.parent / "alphabet.txt"
            if alphabet_file.exists():
                stored = alphabet_file.read_text(encoding="utf-8").strip()
                if stored != ALPHABET:
                    logger.error(
                        "el alfabeto del modelo no coincide con el del servicio; "
                        "se desactiva el reconocedor",
                        extra={"modelo": stored, "servicio": ALPHABET},
                    )
                    self._session = None
                    self._failed = True
                    return

            logger.info("reconocedor de seriales cargado",
                        extra={"path": str(self._model_path)})

    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._session is not None

    def describe(self) -> str:
        if not self._model_path:
            return "crnn (sin configurar)"
        if self._failed:
            return f"crnn (no disponible: {self._model_path.name})"
        return f"crnn:{self._model_path.name}"

    def read(self, image: np.ndarray) -> tuple[str, float] | None:
        """Devuelve (texto, confianza) o None si no hay modelo o entrada válida.

        La confianza es la probabilidad media de los símbolos emitidos, que es
        una medida honesta de lo segura que está la red carácter a carácter.
        """
        self._ensure_loaded()
        if self._session is None or image.size == 0:
            return None

        blob = preprocess(image)[None, None, :, :].astype(np.float32)
        try:
            logits = self._session.run(None, {self._input_name: blob})[0][0]
        except Exception:
            logger.exception("fallo al reconocer el serial")
            return None

        # Softmax estable sobre el eje de clases.
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        indices = probabilities.argmax(axis=1)
        text = decode(indices.tolist())
        if not text:
            return None

        emitted = probabilities.max(axis=1)[indices != 0]
        confidence = float(emitted.mean()) if emitted.size else 0.0
        return text, round(confidence, 4)
