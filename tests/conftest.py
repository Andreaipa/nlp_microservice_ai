"""Fixtures compartidas de la suite de pruebas."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuración aislada: sin OCR real, sin backend y con auditoría temporal.

    `_env_file=None` es deliberado: sin él, las pruebas heredarían el .env de
    quien las ejecuta y pasarían o fallarían según su configuración local.
    """
    return Settings(
        _env_file=None,
        detector_backend="heuristic",
        ocr_backend="null",
        backend_enabled=False,
        catalog_path=ROOT / "datasets" / "catalog.json",
        audit_dir=tmp_path / "audit",
        audit_enabled=True,
        thresholds_calibrated=False,
        log_json=False,
    )


def render_stamped_plate(
    text: str = "JP5 19S206055",
    second_line: str = "FAB 09/19  CO2  45.2KG",
    *,
    width: int = 900,
    height: int = 320,
    contrast: int = 40,
    noise: int = 6,
) -> np.ndarray:
    """Genera una imagen que imita un collarín troquelado.

    No sustituye a fotografías reales: sirve para ejercitar el pipeline de
    extremo a extremo de forma determinista, sin depender de un dataset que
    todavía no está capturado. El texto se dibuja con muy poco contraste sobre
    fondo metálico, que es la dificultad característica del troquelado.
    """
    base = np.full((height, width), 128, dtype=np.uint8)
    # Veteado metálico.
    gradient = np.linspace(-18, 18, width, dtype=np.float32)
    base = np.clip(base + gradient[None, :], 0, 255).astype(np.uint8)

    layer = np.zeros_like(base, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(layer, text, (40, 130), font, 2.2, 255, 5, cv2.LINE_AA)
    if second_line:
        cv2.putText(layer, second_line, (40, 250), font, 1.5, 255, 4, cv2.LINE_AA)

    # Relieve: un borde claro y otro oscuro, como haría una luz rasante.
    shifted_up = np.roll(layer, -3, axis=0)
    shifted_down = np.roll(layer, 3, axis=0)
    embossed = base.astype(np.int16)
    embossed += (shifted_up.astype(np.int16) * contrast) // 255
    embossed -= (shifted_down.astype(np.int16) * contrast) // 255
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)

    if noise:
        embossed = np.clip(
            embossed.astype(np.int16)
            + np.random.default_rng(7).integers(-noise, noise + 1, embossed.shape),
            0, 255,
        ).astype(np.uint8)

    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def stamped_image() -> np.ndarray:
    return render_stamped_plate()


@pytest.fixture
def stamped_jpeg(stamped_image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", stamped_image)
    assert ok
    return buffer.tobytes()
