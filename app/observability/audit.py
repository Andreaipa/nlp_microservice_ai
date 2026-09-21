"""Registro de auditoría de inferencias.

Permite reconstruir por qué el sistema dijo lo que dijo: qué detectó, qué leyó
el OCR, qué serial resolvió y con qué versión de modelo. Es lo que hace
depurable un fallo reportado desde la planta.

Criterio de privacidad y volumen: se guarda siempre el registro estructurado,
que es pequeño; la imagen sólo cuando el caso requirió confirmación manual,
que es cuando hace falta mirarla. Las imágenes correctas no aportan nada y
ocuparían disco sin motivo.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditLog:
    def __init__(self, directory: Path, *, enabled: bool = True,
                 store_images: bool = True, retention_days: int = 30) -> None:
        self._dir = Path(directory)
        self._enabled = enabled
        self._store_images = store_images
        self._retention_days = retention_days
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        request_id: str,
        payload: dict[str, Any],
        image: np.ndarray | None = None,
        needs_review: bool = False,
    ) -> None:
        if not self._enabled:
            return

        timestamp = datetime.now(UTC)
        day_dir = self._dir / timestamp.strftime("%Y-%m-%d")
        try:
            day_dir.mkdir(parents=True, exist_ok=True)
            entry = {"timestamp": timestamp.isoformat(), "request_id": request_id, **payload}

            with (day_dir / "inferences.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

            if image is not None and needs_review and self._store_images:
                cv2.imwrite(str(day_dir / f"{request_id}.jpg"), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except OSError:
            # La auditoría nunca debe tumbar una inferencia.
            logger.exception("no se pudo escribir el registro de auditoría")

    def purge_expired(self) -> int:
        """Elimina carpetas anteriores al periodo de retención."""
        if not self._enabled or not self._dir.exists():
            return 0

        cutoff = datetime.now(UTC).date() - timedelta(days=self._retention_days)
        removed = 0
        for child in self._dir.iterdir():
            if not child.is_dir():
                continue
            try:
                day = datetime.strptime(child.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                for item in child.iterdir():
                    item.unlink(missing_ok=True)
                child.rmdir()
                removed += 1
        if removed:
            logger.info("auditoría purgada", extra={"carpetas_eliminadas": removed})
        return removed
