"""Cliente del backend principal.

Frontera de responsabilidades: este microservicio informa de lo que observó en
la imagen. El backend es el dueño del dato maestro y de la lógica de negocio.
Aquí sólo se consulta información complementaria de un cilindro ya
identificado.

La consulta es best-effort: si el backend no responde, el análisis se devuelve
igualmente con los campos leídos por IA y `backend_match.error` explicando por
qué no hubo enriquecimiento. Una caída del backend no debe dejar ciega a la
aplicación móvil.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BackendClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._settings.backend_enabled and self._settings.backend_base_url)

    async def startup(self) -> None:
        if not self.enabled:
            logger.info("integración con backend desactivada")
            return
        headers = {"Accept": "application/json"}
        if self._settings.backend_api_key:
            headers["Authorization"] = f"Bearer {self._settings.backend_api_key}"
        self._client = httpx.AsyncClient(
            base_url=str(self._settings.backend_base_url),
            timeout=self._settings.backend_timeout_seconds,
            headers=headers,
        )
        logger.info("cliente de backend listo",
                    extra={"base_url": self._settings.backend_base_url})

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_cylinder(self, serial: str) -> tuple[dict[str, Any] | None, str | None]:
        """Consulta el maestro de un cilindro.

        Devuelve (datos, error). Un 404 no es un error: significa que el
        cilindro no está registrado, lo que es información útil para el
        operador.
        """
        if not self.enabled or self._client is None:
            return None, "integración con backend desactivada"

        try:
            response = await self._client.get(f"/cylinders/{serial}")
        except httpx.TimeoutException:
            logger.warning("timeout consultando el backend", extra={"serial": serial})
            return None, "timeout al consultar el backend principal"
        except httpx.HTTPError as exc:
            logger.warning("error de red contra el backend",
                           extra={"serial": serial, "error": str(exc)})
            return None, f"error de red: {exc.__class__.__name__}"

        if response.status_code == 404:
            return None, None
        if response.status_code >= 400:
            logger.warning("respuesta de error del backend",
                           extra={"serial": serial, "status": response.status_code})
            return None, f"el backend respondió {response.status_code}"

        try:
            payload = response.json()
        except ValueError:
            return None, "el backend devolvió una respuesta no JSON"

        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        return (payload if isinstance(payload, dict) else None), None

    async def health(self) -> bool:
        if not self.enabled or self._client is None:
            return False
        try:
            response = await self._client.get("/health")
            return response.status_code < 400
        except httpx.HTTPError:
            return False
