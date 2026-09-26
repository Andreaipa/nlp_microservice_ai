"""Sincronización del catálogo con la tabla `cilindros` de Supabase.

Por qué existe: el catálogo es lo que convierte una lectura OCR imperfecta en
una identificación fiable, porque el serial leído se resuelve contra los
cilindros conocidos. Si ese catálogo fuese sólo el fichero estático generado a
partir de las fichas del dataset, un cilindro que el operario registre hoy en
la aplicación móvil no se reconocería nunca en una fotografía.

La fuente de verdad del inventario es la base de datos de la aplicación. Este
módulo la lee periódicamente y la combina con el catálogo estático:

* un serial que está en Supabase usa los datos de Supabase, que son los que
  mantiene el operario;
* un serial que sólo está en el fichero estático se conserva, para no perder
  los cilindros del dataset que aún no se han registrado en la aplicación.

Si Supabase no responde, el servicio sigue funcionando con el último catálogo
bueno. Nunca se vacía el catálogo por un fallo de red: una identificación con
datos de hace un minuto es mucho mejor que ninguna.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.parsing.records import normalize_record
from app.parsing.serial import normalize_serial_charset
from app.services.catalog import CylinderCatalog, entries_from_records

logger = get_logger(__name__)

# Columnas que se leen de la tabla. Se piden explícitamente en lugar de `*`
# para no traer datos que el servicio no necesita, como el cliente propietario.
SUPABASE_COLUMNS = (
    "id,numero_serie,marca,anio_fabricacion,tipo_gas,unidad_medida,peso,"
    "ancho_diametro,largo_altura,litros,contenido,color,ph,procedencia"
)

# PostgREST limita las filas por respuesta; se pagina para no truncar el
# inventario en silencio si algún día supera ese límite.
PAGE_SIZE = 1000


def supabase_row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convierte una fila de `cilindros` al formato del catálogo.

    La aplicación guarda los campos tal como los escribe el operario ("DIÓXIDO
    DE CARBONO", "CADA 10 AÑOS PH"...). Se pasan por los mismos normalizadores
    que las fichas del dataset, para que el catálogo sea uniforme venga de
    donde venga cada cilindro.
    """
    raw = {
        "numero_serie": row.get("numero_serie"),
        "marca": row.get("marca"),
        "fecha_fabricacion": row.get("anio_fabricacion"),
        "tipo_gas": row.get("tipo_gas"),
        "unidad_medida": row.get("unidad_medida"),
        "peso": row.get("peso"),
        # El acetileno se registra por contenido en lugar de por litros.
        "litros": row.get("litros") if row.get("litros") is not None else row.get("contenido"),
        "ancho_diametro": row.get("ancho_diametro"),
        "largo_altura": row.get("largo_altura"),
        "color": row.get("color"),
        "ph": row.get("ph"),
        "procedencia": row.get("procedencia"),
    }
    text = {key: str(value) for key, value in raw.items() if value not in (None, "")}

    label = f"cilindros#{row.get('id', '?')}"
    record, _issues = normalize_record(label, text)
    if not record:
        return None
    record["cilindro"] = label
    return record


@dataclass
class SyncStatus:
    """Estado de la última sincronización, para /ready y /version."""

    enabled: bool = False
    static_count: int = 0
    supabase_count: int = 0
    total: int = 0
    last_success_at: str | None = None
    last_attempt_at: str | None = None
    last_error: str | None = None
    duration_ms: int | None = None

    def describe(self) -> str:
        if not self.enabled:
            return f"{self.total} cilindros (catálogo estático)"
        if self.last_success_at is None:
            reason = self.last_error or "aún no se ha sincronizado"
            return f"{self.total} cilindros (catálogo estático; Supabase: {reason})"
        detail = (
            f"{self.total} cilindros ({self.supabase_count} de Supabase, "
            f"{self.total - self.supabase_count} sólo del catálogo estático)"
        )
        if self.last_error:
            detail += f"; último intento fallido: {self.last_error}"
        return detail


class CatalogSync:
    """Mantiene el catálogo del servicio alineado con Supabase."""

    def __init__(
        self,
        settings: Settings,
        catalog: CylinderCatalog,
        static_records: list[dict[str, Any]],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._catalog = catalog
        self._static_records = static_records
        self._client: httpx.AsyncClient | None = None
        self.status = SyncStatus(
            enabled=self.enabled,
            static_count=len(static_records),
            total=len(catalog),
        )

    @property
    def enabled(self) -> bool:
        return bool(self._settings.supabase_url and self._settings.supabase_key)

    async def startup(self) -> None:
        if not self.enabled:
            logger.info("sincronización con Supabase desactivada: catálogo estático")
            return
        key = str(self._settings.supabase_key)
        self._client = httpx.AsyncClient(
            base_url=str(self._settings.supabase_url).rstrip("/") + "/rest/v1",
            timeout=self._settings.supabase_timeout_seconds,
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
            transport=self._transport,
        )
        # La primera sincronización se espera para que las primeras peticiones
        # ya vean el inventario real. Si falla, se arranca con el estático.
        await self.refresh()

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_rows(self) -> list[dict[str, Any]]:
        """Lee toda la tabla, página a página."""
        assert self._client is not None
        table = self._settings.supabase_catalog_table
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = await self._client.get(
                f"/{table}",
                params={"select": SUPABASE_COLUMNS, "order": "id.asc"},
                headers={"Range": f"{offset}-{offset + PAGE_SIZE - 1}"},
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("Supabase devolvió una respuesta inesperada")
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                return rows
            offset += PAGE_SIZE

    def merge(self, supabase_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Combina Supabase con el catálogo estático; Supabase prevalece."""
        live_serials = {
            normalize_serial_charset(str(r["numero_serie"])) for r in supabase_records
        }
        kept_static = [
            r for r in self._static_records
            if normalize_serial_charset(str(r.get("numero_serie", ""))) not in live_serials
        ]
        return supabase_records + kept_static

    async def refresh(self) -> bool:
        """Sincroniza una vez. Devuelve True si se actualizó el catálogo."""
        if not self.enabled or self._client is None:
            return False

        started = time.perf_counter()
        self.status.last_attempt_at = _now()
        try:
            rows = await self.fetch_rows()
        except (httpx.HTTPError, ValueError) as exc:
            reason = _describe_error(exc)
            self.status.last_error = reason
            logger.warning("no se pudo sincronizar el catálogo con Supabase",
                           extra={"motivo": reason})
            return False

        records = [r for r in (supabase_row_to_record(row) for row in rows) if r]
        entries = entries_from_records(self.merge(records))
        self._catalog.replace_entries(entries)

        self.status.supabase_count = len({r["numero_serie"] for r in records})
        self.status.total = len(self._catalog)
        self.status.last_success_at = _now()
        self.status.last_error = None
        self.status.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "catálogo sincronizado con Supabase",
            extra={"supabase": self.status.supabase_count,
                   "total": self.status.total,
                   "ms": self.status.duration_ms},
        )
        return True

    async def run_forever(self) -> None:
        """Bucle de refresco en segundo plano. Nunca termina por un error."""
        interval = max(5, self._settings.catalog_refresh_seconds)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.refresh()
            except Exception:  # pragma: no cover - defensa ante lo imprevisto
                logger.exception("error inesperado sincronizando el catálogo")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "tiempo de espera agotado"
    if isinstance(exc, httpx.HTTPError):
        return f"error de red ({exc.__class__.__name__})"
    return str(exc)
