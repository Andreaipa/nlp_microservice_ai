"""Logging estructurado.

En producción emite JSON por línea para que sea consumible por cualquier
agregador. En local usa texto plano legible.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from app.core.context import get_request_id

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} " \
               f"[{get_request_id()}] {record.name}: {record.getMessage()}"
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Silenciar ruido de terceros.
    for noisy in ("httpx", "httpcore", "PIL", "urllib3", "ppocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class SafeExtraLogger(logging.LoggerAdapter):
    """Adaptador que evita colisiones al pasar `extra`.

    `logging` reserva ciertos nombres en LogRecord ('filename', 'module',
    'args'...) y lanza KeyError si un `extra` intenta reutilizarlos. Es un
    error fácil de cometer y que sólo se manifiesta en la ruta que lo comete,
    así que en vez de confiar en recordarlo en cada llamada, las claves en
    conflicto se renombran con el prefijo 'ctx_'.
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"ctx_{key}" if key in _RESERVED else key): value
                for key, value in extra.items()
            }
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Devuelve un logger con `extra` a prueba de colisiones.

    El tipo devuelto es un LoggerAdapter, no un Logger: expone la misma
    interfaz (`info`, `warning`, `exception`...) y añade el saneado de las
    claves reservadas.
    """
    return SafeExtraLogger(logging.getLogger(name), {})
