"""Ensamblado del router de la versión 1 de la API."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import analyze, system

api_router = APIRouter()
api_router.include_router(analyze.router)
api_router.include_router(system.router)
