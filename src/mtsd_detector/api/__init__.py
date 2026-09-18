"""HTTP API routers."""

from __future__ import annotations

from fastapi import APIRouter

from mtsd_detector.api import healthz
from mtsd_detector.api.v1 import v1_router


def build_api_router() -> APIRouter:
    router = APIRouter()
    router.include_router(healthz.router)
    router.include_router(v1_router)
    return router
