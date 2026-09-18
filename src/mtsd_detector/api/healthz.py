"""Liveness probe — intentionally outside /api/v1."""

from __future__ import annotations

from fastapi import APIRouter

from mtsd_detector.schemas import HealthzResponse

router = APIRouter(tags=["healthz"])


@router.get("/healthz", response_model=HealthzResponse)
def healthz() -> HealthzResponse:
    """Fast liveness check for Docker/K8s — not versioned under /api/v1."""
    return HealthzResponse(status="ok")
