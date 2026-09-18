"""GET /api/v1/health — end-to-end dependency health."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mtsd_detector.schemas import HealthResponse
from mtsd_detector.services.health import build_health_report

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def health(request: Request) -> HealthResponse | JSONResponse:
    pool = getattr(request.app.state, "db_pool", None)
    report = await build_health_report(pool)
    if report.status == "error" or (
        report.status == "degraded"
        and any(c.name == "postgres" and not c.ok for c in report.components)
    ):
        return JSONResponse(
            status_code=503,
            content=report.model_dump(),
        )
    return report
