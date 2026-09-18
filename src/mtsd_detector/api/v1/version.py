"""GET /api/v1/version."""

from __future__ import annotations

from fastapi import APIRouter

from mtsd_detector.schemas import VersionResponse
from mtsd_detector.services.version import get_app_version

router = APIRouter(tags=["version"])


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    return VersionResponse(version=get_app_version())
