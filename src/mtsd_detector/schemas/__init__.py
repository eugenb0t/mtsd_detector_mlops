"""Pydantic response DTOs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "error"]
LivenessStatus = Literal["ok"]


class HealthzResponse(BaseModel):
    status: LivenessStatus = "ok"


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    name: str
    version: str
    latency_ms: float = Field(ge=0)
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    components: list[ComponentHealth]
