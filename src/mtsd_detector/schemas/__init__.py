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


class ProcessRequest(BaseModel):
    """An RGB image encoded as base64 for full object detection."""

    image_base64: str = Field(min_length=32)
    confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=50, ge=1, le=200)


class Detection(BaseModel):
    class_id: int = Field(ge=0)
    class_name: str
    score: float = Field(ge=0.0, le=1.0)
    box: tuple[float, float, float, float]


class ProcessResponse(BaseModel):
    detections: list[Detection]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    inference_ms: float = Field(ge=0)
    model_uri: str
    model_version: str | None = None
    loaded_at: str
