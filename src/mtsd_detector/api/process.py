"""Inference endpoint backed by the model loaded during application startup."""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, HTTPException, Request

from mtsd_detector.schemas import ProcessRequest, ProcessResponse
from mtsd_detector.services.model import LoadedModel

router = APIRouter(tags=["inference"])


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    """Detect signs with the already loaded registry model."""

    loaded: LoadedModel | None = getattr(request.app.state, "model", None)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "MLflow model is not loaded; configure the registry alias and restart"
            ),
        )
    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
        if not image_bytes:
            raise ValueError("empty image")
        result = await loaded.detect_async(
            image_bytes,
            payload.confidence,
            payload.iou_threshold,
            payload.max_detections,
        )
    except (binascii.Error, OSError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"invalid image: {error}",
        ) from error
    return ProcessResponse(
        detections=result["detections"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        inference_ms=result["inference_ms"],
        model_uri=loaded.model_uri,
        model_version=loaded.model_version,
        loaded_at=loaded.loaded_at,
    )
