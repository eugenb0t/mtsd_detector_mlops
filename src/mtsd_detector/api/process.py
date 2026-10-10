"""Inference endpoint backed by the model loaded during application startup."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from mtsd_detector.schemas import ProcessRequest, ProcessResponse
from mtsd_detector.services.model import LoadedModel

router = APIRouter(tags=["inference"])


@router.post("/process", response_model=ProcessResponse)
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    """Predict with the already loaded registry model; no MLflow call is made here."""

    loaded: LoadedModel | None = getattr(request.app.state, "model", None)
    if loaded is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "MLflow model is not loaded; configure the registry alias and restart"
            ),
        )
    prediction, probabilities = loaded.predict(payload.features)
    return ProcessResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_uri=loaded.model_uri,
        model_version=loaded.model_version,
        loaded_at=loaded.loaded_at,
    )
