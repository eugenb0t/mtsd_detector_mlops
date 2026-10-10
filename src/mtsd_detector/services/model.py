"""One-time MLflow model loading and in-process inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Immutable runtime handle shared by all requests."""

    model: Any
    model_uri: str
    model_version: str | None
    loaded_at: str

    def predict(self, features: list[float]) -> tuple[int, list[float] | None]:
        frame = pd.DataFrame([features], dtype="float32")
        prediction = self.model.predict(frame)
        label = int(prediction[0])
        probabilities: list[float] | None = None
        if hasattr(self.model, "predict_proba"):
            probabilities = [
                float(value) for value in self.model.predict_proba(frame)[0]
            ]
        return label, probabilities


def load_registry_model(
    tracking_uri: str | None,
    model_name: str,
    model_alias: str,
) -> LoadedModel | None:
    """Load the registry alias exactly once during application lifespan.

    ``None`` is intentional for local API tests and for starting the service before
    the first training run. Once configured, a missing alias is surfaced as a
    startup error to make deployment misconfiguration visible.
    """

    if not tracking_uri:
        logger.info("MLflow model loading disabled: no tracking URI configured")
        return None

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info("loading MLflow model", extra={"model_uri": model_uri})
    model = mlflow.pyfunc.load_model(model_uri)
    version: str | None = None
    try:
        version = str(
            MlflowClient(tracking_uri=tracking_uri)
            .get_model_version_by_alias(model_name, model_alias)
            .version,
        )
    except Exception:
        logger.warning("model version lookup failed", exc_info=True)

    loaded = LoadedModel(
        model=model,
        model_uri=model_uri,
        model_version=version,
        loaded_at=datetime.now(UTC).isoformat(),
    )
    logger.info(
        "MLflow model loaded",
        extra={"model_uri": model_uri, "model_version": version},
    )
    return loaded
