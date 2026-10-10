"""MLflow models-from-code entry point for the ONNX detector."""

import mlflow

from mtsd_detector.detector import DetectorPythonModel

mlflow.models.set_model(DetectorPythonModel())
