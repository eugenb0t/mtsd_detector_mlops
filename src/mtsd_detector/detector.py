"""Portable MTSD20 object detector preprocessing, decoding, and MLflow wrapper.

The model artifact is an externally trained YOLO-style ONNX detector.  This
module keeps the serving contract in this repository: RGB image bytes in,
absolute pixel ``xyxy`` boxes and MTSD20 class names out.  No training
framework is required by the API container.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from PIL import Image

MTSD20_CLASS_NAMES = [
    "regulatory--stop",
    "regulatory--yield",
    "regulatory--no-entry",
    "regulatory--no-parking",
    "regulatory--no-stopping",
    "regulatory--keep-right",
    "regulatory--keep-left",
    "regulatory--priority-road",
    "regulatory--roundabout",
    "regulatory--go-straight",
    "regulatory--turn-left",
    "regulatory--turn-right",
    "regulatory--no-left-turn",
    "regulatory--no-right-turn",
    "regulatory--no-u-turn",
    "regulatory--maximum-speed-limit",
    "information--pedestrians-crossing",
    "warning--pedestrians-crossing",
    "warning--roadworks",
    "warning--children",
    "other-sign",
]


@dataclass(frozen=True)
class Letterbox:
    """Geometry needed to map model-space boxes back to source pixels."""

    scale: float
    pad_x: float
    pad_y: float
    size: int
    source_width: int
    source_height: int


def _letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, Letterbox]:
    image = image.convert("RGB")
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_width) / 2.0
    pad_y = (size - new_height) / 2.0
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]
    return tensor.astype(np.float32, copy=False), Letterbox(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        size=size,
        source_width=source_width,
        source_height=source_height,
    )


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        xx0 = np.maximum(x0[current], x0[order[1:]])
        yy0 = np.maximum(y0[current], y0[order[1:]])
        xx1 = np.minimum(x1[current], x1[order[1:]])
        yy1 = np.minimum(y1[current], y1[order[1:]])
        intersection = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        union = areas[current] + areas[order[1:]] - intersection
        overlap = intersection / np.maximum(union, 1e-8)
        order = order[1:][overlap <= threshold]
    return keep


def _clip_box(box: np.ndarray, width: int, height: int) -> list[float]:
    return [
        max(0.0, min(float(box[0]), float(width))),
        max(0.0, min(float(box[1]), float(height))),
        max(0.0, min(float(box[2]), float(width))),
        max(0.0, min(float(box[3]), float(height))),
    ]


def _decode_yolo_output(
    output: np.ndarray,
    geometry: Letterbox,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Decode an Ultralytics export shaped ``[1, 4+C, N]``."""
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"unexpected detector output shape: {array.shape}")
    if array.shape[0] == 4 + len(MTSD20_CLASS_NAMES):
        array = array.T
    if array.shape[1] != 4 + len(MTSD20_CLASS_NAMES):
        raise ValueError(f"unexpected detector channels: {array.shape}")

    boxes = array[:, :4]
    class_scores = array[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    selected = scores >= confidence
    boxes, scores, class_ids = boxes[selected], scores[selected], class_ids[selected]
    if len(boxes) == 0:
        return []

    letterbox_boxes = np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2.0,
            boxes[:, 1] - boxes[:, 3] / 2.0,
            boxes[:, 0] + boxes[:, 2] / 2.0,
            boxes[:, 1] + boxes[:, 3] / 2.0,
        ],
        axis=1,
    )
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        local = _nms(
            letterbox_boxes[class_indices],
            scores[class_indices],
            iou_threshold,
        )
        kept.extend(int(class_indices[index]) for index in local)
    kept.sort(key=lambda index: float(scores[index]), reverse=True)

    detections: list[dict[str, Any]] = []
    for index in kept[:max_detections]:
        box = letterbox_boxes[index]
        source_box = np.array(
            [
                (box[0] - geometry.pad_x) / geometry.scale,
                (box[1] - geometry.pad_y) / geometry.scale,
                (box[2] - geometry.pad_x) / geometry.scale,
                (box[3] - geometry.pad_y) / geometry.scale,
            ],
            dtype=np.float32,
        )
        class_index = int(class_ids[index])
        detections.append(
            {
                "class_id": class_index,
                "class_name": MTSD20_CLASS_NAMES[class_index],
                "score": round(float(scores[index]), 6),
                "box": _clip_box(
                    source_box,
                    geometry.source_width,
                    geometry.source_height,
                ),
            },
        )
    return detections


class OnnxTrafficSignDetector:
    """CPU ONNX detector with explicit preprocessing and class-aware NMS."""

    def __init__(self, session: Any, input_size: int = 640) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.input_size = int(input_size)

    @classmethod
    def from_path(cls, path: str, input_size: int = 640) -> OnnxTrafficSignDetector:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # The Compose service is intentionally CPU-limited. Explicit single-thread
        # execution avoids oversubscribing a constrained container with ORT's
        # host-wide default thread pool.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return cls(session, input_size=input_size)

    def detect(
        self,
        image_bytes: bytes,
        confidence: float = 0.05,
        iou_threshold: float = 0.5,
        max_detections: int = 50,
    ) -> tuple[list[dict[str, Any]], tuple[int, int], float]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor, geometry = _letterbox(image, self.input_size)
            image_size = (geometry.source_width, geometry.source_height)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detections = _decode_yolo_output(
            outputs[0],
            geometry,
            confidence,
            iou_threshold,
            max_detections,
        )
        return detections, image_size, elapsed_ms


class DetectorPythonModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ONNX detector artifact."""

    def load_context(self, context: Any) -> None:
        self.detector = OnnxTrafficSignDetector.from_path(context.artifacts["onnx"])

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        del context, params
        rows: list[str] = []
        for row in model_input.to_dict(orient="records"):
            image_bytes = base64.b64decode(str(row["image_base64"]))
            detections, image_size, inference_ms = self.detector.detect(
                image_bytes,
                confidence=float(row.get("confidence", 0.05)),
                iou_threshold=float(row.get("iou_threshold", 0.5)),
                max_detections=int(row.get("max_detections", 50)),
            )
            rows.append(
                json.dumps(
                    {
                        "detections": detections,
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "inference_ms": round(inference_ms, 3),
                    },
                ),
            )
        return rows
