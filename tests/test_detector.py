"""Unit tests for detector preprocessing and class-aware NMS."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mtsd_detector.detector import OnnxTrafficSignDetector


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _outputs: object, _feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.output]


def _image_bytes() -> bytes:
    image = Image.new("RGB", (64, 32), (120, 80, 40))
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return handle.getvalue()


def test_detector_decodes_boxes_and_suppresses_same_class() -> None:
    output = np.zeros((1, 25, 2), dtype=np.float32)
    output[0, 0:4, 0] = [32.0, 32.0, 20.0, 20.0]
    output[0, 4, 0] = 0.9
    output[0, 0:4, 1] = [32.0, 32.0, 18.0, 18.0]
    output[0, 4, 1] = 0.8
    detector = OnnxTrafficSignDetector(FakeSession(output), input_size=64)

    detections, image_size, inference_ms = detector.detect(
        _image_bytes(),
        confidence=0.5,
        iou_threshold=0.5,
    )

    assert image_size == (64, 32)
    assert len(detections) == 1
    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "regulatory--stop"
    assert detections[0]["score"] == 0.9
    assert detections[0]["box"][0] >= 0
    assert inference_ms >= 0
