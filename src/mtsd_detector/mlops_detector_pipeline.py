"""Evaluate and register a real MTSD20 ONNX object detector.

The heavy trained checkpoints remain on the external MTSD storage.  This
pipeline evaluates them against the official labeled MTSD ``val.jsonl``
manifest, logs metrics/diagnostics to MLflow, and packages the selected ONNX
detector as a registry-backed pyfunc model for the FastAPI service.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtsd_detector.detector import (
    MTSD20_CLASS_NAMES,
    OnnxTrafficSignDetector,
)

EXPERIMENT_NAME = "mlops-mtsd-detector"
MODEL_NAME = "mtsd20-detector"
MODEL_ALIAS = "champion"
DATASET_NAME = "mtsd20-detector-val-v1"
DEFAULT_MAX_IMAGES = 256
MIN_MAP50_PRESENT = 0.40
# The Compose app is deliberately limited to one CPU; leave a small scheduling
# margin above the observed ~0.5 s quality-checkpoint latency.
MAX_P99_INFERENCE_MS = 600.0


def _discover(explicit: str | None, candidates: list[Path], label: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() or path.is_dir():
            return path
        raise FileNotFoundError(f"{label} does not exist: {path}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"could not discover {label}: {candidates}")


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path, max_images: int) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_images <= 0 or max_images >= len(records):
        return records
    # Evenly sample the official holdout instead of evaluating only its first
    # (heavily other-sign-biased) records.
    indices = np.linspace(0, len(records) - 1, max_images, dtype=int)
    return [records[int(index)] for index in indices]


def _image_path(images_root: Path, record: dict[str, Any]) -> Path:
    path = images_root / str(record["image"])
    if path.is_file():
        return path
    return images_root / Path(str(record["image"])).name


def _xyxy_from_norm(box: list[float], width: int, height: int) -> tuple[float, ...]:
    center_x, center_y, box_width, box_height = (float(value) for value in box)
    return (
        (center_x - box_width / 2.0) * width,
        (center_y - box_height / 2.0) * height,
        (center_x + box_width / 2.0) * width,
        (center_y + box_height / 2.0) * height,
    )


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(
    predictions: list[dict[str, Any]],
    truth: dict[int, list[tuple[float, ...]]],
) -> tuple[float, int, int, int]:
    positives = sum(len(items) for items in truth.values())
    matched = {image_id: [False] * len(items) for image_id, items in truth.items()}
    true_positive: list[int] = []
    false_positive: list[int] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        image_id = int(prediction["image_id"])
        candidates = truth.get(image_id, [])
        overlaps = [_iou(prediction["box"], target) for target in candidates]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] >= 0.5 and not matched[image_id][best]:
            matched[image_id][best] = True
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    if not true_positive:
        return 0.0, 0, 0, positives
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recalls = cumulative_tp / max(1, positives)
    precisions = cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp)
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]
    previous = 0.0
    average_precision = 0.0
    for recall, precision in zip(recalls, envelope, strict=True):
        average_precision += float(recall - previous) * float(precision)
        previous = float(recall)
    return (
        float(average_precision),
        int(cumulative_tp[-1]),
        int(cumulative_fp[-1]),
        positives,
    )


def evaluate_detector(
    detector: OnnxTrafficSignDetector,
    records: list[dict[str, Any]],
    images_root: Path,
    confidence: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    missing_images = 0
    for image_id, record in enumerate(records):
        path = _image_path(images_root, record)
        if not path.is_file():
            missing_images += 1
            continue
        image_bytes = path.read_bytes()
        detections, _, inference_ms = detector.detect(
            image_bytes,
            confidence=confidence,
            iou_threshold=iou_threshold,
        )
        latencies.append(inference_ms)
        width, height = int(record["width"]), int(record["height"])
        for detection in detections:
            box = detection["box"]
            predictions.append(
                {
                    "image_id": image_id,
                    "class_id": int(detection["class_id"]),
                    "score": float(detection["score"]),
                    "box": tuple(float(value) for value in box),
                    "class_name": detection["class_name"],
                    "width": width,
                    "height": height,
                },
            )

    per_class: list[dict[str, Any]] = []
    present_scores: list[float] = []
    total_tp = total_fp = total_gt = 0
    for class_id, class_name in enumerate(MTSD20_CLASS_NAMES):
        truth = {
            image_id: [
                _xyxy_from_norm(
                    item.get("bbox_letterbox", item["bbox"]),
                    detector.input_size
                    if item.get("bbox_letterbox") is not None
                    else int(records[image_id]["width"]),
                    detector.input_size
                    if item.get("bbox_letterbox") is not None
                    else int(records[image_id]["height"]),
                )
                for item in record.get("objects", [])
                if int(item["class_id"]) == class_id
            ]
            for image_id, record in enumerate(records)
        }
        class_predictions = [
            item for item in predictions if item["class_id"] == class_id
        ]
        ap, true_positive, false_positive, ground_truth = _average_precision(
            class_predictions,
            truth,
        )
        if ground_truth:
            present_scores.append(ap)
        total_tp += true_positive
        total_fp += false_positive
        total_gt += ground_truth
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ap50": ap,
                "ground_truth": ground_truth,
                "predictions": len(class_predictions),
            },
        )
    latencies.sort()
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p99_index = min(len(latencies) - 1, int(round(0.99 * (len(latencies) - 1))))
    p99 = float(latencies[p99_index]) if latencies else 0.0
    gate_passed = (
        float(np.mean(present_scores)) if present_scores else 0.0
    ) >= MIN_MAP50_PRESENT and p99 <= MAX_P99_INFERENCE_MS
    return {
        "images_evaluated": len(records) - missing_images,
        "images_requested": len(records),
        "missing_images": missing_images,
        "predictions": len(predictions),
        "mAP50": float(np.mean([item["ap50"] for item in per_class])),
        "mAP50_present_classes": (
            float(np.mean(present_scores)) if present_scores else 0.0
        ),
        "precision_at_iou50": total_tp / max(1, total_tp + total_fp),
        "recall_at_iou50": total_tp / max(1, total_gt),
        "mean_inference_ms": mean_latency,
        "p99_inference_ms": p99,
        "fps": 1000.0 / mean_latency if mean_latency else 0.0,
        "per_class": per_class,
        "class_names": MTSD20_CLASS_NAMES,
        "iou_threshold": iou_threshold,
        "confidence": confidence,
        "quality_gate": {
            "passed": gate_passed,
            "min_mAP50_present_classes": MIN_MAP50_PRESENT,
            "max_p99_inference_ms": MAX_P99_INFERENCE_MS,
        },
    }


def log_detector_eda(bundle: dict[str, Any]) -> str:
    """Log bbox and class-distribution EDA for the exact evaluation manifest."""
    class_counts = Counter(
        int(item["class_id"])
        for record in bundle["records"]
        for item in record.get("objects", [])
    )
    min_sides = []
    for record in bundle["records"]:
        for item in record.get("objects", []):
            box = item.get("bbox_letterbox", item["bbox"])
            min_sides.append(min(float(box[2]), float(box[3])) * 640.0)
    summary = {
        "dataset": DATASET_NAME,
        "source": str(bundle["prepared_root"]),
        "digest": bundle["digest"],
        "images": len(bundle["records"]),
        "images_with_objects": sum(
            bool(record.get("objects")) for record in bundle["records"]
        ),
        "objects": len(min_sides),
        "ignore_regions": sum(
            len(record.get("ignores", [])) for record in bundle["records"]
        ),
        "class_counts": {
            MTSD20_CLASS_NAMES[class_id]: count
            for class_id, count in sorted(class_counts.items())
        },
        "bbox_min_side_px": {
            "min": min(min_sides) if min_sides else 0.0,
            "median": float(np.median(min_sides)) if min_sides else 0.0,
            "max": max(min_sides) if min_sides else 0.0,
        },
    }
    with mlflow.start_run(run_name="detector-eda") as run:
        mlflow.log_input(bundle["dataset"], context="eda")
        mlflow.set_tags(
            {
                "run.type": "eda",
                "dataset.source": str(bundle["prepared_root"]),
                "dataset.digest": bundle["digest"],
                "dataset.images_source": str(bundle["images_root"]),
            },
        )
        mlflow.log_dict(summary, "eda/summary.json")
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.bar(
            range(len(MTSD20_CLASS_NAMES)),
            [class_counts.get(index, 0) for index in range(len(MTSD20_CLASS_NAMES))],
        )
        axis.set_title("MTSD20 detector class distribution")
        axis.set_xlabel("class")
        axis.set_ylabel("objects")
        axis.set_xticks(range(len(MTSD20_CLASS_NAMES)), range(len(MTSD20_CLASS_NAMES)))
        fig.tight_layout()
        mlflow.log_figure(fig, "eda/class_distribution.png")
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.hist(min_sides, bins=20, color="#2563eb")
        axis.set_title("MTSD20 bbox minimum side")
        axis.set_xlabel("minimum side in letterbox pixels")
        axis.set_ylabel("objects")
        fig.tight_layout()
        mlflow.log_figure(fig, "eda/bbox_min_side_histogram.png")
        plt.close(fig)
        return run.info.run_id


def _version_for_run(client: MlflowClient, run_id: str) -> str:
    for _ in range(30):
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        matches = [version for version in versions if version.run_id == run_id]
        if matches:
            return str(max(matches, key=lambda item: int(item.version)).version)
        time.sleep(1)
    raise RuntimeError(f"registered detector version not found for run {run_id}")


def run_candidate(
    client: MlflowClient,
    bundle: dict[str, Any],
    name: str,
    onnx_path: Path,
    confidence: float,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(bundle["tracking_uri"])
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_input(bundle["dataset"], context="evaluation")
        mlflow.set_tags(
            {
                "model.kind": "object-detection",
                "model.format": "onnx",
                "dataset.name": DATASET_NAME,
                "dataset.source": str(bundle["prepared_root"]),
                "dataset.digest": bundle["digest"],
                "dataset.images_source": str(bundle["images_root"]),
                "lineage.contract": (
                    "MTSD20 bbox manifest -> ONNX detector -> POST /process"
                ),
                "evaluation.split": "official labeled val holdout sample",
            },
        )
        mlflow.log_params(
            {
                "checkpoint": str(onnx_path),
                "input_size": 640,
                "confidence": confidence,
                "iou_threshold": 0.5,
                "evaluation_images": len(bundle["records"]),
            },
        )
        detector = OnnxTrafficSignDetector.from_path(str(onnx_path))
        report = evaluate_detector(
            detector,
            bundle["records"],
            bundle["images_root"],
            confidence,
        )
        mlflow.log_metrics(
            {
                "mAP50": report["mAP50"],
                "mAP50_present_classes": report["mAP50_present_classes"],
                "precision_at_iou50": report["precision_at_iou50"],
                "recall_at_iou50": report["recall_at_iou50"],
                "mean_inference_ms": report["mean_inference_ms"],
                "p99_inference_ms": report["p99_inference_ms"],
                "fps": report["fps"],
            },
        )
        mlflow.set_tag("primary_metric", "mAP50_present_classes")
        mlflow.set_tag("quality_gate_passed", str(report["quality_gate"]["passed"]))
        mlflow.log_text(json.dumps(report, indent=2), "evaluation/report.json")
        example_path = _image_path(bundle["images_root"], bundle["records"][0])
        input_example = pd.DataFrame(
            [
                {
                    "image_base64": base64.b64encode(example_path.read_bytes()).decode(
                        "ascii"
                    ),
                    "confidence": confidence,
                    "iou_threshold": 0.5,
                    "max_detections": 50,
                },
            ],
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model="src/mtsd_detector/mlflow_detector_model.py",
            artifacts={"onnx": str(onnx_path)},
            code_paths=["src"],
            input_example=input_example,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "mlflow==2.22.0",
                "numpy>=1.26.0",
                "onnxruntime>=1.20.0",
                "pandas>=2.2.0",
                "pillow>=10.0.0",
            ],
        )
        version = _version_for_run(client, run.info.run_id)
        report.update({"run_id": run.info.run_id, "version": version, "name": name})
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--b1-onnx", default=None)
    parser.add_argument("--quality-onnx", default=None)
    args = parser.parse_args()

    prepared_root = _discover(
        args.prepared_root,
        [
            Path("/media/eugene/data21/mtsd/prepared_mtsd20"),
            Path("/media/eugene/data2/mtsd/prepared_mtsd20"),
        ],
        "MTSD prepared root",
    )
    images_root = _discover(
        args.images_root,
        [
            prepared_root.parent / "runs/parallel_b0_b1/kaggle_stage/dataset/images",
            prepared_root.parent
            / "runs/quality_ladder/packs/mtsd20_letterbox_640_v2/images",
        ],
        "MTSD image root",
    )
    val_path = prepared_root / "val.jsonl"
    records = _read_records(val_path, args.max_images)
    digest = _sha256((prepared_root / "dataset.json", val_path))
    dataset_frame = pd.DataFrame(
        {
            "image": [str(record["image"]) for record in records],
            "split": ["val"] * len(records),
            "objects": [len(record.get("objects", [])) for record in records],
        },
    )
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    dataset = mlflow.data.from_pandas(
        dataset_frame,
        source=str(prepared_root),
        name=DATASET_NAME,
        targets="objects",
    )
    bundle = {
        "tracking_uri": args.tracking_uri,
        "prepared_root": prepared_root,
        "images_root": images_root,
        "records": records,
        "dataset": dataset,
        "digest": digest,
    }
    log_detector_eda(bundle)
    candidates = [
        (
            "detector-portable-b1",
            _discover(
                args.b1_onnx,
                [
                    prepared_root.parent / "runs/parallel_b0_b1/artifacts/b1_best.onnx",
                ],
                "portable detector ONNX",
            ),
        ),
        (
            "detector-quality-pc",
            _discover(
                args.quality_onnx,
                [
                    prepared_root.parent
                    / "runs/quality_ladder/autoloop_map075/viewer/pc_s640_best.onnx",
                    prepared_root.parent
                    / "runs/wave3_ops_eval/weights_ds/pc_s640_best.onnx",
                ],
                "quality detector ONNX",
            ),
        ),
    ]
    client = MlflowClient(args.tracking_uri)
    results = [
        run_candidate(client, bundle, name, path, args.confidence)
        for name, path in candidates
    ]
    winner = max(results, key=lambda item: item["mAP50_present_classes"])
    if not winner["quality_gate"]["passed"]:
        raise RuntimeError(f"no detector candidate passed quality gate: {winner}")
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, winner["version"])
    client.set_model_version_tag(MODEL_NAME, winner["version"], "role", MODEL_ALIAS)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "winner": winner,
        "runs": results,
        "dataset": {
            "name": DATASET_NAME,
            "source": str(prepared_root),
            "digest": digest,
            "images_evaluated": len(records),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
