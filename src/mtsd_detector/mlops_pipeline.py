"""End-to-end MLflow homework pipeline for the prepared MTSD20 dataset.

The pipeline intentionally trains a small, CPU-friendly crop classifier rather
than copying the heavy YOLO R&D trainer into this repository. It uses the same
prepared MTSD20 manifests and images already used by the R&D repository:
``train.jsonl`` and official labeled ``val.jsonl``. Each object bbox is cropped,
resized to 16x16 grayscale, and classified into one of the 21 MTSD classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_NAME = "mtsd20-crops-v1"
EXPERIMENT_NAME = "mlops-mlflow-homework"
MODEL_NAME = "mtsd20-crop-classifier"
MODEL_ALIAS = "champion"
RANDOM_STATE = 42
CROP_SIZE = 16
DEFAULT_MAX_TRAIN_PER_CLASS = 500
DEFAULT_MAX_VAL_PER_CLASS = 120


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    train_features: np.ndarray
    train_targets: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray
    prepared_root: Path
    images_root: Path
    digest: str
    class_names: list[str]


def _sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _discover_path(explicit: str | None, candidates: list[Path], label: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"{label} does not exist: {path}")
    for path in candidates:
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not discover {label}; checked: {formatted}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _crop_features(image_path: Path, bbox: list[float]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size
        center_x, center_y, box_width, box_height = bbox
        pixel_width = box_width * width
        pixel_height = box_height * height
        pad_x = max(pixel_width * 0.15, 2.0)
        pad_y = max(pixel_height * 0.15, 2.0)
        left = max(0, int((center_x * width) - pixel_width / 2 - pad_x))
        top = max(0, int((center_y * height) - pixel_height / 2 - pad_y))
        right = min(width, int((center_x * width) + pixel_width / 2 + pad_x))
        bottom = min(height, int((center_y * height) + pixel_height / 2 + pad_y))
        crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(crop, dtype=np.float32).reshape(-1) / 255.0


def _extract_split(
    records: list[dict[str, Any]],
    images_root: Path,
    max_per_class: int,
    split: str,
    class_names: list[str],
) -> tuple[list[np.ndarray], list[int], list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    rows: list[dict[str, Any]] = []
    counts = [0] * len(class_names)
    for record in records:
        image_path = images_root / record["image"]
        if not image_path.is_file():
            image_path = images_root / Path(record["image"]).name
        if not image_path.is_file():
            continue
        for object_index, obj in enumerate(record.get("objects", [])):
            class_id = int(obj["class_id"])
            if class_id >= len(class_names) or counts[class_id] >= max_per_class:
                continue
            try:
                vector = _crop_features(image_path, obj["bbox"])
            except (OSError, ValueError):
                continue
            features.append(vector)
            targets.append(class_id)
            counts[class_id] += 1
            rows.append(
                {
                    "split": split,
                    "image": record["image"],
                    "object_index": object_index,
                    "target": class_id,
                    "class_name": class_names[class_id],
                }
            )
        if all(count >= max_per_class for count in counts):
            break
    if not features:
        raise RuntimeError(f"no usable MTSD crops found for split={split}")
    return features, targets, rows


def load_dataset(
    prepared_root: Path | None = None,
    images_root: Path | None = None,
    max_train_per_class: int = DEFAULT_MAX_TRAIN_PER_CLASS,
    max_val_per_class: int = DEFAULT_MAX_VAL_PER_CLASS,
    data_dir: Path = Path("data/mlflow"),
) -> DatasetBundle:
    prepared_root = _discover_path(
        str(prepared_root) if prepared_root else os.getenv("MTSD_PREPARED_ROOT"),
        [
            Path("/media/eugene/data21/mtsd/prepared_mtsd20"),
            Path("/media/eugene/data2/mtsd/prepared_mtsd20"),
        ],
        "MTSD prepared root",
    )
    images_root = _discover_path(
        str(images_root) if images_root else os.getenv("MTSD_IMAGES_ROOT"),
        [
            prepared_root.parent / "runs/parallel_b0_b1/kaggle_stage/dataset/images",
            prepared_root.parent
            / "runs/quality_ladder/packs/mtsd20_letterbox_640_v2/images",
        ],
        "MTSD image root",
    )
    metadata = json.loads((prepared_root / "dataset.json").read_text(encoding="utf-8"))
    class_names = list(metadata["class_names"])
    train_jsonl = prepared_root / metadata["splits"]["train"]
    val_jsonl = prepared_root / metadata["splits"]["val"]
    train_features, train_targets, train_rows = _extract_split(
        _read_jsonl(train_jsonl),
        images_root,
        max_train_per_class,
        "train",
        class_names,
    )
    val_features, val_targets, val_rows = _extract_split(
        _read_jsonl(val_jsonl),
        images_root,
        max_val_per_class,
        "val",
        class_names,
    )
    frame = pd.DataFrame(train_rows + val_rows)
    feature_columns = [f"pixel_{index:03d}" for index in range(CROP_SIZE * CROP_SIZE)]
    all_features = np.vstack([train_features, val_features])
    frame = pd.concat(
        [frame, pd.DataFrame(all_features, columns=feature_columns)],
        axis=1,
    )
    frame = frame[
        ["split", "image", "object_index", "target", "class_name", *feature_columns]
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    canonical_csv = data_dir / "mtsd20_crops_v1.csv"
    frame.to_csv(canonical_csv, index=False)
    digest = _sha256(
        (prepared_root / "dataset.json", train_jsonl, val_jsonl, canonical_csv),
    )
    return DatasetBundle(
        frame=frame,
        train_features=np.asarray(train_features),
        train_targets=np.asarray(train_targets),
        val_features=np.asarray(val_features),
        val_targets=np.asarray(val_targets),
        prepared_root=prepared_root,
        images_root=images_root,
        digest=digest,
        class_names=class_names,
    )


def _dataset_tags(bundle: DatasetBundle) -> dict[str, str]:
    return {
        "dataset.name": DATASET_NAME,
        "dataset.source": str(bundle.prepared_root),
        "dataset.images_source": str(bundle.images_root),
        "dataset.digest": bundle.digest,
        "dataset.classes": str(len(bundle.class_names)),
        "dataset.preprocessing": "bbox crop + 15% context + grayscale 16x16 / 255",
        "lineage.contract": "mlflow.log_input(context=training|evaluation)",
    }


def _mlflow_dataset(bundle: DatasetBundle) -> Any:
    return mlflow.data.from_pandas(
        bundle.frame,
        source=str(bundle.prepared_root),
        name=DATASET_NAME,
        targets="target",
    )


def _log_common(bundle: DatasetBundle, context: str) -> None:
    mlflow.log_input(_mlflow_dataset(bundle), context=context)
    mlflow.set_tags(_dataset_tags(bundle))
    mlflow.log_params(
        {
            "dataset_name": DATASET_NAME,
            "dataset_digest": bundle.digest,
            "dataset_train_crops": len(bundle.train_targets),
            "dataset_val_crops": len(bundle.val_targets),
            "crop_size": CROP_SIZE,
            "random_state": RANDOM_STATE,
        }
    )


def _write_eda_artifacts(bundle: DatasetBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        bundle.frame["target"]
        .value_counts()
        .reindex(
            range(len(bundle.class_names)),
            fill_value=0,
        )
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(counts)), counts.values, color="#2563eb")
    ax.set(title="MTSD20 crop class distribution", xlabel="class id", ylabel="crops")
    ax.set_xticks(range(len(counts)), [str(i) for i in range(len(counts))], rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=140)
    plt.close(fig)

    means = (
        np.vstack([bundle.train_features, bundle.val_features])
        .reshape(-1, CROP_SIZE, CROP_SIZE)
        .mean(axis=0)
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(means, cmap="gray", vmin=0, vmax=1)
    ax.set(title="Mean MTSD crop", xticks=[], yticks=[])
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_crop.png", dpi=140)
    plt.close(fig)

    summary = {
        "dataset": DATASET_NAME,
        "prepared_root": str(bundle.prepared_root),
        "images_root": str(bundle.images_root),
        "sha256": bundle.digest,
        "train_crops": int(len(bundle.train_targets)),
        "val_crops": int(len(bundle.val_targets)),
        "classes": dict(enumerate(bundle.class_names)),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "observations": [
            "Train and official labeled val are split by source image manifest.",
            (
                "Class 20 is an intentional other-sign aggregation from the MTSD20 "
                "contract."
            ),
            (
                "Small-object crops are resized with context; this is a classification "
                "demo, not YOLO detection training."
            ),
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# MTSD20 EDA\n\n"
        f"- Source: `{bundle.prepared_root}`\n"
        f"- Image source: `{bundle.images_root}`\n"
        f"- Crops: train `{len(bundle.train_targets)}`, "
        f"val `{len(bundle.val_targets)}`\n"
        f"- SHA-256 lineage digest: `{bundle.digest}`\n"
        "- Class distribution and mean crop are logged as PNG artifacts.\n",
        encoding="utf-8",
    )


def run_eda(bundle: DatasetBundle) -> str:
    with mlflow.start_run(run_name="eda") as run:
        _log_common(bundle, "eda")
        with tempfile.TemporaryDirectory(prefix="mlflow-eda-") as temp_dir:
            _write_eda_artifacts(bundle, Path(temp_dir))
            mlflow.log_artifacts(temp_dir, artifact_path="eda")
        mlflow.log_metric("class_count", float(len(bundle.class_names)))
        return run.info.run_id


def _diagnostic_artifacts(
    output_dir: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(targets, predictions, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=range(len(class_names))).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation="vertical",
    )
    ax.set_title("MTSD20 crop confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    report = classification_report(
        targets,
        predictions,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def _candidates() -> list[tuple[str, Any, dict[str, Any]]]:
    return [
        (
            "baseline-logistic",
            LogisticRegression(max_iter=120, C=1.0, random_state=RANDOM_STATE),
            {"variant": "baseline", "model": "logistic_regression", "C": 1.0},
        ),
        (
            "scaled-logistic",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=120,
                            C=3.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ],
            ),
            {
                "variant": "preprocessing",
                "model": "standard_scaler+logistic_regression",
                "C": 3.0,
            },
        ),
        (
            "random-forest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=18,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "variant": "hyperparameters",
                "model": "random_forest",
                "n_estimators": 80,
                "max_depth": 18,
            },
        ),
    ]


def _version_for_run(model_name: str, run_id: str) -> str:
    client = MlflowClient()
    for _ in range(30):
        versions = client.search_model_versions(f"name='{model_name}'")
        matches = [version for version in versions if version.run_id == run_id]
        if matches:
            return str(max(matches, key=lambda item: int(item.version)).version)
        time.sleep(1)
    raise RuntimeError(f"registered model version for run {run_id} was not created")


def run_candidate(
    bundle: DatasetBundle,
    run_name: str,
    model: Any,
    params: dict[str, Any],
) -> tuple[str, float, str]:
    with mlflow.start_run(run_name=run_name) as run:
        _log_common(bundle, "training")
        mlflow.log_params(params)
        model.fit(bundle.train_features, bundle.train_targets)
        started = time.perf_counter()
        predictions = model.predict(bundle.val_features)
        elapsed_ms = (time.perf_counter() - started) * 1000
        accuracy = float(accuracy_score(bundle.val_targets, predictions))
        macro_f1 = float(f1_score(bundle.val_targets, predictions, average="macro"))
        precision = float(
            precision_score(
                bundle.val_targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        recall = float(
            recall_score(
                bundle.val_targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_macro": macro_f1,
                "precision_macro": precision,
                "recall_macro": recall,
                "inference_ms_per_batch": elapsed_ms,
            },
        )
        mlflow.set_tag("primary_metric", "f1_macro")
        with tempfile.TemporaryDirectory(prefix="mlflow-run-") as temp_dir:
            _diagnostic_artifacts(
                Path(temp_dir),
                bundle.val_targets,
                predictions,
                bundle.class_names,
            )
            mlflow.log_artifacts(temp_dir, artifact_path="diagnostics")
            mlflow.log_text(
                json.dumps(
                    {
                        "primary_metric": "f1_macro",
                        "f1_macro": macro_f1,
                        "accuracy": accuracy,
                    },
                    indent=2,
                ),
                "metrics.json",
            )
        signature = infer_signature(
            bundle.train_features,
            model.predict(bundle.train_features[:5]),
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=bundle.train_features[:2],
            registered_model_name=MODEL_NAME,
        )
        version = _version_for_run(MODEL_NAME, run.info.run_id)
        return run.info.run_id, macro_f1, version


def run_training(bundle: DatasetBundle) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for run_name, model, params in _candidates():
        run_id, score, version = run_candidate(bundle, run_name, model, params)
        results.append(
            {
                "run_name": run_name,
                "run_id": run_id,
                "f1_macro": score,
                "version": version,
            },
        )
    winner = max(results, key=lambda item: item["f1_macro"])
    client = MlflowClient()
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
            "source": str(bundle.prepared_root),
            "digest": bundle.digest,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--images-root", default=None)
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=DEFAULT_MAX_TRAIN_PER_CLASS,
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=DEFAULT_MAX_VAL_PER_CLASS,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("MLOPS_DATA_DIR", "data/mlflow")),
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    bundle = load_dataset(
        prepared_root=Path(args.prepared_root) if args.prepared_root else None,
        images_root=Path(args.images_root) if args.images_root else None,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        data_dir=args.data_dir,
    )
    eda_run_id = run_eda(bundle)
    run_training(bundle)
    print(json.dumps({"completed": True, "eda_run_id": eda_run_id}, indent=2))


if __name__ == "__main__":
    main()
