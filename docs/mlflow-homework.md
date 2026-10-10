# Практическая работа — MLflow

## Архитектура

`docker-compose.yml` поднимает независимые хранилища:

```text
MLflow UI/API :5000
   ├── Backend Store: mlflow-db (PostgreSQL 16)
   └── Artifact Store: separate Docker volume mounted at /mlartifacts
```

MLflow работает с `--serve-artifacts`: клиентам pipeline не нужен доступ к
внутреннему artifact volume, а сервер проксирует загрузку артефактов. В UI видны experiment, runs,
params, metrics, artifacts и Model Registry.

## Датасет и lineage

Используется уже подготовленный в соседнем R&D-контуре MTSD20:

- `prepared_mtsd20/dataset.json` — 21 класс и контракт MTSD20;
- `train.jsonl` и официальный labeled `val.jsonl`;
- изображения из `runs/parallel_b0_b1/kaggle_stage/dataset/images`.

Основной detector pipeline не копирует датасет или веса в репозиторий. Он
оценивает готовые обученные MTSD20 YOLO/ONNX checkpoints с входом RGB 640×640,
декодирует `xywh`-выход, применяет class-aware NMS и возвращает bounding boxes
в пикселях исходного изображения. Crop-classifier pipeline из
`mlflow-pipeline` оставлен как отдельный baseline для сравнения, но в Registry
и API используется полноценный object detector.

`mlflow.log_input()` записывает dataset input в каждый detector run.
Дополнительно tags содержат `dataset.source`, `dataset.digest`, источник
изображений и контракт `MTSD20 bbox manifest -> ONNX detector -> POST /process`.
Digest detector-а — SHA-256 `dataset.json` + официального `val.jsonl`; это
явная версия evaluation manifest, а не скрытая копия датасета.

## Запуск

```bash
uv sync
cp .env.example .env
docker compose up -d mlflow
curl -fsS http://127.0.0.1:5000/health

uv run mlflow-pipeline \
  --tracking-uri http://127.0.0.1:5000 \
  --prepared-root /media/eugene/data21/mtsd/prepared_mtsd20 \
  --images-root /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images

uv run mlflow-detector \
  --tracking-uri http://127.0.0.1:5000 \
  --prepared-root /media/eugene/data21/mtsd/prepared_mtsd20 \
  --images-root /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images \
  --max-images 128
```

`mlflow-detector` равномерно выбирает `--max-images` изображений из официального
`val.jsonl`, поэтому быстрый CPU-run не смещён к первым записям манифеста.
Полный dataset не изменяется. Checkpoints остаются на внешнем разделе, а при
регистрации копируются только в MLflow Artifact Store.

В experiment `mlops-mtsd-detector` появляются два осмысленных detector runs:

1. `detector-portable-b1` — компактный portable ONNX checkpoint;
2. `detector-quality-pc` — более точный PC checkpoint.

Каждый run содержит `mAP50`, `mAP50_present_classes`, precision/recall при
IoU=0.5, mean/P99 latency, FPS и `evaluation/report.json`. Лучший detector
регистрируется как `mtsd20-detector`, а alias `champion` указывает на его
неизменяемую версию. Promotion gate: `mAP50_present_classes >= 0.40` и
`P99 <= 600 ms` на CPU-limited Compose runtime.

Дополнительный baseline experiment `mlops-mlflow-homework` содержит:

1. `eda` — class distribution, mean crop, `summary.json`, report;
2. `baseline-logistic` — baseline;
3. `scaled-logistic` — preprocessing variant;
4. `random-forest` — hyperparameter/model variant.

Каждый baseline model run содержит `accuracy`, основную `f1_macro`, вспомогательные
`precision_macro`, `recall_macro`, inference latency, confusion matrix и
classification report. Каждая модель регистрируется как новая версия
`mtsd20-crop-classifier`. Этот baseline не используется production inference.

## Inference

После pipeline запусти API. Для Docker Compose:

```bash
docker compose --profile prod up --build -d app
curl -fsS http://127.0.0.1:8000/healthz
```

`app` получает `APP_MLFLOW__TRACKING_URI=http://mlflow:5000` и при startup
однократно загружает `models:/mtsd20-detector@champion`. На `/process` нет
вызовов MLflow: endpoint использует объект `app.state.model`.

Payload — base64-encoded RGB image и параметры postprocessing:

```bash
uv run python - <<'PY'
import base64, json, pathlib, urllib.request
image = next(pathlib.Path(
    "/media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images"
).glob("*.jpg")).read_bytes()
payload = json.dumps({
    "image_base64": base64.b64encode(image).decode(),
    "confidence": 0.05,
    "iou_threshold": 0.5,
    "max_detections": 50,
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/process", data=payload,
    headers={"Content-Type": "application/json"}, method="POST",
)
print(urllib.request.urlopen(request).read().decode())
PY
```

Ответ содержит `detections[]` с `class_id`, `class_name`, `score`, `box`
(`x0,y0,x1,y1` в пикселях исходного изображения), а также image size,
latency, `model_uri`, `model_version` и `loaded_at`. Чтобы
продемонстрировать alias switching, в UI или CLI переключи `champion` на другую
версию, затем перезапусти app:

```bash
uv run python - <<'PY'
from mlflow import MlflowClient
client = MlflowClient("http://127.0.0.1:5000")
client.set_registered_model_alias("mtsd20-detector", "champion", "13")
PY
docker compose --profile prod restart app
```

Версия меняется без изменения inference-кода. Горячая смена alias без reload
опасна: процесс продолжит использовать старый объект модели, а несогласованное
обновление может дать разные ответы у разных workers. Для production лучше
делать controlled rollout: загрузить новую версию в новый процесс, выполнить
health/smoke и переключить traffic (либо периодически reload с lock и rollback).

## Защита

- Backend store — PostgreSQL, artifact store — отдельный Docker volume;
  это разные роли и данные.
- Dataset input — lineage link run → dataset; `source` говорит откуда, digest
  позволяет обнаружить изменение содержимого.
- Alias — подвижная роль (`champion`), version — неизменяемая запись модели.
- `/process` асинхронный, модель грузится в lifespan startup и не загружается
  заново на каждый запрос; CPU inference вынесен в worker thread.
- Model Registry alias выбирает версию detector-а без изменения inference-кода.
- MTSD RGB и SoftAP Himax grayscale — разные домены; этот API явно принимает
  RGB MTSD-style images и не заявляет эквивалентность live AI-deck inference.
