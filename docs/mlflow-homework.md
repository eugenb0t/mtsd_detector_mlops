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

Pipeline не копирует датасет в репозиторий. Он извлекает ограниченное число
объектных crops на класс, добавляет 15% контекста, переводит в grayscale 16×16
и нормирует в `[0, 1]`. Это CPU-friendly классификационный слой для демонстрации
MLflow, а не замена YOLO detector из R&D.

`mlflow.log_input()` записывает dataset input в каждый run. Дополнительно
параметры/tags содержат `dataset.source`, `dataset.digest`, источники images и
preprocessing. `digest` — SHA-256 манифеста, JSONL и канонического CSV crops;
`source` — путь подготовленного MTSD root. Поэтому lineage можно объяснить от
конкретного run к конкретной версии данных.

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
```

По умолчанию pipeline берёт до 500 train и 120 val crops на класс. Это даёт
достаточную выборку для сравнения, но не требует обучать YOLO и не создаёт
тяжёлые git-артефакты. Полный dataset не изменяется.

В experiment `mlops-mlflow-homework` появляются:

1. `eda` — class distribution, mean crop, `summary.json`, report;
2. `baseline-logistic` — baseline;
3. `scaled-logistic` — preprocessing variant;
4. `random-forest` — hyperparameter/model variant.

Каждый model run содержит `accuracy`, основную `f1_macro`, вспомогательные
`precision_macro`, `recall_macro`, inference latency, confusion matrix и
classification report. Каждая модель регистрируется как новая версия
`mtsd20-crop-classifier`; pipeline выбирает лучший `f1_macro` и устанавливает
ему alias `champion`.

## Inference

После pipeline запусти API. Для Docker Compose:

```bash
docker compose --profile prod up --build -d app
curl -fsS http://127.0.0.1:8000/healthz
```

`app` получает `APP_MLFLOW__TRACKING_URI=http://mlflow:5000` и при startup
однократно загружает `models:/mtsd20-crop-classifier@champion`. На `/process`
нет вызовов MLflow: endpoint использует объект `app.state.model`.

Payload — crop после того же preprocessing, 256 чисел (16×16 grayscale,
row-major, значения 0…1):

```bash
uv run python - <<'PY'
import json, urllib.request
payload = json.dumps({"features": [0.0] * 256}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/process", data=payload,
    headers={"Content-Type": "application/json"}, method="POST",
)
print(urllib.request.urlopen(request).read().decode())
PY
```

Ответ содержит `prediction`, `model_uri`, `model_version` и `loaded_at`. Чтобы
продемонстрировать alias switching, в UI или CLI переключи `champion` на другую
версию, затем перезапусти app:

```bash
uv run python - <<'PY'
from mlflow import MlflowClient
client = MlflowClient("http://127.0.0.1:5000")
client.set_registered_model_alias("mtsd20-crop-classifier", "champion", "1")
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
  заново на каждый запрос.
- MTSD RGB и SoftAP Himax grayscale — разные домены; этот crop demo не заявляет
  эквивалентность live AI-deck inference.
