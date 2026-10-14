# MTSD Detector MLOps

Учебный репозиторий курса MLOps: асинхронный FastAPI-сервис как каркас пайплайна детекции дорожных знаков (Mapillary MTSD → train → gates → artifacts).

Репозиторий: [eugenb0t/mtsd_detector_mlops](https://github.com/eugenb0t/mtsd_detector_mlops)
Образ: `ghcr.io/eugenb0t/mtsd_detector_mlops:<version>`

## Стек

- **uv** — пакетный менеджер + `uv.lock` (воспроизводимая установка)
- **FastAPI** + **uvicorn** + **asyncpg** (runtime / «web»-зависимости в `[project]`)
- **dev**-группа: ruff, pytest, coverage, pre-commit
- **pydantic-settings** + `config.yaml` (env перекрывает YAML: `APP_`, nested `__`)
- **Docker / Compose**, **GitHub Actions** (CI → CD/GHCR)

## Быстрый старт

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cp .env.example .env   # optional local overrides
uv sync
uv run pre-commit install
make cov
make up-db
uv run uvicorn mtsd_detector.app:create_app --factory --host 0.0.0.0 --port 8000
```

Браузер: http://127.0.0.1:8000/ → редирект на Swagger (`/docs`).

Полный стек: `make up-prod`. Smoke: `make smoke`.

### Эндпоинты

| Метод | Путь | Зачем |
|-------|------|--------|
| GET | `/healthz` | liveness **вне** `/api/v1` (стабильный путь проб, без БД) |
| GET | `/api/v1/version` | версия из package metadata (= `pyproject.toml`) |
| GET | `/api/v1/health` | версии third-party + latency (async Postgres) |

Подробнее и ответы на ловушки защиты: [docs/hw2.md](docs/hw2.md).

## Практическая работа MLflow

Полный контур ДЗ находится в [docs/mlflow-homework.md](docs/mlflow-homework.md):
Compose поднимает отдельный Postgres backend store, файловый artifact store в
отдельном Docker volume и
MLflow Tracking/Registry Server. Detector pipeline использует найденный
подготовленный MTSD20 (`val.jsonl` с bbox-аннотациями), оценивает два внешних
ONNX detector checkpoint’а, логирует mAP/latency/диагностику и регистрирует
выбранный полноценный detector с alias `champion`. Датасет и ONNX-веса остаются
вне git.

```bash
cp .env.example .env
uv sync
docker compose up -d mlflow
uv run mlflow-pipeline \
  --prepared-root /media/eugene/data21/mtsd/prepared_mtsd20 \
  --images-root /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images
uv run mlflow-detector \
  --prepared-root /media/eugene/data21/mtsd/prepared_mtsd20 \
  --images-root /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images
docker compose --profile prod up --build -d app
```

UI: http://127.0.0.1:5000. После регистрации alias приложение загружает
`models:/mtsd20-detector@champion` один раз при старте. `/process` принимает
base64-encoded RGB image и возвращает классы, scores и bounding boxes. POST
контракт и проверка версии описаны в документации ДЗ.

## ДЗ4: FastStream/RabbitMQ backpressure

Практическая работа по конвейеру задач описана в
[docs/hw4-backpressure.md](docs/hw4-backpressure.md). Архитектура состоит из
FastAPI/batch source → bounded RabbitMQ queue → независимые FastStream workers →
PostgreSQL results. Очередь использует `x-max-length=8` и
`x-overflow=reject-publish`; publisher-confirm NACK автоматически замедляет
batch source. Внешний `POST /api/v1/tasks` при переполнении немедленно отвечает
`503` с `Retry-After` (это load shedding на границе системы), а не ждёт очередь.

```bash
docker compose --profile prod --profile hw4 up --build -d --scale worker=1
APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/ \
APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd \
uv run mtsd-source \
  --image /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg \
  --count 100
APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/ \
APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd \
uv run mtsd-experiment \
  --image /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg \
  --count 100 \
  --output artifacts/hw4/bounded-1-worker.jsonl
docker compose --profile hw4 up -d --scale worker=2
```

`prefetch_count=2` явно равен `worker_concurrency=2`; inference вынесен из
event loop, модель загружается один раз по MLflow alias, ручной ACK выполняется
после записи результата, а poison messages после ограниченных retry попадают в
DLQ. Эксперимент пишет в динамике queue depth, queue memory, source rate, RSS,
latency и число завершённых задач. Полный сценарий bounded/unbounded и
интеграционный Testcontainers-тест — в документации ДЗ4. Фактический прогон
зафиксировал: bounded/1 worker — ready queue максимум 8, 40/40 DONE;
bounded/2 workers — 40/40 DONE и средняя latency 4240 ms; unbounded/1 worker —
ready queue максимум 38 сообщений при burst из 40. Подробная таблица и
сценарии SIGTERM, DLQ и HTTP load shedding находятся в документации ДЗ4.

## Конфигурация

- YAML: `config.yaml`
- Env example: [.env.example](.env.example) → `.env` (не в git)
- Compose подхватывает `${POSTGRES_*}` / `APP_DATABASE__URL`

Внутри compose-сети DSN: `postgresql://…@postgres:5432/…` (DNS-имя сервиса, порт контейнера). С хоста для `uv run`: `…@127.0.0.1:5433/…`.

## CI / CD

| Workflow | Триггер | Действие |
|----------|---------|----------|
| CI | push/PR `main`, теги `v*`, reusable | ruff lint + format `--check`, `uv lock --check`, pytest + coverage ≥80% |
| CD | тег `v*` или `workflow_dispatch` | **сначала CI**, затем build/push GHCR |

**Почему CD на тегах:** не плодим образы на каждый коммит; semver тега = `pyproject.toml` = `/api/v1/version`.

```bash
# после bump version в pyproject.toml
git tag v0.1.2 && git push origin v0.1.2
```

Теги образа: `:0.1.2`, `:sha-<short>`, `:latest`.
Пакет GHCR должен быть **Public** (Settings пакета), иначе anonymous `docker pull` даст 401.

## Структура

```text
src/mtsd_detector/   # async FastAPI (api / services / schemas)
config.yaml
.env.example
tests/
Dockerfile
docker-compose.yml
.github/workflows/
architecture/
docs/
```

## Лицензия

MIT — [LICENSE](LICENSE).
