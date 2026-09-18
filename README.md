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
