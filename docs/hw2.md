# ДЗ 2 — чеклист, ловушки защиты, CD

## Эндпоинты

| Путь | Назначение |
|------|------------|
| `GET /` | Редирект на `/docs` (Swagger UI) |
| `GET /healthz` | Liveness для Docker/K8s. **Без `/api/v1`**: пробы оркестратора должны иметь стабильный путь, не зависящий от версии API. Не проверяет БД. Sync `def` — нет I/O, event loop не блокируется. |
| `GET /api/v1/version` | Версия из `importlib.metadata` (= `version` в `pyproject.toml`). **Ловушка**: не хардкодить в роутере. |
| `GET /api/v1/health` | E2E: async `SELECT version()` + latency; версии fastapi/asyncpg/uvicorn. Мёртвая БД → HTTP 503. |

## Конфиг и секреты

- `config.yaml` — дефолты для локальной разработки.
- `.env.example` → скопировать в `.env` (gitignore) для переопределений.
- Compose: `APP_DATABASE__URL` (префикс `APP_` + nested `DATABASE__URL`), внутри сети хост `postgres:5432`.

## Ruff (`select`)

| Код | Смысл |
|-----|--------|
| E/W | pycodestyle errors/warnings |
| F | pyflakes (неиспользуемое, undefined) |
| I | isort |
| B | bugbear (опасные паттерны, bare except) |
| UP | pyupgrade под target py312 |

## CD и версионирование

Реестр: `ghcr.io/eugenb0t/mtsd_detector_mlops` (**package должен быть Public** — отдельно от visibility репозитория).

Теги образа: `<semver>` (= pyproject / `/api/v1/version`), `sha-<7>`, `latest`.

**Триггеры**

1. `push` тегов `v*` — semver тега **обязан** совпасть с `pyproject.toml`; иначе workflow падает.
2. `workflow_dispatch` — пересборка текущей версии из pyproject **без** ручного override (чтобы не опубликовать чужой тег).

Перед push образа всегда проходит reusable **CI quality** job (lint/format/lock/tests+coverage). CI также триггерится на теги `v*`.

```bash
# bump version in pyproject.toml, then:
git tag v0.1.2 && git push origin v0.1.2
docker pull ghcr.io/eugenb0t/mtsd_detector_mlops:0.1.2
```

Если `docker pull` → 401: GitHub → Packages → `mtsd_detector_mlops` → Package settings → Change visibility → Public.

## Локальный цикл

```bash
cp .env.example .env   # optional
uv sync
uv run pre-commit install
make lint && make cov
make up-db
uv run uvicorn mtsd_detector.app:create_app --factory --host 0.0.0.0 --port 8000
# or: make up-prod
make smoke
```
