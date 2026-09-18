# Архитектура проекта (MLOps / MTSD20)

## Задача
Детекция навигационных дорожных знаков с воспроизводимым MLOps-циклом:
подготовка данных → remote train → оценка/гейты → артефакты (PT/ONNX) → smoke/live.

## Источник данных
- **Mapillary Traffic Sign Dataset (MTSD) v2** — street-view, богатая таксономия знаков.
- Рабочая таксономия проекта: **MTSD20+other** (20 навигационных классов + `other-sign` = 21).
- Ignore regions: `ambiguous`, `dummy`, `out-of-frame` (не фон).
- Подготовленный пак (на машине разработки):
  `/media/eugene/data2/mtsd/prepared_mtsd20`
  Официальный root: `/data/datasets/mtsd`

## Модельный контур (из R&D)
| ID / alias | Модель | imgsz | Роль |
|------------|--------|-------|------|
| B0 | YOLOv8n @640 | 640 | control |
| B1 / P2 | YOLOv8n-P2 @640 | 640 | laptop / portable |
| pc_s640 | YOLOv8s-P2 @640 | 640 | PC quality (viewer) |
| B2m | YOLOv8m-P2 @640 | 640 | heavier PC |

Чекпоинт viewer (пример):
`/media/eugene/data2/mtsd/runs/quality_ladder/autoloop_map075/viewer/pc_s640_best.{pt,onnx}`

Текущий лучший mAP50 в autoloop ~**0.45** (цель курса/гейта была ≥0.75 — ещё не достигнута).

## MLOps-слои
1. **Data** — audit → prepare MTSD20 → letterbox packs → versioned manifests (`dataset.json` / JSONL).
2. **Train** — remote (Kaggle/Colab) через общий trainer `mtsd_remote_train.py`.
3. **Eval / gates** — all-scale mAP50 + operational AP (min-side ≥8/16/32); latency P99; ONNX parity.
4. **Artifacts** — `.pt` + `.onnx`; пути вне git на `data2`.
5. **Serve/smoke** — курсовой FastAPI-сервис в этом репо (`src/mtsd_detector`): `/healthz`, `/api/v1/version`, `/api/v1/health`; образ в GHCR. Live PC viewer из R&D: `live_pc_mtsd20_viewer.py`; SoftAP JPEG с AI-deck — отдельный домен.

## Сервис в этом репо (ДЗ 2)
- Пакет: `src/mtsd_detector` (async FastAPI, YAML-конфиг, asyncpg).
- Compose: `postgres` + `app` (profile `prod`); лимиты CPU/RAM; `.env.example`.
- CI на `main` и тегах `v*`; CD публикует `ghcr.io/eugenb0t/mtsd_detector_mlops` только после зелёного quality job.
- Детали сдачи / ловушки защиты: [docs/hw2.md](../docs/hw2.md).

## Связанный код (read-only ссылки)
- Подготовка/таксономия: `tinyml_demo/tinyml_demo/mtsd.py`
- Eval: `tinyml_demo/tinyml_demo/mtsd_evaluation.py`
- ONNX parse: `tinyml_demo/tinyml_demo/mtsd_onnx_eval.py`
- Live viewer: `tinyml_demo/scripts/live_pc_mtsd20_viewer.py`
- Research notes: `tinyml_demo/docs/mtsd_research.md`, `mtsd_drone_domain.md`

Корень R&D: `/data/projects/crazyflie/tinyml_demo`

## Домены
| Домен | Сенсор | Цвет | Совместимость с MTSD-весами |
|-------|--------|------|-----------------------------|
| Train/eval MTSD | Mapillary street | RGB | да |
| SoftAP AI-deck | Himax | grayscale JPEG | **слабая** (STOP/цвет критичен) |
| Целевой drone companion | RGB GS camera | RGB | план адаптации |

## Граница этого репо
`~/mlops` держит курсовые артефакты, описание пайплайна и воспроизводимые рецепты.
Тяжёлые данные/веса — только по абсолютным путям или внешнему storage, не в git.
