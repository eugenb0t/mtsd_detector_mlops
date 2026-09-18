# Handoff: контекст для продолжения работы

## Тема (курс MLOps)
**MLOps-пайплайн детекции дорожных знаков: от Mapillary MTSD до воспроизводимого обучения, валидации и деплоя артефактов**

### Краткое описание проекта
Воспроизводимый цикл: подготовка MTSD → таксономия MTSD20+other → удалённое обучение детекторов (YOLO/P2) → трекинг экспериментов и quality gates (mAP50, operational AP, latency) → экспорт ONNX → smoke на PC. Цель — управляемый пайплайн, а не разовое обучение.

### Датасет
**Mapillary Traffic Sign Dataset (MTSD)** — открытый street-view датасет дорожных знаков (сотни тысяч изображений, детальные bbox + семантика). В проекте: MTSD v2 → свёртка в **21 класс** (20 навигационных + `other-sign`); ambiguous/dummy/out-of-frame = ignore.

Классы (порядок = class_id): stop, yield, no-entry, no-parking, no-stopping, keep-right/left, priority-road, roundabout, go-straight, turn-left/right, no-left/right-turn, no-u-turn, maximum-speed-limit, pedestrians-crossing (info/warning), roadworks, children, **other-sign**.

## Известные факты / ловушки (важно)
1. **Domain gap Himax gray**: live SoftAP с AI-deck — grayscale. RGB MTSD-веса на большом STOP с телефона в комнате дают `dets=0` даже при `conf=0.05`. На цветном MTSD-кадре тот же чекпоинт детектит STOP (~0.5).
2. Viewer по умолчанию на CPU берёт **ONNX**, плюс **TemporalSignTracker (window=3)** — HUD показывает confirmed dets, не raw. Для отладки: `--backend torch --no-temporal --conf 0.001`.
3. Autoloop map≥0.75 сейчас **blocked** около mAP50≈0.45 (`mtsd20-b2m-yolov8m-p2-640`).
4. Smoke «модель жива»: цветное RGB фото знака + `--source image --once`, не SoftAP.

## Полезные команды (из R&D)
```bash
cd /data/projects/crazyflie/tinyml_demo
# smoke RGB
/data/projects/crazyflie/.venv/bin/python scripts/live_pc_mtsd20_viewer.py \
  --source image --image /path/to/color_stop.jpg \
  --checkpoint /media/eugene/data2/mtsd/runs/quality_ladder/autoloop_map075/viewer/pc_s640_best.pt \
  --backend torch --no-temporal --conf 0.05 --once --save /tmp/mtsd_smoke.jpg

# live SoftAP (ожидать слабый recall на gray без fine-tune)
.../live_pc_mtsd20_viewer.py \
  --checkpoint .../pc_s640_best.pt \
  --backend torch --no-temporal --conf 0.001
```

## Предыдущий чат (Crazyflie)
Тема курса + диагностика live viewer: поиск по agent transcripts в
`/home/eugene/.cursor/projects/data-projects-crazyflie/agent-transcripts/`
(сессия с формулировкой MLOps-темы и разбором `dets=0` на SoftAP).

## Что логично делать дальше в `~/mlops`
- Оформить курсовой пайплайн (data → train → eval → register artifacts) без зависимости от прошивки.
- Зафиксировать контракт датасета/метрик и report template.
- Отдельно: план gray/drone domain adaptation (не смешивать с «модель сломана»).
