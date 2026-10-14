# ДЗ4 — FastStream, RabbitMQ и backpressure

## Архитектура

```text
external client
      │ POST /api/v1/tasks
      ▼
FastAPI source ── publisher confirm ──▶ RabbitMQ
                                      │
                         x-max-length=8, x-overflow=reject-publish
                                      │
                                      ▼
                    FastStream worker × N
                    prefetch=2, concurrency=2
                                      │
                         MLflow models:/...@champion
                                      │
                                      ▼
                              PostgreSQL results
                                      │
                                      └── poison messages → DLQ
```

Источник/API и worker — отдельные процессы. Они не вызывают друг друга по
HTTP: единственный транспорт между ними — RabbitMQ. Batch source запускается
командой `mtsd-source`; API — это тот же publisher-контур для внешнего
клиента.

## Почему это backpressure

Очередь объявляется с `x-max-length=8` и `x-overflow=reject-publish`. Когда
готовая часть очереди заполнена, RabbitMQ возвращает publisher-confirm NACK.
`TaskSource.submit_with_backpressure()` повторяет тот же элемент после
`Retry-After`, поэтому источник не держит неограниченный список задач и не
теряет текущую задачу. Это сигнал от брокера, а не опрос Management API.

Опрос Management API выполняется только измерительным скриптом и не влияет на
принятие решений источника.

Для сравнения были рассмотрены альтернативы. Rate limiting ограничил бы
источник заранее выбранной частотой и не реагировал бы на фактическое число
воркеров. Credit-based протокол потребовал бы отдельного счётчика кредитов и
его восстановления при сбоях. Опрос Management API проще, но даёт устаревшее
значение, создаёт дополнительную зависимость от административного API и
подвержен гонке «прочитал место — очередь уже заполнилась». Поэтому выбран
нативный сигнал RabbitMQ: publisher confirm NACK от `reject-publish`.

`prefetch_count=2` выбран равным `worker_concurrency=2`: в одном процессе две
задачи могут одновременно находиться в обработчике и ровно две доставки могут
быть неподтверждены. При двух процессах совокупная конкурентность равна четырём
без изменения кода.

Внешний API является границей системы. Он не ждёт освобождения очереди: при
NACK немедленно отвечает `503 Service Unavailable` и `Retry-After`. Значение
`Retry-After` по умолчанию равно `max(1 с, ceil(inference_delay_s))`: это один
минимальный интервал восстановления, а не произвольная задержка; его можно
переопределить конфигурацией. До publish
создаётся только временная запись `PUBLISHING`; при отказе RabbitMQ она
удаляется, поэтому `PENDING` не остаётся. Это load shedding, а не
backpressure: внешний клиент нельзя замедлить внутренним сигналом, поэтому
перегруженный запрос отбрасывается и должен быть повторён клиентом.

## Надёжность

- worker загружает `models:/mtsd20-detector@champion` один раз в FastStream
  startup hook;
- синхронное ONNX inference выполняется через `asyncio.to_thread`, а задержка
  эксперимента — через `asyncio.sleep`, event loop не блокируется;
- используется `AckPolicy.MANUAL`, ACK отправляется только после записи
  результата в PostgreSQL;
- ошибка повторяется ограниченное число раз (`max_retries=2`) с backoff; после
  последней попытки сообщение сначала публикуется в durable DLQ с confirm, затем
  ACK-ается исходное сообщение;
- `task_id` — первичный ключ, а `idempotency_key` — уникальный ключ; повторная
  доставка после уже записанного `DONE` только ACK-ается;
- FastStream graceful timeout ждёт уже принятые сообщения; SIGTERM прекращает
  новые доставки, а ручные ACK не подтверждённых сообщений позволяют RabbitMQ
  безопасно вернуть их в очередь.

## Запуск

Сначала должен существовать зарегистрированный detector alias из ДЗ3:

```bash
docker compose up -d postgres mlflow rabbitmq
uv run mlflow-detector --tracking-uri http://127.0.0.1:5000 \
  --prepared-root /media/eugene/data21/mtsd/prepared_mtsd20 \
  --images-root /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images \
  --max-images 128
```

Запуск API и одного worker:

```bash
docker compose --profile prod --profile hw4 up --build -d --scale worker=1
```

API принимает `POST /api/v1/tasks` и возвращает `202` с `task_id`. Старый
`POST /process` оставлен как совместимый alias, но теперь тоже создаёт
асинхронную задачу. Результат читается через `GET /api/v1/tasks/{task_id}`.

Batch source не копит задачи в RAM:

```bash
APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/ \
APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd \
uv run mtsd-source \
  --image /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg \
  --count 100
```

## Эксперимент

`mtsd-experiment` пишет динамику в JSONL и итог в соседний
`metrics.summary.json`:

```bash
APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/ \
APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd \
uv run mtsd-experiment \
  --image /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg \
  --count 100 \
  --output artifacts/hw4/bounded-1-worker.jsonl
```

Фиксируются: глубина очереди и её memory из RabbitMQ Management API, число
consumer-ов, скорость подтверждённых публикаций, RSS процесса source,
количество `DONE/FAILED` и среднее время задачи из PostgreSQL. Management API
используется только для наблюдения.

Для контрольного запуска без ограничения очереди нужно использовать другое имя
очереди и отдельный worker без `x-max-length`:

```bash
APP_RABBIT__QUEUE_NAME=mtsd.tasks.unbounded \
APP_RABBIT__BOUNDED_QUEUE=false \
docker compose --profile hw4 up --build -d --scale worker=1

APP_RABBIT__QUEUE_NAME=mtsd.tasks.unbounded \
APP_RABBIT__BOUNDED_QUEUE=false \
APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/ \
APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd \
uv run mtsd-experiment \
  --image /media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg \
  --count 100 \
  --output artifacts/hw4/unbounded-1-worker.jsonl
```

Затем, без изменения программного кода, повторяется controlled-run с двумя
worker-процессами:

```bash
docker compose --profile hw4 up -d --scale worker=2
uv run mtsd-experiment --image /path/to/image.jpg --count 100 \
  --output artifacts/hw4/bounded-2-workers.jsonl
```

Ожидаемая картина:

| Режим | Очередь | Скорость source | Память source | Результат |
|---|---|---:|---:|---|
| bounded, 1 worker | достигает 8 и стабилизируется | сходится к throughput worker | ограничена | все задачи DONE |
| unbounded, 1 worker | растёт вместе с burst | близка к скорости генерации | растёт вместе с backlog | все задачи позже DONE |
| bounded, 2 workers | тот же лимит, быстрее дренируется | примерно удваивается до предела source | ограничена | все задачи DONE быстрее |

Фактические числа защиты берутся из `*.summary.json`; они зависят от CPU,
размера модели и числа задач. Смена `--scale worker=1` на `--scale worker=2`
не требует изменения `prefetch_count`, кода source или API.

Фактический прогон на одном и том же RGB-изображении MTSD, `queue_limit=8`,
`prefetch=2`, `concurrency=2`:

| Режим | Задач | max ready | max total | max unacked | source RSS | средняя latency | DONE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bounded, 1 worker | 40 | 8 | 10 | 2 | 53.12 MiB | 5645 ms | 40 |
| bounded, 2 workers | 40 | 5 | 9 | 4 | 52.88 MiB | 4240 ms | 40 |
| unbounded, 1 worker | 40 | 38 | 40 | 2 | 52.70 MiB | 14880 ms | 40 |

`max ready` — именно глубина готовой части очереди, ограниченная
`x-max-length`; `max total = ready + unacked`, поэтому в bounded-режиме
значение 10 не является нарушением лимита. Через добавление второго worker
совокупная конкурентность выросла с 2 до 4, latency прогона снизилась, а
лимит очереди и код source не менялись. В unbounded-контроле очередь приняла
весь burst (40 сообщений), тогда как bounded-вариант остановил ready-часть на
8 и заставил source ждать подтверждения.

Дополнительные защитные прогоны:

- burst из 30 HTTP-запросов дал `202=10`, `503=20`; все 503 содержали
  `Retry-After: 1`. После завершения принятых задач запрос к PostgreSQL дал
  `PENDING/PROCESSING=0`;
- при отправке `SIGTERM` одному worker во время batch из 24 задач итог был
  `DONE=24`, без потери, а RabbitMQ после восстановления consumer-а показал
  `messages=0`;
- poison-задача `poison-demo-20261014` завершилась как `FAILED`,
  `attempts=2`, и оказалась в `mtsd.tasks.dlq` после ограниченных повторов.

Компактные итоговые артефакты этих запусков сохранены в репозитории:
`artifacts/hw4/bounded-1-worker-40.summary.json`,
`artifacts/hw4/bounded-2-workers-40.summary.json`,
`artifacts/hw4/unbounded-1-worker-final.summary.json` и
`artifacts/hw4/graceful-sigterm.summary.json`. Полные JSONL-трассы намеренно
игнорируются Git, чтобы не раздувать учебный репозиторий; их можно получить
повторным запуском приведённых команд.

## Проверки

```bash
uv run pytest -q
RUN_RABBIT_INTEGRATION=1 uv run pytest -q tests/test_task_integration.py -m integration
```

Первый запуск содержит модульные тесты worker через `TestRabbitBroker`; второй
поднимает настоящий RabbitMQ через Testcontainers и проверяет publisher-confirm
NACK при заполнении bounded queue.
