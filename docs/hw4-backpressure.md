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
  результата в PostgreSQL. ACK до записи результата мог бы потерять задачу при
  падении worker-а, а автоматическое подтверждение до выхода handler-а имело бы
  тот же риск. Подтверждение после записи сохраняет возможность повторной
  доставки при падении между commit и ACK; в этом случае `task_id` и уникальные
  ограничения хранилища превращают повтор в идемпотентную операцию;
- ошибка повторяется ограниченное число раз (`max_retries=2`) с backoff; после
  последней попытки сообщение сначала публикуется в durable DLQ с confirm, затем
  ACK-ается исходное сообщение. Если сама публикация retry получает NACK из-за
  заполненной очереди, передача повторяется ограниченное число раз с backoff;
  если очередь всё ещё не принимает retry, исходное сообщение надёжно
  помещается в DLQ и ACK-ается после confirm. Так оно не зависает unacked
  навсегда и не теряется;
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

Фиксируются: ready depth и queue memory из RabbitMQ Management API, число
consumer-ов, скорость подтверждённых публикаций, RSS процесса source,
использование памяти узлом RabbitMQ и его изменение от baseline, количество
`DONE/FAILED` и среднее время задачи из PostgreSQL. Management API используется
только для наблюдения.

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

| Режим | Очередь | Скорость source | Память source / broker | Результат |
|---|---|---:|---:|---|
| bounded, 1 worker | достигает 8 и стабилизируется | сходится к throughput worker | source ограничен; broker хранит до лимита | все задачи DONE |
| unbounded, 1 worker | растёт вместе с burst | близка к скорости генерации | source остаётся ограничен; backlog хранится в broker | все задачи позже DONE |
| bounded, 2 workers | тот же лимит, быстрее дренируется | throughput source растёт | source ограничен; broker хранит до лимита | все задачи DONE быстрее |

Фактические числа защиты берутся из `*.summary.json`; они зависят от CPU,
размера модели и числа задач. Смена `--scale worker=1` на `--scale worker=2`
не требует изменения `prefetch_count`, кода source или API.

Фактический прогон на одном и том же RGB-изображении MTSD, `queue_limit=8`,
`prefetch=2`, `concurrency=2`:

| Режим | Задач | max ready | max total | max unacked | source RSS | средняя latency | DONE |
|---|---:|---:|---:|---:|---:|---:|---:|
| bounded, 1 worker | 40 | 8 | 10 | 2 | 52.79 MiB | 5642 ms | 40 |
| bounded, 2 workers | 40 | 8 | 12 | 4 | 53.12 MiB | 4275 ms | 40 |
| unbounded, 1 worker | 40 | 38 | 40 | 2 | 52.50 MiB | 14088 ms | 40 |

Динамика во время трёх прогонов (source rate — число новых подтверждённых
публикаций за предыдущий пятисекундный интервал; latency — средняя от начала
прогона для уже завершённых задач):

| Время | bounded / 1 worker: ready, rate/s, DONE, latency | bounded / 2 workers: ready, rate/s, DONE, latency | unbounded / 1 worker: ready, rate/s, DONE, latency |
|---:|---|---|---|
| 5 s | 0, 2.17, 4, 3032 ms | 0, 3.39, 6, 3139 ms | 38, 7.62, 4, 3208 ms |
| 10 s | 8, 1.61, 12, 4953 ms | 8, 2.48, 18, 4413 ms | 30, 0, 12, 5838 ms |
| 15 s | 8, 1.67, 20, 5419 ms | 8, 1.99, 32, 4376 ms | 22, 0, 20, 8384 ms |
| 20 s | 8, 1.54, 28, 5553 ms | уже завершён | 12, 0, 30, 11340 ms |
| 25 s | 8, 0.40, 36, 5625 ms | уже завершён | 2, 0, 39, 13822 ms |

Пиковое queue memory составило 198432 B (bounded/1), 198240 B (bounded/2) и
143576 B (unbounded/1); максимум памяти узла RabbitMQ относительно baseline
внутри каждого прогона — 5226496 B, 3649536 B и 2957312 B соответственно.
RSS producer-а оставался стабильным около 53 MiB, поскольку источник создаёт
задачи лениво и не копит batch в памяти. Node heap зависит также от фоновой
активности брокера и в этих прогонах не вырос пропорционально ready depth.
Полная временная серия каждой метрики сохраняется в локальном JSONL при запуске
`mtsd-experiment`.

`max ready` — именно глубина готовой части очереди, ограниченная
`x-max-length`; `max total = ready + unacked`, поэтому в bounded-режиме
значение 10 не является нарушением лимита. Через добавление второго worker
совокупная конкурентность выросла с 2 до 4, average latency снизилась примерно
на 24%, а лимит очереди и код source не менялись. В unbounded-контроле очередь
приняла весь burst (40 сообщений), тогда как bounded-вариант остановил
ready-часть на 8 и заставил source ждать подтверждения. JSONL также содержит
`rabbit_node_memory_bytes` и `rabbit_node_memory_delta_bytes` для динамики памяти
узла RabbitMQ отдельно от RSS producer-а и per-queue memory.

Дополнительные защитные прогоны:

- burst из 30 HTTP-запросов дал `202=12`, `503=18`; все 503 содержали
  `Retry-After: 1`. В PostgreSQL не осталось записей с idempotency keys
  отказанных запросов;
- SIGTERM отправлен при двух unacked задачах у одного worker-а. До его
  перезапуска обе задачи были `DONE`, а RabbitMQ показал ready=0 и unacked=0;
- poison-задача из `poison-dlq.json` завершилась как `FAILED`, `attempts=2`,
  и увеличила ready-глубину `mtsd.tasks.dlq` на одно сообщение.

Те же три проверки можно повторить одной командой на запущенном compose-стеке.
Сценарии завершаются ненулевым кодом, если нарушены ожидания; итоговые JSON
отчёты печатаются в stdout. `api-burst` использует реальное RGB-изображение,
`sigterm` отправляет сигнал одному из worker-контейнеров во время batch.

```bash
IMAGE=/media/eugene/data21/mtsd/runs/parallel_b0_b1/kaggle_stage/dataset/images/mAU-BrDS6uvsqWDAaQKBGQ.jpg
export APP_DATABASE__URL=postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd
export APP_RABBIT__URL=amqp://guest:guest@127.0.0.1:5672/
uv run mtsd-protection-demo api-burst --count 30 --image "$IMAGE" \
  --report artifacts/hw4/api-overload.json
uv run mtsd-protection-demo poison --report artifacts/hw4/poison-dlq.json
APP_RABBIT__INFERENCE_DELAY_S=10 \
docker compose --profile hw4 up -d --scale worker=1
uv run mtsd-protection-demo sigterm --count 2 --image "$IMAGE" \
  --output artifacts/hw4/graceful-sigterm.jsonl \
  --report artifacts/hw4/graceful-sigterm-demo.json
```

Сценарий `api-burst` сверяет ответы 202/503 и `Retry-After`, затем ждёт
терминальных состояний принятых задач и проверяет отсутствие их `PENDING` и
`PROCESSING` в PostgreSQL. `poison` публикует заведомо некорректный base64,
проверяет итоговый статус, счётчик попыток и увеличение ready-глубины DLQ.
`sigterm` требует ровно один worker и batch из двух задач, равный его
конкурентности. Чтобы Management API успел отразить unacked доставки до
завершения короткого inference, worker запускается с `APP_RABBIT__INFERENCE_DELAY_S=10`.
Сценарий ждёт две unacked доставки при пустой ready-очереди, посылает SIGTERM и
до перезапуска проверяет graceful shutdown в логах, две завершённые обработки,
нулевые ready и unacked, а также две строки DONE в PostgreSQL. Затем контейнер
запускается снова.

Компактные итоговые артефакты нагрузочных запусков сохранены в репозитории:
`artifacts/hw4/bounded-1-worker-40.summary.json`,
`artifacts/hw4/bounded-2-workers-40.summary.json`,
`artifacts/hw4/unbounded-1-worker-final.summary.json` и
`artifacts/hw4/graceful-sigterm-script.summary.json`. Защитные проверки также
сохраняют отчёты в `api-overload.json`, `poison-dlq.json` и
`graceful-sigterm-demo.json`. Полные JSONL-трассы намеренно игнорируются Git,
чтобы не раздувать учебный репозиторий; их можно получить повторным запуском
приведённых команд.

## Проверки

```bash
uv run pytest -q
RUN_RABBIT_INTEGRATION=1 uv run pytest -q tests/test_task_integration.py -m integration
```

Первый запуск содержит модульные тесты worker через `TestRabbitBroker`; второй
поднимает настоящий RabbitMQ через Testcontainers и проверяет publisher-confirm
NACK при заполнении bounded queue.
