"""FastStream worker for RabbitMQ-backed MTSD inference tasks."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from aiormq.exceptions import DeliveryError
from faststream import FastStream
from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import RabbitBroker, RabbitMessage

from mtsd_detector.config import Settings, get_settings
from mtsd_detector.services.health import create_pool
from mtsd_detector.services.model import LoadedModel, load_registry_model
from mtsd_detector.task_source import PublishRejected
from mtsd_detector.task_store import TaskStore
from mtsd_detector.task_types import (
    DetectionTask,
    dead_letter_queue,
    task_queue,
)

logger = logging.getLogger(__name__)


class PrefetchRabbitBroker(RabbitBroker):
    """Rabbit broker that applies AMQP QoS before subscribers start."""

    def __init__(self, *args: Any, prefetch_count: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefetch_count = prefetch_count

    async def _connect(self) -> Any:
        connection = await super()._connect()
        if self._channel is None:  # pragma: no cover - guarded by RabbitBroker
            raise RuntimeError("RabbitMQ channel was not created")
        await self._channel.set_qos(prefetch_count=self.prefetch_count)
        return connection


class WorkerRuntime:
    """State owned by one worker process and initialized exactly once."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: TaskStore | None = None,
        model: LoadedModel | None = None,
        publisher: Any = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.model = model
        self.publisher = publisher
        self.semaphore = asyncio.Semaphore(settings.rabbit.worker_concurrency)
        self.pool: Any = None

    async def initialize(self) -> None:
        """Open dependencies and load the MLflow alias before consuming."""
        if self.settings.rabbit.url is None:
            raise RuntimeError("APP_RABBIT__URL is required for mtsd-worker")
        if self.store is None:
            self.pool = await create_pool(
                dsn=self.settings.database.url,
                min_size=self.settings.database.pool_min_size,
                max_size=self.settings.database.pool_max_size,
                timeout=self.settings.database.connect_timeout_s,
            )
            self.store = TaskStore(self.pool)
            await self.store.initialize()
        if self.model is None:
            self.model = load_registry_model(
                tracking_uri=self.settings.mlflow.tracking_uri,
                model_name=self.settings.mlflow.model_name,
                model_alias=self.settings.mlflow.model_alias,
            )
        if self.model is None:
            raise RuntimeError("worker could not load model from MLflow alias")
        logger.info(
            "worker initialized",
            extra={
                "model_uri": self.model.model_uri,
                "queue": self.settings.rabbit.queue_max_length,
                "prefetch_count": self.settings.rabbit.prefetch_count,
                "worker_concurrency": self.settings.rabbit.worker_concurrency,
            },
        )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def handle(
        self,
        raw_payload: dict[str, Any],
        message: RabbitMessage,
    ) -> None:
        """Process once, persist before ACK, or route after bounded retries."""
        async with self.semaphore:
            attempt = int(message.headers.get("x-retry-count", 0))
            task_id = str(raw_payload.get("task_id", message.message_id))
            try:
                task = DetectionTask.model_validate(raw_payload)
            except Exception as error:
                await self._dead_letter(
                    raw_payload,
                    message,
                    task_id=task_id,
                    error=f"invalid task: {error}",
                    attempt=attempt,
                )
                return

            assert self.store is not None
            existing = await self.store.get(task.task_id)
            if existing is not None and existing.status.value == "DONE":
                # Redelivery after a worker crash is harmless: the durable
                # result wins and no second result row can be created.
                await message.ack()
                return

            await self.store.mark_processing(task.task_id, attempt)
            try:
                if self.settings.rabbit.inference_delay_s:
                    await asyncio.sleep(self.settings.rabbit.inference_delay_s)
                assert self.model is not None
                inference_result = await self.model.detect_async(
                    _decode_image(task.image_base64),
                    task.confidence,
                    task.iou_threshold,
                    task.max_detections,
                )
                if isinstance(inference_result, dict):
                    result = inference_result
                else:
                    # Keep the worker usable with small test doubles that
                    # return the pre-existing detector tuple contract.
                    detections, image_size, inference_ms = inference_result
                    result = {
                        "detections": detections,
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "inference_ms": inference_ms,
                    }
                await self.store.complete(
                    task,
                    result,
                    attempt,
                )
                # ACK is deliberately after the durable result write.
                await message.ack()
            except Exception as error:
                await self._handle_failure(task, raw_payload, message, error, attempt)

    async def _handle_failure(
        self,
        task: DetectionTask,
        raw_payload: dict[str, Any],
        message: RabbitMessage,
        error: Exception,
        attempt: int,
    ) -> None:
        if attempt < self.settings.rabbit.max_retries:
            for handoff_attempt in range(self.settings.rabbit.max_retries + 1):
                if self.settings.rabbit.retry_backoff_s:
                    await asyncio.sleep(
                        self.settings.rabbit.retry_backoff_s
                        * (2 ** (attempt + handoff_attempt)),
                    )
                try:
                    await self._publish_confirmed(
                        raw_payload,
                        queue=task_queue(
                            max_length=self.settings.rabbit.queue_max_length,
                            name=self.settings.rabbit.queue_name,
                            bounded=self.settings.rabbit.bounded_queue,
                        ),
                        headers={"x-retry-count": attempt + 1},
                        message_id=task.task_id,
                    )
                    await message.ack()
                    return
                except PublishRejected as publish_error:
                    logger.warning(
                        "retry publish rejected",
                        extra={
                            "task_id": task.task_id,
                            "handoff_attempt": handoff_attempt + 1,
                        },
                    )
                    if handoff_attempt == self.settings.rabbit.max_retries:
                        await self._dead_letter(
                            raw_payload,
                            message,
                            task_id=task.task_id,
                            error=(
                                f"retry handoff rejected after bounded attempts: "
                                f"{publish_error}; inference error: {error}"
                            ),
                            attempt=attempt,
                            task=task,
                        )
                        return
        await self._dead_letter(
            raw_payload,
            message,
            task_id=task.task_id,
            error=str(error),
            attempt=attempt,
            task=task,
        )

    async def _dead_letter(
        self,
        raw_payload: dict[str, Any],
        message: RabbitMessage,
        *,
        task_id: str,
        error: str,
        attempt: int,
        task: DetectionTask | None = None,
    ) -> None:
        await self._publish_confirmed(
            raw_payload,
            queue=dead_letter_queue(),
            headers={"x-error": error[:1000], "x-retry-count": attempt},
            message_id=task_id,
        )
        if task is not None and self.store is not None:
            await self.store.fail(task, error, attempt)
        # The DLQ copy is durable before the source delivery is acknowledged.
        await message.ack()

    async def _publish_confirmed(
        self,
        payload: dict[str, Any],
        *,
        queue: Any,
        headers: dict[str, Any],
        message_id: str,
    ) -> None:
        try:
            confirmation = await self.publisher.publish(
                message=payload,
                queue=queue,
                mandatory=True,
                persist=True,
                message_id=message_id,
                headers=headers,
                timeout=self.settings.rabbit.publish_confirm_timeout_s,
            )
        except DeliveryError as error:
            raise PublishRejected(str(error)) from error
        from pamqp import commands

        if not isinstance(confirmation, commands.Basic.Ack):
            raise PublishRejected(f"publish was not confirmed: {confirmation!r}")


def _decode_image(value: str) -> bytes:
    import base64

    return base64.b64decode(value, validate=True)


def build_worker(
    settings: Settings,
    *,
    store: TaskStore | None = None,
    model: LoadedModel | None = None,
    broker: RabbitBroker | None = None,
) -> tuple[FastStream, WorkerRuntime]:
    """Build a FastStream application, also convenient for TestRabbitBroker."""
    if settings.rabbit.prefetch_count != settings.rabbit.worker_concurrency:
        raise ValueError(
            "prefetch_count must equal worker_concurrency: each delivery occupies "
            "one executor slot and one unacked slot",
        )
    if broker is None:
        if settings.rabbit.url is None:
            raise RuntimeError("APP_RABBIT__URL is required for mtsd-worker")
        broker = PrefetchRabbitBroker(
            settings.rabbit.url,
            graceful_timeout=settings.rabbit.graceful_timeout_s,
            prefetch_count=settings.rabbit.prefetch_count,
        )
    runtime = WorkerRuntime(settings, store=store, model=model)
    # Also set this eagerly for TestRabbitBroker, which starts subscribers
    # without executing FastStream application hooks.
    runtime.publisher = broker
    app = FastStream(broker)

    @app.on_startup
    async def startup() -> None:
        await runtime.initialize()
        runtime.publisher = broker

    @app.after_shutdown
    async def shutdown() -> None:
        await runtime.close()

    @app.after_startup
    async def declare_dlq() -> None:
        await broker.declare_queue(dead_letter_queue())

    @broker.subscriber(
        task_queue(
            max_length=settings.rabbit.queue_max_length,
            name=settings.rabbit.queue_name,
            bounded=settings.rabbit.bounded_queue,
        ),
        ack_policy=AckPolicy.MANUAL,
    )
    async def process_task(
        payload: dict[str, Any],
        message: RabbitMessage,
    ) -> None:
        await runtime.handle(payload, message)

    return app, runtime


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default=None)
    parser.parse_args()
    settings = get_settings()
    app, _ = build_worker(settings)
    asyncio.run(app.run())
