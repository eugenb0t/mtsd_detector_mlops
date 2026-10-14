"""RabbitMQ source: HTTP submission and streaming batch producer."""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from aiormq.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker
from pamqp import commands

from mtsd_detector.config import RabbitSettings, Settings, get_settings
from mtsd_detector.task_store import TaskStore
from mtsd_detector.task_types import (
    DetectionTask,
    TaskAccepted,
    TaskResult,
    TaskStatus,
    task_queue,
)

logger = logging.getLogger(__name__)


class PublishRejected(RuntimeError):
    """RabbitMQ publisher-confirm NACK caused by a full bounded queue."""


class TaskSource:
    """One RabbitMQ publisher with confirm-based backpressure."""

    def __init__(
        self,
        settings: RabbitSettings,
        store: TaskStore,
        broker: RabbitBroker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.queue = task_queue(
            max_length=settings.queue_max_length,
            name=settings.queue_name,
            bounded=settings.bounded_queue,
        )
        self.broker = broker or RabbitBroker(
            settings.url,
            graceful_timeout=settings.graceful_timeout_s,
        )
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self.broker.start()
            self._started = True

    async def close(self) -> None:
        if self._started:
            await self.broker.stop()
            self._started = False

    async def submit(self, task: DetectionTask) -> TaskAccepted:
        """Publish one task and wait for the broker's confirm frame.

        This method never retries: it is the HTTP boundary and must answer
        503 immediately when the queue rejects the message. The batch source
        uses ``submit_with_backpressure`` below and retries the same task only
        after a bounded delay, so it never builds an in-memory backlog.
        """
        created, existing = await self.store.create_publishing(task)
        if not created:
            if existing is None:
                raise RuntimeError("idempotency reservation disappeared")
            return self._accepted(existing)

        try:
            confirmation = await self.broker.publish(
                message=task.model_dump(mode="json"),
                queue=self.queue,
                mandatory=True,
                persist=True,
                message_id=task.task_id,
                headers={"x-retry-count": 0},
                timeout=self.settings.publish_confirm_timeout_s,
            )
            if not isinstance(confirmation, commands.Basic.Ack):
                raise PublishRejected(
                    f"RabbitMQ did not confirm task {task.task_id}: {confirmation!r}",
                )
            await self.store.mark_pending(task.task_id)
        except (PublishRejected, DeliveryError) as error:
            await self.store.remove_unpublished(task.task_id)
            if isinstance(error, DeliveryError):
                raise PublishRejected(str(error)) from error
            raise
        except Exception:
            # A mandatory publish can fail before a confirm (broker outage or
            # unroutable queue). It is safe to remove only PUBLISHING rows; if
            # RabbitMQ accepted an ambiguous message, the worker's task_id
            # idempotency key prevents a duplicate result.
            await self.store.remove_unpublished(task.task_id)
            raise
        return TaskAccepted(
            task_id=task.task_id,
            status=TaskStatus.PENDING,
            status_url=f"/api/v1/tasks/{task.task_id}",
            accepted_at=datetime.now(UTC),
        )

    async def submit_with_backpressure(self, task: DetectionTask) -> TaskAccepted:
        """Retry the same item after broker rejection, with no busy loop."""
        while True:
            try:
                return await self.submit(task)
            except PublishRejected:
                await asyncio.sleep(self.settings.effective_retry_after_s)

    async def run_batch(
        self,
        tasks: AsyncIterator[DetectionTask],
    ) -> int:
        """Consume one task at a time and adapt to publisher NACKs."""
        count = 0
        async for task in tasks:
            await self.submit_with_backpressure(task)
            count += 1
            logger.info("batch task accepted", extra={"task_id": task.task_id})
        return count

    @staticmethod
    def _accepted(existing: TaskResult) -> TaskAccepted:
        return TaskAccepted(
            task_id=existing.task_id,
            status=existing.status,
            status_url=f"/api/v1/tasks/{existing.task_id}",
            accepted_at=existing.updated_at,
        )


async def image_batch(
    image_path: Path,
    count: int,
    *,
    confidence: float = 0.05,
) -> AsyncIterator[DetectionTask]:
    """Generate tasks lazily; only one message payload is live at a time."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    for _ in range(count):
        yield DetectionTask(
            task_id=str(uuid.uuid4()),
            image_base64=encoded,
            confidence=confidence,
            submitted_at=datetime.now(UTC),
        )


async def run_source(settings: Settings, image_path: Path, count: int) -> int:
    """CLI entry point used by the reproducible batch experiment."""
    if settings.rabbit.url is None:
        raise RuntimeError("APP_RABBIT__URL is required for mtsd-source")
    from mtsd_detector.services.health import create_pool

    pool = await create_pool(
        dsn=settings.database.url,
        min_size=settings.database.pool_min_size,
        max_size=settings.database.pool_max_size,
        timeout=settings.database.connect_timeout_s,
    )
    store = TaskStore(pool)
    await store.initialize()
    source = TaskSource(settings.rabbit, store)
    await source.start()
    try:
        return await source.run_batch(image_batch(image_path, count))
    finally:
        await source.close()
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    settings = get_settings()
    print(asyncio.run(run_source(settings, args.image, args.count)))
