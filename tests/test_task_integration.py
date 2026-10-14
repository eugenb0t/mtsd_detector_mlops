"""RabbitMQ integration test for publisher-confirm overflow backpressure."""

from __future__ import annotations

import base64
import os
import uuid
from datetime import UTC, datetime

import aio_pika
import pytest
from testcontainers.community.rabbitmq import RabbitMqContainer

from mtsd_detector.config import RabbitSettings
from mtsd_detector.task_source import PublishRejected, TaskSource
from mtsd_detector.task_types import DetectionTask


class MemoryStore:
    async def create_publishing(self, task: DetectionTask) -> tuple[bool, None]:
        return True, None

    async def mark_pending(self, task_id: str) -> None:
        del task_id

    async def remove_unpublished(self, task_id: str) -> None:
        del task_id


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_RABBIT_INTEGRATION") != "1",
    reason="set RUN_RABBIT_INTEGRATION=1 to start Testcontainers",
)
@pytest.mark.asyncio
async def test_real_rabbitmq_rejects_publish_when_bounded_queue_is_full() -> None:
    with RabbitMqContainer("rabbitmq:3.13-alpine") as rabbit:
        host = rabbit.get_container_host_ip()
        port = rabbit.get_exposed_port(5672)
        url = f"amqp://guest:guest@{host}:{port}/"
        connection = await aio_pika.connect_robust(url)
        channel = await connection.channel(publisher_confirms=True)
        await channel.declare_queue(
            "mtsd.tasks.pending",
            durable=True,
            arguments={
                "x-max-length": 1,
                "x-overflow": "reject-publish",
            },
        )
        await connection.close()

        settings = RabbitSettings(
            url=url,
            queue_max_length=1,
            publish_confirm_timeout_s=3,
        )
        source = TaskSource(settings, MemoryStore())
        await source.start()
        try:
            for index in range(2):
                task = DetectionTask(
                    task_id=f"integration-{uuid.uuid4().hex}-{index}",
                    image_base64=base64.b64encode(b"image" * 8).decode(),
                    submitted_at=datetime.now(UTC),
                )
                if index == 0:
                    await source.submit(task)
                else:
                    with pytest.raises(PublishRejected):
                        await source.submit(task)
        finally:
            await source.close()
