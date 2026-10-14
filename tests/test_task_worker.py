"""Unit tests for the FastStream worker using the in-memory Rabbit harness."""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from faststream.rabbit import RabbitBroker
from faststream.rabbit.testing import TestRabbitBroker
from pamqp import commands

from mtsd_detector.config import (
    DatabaseSettings,
    MLflowSettings,
    RabbitSettings,
    Settings,
)
from mtsd_detector.services.model import LoadedModel
from mtsd_detector.task_types import DetectionTask, task_queue
from mtsd_detector.task_worker import build_worker


class FakeStore:
    def __init__(self) -> None:
        self.done: set[str] = set()
        self.completed: list[str] = []
        self.failed: list[str] = []

    async def get(self, task_id: str) -> Any:
        if task_id not in self.done:
            return None
        return type(
            "Record",
            (),
            {"status": type("Status", (), {"value": "DONE"})()},
        )()

    async def mark_processing(self, task_id: str, attempt: int) -> None:
        del task_id, attempt

    async def complete(
        self,
        task: DetectionTask,
        result: dict[str, Any],
        attempt: int,
    ) -> None:
        del result, attempt
        self.done.add(task.task_id)
        self.completed.append(task.task_id)

    async def fail(self, task: DetectionTask, error: str, attempt: int) -> None:
        del error, attempt
        self.failed.append(task.task_id)


class FakeModel:
    model_uri = "models:/mtsd20-detector@champion"

    async def detect_async(
        self,
        image_bytes: bytes,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> tuple[list[dict[str, Any]], tuple[int, int], float]:
        await asyncio.sleep(0.01)
        assert image_bytes == b"image" * 8
        assert confidence == 0.05
        assert iou_threshold == 0.5
        assert max_detections == 50
        return [], (1, 1), 2.0


class FailingModel(FakeModel):
    async def detect_async(
        self,
        image_bytes: bytes,
        confidence: float,
        iou_threshold: float,
        max_detections: int,
    ) -> tuple[list[dict[str, Any]], tuple[int, int], float]:
        del image_bytes, confidence, iou_threshold, max_detections
        raise RuntimeError("synthetic inference failure")


class FakePublisher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def publish(self, **kwargs: Any) -> commands.Basic.Ack:
        self.calls.append(kwargs)
        return commands.Basic.Ack(delivery_tag=1)


class FakeMessage:
    def __init__(self, headers: dict[str, Any] | None = None) -> None:
        self.headers = headers or {}
        self.message_id = "malformed-message"
        self.ack = AsyncMock()


def _settings() -> Settings:
    return Settings(
        database=DatabaseSettings(url="postgresql://unused"),
        mlflow=MLflowSettings(tracking_uri="http://mlflow"),
        rabbit=RabbitSettings(
            url="amqp://unused",
            queue_max_length=2,
            prefetch_count=1,
            worker_concurrency=1,
            inference_delay_s=0,
        ),
    )


@pytest.mark.asyncio
async def test_worker_acks_after_idempotent_result_with_test_rabbit_broker() -> None:
    broker = RabbitBroker()
    store = FakeStore()
    build_worker(_settings(), store=store, model=FakeModel(), broker=broker)
    task = DetectionTask(
        task_id="task-success-1",
        image_base64=base64.b64encode(b"image" * 8).decode(),
        submitted_at=datetime.now(UTC),
    )

    async with TestRabbitBroker(broker):
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        await broker.publish(
            task.model_dump(mode="json"),
            queue=task_queue(max_length=2),
        )
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task

    assert store.completed == [task.task_id]
    assert store.failed == []
    assert ticks > 0


@pytest.mark.asyncio
async def test_loaded_model_moves_blocking_inference_off_event_loop() -> None:
    class BlockingDetector:
        def predict(self, _payload: Any) -> list[dict[str, Any]]:
            time.sleep(0.05)
            return [{"detections": [], "image_width": 1, "image_height": 1}]

    model = LoadedModel(
        model=BlockingDetector(),
        model_uri="models:/mtsd20-detector@champion",
        model_version="1",
        loaded_at="2026-01-01T00:00:00+00:00",
    )
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(heartbeat())
    result = await model.detect_async(b"image", 0.05, 0.5, 50)
    heartbeat_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await heartbeat_task

    assert result["detections"] == []
    assert ticks > 10


@pytest.mark.asyncio
async def test_worker_rejects_prefetch_not_equal_to_concurrency() -> None:
    settings = _settings()
    settings.rabbit.prefetch_count = 2
    settings.rabbit.worker_concurrency = 1
    with pytest.raises(ValueError, match="prefetch_count"):
        build_worker(
            settings,
            store=FakeStore(),
            model=FakeModel(),
            broker=RabbitBroker(),
        )


@pytest.mark.asyncio
async def test_worker_routes_exhausted_failure_to_dlq() -> None:
    from mtsd_detector.task_worker import WorkerRuntime

    settings = _settings()
    settings.rabbit.max_retries = 0
    publisher = FakePublisher()
    store = FakeStore()
    runtime = WorkerRuntime(
        settings,
        store=store,
        model=FailingModel(),
        publisher=publisher,
    )
    task = DetectionTask(
        task_id="task-failure-1",
        image_base64=base64.b64encode(b"image" * 8).decode(),
        submitted_at=datetime.now(UTC),
    )
    message = FakeMessage()

    await runtime.handle(task.model_dump(mode="json"), message)  # type: ignore[arg-type]

    assert publisher.calls[0]["queue"].name == "mtsd.tasks.dlq"
    assert store.failed == [task.task_id]
    message.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_republishes_transient_failure_with_incremented_attempt() -> None:
    from mtsd_detector.task_worker import WorkerRuntime

    settings = _settings()
    settings.rabbit.max_retries = 1
    settings.rabbit.retry_backoff_s = 0
    publisher = FakePublisher()
    runtime = WorkerRuntime(
        settings,
        store=FakeStore(),
        model=FailingModel(),
        publisher=publisher,
    )
    task = DetectionTask(
        task_id="task-retry-1",
        image_base64=base64.b64encode(b"image" * 8).decode(),
        submitted_at=datetime.now(UTC),
    )
    message = FakeMessage()

    await runtime.handle(task.model_dump(mode="json"), message)  # type: ignore[arg-type]

    assert publisher.calls[0]["queue"].name == "mtsd.tasks.pending"
    assert publisher.calls[0]["headers"] == {"x-retry-count": 1}
    message.ack.assert_awaited_once()
