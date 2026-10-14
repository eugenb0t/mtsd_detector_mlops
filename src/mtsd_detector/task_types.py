"""Shared task contracts and RabbitMQ topology for the HW4 pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import RabbitExchange, RabbitQueue
from pydantic import BaseModel, Field

TASK_EXCHANGE_NAME = "mtsd.tasks"
TASK_QUEUE_NAME = "mtsd.tasks.pending"
TASK_DLQ_NAME = "mtsd.tasks.dlq"


class TaskStatus(StrEnum):
    """Durable states stored in PostgreSQL."""

    PUBLISHING = "PUBLISHING"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class DetectionTask(BaseModel):
    """Message sent through RabbitMQ.

    The payload is deliberately self-contained so the worker can process it
    after the HTTP request has finished. ``task_id`` is also the idempotency
    key used by the result store.
    """

    task_id: str = Field(min_length=8, max_length=128)
    image_base64: str = Field(min_length=32)
    confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=50, ge=1, le=200)
    submitted_at: datetime
    idempotency_key: str | None = Field(default=None, max_length=256)

    def payload_digest(self) -> str:
        """Stable digest for audit and safe duplicate handling."""
        body = self.model_dump(mode="json", exclude={"submitted_at"})
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()


class TaskSubmission(BaseModel):
    """HTTP payload accepted by the external API boundary."""

    image_base64: str = Field(min_length=32)
    confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=50, ge=1, le=200)
    idempotency_key: str | None = Field(default=None, max_length=256)

    def to_task(self) -> DetectionTask:
        import uuid

        return DetectionTask(
            task_id=str(uuid.uuid4()),
            image_base64=self.image_base64,
            confidence=self.confidence,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            submitted_at=datetime.now(UTC),
            idempotency_key=self.idempotency_key,
        )


class TaskAccepted(BaseModel):
    """Immediate response returned after publisher confirmation."""

    task_id: str
    status: TaskStatus
    status_url: str
    accepted_at: datetime


class TaskResult(BaseModel):
    """Result returned by the asynchronous task status endpoint."""

    task_id: str
    status: TaskStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


def task_queue(
    *,
    max_length: int,
    name: str = TASK_QUEUE_NAME,
    bounded: bool = True,
) -> RabbitQueue:
    """Declare a bounded classic queue whose overflow is rejected to producer."""
    arguments: dict[str, Any] = {}
    if bounded:
        arguments = {
            "x-max-length": max_length,
            # The producer receives a publisher-confirm NACK instead of silently
            # dropping the oldest task. That NACK is the source backpressure.
            "x-overflow": "reject-publish",
        }
    return RabbitQueue(
        name,
        durable=True,
        arguments=arguments,
    )


def dead_letter_queue() -> RabbitQueue:
    """Durable queue for poison messages after bounded retries."""
    return RabbitQueue(TASK_DLQ_NAME, durable=True)


def task_exchange() -> RabbitExchange:
    """Durable direct exchange used by the task and DLQ publishers."""
    return RabbitExchange(TASK_EXCHANGE_NAME, durable=True)


ACK_POLICY = AckPolicy.MANUAL
