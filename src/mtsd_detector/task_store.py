"""PostgreSQL result store for idempotent asynchronous tasks."""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from mtsd_detector.task_types import DetectionTask, TaskResult, TaskStatus

TASK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detector_tasks (
    task_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    payload_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    result JSONB,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS detector_tasks_status_idx
    ON detector_tasks (status, updated_at);
"""


class TaskStore:
    """Small repository with database-level idempotency guarantees."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def initialize(self) -> None:
        await self.pool.execute(TASK_SCHEMA_SQL)

    async def create_publishing(
        self,
        task: DetectionTask,
    ) -> tuple[bool, TaskResult | None]:
        """Reserve a task before publish, but never reserve it as PENDING.

        The temporary PUBLISHING state lets the API roll the row back if
        RabbitMQ rejects a publish. A unique idempotency key makes concurrent
        HTTP retries converge on one task row.
        """
        async with self.pool.acquire() as connection:
            if task.idempotency_key is None:
                row = await connection.fetchrow(
                    """
                    INSERT INTO detector_tasks
                        (task_id, payload_digest, status)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (task_id) DO NOTHING
                    RETURNING *
                    """,
                    task.task_id,
                    task.payload_digest(),
                    TaskStatus.PUBLISHING.value,
                )
            else:
                row = await connection.fetchrow(
                    """
                    INSERT INTO detector_tasks
                        (task_id, idempotency_key, payload_digest, status)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    task.task_id,
                    task.idempotency_key,
                    task.payload_digest(),
                    TaskStatus.PUBLISHING.value,
                )
            if row is not None:
                return True, self._result_from_row(row)
            existing = await self._fetch_by_key(connection, task)
            return False, existing

    async def mark_pending(self, task_id: str) -> None:
        await self.pool.execute(
            """
            UPDATE detector_tasks
            SET status = $2, updated_at = NOW()
            WHERE task_id = $1 AND status = $3
            """,
            task_id,
            TaskStatus.PENDING.value,
            TaskStatus.PUBLISHING.value,
        )

    async def remove_unpublished(self, task_id: str) -> None:
        """Remove only a row that was never confirmed by RabbitMQ."""
        await self.pool.execute(
            "DELETE FROM detector_tasks WHERE task_id = $1 AND status = $2",
            task_id,
            TaskStatus.PUBLISHING.value,
        )

    async def get(self, task_id: str) -> TaskResult | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM detector_tasks WHERE task_id = $1",
            task_id,
        )
        return None if row is None else self._result_from_row(row)

    async def mark_processing(self, task_id: str, attempt: int) -> None:
        await self.pool.execute(
            """
            UPDATE detector_tasks
            SET status = $2, attempts = GREATEST(attempts, $3), updated_at = NOW()
            WHERE task_id = $1 AND status <> $4
            """,
            task_id,
            TaskStatus.PROCESSING.value,
            attempt,
            TaskStatus.DONE.value,
        )

    async def complete(
        self,
        task: DetectionTask,
        result: dict[str, Any],
        attempt: int,
    ) -> None:
        """Insert or update exactly one durable result for a task id."""
        await self.pool.execute(
            """
            INSERT INTO detector_tasks
                (task_id, idempotency_key, payload_digest, status, result,
                 attempts, completed_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, NOW())
            ON CONFLICT (task_id) DO UPDATE SET
                status = EXCLUDED.status,
                result = EXCLUDED.result,
                attempts = GREATEST(detector_tasks.attempts, EXCLUDED.attempts),
                updated_at = NOW(),
                completed_at = NOW()
            WHERE detector_tasks.status <> 'DONE'
            """,
            task.task_id,
            task.idempotency_key,
            task.payload_digest(),
            TaskStatus.DONE.value,
            json.dumps(result),
            attempt,
        )

    async def fail(
        self,
        task: DetectionTask,
        error: str,
        attempt: int,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO detector_tasks
                (task_id, idempotency_key, payload_digest, status, error, attempts)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (task_id) DO UPDATE SET
                status = EXCLUDED.status,
                error = EXCLUDED.error,
                attempts = GREATEST(detector_tasks.attempts, EXCLUDED.attempts),
                updated_at = NOW()
            WHERE detector_tasks.status <> 'DONE'
            """,
            task.task_id,
            task.idempotency_key,
            task.payload_digest(),
            TaskStatus.FAILED.value,
            error[:2000],
            attempt,
        )

    async def _fetch_by_key(
        self,
        connection: asyncpg.Connection,
        task: DetectionTask,
    ) -> TaskResult | None:
        if task.idempotency_key is None:
            row = await connection.fetchrow(
                "SELECT * FROM detector_tasks WHERE task_id = $1",
                task.task_id,
            )
        else:
            row = await connection.fetchrow(
                "SELECT * FROM detector_tasks WHERE idempotency_key = $1",
                task.idempotency_key,
            )
        return None if row is None else self._result_from_row(row)

    @staticmethod
    def _result_from_row(row: asyncpg.Record) -> TaskResult:
        result = row["result"]
        if isinstance(result, str):
            result = json.loads(result)
        return TaskResult(
            task_id=row["task_id"],
            status=TaskStatus(row["status"]),
            result=result,
            error=row["error"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
