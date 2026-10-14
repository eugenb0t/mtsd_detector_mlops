"""Measure bounded versus unbounded RabbitMQ task flow for the HW4 report."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import resource
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mtsd_detector.config import Settings, get_settings
from mtsd_detector.services.health import create_pool
from mtsd_detector.task_source import TaskSource
from mtsd_detector.task_store import TaskStore
from mtsd_detector.task_types import DetectionTask


def _management_snapshot(
    base_url: str,
    user: str,
    password: str,
    queue_name: str,
) -> dict[str, Any]:
    encoded = urllib.parse.quote(queue_name, safe="")
    url = f"{base_url.rstrip('/')}/api/queues/%2F/{encoded}"
    request = urllib.request.Request(url)
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
    request.add_header("Authorization", f"Basic {credentials}")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            body = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError):
        return {
            "messages": None,
            "messages_ready": None,
            "messages_unacknowledged": None,
            "memory": None,
            "consumers": None,
        }
    return {
        "messages": body.get("messages"),
        "messages_ready": body.get("messages_ready"),
        "messages_unacknowledged": body.get("messages_unacknowledged"),
        "memory": body.get("memory"),
        "consumers": body.get("consumers"),
    }


async def _db_snapshot(pool: Any, prefix: str) -> dict[str, Any]:
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'DONE') AS done,
            COUNT(*) FILTER (WHERE status = 'FAILED') AS failed,
            COUNT(*) AS total,
            AVG(EXTRACT(EPOCH FROM (updated_at - created_at)) * 1000)
                FILTER (WHERE status IN ('DONE', 'FAILED')) AS avg_latency_ms
        FROM detector_tasks
        WHERE task_id LIKE $1
        """,
        f"{prefix}%",
    )
    return {
        "done": int(row["done"] or 0),
        "failed": int(row["failed"] or 0),
        "total": int(row["total"] or 0),
        "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
    }


async def run_experiment(
    settings: Settings,
    image_path: Path,
    count: int,
    output: Path,
    management_url: str,
    management_user: str,
    management_password: str,
    sample_interval_s: float,
    drain_timeout_s: float,
) -> dict[str, Any]:
    """Produce tasks as fast as confirms allow and record dynamics."""
    if settings.rabbit.url is None:
        raise RuntimeError("APP_RABBIT__URL is required for mtsd-experiment")
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prefix = f"hw4-{uuid.uuid4().hex[:12]}-"
    started = time.perf_counter()
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
    output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    accepted = 0
    producer_done = asyncio.Event()

    async def produce() -> None:
        nonlocal accepted
        try:
            for index in range(count):
                task = DetectionTask(
                    task_id=f"{prefix}{index}",
                    image_base64=image_base64,
                    submitted_at=datetime.now(UTC),
                )
                await source.submit_with_backpressure(task)
                accepted += 1
        finally:
            producer_done.set()

    async def monitor() -> None:
        previous_accepted = 0
        previous_time = started
        while True:
            now = time.perf_counter()
            queue = await asyncio.to_thread(
                _management_snapshot,
                management_url,
                management_user,
                management_password,
                settings.rabbit.queue_name,
            )
            database = await _db_snapshot(pool, prefix)
            rate = (accepted - previous_accepted) / max(0.001, now - previous_time)
            sample = {
                "elapsed_s": round(now - started, 3),
                # RabbitMQ reports `messages` as ready + unacknowledged. The
                # configured x-max-length bounds ready messages; unacked
                # deliveries are separately bounded by prefetch_count.
                "queue_depth": queue["messages_ready"],
                "queue_total_messages": queue["messages"],
                "queue_unacknowledged": queue["messages_unacknowledged"],
                "queue_memory_bytes": queue["memory"],
                "consumers": queue["consumers"],
                "source_accepted_total": accepted,
                "source_rate_tasks_s": round(rate, 3),
                "source_rss_mb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    3,
                ),
                **database,
            }
            samples.append(sample)
            previous_accepted, previous_time = accepted, now
            if (
                producer_done.is_set()
                and database["done"] + database["failed"] >= count
            ):
                return
            if now - started >= drain_timeout_s:
                return
            await asyncio.sleep(sample_interval_s)

    try:
        await asyncio.gather(produce(), monitor())
    finally:
        await source.close()
        await pool.close()

    final = samples[-1] if samples else {}
    summary = {
        "mode": "bounded" if settings.rabbit.bounded_queue else "unbounded",
        "queue_name": settings.rabbit.queue_name,
        "queue_limit": settings.rabbit.queue_max_length
        if settings.rabbit.bounded_queue
        else None,
        "prefetch_count": settings.rabbit.prefetch_count,
        "worker_concurrency_per_process": settings.rabbit.worker_concurrency,
        "tasks_requested": count,
        "tasks_accepted": accepted,
        "max_queue_depth": max(
            (item["queue_depth"] or 0 for item in samples),
            default=0,
        ),
        "max_queue_total_messages": max(
            (item["queue_total_messages"] or 0 for item in samples),
            default=0,
        ),
        "max_queue_unacknowledged": max(
            (item["queue_unacknowledged"] or 0 for item in samples),
            default=0,
        ),
        "max_queue_memory_bytes": max(
            (item["queue_memory_bytes"] or 0 for item in samples),
            default=0,
        ),
        "max_source_rss_mb": max(
            (item["source_rss_mb"] for item in samples),
            default=0.0,
        ),
        "final_done": final.get("done", 0),
        "final_failed": final.get("failed", 0),
        "final_avg_task_latency_ms": final.get("avg_latency_ms", 0.0),
        "observed_consumers": final.get("consumers"),
        "samples": len(samples),
    }
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in samples),
        encoding="utf-8",
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hw4/metrics.jsonl"),
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:15672")
    parser.add_argument("--management-user", default="guest")
    parser.add_argument("--management-password", default="guest")
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument("--drain-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    summary = asyncio.run(
        run_experiment(
            get_settings(),
            args.image,
            args.count,
            args.output,
            args.management_url,
            args.management_user,
            args.management_password,
            args.sample_interval_s,
            args.drain_timeout_s,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
