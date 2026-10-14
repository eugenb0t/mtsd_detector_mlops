"""Reproducible live demonstrations for HW4 reliability requirements."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aio_pika

from mtsd_detector.config import Settings, get_settings
from mtsd_detector.services.health import create_pool


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        payload = json.loads(response.read())
        return response.status, payload, headers


def _management_json(settings: Settings, resource: str) -> Any:
    base_url = os.getenv("HW4_MANAGEMENT_URL", "http://127.0.0.1:15672")
    user = os.getenv("HW4_MANAGEMENT_USER", "guest")
    password = os.getenv("HW4_MANAGEMENT_PASSWORD", "guest")
    request = urllib.request.Request(f"{base_url.rstrip('/')}{resource}")
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    request.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _quote_queue(queue: str) -> str:
    return urllib.parse.quote(queue, safe="")


async def _api_burst(
    settings: Settings,
    count: int,
    image: Path,
) -> dict[str, Any]:
    image_base64 = base64.b64encode(image.read_bytes()).decode()
    endpoint = os.getenv("HW4_API_URL", "http://127.0.0.1:8000")

    def submit(
        index: int,
    ) -> tuple[str, tuple[int, dict[str, Any], dict[str, str]]]:
        idempotency_key = f"hw4-demo-{uuid.uuid4().hex}-{index}"
        return idempotency_key, _http_json(
            f"{endpoint.rstrip('/')}/api/v1/tasks",
            method="POST",
            body={
                "image_base64": image_base64,
                "idempotency_key": idempotency_key,
            },
            timeout_s=15,
        )

    with ThreadPoolExecutor(max_workers=count) as executor:
        requests = list(executor.map(submit, range(count)))
    idempotency_keys = [key for key, _ in requests]
    responses = [response for _, response in requests]
    rejected_idempotency_keys = [
        key for key, (status, _, _) in requests if status == 503
    ]
    accepted_ids = [
        payload["task_id"] for status, payload, _ in responses if status == 202
    ]
    rejected = [
        headers.get("retry-after") for status, _, headers in responses if status == 503
    ]
    statuses: dict[str, int] = {}
    for status, _, _ in responses:
        statuses[str(status)] = statuses.get(str(status), 0) + 1

    deadline = time.monotonic() + 90
    while accepted_ids and time.monotonic() < deadline:
        states = await asyncio.gather(
            *[
                asyncio.to_thread(
                    _http_json,
                    f"{endpoint.rstrip('/')}/api/v1/tasks/{task_id}",
                )
                for task_id in accepted_ids
            ],
        )
        if all(
            status == 200 and payload["status"] in {"DONE", "FAILED"}
            for status, payload, _ in states
        ):
            break
        await asyncio.sleep(0.25)

    pool = await create_pool(
        dsn=settings.database.url,
        min_size=1,
        max_size=2,
        timeout=settings.database.connect_timeout_s,
    )
    try:
        records = await pool.fetch(
            """
            SELECT task_id, idempotency_key, status FROM detector_tasks
            WHERE idempotency_key = ANY($1::text[])
            ORDER BY task_id
            """,
            idempotency_keys,
        )
        state_counts: dict[str, int] = {}
        rejected_rows = 0
        for record in records:
            state_counts[record["status"]] = state_counts.get(record["status"], 0) + 1
            if record["idempotency_key"] in rejected_idempotency_keys:
                rejected_rows += 1
    finally:
        await pool.close()

    report = {
        "requests": count,
        "http_statuses": statuses,
        "retry_after_values_for_503": rejected,
        "accepted_tasks": len(accepted_ids),
        "accepted_task_states_after_drain": state_counts,
        "pending_or_processing_after_drain": sum(
            state_counts.get(state, 0) for state in ("PENDING", "PROCESSING")
        ),
        "rejected_requests_with_database_rows": rejected_rows,
    }
    if statuses.get("202", 0) == 0 or statuses.get("503", 0) == 0:
        raise RuntimeError(
            f"burst did not exercise both acceptance and shedding: {report}"
        )
    if statuses.get("503", 0) and any(value is None for value in rejected):
        raise RuntimeError("one or more 503 responses did not include Retry-After")
    if report["pending_or_processing_after_drain"] != 0:
        raise RuntimeError("accepted task rows did not drain to a terminal state")
    if rejected_rows != 0:
        raise RuntimeError("rejected requests left rows in the task database")
    return report


async def _poison_message(settings: Settings) -> dict[str, Any]:
    if settings.rabbit.url is None:
        raise RuntimeError("APP_RABBIT__URL is required")
    task_id = f"poison-demo-{uuid.uuid4().hex}"
    queue_path = f"/api/queues/%2F/{_quote_queue('mtsd.tasks.dlq')}"
    before = await asyncio.to_thread(_management_json, settings, queue_path)
    connection = await aio_pika.connect_robust(settings.rabbit.url)
    try:
        channel = await connection.channel(publisher_confirms=True)
        message = aio_pika.Message(
            body=json.dumps(
                {
                    "task_id": task_id,
                    "image_base64": "!" * 32,
                    "submitted_at": datetime.now(UTC).isoformat(),
                },
            ).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=task_id,
        )
        await channel.default_exchange.publish(
            message,
            routing_key=settings.rabbit.queue_name,
            mandatory=True,
        )
    finally:
        await connection.close()

    pool = await create_pool(
        dsn=settings.database.url,
        min_size=1,
        max_size=2,
        timeout=settings.database.connect_timeout_s,
    )
    deadline = time.monotonic() + 60
    row = None
    current_depth = int(before.get("messages_ready", 0))
    try:
        while time.monotonic() < deadline:
            row = await pool.fetchrow(
                "SELECT status, attempts, error FROM detector_tasks WHERE task_id=$1",
                task_id,
            )
            current = await asyncio.to_thread(
                _management_json,
                settings,
                queue_path,
            )
            current_depth = int(current.get("messages_ready", 0))
            if (
                row is not None
                and row["status"] == "FAILED"
                and current_depth > int(before.get("messages_ready", 0))
            ):
                break
            await asyncio.sleep(0.5)
    finally:
        await pool.close()

    report = {
        "task_id": task_id,
        "status": None if row is None else row["status"],
        "attempts": None if row is None else row["attempts"],
        "error": None if row is None else row["error"],
        "dlq_ready_before": int(before.get("messages_ready", 0)),
        "dlq_ready_after": current_depth,
    }
    if row is None or row["status"] != "FAILED":
        raise RuntimeError(f"poison task did not reach FAILED: {report}")
    if int(row["attempts"]) != settings.rabbit.max_retries:
        raise RuntimeError(f"unexpected retry count for poison task: {report}")
    if current_depth <= int(before.get("messages_ready", 0)):
        raise RuntimeError(f"poison task was not observed in DLQ: {report}")
    return report


async def _sigterm_during_batch(
    settings: Settings,
    image: Path,
    count: int,
    output: Path,
) -> dict[str, Any]:
    if settings.rabbit.url is None:
        raise RuntimeError("APP_RABBIT__URL is required")
    if count != settings.rabbit.worker_concurrency:
        raise ValueError(
            "use a batch equal to worker_concurrency so every accepted task is "
            "already in flight when SIGTERM is sent",
        )
    containers = subprocess.check_output(
        ["docker", "compose", "--profile", "hw4", "ps", "-q", "worker"],
        text=True,
    ).split()
    if len(containers) != 1:
        raise RuntimeError("SIGTERM proof requires exactly one running worker")

    queue_path = f"/api/queues/%2F/{_quote_queue(settings.rabbit.queue_name)}"
    initial_queue = await asyncio.to_thread(_management_json, settings, queue_path)
    if int(initial_queue.get("messages", 0)) != 0:
        raise RuntimeError("drain the task queue before the SIGTERM proof")

    target = containers[0]
    started_at = datetime.now(UTC)
    command = [
        "uv",
        "run",
        "mtsd-experiment",
        "--image",
        str(image),
        "--count",
        str(count),
        "--output",
        str(output),
        "--drain-timeout-s",
        "180",
    ]
    experiment = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    deadline = time.monotonic() + 60
    queue = {}
    while time.monotonic() < deadline:
        if experiment.returncode is not None:
            break
        queue = await asyncio.to_thread(_management_json, settings, queue_path)
        if (
            int(queue.get("messages_unacknowledged", 0)) == count
            and int(queue.get("messages_ready", 0)) == 0
        ):
            break
        await asyncio.sleep(0.25)
    if (
        int(queue.get("messages_unacknowledged", 0)) != count
        or int(queue.get("messages_ready", 0)) != 0
    ):
        if experiment.returncode is None:
            experiment.terminate()
        output_text, _ = await experiment.communicate()
        raise RuntimeError(
            "all batch messages did not enter the worker before SIGTERM; "
            "experiment output:\n" + output_text.decode(errors="replace"),
        )

    signal_at = datetime.now(UTC).isoformat(timespec="seconds")
    subprocess.run(
        ["docker", "kill", "--signal=SIGTERM", target],
        check=True,
        capture_output=True,
        text=True,
    )
    stop_deadline = time.monotonic() + settings.rabbit.graceful_timeout_s + 10
    while time.monotonic() < stop_deadline:
        running = await asyncio.to_thread(
            subprocess.check_output,
            ["docker", "inspect", "--format={{.State.Running}}", target],
            text=True,
        )
        if running.strip() == "false":
            break
        await asyncio.sleep(0.2)
    else:
        raise RuntimeError("worker did not exit after SIGTERM within graceful timeout")

    worker_logs = await asyncio.to_thread(
        subprocess.check_output,
        ["docker", "logs", "--since", signal_at, target],
        stderr=subprocess.STDOUT,
        text=True,
    )
    queue_after_exit = {}
    queue_exit_deadline = time.monotonic() + 15
    while time.monotonic() < queue_exit_deadline:
        queue_after_exit = await asyncio.to_thread(
            _management_json,
            settings,
            queue_path,
        )
        if (
            int(queue_after_exit.get("messages_ready", 0)) == 0
            and int(queue_after_exit.get("messages_unacknowledged", 0)) == 0
        ):
            break
        await asyncio.sleep(0.5)
    pool = await create_pool(
        dsn=settings.database.url,
        min_size=1,
        max_size=2,
        timeout=settings.database.connect_timeout_s,
    )
    try:
        drained = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE status = 'DONE') AS done,
                COUNT(*) FILTER (WHERE status IN ('PENDING', 'PROCESSING')) AS active
            FROM detector_tasks
            WHERE created_at >= $1 AND task_id LIKE 'hw4-%'
            """,
            started_at,
        )
    finally:
        await pool.close()

    shutdown_log = worker_logs.partition("FastStream app shutting down...")[2]
    processed_during_shutdown = sum(
        " - Processed" in line for line in shutdown_log.splitlines()
    )
    graceful_evidence = {
        "log_reports_graceful_shutdown": "shut down gracefully" in worker_logs,
        "processed_messages_during_shutdown": processed_during_shutdown,
        "queue_ready_after_worker_exit": int(queue_after_exit.get("messages_ready", 0)),
        "queue_unacknowledged_after_worker_exit": int(
            queue_after_exit.get("messages_unacknowledged", 0)
        ),
        "batch_rows_at_worker_exit": int(drained["total"]),
        "batch_done_at_worker_exit": int(drained["done"]),
        "batch_active_at_worker_exit": int(drained["active"]),
    }

    # Restore the compose service after collecting evidence from the stopped
    # process. Restarting earlier could let the redelivered messages pass and
    # conceal whether the original worker drained its in-flight deliveries.
    await asyncio.to_thread(
        subprocess.run,
        ["docker", "start", target],
        check=True,
        capture_output=True,
        text=True,
    )
    output_text, _ = await asyncio.wait_for(experiment.communicate(), timeout=190)
    if experiment.returncode != 0:
        raise RuntimeError(
            f"experiment exited {experiment.returncode}:\n"
            + output_text.decode(errors="replace"),
        )
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    report = {
        "sigterm_container": target,
        "unacknowledged_at_signal": queue.get("messages_unacknowledged"),
        "graceful_shutdown_evidence": graceful_evidence,
        "tasks_requested": summary["tasks_requested"],
        "tasks_done": summary["final_done"],
        "tasks_failed": summary["final_failed"],
        "worker_restarted_after_evidence": True,
        "summary": str(output.with_suffix(".summary.json")),
    }
    if (
        not graceful_evidence["log_reports_graceful_shutdown"]
        or graceful_evidence["processed_messages_during_shutdown"] != count
        or graceful_evidence["queue_ready_after_worker_exit"] != 0
        or graceful_evidence["queue_unacknowledged_after_worker_exit"] != 0
        or graceful_evidence["batch_rows_at_worker_exit"] != count
        or graceful_evidence["batch_done_at_worker_exit"] != count
        or graceful_evidence["batch_active_at_worker_exit"] != 0
        or summary["final_done"] != count
        or summary["final_failed"] != 0
    ):
        raise RuntimeError(
            f"worker did not gracefully drain its accepted messages: {report}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("api-burst", "poison", "sigterm"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hw4/sigterm.jsonl")
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    if args.scenario == "api-burst":
        if args.image is None:
            parser.error("api-burst scenario requires --image")
        report = asyncio.run(_api_burst(settings, args.count, args.image))
    elif args.scenario == "poison":
        report = asyncio.run(_poison_message(settings))
    else:
        if args.image is None:
            parser.error("sigterm scenario requires --image")
        report = asyncio.run(
            _sigterm_during_batch(settings, args.image, args.count, args.output),
        )
    encoded_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded_report, encoding="utf-8")
    print(encoded_report, end="")


if __name__ == "__main__":
    main()
