"""External task boundary with load shedding and status polling."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from mtsd_detector.task_source import PublishRejected, TaskSource
from mtsd_detector.task_store import TaskStore
from mtsd_detector.task_types import TaskAccepted, TaskResult, TaskSubmission

router = APIRouter(prefix="/api/v1", tags=["tasks"])
legacy_router = APIRouter(tags=["tasks"])


@router.post(
    "/tasks",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task(
    payload: TaskSubmission,
    request: Request,
    response: Response,
) -> TaskAccepted:
    """Submit one task without waiting for inference.

    RabbitMQ's publisher-confirm NACK is translated directly to 503. This is
    load shedding at the system boundary: the external caller cannot be
    slowed by our internal queue, so the request is rejected and can retry.
    """
    source: TaskSource | None = getattr(request.app.state, "task_source", None)
    if source is None:
        response.headers["Retry-After"] = "1"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="task pipeline is not accepting work",
            headers={"Retry-After": "1"},
        )
    task = payload.to_task()
    try:
        accepted = await source.submit(task)
    except PublishRejected as error:
        retry_after = str(source.settings.effective_retry_after_s)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="task queue is full; retry later",
            headers={"Retry-After": retry_after},
        ) from error
    except Exception as error:
        retry_after = str(source.settings.effective_retry_after_s)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="task broker is unavailable; retry later",
            headers={"Retry-After": retry_after},
        ) from error
    return accepted


@router.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task(task_id: str, request: Request) -> TaskResult:
    """Return one durable task state or result."""
    store: TaskStore | None = getattr(request.app.state, "task_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="task store is unavailable")
    task = await store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


# The old coursework endpoint remains a compatibility alias, but now submits
# asynchronous work instead of pretending to perform synchronous inference.
router.add_api_route(
    "/process",
    submit_task,
    methods=["POST"],
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
legacy_router.add_api_route(
    "/process",
    submit_task,
    methods=["POST"],
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
