"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from mtsd_detector.api import build_api_router
from mtsd_detector.config import Settings, get_settings
from mtsd_detector.logging import setup_logging
from mtsd_detector.middleware import RequestLoggingMiddleware
from mtsd_detector.services.health import create_pool
from mtsd_detector.services.version import get_app_version
from mtsd_detector.task_source import TaskSource
from mtsd_detector.task_store import TaskStore

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.db_pool = None
        app.state.task_store = None
        app.state.task_source = None
        try:
            app.state.db_pool = await create_pool(
                dsn=settings.database.url,
                min_size=settings.database.pool_min_size,
                max_size=settings.database.pool_max_size,
                timeout=settings.database.connect_timeout_s,
            )
            logger.info("database pool created")
        except Exception:
            logger.exception(
                "failed to create database pool; /api/v1/health will report error",
            )
        if app.state.db_pool is not None:
            try:
                store = TaskStore(app.state.db_pool)
                await store.initialize()
                app.state.task_store = store
                if settings.rabbit.url:
                    source = TaskSource(settings.rabbit, store)
                    await source.start()
                    app.state.task_source = source
                    logger.info(
                        "task source connected",
                        extra={
                            "queue_max_length": settings.rabbit.queue_max_length,
                            "prefetch_count": settings.rabbit.prefetch_count,
                            "worker_concurrency": settings.rabbit.worker_concurrency,
                        },
                    )
            except Exception:
                logger.exception("failed to initialize task pipeline")
                if settings.rabbit.url:
                    raise
        yield
        source = getattr(app.state, "task_source", None)
        if source is not None:
            await source.close()
        pool = getattr(app.state, "db_pool", None)
        if pool is not None:
            await pool.close()
            logger.info("database pool closed")

    app = FastAPI(
        title="MTSD Detector MLOps",
        version=get_app_version(),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(build_api_router())

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Browser entrypoint: redirect to Swagger UI."""
        return RedirectResponse(url="/docs")

    return app
