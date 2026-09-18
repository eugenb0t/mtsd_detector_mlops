"""Async health checks for third-party components."""

from __future__ import annotations

import logging
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import asyncpg

from mtsd_detector.schemas import ComponentHealth, HealthResponse, HealthStatus

logger = logging.getLogger(__name__)

LIBRARY_PACKAGES = ("fastapi", "asyncpg", "uvicorn")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


async def check_postgres(pool: asyncpg.Pool | None) -> ComponentHealth:
    if pool is None:
        return ComponentHealth(
            name="postgres",
            version="unavailable",
            latency_ms=0.0,
            ok=False,
            detail="database pool is not initialized",
        )

    started = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            pg_version = await conn.fetchval("SELECT version()")
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return ComponentHealth(
            name="postgres",
            version=str(pg_version),
            latency_ms=latency_ms,
            ok=True,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        logger.error(
            "postgres health check failed",
            extra={"component": "postgres", "latency_ms": latency_ms},
            exc_info=True,
        )
        return ComponentHealth(
            name="postgres",
            version="unavailable",
            latency_ms=latency_ms,
            ok=False,
            detail=str(exc),
        )


def check_library(name: str) -> ComponentHealth:
    started = time.perf_counter()
    pkg_version = _package_version(name)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return ComponentHealth(
        name=name,
        version=pkg_version,
        latency_ms=latency_ms,
        ok=pkg_version != "unknown",
    )


def aggregate_status(components: list[ComponentHealth]) -> HealthStatus:
    if all(c.ok for c in components):
        return "ok"
    if any(c.ok for c in components):
        return "degraded"
    return "error"


async def build_health_report(pool: asyncpg.Pool | None) -> HealthResponse:
    components: list[ComponentHealth] = [await check_postgres(pool)]
    components.extend(check_library(name) for name in LIBRARY_PACKAGES)
    status = aggregate_status(components)
    return HealthResponse(status=status, components=components)


async def create_pool(dsn: str, **kwargs: Any) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn=dsn, **kwargs)
