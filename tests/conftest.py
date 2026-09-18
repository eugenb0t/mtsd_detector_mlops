"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from mtsd_detector.app import create_app
from mtsd_detector.config import DatabaseSettings, Settings, clear_settings_cache


@pytest.fixture
def settings() -> Settings:
    clear_settings_cache()
    return Settings(
        environment="development",
        host="127.0.0.1",
        port=8000,
        log_level="WARNING",
        database=DatabaseSettings(
            url="postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd",
            pool_min_size=1,
            pool_max_size=2,
            connect_timeout_s=1.0,
        ),
    )


def _mock_pool(*, fetch_error: Exception | None = None) -> MagicMock:
    mock_conn = MagicMock()
    if fetch_error is None:
        mock_conn.fetchval = AsyncMock(return_value="PostgreSQL 16.0 (test)")
    else:
        mock_conn.fetchval = AsyncMock(side_effect=fetch_error)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=acquire_cm)
    mock_pool.close = AsyncMock()
    return mock_pool


async def _client_with_lifespan(
    app: object,
) -> AsyncIterator[AsyncClient]:
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))  # type: ignore[attr-defined]
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        client = await stack.enter_async_context(
            AsyncClient(transport=transport, base_url="http://test"),
        )
        yield client


@pytest.fixture
async def client_ok_db(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    pool = _mock_pool()

    async def fake_create_pool(*_args: object, **_kwargs: object) -> MagicMock:
        return pool

    monkeypatch.setattr("mtsd_detector.app.create_pool", fake_create_pool)
    app = create_app(settings)
    async for client in _client_with_lifespan(app):
        yield client


@pytest.fixture
async def client_no_db(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    async def fail_create_pool(*_args: object, **_kwargs: object) -> MagicMock:
        raise OSError("connection refused")

    monkeypatch.setattr("mtsd_detector.app.create_pool", fail_create_pool)
    app = create_app(settings)
    async for client in _client_with_lifespan(app):
        yield client


@pytest.fixture
async def client_db_query_fail(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    pool = _mock_pool(fetch_error=RuntimeError("query failed"))

    async def fake_create_pool(*_args: object, **_kwargs: object) -> MagicMock:
        return pool

    monkeypatch.setattr("mtsd_detector.app.create_pool", fake_create_pool)
    app = create_app(settings)
    async for client in _client_with_lifespan(app):
        yield client
