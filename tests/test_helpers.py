"""Unit tests for helpers and version single-source-of-truth."""

from __future__ import annotations

import json
import logging
import tomllib
from importlib.metadata import version as pkg_version
from pathlib import Path

import pytest

from mtsd_detector.config import Settings, clear_settings_cache, get_settings
from mtsd_detector.logging import JsonFormatter, setup_logging
from mtsd_detector.services.health import check_postgres, create_pool
from mtsd_detector.services.version import get_app_version

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_get_app_version_matches_pyproject_and_metadata() -> None:
    expected = _pyproject_version()
    assert get_app_version() == expected
    assert pkg_version("mtsd-detector") == expected
    assert expected != "0.0.0"


def test_json_formatter_with_extras() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc"
    record.method = "GET"
    record.path = "/healthz"
    record.status_code = 200
    record.duration_ms = 1.5
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello"
    assert payload["request_id"] == "abc"
    assert payload["status_code"] == 200


def test_setup_logging() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_settings_cache()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "environment: development\nhost: 127.0.0.1\nport: 9000\nlog_level: INFO\n"
        "database:\n  url: postgresql://x:y@localhost:5432/z\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_CONFIG_PATH", str(cfg))
    monkeypatch.delenv("APP_HOST", raising=False)
    settings = get_settings()
    assert settings.port == 9000
    assert get_settings() is settings
    clear_settings_cache()


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    monkeypatch.setenv("APP_PORT", "9001")
    settings = Settings()
    assert settings.port == 9001
    clear_settings_cache()


async def test_check_postgres_none_pool() -> None:
    result = await check_postgres(None)
    assert result.ok is False
    assert result.name == "postgres"


async def test_create_pool_signature() -> None:
    assert callable(create_pool)
