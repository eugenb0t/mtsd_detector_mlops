"""API endpoint tests."""

from __future__ import annotations

from httpx import AsyncClient

from mtsd_detector import __version__
from mtsd_detector.schemas import ComponentHealth
from mtsd_detector.services.health import aggregate_status, check_library


async def test_healthz(client_ok_db: AsyncClient) -> None:
    response = await client_ok_db.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_root_redirects_to_docs(client_ok_db: AsyncClient) -> None:
    response = await client_ok_db.get("/", follow_redirects=False)
    assert response.status_code in {307, 302}
    assert response.headers["location"] == "/docs"


async def test_version_matches_package_metadata(client_ok_db: AsyncClient) -> None:
    response = await client_ok_db.get("/api/v1/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == __version__
    assert body["version"] != "0.0.0"
    # Single source of truth: endpoint == installed package == pyproject.toml
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8",
        ),
    )
    assert body["version"] == pyproject["project"]["version"]


async def test_health_ok(client_ok_db: AsyncClient) -> None:
    response = await client_ok_db.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    names = {c["name"] for c in body["components"]}
    assert {"postgres", "fastapi", "asyncpg", "uvicorn"} <= names
    postgres = next(c for c in body["components"] if c["name"] == "postgres")
    assert postgres["ok"] is True
    assert postgres["latency_ms"] >= 0
    assert "PostgreSQL" in postgres["version"]


async def test_health_db_unavailable(client_no_db: AsyncClient) -> None:
    response = await client_no_db.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] in {"degraded", "error"}
    postgres = next(c for c in body["components"] if c["name"] == "postgres")
    assert postgres["ok"] is False


async def test_health_db_query_failure(client_db_query_fail: AsyncClient) -> None:
    response = await client_db_query_fail.get("/api/v1/health")
    assert response.status_code == 503
    body = response.json()
    postgres = next(c for c in body["components"] if c["name"] == "postgres")
    assert postgres["ok"] is False
    assert postgres["detail"] is not None


def test_check_library_fastapi() -> None:
    component = check_library("fastapi")
    assert component.ok is True
    assert component.version != "unknown"


def test_aggregate_status() -> None:
    ok = ComponentHealth(name="a", version="1", latency_ms=0, ok=True)
    bad = ComponentHealth(name="b", version="1", latency_ms=0, ok=False)
    assert aggregate_status([ok, ok]) == "ok"
    assert aggregate_status([ok, bad]) == "degraded"
    assert aggregate_status([bad, bad]) == "error"


async def test_healthz_still_ok_without_db(client_no_db: AsyncClient) -> None:
    response = await client_no_db.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_process_uses_startup_model(client_with_model: AsyncClient) -> None:
    response = await client_with_model.post(
        "/process",
        json={"features": [0.0] * 256},
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 7,
        "probabilities": [0.25, 0.75],
        "model_uri": "models:/mtsd20-crop-classifier@champion",
        "model_version": "1",
        "loaded_at": "2026-01-01T00:00:00+00:00",
    }


async def test_process_reports_missing_startup_model(client_ok_db: AsyncClient) -> None:
    response = await client_ok_db.post(
        "/process",
        json={"features": [0.0] * 256},
    )
    assert response.status_code == 503
