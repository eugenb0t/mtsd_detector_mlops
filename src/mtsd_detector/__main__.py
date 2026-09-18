"""CLI entrypoint: ``uv run mtsd-detector`` or project script."""

from __future__ import annotations

import uvicorn

from mtsd_detector.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "mtsd_detector.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
