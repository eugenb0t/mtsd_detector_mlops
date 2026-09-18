"""Version service — single source of truth is package metadata."""

from __future__ import annotations

from mtsd_detector import __version__


def get_app_version() -> str:
    return __version__
