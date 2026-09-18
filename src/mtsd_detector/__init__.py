"""MTSD detector MLOps service package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mtsd-detector")
except PackageNotFoundError:  # pragma: no cover - editable/dev edge case
    __version__ = "0.0.0"

__all__ = ["__version__"]
