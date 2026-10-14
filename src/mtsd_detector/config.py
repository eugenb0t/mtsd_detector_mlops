"""Application settings: YAML primary, env overrides via nested delimiter."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

Environment = Literal["development", "staging", "production"]


class DatabaseSettings(BaseModel):
    # Local-lab default only; override via APP_DATABASE__URL / compose env_file.
    url: str = "postgresql://mtsd:mtsd@127.0.0.1:5433/mtsd"
    pool_min_size: int = 1
    pool_max_size: int = 5
    connect_timeout_s: float = 5.0


class MLflowSettings(BaseModel):
    """Registry settings used by the inference process at startup."""

    tracking_uri: str | None = None
    model_name: str = "mtsd20-detector"
    model_alias: str = "champion"

    @property
    def model_uri(self) -> str | None:
        if self.tracking_uri is None:
            return None
        return f"models:/{self.model_name}@{self.model_alias}"


class RabbitSettings(BaseModel):
    """RabbitMQ topology and explicit concurrency/backpressure contract."""

    url: str | None = None
    queue_name: str = "mtsd.tasks.pending"
    bounded_queue: bool = True
    queue_max_length: int = Field(default=8, ge=1)
    prefetch_count: int = Field(default=2, ge=1)
    worker_concurrency: int = Field(default=2, ge=1)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=0.25, ge=0.0)
    inference_delay_s: float = Field(default=0.25, ge=0.0)
    publish_confirm_timeout_s: float = Field(default=2.0, gt=0.0)
    retry_after_s: int | None = Field(default=None, ge=1)
    graceful_timeout_s: float = Field(default=30.0, gt=0.0)

    @property
    def effective_retry_after_s(self) -> int:
        """Retry hint based on one configured service-time interval."""
        if self.retry_after_s is not None:
            return self.retry_after_s
        return max(1, math.ceil(self.inference_delay_s))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    rabbit: RabbitSettings = Field(default_factory=RabbitSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (highest first): init > env > yaml
        yaml_path = resolve_config_path()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if yaml_path is not None and yaml_path.is_file():
            sources.append(
                YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
            )
        return tuple(sources)


def resolve_config_path() -> Path | None:
    """Locate config.yaml: APP_CONFIG_PATH env, then cwd, then repo root."""
    override = os.environ.get("APP_CONFIG_PATH")
    if override:
        return Path(override)
    candidates = (
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parents[2] / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
