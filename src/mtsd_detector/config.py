"""Application settings: YAML primary, env overrides via nested delimiter."""

from __future__ import annotations

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
