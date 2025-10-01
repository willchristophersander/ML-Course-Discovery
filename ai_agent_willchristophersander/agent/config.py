"""Configuration loading utilities for the financial data collection agent."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class APISourceConfig:
    """Settings for a single upstream API."""

    name: str
    base_url: str
    enabled: bool = True
    api_key_env: Optional[str] = None
    function: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    series_id: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    request_delay_seconds: Optional[float] = None

    def resolve_api_key(self) -> Optional[str]:
        """Return the API key from the environment when configured."""

        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


@dataclass
class CollectionConfig:
    """Global collection parameters for retries and timing."""

    request_delay_seconds: float
    max_retries: int
    retry_backoff_factor: float
    target_timezone: str = "UTC"


@dataclass
class QualityConfig:
    """Thresholds that govern quality evaluation."""

    freshness_days_threshold: int
    required_columns: List[str]
    completeness_threshold: float
    anomaly_threshold_sigma: float = 3.0


@dataclass
class StorageConfig:
    """Filesystem layout for persisted artefacts."""

    raw_data_dir: Path
    processed_data_dir: Path
    metadata_dir: Path

    @classmethod
    def from_dict(cls, data: Dict[str, str], base_path: Path) -> "StorageConfig":
        """Construct storage config using paths relative to config file."""

        return cls(
            raw_data_dir=(base_path / data["raw_data_dir"]).resolve(),
            processed_data_dir=(base_path / data["processed_data_dir"]).resolve(),
            metadata_dir=(base_path / data["metadata_dir"]).resolve(),
        )


@dataclass
class AgentConfig:
    """Top level agent configuration."""

    sources: Dict[str, APISourceConfig]
    collection: CollectionConfig
    quality: QualityConfig
    storage: StorageConfig

    @classmethod
    def load(cls, file_path: Path) -> "AgentConfig":
        """Load configuration data from JSON file."""

        path = file_path.resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        base_dir = path.parent
        raw_sources: Dict[str, Dict[str, object]] = payload.get("api_sources", {})
        sources: Dict[str, APISourceConfig] = {}
        for name, cfg in raw_sources.items():
            sources[name] = APISourceConfig(name=name, **cfg)

        collection = CollectionConfig(**payload["collection"])
        quality_kwargs = payload["quality"].copy()
        quality = QualityConfig(**quality_kwargs)
        storage = StorageConfig.from_dict(payload["storage"], base_dir)

        return cls(
            sources=sources,
            collection=collection,
            quality=quality,
            storage=storage,
        )

    def enabled_sources(self) -> List[APISourceConfig]:
        """Return sources flagged as enabled."""

        return [source for source in self.sources.values() if source.enabled]


def load_config(path: str | Path) -> AgentConfig:
    """Convenience wrapper for `AgentConfig.load`."""

    return AgentConfig.load(Path(path))


__all__ = [
    "APISourceConfig",
    "CollectionConfig",
    "QualityConfig",
    "StorageConfig",
    "AgentConfig",
    "load_config",
]
