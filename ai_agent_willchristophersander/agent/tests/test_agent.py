"""Unit tests for the financial data collection agent."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from ai_agent_willchristophersander.agent.api_clients import FetchResult
from ai_agent_willchristophersander.agent.data_collection_agent import FinancialDataCollectionAgent


class StubAlphaClient:
    """Stub Alpha Vantage client returning deterministic payloads."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.called_with: List[str] = []

    def fetch_daily_series(self, symbol: str, function: str | None = None) -> FetchResult:
        self.called_with.append(symbol)
        payload = [
            {
                "timestamp": "2024-09-01T00:00:00+00:00",
                "value": 150.0,
                "symbol": symbol,
                "source": "alpha_vantage",
            },
            {
                "timestamp": "2024-09-02T00:00:00+00:00",
                "value": 151.5,
                "symbol": symbol,
                "source": "alpha_vantage",
            },
        ]
        return FetchResult(
            source="alpha_vantage",
            task_id=f"alpha_vantage:{symbol}",
            success=True,
            payload=payload,
            raw={"payload": payload},
            fetched_at=datetime.now(timezone.utc),
            latency_seconds=0.1,
        )


class StubFREDClient:
    """Stub FRED client returning deterministic payloads."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.called_with: List[str] = []

    def fetch_series(self, series_id: str) -> FetchResult:
        self.called_with.append(series_id)
        payload = [
            {
                "timestamp": "2024-09-01T00:00:00+00:00",
                "value": 1.0,
                "series_id": series_id,
                "source": "fred",
            },
            {
                "timestamp": "2024-09-02T00:00:00+00:00",
                "value": 1.1,
                "series_id": series_id,
                "source": "fred",
            },
        ]
        return FetchResult(
            source="fred",
            task_id=f"fred:{series_id}",
            success=True,
            payload=payload,
            raw={"payload": payload},
            fetched_at=datetime.now(timezone.utc),
            latency_seconds=0.05,
        )


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create a temporary configuration for testing."""

    project_root = tmp_path
    (project_root / "agent").mkdir()
    (project_root / "data").mkdir()
    (project_root / "logs").mkdir()
    (project_root / "reports").mkdir()

    config_payload = {
        "api_sources": {
            "alpha_vantage": {
                "enabled": True,
                "base_url": "https://example.com",
                "function": "TIME_SERIES_DAILY",
                "symbols": ["AAPL"],
            },
            "fred": {
                "enabled": True,
                "base_url": "https://example.com",
                "series_id": "CPIAUCSL",
            },
        },
        "collection": {
            "request_delay_seconds": 0.0,
            "max_retries": 1,
            "retry_backoff_factor": 1.0,
            "target_timezone": "UTC",
        },
        "quality": {
            "freshness_days_threshold": 5,
            "required_columns": ["timestamp", "value", "source"],
            "completeness_threshold": 0.9,
        },
        "storage": {
            "raw_data_dir": "../data/raw",
            "processed_data_dir": "../data/processed",
            "metadata_dir": "../data/metadata",
        },
    }
    config_path = project_root / "agent" / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return config_path


def test_agent_run_creates_reports(monkeypatch: pytest.MonkeyPatch, temp_config: Path) -> None:
    """Agent run should complete and persist artefacts when clients succeed."""

    clients = {
        "alpha_vantage": StubAlphaClient(),
        "fred": StubFREDClient(),
    }

    def _fake_build_client(source, _collection):
        return clients[source.name]

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "dummy")
    monkeypatch.setenv("FRED_API_KEY", "dummy")
    monkeypatch.setattr(
        "ai_agent_willchristophersander.agent.data_collection_agent.build_client",
        _fake_build_client,
        raising=True,
    )
    monkeypatch.setattr(
        "ai_agent_willchristophersander.agent.api_clients.build_client",
        _fake_build_client,
        raising=True,
    )

    agent = FinancialDataCollectionAgent(temp_config)
    agent.run()

    metadata_path = (temp_config.parent / "../data/metadata/collection_metadata.json").resolve()
    summary_path = (temp_config.parent / "../reports/collection_summary.md").resolve()
    quality_report_path = (temp_config.parent / "../reports/quality_report.html").resolve()

    assert metadata_path.exists(), "metadata file should be created"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["total_records"] == 4

    assert summary_path.exists(), "collection summary should be written"
    assert quality_report_path.exists(), "quality report should be generated"
