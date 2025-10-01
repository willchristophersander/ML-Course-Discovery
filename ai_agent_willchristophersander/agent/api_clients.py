"""Reusable API client implementations for the data collection agent."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .config import APISourceConfig, CollectionConfig
from .logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class FetchResult:
    """Container for API collection responses."""

    source: str
    task_id: str
    success: bool
    payload: List[Dict[str, Any]]
    raw: Dict[str, Any] | None
    fetched_at: datetime
    latency_seconds: float
    error: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Return metadata fields for persistence."""

        return {
            "source": self.source,
            "task_id": self.task_id,
            "success": self.success,
            "records": len(self.payload),
            "fetched_at": self.fetched_at.isoformat(),
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }


class RateLimiter:
    """Simple sleep-based rate limiter."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        """Sleep until the minimum interval has passed."""

        if self._last_call is None:
            self._last_call = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class APIClient:
    """Base class implementing retry and error handling."""

    def __init__(
        self,
        source_config: APISourceConfig,
        collection_config: CollectionConfig,
    ) -> None:
        self.source_config = source_config
        self.collection_config = collection_config
        delay = source_config.request_delay_seconds or collection_config.request_delay_seconds
        self.rate_limiter = RateLimiter(delay)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Will-Finance-Agent/1.0 (+mailto:data-team@example.edu)",
                "Accept": "application/json",
            }
        )

    def get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a GET request with retries and returns JSON payload."""

        attempts = 0
        last_error: Optional[str] = None
        while attempts < self.collection_config.max_retries:
            self.rate_limiter.wait()
            attempts += 1
            start = time.monotonic()
            try:
                response = self.session.get(
                    self.source_config.base_url,
                    params=params,
                    timeout=30,
                )
                latency = time.monotonic() - start
                response.raise_for_status()
                return response.json(), latency
            except requests.RequestException as exc:
                last_error = str(exc)
                backoff = self.collection_config.retry_backoff_factor ** (attempts - 1)
                sleep_for = backoff * self.collection_config.request_delay_seconds
                logger.warning(
                    "Request error for %s attempt %s/%s: %s (sleep %.1fs)",
                    self.source_config.name,
                    attempts,
                    self.collection_config.max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        raise RuntimeError(last_error or "Unknown request failure")


class AlphaVantageClient(APIClient):
    """Client for Alpha Vantage equity price time series."""

    def fetch_daily_series(self, symbol: str, function: Optional[str] = None) -> FetchResult:
        api_key = self.source_config.resolve_api_key()
        if not api_key:
            raise RuntimeError("Alpha Vantage API key not found in environment")

        params: Dict[str, Any] = {
            "function": function or self.source_config.function or "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": api_key,
        }
        params.update(self.source_config.params)

        payload, latency = self.get(params)
        records = self._parse_time_series(payload, symbol)
        return FetchResult(
            source=self.source_config.name,
            task_id=f"{self.source_config.name}:{symbol}",
            success=True,
            payload=records,
            raw=payload,
            fetched_at=datetime.now(timezone.utc),
            latency_seconds=latency,
        )

    def _parse_time_series(self, payload: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        key = next((k for k in payload.keys() if "Time Series" in k), None)
        if not key:
            message = payload.get("Note") or payload.get("Error Message")
            raise RuntimeError(message or "Unexpected Alpha Vantage response")

        series = payload[key]
        records: List[Dict[str, Any]] = []
        for date_str, metrics in series.items():
            record = {
                "timestamp": f"{date_str}T00:00:00+00:00",
                "value": float(metrics.get("4. close", 0.0)),
                "open": float(metrics.get("1. open", 0.0)),
                "high": float(metrics.get("2. high", 0.0)),
                "low": float(metrics.get("3. low", 0.0)),
                "volume": int(float(metrics.get("5. volume", 0.0))),
                "symbol": symbol,
                "source": self.source_config.name,
            }
            records.append(record)
        records.sort(key=lambda item: item["timestamp"])
        return records


class FREDClient(APIClient):
    """Client for the FRED economic indicators API."""

    def fetch_series(self, series_id: str) -> FetchResult:
        api_key = self.source_config.resolve_api_key()
        if not api_key:
            raise RuntimeError("FRED API key not found in environment")

        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        params.update(self.source_config.params)

        payload, latency = self.get(params)
        observations = payload.get("observations")
        if observations is None:
            message = payload.get("error_message") or "No observations field in response"
            raise RuntimeError(message)

        records: List[Dict[str, Any]] = []
        for item in observations:
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            record = {
                "timestamp": f"{item.get('date')}T00:00:00+00:00",
                "value": value,
                "series_id": series_id,
                "source": self.source_config.name,
            }
            records.append(record)
        records.sort(key=lambda item: item["timestamp"])
        return FetchResult(
            source=self.source_config.name,
            task_id=f"{self.source_config.name}:{series_id}",
            success=True,
            payload=records,
            raw=payload,
            fetched_at=datetime.now(timezone.utc),
            latency_seconds=latency,
        )


class FinancialModelingPrepClient(APIClient):
    """Client for backup fundamentals from Financial Modeling Prep."""

    def fetch_profile(self, symbol: str) -> FetchResult:
        api_key = self.source_config.resolve_api_key()
        if not api_key:
            raise RuntimeError("Financial Modeling Prep API key not found in environment")

        params: Dict[str, Any] = {"apikey": api_key}
        params.update(self.source_config.params)
        url = f"{self.source_config.base_url.rstrip('/')}/{symbol.upper()}"

        attempts = 0
        last_error: Optional[str] = None
        while attempts < self.collection_config.max_retries:
            self.rate_limiter.wait()
            attempts += 1
            start = time.monotonic()
            try:
                response = self.session.get(url, params=params, timeout=30)
                latency = time.monotonic() - start
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or not payload:
                    raise RuntimeError("Unexpected profile response shape")
                record_raw = payload[0]
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": float(record_raw.get("beta", 0.0)),
                    "symbol": symbol,
                    "market_cap": record_raw.get("mktCap"),
                    "source": self.source_config.name,
                }
                return FetchResult(
                    source=self.source_config.name,
                    task_id=f"{self.source_config.name}:{symbol}",
                    success=True,
                    payload=[record],
                    raw={"profile": record_raw},
                    fetched_at=datetime.now(timezone.utc),
                    latency_seconds=latency,
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_error = str(exc)
                sleep_for = self.collection_config.request_delay_seconds * (
                    self.collection_config.retry_backoff_factor ** (attempts - 1)
                )
                logger.warning(
                    "Profile fetch failed for %s (%s/%s): %s",
                    symbol,
                    attempts,
                    self.collection_config.max_retries,
                    exc,
                )
                time.sleep(sleep_for)
        raise RuntimeError(last_error or "Profile retrieval failed")


def build_client(source: APISourceConfig, collection: CollectionConfig) -> APIClient:
    """Factory returning the appropriate API client instance."""

    name = source.name.lower()
    if "alpha" in name:
        return AlphaVantageClient(source, collection)
    if "fred" in name:
        return FREDClient(source, collection)
    if "financial" in name or "fmp" in name:
        return FinancialModelingPrepClient(source, collection)
    return APIClient(source, collection)


__all__ = [
    "FetchResult",
    "RateLimiter",
    "APIClient",
    "AlphaVantageClient",
    "FREDClient",
    "FinancialModelingPrepClient",
    "build_client",
]
