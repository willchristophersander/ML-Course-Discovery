"""Collection strategy helpers for the data collection agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .config import AgentConfig, APISourceConfig


@dataclass
class CollectionTask:
    """Atomic unit of work for the agent."""

    task_id: str
    source_name: str
    symbol: Optional[str] = None
    series_id: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """Human readable description for logging."""

        parts = [self.source_name]
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.series_id:
            parts.append(f"series={self.series_id}")
        return ", ".join(parts)


@dataclass
class TaskOutcome:
    """Represents a collection attempt outcome used for adaptive decisions."""

    task_id: str
    success: bool
    latency_seconds: float
    timestamp: datetime
    error: Optional[str] = None


class StrategyPlanner:
    """Builds collection tasks using the agent configuration."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def build_tasks(self) -> List[CollectionTask]:
        """Create a task list expanding tickers and series IDs."""

        tasks: List[CollectionTask] = []
        for source in self._config.enabled_sources():
            if source.symbols:
                for symbol in source.symbols:
                    params = {**source.params}
                    if source.function:
                        params.setdefault("function", source.function)
                    tasks.append(
                        CollectionTask(
                            task_id=f"{source.name}:{symbol}",
                            source_name=source.name,
                            symbol=symbol,
                            params=params,
                        )
                    )
            elif source.series_id:
                params = {**source.params}
                params.setdefault("series_id", source.series_id)
                tasks.append(
                    CollectionTask(
                        task_id=f"{source.name}:{source.series_id}",
                        source_name=source.name,
                        series_id=source.series_id,
                        params=params,
                    )
                )
            else:
                tasks.append(
                    CollectionTask(
                        task_id=f"{source.name}:default",
                        source_name=source.name,
                        params={**source.params},
                    )
                )
        return tasks


class CircuitBreaker:
    """Simple circuit breaker to avoid hammering failing APIs."""

    def __init__(self, failure_threshold: int = 3, reset_after: int = 5) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after
        self._failure_counts: Dict[str, int] = {}
        self._open_cycles: Dict[str, int] = {}

    def record(self, outcome: TaskOutcome) -> None:
        """Update internal counters based on the latest task outcome."""

        if outcome.success:
            self._failure_counts.pop(outcome.task_id, None)
            self._open_cycles.pop(outcome.task_id, None)
            return

        failures = self._failure_counts.get(outcome.task_id, 0) + 1
        self._failure_counts[outcome.task_id] = failures
        if failures >= self.failure_threshold:
            self._open_cycles[outcome.task_id] = self.reset_after

    def should_skip(self, task: CollectionTask) -> bool:
        """Return True when the breaker is open for the task."""

        cycles_remaining = self._open_cycles.get(task.task_id)
        if cycles_remaining is None:
            return False

        cycles_remaining -= 1
        if cycles_remaining <= 0:
            self._open_cycles.pop(task.task_id, None)
            self._failure_counts.pop(task.task_id, None)
            return False

        self._open_cycles[task.task_id] = cycles_remaining
        return True


__all__ = [
    "CollectionTask",
    "TaskOutcome",
    "StrategyPlanner",
    "CircuitBreaker",
]
