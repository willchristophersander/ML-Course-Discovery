"""Metadata aggregation helpers for the data collection agent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence

from .api_clients import FetchResult


@dataclass
class DatasetMetadata:
    """Structured metadata describing a collected dataset."""

    dataset_name: str
    source: str
    records: int
    earliest_timestamp: str | None
    latest_timestamp: str | None
    issues: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "source": self.source,
            "records": self.records,
            "earliest_timestamp": self.earliest_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "issues": self.issues,
        }


def summarise_results(results: Sequence[FetchResult], issues: Dict[str, Iterable[str]]) -> List[DatasetMetadata]:
    """Build dataset summaries for downstream metadata persistence."""

    summaries: List[DatasetMetadata] = []
    for result in results:
        timestamps = [record.get("timestamp") for record in result.payload]
        sorted_ts = sorted(ts for ts in timestamps if isinstance(ts, str))
        dataset_name = result.task_id.replace(":", "_")
        dataset_issues = list(issues.get(result.task_id, []))
        summaries.append(
            DatasetMetadata(
                dataset_name=dataset_name,
                source=result.source,
                records=len(result.payload),
                earliest_timestamp=sorted_ts[0] if sorted_ts else None,
                latest_timestamp=sorted_ts[-1] if sorted_ts else None,
                issues=dataset_issues,
            )
        )
    return summaries


def compile_metadata_document(
    results: Sequence[FetchResult],
    issues: Dict[str, Iterable[str]],
) -> Dict[str, object]:
    """Convert fetch results and issues into a repository metadata document."""

    summaries = summarise_results(results, issues)
    processed_at = datetime.now(timezone.utc).isoformat()
    return {
        "processed_at": processed_at,
        "datasets": [summary.to_dict() for summary in summaries],
        "total_records": sum(summary.records for summary in summaries),
    }


__all__ = [
    "DatasetMetadata",
    "summarise_results",
    "compile_metadata_document",
]
