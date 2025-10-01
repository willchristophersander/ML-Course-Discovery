"""Quality evaluation utilities for collected datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import QualityConfig


@dataclass
class QualityMetrics:
    """Aggregate quality metrics for a dataset."""

    total_records: int
    completeness_ratio: float
    freshness_hours: float
    anomaly_rate: float
    quality_score: float

    def to_dict(self) -> Dict[str, float]:
        """Return metrics as a JSON serialisable dictionary."""

        return asdict(self)


class QualityEvaluator:
    """Compute quality metrics and surface issues for collected data."""

    def __init__(self, config: QualityConfig) -> None:
        self._config = config

    def evaluate(
        self,
        records: Sequence[Dict[str, object]],
        *,
        now: datetime | None = None,
    ) -> Tuple[QualityMetrics, List[str]]:
        """Evaluate quality metrics for the provided records."""

        now = now or datetime.now(timezone.utc)
        total_records = len(records)

        if total_records == 0:
            metrics = QualityMetrics(
                total_records=0,
                completeness_ratio=0.0,
                freshness_hours=float("inf"),
                anomaly_rate=1.0,
                quality_score=0.0,
            )
            return metrics, ["Dataset empty"]

        completeness_ratio = self._compute_completeness(records)
        freshness_hours = self._compute_freshness(records, now)
        anomaly_rate = self._compute_anomaly_rate(records)
        quality_score = self._score_dataset(
            completeness_ratio=completeness_ratio,
            freshness_hours=freshness_hours,
            anomaly_rate=anomaly_rate,
        )

        issues: List[str] = []
        if completeness_ratio < self._config.completeness_threshold:
            issues.append(
                f"Completeness below threshold ({completeness_ratio:.2%} < {self._config.completeness_threshold:.2%})"
            )
        if freshness_hours > self._config.freshness_days_threshold * 24:
            issues.append(
                f"Data older than threshold ({freshness_hours:.1f}h > {self._config.freshness_days_threshold * 24}h)"
            )
        if anomaly_rate > 0.1:
            issues.append(f"High anomaly rate detected ({anomaly_rate:.2%})")

        metrics = QualityMetrics(
            total_records=total_records,
            completeness_ratio=completeness_ratio,
            freshness_hours=freshness_hours,
            anomaly_rate=anomaly_rate,
            quality_score=quality_score,
        )
        return metrics, issues

    def _compute_completeness(self, records: Sequence[Dict[str, object]]) -> float:
        required_columns = self._config.required_columns
        total_cells = len(records) * max(len(required_columns), 1)
        if total_cells == 0:
            return 1.0

        filled = 0
        for record in records:
            for column in required_columns:
                value = record.get(column)
                if value not in (None, ""):
                    filled += 1
        return filled / total_cells

    def _compute_freshness(self, records: Sequence[Dict[str, object]], now: datetime) -> float:
        timestamps: List[datetime] = []
        for record in records:
            value = record.get("timestamp")
            if isinstance(value, datetime):
                timestamps.append(value.astimezone(timezone.utc))
            elif isinstance(value, str):
                try:
                    timestamps.append(datetime.fromisoformat(value).astimezone(timezone.utc))
                except ValueError:
                    continue
        if not timestamps:
            return float("inf")

        newest = max(timestamps)
        delta = now - newest
        return delta.total_seconds() / 3600.0

    def _compute_anomaly_rate(self, records: Sequence[Dict[str, object]]) -> float:
        values: List[float] = []
        for record in records:
            candidate = record.get("value")
            if candidate is None:
                continue
            try:
                values.append(float(candidate))
            except (TypeError, ValueError):
                continue
        if len(values) < 3:
            return 0.0

        deltas = [curr - prev for prev, curr in zip(values, values[1:])]
        std = pstdev(deltas) if len(deltas) > 1 else 0.0
        if std == 0:
            return 0.0

        avg = mean(deltas)
        anomalous = sum(
            1
            for delta in deltas
            if abs(delta - avg) > self._config.anomaly_threshold_sigma * std
        )
        return anomalous / len(deltas)

    def _score_dataset(
        self,
        *,
        completeness_ratio: float,
        freshness_hours: float,
        anomaly_rate: float,
    ) -> float:
        """Combine individual metrics into a 0-1 score."""

        freshness_component = 1.0
        freshness_threshold_hours = self._config.freshness_days_threshold * 24
        if freshness_hours > freshness_threshold_hours:
            freshness_component = max(0.0, 1 - (freshness_hours - freshness_threshold_hours) / freshness_threshold_hours)

        anomaly_component = max(0.0, 1 - anomaly_rate)

        components = [
            max(0.0, min(completeness_ratio, 1.0)),
            max(0.0, min(freshness_component, 1.0)),
            max(0.0, min(anomaly_component, 1.0)),
        ]
        return sum(components) / len(components)


class QualityReportBuilder:
    """Generate human readable quality artefacts."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write_html_report(
        self,
        metrics: QualityMetrics,
        issues: Iterable[str],
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        """Persist a simple HTML quality report."""

        context = context or {}
        issues_list = list(issues)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        html = self._render(metrics, issues_list, context)
        self.output_path.write_text(html, encoding="utf-8")

    def _render(
        self,
        metrics: QualityMetrics,
        issues: Sequence[str],
        context: Dict[str, object],
    ) -> str:
        rows = "".join(
            f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"
            for key, value in metrics.to_dict().items()
        )
        issues_html = "".join(f"<li>{issue}</li>" for issue in issues) or "<li>No critical issues detected.</li>"
        context_html = "".join(
            f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in context.items()
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Quality Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 640px; }}
    th, td {{ padding: 0.5rem; border: 1px solid #ccc; text-align: left; }}
    th {{ background: #f5f5f5; width: 40%; }}
  </style>
</head>
<body>
  <h1>Collection Quality Report</h1>
  <p>Generated at {timestamp}</p>
  <h2>Context</h2>
  <table>{context_html}</table>
  <h2>Metrics</h2>
  <table>{rows}</table>
  <h2>Issues</h2>
  <ul>{issues_html}</ul>
</body>
</html>
"""


def write_metadata(path: Path, payload: Dict[str, object]) -> None:
    """Write metadata JSON to disk with indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


__all__ = [
    "QualityEvaluator",
    "QualityMetrics",
    "QualityReportBuilder",
    "write_metadata",
]
