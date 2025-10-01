"""Financial data collection agent that orchestrates API ingestion and quality checks."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from .api_clients import FetchResult, build_client
from .config import AgentConfig, load_config
from .logging_utils import get_logger, setup_logging
from .metadata import compile_metadata_document
from .quality import QualityEvaluator, QualityReportBuilder, write_metadata
from .strategies import CircuitBreaker, CollectionTask, StrategyPlanner, TaskOutcome


logger = get_logger(__name__)


class FinancialDataCollectionAgent:
    """Agent responsible for fetching, persisting, and validating datasets."""

    def __init__(self, config_path: Path) -> None:
        self.config = load_config(config_path)
        log_path = (config_path.parent / "../logs/collection.log").resolve()
        setup_logging(log_path)
        self.quality_evaluator = QualityEvaluator(self.config.quality)
        self.quality_report_path = (config_path.parent / "../reports/quality_report.html").resolve()
        self.collection_summary_path = (config_path.parent / "../reports/collection_summary.md").resolve()
        self.metadata_path = (config_path.parent / "../data/metadata/collection_metadata.json").resolve()

    def run(self) -> None:
        """Execute the full collection workflow."""

        logger.info("Starting financial data collection agent")
        planner = StrategyPlanner(self.config)
        tasks = planner.build_tasks()
        breaker = CircuitBreaker()
        clients = {name: build_client(cfg, self.config.collection) for name, cfg in self.config.sources.items()}

        results: List[FetchResult] = []
        per_task_issues: Dict[str, List[str]] = {}

        for task in tasks:
            if breaker.should_skip(task):
                logger.warning("Skipping task %s due to open circuit breaker", task.describe())
                per_task_issues.setdefault(task.task_id, []).append("Skipped due to repeated failures earlier")
                continue

            logger.info("Executing task: %s", task.describe())
            try:
                result = self._execute_task(task, clients[task.source_name])
                results.append(result)
                outcome = TaskOutcome(
                    task_id=task.task_id,
                    success=True,
                    latency_seconds=result.latency_seconds,
                    timestamp=result.fetched_at,
                )
                breaker.record(outcome)
                dataset_issues = self._evaluate_dataset(result)
                if dataset_issues:
                    per_task_issues[task.task_id] = dataset_issues
                self._persist_result(result)
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Task %s failed: %s", task.task_id, exc)
                outcome = TaskOutcome(
                    task_id=task.task_id,
                    success=False,
                    latency_seconds=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error=str(exc),
                )
                breaker.record(outcome)
                per_task_issues.setdefault(task.task_id, []).append(str(exc))

        if not results:
            logger.error("No datasets collected; aborting report generation")
            return

        self._generate_global_reports(results, per_task_issues)
        logger.info("Collection workflow complete")

    def _execute_task(self, task: CollectionTask, client) -> FetchResult:
        """Execute the appropriate call for the client/task combination."""

        if hasattr(client, "fetch_daily_series") and task.symbol:
            return client.fetch_daily_series(task.symbol, function=task.params.get("function"))
        if hasattr(client, "fetch_series") and task.series_id:
            return client.fetch_series(task.series_id)
        if hasattr(client, "fetch_profile") and task.symbol:
            return client.fetch_profile(task.symbol)

        payload, latency = client.get(task.params)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                records = data
            else:
                records = [payload]
        else:
            records = [{"raw": payload}]
        return FetchResult(
            source=task.source_name,
            task_id=task.task_id,
            success=True,
            payload=records,
            raw=payload,
            fetched_at=datetime.now(timezone.utc),
            latency_seconds=latency,
        )

    def _persist_result(self, result: FetchResult) -> None:
        """Persist raw and processed datasets to disk."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_filename = f"{result.task_id.replace(':', '_')}_{timestamp}.json"
        processed_filename = f"{result.task_id.replace(':', '_')}_{timestamp}.csv"

        raw_path = self.config.storage.raw_data_dir / raw_filename
        processed_path = self.config.storage.processed_data_dir / processed_filename

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        raw_path.write_text(json.dumps(result.raw or {}, indent=2, default=str), encoding="utf-8")
        self._write_processed_csv(processed_path, result.payload)

    def _write_processed_csv(self, path: Path, records: Sequence[Dict[str, object]]) -> None:
        """Write processed records enriched with change metrics."""

        if not records:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = sorted({key for record in records for key in record.keys()})
        if "percent_change" not in fieldnames:
            fieldnames.append("percent_change")

        previous_value: float | None = None
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                current = record.copy()
                percent_change = None
                value = current.get("value")
                try:
                    numeric_value = float(value) if value is not None else None
                except (TypeError, ValueError):
                    numeric_value = None
                if numeric_value is not None and previous_value not in (None, 0):
                    percent_change = (numeric_value - previous_value) / previous_value
                current["percent_change"] = percent_change
                writer.writerow(current)
                previous_value = numeric_value

    def _evaluate_dataset(self, result: FetchResult) -> List[str]:
        """Evaluate dataset quality and return issue list."""

        metrics, issues = self.quality_evaluator.evaluate(result.payload)
        logger.info(
            "Quality metrics for %s | records=%s quality_score=%.2f",
            result.task_id,
            metrics.total_records,
            metrics.quality_score,
        )
        return issues

    def _generate_global_reports(
        self,
        results: Sequence[FetchResult],
        per_task_issues: Dict[str, List[str]],
    ) -> None:
        """Generate metadata, quality report, and collection summary."""

        combined_records = []
        for result in results:
            combined_records.extend(result.payload)
        metrics, issues = self.quality_evaluator.evaluate(combined_records)
        issues.extend([f"{task}: {issue}" for task, task_issues in per_task_issues.items() for issue in task_issues])

        report_builder = QualityReportBuilder(self.quality_report_path)
        context = {
            "datasets": len(results),
            "total_records": metrics.total_records,
        }
        report_builder.write_html_report(metrics, issues, context)

        metadata_doc = compile_metadata_document(results, per_task_issues)
        write_metadata(self.metadata_path, metadata_doc)

        self._write_collection_summary(results, metrics, per_task_issues)

    def _write_collection_summary(
        self,
        results: Sequence[FetchResult],
        metrics,
        per_task_issues: Dict[str, List[str]],
    ) -> None:
        """Persist collection summary markdown."""

        lines = ["# Collection Summary", ""]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Total datasets: {len(results)}")
        lines.append(f"Total records: {metrics.total_records}")
        lines.append("")

        for result in results:
            lines.append(f"## {result.task_id}")
            lines.append(f"- Source: {result.source}")
            lines.append(f"- Records: {len(result.payload)}")
            lines.append(f"- Latency (s): {result.latency_seconds:.2f}")
            issues = per_task_issues.get(result.task_id, [])
            if issues:
                lines.append("- Issues: " + "; ".join(issues))
            else:
                lines.append("- Issues: None")
            lines.append("")

        self.collection_summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.collection_summary_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = ["FinancialDataCollectionAgent"]
