"""Utility helpers for tracking ML training runs and artefacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .path_utils import get_project_root


@dataclass
class TrainingRunRecord:
    model_name: str
    timestamp: str
    artifact_dir: str
    data_sources: List[str]
    data_hash: Optional[str]
    metadata: Dict[str, object]
    metrics: Dict[str, float]


class TrainingRunRegistry:
    """Centralised registry for persisting lightweight training metadata."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (get_project_root() / "data" / "training_runs")
        self.root.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        *,
        model_name: str,
        data_sources: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        metrics: Optional[Dict[str, float]] = None,
        inline_snapshot: Optional[Dict[str, object]] = None,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"{timestamp}_{model_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        resolved_sources: List[str] = []
        hash_inputs: List[bytes] = []

        if data_sources:
            for source in data_sources:
                resolved_sources.append(str(source))
                candidate_path = Path(source)
                if candidate_path.exists():
                    destination = run_dir / candidate_path.name
                    if candidate_path.is_file():
                        shutil.copy2(candidate_path, destination)
                        hash_inputs.append(destination.read_bytes())
                elif source.startswith("http"):
                    hash_inputs.append(source.encode("utf-8"))

        if inline_snapshot:
            snapshot_file = run_dir / "inline_snapshot.json"
            snapshot_file.write_text(json.dumps(inline_snapshot, indent=2, default=str))
            hash_inputs.append(snapshot_file.read_bytes())

        data_hash = self._hash_inputs(hash_inputs) if hash_inputs else None

        record = TrainingRunRecord(
            model_name=model_name,
            timestamp=timestamp,
            artifact_dir=str(run_dir),
            data_sources=resolved_sources,
            data_hash=data_hash,
            metadata=metadata or {},
            metrics=metrics or {},
        )

        manifest_path = run_dir / "training_run.json"
        manifest_path.write_text(json.dumps(asdict(record), indent=2, default=str))
        return manifest_path

    @staticmethod
    def _hash_inputs(chunks: List[bytes]) -> str:
        digest = hashlib.md5()
        for chunk in chunks:
            digest.update(chunk)
        return digest.hexdigest()


__all__ = ["TrainingRunRegistry"]
