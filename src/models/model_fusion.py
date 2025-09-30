"""Scaffolding for future model-fusion experiments (Recommendation 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class EnsemblePrediction:
    label: Any
    confidence: float
    details: Dict[str, Any]


class CourseSearchEnsemble:
    """
    Lightweight wrapper that can combine multiple classifiers without
    disrupting the current production pipeline. The ensemble currently defers
    to the primary model and records secondary opinions for later analysis.
    """

    def __init__(
        self,
        primary_model,
        *,
        secondary_model=None,
        blend_weight: float = 0.5,
    ) -> None:
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.blend_weight = blend_weight

    def predict(self, url: str, html_content: str) -> EnsemblePrediction:
        """
        Execute the primary classifier and, if available, collect secondary
        model outputs for offline comparison. Confidence blending is gated so
        the existing behaviour remains unchanged until explicitly enabled.
        """
        primary_result = self._call_model(self.primary_model, url, html_content)

        if not self.secondary_model:
            return primary_result

        secondary_result = self._call_model(self.secondary_model, url, html_content)

        blended_confidence = self._blend_confidence(
            primary_result.confidence,
            secondary_result.confidence,
        )

        merged_details = {
            "primary": primary_result.details,
            "secondary": secondary_result.details,
        }

        return EnsemblePrediction(
            label=primary_result.label,
            confidence=blended_confidence,
            details=merged_details,
        )

    def _blend_confidence(self, primary: float, secondary: float) -> float:
        if self.secondary_model is None:
            return primary
        return max(0.0, min(1.0, (primary * (1 - self.blend_weight)) + (secondary * self.blend_weight)))

    @staticmethod
    def _call_model(model, url: str, html_content: str) -> EnsemblePrediction:
        prediction, confidence, details = model.predict_course_search(url, html_content)
        label = details.get("label") if isinstance(details, dict) else prediction
        return EnsemblePrediction(label=label, confidence=confidence, details=details or {})

    def update_secondary(self, model, *, weight: Optional[float] = None) -> None:
        """Attach or update the secondary model and optional blending weight."""
        self.secondary_model = model
        if weight is not None:
            self.blend_weight = weight


__all__ = ["CourseSearchEnsemble", "EnsemblePrediction"]
