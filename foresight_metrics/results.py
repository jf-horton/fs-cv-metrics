"""
MetricResult - Standardized output from metric computations.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """
    Container for metric computation results.

    Attributes:
        task_name: Name of the vision task (e.g., "object_detection")
        metrics: Dictionary mapping metric names to values
        per_class: Optional per-class breakdown of metrics
        metadata: Optional additional information about the evaluation

    Example:
        >>> result = MetricResult(
        ...     task_name="object_detection",
        ...     metrics={"mAP": 0.75, "precision": 0.82},
        ...     per_class={"person": {"precision": 0.85}, "car": {"precision": 0.78}},
        ... )
        >>> print(result.pretty())
    """

    task_name: str
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        """
        Format results for terminal display.

        Returns:
            Human-readable string representation of metrics.
        """
        lines = [f"=== {self.task_name} ==="]

        for name, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")

        if self.per_class:
            lines.append("  Per-class:")
            for cls_name, cls_metrics in self.per_class.items():
                for metric_name, value in cls_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"    {cls_name}/{metric_name}: {value:.4f}")
                    else:
                        lines.append(f"    {cls_name}/{metric_name}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Export as plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "task_name": self.task_name,
            "metrics": self.metrics,
            "per_class": self.per_class,
            "metadata": self.metadata,
        }
