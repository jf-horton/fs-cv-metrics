"""
Metric protocol for object detection.
"""
from typing import Protocol

from foresight_metrics.tasks.object_detection.types import ODData


class ODMetric(Protocol):
    """
    Protocol for object detection metrics.

    Implement this protocol to add a new metric. Metrics should be
    stateless and operate only on the canonical ODData format.

    Attributes:
        name: Unique identifier for this metric (e.g., "mAP", "precision").

    Example:
        >>> class MyMetric:
        ...     name = "my_metric"
        ...
        ...     def compute(self, data: ODData) -> dict[str, float]:
        ...         # Calculate metrics and return as dict
        ...         return {"my_metric": 0.85}
    """

    name: str

    def compute(self, data: ODData) -> dict[str, float]:
        """
        Compute metric(s) from internal data.

        Args:
            data: Object detection data in canonical ODData format.

        Returns:
            Dictionary mapping metric names to their computed values.
            Can return multiple related metrics (e.g., mAP@0.5, mAP@0.75).
        """
        ...
