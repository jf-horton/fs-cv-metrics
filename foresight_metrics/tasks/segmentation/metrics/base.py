"""
Metric protocol for segmentation.
"""
from typing import Protocol

from foresight_metrics.tasks.segmentation.types import SegData


class SegMetric(Protocol):
    """
    Protocol for segmentation metrics.

    Implement this protocol to add a new metric. Metrics should be
    stateless and operate only on the canonical SegData format.

    Attributes:
        name: Unique identifier for this metric (e.g., "iou", "dice").

    Example:
        >>> class MyMetric:
        ...     name = "my_metric"
        ...
        ...     def compute(self, data: SegData) -> dict[str, float]:
        ...         # Calculate metrics and return as dict
        ...         return {"my_metric": 0.85}
    """

    name: str

    def compute(self, data: SegData) -> dict[str, float]:
        """
        Compute metric(s) from internal data.

        Args:
            data: Segmentation data in canonical SegData format.

        Returns:
            Dictionary mapping metric names to their computed values.
            Can return multiple related metrics (e.g., mIoU, per-class IoU).
        """
        ...
