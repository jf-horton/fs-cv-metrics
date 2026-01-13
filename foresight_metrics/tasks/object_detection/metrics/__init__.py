"""
foresight_metrics.tasks.object_detection.metrics - Metrics for object detection.
"""
from foresight_metrics.tasks.object_detection.metrics.map import MAPMetric
from foresight_metrics.tasks.object_detection.metrics.precision_recall import PrecisionRecallMetric

__all__ = [
    "MAPMetric",
    "PrecisionRecallMetric",
]
