"""
foresight_metrics.tasks - Vision task implementations.
"""
from foresight_metrics.tasks.object_detection import ObjectDetection
from foresight_metrics.tasks.segmentation import Segmentation

__all__ = [
    "ObjectDetection",
    "Segmentation",
]
