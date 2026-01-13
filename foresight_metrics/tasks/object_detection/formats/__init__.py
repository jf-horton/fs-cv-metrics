"""
foresight_metrics.tasks.object_detection.formats - Format adapters for object detection.
"""
from foresight_metrics.tasks.object_detection.formats.coco import CocoFormat
from foresight_metrics.tasks.object_detection.formats.internal import InternalFormat

__all__ = [
    "CocoFormat",
    "InternalFormat",
]
