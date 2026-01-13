"""
foresight_metrics - Computer Vision Metrics Package

A Python utility package for computing computer vision metrics with a focus
on maintainability, readability, and extensibility.

Example:
    >>> from foresight_metrics import ObjectDetection, StdoutLogger
    >>>
    >>> od = ObjectDetection(
    ...     data_format="coco",
    ...     metrics=["mAP", "precision_recall"],
    ...     loggers=[StdoutLogger()],
    ... )
    >>> result = od.evaluate("ground_truth.json", "predictions.json")
    >>> print(result.pretty())

Supported Tasks:
    - ObjectDetection: mAP, precision, recall, F1 metrics
    - Segmentation: mIoU, Dice, pixel accuracy metrics

Supported Formats:
    - COCO: Standard COCO JSON annotation format (object detection)
    - Numpy: Direct numpy array input (segmentation)

Loggers:
    - StdoutLogger: Print metrics to terminal
    - ClearMLLogger: Log metrics to ClearML experiment tracker
"""
from foresight_metrics.results import MetricResult
from foresight_metrics.loggers import StdoutLogger, ClearMLLogger
from foresight_metrics.tasks.object_detection import ObjectDetection
from foresight_metrics.tasks.segmentation import Segmentation

__version__ = "0.1.0"

__all__ = [
    # Core
    "MetricResult",
    # Tasks
    "ObjectDetection",
    "Segmentation",
    # Loggers
    "StdoutLogger",
    "ClearMLLogger",
]
