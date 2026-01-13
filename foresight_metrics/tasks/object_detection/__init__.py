"""
ObjectDetection - Main API for object detection evaluation.
"""
from typing import Sequence, Type

from foresight_metrics.results import MetricResult
from foresight_metrics.loggers.base import MetricsLogger
from foresight_metrics.tasks.object_detection.types import ODData
from foresight_metrics.tasks.object_detection.formats.coco import CocoFormat
from foresight_metrics.tasks.object_detection.formats.base import ODFormatAdapter
from foresight_metrics.tasks.object_detection.metrics.base import ODMetric
from foresight_metrics.tasks.object_detection.metrics.map import MAPMetric
from foresight_metrics.tasks.object_detection.metrics.precision_recall import PrecisionRecallMetric


# Built-in format adapters
BUILTIN_FORMATS: dict[str, Type] = {
    "coco": CocoFormat,
}

# Built-in metrics
BUILTIN_METRICS: dict[str, Type] = {
    "mAP": MAPMetric,
    "precision_recall": PrecisionRecallMetric,
}


class ObjectDetection:
    """
    Evaluate object detection models.

    This is the main entry point for computing object detection metrics.
    It handles loading data from various formats, computing metrics, and
    logging results to configured destinations.

    Attributes:
        data_format: Name of the data format being used.
        metric_names: Names of metrics to compute.
        loggers: List of loggers for output.

    Example:
        >>> from foresight_metrics.tasks.object_detection import ObjectDetection
        >>> from foresight_metrics.loggers import StdoutLogger, ClearMLLogger
        >>>
        >>> od = ObjectDetection(
        ...     data_format="coco",
        ...     metrics=["mAP", "precision_recall"],
        ...     loggers=[StdoutLogger()],
        ... )
        >>> result = od.evaluate("ground_truth.json", "predictions.json")
        >>> print(result.metrics)
        {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48, 'precision': 0.85, ...}

    Extension Example:
        >>> # Add a custom format
        >>> class YoloFormat:
        ...     name = "yolo"
        ...     def load(self, gt, preds): ...
        >>>
        >>> od = ObjectDetection(
        ...     data_format="yolo",
        ...     custom_formats={"yolo": YoloFormat},
        ... )

        >>> # Add a custom metric
        >>> class MyMetric:
        ...     name = "my_metric"
        ...     def compute(self, data): return {"my_metric": 0.5}
        >>>
        >>> od = ObjectDetection(
        ...     metrics=["mAP", "my_metric"],
        ...     custom_metrics={"my_metric": MyMetric},
        ... )
    """

    def __init__(
        self,
        data_format: str = "coco",
        metrics: Sequence[str] | None = None,
        loggers: Sequence[MetricsLogger] | None = None,
        custom_formats: dict[str, Type[ODFormatAdapter]] | None = None,
        custom_metrics: dict[str, Type[ODMetric]] | None = None,
    ) -> None:
        """
        Initialize the ObjectDetection evaluator.

        Args:
            data_format: Name of the data format ("coco", "yolo", etc.).
                        Use custom_formats to add new formats.
            metrics: List of metric names to compute. Defaults to all
                    built-in metrics. Use custom_metrics to add new metrics.
            loggers: List of loggers for output (StdoutLogger, ClearMLLogger, etc.).
            custom_formats: Additional format adapters as {name: AdapterClass}.
            custom_metrics: Additional metrics as {name: MetricClass}.

        Raises:
            ValueError: If data_format or metric name is not recognized.
        """
        # Merge built-ins with custom
        self._formats = {**BUILTIN_FORMATS, **(custom_formats or {})}
        self._metrics = {**BUILTIN_METRICS, **(custom_metrics or {})}

        # Validate format
        if data_format not in self._formats:
            available = list(self._formats.keys())
            raise ValueError(
                f"Unknown format '{data_format}'. Available formats: {available}"
            )

        self._format_adapter = self._formats[data_format]()
        self._metric_names = list(metrics) if metrics else list(BUILTIN_METRICS.keys())
        self._loggers = list(loggers) if loggers else []

        # Validate metric names
        for name in self._metric_names:
            if name not in self._metrics:
                available = list(self._metrics.keys())
                raise ValueError(
                    f"Unknown metric '{name}'. Available metrics: {available}"
                )

    @property
    def data_format(self) -> str:
        """Name of the configured data format."""
        return self._format_adapter.name

    @property
    def metric_names(self) -> list[str]:
        """Names of metrics that will be computed."""
        return self._metric_names.copy()

    @property
    def available_formats(self) -> list[str]:
        """All available format names (built-in + custom)."""
        return list(self._formats.keys())

    @property
    def available_metrics(self) -> list[str]:
        """All available metric names (built-in + custom)."""
        return list(self._metrics.keys())

    def evaluate(self, ground_truth, predictions) -> MetricResult:
        """
        Evaluate predictions against ground truth.

        This method:
        1. Loads data using the configured format adapter
        2. Computes all requested metrics
        3. Logs results to all configured loggers
        4. Returns a MetricResult with all computed values

        Args:
            ground_truth: Path to GT file, or data structure in the
                         configured format.
            predictions: Path to predictions file, or data structure
                        in the configured format.

        Returns:
            MetricResult containing all computed metrics, with fields:
            - task_name: "object_detection"
            - metrics: dict of metric name -> value
            - metadata: additional info about the evaluation

        Raises:
            ValueError: If data format is invalid.
            FileNotFoundError: If file paths don't exist.
        """
        # 1. Load and convert to internal format
        data: ODData = self._format_adapter.load(ground_truth, predictions)

        # 2. Compute all requested metrics
        all_metrics: dict[str, float] = {}
        for name in self._metric_names:
            metric_cls = self._metrics[name]
            metric_instance = metric_cls()
            result = metric_instance.compute(data)
            all_metrics.update(result)

        # 3. Build result
        result = MetricResult(
            task_name="object_detection",
            metrics=all_metrics,
            metadata={
                "format": self._format_adapter.name,
                "num_gt_boxes": data.num_gt_boxes,
                "num_predictions": data.num_predictions,
                "num_images": data.num_images,
                "num_classes": len(data.unique_labels),
            },
        )

        # 4. Log to all loggers
        for logger in self._loggers:
            logger.log(result)

        return result

    def evaluate_data(self, data: ODData) -> MetricResult:
        """
        Evaluate metrics directly from ODData.

        Use this when you've already constructed ODData in memory,
        e.g., from running inference in a loop. This bypasses the
        format adapter entirely.

        Args:
            data: ODData containing ground truth and predictions.

        Returns:
            MetricResult containing all computed metrics.

        Example:
            >>> from foresight_metrics.tasks.object_detection.types import ODData
            >>> import numpy as np
            >>>
            >>> data = ODData(
            ...     gt_image_ids=np.array([0, 0]),
            ...     gt_boxes=np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            ...     gt_labels=np.array([0, 1]),
            ...     pred_image_ids=np.array([0, 0]),
            ...     pred_boxes=np.array([[12, 12, 48, 48], [58, 58, 102, 102]]),
            ...     pred_labels=np.array([0, 1]),
            ...     pred_scores=np.array([0.9, 0.85]),
            ... )
            >>> od = ObjectDetection()
            >>> result = od.evaluate_data(data)
        """
        # Compute all requested metrics
        all_metrics: dict[str, float] = {}
        for name in self._metric_names:
            metric_cls = self._metrics[name]
            metric_instance = metric_cls()
            result = metric_instance.compute(data)
            all_metrics.update(result)

        # Build result
        result = MetricResult(
            task_name="object_detection",
            metrics=all_metrics,
            metadata={
                "format": "internal",
                "num_gt_boxes": data.num_gt_boxes,
                "num_predictions": data.num_predictions,
                "num_images": data.num_images,
                "num_classes": len(data.unique_labels),
            },
        )

        # Log to all loggers
        for logger in self._loggers:
            logger.log(result)

        return result
