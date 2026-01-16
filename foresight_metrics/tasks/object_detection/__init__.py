"""
ObjectDetection - Main API for object detection evaluation.
"""
from typing import Sequence, Type

import numpy as np

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

# Built-in metrics (all available)
BUILTIN_METRICS: dict[str, Type] = {
    "mAP": MAPMetric,
    "precision_recall": PrecisionRecallMetric,
}

# Standard metrics (default when not specified)
STANDARD_METRICS: list[str] = ["mAP", "precision_recall"]


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

    Example (update/compute pattern):
        >>> od = ObjectDetection(metrics=["mAP"], loggers=[StdoutLogger()])
        >>> for image, annotation in dataset:
        ...     preds = model.predict(image)
        ...     od.update(
        ...         gt_boxes=annotation.xyxy,
        ...         gt_labels=annotation.class_id,
        ...         pred_boxes=preds.xyxy,
        ...         pred_labels=preds.class_id,
        ...         pred_scores=preds.confidence,
        ...     )
        >>> result = od.compute()

    Example (file-based):
        >>> od = ObjectDetection(data_format="coco")
        >>> result = od.evaluate("ground_truth.json", "predictions.json")
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
        self._metric_names = list(metrics) if metrics else STANDARD_METRICS.copy()
        self._loggers = list(loggers) if loggers else []

        # Validate metric names
        for name in self._metric_names:
            if name not in self._metrics:
                available = list(self._metrics.keys())
                raise ValueError(
                    f"Unknown metric '{name}'. Available metrics: {available}"
                )

        # Internal accumulators for update/compute pattern
        self._image_count = 0
        self._gt_image_ids: list[int] = []
        self._gt_boxes: list[np.ndarray] = []
        self._gt_labels: list[int] = []
        self._pred_image_ids: list[int] = []
        self._pred_boxes: list[np.ndarray] = []
        self._pred_labels: list[int] = []
        self._pred_scores: list[float] = []
        self._class_names: dict[int, str] | None = None

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

    def reset(self) -> None:
        """Clear all accumulated data for a fresh evaluation."""
        self._image_count = 0
        self._gt_image_ids = []
        self._gt_boxes = []
        self._gt_labels = []
        self._pred_image_ids = []
        self._pred_boxes = []
        self._pred_labels = []
        self._pred_scores = []

    def update(
        self,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        pred_boxes: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
    ) -> None:
        """
        Add detections for one image to internal accumulator.

        Call this for each image during inference, then call compute()
        to get final metrics.

        Args:
            gt_boxes: Ground truth boxes [N, 4] in xyxy format.
            gt_labels: Ground truth class labels [N].
            pred_boxes: Predicted boxes [M, 4] in xyxy format.
            pred_labels: Predicted class labels [M].
            pred_scores: Prediction confidence scores [M].

        Example:
            >>> od = ObjectDetection()
            >>> for image, annotation in dataset:
            ...     preds = model.predict(image)
            ...     od.update(
            ...         gt_boxes=annotation.xyxy,
            ...         gt_labels=annotation.class_id,
            ...         pred_boxes=preds.xyxy,
            ...         pred_labels=preds.class_id,
            ...         pred_scores=preds.confidence,
            ...     )
        """
        image_id = self._image_count
        self._image_count += 1

        # Accumulate ground truth
        if gt_boxes is not None and len(gt_boxes) > 0:
            gt_boxes = np.asarray(gt_boxes)
            gt_labels = np.asarray(gt_labels)
            self._gt_image_ids.extend([image_id] * len(gt_boxes))
            self._gt_boxes.append(gt_boxes)
            self._gt_labels.extend(gt_labels.tolist())

        # Accumulate predictions
        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_boxes = np.asarray(pred_boxes)
            pred_labels = np.asarray(pred_labels)
            pred_scores = np.asarray(pred_scores)
            self._pred_image_ids.extend([image_id] * len(pred_boxes))
            self._pred_boxes.append(pred_boxes)
            self._pred_labels.extend(pred_labels.tolist())
            self._pred_scores.extend(pred_scores.tolist())

    def compute(self) -> MetricResult:
        """
        Compute metrics on accumulated data.

        This method builds ODData from accumulated updates, computes all
        requested metrics, logs results, and resets the internal state.

        Returns:
            MetricResult containing all computed metrics.

        Raises:
            ValueError: If no data has been accumulated via update().

        Example:
            >>> od = ObjectDetection()
            >>> for image, annotation in dataset:
            ...     od.update(gt_boxes=..., pred_boxes=...)
            >>> result = od.compute()  # Computes and auto-resets
        """
        if self._image_count == 0:
            raise ValueError("No data accumulated. Call update() before compute().")

        # Build ODData from accumulated data
        data = ODData(
            gt_image_ids=np.array(self._gt_image_ids, dtype=np.int64),
            gt_boxes=np.vstack(self._gt_boxes).astype(np.float32) if self._gt_boxes else np.empty((0, 4), dtype=np.float32),
            gt_labels=np.array(self._gt_labels, dtype=np.int64),
            pred_image_ids=np.array(self._pred_image_ids, dtype=np.int64),
            pred_boxes=np.vstack(self._pred_boxes).astype(np.float32) if self._pred_boxes else np.empty((0, 4), dtype=np.float32),
            pred_labels=np.array(self._pred_labels, dtype=np.int64),
            pred_scores=np.array(self._pred_scores, dtype=np.float32),
            class_names=self._class_names,
        )

        # Compute metrics
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
                "format": "accumulated",
                "num_gt_boxes": data.num_gt_boxes,
                "num_predictions": data.num_predictions,
                "num_images": data.num_images,
                "num_classes": len(data.unique_labels),
            },
        )

        # Log to all loggers
        for logger in self._loggers:
            logger.log(result)

        # Auto-reset after compute
        self.reset()

        return result

    def evaluate(self, ground_truth, predictions) -> MetricResult:
        """
        Evaluate predictions against ground truth from files.

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
            MetricResult containing all computed metrics.

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

        Use this when you've already constructed ODData in memory.

        Args:
            data: ODData containing ground truth and predictions.

        Returns:
            MetricResult containing all computed metrics.
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

