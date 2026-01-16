"""
Segmentation - Main API for segmentation evaluation.
"""
from typing import Sequence, Type

import numpy as np

from foresight_metrics.results import MetricResult
from foresight_metrics.loggers.base import MetricsLogger
from foresight_metrics.tasks.segmentation.types import SegData
from foresight_metrics.tasks.segmentation.formats.numpy_format import NumpyFormat
from foresight_metrics.tasks.segmentation.formats.base import SegFormatAdapter
from foresight_metrics.tasks.segmentation.metrics.base import SegMetric
from foresight_metrics.tasks.segmentation.metrics.iou import IoUMetric
from foresight_metrics.tasks.segmentation.metrics.dice import DiceMetric, PixelAccuracyMetric


# Built-in format adapters
BUILTIN_FORMATS: dict[str, Type] = {
    "numpy": NumpyFormat,
}

# Built-in metrics
BUILTIN_METRICS: dict[str, Type] = {
    "iou": IoUMetric,
    "dice": DiceMetric,
    "pixel_accuracy": PixelAccuracyMetric,
}


class Segmentation:
    """
    Evaluate semantic segmentation models.

    This is the main entry point for computing segmentation metrics.
    It handles loading data from various formats, computing metrics, and
    logging results to configured destinations.

    Attributes:
        data_format: Name of the data format being used.
        metric_names: Names of metrics to compute.
        loggers: List of loggers for output.

    Example (update/compute pattern):
        >>> seg = Segmentation(num_classes=3, loggers=[StdoutLogger()])
        >>> for image, annotation in dataset:
        ...     pred = model.predict(image)
        ...     seg.update(gt_mask=annotation, pred_mask=pred)
        >>> result = seg.compute()

    Example (file-based):
        >>> seg = Segmentation(num_classes=3, data_format="numpy")
        >>> result = seg.evaluate(gt_masks, pred_masks)
    """

    def __init__(
        self,
        num_classes: int,
        data_format: str = "numpy",
        metrics: Sequence[str] | None = None,
        loggers: Sequence[MetricsLogger] | None = None,
        custom_formats: dict[str, Type[SegFormatAdapter]] | None = None,
        custom_metrics: dict[str, Type[SegMetric]] | None = None,
        ignore_index: int = 255,
    ) -> None:
        """
        Initialize the Segmentation evaluator.

        Args:
            num_classes: Number of classes in the dataset (required).
            data_format: Name of the data format ("numpy", etc.).
                        Use custom_formats to add new formats.
            metrics: List of metric names to compute. Defaults to all
                    built-in metrics. Use custom_metrics to add new metrics.
            loggers: List of loggers for output (StdoutLogger, ClearMLLogger, etc.).
            custom_formats: Additional format adapters as {name: AdapterClass}.
            custom_metrics: Additional metrics as {name: MetricClass}.
            ignore_index: Class index to ignore in metric computation.

        Raises:
            ValueError: If data_format or metric name is not recognized.
        """
        self._num_classes = num_classes
        self._ignore_index = ignore_index

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

        # Internal accumulators for update/compute pattern
        self._gt_masks: list[np.ndarray] = []
        self._pred_masks: list[np.ndarray] = []

    @property
    def data_format(self) -> str:
        """Name of the configured data format."""
        return self._format_adapter.name

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        return self._num_classes

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
        self._gt_masks = []
        self._pred_masks = []

    def update(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
        """
        Add masks for one image to internal accumulator.

        Call this for each image during inference, then call compute()
        to get final metrics.

        Args:
            gt_mask: Ground truth mask [H, W] with class indices.
            pred_mask: Predicted mask [H, W] with class indices.

        Example:
            >>> seg = Segmentation(num_classes=3)
            >>> for image, annotation in dataset:
            ...     pred = model.predict(image)
            ...     seg.update(gt_mask=annotation, pred_mask=pred)
        """
        gt_mask = np.asarray(gt_mask)
        pred_mask = np.asarray(pred_mask)
        self._gt_masks.append(gt_mask)
        self._pred_masks.append(pred_mask)

    def compute(self) -> MetricResult:
        """
        Compute metrics on accumulated data.

        This method builds SegData from accumulated updates, computes all
        requested metrics, logs results, and resets the internal state.

        Returns:
            MetricResult containing all computed metrics.

        Raises:
            ValueError: If no data has been accumulated via update().

        Example:
            >>> seg = Segmentation(num_classes=3)
            >>> for image, annotation in dataset:
            ...     seg.update(gt_mask=annotation, pred_mask=pred)
            >>> result = seg.compute()  # Computes and auto-resets
        """
        if len(self._gt_masks) == 0:
            raise ValueError("No data accumulated. Call update() before compute().")

        # Build SegData from accumulated data
        data = SegData(
            gt_masks=np.stack(self._gt_masks),
            pred_masks=np.stack(self._pred_masks),
            num_classes=self._num_classes,
            ignore_index=self._ignore_index,
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
            task_name="segmentation",
            metrics=all_metrics,
            metadata={
                "format": "accumulated",
                "num_images": data.num_images,
                "num_classes": data.num_classes,
                "image_shape": data.image_shape,
                "total_pixels": data.total_pixels,
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
        Evaluate predictions against ground truth.

        This method:
        1. Loads data using the configured format adapter
        2. Computes all requested metrics
        3. Logs results to all configured loggers
        4. Returns a MetricResult with all computed values

        Args:
            ground_truth: Path to GT file(s), or numpy array of masks.
            predictions: Path to prediction file(s), or numpy array of masks.

        Returns:
            MetricResult containing all computed metrics.

        Raises:
            ValueError: If data format is invalid.
            FileNotFoundError: If file paths don't exist.
        """
        # 1. Load and convert to internal format
        data: SegData = self._format_adapter.load(
            ground_truth, predictions, self._num_classes
        )
        # Override ignore_index if specified
        data.ignore_index = self._ignore_index

        # 2. Compute all requested metrics
        all_metrics: dict[str, float] = {}
        for name in self._metric_names:
            metric_cls = self._metrics[name]
            metric_instance = metric_cls()
            result = metric_instance.compute(data)
            all_metrics.update(result)

        # 3. Build result
        result = MetricResult(
            task_name="segmentation",
            metrics=all_metrics,
            metadata={
                "format": self._format_adapter.name,
                "num_images": data.num_images,
                "num_classes": data.num_classes,
                "image_shape": data.image_shape,
                "total_pixels": data.total_pixels,
            },
        )

        # 4. Log to all loggers
        for logger in self._loggers:
            logger.log(result)

        return result

