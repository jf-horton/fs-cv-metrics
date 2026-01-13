"""
Segmentation - Main API for segmentation evaluation.
"""
from typing import Sequence, Type

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

    Example:
        >>> from foresight_metrics.tasks.segmentation import Segmentation
        >>> from foresight_metrics.loggers import StdoutLogger
        >>> import numpy as np
        >>>
        >>> seg = Segmentation(
        ...     data_format="numpy",
        ...     num_classes=3,
        ...     metrics=["iou", "dice"],
        ...     loggers=[StdoutLogger()],
        ... )
        >>> gt = np.array([[[0, 0, 1], [1, 1, 2], [2, 2, 2]]])
        >>> pred = np.array([[[0, 0, 1], [1, 1, 1], [2, 2, 2]]])
        >>> result = seg.evaluate(gt, pred)
        >>> print(result.metrics)

    Extension Example:
        >>> # Add a custom metric
        >>> class MyMetric:
        ...     name = "my_metric"
        ...     def compute(self, data): return {"my_metric": 0.5}
        >>>
        >>> seg = Segmentation(
        ...     num_classes=3,
        ...     metrics=["iou", "my_metric"],
        ...     custom_metrics={"my_metric": MyMetric},
        ... )
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
            MetricResult containing all computed metrics, with fields:
            - task_name: "segmentation"
            - metrics: dict of metric name -> value
            - metadata: additional info about the evaluation

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
