"""
Format adapter protocol for object detection.
"""
from pathlib import Path
from typing import Protocol, Union

import numpy as np

from foresight_metrics.tasks.object_detection.types import ODData


# Type alias for data sources
DataSource = Union[str, Path, np.ndarray, dict]


class ODFormatAdapter(Protocol):
    """
    Protocol for object detection format adapters.

    Implement this protocol to add support for a new data format.
    Format adapters are responsible for loading external data formats
    (COCO, YOLO, Pascal VOC, etc.) and converting them to the internal
    ODData representation.

    Attributes:
        name: Unique identifier for this format (e.g., "coco", "yolo").

    Example:
        >>> class MyFormat:
        ...     name = "my_format"
        ...
        ...     def load(self, ground_truth, predictions) -> ODData:
        ...         # Parse files/data and return ODData
        ...         ...
    """

    name: str

    def load(self, ground_truth: DataSource, predictions: DataSource) -> ODData:
        """
        Load ground truth and predictions, convert to internal ODData format.

        Args:
            ground_truth: Path to file, numpy array, or dict containing GT data.
            predictions: Path to file, numpy array, or dict containing predictions.

        Returns:
            ODData in canonical internal format.

        Raises:
            ValueError: If the data format is invalid or unsupported.
            FileNotFoundError: If a file path is provided but doesn't exist.
        """
        ...
