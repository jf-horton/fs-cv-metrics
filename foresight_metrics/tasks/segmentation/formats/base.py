"""
Format adapter protocol for segmentation.
"""
from pathlib import Path
from typing import Protocol, Union

import numpy as np

from foresight_metrics.tasks.segmentation.types import SegData


# Type alias for data sources
DataSource = Union[str, Path, np.ndarray, dict, list]


class SegFormatAdapter(Protocol):
    """
    Protocol for segmentation format adapters.

    Implement this protocol to add support for a new data format.
    Format adapters are responsible for loading external data formats
    (COCO, PNG masks, numpy arrays, etc.) and converting them to the
    internal SegData representation.

    Attributes:
        name: Unique identifier for this format (e.g., "coco", "png").

    Example:
        >>> class MyFormat:
        ...     name = "my_format"
        ...
        ...     def load(self, ground_truth, predictions, num_classes) -> SegData:
        ...         # Parse files/data and return SegData
        ...         ...
    """

    name: str

    def load(
        self,
        ground_truth: DataSource,
        predictions: DataSource,
        num_classes: int,
    ) -> SegData:
        """
        Load ground truth and predictions, convert to internal SegData format.

        Args:
            ground_truth: Path to file(s), numpy array, or dict containing GT masks.
            predictions: Path to file(s), numpy array, or dict containing pred masks.
            num_classes: Number of classes in the dataset.

        Returns:
            SegData in canonical internal format.

        Raises:
            ValueError: If the data format is invalid or unsupported.
            FileNotFoundError: If a file path is provided but doesn't exist.
        """
        ...
