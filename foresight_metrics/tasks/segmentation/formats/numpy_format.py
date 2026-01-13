"""
Numpy array format adapter for segmentation.

This is the simplest format - directly accepts numpy arrays of masks.
"""
from pathlib import Path
from typing import Union

import numpy as np

from foresight_metrics.tasks.segmentation.types import SegData


class NumpyFormat:
    """
    Format adapter for numpy arrays.

    Accepts ground truth and predictions as numpy arrays directly.
    This is the simplest format adapter and is useful when masks
    are already loaded in memory.

    Expected format:
        - Ground truth: np.ndarray of shape [N, H, W] or [H, W]
        - Predictions: np.ndarray of shape [N, H, W] or [H, W]
        - Values are class labels (integers)

    Example:
        >>> adapter = NumpyFormat()
        >>> gt = np.array([[[0, 0, 1], [1, 1, 2], [2, 2, 2]]])
        >>> pred = np.array([[[0, 0, 1], [1, 1, 1], [2, 2, 2]]])
        >>> data = adapter.load(gt, pred, num_classes=3)
    """

    name = "numpy"

    def load(
        self,
        ground_truth: Union[np.ndarray, str, Path],
        predictions: Union[np.ndarray, str, Path],
        num_classes: int,
    ) -> SegData:
        """
        Load numpy arrays and convert to SegData.

        Args:
            ground_truth: Numpy array of shape [N, H, W] or [H, W], or path to .npy file.
            predictions: Numpy array of shape [N, H, W] or [H, W], or path to .npy file.
            num_classes: Number of classes in the dataset.

        Returns:
            SegData with masks in canonical format.
        """
        gt_masks = self._load_array(ground_truth)
        pred_masks = self._load_array(predictions)

        return SegData(
            gt_masks=gt_masks,
            pred_masks=pred_masks,
            num_classes=num_classes,
        )

    def _load_array(self, source: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load array from numpy array or .npy file."""
        if isinstance(source, np.ndarray):
            return source
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == ".npy":
            return np.load(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .npy files.")
