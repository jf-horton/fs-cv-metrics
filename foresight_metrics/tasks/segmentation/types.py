"""
Internal data types for semantic/instance segmentation.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SegData:
    """
    Internal representation of segmentation data for metric computation.

    All metrics operate on this format. Format adapters convert external
    formats (COCO, PNG masks, etc.) into this structure.

    Supports both semantic and instance segmentation:
    - For semantic segmentation: use gt_masks and pred_masks as 2D label maps
    - For instance segmentation: use per-instance masks with labels

    Attributes:
        gt_masks: Ground truth segmentation masks. Shape [N, H, W] for N images,
                  where each pixel contains the class label.
        pred_masks: Predicted segmentation masks. Shape [N, H, W].
        class_names: Optional mapping from class ID to class name.
        ignore_index: Class index to ignore in metric computation (default: 255).
        num_classes: Number of classes (excluding ignore index).

    Example (semantic segmentation):
        >>> data = SegData(
        ...     gt_masks=np.array([[[0, 0, 1], [1, 1, 2], [2, 2, 2]]]),  # [1, 3, 3]
        ...     pred_masks=np.array([[[0, 0, 1], [1, 1, 1], [2, 2, 2]]]),
        ...     num_classes=3,
        ... )
    """

    # Ground truth masks - shape [N, H, W] where value is class label
    gt_masks: np.ndarray

    # Predicted masks - shape [N, H, W] where value is class label
    pred_masks: np.ndarray

    # Number of classes (required for proper metric computation)
    num_classes: int

    # Optional metadata
    class_names: dict[int, str] | None = None
    ignore_index: int = 255
    image_paths: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data shapes after initialization."""
        # Ensure 3D arrays [N, H, W]
        if self.gt_masks.ndim == 2:
            self.gt_masks = self.gt_masks[np.newaxis, ...]
        if self.pred_masks.ndim == 2:
            self.pred_masks = self.pred_masks[np.newaxis, ...]

        assert self.gt_masks.ndim == 3, (
            f"gt_masks must have shape [N, H, W], got {self.gt_masks.shape}"
        )
        assert self.pred_masks.ndim == 3, (
            f"pred_masks must have shape [N, H, W], got {self.pred_masks.shape}"
        )
        assert self.gt_masks.shape == self.pred_masks.shape, (
            f"gt_masks shape {self.gt_masks.shape} must match "
            f"pred_masks shape {self.pred_masks.shape}"
        )
        assert self.num_classes > 0, "num_classes must be positive"

    @property
    def num_images(self) -> int:
        """Number of images in the dataset."""
        return self.gt_masks.shape[0]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Height and width of the masks."""
        return (self.gt_masks.shape[1], self.gt_masks.shape[2])

    @property
    def total_pixels(self) -> int:
        """Total number of pixels across all images."""
        return int(np.prod(self.gt_masks.shape))
