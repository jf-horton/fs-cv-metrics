"""
Internal data types for object detection.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ODData:
    """
    Internal representation of object detection data for metric computation.

    All metrics operate on this format. Format adapters convert external
    formats (COCO, YOLO, etc.) into this structure.

    Supports multi-image evaluation by including image_id arrays.

    Attributes:
        gt_image_ids: Image ID for each ground truth box. Shape [N].
        gt_boxes: Ground truth bounding boxes in xyxy format. Shape [N, 4].
        gt_labels: Class labels for each ground truth box. Shape [N].
        pred_image_ids: Image ID for each prediction. Shape [M].
        pred_boxes: Predicted bounding boxes in xyxy format. Shape [M, 4].
        pred_labels: Predicted class labels. Shape [M].
        pred_scores: Confidence scores for predictions. Shape [M].
        class_names: Optional mapping from label ID to class name.
        image_paths: Optional mapping from image ID to file path.

    Example:
        >>> data = ODData(
        ...     gt_image_ids=np.array([0, 0, 1]),
        ...     gt_boxes=np.array([[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 80, 80]]),
        ...     gt_labels=np.array([0, 1, 0]),
        ...     pred_image_ids=np.array([0, 0, 1]),
        ...     pred_boxes=np.array([[12, 12, 48, 48], [58, 58, 102, 102], [22, 22, 78, 78]]),
        ...     pred_labels=np.array([0, 1, 0]),
        ...     pred_scores=np.array([0.9, 0.85, 0.75]),
        ... )
    """

    # Ground truth
    gt_image_ids: np.ndarray
    gt_boxes: np.ndarray
    gt_labels: np.ndarray

    # Predictions
    pred_image_ids: np.ndarray
    pred_boxes: np.ndarray
    pred_labels: np.ndarray
    pred_scores: np.ndarray

    # Optional metadata
    class_names: dict[int, str] | None = None
    image_paths: dict[int, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data shapes after initialization."""
        # Ground truth validation
        if len(self.gt_boxes) > 0:
            assert self.gt_boxes.ndim == 2 and self.gt_boxes.shape[1] == 4, (
                f"gt_boxes must have shape [N, 4], got {self.gt_boxes.shape}"
            )
        assert len(self.gt_image_ids) == len(self.gt_boxes), (
            f"gt_image_ids length ({len(self.gt_image_ids)}) must match "
            f"gt_boxes length ({len(self.gt_boxes)})"
        )
        assert len(self.gt_labels) == len(self.gt_boxes), (
            f"gt_labels length ({len(self.gt_labels)}) must match "
            f"gt_boxes length ({len(self.gt_boxes)})"
        )

        # Prediction validation
        if len(self.pred_boxes) > 0:
            assert self.pred_boxes.ndim == 2 and self.pred_boxes.shape[1] == 4, (
                f"pred_boxes must have shape [M, 4], got {self.pred_boxes.shape}"
            )
        assert len(self.pred_image_ids) == len(self.pred_boxes), (
            f"pred_image_ids length ({len(self.pred_image_ids)}) must match "
            f"pred_boxes length ({len(self.pred_boxes)})"
        )
        assert len(self.pred_labels) == len(self.pred_boxes), (
            f"pred_labels length ({len(self.pred_labels)}) must match "
            f"pred_boxes length ({len(self.pred_boxes)})"
        )
        assert len(self.pred_scores) == len(self.pred_boxes), (
            f"pred_scores length ({len(self.pred_scores)}) must match "
            f"pred_boxes length ({len(self.pred_boxes)})"
        )

    @property
    def num_gt_boxes(self) -> int:
        """Total number of ground truth boxes."""
        return len(self.gt_boxes)

    @property
    def num_predictions(self) -> int:
        """Total number of predictions."""
        return len(self.pred_boxes)

    @property
    def num_images(self) -> int:
        """Number of unique images in the dataset."""
        all_ids = np.concatenate([self.gt_image_ids, self.pred_image_ids])
        return len(np.unique(all_ids)) if len(all_ids) > 0 else 0

    @property
    def unique_labels(self) -> np.ndarray:
        """All unique class labels across GT and predictions."""
        all_labels = np.concatenate([self.gt_labels, self.pred_labels])
        return np.unique(all_labels) if len(all_labels) > 0 else np.array([])
