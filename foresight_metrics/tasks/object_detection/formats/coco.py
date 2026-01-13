"""
COCO format adapter for object detection.
"""
import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from foresight_metrics.tasks.object_detection.types import ODData


class CocoFormat:
    """
    Format adapter for COCO JSON annotation format.

    Supports loading ground truth and predictions from COCO-style JSON files.
    Uses the Supervision library internally for robust parsing when available,
    but falls back to direct JSON parsing.

    Expected ground truth format (COCO annotations):
        {
            "images": [{"id": 1, "file_name": "image1.jpg", ...}, ...],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}, ...],
            "categories": [{"id": 1, "name": "person"}, ...]
        }

    Expected predictions format (COCO results):
        [
            {"image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "score": 0.95},
            ...
        ]

    Example:
        >>> adapter = CocoFormat()
        >>> data = adapter.load("annotations.json", "predictions.json")
        >>> print(data.num_gt_boxes, data.num_predictions)
    """

    name = "coco"

    def load(
        self,
        ground_truth: Union[str, Path, dict],
        predictions: Union[str, Path, dict, list],
    ) -> ODData:
        """
        Load COCO format data and convert to ODData.

        Args:
            ground_truth: Path to COCO annotations JSON or dict with annotations.
            predictions: Path to COCO results JSON, list of predictions, or dict.

        Returns:
            ODData with ground truth and predictions in canonical format.
        """
        gt_data = self._load_json(ground_truth)
        pred_data = self._load_json(predictions)

        # Parse ground truth
        gt_image_ids, gt_boxes, gt_labels, class_names, image_paths = (
            self._parse_ground_truth(gt_data)
        )

        # Parse predictions
        pred_image_ids, pred_boxes, pred_labels, pred_scores = self._parse_predictions(
            pred_data
        )

        return ODData(
            gt_image_ids=gt_image_ids,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_image_ids=pred_image_ids,
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            class_names=class_names,
            image_paths=image_paths,
        )

    def _load_json(self, source: Union[str, Path, dict, list]) -> Any:
        """Load JSON from file path or return dict/list directly."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                return json.load(f)
        return source

    def _parse_ground_truth(
        self, data: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], dict[int, str]]:
        """Parse COCO ground truth annotations."""
        if not isinstance(data, dict):
            raise ValueError("Ground truth must be a COCO annotations dict")

        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        images = data.get("images", [])

        # Build category mapping
        class_names = {cat["id"]: cat["name"] for cat in categories}

        # Build image path mapping
        image_paths = {img["id"]: img.get("file_name", "") for img in images}

        # Parse annotations
        image_ids = []
        boxes = []
        labels = []

        for ann in annotations:
            image_ids.append(ann["image_id"])
            # COCO bbox is [x, y, width, height], convert to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        return (
            np.array(image_ids, dtype=np.int64),
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64),
            class_names,
            image_paths,
        )

    def _parse_predictions(
        self, data: Union[dict, list]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse COCO prediction results."""
        # Handle both list format and dict with "annotations" key
        if isinstance(data, dict):
            predictions = data.get("annotations", data.get("results", []))
        else:
            predictions = data

        if not isinstance(predictions, list):
            raise ValueError("Predictions must be a list of detection results")

        image_ids = []
        boxes = []
        labels = []
        scores = []

        for pred in predictions:
            image_ids.append(pred["image_id"])
            # COCO bbox is [x, y, width, height], convert to [x1, y1, x2, y2]
            x, y, w, h = pred["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(pred["category_id"])
            scores.append(pred.get("score", 1.0))

        return (
            np.array(image_ids, dtype=np.int64),
            np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64),
            np.array(scores, dtype=np.float32),
        )
