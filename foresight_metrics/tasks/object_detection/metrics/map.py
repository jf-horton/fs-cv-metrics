"""
Mean Average Precision (mAP) metric for object detection.
"""
from typing import Sequence

import numpy as np

from foresight_metrics.tasks.object_detection.types import ODData


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _compute_ap(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
    class_id: int,
) -> float:
    """
    Compute Average Precision for a single class at a single IoU threshold.

    Args:
        gt_boxes: Ground truth boxes [N, 4]
        gt_labels: Ground truth labels [N]
        pred_boxes: Predicted boxes [M, 4]
        pred_labels: Predicted labels [M]
        pred_scores: Prediction scores [M]
        iou_threshold: IoU threshold for matching
        class_id: Class to evaluate

    Returns:
        Average Precision value.
    """
    # Filter by class
    gt_mask = gt_labels == class_id
    pred_mask = pred_labels == class_id

    gt_cls_boxes = gt_boxes[gt_mask]
    pred_cls_boxes = pred_boxes[pred_mask]
    pred_cls_scores = pred_scores[pred_mask]

    if len(gt_cls_boxes) == 0:
        return 0.0 if len(pred_cls_boxes) > 0 else 1.0

    if len(pred_cls_boxes) == 0:
        return 0.0

    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_cls_scores)
    pred_cls_boxes = pred_cls_boxes[sorted_indices]

    # Track matched GT boxes
    gt_matched = np.zeros(len(gt_cls_boxes), dtype=bool)

    # Calculate TP/FP for each prediction
    tp = np.zeros(len(pred_cls_boxes))
    fp = np.zeros(len(pred_cls_boxes))

    for i, pred_box in enumerate(pred_cls_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_cls_boxes):
            if gt_matched[j]:
                continue
            iou = _compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(gt_cls_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 11-point interpolation or all-point interpolation
    # Using all-point interpolation (COCO style)
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    recall_changes = np.where(recalls[1:] != recalls[:-1])[0]

    # Sum up the areas
    ap = np.sum((recalls[recall_changes + 1] - recalls[recall_changes]) * precisions[recall_changes + 1])

    return float(ap)


class MAPMetric:
    """
    Mean Average Precision (mAP) metric.

    Computes COCO-style mAP at multiple IoU thresholds. The default
    thresholds follow the COCO evaluation standard: [0.5, 0.55, ..., 0.95].

    Attributes:
        name: "mAP"
        iou_thresholds: List of IoU thresholds for evaluation.

    Example:
        >>> metric = MAPMetric(iou_thresholds=[0.5, 0.75])
        >>> result = metric.compute(data)
        >>> print(result)
        {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48}
    """

    name = "mAP"

    def __init__(self, iou_thresholds: Sequence[float] | None = None) -> None:
        """
        Initialize the mAP metric.

        Args:
            iou_thresholds: IoU thresholds for mAP calculation.
                           Defaults to COCO standard [0.5, 0.55, ..., 0.95].
        """
        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = list(iou_thresholds)

    def compute(self, data: ODData) -> dict[str, float]:
        """
        Compute mAP across all classes and IoU thresholds.

        Args:
            data: Object detection data in canonical format.

        Returns:
            Dictionary with mAP values:
            - "mAP": Mean AP across all thresholds
            - "mAP@{threshold}": AP at specific thresholds (0.5, 0.75)
        """
        unique_labels = data.unique_labels

        if len(unique_labels) == 0:
            return {"mAP": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0}

        # Compute AP for each class and threshold
        all_aps = []
        aps_by_threshold: dict[float, list[float]] = {t: [] for t in self.iou_thresholds}

        for class_id in unique_labels:
            for threshold in self.iou_thresholds:
                ap = _compute_ap(
                    data.gt_boxes,
                    data.gt_labels,
                    data.pred_boxes,
                    data.pred_labels,
                    data.pred_scores,
                    threshold,
                    class_id,
                )
                all_aps.append(ap)
                aps_by_threshold[threshold].append(ap)

        # Compute mean values
        result = {
            "mAP": float(np.mean(all_aps)) if all_aps else 0.0,
        }

        # Add specific thresholds that are commonly reported
        if 0.5 in aps_by_threshold:
            result["mAP@0.5"] = float(np.mean(aps_by_threshold[0.5]))
        if 0.75 in aps_by_threshold:
            result["mAP@0.75"] = float(np.mean(aps_by_threshold[0.75]))

        return result
