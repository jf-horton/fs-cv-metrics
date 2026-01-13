"""
Precision, Recall, and F1 metrics for object detection.
"""
import numpy as np

from foresight_metrics.tasks.object_detection.types import ODData


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class PrecisionRecallMetric:
    """
    Precision, Recall, and F1 score at a fixed IoU threshold.

    Computes detection metrics by matching predictions to ground truth
    boxes using IoU and class labels.

    Attributes:
        name: "precision_recall"
        iou_threshold: IoU threshold for counting a detection as correct.
        score_threshold: Minimum confidence score for predictions.

    Example:
        >>> metric = PrecisionRecallMetric(iou_threshold=0.5)
        >>> result = metric.compute(data)
        >>> print(result)
        {'precision': 0.85, 'recall': 0.72, 'f1': 0.78}
    """

    name = "precision_recall"

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
    ) -> None:
        """
        Initialize the precision/recall metric.

        Args:
            iou_threshold: IoU threshold for matching predictions to GT.
            score_threshold: Minimum confidence score for predictions.
        """
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def compute(self, data: ODData) -> dict[str, float]:
        """
        Compute precision, recall, and F1 score.

        A prediction is considered a true positive if:
        1. Its IoU with a GT box >= iou_threshold
        2. Its class matches the GT class
        3. The GT box has not already been matched

        Args:
            data: Object detection data in canonical format.

        Returns:
            Dictionary with:
            - "precision": TP / (TP + FP)
            - "recall": TP / (TP + FN)
            - "f1": Harmonic mean of precision and recall
        """
        # Filter predictions by score threshold
        score_mask = data.pred_scores >= self.score_threshold
        pred_boxes = data.pred_boxes[score_mask]
        pred_labels = data.pred_labels[score_mask]
        pred_scores = data.pred_scores[score_mask]

        gt_boxes = data.gt_boxes
        gt_labels = data.gt_labels

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        if len(gt_boxes) == 0:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

        if len(pred_boxes) == 0:
            return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]

        # Track matched GT boxes
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        tp = 0
        fp = 0

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0.0
            best_gt_idx = -1

            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_matched[j] or gt_label != pred_label:
                    continue

                iou = _compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = len(gt_boxes) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
