"""
Intersection over Union (IoU) metrics for segmentation.
"""
import numpy as np

from foresight_metrics.tasks.segmentation.types import SegData


class IoUMetric:
    """
    Intersection over Union (IoU) metric for semantic segmentation.

    Computes per-class IoU and mean IoU (mIoU) across all classes.
    IoU is defined as: intersection / union = TP / (TP + FP + FN)

    Attributes:
        name: "iou"

    Example:
        >>> metric = IoUMetric()
        >>> result = metric.compute(data)
        >>> print(result)
        {'mIoU': 0.75, 'IoU_class_0': 0.80, 'IoU_class_1': 0.70}
    """

    name = "iou"

    def compute(self, data: SegData) -> dict[str, float]:
        """
        Compute IoU metrics.

        Args:
            data: Segmentation data in canonical format.

        Returns:
            Dictionary with:
            - "mIoU": Mean IoU across all classes
            - "IoU_class_{i}": Per-class IoU (if class_names not provided)
            - "IoU_{name}": Per-class IoU (if class_names provided)
        """
        gt = data.gt_masks.flatten()
        pred = data.pred_masks.flatten()
        
        # Create valid mask (exclude ignore_index)
        valid_mask = gt != data.ignore_index
        gt = gt[valid_mask]
        pred = pred[valid_mask]

        per_class_iou = {}
        ious = []

        for class_id in range(data.num_classes):
            gt_mask = gt == class_id
            pred_mask = pred == class_id

            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()

            if union == 0:
                # Class not present in GT or predictions
                iou = float('nan')
            else:
                iou = float(intersection / union)
                ious.append(iou)

            # Use class name if available
            if data.class_names and class_id in data.class_names:
                key = f"IoU_{data.class_names[class_id]}"
            else:
                key = f"IoU_class_{class_id}"
            
            per_class_iou[key] = iou

        # Compute mIoU (excluding NaN values)
        valid_ious = [i for i in ious if not np.isnan(i)]
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0

        return {"mIoU": miou, **per_class_iou}
