"""
Dice coefficient (F1 score) for segmentation.
"""
import numpy as np

from foresight_metrics.tasks.segmentation.types import SegData


class DiceMetric:
    """
    Dice coefficient (F1 score) for semantic segmentation.

    Computes per-class Dice score and mean Dice across all classes.
    Dice is defined as: 2 * intersection / (|A| + |B|) = 2*TP / (2*TP + FP + FN)

    This is equivalent to the F1 score for binary classification.

    Attributes:
        name: "dice"

    Example:
        >>> metric = DiceMetric()
        >>> result = metric.compute(data)
        >>> print(result)
        {'mean_dice': 0.82, 'dice_class_0': 0.88, 'dice_class_1': 0.76}
    """

    name = "dice"

    def compute(self, data: SegData) -> dict[str, float]:
        """
        Compute Dice coefficient metrics.

        Args:
            data: Segmentation data in canonical format.

        Returns:
            Dictionary with:
            - "mean_dice": Mean Dice across all classes
            - "dice_class_{i}": Per-class Dice (if class_names not provided)
            - "dice_{name}": Per-class Dice (if class_names provided)
        """
        gt = data.gt_masks.flatten()
        pred = data.pred_masks.flatten()
        
        # Create valid mask (exclude ignore_index)
        valid_mask = gt != data.ignore_index
        gt = gt[valid_mask]
        pred = pred[valid_mask]

        per_class_dice = {}
        dices = []

        for class_id in range(data.num_classes):
            gt_mask = gt == class_id
            pred_mask = pred == class_id

            intersection = np.logical_and(gt_mask, pred_mask).sum()
            total = gt_mask.sum() + pred_mask.sum()

            if total == 0:
                # Class not present in GT or predictions
                dice = float('nan')
            else:
                dice = float(2 * intersection / total)
                dices.append(dice)

            # Use class name if available
            if data.class_names and class_id in data.class_names:
                key = f"dice_{data.class_names[class_id]}"
            else:
                key = f"dice_class_{class_id}"
            
            per_class_dice[key] = dice

        # Compute mean Dice (excluding NaN values)
        valid_dices = [d for d in dices if not np.isnan(d)]
        mean_dice = float(np.mean(valid_dices)) if valid_dices else 0.0

        return {"mean_dice": mean_dice, **per_class_dice}


class PixelAccuracyMetric:
    """
    Pixel accuracy for semantic segmentation.

    Computes overall pixel accuracy and per-class accuracy.

    Attributes:
        name: "pixel_accuracy"

    Example:
        >>> metric = PixelAccuracyMetric()
        >>> result = metric.compute(data)
        >>> print(result)
        {'pixel_accuracy': 0.92, 'mean_class_accuracy': 0.85}
    """

    name = "pixel_accuracy"

    def compute(self, data: SegData) -> dict[str, float]:
        """
        Compute pixel accuracy metrics.

        Args:
            data: Segmentation data in canonical format.

        Returns:
            Dictionary with:
            - "pixel_accuracy": Overall pixel accuracy
            - "mean_class_accuracy": Mean per-class accuracy
        """
        gt = data.gt_masks.flatten()
        pred = data.pred_masks.flatten()
        
        # Create valid mask (exclude ignore_index)
        valid_mask = gt != data.ignore_index
        gt = gt[valid_mask]
        pred = pred[valid_mask]

        if len(gt) == 0:
            return {"pixel_accuracy": 0.0, "mean_class_accuracy": 0.0}

        # Overall pixel accuracy
        correct = (gt == pred).sum()
        total = len(gt)
        pixel_acc = float(correct / total)

        # Per-class accuracy
        class_accs = []
        for class_id in range(data.num_classes):
            gt_mask = gt == class_id
            if gt_mask.sum() == 0:
                continue
            correct_class = np.logical_and(gt_mask, pred == class_id).sum()
            class_accs.append(float(correct_class / gt_mask.sum()))

        mean_class_acc = float(np.mean(class_accs)) if class_accs else 0.0

        return {
            "pixel_accuracy": pixel_acc,
            "mean_class_accuracy": mean_class_acc,
        }
