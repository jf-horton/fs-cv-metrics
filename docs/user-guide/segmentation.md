# Segmentation

The `Segmentation` class provides metrics for evaluating semantic segmentation models.

## Supported Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| **mIoU** | `mIoU` | Mean Intersection over Union across all classes |
| **IoU per class** | `IoU_class_{i}` | Per-class IoU values |
| **Mean Dice** | `mean_dice` | Mean Dice coefficient (F1 score) |
| **Dice per class** | `dice_class_{i}` | Per-class Dice values |
| **Pixel Accuracy** | `pixel_accuracy` | Overall pixel classification accuracy |
| **Mean Class Accuracy** | `mean_class_accuracy` | Average per-class accuracy |

## Supported Formats

| Format | Key | Description |
|--------|-----|-------------|
| **NumPy** | `numpy` | Direct numpy array input |

## Usage

### Basic Usage

```python
import numpy as np
from foresight_metrics import Segmentation, StdoutLogger

gt_masks = np.array([...])    # Shape: [N, H, W] or [H, W]
pred_masks = np.array([...])  # Shape: [N, H, W] or [H, W]

seg = Segmentation(
    num_classes=3,
    data_format="numpy",
    metrics=["iou", "dice", "pixel_accuracy"],
    loggers=[StdoutLogger()],
)

result = seg.evaluate(gt_masks, pred_masks)
```

### Complete Example

```python
import numpy as np
from foresight_metrics import Segmentation, StdoutLogger

# Ground truth: 2 images, 4x4 pixels, 3 classes
gt_masks = np.array([
    [[0, 0, 1, 1],
     [0, 0, 1, 1],
     [2, 2, 2, 2],
     [2, 2, 2, 2]],
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [2, 2, 2, 2],
     [2, 2, 2, 2]],
])

# Predictions (with some errors)
pred_masks = np.array([
    [[0, 0, 1, 1],
     [0, 1, 1, 1],  # one error
     [2, 2, 2, 2],
     [2, 2, 2, 2]],
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [2, 2, 1, 2],  # one error
     [2, 2, 2, 2]],
])

seg = Segmentation(
    num_classes=3,
    metrics=["iou", "dice"],
    loggers=[StdoutLogger()],
)

result = seg.evaluate(gt_masks, pred_masks)
# Output:
# mIoU: 0.8708
# mean_dice: 0.9300
```

## SegData Format

The internal `SegData` format represents segmentation data:

| Field | Type | Description |
|-------|------|-------------|
| `gt_masks` | `np.ndarray[N, H, W]` | Ground truth masks (pixel values = class IDs) |
| `pred_masks` | `np.ndarray[N, H, W]` | Predicted masks |
| `num_classes` | `int` | Number of classes in the dataset |
| `ignore_index` | `int` | Class index to ignore (default: 255) |
| `class_names` | `dict[int, str]` | Optional label → name mapping |

## Ignoring Classes

Use `ignore_index` to exclude certain pixels (e.g., unlabeled regions):

```python
seg = Segmentation(
    num_classes=3,
    ignore_index=255,  # Pixels with value 255 are ignored
)
```
