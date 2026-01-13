# Object Detection

The `ObjectDetection` class provides metrics for evaluating object detection models.

## Supported Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| **mAP** | `mAP` | Mean Average Precision across IoU thresholds 0.5-0.95 |
| **mAP@0.5** | `mAP@0.5` | AP at IoU threshold 0.5 |
| **mAP@0.75** | `mAP@0.75` | AP at IoU threshold 0.75 |
| **Precision** | `precision` | True positives / (True positives + False positives) |
| **Recall** | `recall` | True positives / (True positives + False negatives) |
| **F1** | `f1` | Harmonic mean of precision and recall |

## Supported Formats

| Format | Key | Description |
|--------|-----|-------------|
| **COCO** | `coco` | Standard COCO JSON annotation format |

## Usage

### Basic Usage

```python
from foresight_metrics import ObjectDetection, StdoutLogger

od = ObjectDetection(
    data_format="coco",
    metrics=["mAP", "precision_recall"],
    loggers=[StdoutLogger()],
)

result = od.evaluate("ground_truth.json", "predictions.json")
```

### Direct Data Evaluation

Use `evaluate_data()` when you have detection data in memory:

```python
from foresight_metrics.tasks.object_detection.types import ODData
import numpy as np

data = ODData(
    gt_image_ids=np.array([0, 0, 1]),
    gt_boxes=np.array([[10, 10, 50, 50], [60, 60, 100, 100], [20, 20, 80, 80]]),
    gt_labels=np.array([0, 1, 0]),
    pred_image_ids=np.array([0, 0, 1]),
    pred_boxes=np.array([[12, 12, 48, 48], [58, 58, 102, 102], [22, 22, 78, 78]]),
    pred_labels=np.array([0, 1, 0]),
    pred_scores=np.array([0.9, 0.85, 0.75]),
)

od = ObjectDetection()
result = od.evaluate_data(data)
```

## ODData Format

The internal `ODData` format represents detection data:

| Field | Type | Description |
|-------|------|-------------|
| `gt_image_ids` | `np.ndarray[N]` | Image ID for each ground truth box |
| `gt_boxes` | `np.ndarray[N, 4]` | GT boxes in `[x1, y1, x2, y2]` format |
| `gt_labels` | `np.ndarray[N]` | Class labels for GT (integers) |
| `pred_image_ids` | `np.ndarray[M]` | Image ID for each prediction |
| `pred_boxes` | `np.ndarray[M, 4]` | Predicted boxes in `[x1, y1, x2, y2]` |
| `pred_labels` | `np.ndarray[M]` | Predicted class labels |
| `pred_scores` | `np.ndarray[M]` | Confidence scores |
| `class_names` | `dict[int, str]` | Optional label → name mapping |
