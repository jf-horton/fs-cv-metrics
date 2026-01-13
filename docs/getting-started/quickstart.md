# Quick Start

This guide shows you how to evaluate your first model with Foresight Metrics.

## Object Detection

### From COCO JSON Files

```python
from foresight_metrics import ObjectDetection, StdoutLogger

od = ObjectDetection(
    data_format="coco",
    metrics=["mAP", "precision_recall"],
    loggers=[StdoutLogger()],
)

result = od.evaluate("ground_truth.json", "predictions.json")
```

### From In-Memory Data (Inference Loop)

When running inference in a loop, accumulate detections and evaluate at the end:

```python
import numpy as np
from foresight_metrics import ObjectDetection, StdoutLogger
from foresight_metrics.tasks.object_detection.types import ODData

# Accumulate during inference
all_gt_image_ids = []
all_gt_boxes = []
all_gt_labels = []
all_pred_image_ids = []
all_pred_boxes = []
all_pred_labels = []
all_pred_scores = []

for idx, (image, annotation) in enumerate(dataset):
    preds = model.predict(image)
    
    # Accumulate ground truth
    all_gt_image_ids.extend([idx] * len(annotation.boxes))
    all_gt_boxes.append(annotation.boxes)
    all_gt_labels.extend(annotation.labels)
    
    # Accumulate predictions
    all_pred_image_ids.extend([idx] * len(preds.boxes))
    all_pred_boxes.append(preds.boxes)
    all_pred_labels.extend(preds.labels)
    all_pred_scores.extend(preds.scores)

# Build ODData
data = ODData(
    gt_image_ids=np.array(all_gt_image_ids),
    gt_boxes=np.vstack(all_gt_boxes),
    gt_labels=np.array(all_gt_labels),
    pred_image_ids=np.array(all_pred_image_ids),
    pred_boxes=np.vstack(all_pred_boxes),
    pred_labels=np.array(all_pred_labels),
    pred_scores=np.array(all_pred_scores),
)

# Evaluate
od = ObjectDetection(loggers=[StdoutLogger()])
result = od.evaluate_data(data)
```

## Segmentation

```python
import numpy as np
from foresight_metrics import Segmentation, StdoutLogger

# Create sample masks (3 classes, 4x4 images)
gt_masks = np.array([
    [[0, 0, 1, 1],
     [0, 0, 1, 1],
     [2, 2, 2, 2],
     [2, 2, 2, 2]],
])

pred_masks = np.array([
    [[0, 0, 1, 1],
     [0, 1, 1, 1],  # one error
     [2, 2, 2, 2],
     [2, 2, 2, 2]],
])

seg = Segmentation(
    num_classes=3,
    data_format="numpy",
    metrics=["iou", "dice", "pixel_accuracy"],
    loggers=[StdoutLogger()],
)

result = seg.evaluate(gt_masks, pred_masks)
```

## With ClearML

```python
from clearml import Task
from foresight_metrics import ObjectDetection, ClearMLLogger, StdoutLogger

task = Task.init(project_name="my_project", task_name="evaluation")

od = ObjectDetection(
    data_format="coco",
    loggers=[
        StdoutLogger(),
        ClearMLLogger(task=task),
    ],
)

result = od.evaluate("gt.json", "preds.json")
# Metrics are logged to ClearML dashboard automatically
```

## Working with Results

The `evaluate()` method returns a `MetricResult` object:

```python
result = od.evaluate("gt.json", "preds.json")

# Access metrics
print(result.metrics)        # {'mAP': 0.65, 'precision': 0.85, ...}
print(result.task_name)      # 'object_detection'
print(result.metadata)       # {'format': 'coco', 'num_gt_boxes': 100, ...}

# Pretty print
print(result.pretty())

# Export to dict
data = result.to_dict()
```
