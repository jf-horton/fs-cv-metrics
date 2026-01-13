# Custom Extensions

Foresight Metrics is designed to be easily extensible. You can add custom formats, metrics, and loggers without modifying the core package.

## Adding a Custom Format

Create a class that implements the format protocol:

```python
from foresight_metrics.tasks.object_detection.types import ODData
import numpy as np

class YoloFormat:
    """Custom format adapter for YOLO text files."""
    name = "yolo"
    
    def load(self, ground_truth, predictions) -> ODData:
        # Parse your format and return ODData
        # ground_truth and predictions can be file paths or data structures
        
        return ODData(
            gt_image_ids=np.array([...]),
            gt_boxes=np.array([...]),
            gt_labels=np.array([...]),
            pred_image_ids=np.array([...]),
            pred_boxes=np.array([...]),
            pred_labels=np.array([...]),
            pred_scores=np.array([...]),
        )
```

Use your custom format:

```python
from foresight_metrics import ObjectDetection

od = ObjectDetection(
    data_format="yolo",
    custom_formats={"yolo": YoloFormat},
)

result = od.evaluate("labels/", "predictions/")
```

## Adding a Custom Metric

Create a class that implements the metric protocol:

```python
from foresight_metrics.tasks.object_detection.types import ODData

class ConfusionMatrixMetric:
    """Custom metric that computes confusion matrix statistics."""
    name = "confusion"
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def compute(self, data: ODData) -> dict[str, float]:
        # Access data fields:
        # - data.gt_boxes, data.gt_labels, data.gt_image_ids
        # - data.pred_boxes, data.pred_labels, data.pred_scores, data.pred_image_ids
        
        # Compute your metrics
        tp = ...
        fp = ...
        fn = ...
        
        return {
            "true_positives": float(tp),
            "false_positives": float(fp),
            "false_negatives": float(fn),
        }
```

Use your custom metric:

```python
from foresight_metrics import ObjectDetection

od = ObjectDetection(
    metrics=["mAP", "confusion"],
    custom_metrics={"confusion": ConfusionMatrixMetric},
)

result = od.evaluate("gt.json", "preds.json")
print(result.metrics["true_positives"])
```

## Adding a Custom Logger

Create a class that implements the logger protocol:

```python
from foresight_metrics.results import MetricResult
import json

class JSONFileLogger:
    """Logger that saves metrics to a JSON file."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
    
    def log(self, result: MetricResult) -> None:
        with open(self.output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
```

Use your custom logger:

```python
from foresight_metrics import ObjectDetection

od = ObjectDetection(
    loggers=[JSONFileLogger("metrics.json")],
)

result = od.evaluate("gt.json", "preds.json")
# Metrics saved to metrics.json
```

## Adding a New Task

To add a completely new task type (e.g., keypoint detection), create a new directory under `tasks/`:

```
foresight_metrics/tasks/keypoints/
├── __init__.py      # KeypointDetection class
├── types.py         # KPData internal format
├── formats/
│   ├── __init__.py
│   ├── base.py      # KPFormatAdapter protocol
│   └── coco.py      # COCO keypoint format
└── metrics/
    ├── __init__.py
    ├── base.py      # KPMetric protocol
    └── oks.py       # Object Keypoint Similarity
```

Follow the same patterns used in `object_detection/` and `segmentation/`.

## Protocol Reference

### Format Adapter Protocol

```python
class FormatAdapter:
    name: str
    
    def load(self, ground_truth, predictions) -> InternalData:
        ...
```

### Metric Protocol

```python
class Metric:
    name: str
    
    def compute(self, data: InternalData) -> dict[str, float]:
        ...
```

### Logger Protocol

```python
class Logger:
    def log(self, result: MetricResult) -> None:
        ...
```
