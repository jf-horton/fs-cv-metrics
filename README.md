# foresight_metrics

Computer Vision Metrics Package - Extensible metrics for CV tasks.

## Installation

```bash
pip install -e .
```

With ClearML support:
```bash
pip install -e ".[clearml]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from foresight_metrics import ObjectDetection, StdoutLogger

od = ObjectDetection(
    data_format="coco",
    metrics=["mAP", "precision_recall"],
    loggers=[StdoutLogger()],
)

result = od.evaluate("ground_truth.json", "predictions.json")
print(result.metrics)
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
```

## Extending

### Add a Custom Format

```python
from foresight_metrics.tasks.object_detection.types import ODData

class YoloFormat:
    name = "yolo"
    
    def load(self, ground_truth, predictions) -> ODData:
        # Parse YOLO format files
        ...
        return ODData(...)

od = ObjectDetection(
    data_format="yolo",
    custom_formats={"yolo": YoloFormat},
)
```

### Add a Custom Metric

```python
from foresight_metrics.tasks.object_detection.types import ODData

class ConfusionMetric:
    name = "confusion"
    
    def compute(self, data: ODData) -> dict[str, float]:
        # Compute your metric
        return {"accuracy": 0.85}

od = ObjectDetection(
    metrics=["mAP", "confusion"],
    custom_metrics={"confusion": ConfusionMetric},
)
```

## Available Metrics

| Metric | Description |
|--------|-------------|
| `mAP` | Mean Average Precision at IoU 0.5-0.95 |
| `precision_recall` | Precision, Recall, F1 at IoU 0.5 |

## License

MIT
