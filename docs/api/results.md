# MetricResult

::: foresight_metrics.results.MetricResult
    options:
      show_root_heading: true
      show_source: false

## Example Usage

```python
from foresight_metrics import ObjectDetection

od = ObjectDetection()
result = od.evaluate("gt.json", "preds.json")

# Access computed metrics
print(result.metrics)
# {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48, 'precision': 0.85, ...}

# Task name
print(result.task_name)
# 'object_detection'

# Metadata about the evaluation
print(result.metadata)
# {'format': 'coco', 'num_gt_boxes': 100, 'num_predictions': 95, ...}

# Pretty print for terminal
print(result.pretty())
# === object_detection ===
#   mAP: 0.6500
#   mAP@0.5: 0.8200
#   ...

# Export to dictionary
data = result.to_dict()
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_name` | `str` | Name of the evaluation task |
| `metrics` | `dict[str, float]` | Computed metric values |
| `per_class` | `dict[str, dict[str, float]]` | Per-class metric breakdown |
| `metadata` | `dict` | Additional evaluation information |

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `pretty()` | `str` | Formatted string for terminal display |
| `to_dict()` | `dict` | Dictionary for JSON serialization |
