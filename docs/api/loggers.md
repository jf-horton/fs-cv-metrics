# Loggers

Loggers handle outputting metrics to various destinations.

## StdoutLogger

::: foresight_metrics.loggers.stdout.StdoutLogger
    options:
      show_root_heading: true
      show_source: false

### Example

```python
from foresight_metrics import ObjectDetection, StdoutLogger

od = ObjectDetection(loggers=[StdoutLogger()])
result = od.evaluate("gt.json", "preds.json")

# Output:
# === object_detection ===
#   mAP: 0.6500
#   mAP@0.5: 0.8200
#   mAP@0.75: 0.4800
#   precision: 0.8500
#   recall: 0.7200
#   f1: 0.7800
```

---

## ClearMLLogger

::: foresight_metrics.loggers.clearml.ClearMLLogger
    options:
      show_root_heading: true
      show_source: false

### Example

```python
from clearml import Task
from foresight_metrics import ObjectDetection, ClearMLLogger

task = Task.init(project_name="my_project", task_name="evaluation")

od = ObjectDetection(loggers=[ClearMLLogger(task=task)])
result = od.evaluate("gt.json", "preds.json")
# Metrics appear in ClearML dashboard
```

---

## Custom Loggers

Implement the logger protocol to create custom loggers:

```python
from foresight_metrics.results import MetricResult

class MyLogger:
    def log(self, result: MetricResult) -> None:
        # Send metrics somewhere
        pass
```

See [Custom Extensions](../user-guide/extensions.md) for more examples.
