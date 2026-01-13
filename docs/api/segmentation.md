# Segmentation

::: foresight_metrics.tasks.segmentation.Segmentation
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - evaluate
        - data_format
        - num_classes
        - metric_names
        - available_formats
        - available_metrics

---

## SegData

::: foresight_metrics.tasks.segmentation.types.SegData
    options:
      show_root_heading: true
      show_source: false

---

## Format Adapters

### NumpyFormat

::: foresight_metrics.tasks.segmentation.formats.numpy_format.NumpyFormat
    options:
      show_root_heading: true
      show_source: false

---

## Metrics

### IoUMetric

::: foresight_metrics.tasks.segmentation.metrics.iou.IoUMetric
    options:
      show_root_heading: true
      show_source: false

### DiceMetric

::: foresight_metrics.tasks.segmentation.metrics.dice.DiceMetric
    options:
      show_root_heading: true
      show_source: false

### PixelAccuracyMetric

::: foresight_metrics.tasks.segmentation.metrics.dice.PixelAccuracyMetric
    options:
      show_root_heading: true
      show_source: false
