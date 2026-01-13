# ObjectDetection

::: foresight_metrics.tasks.object_detection.ObjectDetection
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - evaluate
        - evaluate_data
        - data_format
        - metric_names
        - available_formats
        - available_metrics

---

## ODData

::: foresight_metrics.tasks.object_detection.types.ODData
    options:
      show_root_heading: true
      show_source: false

---

## Format Adapters

### CocoFormat

::: foresight_metrics.tasks.object_detection.formats.coco.CocoFormat
    options:
      show_root_heading: true
      show_source: false

---

## Metrics

### MAPMetric

::: foresight_metrics.tasks.object_detection.metrics.map.MAPMetric
    options:
      show_root_heading: true
      show_source: false

### PrecisionRecallMetric

::: foresight_metrics.tasks.object_detection.metrics.precision_recall.PrecisionRecallMetric
    options:
      show_root_heading: true
      show_source: false
