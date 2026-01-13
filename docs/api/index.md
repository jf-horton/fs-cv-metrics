# API Reference

This section provides detailed API documentation for all public classes and functions.

## Tasks

- [**ObjectDetection**](object-detection.md) - Object detection metrics (mAP, precision, recall)
- [**Segmentation**](segmentation.md) - Segmentation metrics (mIoU, Dice, pixel accuracy)

## Loggers

- [**StdoutLogger**](loggers.md#stdoutlogger) - Print metrics to terminal
- [**ClearMLLogger**](loggers.md#clearmllogger) - Log metrics to ClearML

## Core

- [**MetricResult**](results.md) - Metric computation result container

## Package Structure

```
foresight_metrics/
├── __init__.py              # Public exports
├── results.py               # MetricResult
├── loggers/
│   ├── stdout.py            # StdoutLogger
│   └── clearml.py           # ClearMLLogger
└── tasks/
    ├── object_detection/
    │   ├── __init__.py      # ObjectDetection
    │   ├── types.py         # ODData
    │   ├── formats/         # CocoFormat, etc.
    │   └── metrics/         # MAPMetric, PrecisionRecallMetric
    └── segmentation/
        ├── __init__.py      # Segmentation
        ├── types.py         # SegData
        ├── formats/         # NumpyFormat
        └── metrics/         # IoUMetric, DiceMetric, PixelAccuracyMetric
```
