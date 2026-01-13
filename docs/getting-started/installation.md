# Installation

## Requirements

- Python 3.10 or higher
- NumPy

## Basic Installation

Install from the repository:

```bash
git clone https://github.com/jf-horton/fs-cv-metrics.git
cd fs-cv-metrics
pip install -e .
```

## Optional Dependencies

### ClearML Integration

For logging metrics to ClearML experiment tracker:

```bash
pip install -e ".[clearml]"
```

### Development Tools

For development (testing, linting, type checking):

```bash
pip install -e ".[dev]"
```

### All Dependencies

Install everything:

```bash
pip install -e ".[all]"
```

## Verify Installation

```python
from foresight_metrics import ObjectDetection, Segmentation

print("ObjectDetection available formats:", ObjectDetection().available_formats)
print("ObjectDetection available metrics:", ObjectDetection().available_metrics)
```

Expected output:
```
ObjectDetection available formats: ['coco']
ObjectDetection available metrics: ['mAP', 'precision_recall']
```
