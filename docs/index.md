# Foresight Metrics

**Computer Vision Metrics Package** - Extensible metrics for CV tasks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Foresight Metrics is a Python utility package for computing computer vision metrics with a focus on **maintainability**, **readability**, and **extensibility**.

### Supported Tasks

| Task | Metrics |
|------|---------|
| **Object Detection** | mAP, mAP@0.5, mAP@0.75, Precision, Recall, F1 |
| **Segmentation** | mIoU, Dice, Pixel Accuracy |

### Key Features

- 🎯 **Simple API** - One class per task, clear method signatures
- 🔌 **Extensible** - Add custom formats and metrics without touching core code
- 📊 **Multiple Loggers** - Console, ClearML, and custom integrations
- 📝 **Self-Documenting** - Full type hints and docstrings

## Quick Example

```python
from foresight_metrics import ObjectDetection, StdoutLogger

od = ObjectDetection(
    data_format="coco",
    metrics=["mAP", "precision_recall"],
    loggers=[StdoutLogger()],
)

result = od.evaluate("ground_truth.json", "predictions.json")
print(result.metrics)
# {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48, 'precision': 0.85, ...}
```

## Installation

```bash
pip install -e .
```

With ClearML support:
```bash
pip install -e ".[clearml]"
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [API Reference](api/index.md)
