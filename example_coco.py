"""
Example: Using foresight_metrics with COCO JSON files (indirect/file-based method).

This shows how to use foresight_metrics when you have:
- Ground truth annotations in COCO JSON format
- Predictions saved in COCO results JSON format
"""
from foresight_metrics import ObjectDetection, ClearMLLogger, StdoutLogger


def evaluate_from_files(gt_path: str, pred_path: str, task=None):
    """
    Evaluate predictions from COCO JSON files.
    
    Args:
        gt_path: Path to COCO annotations JSON (ground truth)
        pred_path: Path to COCO results JSON (predictions)
        task: Optional ClearML task for logging
    
    Returns:
        MetricResult with computed metrics
    """
    # Configure loggers
    loggers = [StdoutLogger()]
    if task is not None:
        loggers.append(ClearMLLogger(task=task))

    # Create evaluator
    od = ObjectDetection(
        data_format="coco",
        metrics=["mAP", "precision_recall"],
        loggers=loggers,
    )
    
    # Evaluate from files
    result = od.evaluate(gt_path, pred_path)
    
    return result


# -----------------------------------------------------------------------------
# Example COCO JSON formats (for reference)
# -----------------------------------------------------------------------------

# Ground truth format (COCO annotations):
EXAMPLE_GT = {
    "images": [
        {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
        {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},  # [x, y, w, h]
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 80, 60]},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 40, 40]},
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
    ]
}

# Predictions format (COCO results):
EXAMPLE_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [102, 98, 48, 52], "score": 0.95},
    {"image_id": 1, "category_id": 2, "bbox": [198, 202, 82, 58], "score": 0.87},
    {"image_id": 2, "category_id": 1, "bbox": [148, 152, 42, 38], "score": 0.92},
]


# -----------------------------------------------------------------------------
# Usage with in-memory dicts (also works!)
# -----------------------------------------------------------------------------

def evaluate_from_dicts():
    """You can also pass dicts directly instead of file paths."""
    od = ObjectDetection(
        data_format="coco",
        metrics=["mAP", "precision_recall"],
        loggers=[StdoutLogger()],
    )
    
    # Pass dicts directly
    result = od.evaluate(EXAMPLE_GT, EXAMPLE_PREDS)
    
    return result


if __name__ == "__main__":
    print("Evaluating from in-memory dicts:")
    result = evaluate_from_dicts()
    print(f"\nMetrics: {result.metrics}")
    print(f"Metadata: {result.metadata}")
