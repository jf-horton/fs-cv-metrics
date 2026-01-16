"""
Example: Using foresight_metrics with a YOLO training script.

Shows the simple update/compute pattern for evaluating during inference.
"""
import cv2

# Your existing imports
# from clearml import Task
# from ultralytics import YOLO

from foresight_metrics import ObjectDetection, ClearMLLogger, StdoutLogger


def evaluate_model(model, test_dataset, config, task=None):
    """
    Evaluate a model on a test dataset using foresight_metrics.
    
    Args:
        model: YOLO model instance
        test_dataset: Supervision DetectionDataset
        config: Dict with 'imgsz' key
        task: Optional ClearML task for logging
    """
    # Configure loggers
    loggers = [StdoutLogger()]
    if task is not None:
        loggers.append(ClearMLLogger(task=task))

    # Create evaluator
    od = ObjectDetection(
        metrics=["mAP", "precision_recall"],
        loggers=loggers,
    )

    # Inference loop - just call update() for each image
    for idx, (image_path, _image, annotation) in enumerate(test_dataset):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        pred_results = model.predict(
            source=img_rgb,
            imgsz=config["imgsz"],
            conf=0.001,
            verbose=False
        )

        # Get predictions
        res = pred_results[0] if len(pred_results) > 0 else None
        boxes = res.boxes if res is not None else None

        # Update metrics with this image's detections
        od.update(
            gt_boxes=annotation.xyxy,
            gt_labels=annotation.class_id,
            pred_boxes=boxes.xyxy.cpu().numpy() if boxes is not None else [],
            pred_labels=boxes.cls.cpu().numpy().astype(int) if boxes is not None else [],
            pred_scores=boxes.conf.cpu().numpy() if boxes is not None else [],
        )

    # Compute final metrics (auto-resets)
    result = od.compute()
    
    return result


# Example usage:
# 
# result = evaluate_model(model, test_dataset, config, task)
# print(result.metrics)
# # {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48, 'precision': 0.85, ...}