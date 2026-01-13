"""
Example: Using foresight_metrics with a YOLO training script.

This shows how to integrate foresight_metrics when running inference in a loop,
accumulating detections, and evaluating at the end.
"""
import cv2
import numpy as np

# Your existing imports
# from clearml import Task
# from ultralytics import YOLO
# import supervision as sv

from foresight_metrics import ObjectDetection, ClearMLLogger, StdoutLogger
from foresight_metrics.tasks.object_detection.types import ODData


def evaluate_model(model, test_dataset, config, task=None):
    """
    Evaluate a model on a test dataset using foresight_metrics.
    
    Args:
        model: YOLO model instance
        test_dataset: Supervision DetectionDataset
        config: Dict with 'imgsz' key
        task: Optional ClearML task for logging
    """
    # Accumulators for detections
    all_gt_image_ids = []
    all_gt_boxes = []
    all_gt_labels = []
    all_pred_image_ids = []
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []

    # Inference loop
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

        # Accumulate ground truth
        if annotation.xyxy is not None and len(annotation.xyxy) > 0:
            all_gt_image_ids.extend([idx] * len(annotation.xyxy))
            all_gt_boxes.append(annotation.xyxy)
            all_gt_labels.extend(annotation.class_id)

        # Accumulate predictions (convert from Ultralytics format)
        if len(pred_results) > 0:
            res = pred_results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                all_pred_image_ids.extend([idx] * len(boxes))
                all_pred_boxes.append(boxes.xyxy.cpu().numpy())
                all_pred_labels.extend(boxes.cls.cpu().numpy().astype(int))
                all_pred_scores.extend(boxes.conf.cpu().numpy())

    # Build ODData
    data = ODData(
        gt_image_ids=np.array(all_gt_image_ids, dtype=np.int64),
        gt_boxes=np.vstack(all_gt_boxes).astype(np.float32) if all_gt_boxes else np.empty((0, 4), dtype=np.float32),
        gt_labels=np.array(all_gt_labels, dtype=np.int64),
        pred_image_ids=np.array(all_pred_image_ids, dtype=np.int64),
        pred_boxes=np.vstack(all_pred_boxes).astype(np.float32) if all_pred_boxes else np.empty((0, 4), dtype=np.float32),
        pred_labels=np.array(all_pred_labels, dtype=np.int64),
        pred_scores=np.array(all_pred_scores, dtype=np.float32),
        class_names={i: name for i, name in enumerate(test_dataset.classes)},
    )

    # Configure loggers
    loggers = [StdoutLogger()]
    if task is not None:
        loggers.append(ClearMLLogger(task=task))

    # Evaluate using foresight_metrics
    od = ObjectDetection(
        metrics=["mAP", "precision_recall"],
        loggers=loggers,
    )
    
    result = od.evaluate_data(data)
    
    return result


# Example usage in your main script:
# 
# result = evaluate_model(model, test_dataset, config, task)
# print(result.metrics)
# # {'mAP': 0.65, 'mAP@0.5': 0.82, 'mAP@0.75': 0.48, 'precision': 0.85, ...}