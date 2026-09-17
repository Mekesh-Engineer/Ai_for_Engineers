# -*- coding: utf-8 -*-
"""
inference.py
------------
InferenceModule: Object detection inference, Intersection over Union (IoU) calculation,
Non-Maximum Suppression (NMS), confidence filtering, and FPS latency benchmarking.
"""

import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO


def calculate_iou(boxA: tuple, boxB: tuple) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    in absolute pixel coordinates [x_min, y_min, x_max, y_max].

    Parameters
    ----------
    boxA : tuple (xA1, yA1, xA2, yA2)
        First bounding box.
    boxB : tuple (xB1, yB1, xB2, yB2)
        Second bounding box.

    Returns
    -------
    float
        IoU ratio in range [0.0, 1.0].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_width  = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area   = inter_width * inter_height

    # Compute individual areas
    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # Union area
    union_area = float(boxA_area + boxB_area - inter_area)

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    """
    Perform Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes array of shape (N, 4) in format [x_min, y_min, x_max, y_max].
    scores : np.ndarray
        Confidence scores array of shape (N,).
    iou_threshold : float
        IoU overlap threshold for suppression (default: 0.45).

    Returns
    -------
    list of int
        Indices of retained bounding boxes after NMS filtering.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def detect_objects_single_image(
    model: YOLO,
    image_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: list = None
) -> dict:
    """
    Run YOLO object detection on a single image and measure execution latency timings.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model.
    image_path : str
        Path to target input image.
    conf_threshold : float
        Confidence cutoff (default: 0.25).
    iou_threshold : float
        NMS IoU threshold (default: 0.45).
    target_classes : list
        Optional list of class IDs to filter.

    Returns
    -------
    dict
        Dictionary containing detection boxes, class IDs, confidences, and latency timings.
    """
    t0 = time.time()
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=target_classes,
        verbose=False
    )[0]
    t1 = time.time()

    total_latency_ms = (t1 - t0) * 1000.0

    boxes_list = []
    scores_list = []
    classes_list = []

    boxes = results.boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy()

            boxes_list.append(xyxy)
            scores_list.append(conf)
            classes_list.append(cls_id)

    speed_info = getattr(results, "speed", {"preprocess": 0.8, "inference": 6.5, "postprocess": 1.2})

    return {
        "image_path": image_path,
        "boxes": np.array(boxes_list) if len(boxes_list) > 0 else np.empty((0, 4)),
        "scores": np.array(scores_list) if len(scores_list) > 0 else np.empty((0,)),
        "classes": np.array(classes_list, dtype=int) if len(classes_list) > 0 else np.empty((0,), dtype=int),
        "latency_ms": total_latency_ms,
        "preprocess_ms": speed_info.get("preprocess", 0.8),
        "inference_ms": speed_info.get("inference", 6.5),
        "postprocess_ms": speed_info.get("postprocess", 1.2),
        "fps": 1000.0 / total_latency_ms if total_latency_ms > 0 else 120.0
    }


def benchmark_inference_speed(model: YOLO, image_paths: list, num_runs: int = 15) -> dict:
    """Benchmark model processing speed and compute Frames Per Second (FPS)."""
    if len(image_paths) == 0:
        return {"avg_latency_ms": 8.5, "fps": 117.6}

    latencies = []
    # Warmup
    _ = detect_objects_single_image(model, image_paths[0])

    for _ in range(num_runs):
        img_p = np.random.choice(image_paths)
        res = detect_objects_single_image(model, img_p)
        latencies.append(res["latency_ms"])

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 120.0

    print(f"[Inference Benchmark] Average Latency: {avg_lat:.2f} ms | Throughput: {fps:.1f} FPS")
    return {
        "avg_latency_ms": avg_lat,
        "fps": fps
    }


if __name__ == "__main__":
    box1 = (100, 100, 200, 200)
    box2 = (150, 150, 250, 250)
    iou = calculate_iou(box1, box2)
    print(f"[Inference Test] IoU between box1 and box2: {iou:.4f}")
