# -*- coding: utf-8 -*-
"""
annotation_converter.py
------------------------
AnnotationConverterModule: Conversion between absolute pixel bounding box coordinates
[x_min, y_min, x_max, y_max] and normalized YOLO format [x_center, y_center, width, height].
Includes box clipping, area validation, and multi-format support.
"""

import os
import numpy as np


def xyxy_to_yolo(box: tuple, img_width: int, img_height: int) -> tuple:
    """
    Convert absolute pixel coordinates [x_min, y_min, x_max, y_max]
    to normalized YOLO format [x_center, y_center, width, height].

    Parameters
    ----------
    box : tuple (x_min, y_min, x_max, y_max)
        Bounding box in absolute pixel coordinates.
    img_width : int
        Width of the image in pixels.
    img_height : int
        Height of the image in pixels.

    Returns
    -------
    tuple (x_center, y_center, width, height)
        Normalized values in range [0.0, 1.0].
    """
    x_min, y_min, x_max, y_max = box
    
    # Clip to image boundaries
    x_min = max(0.0, min(float(x_min), float(img_width)))
    y_min = max(0.0, min(float(y_min), float(img_height)))
    x_max = max(0.0, min(float(x_max), float(img_width)))
    y_max = max(0.0, min(float(y_max), float(img_height)))

    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 0.0, 0.0

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = (x_min + x_max) / 2.0 * dw
    y_center = (y_min + y_max) / 2.0 * dh
    width = (x_max - x_min) * dw
    height = (y_max - y_min) * dh

    # Clip to [0.0, 1.0] boundary
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    width    = float(np.clip(width, 0.0, 1.0))
    height   = float(np.clip(height, 0.0, 1.0))

    return x_center, y_center, width, height


def yolo_to_xyxy(yolo_box: tuple, img_width: int, img_height: int) -> tuple:
    """
    Convert normalized YOLO format [x_center, y_center, width, height]
    to absolute pixel coordinates [x_min, y_min, x_max, y_max].

    Parameters
    ----------
    yolo_box : tuple (x_center, y_center, width, height)
        Normalized values in range [0.0, 1.0].
    img_width : int
        Width of the image in pixels.
    img_height : int
        Height of the image in pixels.

    Returns
    -------
    tuple (x_min, y_min, x_max, y_max)
        Absolute pixel coordinates as integers.
    """
    x_center, y_center, w, h = yolo_box

    x_min = int(round((x_center - w / 2.0) * img_width))
    y_min = int(round((y_center - h / 2.0) * img_height))
    x_max = int(round((x_center + w / 2.0) * img_width))
    y_max = int(round((y_center + h / 2.0) * img_height))

    # Boundary clip
    x_min = max(0, min(img_width, x_min))
    y_min = max(0, min(img_height, y_min))
    x_max = max(0, min(img_width, x_max))
    y_max = max(0, min(img_height, y_max))

    return x_min, y_min, x_max, y_max


def read_yolo_label_file(label_path: str) -> list:
    """
    Read a YOLO annotation .txt file.

    Parameters
    ----------
    label_path : str
        Path to YOLO format label text file.

    Returns
    -------
    list of dict
        List of bounding box dicts: [{'class_id': int, 'bbox': (xc, yc, w, h)}]
    """
    if not os.path.exists(label_path):
        return []

    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                # Skip invalid or empty bounding boxes
                if w > 0 and h > 0:
                    annotations.append({
                        "class_id": class_id,
                        "bbox": (xc, yc, w, h)
                    })
    return annotations


def write_yolo_label_file(label_path: str, annotations: list) -> None:
    """
    Write annotations to a YOLO format .txt file.

    Parameters
    ----------
    label_path : str
        Target file path.
    annotations : list of dict
        List of dicts: [{'class_id': int, 'bbox': (xc, yc, w, h)}]
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            cls_id = ann["class_id"]
            xc, yc, w, h = ann["bbox"]
            if w > 0 and h > 0:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


if __name__ == "__main__":
    w, h = 640, 640
    pixel_box = (100, 150, 350, 400)
    yolo_box = xyxy_to_yolo(pixel_box, w, h)
    recon_box = yolo_to_xyxy(yolo_box, w, h)
    print(f"[AnnotationConverter Test] Original Pixel: {pixel_box} -> YOLO: {yolo_box} -> Reconstructed: {recon_box}")
