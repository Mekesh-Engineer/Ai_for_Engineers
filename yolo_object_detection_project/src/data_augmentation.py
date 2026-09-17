# -*- coding: utf-8 -*-
"""
data_augmentation.py
--------------------
DataAugmentationModule: Data augmentation pipeline for YOLO object detection,
including random horizontal flipping, color jitter, and bounding box coordinate transformation.
"""

import numpy as np
import cv2


def apply_horizontal_flip(image: np.ndarray, annotations: list) -> tuple:
    """
    Flip image horizontally and adjust YOLO bounding box coordinates accordingly.

    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W, C).
    annotations : list of dict
        List of dicts: [{'class_id': int, 'bbox': (xc, yc, w, h)}]

    Returns
    -------
    tuple (flipped_image, flipped_annotations)
    """
    flipped_image = cv2.flip(image, 1)  # 1 = horizontal flip
    flipped_annotations = []

    for ann in annotations:
        cls_id = ann["class_id"]
        xc, yc, w, h = ann["bbox"]
        
        # Horizontally flipped x_center = 1.0 - x_center
        new_xc = 1.0 - xc

        flipped_annotations.append({
            "class_id": cls_id,
            "bbox": (new_xc, yc, w, h)
        })

    return flipped_image, flipped_annotations


def apply_color_jitter(image: np.ndarray) -> np.ndarray:
    """
    Randomly adjust brightness, contrast, and saturation.

    Parameters
    ----------
    image : np.ndarray
        RGB/BGR image array in uint8.

    Returns
    -------
    np.ndarray
        Color jittered image array.
    """
    img_float = image.astype(np.float32)

    # Random brightness shift [-20, +20]
    brightness = np.random.uniform(-20, 20)
    img_float = np.clip(img_float + brightness, 0, 255)

    # Random contrast scaling [0.85, 1.15]
    contrast = np.random.uniform(0.85, 1.15)
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = np.clip((img_float - mean_val) * contrast + mean_val, 0, 255)

    return img_float.astype(np.uint8)


def augment_training_sample(image: np.ndarray, annotations: list) -> tuple:
    """Apply full augmentation pipeline to a training sample."""
    aug_image = apply_color_jitter(image)

    # 50% probability of horizontal flip
    if np.random.rand() < 0.5:
        aug_image, aug_annotations = apply_horizontal_flip(aug_image, annotations)
    else:
        aug_annotations = annotations

    return aug_image, aug_annotations


if __name__ == "__main__":
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_anns = [{"class_id": 0, "bbox": (0.3, 0.4, 0.2, 0.3)}]
    f_img, f_anns = apply_horizontal_flip(dummy_img, dummy_anns)
    print(f"[Augmentation Test] Original xc: {dummy_anns[0]['bbox'][0]} -> Flipped xc: {f_anns[0]['bbox'][0]}")
