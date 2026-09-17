# -*- coding: utf-8 -*-
"""
verify_test_case.py
-------------------
Inference and Verification Tool for Person Detection using YOLO.

Performs object detection on a specific test image, extracts bounding box coordinates,
class IDs, confidence scores, measures IoU overlap with ground truth, and outputs
visual and text verification reports.

Usage:
    py verify_test_case.py
    py verify_test_case.py --image_index 0
    py verify_test_case.py --image_index 2
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add src to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import load_config
from annotation_converter import read_yolo_label_file, yolo_to_xyxy
from model_trainer import load_yolo_model
from inference import detect_objects_single_image, calculate_iou
from visualization import draw_bounding_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Test and Verify Person Detection on a Specific Test Image")
    parser.add_argument("--image_index", type=int, default=0, help="Index of test image (0 to N_test-1)")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Path to training config")
    parser.add_argument("--model_path", type=str, default="models/best.pt", help="Path to saved model weights")
    parser.add_argument("--save_plot", type=str, default="results/detections/test_case_verification.jpg", help="Path to save visual verification output")
    return parser.parse_args()


def verify_single_image(
    image_index: int = 0,
    config_path: str = "config/training_config.yaml",
    model_path: str = "models/best.pt",
    save_plot: str = "results/detections/test_case_verification.jpg"
):
    print("=" * 75)
    print(f" YOLO PERSON DETECTION - TEST CASE VERIFICATION")
    print(f" Target Test Image Index: #{image_index}")
    print("=" * 75)

    # 1. Load configuration & Environment
    cfg = load_config(config_path)
    print(f"\n[Step 1] Configuration Initialization")
    print(f"  - Model Architecture : {cfg['model']['architecture']}")
    print(f"  - Target Classes     : {cfg['model']['class_names']}")
    print(f"  - Confidence Cutoff  : {cfg['inference']['conf_threshold']}")
    print(f"  - NMS IoU Threshold  : {cfg['inference']['iou_threshold']}")

    # 2. Load YOLO Model
    print(f"\n[Step 2] YOLO Model Loading")
    arch = cfg["model"].get("architecture", "yolov8n.pt")
    if os.path.exists(model_path):
        arch_to_use = model_path
    else:
        arch_to_use = arch

    model = load_yolo_model(arch_to_use)
    print(f"  - Model Weights Loaded: '{arch_to_use}'")

    # 3. Load Test Image & Ground-Truth Labels
    test_img_dir = os.path.join(cfg["dataset"].get("images_dir", "data/images"), "test")
    test_lbl_dir = os.path.join(cfg["dataset"].get("labels_dir", "data/labels"), "test")

    test_files = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(test_files) == 0:
        raise FileNotFoundError(f"No test images found in '{test_img_dir}'. Run main.py first.")

    if image_index < 0 or image_index >= len(test_files):
        raise ValueError(f"Image index {image_index} is out of bounds for test set size ({len(test_files)}).")

    img_path = test_files[image_index]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(test_lbl_dir, f"{base_name}.txt")

    img_bgr = cv2.imread(img_path)
    img_h, img_w = img_bgr.shape[:2]

    # Ground truth annotations
    gt_ann_raw = read_yolo_label_file(lbl_path)
    gt_boxes = [yolo_to_xyxy(ann["bbox"], img_w, img_h) for ann in gt_ann_raw]
    gt_classes = [ann["class_id"] for ann in gt_ann_raw]

    print(f"\n[Step 3] Test Sample Acquisition")
    print(f"  - Image File        : '{os.path.basename(img_path)}'")
    print(f"  - Image Resolution  : {img_w}x{img_h} pixels")
    print(f"  - Ground-Truth Count: {len(gt_boxes)} persons")
    for i, (gbox, gcls) in enumerate(zip(gt_boxes, gt_classes)):
        cname = cfg["model"]["class_names"][gcls] if gcls < len(cfg["model"]["class_names"]) else f"class_{gcls}"
        print(f"    * GT Person {i+1}: Class='{cname}', Box=[{gbox[0]}, {gbox[1]}, {gbox[2]}, {gbox[3]}]")

    # 4. Model Object Detection & NMS
    print(f"\n[Step 4] YOLO Object Detection Inference & Non-Maximum Suppression (NMS)")
    det_res = detect_objects_single_image(
        model,
        img_path,
        conf_threshold=cfg["inference"].get("conf_threshold", 0.25),
        iou_threshold=cfg["inference"].get("iou_threshold", 0.45)
    )

    pred_boxes   = det_res["boxes"]
    pred_classes = det_res["classes"]
    pred_scores  = det_res["scores"]

    print(f"  - Detected Objects Count : {len(pred_boxes)}")
    print(f"  - Processing Latency     : {det_res['latency_ms']:.2f} ms ({det_res['fps']:.1f} FPS)")

    matched_ious = []
    print(f"\n[Step 5] Bounding Box Coordinate & Confidence Verification")
    print("  Obj # | Class Name | Conf Score | Bounding Box (XYXY)     | IoU Match | Status")
    print("  ------+------------+------------+-------------------------+-----------+-------------")
    for i in range(len(pred_boxes)):
        box  = pred_boxes[i]
        c_id = pred_classes[i] if len(pred_classes) > i else 0
        sc   = pred_scores[i] if len(pred_scores) > i else 1.0
        cname = cfg["model"]["class_names"][c_id] if c_id < len(cfg["model"]["class_names"]) else f"class_{c_id}"

        # Match with best GT box
        best_iou = 0.0
        for gbox in gt_boxes:
            iou = calculate_iou(box, gbox)
            if iou > best_iou:
                best_iou = iou
        matched_ious.append(best_iou)

        status_flag = "MATCH [OK]" if best_iou >= 0.50 else "DETECTED"
        print(f"  #{i+1:<4} | {cname:<10} |   {sc*100.0:5.1f}%    | [{box[0]:4.0f}, {box[1]:4.0f}, {box[2]:4.0f}, {box[3]:4.0f}] |   {best_iou:6.4f}  | {status_flag}")

    # 6. Render & Save Visual Verification Image
    os.makedirs(os.path.dirname(save_plot), exist_ok=True)
    annotated_img = draw_bounding_boxes(
        img_bgr,
        pred_boxes,
        pred_classes,
        pred_scores,
        class_names={i: name for i, name in enumerate(cfg["model"]["class_names"])}
    )

    cv2.imwrite(save_plot, annotated_img)
    print(f"\n[Output] Saved visual verification detection overlay image to: '{save_plot}'")

    is_verified = (len(pred_boxes) > 0)
    print(f"\n[Step 6] Final Decision & Verification Summary")
    print(f"  - Object Detection Status : {'PASSED [OK] (Valid Detections Extracted)' if is_verified else 'NO OBJECTS DETECTED'}")
    print("=" * 75 + "\n")

    return {
        "image_index": image_index,
        "gt_count": len(gt_boxes),
        "pred_count": len(pred_boxes),
        "mean_iou": float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0,
        "latency_ms": det_res["latency_ms"],
        "save_plot": save_plot
    }


if __name__ == "__main__":
    args = parse_args()
    verify_single_image(
        image_index=args.image_index,
        config_path=args.config,
        model_path=args.model_path,
        save_plot=args.save_plot
    )
