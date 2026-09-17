# YOLO Moving Object Detection & Real-Time Tracking Project

## 📋 Overview
A complete, modular, real-time object detection and multi-object tracking (MOT) pipeline using the **YOLO (YOLOv8 / YOLO11)** framework and **ByteTrack** applied to **Real-World Pedestrian / Moving Object Detection** on the **Penn-Fudan Pedestrian Dataset**.

The project orchestrates:
- Real-world dataset acquisition, normalization, and automatic `train/val/test` (70% / 15% / 15%) partitioning.
- Coordinate transformation between absolute pixel bounding boxes `[xmin, ymin, xmax, ymax]` and normalized YOLO coordinates `<class_id> <x_center> <y_center> <width> <height>`.
- Pre-trained backbone loading (`yolov8n.pt`) and fine-tuning with spatial & photometric augmentations (Mosaic, MixUp, HSV jitter, random flips).
- Non-Maximum Suppression (NMS) and Intersection over Union (IoU) overlap calculation.
- Mean Average Precision ($\text{mAP@0.5}$, $\text{mAP@0.5:0.95}$), Precision, Recall, and F1-Score evaluation.
- High-throughput inference latency benchmarking (FPS).
- Multi-Object Tracking (MOT) using **ByteTrack** for persistent identity assignment and motion trajectory visualization on video streams and webcam feeds.
- Diagnostic visualization suite (annotated detection grids, Precision-Recall curves, confusion matrix heatmaps, and IoU histograms).

---

## 🗂️ Project Structure
```
yolo_object_detection_project/
├── config/
│   └── training_config.yaml       # Central configuration file
├── data/
│   ├── images/                    # Train / val / test image partitions
│   ├── labels/                    # YOLO format normalized annotation files (.txt)
│   └── raw_penn_fudan/            # Downloaded raw Penn-Fudan dataset archive
├── datasets/
│   └── data.yaml                  # YOLO dataset specification file
├── models/
│   └── best.pt                    # Fine-tuned YOLO model checkpoint
├── notebooks/
│   └── yolo_training.ipynb        # Interactive Jupyter notebook
├── results/
│   ├── detections/
│   │   ├── sample_1.jpg           # Detection overlays with bboxes & confidence
│   │   ├── sample_2.jpg
│   │   └── test_case_verification.jpg
│   ├── metrics/
│   │   ├── precision_recall_curve.png # Precision-Recall curves
│   │   ├── confusion_matrix.png   # Confusion matrix heatmap
│   │   ├── iou_distribution.png   # Bounding box IoU fit histogram
│   │   └── detection_report.txt   # Comprehensive evaluation report
│   ├── tracked_output.mp4         # Rendered video tracking output with trajectories
│   └── training_results.csv       # Training loss and mAP progression history
├── src/
│   ├── data_loader.py             # Penn-Fudan dataset acquisition & YOLO YAML generation
│   ├── annotation_converter.py    # Pixel [xyxy] <-> Normalized YOLO format conversion
│   ├── data_augmentation.py       # Random flips, color jitter & bbox transforms
│   ├── model_trainer.py           # YOLO model fine-tuning & checkpointing
│   ├── inference.py               # Inference, NMS, IoU & FPS speed benchmark
│   ├── evaluation.py              # mAP@0.5, Precision, Recall & F1 metrics
│   ├── tracker.py                 # Multi-Object Tracking (ByteTrack) & trajectory tracing
│   └── visualization.py           # Bounding box rendering & diagnostic plots
├── main.py                        # End-to-end pipeline runner
├── verify_test_case.py            # Test image detection & verification tool
└── README.md
```

---

## ⚙️ Installation

```bash
pip install ultralytics opencv-python pyyaml torch torchvision numpy matplotlib
```

---

## 🚀 Quick Start

### 1. Run the Full End-to-End Pipeline
```bash
cd yolo_object_detection_project
py main.py
```

### 2. Prepare the Real-World Dataset Only
```bash
py main.py --mode prepare_data
```

### 3. Retrain the YOLO Model
```bash
py main.py --mode train --epochs 15 --batch_size 8
```

### 4. Evaluate Held-Out Test Set Metrics
```bash
py main.py --mode eval
```

### 5. Run Multi-Object Video Tracking (ByteTrack)
```bash
# Test on synthetic/sample moving video:
py main.py --mode track

# Test on custom video file:
py main.py --mode track --source path/to/video.mp4

# Test on live webcam feed:
py main.py --mode track --source 0
```

### 6. Verify Detection on a Specific Test Sample
```bash
py verify_test_case.py --image_index 0
py verify_test_case.py --image_index 3
```

---

## 🔬 Methodology & Workflow

| Step | Module | Description |
|------|--------|-------------|
| 1. Configuration | `config/` | Define image size (640), confidence cutoff (0.25), IoU cutoff (0.45) |
| 2. Data Acquisition | `data_loader.py` | Extract real-world Penn-Fudan pedestrian images & bounding boxes |
| 3. Format Conversion | `annotation_converter.py` | Convert absolute pixels to normalized YOLO $[0.0, 1.0]$ bounds |
| 4. YAML Specification | `data_loader.py` | Create `data.yaml` defining paths and class names `['person']` |
| 5. YOLO Architecture | `model_trainer.py` | Load pre-trained `yolov8n.pt` backbone with FPN feature head |
| 6. Fine-Tuning | `model_trainer.py` | Fine-tune network weights on moving pedestrian dataset |
| 7. Inference & NMS | `inference.py` | Predict bounding boxes, apply Non-Maximum Suppression (NMS) |
| 8. Metrics Evaluation | `evaluation.py` | Calculate IoU, Precision, Recall, F1, $\text{mAP@0.5}$, $\text{mAP@0.5:0.95}$ |
| 9. Speed Benchmark | `inference.py` | Benchmark per-frame processing latency and calculate FPS |
| 10. Multi-Object Tracking | `tracker.py` | Track persistent identities (ByteTrack) and draw trajectory trails |
| 11. Visual Overlay | `visualization.py` | Draw bounding box rectangles, class labels, confidence scores |

---

## 🏗️ YOLO Detection Pipeline

```
Input Image (640 × 640 × 3 RGB)
    ↓
[CSPDarknet Backbone (Multi-Scale Feature Extraction)]
    ↓
[Feature Pyramid Network (FPN / PANet Path Aggregation)]
    ↓
[Multi-Scale Detection Head]
    ├─ Bounding Box Offsets (x_center, y_center, width, height)
    ├─ Objectness Confidence Score P(Object)
    └─ Class Probabilities P(Class | Object)
    ↓
[Non-Maximum Suppression (NMS, Conf >= 0.25, IoU >= 0.45)]
    ↓
[ByteTrack Multi-Object Tracker (Kalman Filter + Hungarian Association)]
    ↓
Output Bounding Boxes + Persistent IDs + Trajectory Trails + HUD Stats
```

---

## 📊 Performance Metrics

- **Mean Average Precision ($\text{mAP@0.5}$)**: Evaluates bounding box overlap at IoU $\ge 0.50$.
- **$\text{mAP@0.5:0.95}$**: Averages mAP across 10 IoU thresholds from $0.50$ to $0.95$ in steps of $0.05$.
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Accuracy of detected boxes).
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Completeness of ground-truth objects found).
- **Inference FPS**: $\frac{1000}{\text{Latency (ms)}}$ real-time throughput.
