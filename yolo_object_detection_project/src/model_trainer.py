# -*- coding: utf-8 -*-
"""
model_trainer.py
----------------
ModelTrainerModule: Model loading, initialization, fine-tuning, and weight
checkpoint saving using Ultralytics YOLOv8 for Person Object Detection.
"""

import os
import shutil
import torch
import yaml
from ultralytics import YOLO


def load_yolo_model(architecture: str = "yolov8n.pt") -> YOLO:
    """
    Load pre-trained YOLO model weights or trained checkpoint.

    Parameters
    ----------
    architecture : str
        Model weight file or architecture name (default: 'yolov8n.pt').

    Returns
    -------
    YOLO
        Ultralytics YOLO model instance.
    """
    print(f"[ModelTrainer] Loading YOLO model architecture/weights '{architecture}'...")
    model = YOLO(architecture)
    return model


def train_yolo_model(config: dict) -> tuple:
    """
    Execute YOLO model training / fine-tuning on Person Detection dataset.

    Parameters
    ----------
    config : dict
        Training configuration dictionary.

    Returns
    -------
    tuple (model, training_results)
    """
    data_yaml_path = config["dataset"].get("data_yaml_path", "datasets/data.yaml")
    arch = config["model"].get("architecture", "yolov8n.pt")
    epochs = config["training"].get("epochs", 25)
    imgsz = config["dataset"].get("img_size", 640)
    batch_size = config["training"].get("batch_size", 8)
    lr0 = config["training"].get("learning_rate", 0.01)
    patience = config["training"].get("early_stopping_patience", 10)
    device_cfg = str(config["training"].get("device", "0"))
    model_save_dir = config["output"].get("model_save_dir", "models/")

    os.makedirs(model_save_dir, exist_ok=True)

    # Check CUDA availability
    if device_cfg != "cpu" and not torch.cuda.is_available():
        print("[ModelTrainer] CUDA GPU not detected. Automatically falling back to 'cpu'.")
        device_cfg = "cpu"

    # Initialize model
    model = load_yolo_model(arch)

    print("\n---------------- Starting YOLO Person Detector Fine-Tuning ----------------")
    print(f"  Architecture      : {arch}")
    print(f"  Dataset YAML Path : {data_yaml_path}")
    print(f"  Image Resolution  : {imgsz}x{imgsz}")
    print(f"  Epochs            : {epochs}")
    print(f"  Batch Size        : {batch_size}")
    print(f"  Compute Device    : {device_cfg}")
    print(f"  Early Stop Patience: {patience}")
    print("----------------------------------------------------------------------------\n")

    results = None
    if epochs > 0:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            patience=patience,
            device=device_cfg,
            project="runs/detect",
            name="person_exp",
            exist_ok=True,
            verbose=True,
            # Augmentation parameters tuned for pedestrians
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            fliplr=0.5,
            flipud=0.0
        )

        # Save best model to models/best.pt
        target_best_path = config["output"].get("best_model_path", "models/best.pt")
        copied = False

        if results is not None and hasattr(results, "save_dir"):
            trainer_best = os.path.join(str(results.save_dir), "weights", "best.pt")
            if os.path.exists(trainer_best):
                shutil.copy(trainer_best, target_best_path)
                print(f"[ModelTrainer] Copied best model weights from '{trainer_best}' to '{target_best_path}'")
                copied = True

        if not copied:
            best_weights_src = os.path.join("runs/detect/person_exp", "weights", "best.pt")
            if os.path.exists(best_weights_src):
                shutil.copy(best_weights_src, target_best_path)
                print(f"[ModelTrainer] Copied best model weights to '{target_best_path}'")
                copied = True

        if not copied:
            model.save(target_best_path)
            print(f"[ModelTrainer] Saved model checkpoint directly to '{target_best_path}'")
    else:
        print("[ModelTrainer] Fast evaluation mode (epochs=0): Using existing trained weights directly.")
        target_best_path = config["output"].get("best_model_path", "models/best.pt")
        if os.path.exists(target_best_path):
            model = load_yolo_model(target_best_path)

    return model, results


if __name__ == "__main__":
    test_cfg = {
        "dataset": {"data_yaml_path": "datasets/data.yaml", "img_size": 640},
        "model": {"architecture": "yolov8n.pt"},
        "training": {"epochs": 1, "batch_size": 4, "device": "cpu"},
        "output": {"best_model_path": "models/test_best.pt"}
    }
    try:
        m, r = train_yolo_model(test_cfg)
        print("[ModelTrainer Test] Training completed successfully!")
    except Exception as e:
        print(f"[ModelTrainer Test] Setup note: {e}")
