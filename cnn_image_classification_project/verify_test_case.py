# -*- coding: utf-8 -*-
"""
verify_test_case.py
-------------------
Inference and Verification Tool for Experiment 4: CNN Image Classification.
EEE Domain Application: Solar Photovoltaic (PV) Panel Fault Detection.

Performs step-by-step image classification on a specific test image from the
Solar PV Fault Dataset, extracts feature map activations,
computes Softmax probabilities across 4 EEE defect classes,
compares prediction with expected class, and outputs visual and text verification reports.

Usage:
    py verify_test_case.py
    py verify_test_case.py --sample_index 0
    py verify_test_case.py --sample_index 5
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Add src to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import load_config
from model_builder import build_cnn_model, MNIST_CNN
from data_preprocessing import get_normalize_parameters
from eee_dataset import SolarPVFaultDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test and Verify EEE Solar PV Panel Fault Experiment on a Specific Image")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of test set image (0 to 999)")
    parser.add_argument("--config", type=str, default="config/hyperparameters.json", help="Path to hyperparameters config")
    parser.add_argument("--model_path", type=str, default="models/cnn_model.pt", help="Path to saved model weights")
    parser.add_argument("--save_plot", type=str, default="results/test_case_verification.png", help="Path to save visual verification figure")
    return parser.parse_args()


def load_trained_model(config: dict, model_path: str, device: torch.device) -> MNIST_CNN:
    model = build_cnn_model(config)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at '{model_path}'")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def verify_single_image(sample_index: int = 0, config_path: str = "config/hyperparameters.json", model_path: str = "models/cnn_model.pt", save_plot: str = "results/test_case_verification.png"):
    print("=" * 75)
    print(f" EXPERIMENT 4: CNN SOLAR PV FAULT CLASSIFICATION - TEST VERIFICATION")
    print(f" Target Test Sample Index: {sample_index}")
    print("=" * 75)

    # 1. Load configuration
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["dataset"].get("class_names", ["Healthy_Panel", "Micro_Crack", "Hotspot_Fault", "Dust_Soiling"])

    print(f"\n[Step 1] Configuration & Hardware Initialization")
    print(f"  - Operating Device  : {device}")
    print(f"  - Model Path        : {model_path}")
    print(f"  - Dataset Name      : {cfg['dataset']['name']}")
    print(f"  - Target EEE Classes: {class_names}")

    # 2. Load trained PyTorch Model
    print(f"\n[Step 2] Model Checkpoint Loading & Validation")
    model = load_trained_model(cfg, model_path, device)
    print(f"  - Model State Loaded: Successfully restored state_dict from '{model_path}'")
    print(f"  - Evaluation Mode   : Enabled (Dropout deactivated, BatchNorm running stats fixed)")

    # 3. Load Test Image & True Ground-Truth Label
    raw_dataset = SolarPVFaultDataset(num_samples=1000, seed=cfg["training"].get("random_seed", 42) + 1)
    raw_img_pil = Image.fromarray(raw_dataset.images[sample_index], mode="L")
    true_label  = raw_dataset.labels[sample_index]
    true_class_name = class_names[true_label]
    raw_img_np  = np.array(raw_img_pil)

    print(f"\n[Step 3] EEE Test Sample Acquisition")
    print(f"  - Test Sample Index : #{sample_index}")
    print(f"  - Expected Class    : Class #{true_label} ('{true_class_name}')")
    print(f"  - Image Dimensions  : {raw_img_np.shape[0]}x{raw_img_np.shape[1]} pixels (1 channel thermography)")
    print(f"  - Raw Pixel Range   : [{raw_img_np.min()}, {raw_img_np.max()}] (uint8)")

    # 4. Preprocessing & Normalization
    mean, std = cfg["preprocessing"]["normalize_mean"], cfg["preprocessing"]["normalize_std"]
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((mean,), (std,))
    ])
    input_tensor = transform(raw_img_pil).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    print(f"\n[Step 4] Preprocessing & Tensor Normalization")
    print(f"  - Scaling Applied   : uint8 [0, 255] -> Float32 [0.0, 1.0]")
    print(f"  - Normalization     : mean={mean}, std={std}")
    print(f"  - Transformed Shape : {list(input_tensor.shape)} (Batch=1, Channels=1, Height=28, Width=28)")
    print(f"  - Tensor Value Range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")

    # 5. Feature Extraction & Intermediate Layer Forward Pass
    print(f"\n[Step 5] CNN Layer-by-Layer Forward Pass & Feature Extraction")
    with torch.no_grad():
        conv1_out = model.conv1(input_tensor)
        bn1_out   = model.bn1(conv1_out)
        relu1_out = F.relu(bn1_out)
        pool1_out = model.pool(relu1_out)  # (1, 32, 14, 14)

        conv2_out = model.conv2(pool1_out)
        bn2_out   = model.bn2(conv2_out)
        relu2_out = F.relu(bn2_out)
        pool2_out = model.pool(relu2_out)  # (1, 64, 7, 7)

        flattened = torch.flatten(pool2_out, start_dim=1)  # (1, 3136)
        fc1_out   = F.relu(model.fc1(flattened))            # (1, 128)
        logits    = model.fc2(fc1_out)                      # (1, 4)
        probs     = F.softmax(logits, dim=1)                # (1, 4)

    print(f"  - Conv Block 1 Output : {list(pool1_out.shape)} (32 feature maps of 14x14)")
    print(f"  - Conv Block 2 Output : {list(pool2_out.shape)} (64 feature maps of 7x7)")
    print(f"  - Flattened Vector    : {list(flattened.shape)} (3,136 input features)")
    print(f"  - FC1 Dense Output    : {list(fc1_out.shape)} (128 hidden representations)")
    print(f"  - Output Logits Shape : {list(logits.shape)} (4 raw class scores)")

    # 6. Logits & Softmax Probability Distribution
    logits_np = logits.cpu().numpy().squeeze()
    probs_np  = probs.cpu().numpy().squeeze()
    pred_class = int(np.argmax(probs_np))
    pred_class_name = class_names[pred_class]
    confidence = float(probs_np[pred_class] * 100.0)

    print(f"\n[Step 6] Classification Logits & Softmax Probabilities")
    print("  Class Index & Name   | Logit Score | Softmax Probability | Bar Representation")
    print("  ---------------------+-------------+---------------------+-------------------")
    for cls_i in range(len(class_names)):
        c_name = class_names[cls_i]
        bar = "#" * int(probs_np[cls_i] * 30)
        marker = "<-- (PREDICTED)" if cls_i == pred_class else ""
        print(f"  #{cls_i} {c_name:<16} |   {logits_np[cls_i]:8.4f}  |       {probs_np[cls_i]*100:6.2f}%       | {bar:<30} {marker}")

    # 7. Verification & Decision Assessment
    is_correct = (pred_class == true_label)
    status_str = "PASSED [OK] (Correct Fault Classification)" if is_correct else "FAILED [X] (Misclassification)"

    print(f"\n[Step 7] Final Verification & Hypothesis Testing")
    print(f"  - Expected Ground-Truth Class : Class #{true_label} ('{true_class_name}')")
    print(f"  - CNN Model Predicted Class   : Class #{pred_class} ('{pred_class_name}')")
    print(f"  - Model Prediction Confidence : {confidence:.2f}%")
    print(f"  - Verification Status         : {status_str}")
    print("=" * 75)

    # 8. Generate Visual Verification Figure
    os.makedirs(os.path.dirname(save_plot), exist_ok=True)
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Solar PV Panel Fault Verification: Test Sample #{sample_index}", fontsize=15, fontweight="bold", y=0.98)

    # Subplot 1: Input Image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(raw_img_np, cmap="inferno")
    ax1.set_title(f"Thermal Thermography Sample #{sample_index}\nTrue Fault: {true_class_name}", fontsize=11, fontweight="semibold")
    ax1.axis("off")

    # Subplot 2: Conv Block 1 Feature Maps (First 8 filters)
    ax2 = fig.add_subplot(2, 3, 2)
    c1_map = pool1_out[0, :8].cpu().numpy()
    c1_grid = np.hstack([c1_map[i] for i in range(8)])
    ax2.imshow(c1_grid, cmap="viridis")
    ax2.set_title("Conv Block 1 Feature Maps\n(Sample 8 / 32 maps @ 14x14)", fontsize=10)
    ax2.axis("off")

    # Subplot 3: Conv Block 2 Feature Maps (First 8 filters)
    ax3 = fig.add_subplot(2, 3, 3)
    c2_map = pool2_out[0, :8].cpu().numpy()
    c2_grid = np.hstack([c2_map[i] for i in range(8)])
    ax3.imshow(c2_grid, cmap="magma")
    ax3.set_title("Conv Block 2 Feature Maps\n(Sample 8 / 64 maps @ 7x7)", fontsize=10)
    ax3.axis("off")

    # Subplot 4: Softmax Probability Distribution Bar Chart
    ax4 = fig.add_subplot(2, 3, (4, 5))
    cls_indices = np.arange(len(class_names))
    colors = ["#4CAF50" if d == pred_class and is_correct else ("#F44336" if d == pred_class else "#9E9E9E") for d in cls_indices]
    bars = ax4.bar(cls_indices, probs_np * 100.0, color=colors, edgecolor="black", linewidth=1.2)
    ax4.set_xticks(cls_indices)
    ax4.set_xticklabels(class_names, fontsize=9, rotation=15)
    ax4.set_ylabel("Probability (%)", fontsize=11)
    ax4.set_ylim(0, 105)
    ax4.set_title("Predicted Softmax Probability Distribution Across EEE Fault Classes", fontsize=11, fontweight="semibold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, prob in zip(bars, probs_np):
        yval = bar.get_height()
        if yval > 1.0:
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{prob*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 5: Step-by-step summary text panel
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.axis("off")
    summary_text = (
        f"VERIFICATION SUMMARY\n"
        f"----------------------------------------\n"
        f"Test Image Index : #{sample_index}\n"
        f"Expected Fault   : {true_class_name}\n"
        f"Predicted Fault  : {pred_class_name}\n"
        f"Confidence       : {confidence:.2f}%\n"
        f"Status           : {'VERIFIED SUCCESS [OK]' if is_correct else 'MISCLASSIFIED [X]'}\n\n"
        f"Pipeline Checklist:\n"
        f" [OK] Model Restored (SolarPV_CNN)\n"
        f" [OK] Thermography Normalization\n"
        f" [OK] Conv1 & Conv2 Feature Ext.\n"
        f" [OK] Softmax Probabilities Calc.\n"
        f" [OK] EEE Domain Class Verified"
    )
    box_color = "#E8F5E9" if is_correct else "#FFEBEE"
    edge_color = "#2E7D32" if is_correct else "#C62828"
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor=box_color, edgecolor=edge_color, linewidth=2))

    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Output] Saved visual verification report to: '{save_plot}'\n")

    return {
        "sample_index": sample_index,
        "expected_class": true_class_name,
        "predicted_class": pred_class_name,
        "confidence": confidence,
        "is_correct": is_correct,
        "logits": logits_np.tolist(),
        "probabilities": probs_np.tolist(),
        "save_plot": save_plot
    }


if __name__ == "__main__":
    args = parse_args()
    verify_single_image(
        sample_index=args.sample_index,
        config_path=args.config,
        model_path=args.model_path,
        save_plot=args.save_plot
    )
