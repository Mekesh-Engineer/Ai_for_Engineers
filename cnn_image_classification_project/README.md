# CNN Image Classification Project — Experiment 4

## 📋 Overview
A complete, modular Convolutional Neural Network (CNN) image classification pipeline applied to the **MNIST Handwritten Digits dataset** (70,000 grayscale 28×28 images, 10 classes: digits 0–9). The project builds a 2-block ConvNet with BatchNorm, MaxPool2D, Dropout, and Dense layers, incorporates on-the-fly data augmentation, trains with early stopping, evaluates test accuracy, and produces 5 publication-quality visualisations.

---

## 🗂️ Project Structure
```
cnn_image_classification_project/
├── config/
│   └── hyperparameters.json       # Central configuration file
├── data/
│   ├── raw/mnist/                 # Raw MNIST dataset (auto-downloaded)
│   └── processed/                 # Scaled tensor files
├── models/
│   └── cnn_model.pt               # Saved trained PyTorch state_dict
├── notebooks/
│   └── cnn_training.ipynb        # Interactive Jupyter notebook
├── results/
│   ├── training_history.png       # Loss & Accuracy curves over epochs
│   ├── confusion_matrix.png       # 10x10 Heatmap of actual vs predicted
│   ├── sample_predictions.png     # Grid of test samples with confidence
│   ├── misclassified_samples.png  # Failure case analysis
│   ├── learned_filters.png        # Visualization of 32 Conv1 3x3 filters
│   ├── accuracy_metrics.csv       # Per-class precision, recall, F1-score
│   └── classification_report.txt  # Comprehensive text summary report
├── src/
│   ├── data_loader.py             # DataLoaderModule
│   ├── data_preprocessing.py      # PreprocessingModule
│   ├── data_augmentation.py       # DataAugmentationModule
│   ├── model_builder.py           # ModelBuilderModule (PyTorch CNN)
│   ├── training.py                # TrainingModule (Train loop & early stopping)
│   ├── evaluation.py              # EvaluationModule (Test set evaluation)
│   └── visualization.py           # VisualizationModule (Plots & filters)
├── main.py                        # End-to-end pipeline runner
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 Quick Start

### Run the full end-to-end pipeline
```bash
cd cnn_image_classification_project
py main.py
```

### Override epochs or batch size
```bash
py main.py --epochs 15 --batch_size 64
```

### Run the Jupyter notebook
```bash
cd notebooks
jupyter notebook cnn_training.ipynb
```

---

## 🔬 Methodology & Workflow

| Step | Module | Description |
|------|--------|-------------|
| 1. Configuration | `config/` | Define batch size, learning rate, dropout, optimizer, and paths |
| 2. Data Acquisition | `data_loader.py` | Load 70,000 MNIST images (60,000 train/val, 10,000 test) |
| 3. Preprocessing | `data_preprocessing.py` | Normalize pixel values [0, 255] → [0, 1] (mean=0.1307, std=0.3081) |
| 4. Data Augmentation | `data_augmentation.py` | On-the-fly random rotation (±10°), translation, padding crop |
| 5. CNN Architecture | `model_builder.py` | Conv2D(32) → BatchNorm → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dropout(0.5) → Dense(10) |
| 6. Training & Validation | `training.py` | Adam optimizer, CrossEntropyLoss, early stopping monitoring |
| 7. Evaluation | `evaluation.py` | Test accuracy, per-class metrics, confusion matrix |
| 8. Visualisation | `visualization.py` | 5 visual plots (curves, heatmap, predictions, failures, filters) |

---

## 🏗️ Model Architecture

```
Input Image (1 × 28 × 28)
    ↓
[Conv2D (32 filters, 3×3, pad=1) → BatchNorm2d → ReLU]
    ↓
[MaxPool2d (2×2, stride=2)]             → Output: 32 × 14 × 14
    ↓
[Conv2D (64 filters, 3×3, pad=1) → BatchNorm2d → ReLU]
    ↓
[MaxPool2d (2×2, stride=2)]             → Output: 64 × 7 × 7
    ↓
[Flatten]                               → Output: 3136 features
    ↓
[Linear (3136 → 128) → ReLU]
    ↓
[Dropout (p=0.5)]
    ↓
[Linear (128 → 10)]                     → Logits (10 classes)
```

---

## 📊 Expected Performance (MNIST)

| Metric | Expected Target |
|--------|-----------------|
| Test Accuracy | **~98.5% – 99.2%** |
| Macro F1-Score | **~0.985+** |
| Epochs to Converge | **10 – 20 epochs** |
| Test Misclassifications | **< 180 / 10,000 samples** |

---

## 🖼️ Visualisations Produced

1. **Training History Plot**: Loss and accuracy curves over epochs for training and validation sets.
2. **Confusion Matrix Heatmap**: 10×10 confusion matrix revealing digit-to-digit classification performance.
3. **Sample Predictions Grid**: Grid of test images displayed with predicted digits and confidence probabilities.
4. **Misclassified Samples Analysis**: Visual analysis of failure cases highlighting ambiguous digit writing.
5. **Learned Conv Filters**: Visual grid displaying the 32 learned 3×3 spatial filters from the first convolutional layer.
