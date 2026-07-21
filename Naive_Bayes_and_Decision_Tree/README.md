# Naïve Bayes & Decision Tree Classification (Fine-Tuned 365-Day Model)

> **Experiment 2** — Predict whether tennis will be played (`PlayTennis`) based on weather and seasonal features across all 365 days of the year using fine-tuned **Gaussian Naïve Bayes** and **Decision Tree** classifiers.

---

## Project Structure

```
Naive_Bayes_and_Decision_Tree/
├── data/
│   ├── raw/
│   │   └── play_tennis.csv          ← Full 365-day dataset (Jan 1 - Dec 31)
│   └── processed/
│       └── encoded_data.csv         ← Encoded dataset
├── notebooks/
│   └── classification_analysis.ipynb  ← 16-section Jupyter Notebook
├── src/
│   ├── data_loader.py               ← Load, check data quality
│   ├── feature_encoding.py          ← LabelEncoder & train/test split
│   ├── model_training.py            ← GaussianNB, DecisionTree, GridSearchCV fine-tuning
│   ├── evaluation.py                ← Metrics, CV, ROC, comparison reports
│   └── visualization.py             ← All plot functions
├── models/
│   ├── naive_bayes_model.pkl        ← Fine-tuned Naïve Bayes model
│   └── decision_tree_model.pkl      ← Fine-tuned Decision Tree model
├── results/                         ← Output figures & text reports
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── decision_tree_visualization.png
│   ├── feature_importance.png
│   ├── depth_sweep.png
│   ├── cv_comparison.png
│   ├── performance_comparison.csv
│   ├── classification_report.txt
│   └── decision_tree_rules.txt
├── config/
│   └── parameters.json              ← Hyperparameters & fine-tuning grid config
├── run_experiment.py                ← Standalone execution script
└── README.md
```

---

## Dataset Attributes

| Column | Description | Role |
|--------|-------------|------|
| `Day` | Day identifier (`D1` ... `D365`) | Identifier (Dropped) |
| `Date` | Calendar date (`01-Jan` ... `31-Dec`) | Identifier (Dropped) |
| `Month` | Calendar month (`January` ... `December`) | **Feature** |
| `Outlook` | Weather condition (`Sunny`, `Rain`, `Overcast`) | **Feature** |
| `Temperature` | Temperature level (`Hot`, `Mild`, `Cool`) | **Feature** |
| `Humidity` | Humidity level (`High`, `Normal`) | **Feature** |
| `Wind` | Wind condition (`Strong`, `Weak`) | **Feature** |
| `PlayTennis` | Target binary outcome (`Yes`, `No`) | **Target** |

- **Total Records**: 365 days
- **Train / Test split**: 70% (255 samples) / 30% (110 samples)

---

## Fine-Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | ROC-AUC | 5-Fold CV Mean |
|-------|----------|-----------|--------|----------|-------------|---------|----------------|
| **Gaussian Naïve Bayes (Tuned)** | 94.55% | 0.9499 | 0.9455 | 0.9439 | 0.8333 | 0.9450 | 94.79% ± 0.55% |
| **Decision Tree (Tuned)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **100.00% ± 0.00%** |

---

## Quick Start

### 1. Install Dependencies

```powershell
$pyexe = "C:\Users\mekes\AppData\Local\Python\pythoncore-3.14-64\python.exe"
& $pyexe -m pip install -r requirements.txt
```

### 2. Run via Jupyter Notebook
```powershell
& $pyexe -m jupyter notebook notebooks/classification_analysis.ipynb
```
