# Experiment 1: Simple and Multiple Linear Regression

## Housing Price Prediction — California Housing Dataset

---

## 1. Problem Statement

Predict median house values in California districts using physical, geographic, and economic features. The goal is to compare **Simple Linear Regression** (single predictor) against **Multiple Linear Regression** (all predictors) and understand the trade-offs between model complexity and accuracy.

## 2. Objective

- Develop and evaluate simple and multiple linear regression models
- Compare univariate vs. multivariate prediction performance
- Understand the impact of feature scaling on model quality
- Perform residual analysis to validate model assumptions

## 3. Dataset

| Attribute | Description |
|-----------|-------------|
| **Name** | California Housing Dataset |
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Samples** | 20,640 districts |
| **Features** | 8 numerical attributes |
| **Target** | `MedHouseVal` — median house value (in $100,000) |

### Feature Dictionary

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age in the block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

## 4. Project Structure

```
linear_regression_project/
├── config/
│   └── parameters.json          ← Hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── housing_data.csv     ← Original dataset
│   └── processed/
│       └── cleaned_housing_data.csv  ← After outlier removal
├── models/
│   ├── simple_lr_model.pkl      ← Trained Simple LR
│   ├── multiple_lr_model.pkl    ← Trained Multiple LR
│   └── scaler.pkl               ← Feature scaler
├── results/
│   ├── performance_metrics.txt  ← Full text report
│   ├── predictions.csv          ← Actual vs predicted
│   ├── model_comparison.csv     ← Side-by-side metrics
│   └── plots/
│       ├── feature_correlation.png
│       ├── feature_distributions.png
│       ├── simple_regression_line.png
│       ├── actual_vs_predicted_simple.png
│       ├── actual_vs_predicted_multiple.png
│       ├── residual_plot_simple.png
│       ├── residual_plot_multiple.png
│       ├── qq_plot_simple.png
│       ├── qq_plot_multiple.png
│       ├── cross_validation_scores.png
│       └── coefficient_importance.png
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    ← Load, clean, scale, split
│   ├── model_training.py        ← Train Simple & Multiple LR
│   ├── evaluation.py            ← Metrics, CV, residuals, report
│   └── visualization.py         ← All plots
├── notebooks/
│   └── (optional Jupyter notebook)
├── main.py                      ← Run full pipeline
├── requirements.txt
└── README.md                    ← This file
```

## 5. Methodology

### 5.1 Data Collection & Acquisition
- Dataset loaded programmatically from scikit-learn
- Raw CSV persisted to `data/raw/` for traceability
- Data integrity verified: no missing values, all numeric

### 5.2 Data Preprocessing
1. **Missing values**: Median imputation (configurable)
2. **Outlier removal**: IQR method with 1.5× multiplier
3. **Feature-target separation**: 8 features → `X`, target → `y`
4. **Train-test split**: 80/20 (stratified by random seed 42)

### 5.3 Feature Engineering
- **Scaling**: `StandardScaler` (zero-mean, unit-variance)
- **Correlation analysis**: Pearson matrix, threshold ≥ 0.90
- **Simple LR feature**: `MedInc` (highest target correlation)

### 5.4 Model Training
| Model | Equation | Features |
|-------|----------|----------|
| Simple LR | `y = m·MedInc + b` | 1 |
| Multiple LR | `y = b₀ + Σ bᵢ·xᵢ` | 8 |

### 5.5 Validation & Testing
- 5-fold cross-validation (R² scoring)
- Residual analysis: mean, std, skewness, kurtosis
- Shapiro-Wilk normality test on residuals

### 5.6 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² Score | `1 - SS_res / SS_tot` | Variance explained (higher ↑ better) |
| MSE | `mean((y - ŷ)²)` | Average squared error (lower ↓ better) |
| RMSE | `√MSE` | Error in target units (lower ↓ better) |
| MAE | `mean(|y - ŷ|)` | Average absolute error (lower ↓ better) |

## 6. System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                        main.py                            │
│         (Pipeline Orchestrator — 10 Steps)                │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   config/parameters.json                                  │
│          ↓                                                │
│   src/data_preprocessing.py                               │
│   ├─ load_housing_data()                                  │
│   ├─ explore_data()                                       │
│   ├─ handle_missing_values()                              │
│   ├─ remove_outliers()                                    │
│   ├─ analyze_correlations()                               │
│   ├─ split_data()                                         │
│   └─ scale_features()                                     │
│          ↓                                                │
│   src/model_training.py                                   │
│   ├─ train_simple_regression()   → y = mx + b             │
│   └─ train_multiple_regression() → y = b₀ + Σbᵢxᵢ        │
│          ↓                                                │
│   src/evaluation.py                                       │
│   ├─ evaluate_model()         (R², MSE, RMSE, MAE)        │
│   ├─ cross_validate_model()   (5-fold CV)                 │
│   ├─ residual_analysis()      (normality, skewness)       │
│   ├─ compare_models()         (side-by-side table)        │
│   └─ generate_report()        (full text report)          │
│          ↓                                                │
│   src/visualization.py                                    │
│   └─ generate_all_plots()     (11 publication plots)      │
│          ↓                                                │
│   Output: models/*.pkl, results/*, results/plots/*        │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## 7. Software Implementation

### Prerequisites

- Python 3.8+
- pip (package manager)

### Installation

```bash
cd linear_regression_project
pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py
```

This single command executes the complete pipeline and generates all outputs.

### Configuration

Edit `config/parameters.json` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_size` | 0.2 | Train/test split ratio |
| `scaling_method` | StandardScaler | Feature normalization |
| `cross_validation_folds` | 5 | Number of CV folds |
| `random_seed` | 42 | Reproducibility seed |
| `outlier_method` | IQR | Outlier detection method |
| `simple_regression_feature` | MedInc | Feature for Simple LR |

### Interactive Prediction

Use the interactive prediction script to test the trained model with custom input values:

```bash
python predict_house_value.py
```

The script loads the pre-trained `multiple_linear_regression_model.joblib` model, accepts 8 housing feature values interactively (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income), scales them, and outputs the predicted median house value in dollars.

### Google Colab

For running the full experiment in Google Colab, refer to `collab.md` which provides step-by-step code cells aligned with Colab's built-in California Housing CSV dataset.

## 8. Expected Outcomes

### Simple Linear Regression
- R² ≈ 0.45–0.55 (single feature captures partial variance)
- Clear positive linear trend between MedInc and price
- Residuals show some heteroscedasticity at high values

### Multiple Linear Regression
- R² ≈ 0.60–0.70 (all features capture more variance)
- MedInc, Latitude, Longitude are top-3 predictors
- Lower MSE/RMSE than Simple LR

### Key Observations
- Multiple features capture significantly more variance
- Feature scaling is essential for coefficient interpretability
- Outlier removal improves model stability
- Residual analysis reveals non-linear patterns (limitation of linear models)

## 9. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any modern processor | Multi-core for CV |
| RAM | 2 GB | 4 GB |
| Storage | 100 MB | 500 MB (with plots) |
| GPU | Not required | Not required |

## 10. Reproducibility

- Random seed: **42** (set in config)
- All parameters stored in `config/parameters.json`
- Models serialized with timestamps via `joblib`
- Environment: capture with `pip freeze > environment.txt`

## 11. References

1. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
2. Pace, R.K. and Barry, R. (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters.
3. James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
4. Montgomery, D.C., Peck, E.A., and Vining, G.G. (2012). *Introduction to Linear Regression Analysis*. Wiley.

---

**Course**: AI for Engineers (22GEX03) | **Experiment 1** | **Semester 7**
