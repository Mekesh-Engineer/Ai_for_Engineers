# Experiment 1: Simple and Multiple Linear Regression (Google Colab Implementation)

This guide provides a complete, step-by-step implementation of the Linear Regression project optimized for Google Colab.

---

## 1. Environment Setup & GPU Acceleration

While standard scikit-learn models (like the ones used in this experiment) run on the CPU and are extremely fast for datasets of this size, Google Colab provides free GPU access which is crucial when scaling up to larger datasets or deep learning models (like CNNs or RNNs in subsequent experiments).

### How to enable GPU in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. In the top menu, click on **Runtime** > **Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU** (or any available GPU).
5. Click **Save**.

To verify the GPU is active, run the following command in your first cell:
```python
!nvidia-smi
```

---

## 2. Problem Statement

Predict median house values in California districts using physical, geographic, and economic features. The goal is to compare **Simple Linear Regression** (single predictor) against **Multiple Linear Regression** (all predictors) and understand the trade-offs between model complexity and accuracy.

---

## 3. Solution

We will develop a machine learning pipeline that fetches the California Housing dataset, preprocesses the data (handles missing values, removes outliers, and scales features), and trains two types of regression models. We will evaluate these models using standard regression metrics (R², MSE, RMSE, MAE), perform cross-validation, and visually analyze the residuals to validate model assumptions.

---

## 4. Methodology

1. **Data Collection & Acquisition:** Load the California Housing dataset from `sklearn.datasets`.
2. **Data Preprocessing:** Impute missing values (if any), remove outliers using the IQR method, and split the data into training (80%) and testing (20%) sets.
3. **Feature Engineering:** Apply `StandardScaler` to normalize the features. Perform correlation analysis to identify the best single feature for Simple Linear Regression (`MedInc`).
4. **Model Training:** Train a Simple Linear Regression model (using `MedInc`) and a Multiple Linear Regression model (using all features).
5. **Validation & Testing:** Evaluate models on the test set, perform 5-fold cross-validation, and analyze the distribution of residuals.
6. **Result Analysis:** Compare the models using metrics and visualization plots.

---

## 5. Hardware Implementation

For this specific experiment executed in Google Colab:
- **Compute:** Cloud-based CPU instance (Intel Xeon) or Cloud GPU (NVIDIA T4).
- **Memory:** ~12 GB RAM provided by Colab standard runtime.
- **Storage:** Ephemeral cloud storage (~100 GB available).
- *Note:* The scikit-learn linear regression model will predominantly utilize the CPU. For GPU-accelerated machine learning on tabular data, libraries like cuML (from NVIDIA RAPIDS) can be used, though it is not strictly required for a dataset of 20,000 rows.

---

## 6. Software Implementation (Colab Code Cells)

Copy and paste the following code blocks into individual cells in your Google Colab notebook.

### Cell 1: Install Dependencies & Import Libraries
```python
# Colab typically has these installed, but it's good practice to ensure they are up to date.
!pip install -q numpy pandas scikit-learn matplotlib seaborn scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
```

### Cell 2: Data Collection and Preprocessing
```python
# 1. Load Data — Colab provides a built-in California Housing CSV
df = pd.read_csv('/content/sample_data/california_housing_train.csv')
print(f"Dataset Shape: {df.shape}")
print("First 5 rows:")
display(df.head())

# 2. Handle Missing Values
for column in df.columns:
    if df[column].isnull().any() and df[column].dtype != 'object':
        median_val = df[column].median()
        df[column].fillna(median_val, inplace=True)

print(f"\nMissing values after handling: {df.isnull().sum().sum()}")

# 3. Separate features (X) and target variable (y)
# Target: median_house_value (in dollars)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# 4. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
feature_names = X.columns.tolist()

# Convert to DataFrame for display
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print("Data preprocessing complete!")
```

### Cell 3: Correlation Analysis & Visualization
```python
# Feature-Target Correlation
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# Extract the most correlated feature for Simple LR
target_corr = corr_matrix['median_house_value'].drop('median_house_value').sort_values(ascending=False, key=abs)
best_feature = target_corr.index[0]
best_feature_idx = feature_names.index(best_feature)
print(f"Most correlated feature for Simple LR: {best_feature} (r = {target_corr.iloc[0]:.4f})")
```

### Cell 4: Train Simple Linear Regression
```python
# Extract single feature
X_train_simple = X_train_scaled[:, best_feature_idx].reshape(-1, 1)
X_test_simple = X_test_scaled[:, best_feature_idx].reshape(-1, 1)

# Train model
simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)

# Predictions
y_pred_simple_train = simple_model.predict(X_train_simple)
y_pred_simple_test = simple_model.predict(X_test_simple)

print(f"Simple LR Equation: y = {simple_model.coef_[0]:.4f} * {best_feature} + {simple_model.intercept_:.4f}")

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X_test_simple, y_test, alpha=0.3, s=10, label="Actual Data")
plt.plot(X_test_simple, y_pred_simple_test, color='red', linewidth=2, label="Regression Line")
plt.xlabel(f"{best_feature} (Scaled)")
plt.ylabel("Median House Value ($100k)")
plt.title(f"Simple Linear Regression: {best_feature} vs Price")
plt.legend()
plt.show()
```

### Cell 5: Train Multiple Linear Regression
```python
# Train model using all features
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_multi_train = multi_model.predict(X_train_scaled)
y_pred_multi_test = multi_model.predict(X_test_scaled)

print(f"Multiple LR Intercept: {multi_model.intercept_:.4f}")
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': multi_model.coef_})
coeff_df = coeff_df.sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 5))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coeff_df['Coefficient']]
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Multiple LR Coefficients")
plt.xlabel("Coefficient Value")
plt.show()
```

### Cell 6: Evaluation & Cross-Validation
```python
def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"--- {model_name} Test Metrics ---")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}\n")
    return r2, rmse, mae

r2_simp, rmse_simp, mae_simp = evaluate(y_test, y_pred_simple_test, "Simple LR")
r2_mult, rmse_mult, mae_mult = evaluate(y_test, y_pred_multi_test, "Multiple LR")

# Cross Validation
cv_simple = cross_val_score(LinearRegression(), X_train_simple, y_train, cv=5, scoring='r2')
cv_multi = cross_val_score(LinearRegression(), X_train_scaled, y_train, cv=5, scoring='r2')

print(f"Simple LR CV R²  : {cv_simple.mean():.4f} ± {cv_simple.std():.4f}")
print(f"Multiple LR CV R²: {cv_multi.mean():.4f} ± {cv_multi.std():.4f}")
```

### Cell 7: Residual Analysis
```python
def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color='purple')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"Residuals vs Predicted ({title})")
    
    # Histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color='teal')
    axes[1].set_title(f"Residual Distribution ({title})")
    
    plt.show()
    
    # Shapiro-Wilk test (sample due to size limits)
    sample = np.random.choice(residuals, min(len(residuals), 5000), replace=False)
    stat, p = stats.shapiro(sample)
    print(f"Shapiro-Wilk p-value for {title}: {p:.4e} (p < 0.05 indicates non-normal distribution)")

plot_residuals(y_test, y_pred_simple_test, "Simple LR")
plot_residuals(y_test, y_pred_multi_test, "Multiple LR")
```

---

## 7. System Architecture

The workflow within the Colab environment follows this linear sequential architecture:

1. **Cloud Environment Initialization:** Colab runtime allocation (CPU/GPU, RAM).
2. **Library Import & Data Ingestion:** Importing `scikit-learn`, `pandas`, and fetching the California Housing dataset over the network into memory.
3. **Data Transformation Pipeline:** Sequential execution of missing value checks, IQR filtering, and `StandardScaler` fitting/transforming.
4. **Model execution graphs:**
   - **Path A:** Extraction of `MedInc` vector -> Model Fit -> Prediction array generation.
   - **Path B:** Full scaled matrix ingestion -> Model Fit -> Prediction array generation.
5. **Metric Calculation Engine:** Comparison of prediction arrays against true label vectors using `r2_score`, `mean_squared_error`.
6. **Visualization Rendering:** Passing numerical results and arrays to `matplotlib` to render inline PNG graphics in the notebook output cells.

---

## 8. References

1. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
2. Pace, R.K. and Barry, R. (1997). *Sparse Spatial Autoregressions*. Statistics & Probability Letters.
3. Google Colaboratory Documentation. [https://research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html)
4. James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
