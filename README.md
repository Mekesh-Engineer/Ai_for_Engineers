# AI for Engineers: Experiments - Methodology & Implementation Plan

**Course Code**: 22GEX03 | **Semester**: 7 | **Credit**: 3  
**Branch**: All branches except CSE, IT, AIDS & AIML  

---

## Table of Contents

1. [Experiment 1: Simple and Multiple Linear Regression](#experiment-1-simple-and-multiple-linear-regression)
2. [Experiment 2: Naïve Bayes and Decision Tree Classification](#experiment-2-naïve-bayes-and-decision-tree-classification)
3. [Experiment 3: K-Means Clustering](#experiment-3-k-means-clustering)
4. [Experiment 4: Image Classification using CNN](#experiment-4-image-classification-using-cnn)
5. [Experiment 5: Sequential Data Modeling using RNN](#experiment-5-sequential-data-modeling-using-rnn)
6. [Experiment 6: Object Detection using YOLO](#experiment-6-object-detection-using-yolo)
7. [Experiment 7: Text Summarization using LLM](#experiment-7-text-summarization-using-llm)
8. [Experiment 8: Grammar Correction and Text Rewriting using LLM](#experiment-8-grammar-correction-and-text-rewriting-using-llm)
9. [Experiment 9: Advanced Domain-Specific EEE Question Answering Chatbot Using LLM, RAG and Ollama/API](#experiment-9-advanced-domain-specific-eee-question-answering-chatbot-using-llm-rag-and-ollamaapi)

---

## Experiment 1: Simple and Multiple Linear Regression

### Objective

Develop and evaluate simple and multiple linear regression models to predict continuous numerical values based on one or more input features. Compare model performance between univariate and multivariate approaches and understand the impact of feature scaling and regularization.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Boston Housing Dataset or California Housing Dataset |
| **Source/URL** | Scikit-learn datasets or Kaggle (kaggle.com/datasets) |
| **Brief Description** | Real estate pricing dataset containing house features and their corresponding prices in a specific geographic region |
| **Important Features/Attributes** | Crime rate, number of rooms, age of house, distance to employment centers, property tax rate, student-teacher ratio, median house price (target) |

### Methodology

1. **Data Collection and Acquisition**
   - Load the dataset from Scikit-learn or download from a repository
   - Verify data integrity and completeness
   - Check for missing values and handle them appropriately
   - Document data source and dictionary

2. **Data Preprocessing**
   - Handle missing values using imputation techniques (mean, median, or forward-fill)
   - Remove or treat outliers using statistical methods (IQR or Z-score)
   - Separate features (X) and target variable (y)
   - Split data into training (70-80%) and testing (20-30%) sets

3. **Feature Engineering**
   - Perform feature scaling using StandardScaler or MinMaxScaler to normalize all features to a common range
   - For multiple linear regression, identify relevant features using correlation analysis
   - Consider polynomial features for simple regression if relationship is non-linear
   - Remove highly correlated features to reduce multicollinearity

4. **Model Selection**
   - **Simple Linear Regression**: For univariate prediction with single feature
   - **Multiple Linear Regression**: For multivariate prediction with multiple features
   - Evaluate both approaches and document selection rationale

5. **Model Training**
   - Fit the selected regression model on training data
   - Generate predictions on training set
   - Document model coefficients and intercept values
   - Store trained model for inference

6. **Validation and Testing**
   - Generate predictions on test set
   - Perform cross-validation (5-fold or 10-fold) to assess model stability
   - Analyze prediction residuals for patterns and heteroscedasticity
   - Create residual plots for visual inspection

7. **Result Analysis**
   - Compare training and testing performance metrics
   - Investigate cases with high prediction errors
   - Analyze coefficient values to understand feature impact
   - Document limitations and areas for improvement

### Model/Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    LINEAR REGRESSION WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input Data                                                       │
│      ↓                                                            │
│  Data Cleaning & Preprocessing                                   │
│      ↓                                                            │
│  Feature Scaling (Normalization/Standardization)                │
│      ↓                                                            │
│  Train-Test Split (70-30 or 80-20)                             │
│      ↓                                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Model Training Path                                      │   │
│  │  ├─ Simple Linear Regression: y = mx + b                │   │
│  │  └─ Multiple Linear Regression: y = b₀ + b₁x₁ + ... + bₙxₙ │
│  └─────────────────────────────────────────────────────────┘   │
│      ↓                                                            │
│  Model Evaluation (R² Score, MSE, RMSE, MAE)                    │
│      ↓                                                            │
│  Cross-Validation (5-fold or 10-fold)                           │
│      ↓                                                            │
│  Predictions & Residual Analysis                                │
│      ↓                                                            │
│  Performance Comparison & Reporting                              │
│      ↓                                                            │
│  Output: Model Coefficients, Predictions, Performance Metrics   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Scikit-learn**: Model development (LinearRegression)
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization of results and residual plots
- **Seaborn**: Statistical data visualization
- **Scipy**: Statistical tests and analysis

### Sample Project Structure

```
linear_regression_project/
├── data/
│   ├── raw/
│   │   └── housing_data.csv
│   └── processed/
│       └── cleaned_housing_data.csv
├── notebooks/
│   └── linear_regression_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── simple_lr_model.pkl
│   └── multiple_lr_model.pkl
├── results/
│   ├── performance_metrics.txt
│   ├── predictions.csv
│   └── plots/
│       ├── residual_plot.png
│       ├── actual_vs_predicted.png
│       └── feature_correlation.png
├── config/
│   └── parameters.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoader Module**: Load and validate dataset from source
2. **PreprocessingModule**: Handle missing values, outliers, and scaling
3. **ModelTrainerModule**: Train simple and multiple regression models
4. **EvaluationModule**: Calculate performance metrics and validation scores
5. **VisualizationModule**: Generate plots for residuals and predictions
6. **ConfigurationModule**: Manage hyperparameters and settings

**Important Functions/Classes (Conceptual):**

- `load_housing_data()`: Load and return dataset
- `preprocess_data()`: Clean and prepare data
- `scale_features()`: Normalize feature values
- `train_simple_regression()`: Train univariate model
- `train_multiple_regression()`: Train multivariate model
- `evaluate_model()`: Calculate metrics (R², MSE, RMSE, MAE)
- `cross_validate_model()`: Perform k-fold cross-validation
- `plot_residuals()`: Visualize prediction errors
- `generate_report()`: Create comprehensive summary document

**Configuration Requirements:**

- Test-train split ratio (default: 0.8-0.2)
- Feature scaling method (StandardScaler or MinMaxScaler)
- Cross-validation folds (default: 5)
- Random seed for reproducibility

**Reproducibility Considerations:**

- Set random seeds for data splitting and cross-validation
- Document all parameter values in configuration file
- Save trained models with timestamp
- Record Python version and library versions used

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | CSV file with feature columns and target variable (price); minimum 200-300 records |
| **Processing** | Numerical feature values; no categorical data for simple approach |
| **Output (Simple)** | Trained model, predicted values, R² score, MSE, RMSE, MAE, coefficients |
| **Output (Multiple)** | Trained model, predicted values, R² score, RMSE, feature importance, residual plots |

### Evaluation Methodology

**Performance Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1, higher is better) |
| **Mean Squared Error (MSE)** | (1/n) × Σ(y_true - y_pred)² | Average squared prediction error (lower is better) |
| **Root Mean Squared Error (RMSE)** | √MSE | Error in original units (lower is better) |
| **Mean Absolute Error (MAE)** | (1/n) × Σ\|y_true - y_pred\| | Average absolute error (lower is better) |

**Assessment Procedure:**

1. Compare R² scores for training and testing sets (identify overfitting)
2. Analyze residual distribution (should be normally distributed)
3. Check for heteroscedasticity using residual plots
4. Perform cross-validation and report average scores
5. Document feature importance in multiple regression

### Expected Outcomes

1. **Simple Linear Regression Results:**
   - R² score typically in range of 0.4-0.7 (depends on dataset)
   - Clear linear relationship visualization
   - Residual plot showing random distribution

2. **Multiple Linear Regression Results:**
   - Improved R² score (0.6-0.85) compared to simple regression
   - Identification of most influential features
   - Balanced training and testing performance (no overfitting)

3. **Observations:**
   - Feature scaling significantly improves model performance
   - Multiple features capture more variance than single feature
   - Outliers and missing values impact model accuracy
   - Some features may have negligible impact on predictions

4. **Deliverables:**
   - Trained models saved in serialized format
   - Performance report with metrics and visualizations
   - Residual analysis plots
   - Comparison document between simple and multiple regression

---

## Experiment 2: Naïve Bayes and Decision Tree Classification

### Objective

Develop and compare Naïve Bayes and Decision Tree classification models to predict categorical outcomes. Understand probabilistic classification and tree-based decision boundaries, evaluate performance using classification metrics, and handle both categorical and numerical features.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Play Tennis Dataset (historical UCI dataset) or Buys Computer Dataset |
| **Source/URL** | UCI Machine Learning Repository (archive.ics.uci.edu) or Kaggle |
| **Brief Description** | Categorical dataset with attributes describing weather conditions and whether tennis is played / computer is purchased |
| **Important Features/Attributes** | Outlook (Sunny/Rainy/Overcast), Temperature, Humidity, Wind (Strong/Weak), PlayTennis (Yes/No) or similar binary/categorical target |

### Methodology

1. **Data Collection and Acquisition**
   - Load dataset from UCI repository or Kaggle
   - Explore data dimensions, feature types, and class distribution
   - Check for missing values and document data characteristics
   - Verify balanced or imbalanced class distribution

2. **Data Preprocessing**
   - Handle missing values using mode (categorical) or deletion
   - Identify categorical and numerical features
   - For numerical features in Play Tennis dataset, apply discretization (binning) if needed
   - Check for duplicate records and remove if present

3. **Feature Engineering**
   - Encode categorical variables using LabelEncoder or OneHotEncoder
   - For Naïve Bayes, keep features in original form for probability calculation
   - For Decision Trees, consider feature interaction and importance
   - Remove or combine highly correlated categorical features
   - Create binary target variable encoding (0/1 or True/False)

4. **Model Selection**
   - **Naïve Bayes**: Assumes feature independence, suitable for categorical data
   - **Decision Tree**: Non-parametric approach, handles both categorical and numerical features
   - Select both models for comparison

5. **Model Training**
   - Fit Naïve Bayes classifier on training data
   - Fit Decision Tree classifier with appropriate depth constraints
   - Experiment with tree depth (max_depth parameter) to avoid overfitting
   - Store both trained models

6. **Validation and Testing**
   - Generate predictions on test set for both models
   - Perform cross-validation (5-fold) for stability assessment
   - Analyze per-class performance metrics
   - Generate confusion matrices for both models

7. **Result Analysis**
   - Compare accuracy, precision, recall, and F1-scores
   - Analyze misclassified instances
   - For Decision Trees, extract and visualize decision rules
   - Document strengths and weaknesses of each approach

### Model/Algorithm Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│               CLASSIFICATION WORKFLOW (NAÏVE BAYES & DT)          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input Data (Mixed Categorical/Numerical)                         │
│      ↓                                                             │
│  Data Preprocessing & Feature Encoding                            │
│      ↓                                                             │
│  Train-Test Split (70-30)                                         │
│      ↓                                                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         NAÏVE BAYES PATH                    DT PATH         │  │
│  │  ├─ Calculate P(Class)                 ├─ Select Best      │  │
│  │  ├─ Calculate P(Features|Class)        │   Split Feature   │  │
│  │  └─ Apply Bayes' Theorem               ├─ Recursively      │  │
│  │     for Prediction                     │   Split Nodes     │  │
│  │                                        └─ Prune Tree (opt) │  │
│  └────────────────────────────────────────────────────────────┘  │
│      ↓                                                             │
│  Generate Predictions (Binary: Yes/No or 0/1)                    │
│      ↓                                                             │
│  Model Evaluation & Comparison                                    │
│  ├─ Accuracy, Precision, Recall, F1-Score                        │
│  ├─ Confusion Matrix Analysis                                    │
│  └─ ROC-AUC Curve                                                │
│      ↓                                                             │
│  Output: Models, Predictions, Performance Metrics, Visualizations │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Scikit-learn**: Classification models (GaussianNB, DecisionTreeClassifier)
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization of decision trees and metrics
- **Seaborn**: Confusion matrix heatmaps and ROC curves
- **Graphviz**: Tree visualization (optional)

### Sample Project Structure

```
classification_project/
├── data/
│   ├── raw/
│   │   └── play_tennis.csv
│   └── processed/
│       └── encoded_data.csv
├── notebooks/
│   └── classification_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_encoding.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── naive_bayes_model.pkl
│   └── decision_tree_model.pkl
├── results/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── decision_tree_visualization.png
│   ├── performance_comparison.csv
│   └── classification_report.txt
├── config/
│   └── parameters.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load and explore categorical dataset
2. **EncodingModule**: Handle categorical feature encoding
3. **ClassifierTrainerModule**: Train both Naïve Bayes and Decision Tree
4. **EvaluationModule**: Calculate classification metrics
5. **VisualizationModule**: Plot confusion matrices, ROC curves, tree diagrams
6. **ComparisonModule**: Generate comparative analysis report

**Important Functions/Classes (Conceptual):**

- `load_classification_data()`: Load dataset
- `encode_categorical_features()`: Convert categorical to numerical
- `split_and_prepare_data()`: Separate features and target
- `train_naive_bayes()`: Train Naïve Bayes classifier
- `train_decision_tree()`: Train Decision Tree with depth constraints
- `generate_predictions()`: Predict on test set
- `evaluate_classification()`: Calculate metrics (accuracy, precision, recall, F1)
- `create_confusion_matrix()`: Generate confusion matrix visualization
- `extract_tree_rules()`: Extract and document decision rules
- `compare_models()`: Create comparison report

**Configuration Requirements:**

- Train-test split ratio (default: 0.7-0.3)
- Decision Tree max_depth (default: 3-5 to avoid overfitting)
- Cross-validation folds (default: 5)
- Feature encoding method (LabelEncoder or OneHotEncoder)

**Reproducibility Considerations:**

- Set random seeds for data splitting and tree initialization
- Document encoding mappings for categorical features
- Save all trained models with metadata
- Record cross-validation results

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | CSV with categorical/mixed features and binary/multi-class target; minimum 100-200 records |
| **Processing** | Categorical encoding, numerical scaling if needed, balanced class consideration |
| **Output (Naïve Bayes)** | Trained model, predicted labels, probability estimates, accuracy, precision, recall, F1-score |
| **Output (Decision Tree)** | Trained model, decision rules, feature importance, tree depth, pruning analysis, performance metrics |

### Evaluation Methodology

**Performance Metrics for Classification:**

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness (0-1, higher is better) |
| **Precision** | TP / (TP + FP) | Correctness of positive predictions |
| **Recall (Sensitivity)** | TP / (TP + FN) | Coverage of actual positives |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **Specificity** | TN / (TN + FP) | Coverage of actual negatives |
| **ROC-AUC** | Area under ROC curve | Model's ability to distinguish classes |

**Assessment Procedure:**

1. Generate confusion matrices for both models
2. Calculate and compare all metrics
3. Plot ROC curves for probabilistic evaluation
4. Perform cross-validation and report average scores
5. Analyze misclassification patterns
6. For Decision Tree, evaluate feature importance

### Expected Outcomes

1. **Naïve Bayes Results:**
   - Accuracy typically 70-85% (depends on dataset)
   - Fast training and prediction
   - Simple probabilistic interpretation
   - May perform well despite independence assumption violation

2. **Decision Tree Results:**
   - Accuracy 80-95% (typically higher than Naïve Bayes)
   - Interpretable decision rules
   - Risk of overfitting if tree is too deep
   - Feature importance clearly identified

3. **Observations:**
   - Feature encoding significantly impacts both models
   - Naïve Bayes assumes independence; violations visible in results
   - Decision Trees capture feature interactions naturally
   - Class imbalance affects both models (handle with class weights)

4. **Deliverables:**
   - Trained models in serialized format
   - Confusion matrices and performance comparison plots
   - Decision tree visualization with rules
   - Comprehensive classification report
   - Cross-validation analysis

---

## Experiment 3: K-Means Clustering

### Objective

Develop and evaluate K-Means clustering models to partition unlabeled data into meaningful clusters. Determine optimal cluster count, analyze cluster characteristics, and visualize clustering results for pattern recognition and data segmentation.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Iris Dataset or Customer Segmentation Dataset |
| **Source/URL** | Scikit-learn datasets or Kaggle (kaggle.com/datasets) |
| **Brief Description** | Iris dataset contains 150 samples of iris flowers with 4 features (sepal/petal length/width); or customer behavior dataset for segmentation |
| **Important Features/Attributes** | Sepal Length, Sepal Width, Petal Length, Petal Width (for Iris); or RFM (Recency, Frequency, Monetary) for customer data |

### Methodology

1. **Data Collection and Acquisition**
   - Load dataset from Scikit-learn or repository
   - Verify data completeness and structure
   - Confirm dataset contains only numerical features
   - Document feature ranges and distributions

2. **Data Preprocessing**
   - Handle missing values using mean/median imputation
   - Remove or treat outliers using statistical methods
   - Separate features (X) for clustering; no target variable used
   - Ensure all features are numerical

3. **Feature Engineering**
   - Standardize all features using StandardScaler (crucial for K-Means)
   - Apply MinMaxScaler alternative if feature ranges vary significantly
   - Reduce dimensionality using PCA if dataset has >10 features (optional)
   - Document scaling parameters for reproducibility

4. **Model Selection**
   - K-Means clustering algorithm
   - Determine optimal number of clusters (k) using:
     - Elbow Method: Plot inertia vs. number of clusters
     - Silhouette Score: Measure of cluster cohesion and separation
     - Gap Statistic: Compare clustering to random data

5. **Model Training**
   - Train K-Means with different k values (typically 2-10)
   - Initialize with k-means++ for better convergence
   - Set random seed for reproducibility
   - Record convergence and iteration counts
   - Extract cluster centers and labels

6. **Validation and Testing**
   - Calculate silhouette scores for each k
   - Perform multiple runs with different random seeds
   - Analyze inertia (within-cluster sum of squares) trend
   - Validate cluster stability across runs

7. **Result Analysis**
   - Identify optimal k from elbow and silhouette analysis
   - Analyze cluster sizes and distributions
   - Examine cluster centers in original feature space
   - Interpret cluster characteristics and patterns
   - Visualize 2D/3D projections of clusters

### Model/Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    K-MEANS CLUSTERING WORKFLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input Data (Unlabeled, Numerical Features)                          │
│      ↓                                                                │
│  Feature Scaling (StandardScaler)                                    │
│      ↓                                                                │
│  Determine Optimal k (Elbow Method, Silhouette, Gap Statistic)       │
│      ↓                                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │          K-MEANS ALGORITHM (for each k value)                 │  │
│  │  1. Randomly initialize k cluster centers (k-means++)         │  │
│  │  2. Assign each point to nearest center (Euclidean distance)  │  │
│  │  3. Recalculate centers as mean of assigned points           │  │
│  │  4. Repeat steps 2-3 until convergence or max iterations      │  │
│  │  5. Compute inertia (sum of squared distances)                │  │
│  │  6. Calculate silhouette score for quality assessment         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│      ↓                                                                │
│  Analyze Clustering Metrics (Inertia, Silhouette, Gap Statistic)    │
│      ↓                                                                │
│  Select Optimal k & Train Final Model                               │
│      ↓                                                                │
│  Visualize Clusters (2D/3D Projections using PCA/t-SNE)            │
│      ↓                                                                │
│  Output: Cluster Labels, Centers, Metrics, Visualizations           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Scikit-learn**: KMeans, metrics (silhouette_score, gap_statistic)
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations and distance calculations
- **Matplotlib**: Cluster visualization and elbow plots
- **Seaborn**: Enhanced visualizations
- **Scikit-learn decomposition**: PCA for dimensionality reduction
- **Scipy**: Clustering distance metrics

### Sample Project Structure

```
kmeans_clustering_project/
├── data/
│   ├── raw/
│   │   └── iris.csv
│   └── processed/
│       └── scaled_iris.csv
├── notebooks/
│   └── kmeans_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── optimal_k_finder.py
│   ├── clustering_model.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   └── kmeans_model.pkl
├── results/
│   ├── elbow_plot.png
│   ├── silhouette_plot.png
│   ├── cluster_visualization_2d.png
│   ├── cluster_visualization_3d.png
│   ├── cluster_centers.csv
│   ├── cluster_labels.csv
│   └── clustering_report.txt
├── config/
│   └── parameters.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load and explore dataset
2. **PreprocessingModule**: Scale and prepare features
3. **OptimalKFinderModule**: Implement elbow method, silhouette analysis, gap statistic
4. **ClusteringModule**: Train K-Means with optimal k
5. **EvaluationModule**: Calculate clustering quality metrics
6. **VisualizationModule**: Plot clusters, elbow curves, silhouette scores
7. **AnalysisModule**: Interpret cluster characteristics

**Important Functions/Classes (Conceptual):**

- `load_iris_data()`: Load dataset
- `scale_features()`: Apply StandardScaler to all features
- `compute_elbow_curve()`: Calculate inertia for different k values
- `compute_silhouette_scores()`: Calculate silhouette score for each k
- `compute_gap_statistic()`: Implement gap statistic method
- `train_kmeans()`: Train K-Means with specified k
- `get_cluster_labels()`: Extract cluster assignments
- `get_cluster_centers()`: Extract and denormalize cluster centers
- `evaluate_clustering()`: Calculate silhouette, Davies-Bouldin index
- `visualize_clusters_2d()`: Plot using PCA projection
- `visualize_clusters_3d()`: Plot using t-SNE or PCA 3D
- `analyze_cluster_characteristics()`: Describe each cluster

**Configuration Requirements:**

- Feature scaling method (StandardScaler)
- Range of k values to test (default: 2-10)
- Maximum iterations for K-Means (default: 300)
- Convergence tolerance (default: 1e-4)
- k-means++ initialization for better results

**Reproducibility Considerations:**

- Set random seed before K-Means training
- Document optimal k selection criteria
- Save scaling parameters for new data
- Record multiple runs to assess stability

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | CSV with numerical features only; no target variable required; Iris: 150 samples × 4 features |
| **Processing** | Feature scaling to zero mean and unit variance; k values range from 2-10 typically |
| **Output** | Cluster labels for each sample, cluster centers, inertia values, silhouette scores, optimal k value |

### Evaluation Methodology

**Clustering Quality Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **Inertia (Within-Cluster SS)** | Measures cluster tightness | Lower is better; use for elbow method |
| **Silhouette Score** | Measures cluster cohesion and separation | Range -1 to 1; closer to 1 is better |
| **Davies-Bouldin Index** | Ratio of within to between cluster distances | Lower is better |
| **Calinski-Harabasz Index** | Ratio of between to within cluster variance | Higher is better |

**Optimal k Selection Procedure:**

1. Compute inertia for k = 2 to 10
2. Plot elbow curve and identify inflection point
3. Calculate silhouette scores for same k range
4. Select k with highest average silhouette score
5. Verify choice using gap statistic
6. Document selection rationale

### Expected Outcomes

1. **K-Means Results for Iris Dataset:**
   - Optimal k = 3 (known from dataset structure)
   - Silhouette score approximately 0.5-0.6
   - Clear elbow point around k = 3
   - Clusters correspond to iris species (validation possible)

2. **Clustering Characteristics:**
   - Tight, spherical clusters formed by K-Means
   - Cluster centers converge after 10-20 iterations
   - Consistency across multiple runs with same seed

3. **Observations:**
   - Feature scaling critical for distance calculations
   - Optimal k identified reliably from elbow plot
   - Silhouette scores validate cluster quality
   - K-Means assumes spherical, equal-size clusters

4. **Deliverables:**
   - Trained K-Means model with optimal k
   - Elbow plot and silhouette analysis graphs
   - 2D and 3D cluster visualizations
   - Cluster assignment CSV file
   - Detailed clustering analysis report
   - Cluster center coordinates in original feature space

---

## Experiment 4: Image Classification using CNN

### Objective

Develop and train a Convolutional Neural Network (CNN) to classify images into predefined categories. Build and optimize the network architecture, implement data augmentation, achieve high accuracy on test data, and analyze model performance.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | MNIST Handwritten Digits Dataset |
| **Source/URL** | Keras/TensorFlow datasets or torchvision |
| **Brief Description** | 70,000 grayscale images (28×28 pixels) of handwritten digits (0-9); 60,000 training, 10,000 testing |
| **Important Features/Attributes** | Pixel intensity values (0-255); 10 classes (digits 0-9); balanced class distribution |

### Methodology

1. **Data Collection and Acquisition**
   - Load MNIST dataset from TensorFlow/Keras or torchvision
   - Verify dataset dimensions (28×28×1 for MNIST)
   - Check class distribution and balance
   - Document data source and sampling methodology

2. **Data Preprocessing**
   - Normalize pixel values to range [0, 1] by dividing by 255
   - Reshape data to match model input requirements (height, width, channels)
   - Ensure data type consistency (float32)
   - Separate training, validation, and testing sets (70-10-20 or 60-20-20)

3. **Data Augmentation** (Optional but Recommended)
   - Apply random rotations (±10 degrees)
   - Apply random shifts (horizontal/vertical)
   - Apply zoom and shear transformations
   - Perform on-the-fly augmentation during training
   - Avoid augmentation on validation/test sets

4. **Model Selection and Architecture Design**
   - Build sequential CNN architecture:
     - Input layer matching image dimensions
     - Convolutional layers with ReLU activation
     - Max pooling layers for downsampling
     - Dropout layers for regularization
     - Flatten layer for transition to dense layers
     - Dense layers with ReLU activation
     - Output softmax layer for 10-class classification
   - Consider model complexity (shallow vs. deep)

5. **Model Training**
   - Compile model with appropriate loss function (categorical cross-entropy)
   - Use optimization algorithm (Adam, SGD with momentum)
   - Set batch size (typically 32 or 64)
   - Train for multiple epochs (30-100 typically)
   - Implement early stopping to prevent overfitting
   - Monitor training and validation loss/accuracy

6. **Validation and Testing**
   - Evaluate model on validation set during training
   - Generate predictions on held-out test set
   - Calculate classification accuracy and other metrics
   - Analyze per-class performance metrics
   - Generate confusion matrix

7. **Result Analysis**
   - Compare training and validation accuracy (detect overfitting)
   - Plot training curves (loss and accuracy vs. epochs)
   - Analyze misclassified samples
   - Visualize learned filters in early convolutional layers
   - Document model strengths and failure cases

### Model/Algorithm Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│                  CNN IMAGE CLASSIFICATION WORKFLOW                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Images (28×28×1 MNIST digits)                               │
│      ↓                                                              │
│  Normalization (Divide by 255)                                     │
│      ↓                                                              │
│  Train-Val-Test Split (60-20-20 or 70-10-20)                      │
│      ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  CNN ARCHITECTURE                                             │ │
│  │  ├─ Conv2D(32 filters, 3×3) → ReLU                           │ │
│  │  ├─ MaxPooling2D(2×2)                                        │ │
│  │  ├─ Conv2D(64 filters, 3×3) → ReLU                           │ │
│  │  ├─ MaxPooling2D(2×2)                                        │ │
│  │  ├─ Flatten                                                  │ │
│  │  ├─ Dense(128) → ReLU                                        │ │
│  │  ├─ Dropout(0.5)                                             │ │
│  │  └─ Dense(10) → Softmax                                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│      ↓                                                              │
│  Data Augmentation (Rotation, Shift, Zoom)                        │
│      ↓                                                              │
│  Model Compilation (Adam optimizer, categorical cross-entropy)    │
│      ↓                                                              │
│  Training Loop (Multiple Epochs)                                  │
│  ├─ Forward Pass through CNN                                      │
│  ├─ Compute Loss                                                  │
│  ├─ Backpropagation & Weight Updates                              │
│  └─ Validation every epoch                                        │
│      ↓                                                              │
│  Evaluation on Test Set                                           │
│      ↓                                                              │
│  Output: Model, Predictions, Accuracy, Confusion Matrix           │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **TensorFlow/Keras**: CNN model building and training
- **PyTorch**: Alternative deep learning framework
- **NumPy**: Numerical operations
- **Pandas**: Data handling
- **Matplotlib**: Visualization of training curves and sample images
- **Scikit-learn**: Evaluation metrics and confusion matrix
- **OpenCV**: Image processing (optional)

### Sample Project Structure

```
cnn_image_classification_project/
├── data/
│   ├── raw/
│   │   └── mnist/
│   │       ├── train_images.npy
│   │       └── train_labels.npy
│   └── processed/
│       ├── normalized_train.npy
│       └── normalized_test.npy
├── notebooks/
│   └── cnn_training.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── model_builder.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── cnn_model.h5
│   └── cnn_model_weights.pkl
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── learned_filters.png
│   ├── accuracy_metrics.csv
│   └── classification_report.txt
├── config/
│   └── hyperparameters.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load MNIST dataset
2. **PreprocessingModule**: Normalize and prepare images
3. **DataAugmentationModule**: Implement augmentation transformations
4. **ModelBuilderModule**: Design CNN architecture
5. **TrainingModule**: Handle training loop and callbacks
6. **EvaluationModule**: Calculate metrics and create confusion matrix
7. **VisualizationModule**: Plot training curves, filters, predictions

**Important Functions/Classes (Conceptual):**

- `load_mnist_dataset()`: Load and return MNIST data
- `normalize_images()`: Scale pixel values to [0, 1]
- `create_data_augmentation()`: Define augmentation pipeline
- `build_cnn_model()`: Construct sequential CNN architecture
- `compile_model()`: Set loss, optimizer, metrics
- `train_model()`: Execute training loop with validation
- `evaluate_model()`: Test set evaluation
- `predict_classes()`: Generate predictions
- `create_confusion_matrix()`: Visualize prediction matrix
- `plot_training_curves()`: Plot loss and accuracy histories
- `visualize_learned_filters()`: Display early layer filters
- `analyze_misclassifications()`: Identify failure patterns

**Configuration Requirements:**

- Batch size (default: 32 or 64)
- Number of epochs (default: 50-100)
- Learning rate (typically 0.001)
- Dropout rate (typically 0.5)
- Data augmentation parameters (rotation angle, shift range)
- Train-validation-test split ratios

**Reproducibility Considerations:**

- Set random seeds for TensorFlow, Keras, NumPy
- Save model architecture and weights separately
- Document all hyperparameter values
- Record GPU/CPU usage if applicable
- Save training history and metrics

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | 28×28×1 grayscale images; pixel values 0-255; 10 classes (digits 0-9) |
| **Processing** | Normalization to [0,1]; train-val-test split; data augmentation during training |
| **Output** | Trained CNN model, predicted class labels, class probabilities, accuracy (~98%+ expected) |

### Evaluation Methodology

**Image Classification Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | Overall correctness | (TP+TN)/(TP+TN+FP+FN); goal ~98%+ for MNIST |
| **Precision (per class)** | Correctness of positive predictions | TP/(TP+FP) for each digit class |
| **Recall (per class)** | Coverage of actual positives | TP/(TP+FN) for each digit class |
| **F1-Score** | Harmonic mean | 2×(Precision×Recall)/(Precision+Recall) |
| **Macro-averaged Metrics** | Average across all classes | Unweighted average over classes |

**Assessment Procedure:**

1. Monitor training/validation accuracy and loss during epochs
2. Detect overfitting by comparing training vs. validation curves
3. Evaluate on independent test set
4. Generate confusion matrix and per-class metrics
5. Analyze misclassified samples visually
6. Compare different architectures if time permits

### Expected Outcomes

1. **MNIST Classification Results:**
   - Training accuracy: ~99%
   - Validation accuracy: ~98-99%
   - Test accuracy: ~97-98% (achievable with basic CNN)
   - Fast convergence within 20-30 epochs

2. **Training Characteristics:**
   - Smooth learning curves without extreme fluctuations
   - Validation loss plateaus after ~20 epochs
   - No significant overfitting for basic architecture

3. **Model Observations:**
   - Early convolutional layers learn edge and shape features
   - Deeper layers learn complex patterns
   - Dropout effectively reduces overfitting
   - Data augmentation may improve generalization slightly

4. **Deliverables:**
   - Trained CNN model in HDF5 or SavedModel format
   - Training history plots (loss and accuracy)
   - Confusion matrix visualization
   - Sample predictions with confidence scores
   - Visualization of learned filters
   - Comprehensive performance report

---

## Experiment 5: Sequential Data Modeling using RNN

### Objective

Develop and train a Recurrent Neural Network (RNN) to model sequential data dependencies and make predictions on time series or sequential patterns. Implement LSTM cells for handling long-term dependencies, evaluate model performance, and analyze predictions.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Book Sales Prediction or Time Series Synthetic Data |
| **Source/URL** | Kaggle, UCI ML Repository, or synthetically generated |
| **Brief Description** | Sequential sales data with temporal patterns; typically 1000-10000 samples; univariate or multivariate time series |
| **Important Features/Attributes** | Time step index, sales values, optional: price, promotion indicators; temporal dependency expected |

### Methodology

1. **Data Collection and Acquisition**
   - Load sequential/time series dataset
   - Verify temporal ordering and continuity
   - Check for missing time steps or gaps
   - Document temporal resolution (daily, hourly, etc.)
   - Analyze temporal patterns and seasonality

2. **Data Preprocessing**
   - Handle missing values using interpolation (linear, forward-fill)
   - Normalize values to range [0, 1] or standardize
   - Remove outliers using statistical methods
   - Ensure consistent time step intervals
   - Create lagged features for sequence generation

3. **Feature Engineering**
   - Generate sliding windows of sequence data
   - Define lookback window (past steps) and lookahead (prediction steps)
   - Create input-output pairs for supervised learning
   - For multivariate, reshape to (samples, timesteps, features)
   - Add temporal features if useful (day of week, month, etc.)

4. **Model Selection and Architecture Design**
   - Design RNN architecture:
     - Input layer for sequence data
     - RNN/LSTM layers for sequence processing
     - Dropout for regularization
     - Dense output layer for prediction
   - Consider sequence length and complexity
   - Decide on return sequences for stacked RNNs

5. **Model Training**
   - Compile with appropriate loss function (MSE for regression)
   - Use Adam or RMSprop optimizer
   - Set batch size considering sequence length
   - Train for multiple epochs with early stopping
   - Monitor training and validation loss
   - Store best model during training

6. **Validation and Testing**
   - Evaluate on validation set
   - Generate predictions on test sequences
   - Calculate regression metrics (MSE, MAE, RMSE)
   - Analyze prediction errors and residuals
   - Perform walk-forward validation if appropriate

7. **Result Analysis**
   - Compare predictions with actual values
   - Analyze prediction accuracy at different time horizons
   - Visualize prediction vs. actual trajectories
   - Investigate failure cases
   - Examine learned temporal patterns

### Model/Algorithm Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│              RNN/LSTM SEQUENTIAL MODELING WORKFLOW                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Time Series Data (Sales, Prices, etc.)                          │
│      ↓                                                            │
│  Normalization & Preprocessing                                   │
│      ↓                                                            │
│  Create Sliding Windows (Lookback=30, Lookahead=1)              │
│  (Reshape to: samples × timesteps × features)                    │
│      ↓                                                            │
│  Train-Validation-Test Split                                     │
│      ↓                                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RNN/LSTM ARCHITECTURE                                    │   │
│  │  ├─ Input Layer (None, timesteps, features)               │   │
│  │  ├─ LSTM Layer(128 units) → Return Sequences              │   │
│  │  ├─ Dropout(0.2)                                          │   │
│  │  ├─ LSTM Layer(64 units) → Return Last Only               │   │
│  │  ├─ Dropout(0.2)                                          │   │
│  │  └─ Dense(1) → Linear (for regression)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│      ↓                                                            │
│  Training Loop (Forward Pass through Time Steps)                 │
│  ├─ Process sequence step-by-step                                │
│  ├─ Compute loss from output vs. target                          │
│  ├─ Backpropagation through time (BPTT)                         │
│  └─ Update weights based on gradient                             │
│      ↓                                                            │
│  Validation & Testing with Sliding Windows                       │
│      ↓                                                            │
│  Output: Predictions, Loss, Metrics, Visualizations              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **TensorFlow/Keras**: RNN and LSTM model building
- **PyTorch**: Alternative deep learning framework
- **NumPy**: Numerical operations and sequence handling
- **Pandas**: Time series data manipulation
- **Matplotlib**: Visualization of time series and predictions
- **Scikit-learn**: Evaluation metrics
- **Statsmodels**: Time series analysis and autocorrelation

### Sample Project Structure

```
rnn_sequential_modeling_project/
├── data/
│   ├── raw/
│   │   └── book_sales.csv
│   └── processed/
│       ├── normalized_sales.csv
│       └── sequences_data.npy
├── notebooks/
│   └── rnn_training.ipynb
├── src/
│   ├── data_loader.py
│   ├── time_series_preprocessing.py
│   ├── sequence_generator.py
│   ├── model_builder.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── rnn_model.h5
│   └── model_weights.pkl
├── results/
│   ├── training_history.png
│   ├── predictions_vs_actual.png
│   ├── error_analysis.png
│   ├── autocorrelation_plot.png
│   ├── predictions.csv
│   └── metrics_report.txt
├── config/
│   └── hyperparameters.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load time series data
2. **PreprocessingModule**: Normalize and clean time series
3. **SequenceGeneratorModule**: Create sliding windows
4. **ModelBuilderModule**: Design RNN/LSTM architecture
5. **TrainingModule**: Handle training with time series considerations
6. **EvaluationModule**: Calculate regression metrics
7. **VisualizationModule**: Plot series, predictions, errors

**Important Functions/Classes (Conceptual):**

- `load_time_series()`: Load sales or time series data
- `normalize_time_series()`: Standardize or scale values
- `create_sequences()`: Generate sliding window samples
- `build_rnn_model()`: Construct RNN architecture
- `build_lstm_model()`: Construct LSTM architecture
- `compile_model()`: Set loss function and optimizer
- `train_model()`: Execute training loop
- `predict_sequences()`: Generate predictions on test sequences
- `inverse_normalize()`: Convert predictions back to original scale
- `calculate_regression_metrics()`: Compute MSE, MAE, RMSE
- `plot_predictions()`: Visualize prediction vs. actual
- `analyze_errors()`: Examine prediction residuals

**Configuration Requirements:**

- Lookback window size (past timesteps, typically 30-60)
- Lookahead/forecast horizon (1 for one-step, >1 for multi-step)
- Batch size (typically 32)
- Number of epochs (typically 50-100)
- LSTM unit counts (typically 64-256)
- Dropout rate (typically 0.1-0.2)

**Reproducibility Considerations:**

- Set random seeds for all libraries
- Document normalization parameters for inference
- Save model with architecture and weights
- Record all hyperparameter values
- Log training metrics and convergence

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | Time series data (1D or multivariate); typically 1000+ samples; temporal ordering essential |
| **Processing** | Normalization; sliding window conversion to 3D array (samples, timesteps, features) |
| **Output** | Predicted next value(s), prediction confidence, MSE/MAE error metrics |

### Evaluation Methodology

**Time Series Regression Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Mean Absolute Error (MAE)** | (1/n)×Σ\|y_true - y_pred\| | Average absolute error in original units |
| **Mean Squared Error (MSE)** | (1/n)×Σ(y_true - y_pred)² | Penalizes larger errors more |
| **Root Mean Squared Error (RMSE)** | √MSE | Error in original scale |
| **Mean Absolute Percentage Error (MAPE)** | (1/n)×Σ(\|(y_true-y_pred)/y_true\|×100) | Percentage error |

**Assessment Procedure:**

1. Perform walk-forward validation on time series
2. Calculate metrics on multiple time horizons
3. Analyze prediction errors for patterns
4. Check for autocorrelation in residuals
5. Visualize actual vs. predicted trajectories
6. Compare with baseline methods (moving average, ARIMA)

### Expected Outcomes

1. **RNN/LSTM Results:**
   - RMSE typically 5-15% of data range
   - Captures temporal patterns and trends
   - Faster training than CNNs for sequential data
   - Smooth prediction curves following actual trends

2. **Training Characteristics:**
   - Convergence within 30-50 epochs typically
   - Smooth learning curves without spikes
   - Validation loss close to training loss

3. **Observations:**
   - LSTM handles long-term dependencies better than simple RNN
   - Dropout important to prevent overfitting
   - Sequence length affects model capacity
   - Normalization critical for convergence

4. **Deliverables:**
   - Trained RNN/LSTM model
   - Training history plots
   - Prediction vs. actual visualization
   - Error analysis and residual plots
   - Performance metrics report
   - Forecast for future periods

---

## Experiment 6: Object Detection using YOLO

### Objective

Implement object detection using YOLO (You Only Look Once) to identify and localize multiple objects in images. Train or fine-tune YOLO model on vehicle detection dataset, achieve real-time detection capability, and analyze performance metrics.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Vehicle Detection Dataset or COCO Dataset subset |
| **Source/URL** | Kaggle (vehicle detection datasets), COCO dataset (cocodataset.org) |
| **Brief Description** | Images containing vehicles (cars, trucks, buses) with bounding box annotations; 500-5000 training images |
| **Important Features/Attributes** | Bounding box coordinates (x, y, width, height), class labels (car, truck, bus), image size, confidence scores |

### Methodology

1. **Data Collection and Acquisition**
   - Download vehicle detection dataset with annotations
   - Verify annotation format (Pascal VOC, COCO JSON, or YOLO txt)
   - Check image quality and variety
   - Document dataset statistics (images, objects, classes)
   - Validate annotation accuracy on sample images

2. **Data Preprocessing and Annotation Conversion**
   - Convert annotations to YOLO format if needed (class_id, x_center, y_center, width, height in normalized coordinates)
   - Resize images to standard size (416×416 for YOLOv3, 640×640 for YOLOv5/v8)
   - Split data into training, validation, testing (70-15-15)
   - Create dataset YAML configuration file with paths and classes
   - Verify annotations align with images

3. **Data Augmentation**
   - Apply random horizontal flips
   - Apply random brightness/contrast adjustments
   - Apply random rotations (±10 degrees)
   - Apply random crops and resizing
   - Augment annotations accordingly
   - Use augmentation only on training set

4. **Model Selection**
   - Select YOLO version:
     - YOLOv5: Balanced performance and speed
     - YOLOv8: Latest with improved accuracy
     - YOLOv3: Older but well-documented
   - Decide between pre-trained weights (transfer learning) or training from scratch
   - Pre-trained weights recommended for vehicle detection

5. **Model Training**
   - Load pre-trained YOLO model weights
   - Configure training parameters (learning rate, momentum, weight decay)
   - Set batch size (typical: 16-32 on GPU)
   - Train for multiple epochs (50-100 typical)
   - Implement learning rate scheduling
   - Monitor loss convergence and validation metrics

6. **Validation and Testing**
   - Evaluate on validation set during training
   - Generate predictions on test set
   - Extract bounding boxes and confidence scores
   - Calculate IoU (Intersection over Union) for bounding boxes
   - Compute precision, recall, and mAP metrics
   - Perform threshold tuning for detection confidence

7. **Result Analysis**
   - Visualize detections on sample images
   - Analyze false positives and false negatives
   - Evaluate performance by object class
   - Assess detection quality for different image conditions
   - Document model inference speed (FPS)

### Model/Algorithm Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│               YOLO OBJECT DETECTION WORKFLOW                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Images (416×416 or 640×640)                                 │
│      ↓                                                              │
│  Image Preprocessing & Normalization                               │
│      ↓                                                              │
│  Annotation Conversion to YOLO Format (Normalized Coordinates)    │
│      ↓                                                              │
│  Train-Val-Test Split (70-15-15)                                   │
│      ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         YOLO ARCHITECTURE & TRAINING                          │ │
│  │  ├─ Pre-trained Backbone (Feature Extraction)                │ │
│  │  ├─ Feature Pyramid (Multi-scale feature maps)              │ │
│  │  ├─ Detection Head (Predicts bboxes, objectness, classes)   │ │
│  │  │                                                           │ │
│  │  │  For each grid cell, predict:                            │ │
│  │  │  - Bounding box: (x, y, width, height)                  │ │
│  │  │  - Objectness: P(object)                                │ │
│  │  │  - Class probabilities: P(class|object)                 │ │
│  │                                                              │ │
│  │  Loss = Localization Loss + Objectness Loss + Classification │ │
│  │         (MSE for boxes + CE for classification)              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│      ↓                                                              │
│  Training Loop (Multiple Epochs)                                   │
│  ├─ Forward pass through YOLO network                              │
│  ├─ Compute multi-task loss                                        │
│  ├─ Backpropagation                                                │
│  └─ Weight update with optimizer                                   │
│      ↓                                                              │
│  Non-Maximum Suppression (NMS) for overlapping boxes              │
│      ↓                                                              │
│  Evaluation: Precision, Recall, mAP, F1-score                     │
│      ↓                                                              │
│  Output: Bboxes, Class Labels, Confidence Scores, Detections      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Ultralytics YOLOv5/v8**: Official YOLO implementation
- **PyTorch**: Deep learning framework
- **OpenCV**: Image reading, preprocessing, and visualization
- **NumPy**: Numerical operations
- **Pandas**: Data handling
- **Matplotlib**: Visualization of detections
- **Scikit-learn**: Evaluation metrics
- **Albumentations**: Advanced image augmentation

### Sample Project Structure

```
yolo_object_detection_project/
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── datasets/
│   └── data.yaml
├── notebooks/
│   └── yolo_training.ipynb
├── src/
│   ├── data_loader.py
│   ├── annotation_converter.py
│   ├── model_trainer.py
│   ├── inference.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── yolov5s.pt
│   ├── best.pt
│   └── last.pt
├── results/
│   ├── detections/
│   │   ├── sample_1.jpg
│   │   └── sample_2.jpg
│   ├── metrics/
│   │   ├── precision_recall_curve.png
│   │   ├── confusion_matrix.png
│   │   └── mAP_scores.txt
│   ├── training_results.csv
│   └── detection_report.txt
├── config/
│   └── training_config.yaml
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load images and annotations
2. **AnnotationConverterModule**: Convert to YOLO format
3. **DataAugmentationModule**: Implement image and bbox augmentation
4. **ModelTrainerModule**: Train YOLO with Ultralytics API
5. **InferenceModule**: Run detection on new images
6. **EvaluationModule**: Calculate mAP, precision, recall
7. **VisualizationModule**: Draw bboxes and visualize results

**Important Functions/Classes (Conceptual):**

- `load_vehicle_dataset()`: Load images and annotations
- `convert_to_yolo_format()`: Convert annotations to YOLO format
- `create_data_yaml()`: Generate dataset configuration
- `load_pretrained_model()`: Load YOLOv5/v8 weights
- `train_yolo()`: Execute training loop
- `detect_objects()`: Run inference on images
- `apply_nms()`: Non-Maximum Suppression on detections
- `calculate_iou()`: Compute Intersection over Union
- `compute_map()`: Calculate mean Average Precision
- `visualize_detections()`: Draw bboxes on images
- `analyze_false_positives()`: Identify detection errors

**Configuration Requirements:**

- Image size (416×416 or 640×640)
- Batch size (typically 16-32)
- Number of epochs (typically 50-100)
- Learning rate and warmup epochs
- Confidence threshold for detections (typically 0.5)
- IoU threshold for NMS (typically 0.5-0.6)
- Dataset YAML with paths and class names

**Reproducibility Considerations:**

- Set random seeds for PyTorch
- Save trained weights with timestamp
- Document all hyperparameters in config file
- Version annotations and preprocessing code
- Log training metrics and convergence

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | Images (416×416 or 640×640 resized) with bounding box annotations in YOLO format |
| **Processing** | Annotation conversion, image normalization, data augmentation, batch preparation |
| **Output** | Bounding boxes (x_min, y_min, x_max, y_max), class labels, confidence scores for each detection |

### Evaluation Methodology

**Object Detection Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **Intersection over Union (IoU)** | Bbox accuracy | Overlap between predicted and ground truth; 0-1, higher is better |
| **Precision** | Correctness of detections | TP/(TP+FP); fraction of detections that are correct |
| **Recall** | Detection coverage | TP/(TP+FN); fraction of ground truth objects found |
| **Average Precision (AP)** | Per-class performance | Area under precision-recall curve for single class |
| **Mean Average Precision (mAP)** | Overall performance | Average AP across all classes; main metric |
| **F1-Score** | Harmonic mean | 2×(Precision×Recall)/(Precision+Recall) |

**Assessment Procedure:**

1. Calculate IoU for each prediction vs. ground truth
2. Classify predictions as TP, FP, or FN based on IoU threshold (0.5)
3. Compute precision and recall for each confidence threshold
4. Plot precision-recall curve
5. Calculate AP as area under curve
6. Average AP across all classes for mAP
7. Analyze per-class performance
8. Measure inference speed (FPS)

### Expected Outcomes

1. **YOLO Object Detection Results:**
   - mAP@0.5 typically 70-85% for vehicle detection
   - mAP@0.5:0.95 (stricter) typically 40-60%
   - Real-time detection (>30 FPS on GPU)
   - High recall (>80%) with reasonable precision

2. **Training Characteristics:**
   - Convergence within 50-100 epochs
   - Stable loss curves during training
   - Validation mAP increases with epochs
   - No extreme overfitting due to data augmentation

3. **Detection Observations:**
   - Early layers detect edges and shapes
   - Deeper layers combine features for object recognition
   - Multi-scale detection handles various sizes
   - NMS reduces duplicate detections effectively

4. **Deliverables:**
   - Trained YOLO model weights
   - Training metrics and loss curves
   - Detection visualizations on test images
   - mAP and per-class performance analysis
   - Precision-recall curves
   - Inference time benchmarks
   - Comprehensive detection report

---

## Experiment 7: Text Summarization using a Large Language Model (LLM)

### Objective

Develop a text summarization system using a Large Language Model to automatically generate concise summaries from longer documents. Implement prompt engineering techniques, evaluate summary quality, and compare abstractive vs. extractive approaches.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | CNN/DailyMail Dataset or arXiv Papers Dataset |
| **Source/URL** | HuggingFace Datasets library, Kaggle, or GitHub repositories |
| **Brief Description** | Articles with human-written summaries; diverse topics; 1000-100,000 document-summary pairs |
| **Important Features/Attributes** | Document text (500-2000 words), reference summary (50-200 words), document length, summary length |

### Methodology

1. **Data Collection and Acquisition**
   - Load pre-built dataset from HuggingFace or repository
   - Verify document-summary pairs align correctly
   - Analyze document and summary length distributions
   - Check language consistency (all English, etc.)
   - Document data source and licensing

2. **Data Preprocessing**
   - Clean text: remove HTML tags, special characters, excess whitespace
   - Tokenize documents into sentences/tokens
   - Handle encoding issues and convert to UTF-8
   - Filter documents by length (e.g., 100-2000 words)
   - Split data into train/validation/test (80-10-10)

3. **LLM Selection and Setup**
   - Choose LLM based on availability:
     - Open-source: BART, T5, Pegasus (via HuggingFace)
     - Commercial API: OpenAI GPT, Claude, Google PaLM
   - Set up API credentials if using commercial LLM
   - Test basic API functionality
   - Configure parameters: temperature, max_tokens, etc.

4. **Prompt Engineering**
   - Design summarization prompts:
     - Task specification: "Summarize this document in 2-3 sentences"
     - Length constraints: "Keep summary under 150 words"
     - Style guidance: "Write in bullet points / paragraph form"
     - Format specification: "Output only the summary, no explanation"
   - Experiment with prompt variations
   - Document best-performing prompts

5. **Summarization Generation**
   - Generate summaries for all test documents using LLM
   - Handle API rate limits and timeouts
   - Extract summaries from LLM responses
   - Store summaries with metadata
   - Clean generated summaries (remove extra whitespace, etc.)

6. **Validation and Testing**
   - Compare generated summaries with reference summaries
   - Calculate similarity metrics (ROUGE, BERTScore)
   - Evaluate summary coherence and informativeness
   - Perform manual quality assessment on sample summaries
   - Analyze length compliance with prompt specifications

7. **Result Analysis**
   - Identify prompt variations that produce best results
   - Analyze failure cases and challenging documents
   - Compare different summarization approaches
   - Assess computational cost and latency
   - Document findings and recommendations

### Model/Algorithm Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│                 LLM TEXT SUMMARIZATION WORKFLOW                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Document (Article, Paper, Report)                           │
│      ↓                                                              │
│  Text Preprocessing & Cleaning                                     │
│      ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         PROMPT ENGINEERING                                    │ │
│  │  ├─ Task specification                                        │ │
│  │  ├─ Format instructions                                       │ │
│  │  ├─ Length constraints                                        │ │
│  │  └─ Quality guidelines                                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│      ↓                                                              │
│  Prompt + Document → LLM                                           │
│      ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  LLM PROCESSING                                               │ │
│  │  ├─ Tokenization                                              │ │
│  │  ├─ Transformer encoding (multi-head attention)              │ │
│  │  ├─ Context understanding across document                    │ │
│  │  ├─ Key information extraction & abstraction                 │ │
│  │  └─ Token generation (auto-regressive)                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│      ↓                                                              │
│  Generated Summary                                                 │
│      ↓                                                              │
│  Post-processing & Validation                                      │
│      ↓                                                              │
│  Evaluation: ROUGE, BERTScore, Manual Assessment                  │
│      ↓                                                              │
│  Output: Summary Text, Quality Scores, Metrics                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Hugging Face Transformers**: Pre-trained LLMs (BART, T5, Pegasus)
- **Hugging Face Datasets**: Easy dataset loading and management
- **LangChain**: LLM prompt management and chaining
- **OpenAI API / Anthropic API**: Commercial LLM access
- **ROUGE**: Evaluation metrics for summarization
- **BERTScore**: Semantic similarity evaluation
- **Pandas**: Data handling
- **Matplotlib**: Results visualization

### Sample Project Structure

```
llm_text_summarization_project/
├── data/
│   ├── raw/
│   │   └── articles.csv
│   ├── processed/
│   │   ├── cleaned_articles.csv
│   │   └── test_documents.json
│   └── summaries/
│       ├── reference_summaries.json
│       └── generated_summaries.json
├── notebooks/
│   ├── data_exploration.ipynb
│   └── summarization_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── prompt_templates.py
│   ├── summarizer.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── prompts/
│   │   ├── prompt_v1.txt
│   │   ├── prompt_v2.txt
│   │   └── best_prompt.txt
│   └── config/
│       └── model_config.json
├── results/
│   ├── summaries_output.csv
│   ├── rouge_scores.csv
│   ├── quality_assessment.txt
│   ├── prompt_comparison.png
│   └── evaluation_report.txt
├── config/
│   └── api_config.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load documents and reference summaries
2. **PreprocessingModule**: Clean and prepare text
3. **PromptEngineerModule**: Manage and test different prompts
4. **SummarizerModule**: Call LLM and generate summaries
5. **EvaluationModule**: Calculate ROUGE and similarity metrics
6. **ComparisonModule**: Compare prompt and LLM variations
7. **VisualizationModule**: Plot evaluation results

**Important Functions/Classes (Conceptual):**

- `load_summarization_data()`: Load articles and references
- `clean_text()`: Remove HTML, special characters
- `create_summarization_prompt()`: Build prompt from template
- `generate_summary()`: Call LLM API with prompt and document
- `extract_summary_text()`: Parse LLM response
- `calculate_rouge_scores()`: Compute ROUGE-1, ROUGE-2, ROUGE-L
- `calculate_bertscore()`: Compute semantic similarity
- `evaluate_summary_quality()`: Multi-metric evaluation
- `compare_prompts()`: Evaluate different prompt versions
- `assess_length_compliance()`: Check if summary meets length constraints
- `visualize_metrics()`: Plot performance across documents

**Configuration Requirements:**

- LLM model name and version
- API key and endpoint configuration
- Temperature (controls randomness, typically 0.5-0.9)
- Max tokens for summary (typically 150-300)
- Summary length constraints (word count or token count)
- Batch size for processing multiple documents
- Evaluation metrics to compute

**Reproducibility Considerations:**

- Set random seed for temperature-based LLMs
- Log all prompts used for each document
- Save API responses with timestamps
- Document LLM version and date used
- Record API rate limit and cost information

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | Document text (500-2000 words); optional reference summary for evaluation |
| **Processing** | Text cleaning, prompt template insertion, API call batching, response parsing |
| **Output** | Generated summary (50-300 words), ROUGE scores, BERTScore, quality assessment |

### Evaluation Methodology

**Text Summarization Evaluation Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **ROUGE-1** | Unigram overlap | Percentage of single words in generated summary matching reference |
| **ROUGE-2** | Bigram overlap | Percentage of word pairs matching |
| **ROUGE-L** | Longest common subsequence | Longest sequence of matching words |
| **BERTScore** | Semantic similarity | Embedding-based similarity; captures semantic alignment |
| **Summary Length** | Constraint compliance | Generated summary length vs. specified length |
| **Coherence (Manual)** | Readability and flow | Human judgment of summary quality and readability |

**Assessment Procedure:**

1. Generate summaries for test documents
2. Calculate ROUGE scores against reference summaries
3. Compute BERTScore for semantic evaluation
4. Assess summary length compliance
5. Perform manual quality assessment on sample summaries
6. Compare different prompt variations
7. Analyze results by document category/length

### Expected Outcomes

1. **Text Summarization Results:**
   - ROUGE-1 score typically 35-50% (depends on LLM and prompt)
   - ROUGE-L score typically 30-45%
   - BERTScore typically 0.85-0.95
   - Most summaries meet length specifications

2. **Prompt Engineering Observations:**
   - Specific task instructions improve quality
   - Length constraints effectively control summary size
   - Format guidance (bullet vs. paragraph) followed reasonably well
   - Different prompts show 5-15% variation in quality

3. **LLM Characteristics:**
   - Larger models generally produce better summaries
   - Temperature affects diversity of outputs
   - Multi-turn prompting can refine summaries
   - API-based models may outperform open-source alternatives

4. **Deliverables:**
   - Generated summaries for test documents
   - ROUGE and BERTScore evaluation metrics
   - Prompt engineering analysis and best prompts
   - Quality assessment report with sample summaries
   - Comparison of different LLM/prompt combinations
   - Recommendations for production deployment

---

## Experiment 8: Grammar Correction and Text Rewriting using a Large Language Model (LLM)

### Objective

Develop a grammar correction and text rewriting system using an LLM to automatically improve written content by fixing errors and enhancing clarity. Implement prompt variations to control rewriting intensity, evaluate correction accuracy, and assess output quality.

### Recommended Dataset

| Aspect | Details |
|--------|---------|
| **Dataset Name** | Lang-8 Learner Corpora or JFLEG Dataset |
| **Source/URL** | HuggingFace Datasets, GitHub repositories, or academic datasets |
| **Brief Description** | Sentences with grammar errors paired with corrected versions; diverse error types; 10,000-100,000 sentence pairs |
| **Important Features/Attributes** | Erroneous sentence, corrected sentence, error type (spelling, grammar, punctuation, style), source language (non-native English) |

### Methodology

1. **Data Collection and Acquisition**
   - Load grammar correction dataset from repository
   - Verify sentence-correction pairs alignment
   - Analyze error type distribution (spelling, grammar, punctuation, style)
   - Check language and difficulty levels
   - Document dataset statistics and characteristics

2. **Data Preprocessing**
   - Clean text: standardize whitespace and encoding
   - Filter by error type or difficulty if needed
   - Split into train/validation/test sets (80-10-10)
   - Separate error types for targeted evaluation (optional)
   - Create categorized datasets by correction scope

3. **LLM Selection and Configuration**
   - Choose LLM suitable for text editing:
     - Specialized models: GECToR (Grammar Error Correction)
     - General LLMs: GPT, Claude (via API)
     - Open-source: T5-based models
   - Set up API or local model
   - Configure generation parameters

4. **Prompt Engineering for Corrections**
   - Design correction prompts:
     - Error detection focus: "Find and explain all grammatical errors"
     - Correction focus: "Fix grammar errors, maintain original meaning"
     - Minimal changes: "Fix only critical errors, preserve voice"
     - Comprehensive rewriting: "Improve writing quality, grammar, clarity, and style"
   - Create prompt variations for different correction levels:
     - Level 1: Spelling and obvious grammar
     - Level 2: Complex grammar structures
     - Level 3: Style and clarity enhancement
   - Experiment with prompt phrasing

5. **Grammar Correction and Rewriting**
   - Process error sentences with different prompts
   - Generate corrections for each input
   - Extract corrected text from LLM responses
   - Apply post-processing (whitespace normalization)
   - Store outputs with metadata (prompt version, LLM, timestamp)

6. **Validation and Testing**
   - Compare generated corrections with reference corrections
   - Calculate exact match accuracy
   - Compute edit distance and word-level F1 scores
   - Evaluate for over/under-correction
   - Perform manual quality assessment
   - Assess preservation of original meaning

7. **Result Analysis**
   - Analyze performance by error type
   - Compare prompt variations and correction levels
   - Identify failure cases (over-correction, under-correction)
   - Evaluate computational efficiency
   - Assess practical usability for different scenarios

### Model/Algorithm Workflow

```
┌───────────────────────────────────────────────────────────────────┐
│          LLM GRAMMAR CORRECTION & REWRITING WORKFLOW               │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input Sentence (With Grammar/Style Errors)                       │
│      ↓                                                             │
│  Text Preprocessing                                               │
│      ↓                                                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  PROMPT DESIGN FOR TEXT CORRECTION                          │ │
│  │  ├─ Error detection prompt                                  │ │
│  │  ├─ Minimal correction prompt                               │ │
│  │  ├─ Standard correction prompt                              │ │
│  │  └─ Comprehensive rewriting prompt                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      ↓                                                             │
│  Prompt + Erroneous Sentence → LLM                               │
│      ↓                                                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  LLM TEXT PROCESSING                                        │ │
│  │  ├─ Error identification (attention to syntax/semantics)   │ │
│  │  ├─ Correction generation (maintains meaning)              │ │
│  │  ├─ Alternative phrasings (if rewriting)                  │ │
│  │  └─ Output generation (auto-regressive token prediction)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│      ↓                                                             │
│  Corrected Sentence                                              │
│      ↓                                                             │
│  Post-processing & Validation                                    │
│      ↓                                                             │
│  Evaluation: Accuracy, Semantic Preservation, Quality            │
│      ↓                                                             │
│  Output: Corrected Text, Quality Metrics, Error Analysis         │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### Tools and Libraries

- **Python 3.8+**
- **Hugging Face Transformers**: Pre-trained correction models
- **LangChain**: Prompt management and LLM integration
- **OpenAI API / Anthropic API**: Commercial LLM access
- **NLTK**: Tokenization and text processing
- **TextBlob**: Linguistic features and basic correction
- **Diff-match-patch**: Compute text differences
- **editdistance**: Calculate Levenshtein distance
- **Pandas**: Data handling
- **Matplotlib**: Visualization

### Sample Project Structure

```
llm_grammar_correction_project/
├── data/
│   ├── raw/
│   │   └── lang8_errors.csv
│   ├── processed/
│   │   ├── cleaned_sentences.csv
│   │   └── error_categorized.csv
│   └── outputs/
│       ├── corrected_sentences.json
│       └── correction_details.json
├── notebooks/
│   ├── data_analysis.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── prompt_manager.py
│   ├── corrector.py
│   ├── evaluation.py
│   ├── error_analyzer.py
│   └── visualization.py
├── models/
│   ├── prompts/
│   │   ├── minimal_correction.txt
│   │   ├── standard_correction.txt
│   │   ├── comprehensive_rewriting.txt
│   │   └── error_detection.txt
│   └── config/
│       └── correction_config.json
├── results/
│   ├── accuracy_scores.csv
│   ├── performance_by_error_type.csv
│   ├── prompt_comparison.png
│   ├── error_type_distribution.png
│   ├── sample_corrections.txt
│   └── evaluation_report.txt
├── config/
│   └── llm_config.json
└── README.md
```

### Implementation Guidelines

**Key Modules/Files to Create:**

1. **DataLoaderModule**: Load error-correction sentence pairs
2. **PreprocessingModule**: Clean and categorize error types
3. **PromptManagerModule**: Manage correction prompts
4. **CorrectorModule**: Call LLM and generate corrections
5. **EvaluationModule**: Calculate accuracy and similarity metrics
6. **ErrorAnalyzerModule**: Categorize and analyze correction errors
7. **VisualizationModule**: Plot evaluation results

**Important Functions/Classes (Conceptual):**

- `load_error_dataset()`: Load Lang-8 or JFLEG data
- `categorize_errors()`: Classify error types (spelling, grammar, etc.)
- `create_correction_prompt()`: Build prompt from template
- `generate_correction()`: Call LLM with error sentence
- `extract_corrected_text()`: Parse LLM response
- `calculate_exact_match()`: Compare with reference correction
- `calculate_edit_distance()`: Compute Levenshtein distance
- `calculate_word_f1_score()`: Token-level F1 score
- `assess_meaning_preservation()`: Check semantic equivalence
- `detect_over_correction()`: Identify unnecessary changes
- `evaluate_by_error_type()`: Performance breakdown
- `compare_prompt_versions()`: Evaluate different prompts

**Configuration Requirements:**

- LLM model name and API endpoint
- Temperature for generation (typically 0.3-0.5 for consistency)
- Max tokens for correction (typically 100-200)
- Correction level (minimal, standard, comprehensive)
- Evaluation metrics to compute
- Error type categories

**Reproducibility Considerations:**

- Document all prompt versions used
- Log LLM responses with timestamps
- Set fixed random seeds for consistency
- Record API version and model version
- Version control for evaluation code

### Input and Output Specifications

| Aspect | Description |
|--------|-------------|
| **Input** | Sentence with grammatical/spelling errors (typically 10-30 words); optionally categorized by error type |
| **Processing** | Error type identification, prompt selection, LLM API call, response parsing |
| **Output** | Corrected sentence, error type identified, correction confidence, comparison to reference |

### Evaluation Methodology

**Grammar Correction Evaluation Metrics:**

| Metric | Purpose | Interpretation |
|--------|---------|-----------------|
| **Exact Match Accuracy** | Perfect corrections | Percentage of corrections exactly matching reference |
| **Edit Distance (Levenshtein)** | Similarity of output | Number of character edits needed to match reference |
| **Token-level F1** | Word overlap | F1 score comparing tokens between output and reference |
| **Over-correction Rate** | Unnecessary changes | Percentage of unnecessary or harmful modifications |
| **Under-correction Rate** | Missed errors | Percentage of errors not corrected |
| **Semantic Preservation** | Meaning maintained | Manual assessment of whether meaning is preserved |

**Assessment Procedure:**

1. Generate corrections for test sentences
2. Compare with reference corrections
3. Calculate exact match accuracy
4. Compute edit distance for each correction
5. Calculate word-level F1 scores
6. Identify over and under-corrections
7. Perform error type-wise evaluation
8. Manual quality assessment on samples

### Expected Outcomes

1. **Grammar Correction Results:**
   - Exact match accuracy typically 50-80% (depends on LLM and error types)
   - Edit distance small when corrections differ (good alternative solutions)
   - Token F1 scores typically 0.85-0.95
   - Low over-correction rate with well-designed prompts

2. **Prompt Effects:**
   - Minimal correction prompts reduce over-correction
   - Standard prompts balance comprehensiveness and safety
   - Comprehensive rewriting shows higher divergence from reference
   - Prompt specificity significantly impacts quality

3. **Error Type Analysis:**
   - Spelling errors: >90% accuracy
   - Simple grammar: 70-85% accuracy
   - Complex structures: 50-70% accuracy
   - Style issues: high variation, LLM-dependent

4. **Deliverables:**
   - Corrected sentences for test set
   - Accuracy metrics and error analysis
   - Comparison of different prompt versions
   - Performance breakdown by error type
   - Sample corrections with explanations
   - Recommendations for optimal prompt design
   - Evaluation report with visualizations

---

## Experiment 9: Advanced Domain-Specific EEE Question Answering Chatbot Using LLM, RAG and Ollama/API

### Objective

Develop a **live, advanced domain-specific Question Answering (QA) chatbot for Electrical and Electronics Engineering (EEE)** using a Large Language Model (LLM), Retrieval-Augmented Generation (RAG), vector search, conversational memory, and real-time response streaming.

The system should behave like a modern AI assistant similar in interaction style to **ChatGPT/Gemini**, while being specifically optimized for the **EEE domain**.

The chatbot must be capable of answering questions related to:

* Electrical Machines
* Power Systems
* Power Electronics
* Control Systems
* Electrical Measurements
* Electric Vehicles
* Renewable Energy Systems
* Energy Storage Systems
* Microgrids
* Smart Grids
* Electrical Drives
* Circuit Theory
* Digital Electronics
* Analog Electronics
* Signals and Systems
* Embedded Systems
* Engineering Mathematics
* AI/ML applications in Electrical Engineering
* Industrial Automation
* MATLAB/Simulink
* Electrical standards and technical documentation

The system should combine **LLM reasoning + domain knowledge retrieval + conversation context + source citations** to generate accurate and explainable answers.

---

### 1. System Objective

Build a production-style EEE AI assistant with the following architecture:

```text
                    ┌─────────────────────────────┐
                    │       USER / STUDENT        │
                    │                             │
                    │  Natural Language Question  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       CHAT INTERFACE        │
                    │                             │
                    │  ChatGPT/Gemini-like UI    │
                    │  • Streaming responses      │
                    │  • Markdown                 │
                    │  • Code blocks              │
                    │  • Tables                   │
                    │  • Math equations           │
                    │  • Source citations         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       CHAT ORCHESTRATOR     │
                    │                             │
                    │ Intent Detection            │
                    │ Query Classification        │
                    │ Conversation Memory         │
                    │ RAG Decision                │
                    └──────────────┬──────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
        │ Conversation   │ │ EEE RAG       │ │ Tool / API     │
        │ Memory         │ │ Pipeline      │ │ Layer          │
        │                │ │               │ │                │
        │ Short-term     │ │ Embeddings    │ │ Calculator     │
        │ Long-term      │ │ Vector DB     │ │ Web Search     │
        │ Chat history   │ │ Retrieval     │ │ MATLAB         │
        └────────────────┘ └───────┬───────┘ │ APIs           │
                                   │         └───────┬────────┘
                                   │                 │
                                   └────────┬────────┘
                                            ▼
                              ┌─────────────────────────┐
                              │       LLM ENGINE        │
                              │                         │
                              │ Ollama Local LLM        │
                              │ OR                      │
                              │ Cloud LLM API           │
                              │                         │
                              │ Qwen / Llama / Mistral  │
                              │ GPT / Gemini / Claude   │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   RESPONSE PROCESSOR    │
                              │                         │
                              │ • Answer validation     │
                              │ • Citation generation   │
                              │ • Formatting            │
                              │ • Confidence estimation│
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │       FINAL ANSWER      │
                              │                         │
                              │ Answer + Sources        │
                              │ Equations + Diagrams    │
                              │ Follow-up suggestions   │
                              └─────────────────────────┘
```

---

### 2. Core System Requirements

The chatbot should not be a simple question-answering script.

It should provide a **real-time conversational AI experience** with:

#### Required Features

* Real-time chat
* Streaming token-by-token responses
* Multi-turn conversation
* Conversation history
* Context-aware follow-up questions
* EEE-domain RAG
* Semantic document search
* Source citations
* PDF/document ingestion
* Mathematical equation support
* Code generation
* MATLAB/Simulink assistance
* Engineering calculations
* Technical explanations
* Exam-oriented answers
* Conceptual explanations
* Problem-solving
* Numerical analysis
* Comparison tables
* Structured responses
* Conversation memory
* Model selection
* Local LLM support through Ollama
* Cloud API support
* Fallback between models
* Hallucination reduction
* Retrieval confidence
* Error handling

---

### 3. LLM Architecture

The system must support **two inference modes**.

#### Mode A — Local LLM using Ollama

Use Ollama as the local inference server.

```text
Frontend
   ↓
Backend API
   ↓
Ollama API
   ↓
Local LLM
```

Possible models:

```text
Qwen
Llama
Mistral
Gemma
DeepSeek
Qwen-Coder
```

The implementation must allow the model to be changed through configuration rather than hard-coded.

Example:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=<selected-model>
```

---

### 4. Mode B — Cloud/API LLM

The architecture must also support API-based models.

```text
Frontend
      ↓
Backend
      ↓
LLM Provider Abstraction Layer
      ↓
┌────────────┬─────────────┬─────────────┐
│ OpenAI API │ Gemini API  │ Other APIs  │
└────────────┴─────────────┴─────────────┘
```

The application should use a common interface:

```python
generate_response(
    messages,
    model,
    temperature,
    stream=True
)
```

This allows the LLM provider to be changed without modifying the entire application.

---

### 5. EEE Domain Knowledge Base

Create a domain-specific knowledge base containing:

#### Core EEE textbooks

Examples:

* Electrical Machines
* Power Systems
* Power Electronics
* Control Systems
* Electrical Measurements
* Network Theory
* Signals and Systems
* Electrical Drives
* Power System Protection
* High Voltage Engineering
* Renewable Energy
* Electric Vehicle Technology
* Energy Storage Systems

#### Technical Documents

Include:

* Lecture notes
* Laboratory manuals
* Technical manuals
* Research papers
* Datasheets
* IEEE papers where legally available
* Manufacturer documentation
* MATLAB documentation
* Simulink documentation
* Course notes
* Standards/documentation where permitted

---

### 6. Document Processing Pipeline

```text
PDF / DOCX / TXT / HTML
          ↓
Document Loader
          ↓
Text Extraction
          ↓
Cleaning
          ↓
Metadata Extraction
          ↓
Semantic Chunking
          ↓
Chunk Validation
          ↓
Embedding Generation
          ↓
Vector Database
```

Each chunk should contain metadata such as:

```json
{
  "document_id": "power_systems_001",
  "title": "Power System Analysis",
  "subject": "Power Systems",
  "topic": "Load Flow Analysis",
  "chapter": "Newton Raphson Method",
  "page": 142,
  "source": "Power_System_Analysis.pdf",
  "chunk_id": "ps_001_142_04"
}
```

---

### 7. Advanced Chunking

Do not rely only on fixed 500-token chunks.

Implement configurable chunking:

```text
Fixed-size chunking
        +
Sentence-aware chunking
        +
Paragraph-aware chunking
        +
Heading-aware chunking
        +
Overlap
```

Recommended starting configuration:

```text
Chunk size: 400–700 tokens
Overlap: 50–100 tokens
Top-K retrieval: 5
```

The system should allow these values to be changed through configuration.

---

### 8. Embedding Model

Use a dedicated embedding model rather than using the LLM itself.

Possible choices:

```text
Sentence-Transformers
BGE
E5
Nomic embeddings
Other compatible embedding models
```

Example architecture:

```text
EEE Document
     ↓
Embedding Model
     ↓
Dense Vector
     ↓
Vector Database
```

The same embedding model must be used for:

```text
Document embeddings
        +
Query embeddings
```

---

### 9. Vector Database

Support a local vector database for experimentation.

Recommended:

```text
FAISS
```

Optional:

```text
Chroma
Qdrant
Weaviate
Pinecone
```

For the laboratory experiment, the default implementation should use:

```text
FAISS + persistent metadata
```

---

### 10. Query Processing

When the user submits:

> "Explain the working principle of an induction motor."

The system should process the query as follows:

```text
User Query
    ↓
Query Analyzer
    ↓
Intent Detection
    ↓
Domain Detection
    ↓
Query Embedding
    ↓
Vector Search
    ↓
Top-K Relevant Chunks
    ↓
Context Re-ranking
    ↓
RAG Prompt
    ↓
LLM
    ↓
Answer
```

---

### 11. Query Intent Classification

The chatbot should identify different types of engineering questions.

Example:

| Intent          | Example                                                    |
| --------------- | ---------------------------------------------------------- |
| Conceptual      | What is slip in an induction motor?                        |
| Definition      | Define power factor.                                       |
| Numerical       | Calculate the efficiency of a transformer.                 |
| Derivation      | Derive the EMF equation of a transformer.                  |
| Comparison      | Compare synchronous and induction motors.                  |
| Procedure       | Explain how to perform load-flow analysis.                 |
| Troubleshooting | Why is my inverter output distorted?                       |
| Programming     | Write MATLAB code for PID control.                         |
| Simulation      | How do I model an SRM in Simulink?                         |
| Design          | Design a buck converter for 24 V to 12 V.                  |
| Exam            | Give a 10-mark answer on synchronous machines.             |
| Research        | Explain recent developments in battery management systems. |

---

### 12. Intelligent RAG Decision

The chatbot should **not blindly perform retrieval for every message**.

Implement a routing mechanism:

```text
User Query
     ↓
Query Router
     │
     ├── Casual conversation
     │        ↓
     │      Direct LLM
     │
     ├── General EEE concept
     │        ↓
     │      RAG + LLM
     │
     ├── Document-specific question
     │        ↓
     │      RAG
     │
     ├── Numerical problem
     │        ↓
     │      Calculator/Reasoning + LLM
     │
     ├── MATLAB question
     │        ↓
     │      Code-oriented LLM
     │
     └── Current information
              ↓
          Web/API retrieval
```

This makes the system more similar to modern AI assistants.

---

### 13. RAG Prompt Construction

The system prompt should instruct the LLM to:

1. Answer using retrieved EEE context when relevant.
2. Do not invent technical facts.
3. Clearly distinguish retrieved information from reasoning.
4. Cite the source document.
5. State when information is unavailable.
6. Show equations correctly.
7. Explain calculations step-by-step.
8. Use SI units.
9. Define technical abbreviations.
10. Adapt explanations to the user's requested level.

Example:

```text
You are an advanced Electrical and Electronics Engineering AI assistant.

Use the retrieved context as the primary source when answering
domain-specific questions.

Rules:

- Do not fabricate technical information.
- If the retrieved context does not contain sufficient information,
  clearly state that additional information is required.
- Provide technically accurate explanations.
- Use SI units.
- Show equations using LaTeX.
- For numerical problems, show:
  Given → Formula → Substitution → Calculation → Answer.
- For comparisons, use tables when useful.
- Cite retrieved sources.
- Preserve the context of previous conversation turns.
- If the user asks for an exam answer, structure it appropriately.
- If the user asks for code, provide executable code with explanation.
```

---

### 14. Conversation Memory

The chatbot must support multi-turn conversations.

Example:

**User:**

> Explain induction motor slip.

**AI:**

> Slip is the difference between synchronous speed and rotor speed...

**User:**

> What happens if it increases?

The system should understand that **"it" refers to slip**.

Architecture:

```text
Current Query
      +
Conversation History
      +
Relevant Retrieved Context
      ↓
Context Manager
      ↓
LLM
```

Implement:

#### Short-Term Memory

Current conversation/session.

#### Long-Term Memory

Optional persistent storage for:

* User preferences
* Frequently discussed subjects
* Learning level
* Previous topics

The system must provide controls for clearing or disabling persistent memory.

---

### 15. ChatGPT/Gemini-Like User Interface

Develop a modern responsive interface.

#### Interface Components

```text
┌──────────────────────────────────────────────────────────┐
│ EEE AI ASSISTANT                              Model ▼    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ User: Explain the working of an SRM.                    │
│                                                          │
│ AI:                                                      │
│     An SRM operates based on the tendency of the rotor   │
│     to move toward the position of minimum reluctance... │
│                                                          │
│     Equation:                                            │
│     T = 1/2 i² dL/dθ                                    │
│                                                          │
│     Sources:                                             │
│     📄 Electrical Drives — Chapter 6                     │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Ask anything about Electrical Engineering...      ➤     │
└──────────────────────────────────────────────────────────┘
```

Required UI features:

* Markdown rendering
* LaTeX equation rendering
* Syntax-highlighted code
* Copy answer
* Copy code
* Regenerate response
* Stop generation
* Edit user message
* Conversation history
* New conversation
* Delete conversation
* Source references
* File upload
* Model selection
* Settings
* Dark/light mode
* Streaming response
* Loading indicator

---

### 16. File Upload and Personal RAG

Allow users to upload:

```text
PDF
DOCX
TXT
CSV
PPTX
```

Example:

> User uploads "Power Electronics Unit 3.pdf"

The system should automatically:

```text
Upload
 ↓
Parse
 ↓
Chunk
 ↓
Embed
 ↓
Index
 ↓
Associate with conversation
```

Then:

> "Explain the PWM techniques from the PDF I uploaded."

The chatbot should answer using the uploaded document.

---

### 17. Engineering Calculation Capability

The chatbot should support engineering calculations.

Example:

> A 10 kVA transformer has 95% efficiency at full load. Calculate losses if output power is 8 kW.

The system should produce:

```text
Given:
Output power = 8 kW
Efficiency = 95%

η = Output / Input

Input = 8 / 0.95
      = 8.421 kW

Losses = Input - Output
       = 0.421 kW

Therefore:
Losses ≈ 421 W
```

For reliable numerical computation, use a calculator/Python tool where appropriate instead of relying solely on LLM arithmetic.

---

### 18. MATLAB/Simulink Assistance

The chatbot should support:

```text
MATLAB
Simulink
Stateflow
Python
C/C++
Arduino
ESP32
```

Example:

> Write MATLAB code to plot the torque-speed characteristics of an induction motor.

The chatbot should provide:

```matlab
% MATLAB implementation
...
```

with an explanation of:

* Inputs
* Equations
* Algorithm
* Expected output
* Modification options

---

### 19. Advanced EEE Domain Agents

For an advanced implementation, introduce specialized agents.

```text
                    EEE AI ORCHESTRATOR
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
 Power Systems Agent   Machines Agent    Power Electronics
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          EV Agent    Renewable Agent   Control Agent
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    General EEE Agent
```

Each agent may use:

* Specialized prompts
* Domain-specific retrieval
* Dedicated tools
* Domain-specific documents

---

### 20. Backend Architecture

Recommended backend:

```text
Python
FastAPI
LangChain or equivalent orchestration layer
FAISS
Sentence-Transformers
Ollama
Pydantic
WebSocket/SSE
```

Architecture:

```text
                 ┌───────────────┐
                 │   Next.js UI  │
                 └───────┬───────┘
                         │
                   HTTP / WebSocket
                         │
                         ▼
                 ┌───────────────┐
                 │    FastAPI    │
                 └───────┬───────┘
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
        ┌──────────────┐   ┌──────────────┐
        │ RAG Engine   │   │ LLM Router   │
        └──────┬───────┘   └──────┬───────┘
               │                  │
               ▼                  ▼
        ┌──────────────┐   ┌──────────────┐
        │ FAISS        │   │ Ollama/API   │
        │ Vector DB    │   │ LLM          │
        └──────────────┘   └──────────────┘
```

---

### 21. API Endpoints

Implement APIs such as:

```text
POST /api/chat
POST /api/chat/stream
POST /api/documents/upload
POST /api/documents/index
GET  /api/documents
DELETE /api/documents/{id}

GET  /api/conversations
POST /api/conversations
GET  /api/conversations/{id}
DELETE /api/conversations/{id}

GET  /api/models
POST /api/models/select

POST /api/search
POST /api/evaluate
GET  /api/health
```

---

### 22. Streaming Response

The chatbot must support real-time generation.

```text
User Query
    ↓
FastAPI
    ↓
Ollama / LLM API
    ↓
Token 1 ─┐
Token 2  │
Token 3  │
Token 4  ├──→ WebSocket/SSE
Token 5  │
...      │
         ↓
Frontend progressively renders response
```

This creates a live conversational experience rather than waiting for the entire response.

---

### 23. Suggested Project Structure

```text
eee-ai-chatbot/
│
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── ChatWindow/
│   │   ├── MessageBubble/
│   │   ├── SourceCitation/
│   │   ├── FileUpload/
│   │   └── ModelSelector/
│   ├── lib/
│   └── styles/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   ├── documents.py
│   │   │   ├── conversations.py
│   │   │   └── models.py
│   │   │
│   │   ├── rag/
│   │   │   ├── loader.py
│   │   │   ├── parser.py
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   └── reranker.py
│   │   │
│   │   ├── llm/
│   │   │   ├── ollama.py
│   │   │   ├── openai.py
│   │   │   ├── gemini.py
│   │   │   └── router.py
│   │   │
│   │   ├── agents/
│   │   │   ├── orchestrator.py
│   │   │   ├── machines_agent.py
│   │   │   ├── power_systems_agent.py
│   │   │   ├── power_electronics_agent.py
│   │   │   ├── ev_agent.py
│   │   │   └── control_agent.py
│   │   │
│   │   ├── memory/
│   │   │   ├── conversation.py
│   │   │   └── persistent.py
│   │   │
│   │   ├── tools/
│   │   │   ├── calculator.py
│   │   │   ├── python_tool.py
│   │   │   └── engineering_tools.py
│   │   │
│   │   └── evaluation/
│   │       ├── evaluator.py
│   │       ├── retrieval_metrics.py
│   │       └── answer_metrics.py
│
├── data/
│   ├── documents/
│   ├── processed/
│   ├── embeddings/
│   └── qa_dataset/
│
├── vector_db/
│   ├── faiss.index
│   └── metadata.json
│
├── evaluation/
│   ├── test_questions.json
│   ├── predictions.json
│   └── metrics.csv
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── embedding_generation.ipynb
│   ├── rag_testing.ipynb
│   └── evaluation.ipynb
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md
```

---

### 24. Evaluation Dataset

Create an EEE QA benchmark containing questions from different categories.

Example:

```text
Electrical Machines       50 questions
Power Systems             50 questions
Power Electronics         50 questions
Control Systems            50 questions
EV Technology              50 questions
Renewable Energy           50 questions
Energy Storage             50 questions
Circuit Theory             50 questions
```

Total:

```text
400+ EEE questions
```

Each question should contain:

```json
{
  "id": "EEE_001",
  "domain": "Electrical Machines",
  "topic": "Induction Motor",
  "question": "What is slip in an induction motor?",
  "reference_answer": "...",
  "difficulty": "medium",
  "answer_type": "conceptual"
}
```

---

### 25. Evaluation Metrics

Evaluate both **retrieval** and **generation**.

#### Retrieval Metrics

```text
Recall@K
Precision@K
MRR
Hit Rate
Context Relevance
Answer-Span Retrieval Rate
```

#### Answer Metrics

```text
Exact Match
BLEU
ROUGE-L
BERTScore
Semantic Similarity
Answer Relevance
Faithfulness / Groundedness
Citation Accuracy
```

#### System Metrics

Also measure:

```text
Response latency
Time to first token
Tokens/second
Memory consumption
CPU/GPU utilization
Retrieval latency
LLM generation latency
```

---

### 26. Baseline Comparison

Compare three configurations:

#### Baseline 1 — LLM Only

```text
Question → LLM → Answer
```

#### Baseline 2 — RAG

```text
Question
   ↓
Retriever
   ↓
Context
   ↓
LLM
   ↓
Answer
```

#### Proposed System

```text
Question
   ↓
Intent Router
   ↓
Memory
   ↓
RAG
   ↓
Tools
   ↓
Specialized Agent
   ↓
LLM
   ↓
Validation
   ↓
Citations
   ↓
Answer
```

The evaluation should report measured differences rather than assuming that one architecture will always perform better.

---

### 27. Example Chatbot Interactions

#### Example 1 — Conceptual

**User:**

> Explain the working principle of an induction motor.

**AI:**

```text
An induction motor operates on the principle of electromagnetic
induction.

1. Three-phase AC is supplied to the stator.
2. A rotating magnetic field is produced.
3. The rotating field cuts the rotor conductors.
4. EMF is induced in the rotor.
5. Rotor current produces electromagnetic torque.
6. The rotor rotates in the direction of the rotating magnetic field.

Synchronous speed:

Ns = 120f / P

where:
f = supply frequency
P = number of poles

Source:
📄 Electrical Machines → Induction Motors → Working Principle
```

---

#### Example 2 — Follow-up Question

**User:**

> Why can't the rotor reach synchronous speed?

The system must understand the previous context and answer accordingly.

---

#### Example 3 — Numerical

**User:**

> A 4-pole induction motor operates at 50 Hz and runs at 1440 rpm. Find the slip.

The chatbot should calculate:

```text
Ns = 120 × 50 / 4
   = 1500 rpm

s = (Ns - Nr) / Ns

s = (1500 - 1440) / 1500
  = 0.04

Slip = 4%
```

---

#### Example 4 — MATLAB

**User:**

> Give MATLAB code to plot the torque-speed curve of an induction motor.

The chatbot should return properly formatted MATLAB code with an explanation.

---

### 28. Safety and Technical Reliability

The chatbot must clearly distinguish:

```text
Retrieved Fact
     ↓
LLM Explanation
     ↓
Engineering Calculation
     ↓
Model Inference
```

For uncertain questions:

```text
"I don't have sufficient information in the indexed
EEE knowledge base to verify this answer."
```

The chatbot should **not fabricate citations, equations, standards, experimental results, or component specifications**.

For safety-critical electrical work, the chatbot should encourage verification against applicable standards, manufacturer documentation, and qualified engineering practice.

---

### 29. Advanced Features

For the advanced version, implement:

#### Multimodal EEE Assistant

Support:

```text
PDF
Images
Circuit diagrams
Oscilloscope screenshots
Simulink diagrams
Graphs
Datasheets
```

Example:

> Upload a circuit diagram → "Explain why this circuit is not working."

---

#### Diagram Understanding

The system should eventually support:

```text
Circuit → Vision Model → Component Recognition → Analysis → LLM
```

---

#### Voice Interface

Optional:

```text
Speech → STT → EEE RAG → LLM → TTS → Speech
```

---

### 30. Configuration

Use a centralized configuration file:

```yaml
llm:
  provider: ollama
  model: selected_model
  temperature: 0.2
  max_tokens: 2048
  streaming: true

embedding:
  model: selected_embedding_model

retrieval:
  vector_store: faiss
  top_k: 5
  chunk_size: 500
  chunk_overlap: 75

memory:
  enabled: true
  max_history: 20

rag:
  enabled: true
  reranking: true
  citations: true
```

Do **not** hard-code API keys.

Use:

```text
.env
```

for secrets.

---

### 31. Google Colab Development Workflow

Use Google Colab primarily for:

```text
Dataset preparation
        ↓
Document processing
        ↓
Chunking experiments
        ↓
Embedding generation
        ↓
FAISS indexing
        ↓
RAG experimentation
        ↓
Evaluation
        ↓
Model benchmarking
```

The live chatbot itself can run locally or on a server:

```text
Browser
   ↓
Frontend
   ↓
FastAPI
   ↓
Ollama
   ↓
Local LLM
```

---

### 32. Expected Final System

The final application should function as:

> **"An intelligent EEE engineering assistant capable of understanding natural-language questions, maintaining conversation context, retrieving relevant technical knowledge, reasoning over engineering problems, generating explanations and code, citing sources, and responding in real time through a ChatGPT/Gemini-style interface."**

The final system should demonstrate:

* **Domain specialization**
* **RAG**
* **LLM integration**
* **Ollama local inference**
* **API-based LLM support**
* **Vector search**
* **Conversation memory**
* **Streaming**
* **Source citations**
* **Engineering calculations**
* **Code generation**
* **EEE-specific reasoning**
* **Document upload**
* **Evaluation**
* **Performance monitoring**

---

### 33. Deliverables

The experiment should produce:

1. **Working EEE AI chatbot**
2. **ChatGPT/Gemini-style web interface**
3. **Ollama integration**
4. **Optional cloud LLM API integration**
5. **EEE document knowledge base**
6. **FAISS vector database**
7. **Embedding pipeline**
8. **RAG pipeline**
9. **Conversation-memory system**
10. **Streaming response system**
11. **Source citation system**
12. **EEE QA evaluation dataset**
13. **Evaluation results**
14. **Retrieval performance analysis**
15. **LLM response-quality analysis**
16. **Baseline comparison**
17. **Latency/performance analysis**
18. **Error and hallucination analysis**
19. **Complete source code**
20. **README and technical documentation**
21. **Demonstration of live conversations**
22. **Final experiment report**

---

### Recommended Technology Stack

| Layer                   | Technology                                       |
| ----------------------- | ------------------------------------------------ |
| Frontend                | **Next.js / React**                              |
| UI                      | **Tailwind CSS**                                 |
| Backend                 | **FastAPI**                                      |
| LLM Runtime             | **Ollama**                                       |
| Local LLM               | **Qwen / Llama / Mistral / compatible model**    |
| Cloud LLM               | **OpenAI / Gemini / other API**                  |
| RAG                     | **LangChain or lightweight custom RAG**          |
| Embeddings              | **Sentence-Transformers / BGE / E5**             |
| Vector DB               | **FAISS**                                        |
| PDF Processing          | **PyMuPDF / pdfplumber**                         |
| Database                | **PostgreSQL / SQLite**                          |
| Real-time Communication | **SSE / WebSocket**                              |
| Evaluation              | **ROUGE / BLEU / BERTScore + retrieval metrics** |
| Numerical Tools         | **Python / SymPy**                               |
| Programming Support     | **Python / MATLAB / C/C++**                      |
| Deployment              | **Docker**                                       |
| Development             | **Google Colab + local machine/server**          |

---

## Summary and Best Practices

### General Recommendations for All Experiments

1. **Data Quality**: Ensure clean, well-formatted data; document preprocessing steps
2. **Reproducibility**: Set random seeds; document all hyperparameters; version code
3. **Evaluation**: Use appropriate metrics; perform cross-validation; analyze failure cases
4. **Documentation**: Maintain detailed lab notebooks; explain decisions and findings
5. **Code Organization**: Use modular design; separate concerns; enable easy testing
6. **Model Management**: Version trained models; store checkpoints; document performance
7. **Visualization**: Create informative plots; visualize predictions; analyze results visually
8. **Testing**: Use held-out test sets; avoid data leakage; validate assumptions

### Tools and Frameworks Used Across Experiments

| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Deep Learning** | TensorFlow/Keras, PyTorch |
| **Computer Vision** | OpenCV, Scikit-image |
| **NLP & LLMs** | HuggingFace Transformers, LangChain, NLTK |
| **Evaluation** | Scikit-learn metrics, ROUGE, BERTScore |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Vector Databases** | FAISS, Pinecone, Weaviate |
| **APIs** | OpenAI, Anthropic, Hugging Face |

### Course Learning Outcomes Alignment

- **CO1**: Experiments 1-4 emphasize applying AI techniques to real-world problems
- **CO2**: Experiments 1-3 focus on regression, classification, and clustering
- **CO3**: Experiments 4-6 cover deep learning architectures
- **CO4**: Experiments 7-9 apply generative AI for practical applications
- **CO5**: Experiments 7-9 demonstrate prompt engineering and prompt refinement

---

**Document Version**: 1.0  
**Last Updated**: June 2026  
**Created for**: 22GEX03 - AI for Engineers (7th Semester)