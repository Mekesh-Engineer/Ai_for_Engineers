# K-Means Clustering Project — Experiment 3

## 📋 Overview
A complete, modular K-Means clustering pipeline applied to the **Iris dataset** (150 samples × 4 features). The project identifies the optimal number of clusters using three complementary methods, trains the final model, evaluates cluster quality, and produces seven publication-quality visualisations.

---

## 🗂️ Project Structure
```
kmeans_clustering_project/
├── config/
│   └── parameters.json          ← All hyperparameters in one place
├── data/
│   ├── raw/iris.csv             ← Raw dataset (auto-generated)
│   └── processed/scaled_iris.csv← StandardScaler output
├── models/
│   └── kmeans_model.pkl         ← Saved trained model
├── notebooks/
│   └── kmeans_analysis.ipynb   ← Interactive step-by-step notebook
├── results/
│   ├── elbow_plot.png
│   ├── silhouette_plot.png
│   ├── gap_statistic_plot.png
│   ├── cluster_visualization_2d.png
│   ├── cluster_visualization_3d.png
│   ├── feature_pair_plot.png
│   ├── cluster_radar_chart.png
│   ├── cluster_centers.csv
│   ├── cluster_labels.csv
│   └── clustering_report.txt
├── src/
│   ├── data_loader.py           ← DataLoaderModule
│   ├── data_preprocessing.py   ← PreprocessingModule
│   ├── optimal_k_finder.py     ← OptimalKFinderModule
│   ├── clustering_model.py     ← ClusteringModule
│   ├── evaluation.py           ← EvaluationModule
│   └── visualization.py        ← VisualizationModule
├── main.py                      ← End-to-end pipeline runner
└── README.md
```

---

## ⚙️ Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 Quick Start

### Run the full pipeline
```bash
cd kmeans_clustering_project
python main.py
```

### Force a specific k
```bash
python main.py --k 3
```

### Use a custom config
```bash
python main.py --config config/parameters.json
```

### Run the Jupyter notebook
```bash
cd notebooks
jupyter notebook kmeans_analysis.ipynb
```

---

## 🔬 Methodology

| Step | Module | Description |
|------|--------|-------------|
| 1. Data Loading | `data_loader.py` | Load Iris from scikit-learn, EDA, validate structure |
| 2. Preprocessing | `data_preprocessing.py` | Mean imputation → IQR outlier capping → StandardScaler |
| 3. Optimal-k Discovery | `optimal_k_finder.py` | Elbow Method + Silhouette Score + Gap Statistic |
| 4. Model Training | `clustering_model.py` | K-Means with k-means++ init, stability check |
| 5. Evaluation | `evaluation.py` | Silhouette, Davies-Bouldin, Calinski-Harabasz, report |
| 6. Visualisation | `visualization.py` | 7 dark-themed plots |

---

## 📊 Evaluation Metrics

| Metric | Purpose | Better When |
|--------|---------|-------------|
| Inertia | Cluster tightness | ↓ Lower |
| Silhouette Score | Cohesion & separation | ↑ Closer to 1 |
| Davies-Bouldin Index | Within vs. between cluster ratio | ↓ Lower |
| Calinski-Harabasz Index | Between vs. within cluster variance | ↑ Higher |

---

## 📈 Expected Results (Iris Dataset)

| Metric | Expected Value |
|--------|---------------|
| Optimal k | **3** |
| Silhouette Score | **~0.55** |
| Davies-Bouldin | **~0.66** |
| Calinski-Harabasz | **~561** |
| Convergence | **10–20 iterations** |

---

## 🖼️ Visualisations Produced

1. **Elbow Curve** — Inertia vs. k with optimal k annotated
2. **Silhouette Plot** — Average score per k + per-sample bar chart
3. **Gap Statistic** — Gap values with error bars
4. **2D Cluster Plot** — PCA projection with decision boundaries
5. **3D Cluster Plot** — PCA 3-component scatter
6. **Feature Pair Plot** — All feature combinations coloured by cluster
7. **Radar Chart** — Normalised cluster center profiles

---

## 🔧 Configuration (`config/parameters.json`)

```json
{
  "model": {
    "algorithm": "KMeans",
    "init": "k-means++",
    "k_range_min": 2,
    "k_range_max": 10,
    "max_iter": 300,
    "tol": 1e-4,
    "n_init": 10,
    "random_seed": 42
  }
}
```

---

## 📝 Key Observations

1. **Feature scaling** (StandardScaler) is critical — without it, sepal length dominates Euclidean distance calculations.
2. **Clear elbow at k=3** matches the known 3-species structure of the Iris dataset.
3. **Silhouette ~0.55** indicates moderate-to-good cluster separation; overlap between Versicolor and Virginica is expected.
4. **High stability** (low inertia std across different seeds) confirms the solution is robust.
5. K-Means assumes **spherical, equal-size clusters**. For non-spherical shapes, DBSCAN or GMM may be more appropriate.

---

## 🐍 Python Version
Python 3.8+ | Scikit-learn 1.x | Pandas | NumPy | Matplotlib | Seaborn
