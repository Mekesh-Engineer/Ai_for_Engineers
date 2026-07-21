"""
model_training.py
=================
ClassifierTrainerModule: Train Naïve Bayes and Decision Tree classifiers.

Functions
---------
train_naive_bayes(X_train, y_train, var_smoothing)
    Fit a GaussianNB model.
train_categorical_naive_bayes(X_train, y_train)
    Fit a CategoricalNB model (better suited to label-encoded categoricals).
train_decision_tree(X_train, y_train, **kwargs)
    Fit a DecisionTreeClassifier with configurable depth constraints.
save_model(model, path)
    Pickle a trained model to disk.
load_model(path)
    Load a pickled model from disk.
"""

import os
import pickle
import numpy as np

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# ──────────────────────────────────────────────────────────────────────────────
# Naïve Bayes
# ──────────────────────────────────────────────────────────────────────────────

def train_naive_bayes(
    X_train,
    y_train,
    var_smoothing: float = 1e-9,
) -> GaussianNB:
    """Train a Gaussian Naïve Bayes classifier.

    GaussianNB models each feature as a Gaussian distribution conditioned on
    the class label.  It is specified in the experiment brief and works after
    LabelEncoder converts categories to integers.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)
    var_smoothing : float
        Portion of the largest variance added to all variances for stability.

    Returns
    -------
    GaussianNB : fitted model
    """
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    print("=" * 60)
    print("NAÏVE BAYES — TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model type    : GaussianNB")
    print(f"  var_smoothing : {var_smoothing}")
    print(f"  Classes       : {model.classes_}")
    print(f"  Class priors  : {np.round(model.class_prior_, 4)}")
    print()
    return model


def tune_naive_bayes(
    X_train,
    y_train,
    param_grid: dict = None,
    cv: int = 5,
) -> tuple[GaussianNB, dict]:
    """Fine-tune Gaussian Naïve Bayes using GridSearchCV.

    Parameters
    ----------
    X_train, y_train : training data
    param_grid : dict, optional
    cv : int

    Returns
    -------
    best_model : GaussianNB
    best_params : dict
    """
    if param_grid is None:
        param_grid = {
            "var_smoothing": np.logspace(-11, -1, 11).tolist()
        }

    grid = GridSearchCV(
        GaussianNB(),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )
    grid.fit(X_train, y_train)

    print("=" * 60)
    print("NAÏVE BAYES — FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best Parameters : {grid.best_params_}")
    print(f"  Best CV Accuracy: {grid.best_score_:.4f}")
    print()

    return grid.best_estimator_, grid.best_params_


def train_categorical_naive_bayes(X_train, y_train) -> CategoricalNB:
    """Train a Categorical Naïve Bayes classifier.

    CategoricalNB is theoretically more appropriate for label-encoded
    categorical features as it does not assume a Gaussian distribution.
    Included for comparison / commentary in the notebook.

    Returns
    -------
    CategoricalNB : fitted model
    """
    model = CategoricalNB()
    model.fit(X_train, y_train)

    print("  CategoricalNB trained (supplementary model).")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Decision Tree
# ──────────────────────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train,
    y_train,
    max_depth: int = 4,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    max_depth : int
        Maximum tree depth.  Setting this prevents overfitting on small datasets.
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required at a leaf node.
    criterion : str
        'gini' or 'entropy'.
    random_state : int

    Returns
    -------
    DecisionTreeClassifier : fitted model
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    print("=" * 60)
    print("DECISION TREE — TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model type       : DecisionTreeClassifier")
    print(f"  Criterion        : {criterion}")
    print(f"  max_depth        : {max_depth}  (actual: {model.get_depth()})")
    print(f"  n_leaves         : {model.get_n_leaves()}")
    print(f"  n_features_in    : {model.n_features_in_}")
    print()
    return model


def tune_decision_tree(
    X_train,
    y_train,
    param_grid: dict = None,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, dict]:
    """Fine-tune Decision Tree classifier using GridSearchCV.

    Parameters
    ----------
    X_train, y_train : training data
    param_grid : dict, optional
    cv : int
    random_state : int

    Returns
    -------
    best_model : DecisionTreeClassifier
    best_params : dict
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 5],
            "ccp_alpha": [0.0, 0.005, 0.01, 0.02],
        }

    base_dt = DecisionTreeClassifier(random_state=random_state)
    grid = GridSearchCV(
        base_dt,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )
    grid.fit(X_train, y_train)

    print("=" * 60)
    print("DECISION TREE — FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best Parameters : {grid.best_params_}")
    print(f"  Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"  Tree Depth      : {grid.best_estimator_.get_depth()}")
    print(f"  Leaf Count      : {grid.best_estimator_.get_n_leaves()}")
    print()

    return grid.best_estimator_, grid.best_params_


# ──────────────────────────────────────────────────────────────────────────────
# Serialization
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Pickle *model* to *path* (creates parent directories)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


def load_model(path: str):
    """Load and return a pickled model from *path*."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Model loaded ← {path}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Experiment helpers: depth sweep
# ──────────────────────────────────────────────────────────────────────────────

def depth_sweep(X_train, X_test, y_train, y_test, depths=None) -> list[dict]:
    """Train Decision Trees at multiple depths and return accuracy records.

    Useful for plotting train-vs-test accuracy to diagnose overfitting.

    Returns
    -------
    list of dict  [{depth, train_acc, test_acc}, ...]
    """
    if depths is None:
        depths = list(range(1, 11))

    records = []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        records.append({
            "depth": d,
            "train_acc": dt.score(X_train, y_train),
            "test_acc": dt.score(X_test, y_test),
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data_loader import load_classification_data, augment_data
    from src.feature_encoding import encode_categorical_features, split_and_prepare_data

    df = load_classification_data("data/raw/play_tennis.csv", drop_cols=["Day"])
    df_aug = augment_data(df, n_samples=140)
    df_enc, _ = encode_categorical_features(df_aug, "PlayTennis")
    X_train, X_test, y_train, y_test = split_and_prepare_data(df_enc, "PlayTennis")

    nb = train_naive_bayes(X_train, y_train)
    dt = train_decision_tree(X_train, y_train, max_depth=4)

    save_model(nb, "models/naive_bayes_model.pkl")
    save_model(dt, "models/decision_tree_model.pkl")
