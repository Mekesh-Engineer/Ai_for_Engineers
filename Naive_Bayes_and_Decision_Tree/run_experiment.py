"""
run_experiment.py
=================
Standalone script to run the full classification experiment without Jupyter.
All results are saved to the results/ directory.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Add project root to path ──────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader      import load_classification_data, check_data_quality, augment_data
from src.feature_encoding import (encode_categorical_features, split_and_prepare_data,
                                   save_processed_data)
from src.model_training   import (train_naive_bayes, tune_naive_bayes, train_categorical_naive_bayes,
                                   train_decision_tree, tune_decision_tree, save_model, depth_sweep)
from src.evaluation       import (evaluate_classification, cross_validate_model,
                                   generate_roc_data, compare_models,
                                   extract_tree_rules, analyze_misclassifications)
from src.visualization    import (plot_confusion_matrices, plot_roc_curves,
                                   plot_decision_tree, plot_feature_importance,
                                   plot_depth_sweep, plot_cv_comparison)

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'parameters.json')
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

def P(rel):
    return os.path.join(PROJECT_ROOT, rel)

os.makedirs(P('results'), exist_ok=True)
os.makedirs(P('models'),  exist_ok=True)
os.makedirs(P('data/processed'), exist_ok=True)

TARGET = CFG['data']['target_column']

print("\n" + "="*65)
print("  EXPERIMENT 2: NAÏVE BAYES & DECISION TREE CLASSIFICATION (365-DAY DATASET)")
print("="*65)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df_raw = load_classification_data(P(CFG['data']['raw_path']),
                                   drop_cols=CFG['data']['drop_columns'])
check_data_quality(df_raw, TARGET)

# ── 2. Preprocess / Augment if enabled ─────────────────────────────────────────
aug_cfg = CFG['preprocessing']
if aug_cfg.get('augment_data', False):
    df_prepared = augment_data(df_raw,
                              n_samples=aug_cfg['augmented_samples'],
                              random_state=aug_cfg['random_state'])
else:
    df_prepared = df_raw.copy()
    print(f"  Using full raw dataset: {len(df_prepared)} records")

# ── 3. Encode ─────────────────────────────────────────────────────────────────
df_enc, mappings = encode_categorical_features(df_prepared, TARGET,
                                                method=aug_cfg['encoding_method'])
save_processed_data(df_enc, P(CFG['data']['processed_path']))

# ── 4. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = split_and_prepare_data(
    df_enc, TARGET,
    test_size=aug_cfg['test_size'],
    random_state=aug_cfg['random_state'],
)
FEATURE_NAMES = [c for c in df_enc.columns if c != TARGET]
CLASS_NAMES   = [k for k, v in sorted(mappings[TARGET].items(), key=lambda x: x[1])]
print(f"\n  Feature names : {FEATURE_NAMES}")
print(f"  Class names   : {CLASS_NAMES}\n")

# ── 5. Train & Fine-Tune Naïve Bayes ──────────────────────────────────────────
nb_cfg = CFG['naive_bayes']
if nb_cfg.get("fine_tune", True):
    print("Fine-tuning Naïve Bayes...")
    nb_model, best_nb_params = tune_naive_bayes(X_train, y_train,
                                                 param_grid=nb_cfg.get("param_grid"),
                                                 cv=CFG['cross_validation']['cv_folds'])
else:
    nb_model = train_naive_bayes(X_train, y_train, var_smoothing=nb_cfg['var_smoothing'])

save_model(nb_model, P(CFG['models']['nb_path']))

# ── 6. Train & Fine-Tune Decision Tree ────────────────────────────────────────
dt_cfg = CFG['decision_tree']
if dt_cfg.get("fine_tune", True):
    print("Fine-tuning Decision Tree...")
    dt_model, best_dt_params = tune_decision_tree(X_train, y_train,
                                                   param_grid=dt_cfg.get("param_grid"),
                                                   cv=CFG['cross_validation']['cv_folds'],
                                                   random_state=dt_cfg['random_state'])
else:
    dt_model = train_decision_tree(
        X_train, y_train,
        max_depth         = dt_cfg['max_depth'],
        min_samples_split = dt_cfg['min_samples_split'],
        min_samples_leaf  = dt_cfg['min_samples_leaf'],
        criterion         = dt_cfg['criterion'],
        random_state      = dt_cfg['random_state'],
    )

save_model(dt_model, P(CFG['models']['dt_path']))

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
nb_results = evaluate_classification(nb_model, X_test, y_test,
                                      model_name='Gaussian Naïve Bayes (Tuned)',
                                      class_names=CLASS_NAMES)
dt_results = evaluate_classification(dt_model, X_test, y_test,
                                      model_name='Decision Tree (Tuned)',
                                      class_names=CLASS_NAMES)

# ── 8. Confusion Matrices ─────────────────────────────────────────────────────
plot_confusion_matrices([nb_results, dt_results],
                        class_names=CLASS_NAMES,
                        save_path=P(CFG['results']['confusion_matrices']))

# ── 9. ROC Curves ─────────────────────────────────────────────────────────────
nb_roc = generate_roc_data(nb_model, X_test, y_test)
dt_roc = generate_roc_data(dt_model, X_test, y_test)
nb_roc['model_name'] = 'Gaussian Naïve Bayes (Tuned)'
dt_roc['model_name'] = 'Decision Tree (Tuned)'
plot_roc_curves([nb_roc, dt_roc],
                save_path=P(CFG['results']['roc_curves']))

# ── 10. Decision Tree Visualization ──────────────────────────────────────────
plot_decision_tree(dt_model, FEATURE_NAMES, CLASS_NAMES,
                   save_path=P(CFG['results']['decision_tree_viz']))
extract_tree_rules(dt_model, FEATURE_NAMES, CLASS_NAMES,
                   save_path=P('results/decision_tree_rules.txt'))

# ── 11. Feature Importance ───────────────────────────────────────────────────
plot_feature_importance(dt_model, FEATURE_NAMES,
                         save_path=P(CFG['results']['feature_importance']))

# ── 12. Cross-Validation ─────────────────────────────────────────────────────
X_all = df_enc.drop(columns=[TARGET])
y_all = df_enc[TARGET]
cv    = CFG['cross_validation']['cv_folds']

cv_nb = cross_validate_model(nb_model, X_all, y_all, cv=cv,
                              model_name='Gaussian Naïve Bayes (Tuned)')
cv_dt = cross_validate_model(dt_model, X_all, y_all, cv=cv,
                              model_name='Decision Tree (Tuned)')
plot_cv_comparison([cv_nb, cv_dt], save_path=P('results/cv_comparison.png'))

# ── 13. Depth Sweep ──────────────────────────────────────────────────────────
sweep = depth_sweep(X_train, X_test, y_train, y_test, depths=list(range(1, 11)))
plot_depth_sweep(sweep, save_path=P('results/depth_sweep.png'))

# ── 14. Misclassification Analysis ───────────────────────────────────────────
import numpy as np
print("\n=== NAÏVE BAYES Misclassifications ===")
nb_misclf = analyze_misclassifications(X_test.values, y_test.values,
                                        nb_results['predictions'],
                                        feature_names=FEATURE_NAMES)
if len(nb_misclf) > 0:
    print(nb_misclf.to_string())

print("\n=== DECISION TREE Misclassifications ===")
dt_misclf = analyze_misclassifications(X_test.values, y_test.values,
                                        dt_results['predictions'],
                                        feature_names=FEATURE_NAMES)
if len(dt_misclf) > 0:
    print(dt_misclf.to_string())

# ── 15. Model Comparison Report ───────────────────────────────────────────────
compare_models([nb_results, dt_results], save_dir=P('results'))

print("\n" + "="*65)
print("  EXPERIMENT COMPLETE — all outputs saved to results/")
print("="*65)
print(f"\n  Files generated:")
for f in sorted(os.listdir(P('results'))):
    size = os.path.getsize(os.path.join(P('results'), f))
    print(f"    {f:<45} {size:>8} bytes")
