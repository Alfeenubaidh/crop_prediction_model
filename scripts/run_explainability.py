# scripts/run_explainability.py
"""
Explainability Validation Experiment
--------------------------------------
SHAP feature importances for the student LightGBM model, explanation
stability across runs, and performance degradation under feature ablation.
Protocol: ml/experinments/04_explainability_validation.md

Outputs -> results/explainability/
  shap_mean_importance.csv    mean |SHAP| per feature, ranked
  feature_stability.csv       Spearman rank correlation across 3 SHAP runs
  ablation_results.csv        RMSE / R² after zeroing top-k features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "explainability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── load artifacts ────────────────────────────────────────────────────────────
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
model        = joblib.load(MODELS_DIR / "student_lightgbm.joblib")
X_raw        = pd.read_csv(MODELS_DIR / "X_test_features.csv")
y_test       = pd.read_csv(MODELS_DIR / "y_test.csv").values.ravel()

expected_cols = list(preprocessor.feature_names_in_)
X_base = X_raw.copy()
for col in expected_cols:
    if col not in X_base.columns:
        X_base[col] = 0.0
X_base = X_base[expected_cols]
X_enc = preprocessor.transform(X_base)

# Encoded feature names (scikit-learn 1.0.2 safe)
try:
    feature_names = preprocessor.get_feature_names_out().tolist()
except AttributeError:
    feature_names = [f"f{i}" for i in range(X_enc.shape[1])]

print(f"Test set: {X_enc.shape[0]} rows, {X_enc.shape[1]} features")


# ============================================================
# 1. SHAP VALUES — 3 independent runs for stability check
# ============================================================
explainer = shap.TreeExplainer(model)

shap_runs = []
for run in range(3):
    sv = explainer.shap_values(X_enc)
    shap_runs.append(sv)
    print(f"  SHAP run {run + 1}/3 done")

# Average |SHAP| across all runs
mean_abs_shap = np.mean(
    [np.abs(sv).mean(axis=0) for sv in shap_runs], axis=0
)

df_importance = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
df_importance.index = df_importance.index + 1   # 1-based rank
df_importance.index.name = "rank"

df_importance.to_csv(RESULTS_DIR / "shap_mean_importance.csv")
print(f"\nSaved shap_mean_importance.csv")
print("\nTop 10 features by mean |SHAP|:")
print(df_importance.head(10).to_string())


# ============================================================
# 2. STABILITY — Spearman rank correlation between run pairs
# ============================================================
run_ranks = []
for sv in shap_runs:
    mean_abs = np.abs(sv).mean(axis=0)
    run_ranks.append(
        pd.Series(mean_abs, index=feature_names).rank(ascending=False)
    )

stability_records = []
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    corr, pval = spearmanr(run_ranks[i], run_ranks[j])
    stability_records.append({
        "run_pair":                  f"run{i+1}_vs_run{j+1}",
        "spearman_rank_correlation": round(float(corr), 4),
        "p_value":                   round(float(pval), 6),
        "stable":                    bool(corr >= 0.95),
    })

df_stability = pd.DataFrame(stability_records)
df_stability.to_csv(RESULTS_DIR / "feature_stability.csv", index=False)
print(f"\nSaved feature_stability.csv")
print(df_stability.to_string(index=False))


# ============================================================
# 3. FEATURE ABLATION — zero out top-k features, measure degradation
# ============================================================
y_pred_base  = model.predict(X_enc)
baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))
baseline_r2   = float(r2_score(y_test, y_pred_base))

top_features = df_importance["feature"].tolist()

ablation_records = []
for k in [1, 3, 5, 10, 15, 20]:
    if k >= len(top_features):
        continue
    X_abl = X_enc.copy()
    for feat in top_features[:k]:
        if feat in feature_names:
            X_abl[:, feature_names.index(feat)] = 0.0

    y_pred_abl = model.predict(X_abl)
    abl_rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred_abl)))
    abl_r2     = float(r2_score(y_test, y_pred_abl))

    ablation_records.append({
        "top_k_removed":    k,
        "rmse":             round(abl_rmse, 4),
        "r2":               round(abl_r2, 4),
        "rmse_degradation": round(abl_rmse - baseline_rmse, 4),
        "r2_degradation":   round(baseline_r2 - abl_r2, 4),
        "features_removed": "; ".join(top_features[:k]),
    })

df_ablation = pd.DataFrame(ablation_records)
df_ablation.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
print(f"\nSaved ablation_results.csv")
print(f"\nBaseline — RMSE: {baseline_rmse:.4f}  R²: {baseline_r2:.4f}")
print(df_ablation[["top_k_removed", "rmse", "r2", "rmse_degradation", "r2_degradation"]].to_string(index=False))
