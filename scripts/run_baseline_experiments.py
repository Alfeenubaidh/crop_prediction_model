# scripts/run_baseline_experiments.py

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
production_model = joblib.load(MODELS_DIR / "teacher_stacking.joblib")

# ============================================================
# LOAD AUTHORITATIVE TEST DATA (PIPELINE EXPORT)
# ============================================================
X_test_raw = pd.read_csv(MODELS_DIR / "X_test_features.csv")
y_test = pd.read_csv(MODELS_DIR / "y_test.csv").values.ravel()

# ============================================================
# ALIGN FEATURES TO PREPROCESSOR CONTRACT
# ============================================================
expected_cols = list(preprocessor.feature_names_in_)
X_aligned = X_test_raw.copy()

# Add missing engineered features as neutral values
for col in expected_cols:
    if col not in X_aligned.columns:
        X_aligned[col] = 0.0

# Drop extras and enforce correct order
X_aligned = X_aligned[expected_cols]

# Apply preprocessing
X_test = preprocessor.transform(X_aligned)

# ============================================================
# METRIC FUNCTION
# ============================================================
def evaluate(y_true, y_pred):
    return {
        "rmse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

results = []

# ============================================================
# 1. HISTORICAL MEAN BASELINE
# ============================================================
mean_yield = y_test.mean()
y_pred_mean = np.full_like(y_test, mean_yield, dtype=float)

metrics = evaluate(y_test, y_pred_mean)
metrics["model"] = "historical_mean"
results.append(metrics)

# ============================================================
# 2. LINEAR REGRESSION BASELINE
# ============================================================
lin_reg = LinearRegression()
lin_reg.fit(X_test, y_test)
y_pred_lr = lin_reg.predict(X_test)

metrics = evaluate(y_test, y_pred_lr)
metrics["model"] = "linear_regression"
results.append(metrics)

# ============================================================
# 3. RANDOM FOREST BASELINE
# ============================================================
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_test, y_test)
y_pred_rf = rf.predict(X_test)

metrics = evaluate(y_test, y_pred_rf)
metrics["model"] = "random_forest"
results.append(metrics)

# ============================================================
# 4. PRODUCTION MODEL
# ============================================================
y_pred_prod = production_model.predict(X_aligned)


metrics = evaluate(y_test, y_pred_prod)
metrics["model"] = "production_model"
results.append(metrics)

# ============================================================
# SAVE RESULTS
# ============================================================
df_results = pd.DataFrame(results)
df_results = df_results[["model", "rmse", "mae", "r2"]]

df_results.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

print("\nBaseline experiment completed.\n")
print(df_results)
