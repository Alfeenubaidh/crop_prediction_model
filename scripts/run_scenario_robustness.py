# scripts/run_scenario_robustness.py
"""
Scenario Robustness Experiment
--------------------------------
Evaluates model sensitivity to controlled perturbations of rainfall,
temperature, and NDVI anomaly. Individual and compound stress scenarios.
Protocol: ml/experinments/03_scenario_robustness.md

Outputs -> results/scenario_robustness/
  scenario_response_curves.csv   per-variable sensitivity sweep
  compound_stress_results.csv    combined high-temp + low-rain scenarios
  robustness_summary.csv         peak sensitivity and monotonicity flags
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "scenario_robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── load artifacts ────────────────────────────────────────────────────────────
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
model        = joblib.load(MODELS_DIR / "student_lightgbm.joblib")
X_raw        = pd.read_csv(MODELS_DIR / "X_test_features.csv")
y_test       = pd.read_csv(MODELS_DIR / "y_test.csv").values.ravel()

# ── align to preprocessor feature contract ───────────────────────────────────
expected_cols = list(preprocessor.feature_names_in_)
X_base = X_raw.copy()
for col in expected_cols:
    if col not in X_base.columns:
        X_base[col] = 0.0
X_base = X_base[expected_cols]

X_base_enc   = preprocessor.transform(X_base)
y_base       = model.predict(X_base_enc)
baseline_mean = float(np.mean(y_base))
print(f"Baseline mean predicted yield: {baseline_mean:.4f}")

RAIN_COL  = "PRECTOTCORR"
TEMP_COLS = ["T2M", "T2M_MAX", "T2M_MIN"]
NDVI_COL  = "NDVI_anomaly"


# ============================================================
# 1. RESPONSE CURVES — single-variable sweeps
# ============================================================
# Rainfall : ±60 % relative to observed values  (13 steps)
# Temp     : −6 °C to +6 °C additive offset     (13 steps)
# NDVI     : −0.30 to +0.30 additive on anomaly  (13 steps)

sweep_config = {
    RAIN_COL:  ("multiplier",        np.linspace(0.4, 1.6, 13)),
    "T2M":     ("additive_celsius",  np.linspace(-6.0, 6.0, 13)),
    NDVI_COL:  ("additive",          np.linspace(-0.3, 0.3, 13)),
}

records = []
for variable, (pert_type, values) in sweep_config.items():
    for val in values:
        X_pert = X_base.copy()
        if pert_type == "multiplier":
            X_pert[variable] = X_pert[variable] * val
        elif pert_type == "additive_celsius":
            # shift all three temperature columns together
            for c in TEMP_COLS:
                if c in X_pert.columns:
                    X_pert[c] = X_pert[c] + val
        else:
            if variable in X_pert.columns:
                X_pert[variable] = X_pert[variable] + val

        y_pred = model.predict(preprocessor.transform(X_pert))
        records.append({
            "variable":             variable,
            "perturbation":         round(float(val), 4),
            "perturbation_type":    pert_type,
            "mean_predicted_yield": round(float(np.mean(y_pred)), 4),
            "std_predicted_yield":  round(float(np.std(y_pred)), 4),
            "mean_change_pct":      round(float(
                (np.mean(y_pred) - baseline_mean) / baseline_mean * 100
            ), 4),
        })

df_curves = pd.DataFrame(records)
df_curves.to_csv(RESULTS_DIR / "scenario_response_curves.csv", index=False)
print(f"Saved scenario_response_curves.csv  ({len(df_curves)} rows)")


# ============================================================
# 2. COMPOUND STRESS — high temperature + low rainfall grid
# ============================================================
compound_records = []
for rain_mult in [0.4, 0.6, 0.8, 1.0]:
    for temp_offset in [0.0, 2.0, 4.0, 6.0]:
        X_pert = X_base.copy()
        X_pert[RAIN_COL] = X_pert[RAIN_COL] * rain_mult
        for c in TEMP_COLS:
            if c in X_pert.columns:
                X_pert[c] = X_pert[c] + temp_offset
        y_pred = model.predict(preprocessor.transform(X_pert))
        compound_records.append({
            "rain_multiplier":        rain_mult,
            "temp_offset_c":          temp_offset,
            "stress_label":           f"rain×{rain_mult}_temp+{temp_offset}°C",
            "mean_predicted_yield":   round(float(np.mean(y_pred)), 4),
            "change_from_baseline_pct": round(float(
                (np.mean(y_pred) - baseline_mean) / baseline_mean * 100
            ), 4),
        })

df_compound = pd.DataFrame(compound_records)
df_compound.to_csv(RESULTS_DIR / "compound_stress_results.csv", index=False)
print(f"Saved compound_stress_results.csv   ({len(df_compound)} rows)")


# ============================================================
# 3. ROBUSTNESS SUMMARY
# ============================================================
summary = []
for var, (pert_type, _) in sweep_config.items():
    subset = df_curves[df_curves["variable"] == var].sort_values("perturbation")
    preds  = subset["mean_predicted_yield"].values
    diffs  = np.diff(preds)

    summary.append({
        "variable":                 var,
        "perturbation_type":        pert_type,
        "max_absolute_change_pct":  round(float(subset["mean_change_pct"].abs().max()), 4),
        "yield_range":              round(float(preds.max() - preds.min()), 4),
        "monotonic_increasing":     bool(np.all(diffs >= -0.005)),
        "monotonic_decreasing":     bool(np.all(diffs <= 0.005)),
        "response_stable":          bool(subset["mean_change_pct"].abs().max() < 50.0),
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv(RESULTS_DIR / "robustness_summary.csv", index=False)
print(f"Saved robustness_summary.csv")
print("\nRobustness summary:")
print(df_summary.to_string(index=False))
