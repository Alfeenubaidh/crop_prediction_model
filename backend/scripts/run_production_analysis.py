# scripts/run_production_analysis.py
"""
Production and Lifecycle Analysis
------------------------------------
Inference latency benchmarking, per-state/season prediction breakdown,
and API input-validation tests.
Protocol: ml/experinments/05_production_lifecycle_analysis.md

Outputs -> results/production/
  inference_latency.csv          latency stats (mean, p50, p95, p99)
  prediction_distribution.csv    per state/season accuracy breakdown
  input_validation_results.csv   API boundary tests (requires uvicorn running)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "production"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "http://localhost:8000"

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


# ============================================================
# 1. INFERENCE LATENCY
# ============================================================
N_WARMUP = 5
N_TRIALS = 100

for _ in range(N_WARMUP):
    model.predict(X_enc)

latencies_ms = []
for _ in range(N_TRIALS):
    t0 = time.perf_counter()
    model.predict(X_enc)
    latencies_ms.append((time.perf_counter() - t0) * 1000)

latencies_ms = np.array(latencies_ms)
df_latency = pd.DataFrame([{
    "n_trials":         N_TRIALS,
    "n_rows_per_call":  len(X_enc),
    "mean_ms":          round(float(np.mean(latencies_ms)), 3),
    "std_ms":           round(float(np.std(latencies_ms)), 3),
    "p50_ms":           round(float(np.percentile(latencies_ms, 50)), 3),
    "p95_ms":           round(float(np.percentile(latencies_ms, 95)), 3),
    "p99_ms":           round(float(np.percentile(latencies_ms, 99)), 3),
    "min_ms":           round(float(np.min(latencies_ms)), 3),
    "max_ms":           round(float(np.max(latencies_ms)), 3),
}])
df_latency.to_csv(RESULTS_DIR / "inference_latency.csv", index=False)
print("Inference latency:")
print(df_latency.to_string(index=False))


# ============================================================
# 2. PREDICTION DISTRIBUTION — per state / season
# ============================================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_enc)
df_preds = X_raw[["State", "Season", "Year"]].copy()
df_preds["y_true"]     = y_test
df_preds["y_pred"]     = y_pred
df_preds["abs_error"]  = np.abs(y_test - y_pred)

dist_records = []
for (state, season), grp in df_preds.groupby(["State", "Season"]):
    if len(grp) < 2:
        r2_val = float("nan")
    else:
        r2_val = round(float(r2_score(grp["y_true"], grp["y_pred"])), 4)
    dist_records.append({
        "state":            state,
        "season":           season,
        "n":                len(grp),
        "mean_actual":      round(float(grp["y_true"].mean()), 4),
        "mean_predicted":   round(float(grp["y_pred"].mean()), 4),
        "mae":              round(float(grp["abs_error"].mean()), 4),
        "max_abs_error":    round(float(grp["abs_error"].max()), 4),
        "r2":               r2_val,
    })

df_dist = pd.DataFrame(dist_records).sort_values(["state", "season"])
df_dist.to_csv(RESULTS_DIR / "prediction_distribution.csv", index=False)
print(f"\nSaved prediction_distribution.csv  ({len(df_dist)} state/season groups)")
print(df_dist.to_string(index=False))


# ============================================================
# 3. INPUT VALIDATION — API boundary tests
# ============================================================
test_cases = [
    {
        "name": "valid_baseline",
        "payload": {"state": "PUNJAB", "year": 2022, "season": "Rabi",
                    "T2M": 18.5, "PRECTOTCORR": 95.0, "RH2M": 62.0},
        "expect_status": 200,
    },
    {
        "name": "missing_state",
        "payload": {"year": 2022, "season": "Rabi", "T2M": 18.5},
        "expect_status": 422,
    },
    {
        "name": "missing_year",
        "payload": {"state": "PUNJAB", "season": "Rabi"},
        "expect_status": 422,
    },
    {
        "name": "missing_season",
        "payload": {"state": "PUNJAB", "year": 2022},
        "expect_status": 422,
    },
    {
        "name": "unsupported_state_chandigarh",
        "payload": {"state": "CHANDIGARH", "year": 2022, "season": "Rabi"},
        "expect_status": 422,
    },
    {
        "name": "invalid_year_type",
        "payload": {"state": "PUNJAB", "year": "not_a_year", "season": "Rabi"},
        "expect_status": 422,
    },
    {
        "name": "empty_payload",
        "payload": {},
        "expect_status": 422,
    },
    {
        "name": "extreme_temperature",
        "payload": {"state": "RAJASTHAN", "year": 2021, "season": "Kharif",
                    "T2M": 55.0, "T2M_MAX": 65.0, "PRECTOTCORR": 50.0},
        "expect_status": 200,
    },
    {
        "name": "zero_rainfall",
        "payload": {"state": "HARYANA", "year": 2022, "season": "Rabi",
                    "T2M": 20.0, "PRECTOTCORR": 0.0},
        "expect_status": 200,
    },
    {
        "name": "all_optional_fields_omitted",
        "payload": {"state": "UTTAR PRADESH", "year": 2021, "season": "Kharif"},
        "expect_status": 200,
    },
]

api_available = False
if _REQUESTS_OK:
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
        api_available = True
    except Exception:
        print(f"\nAPI not reachable at {API_BASE} — recording as skipped")

validation_records = []
for tc in test_cases:
    if not api_available:
        validation_records.append({
            "test_name":        tc["name"],
            "payload_summary":  ", ".join(str(k) for k in tc["payload"]),
            "expected_status":  tc["expect_status"],
            "actual_status":    "skipped",
            "passed":           None,
            "note":             "API offline during analysis",
        })
        continue

    try:
        r = requests.post(f"{API_BASE}/predict", json=tc["payload"], timeout=5)
        actual = r.status_code
        passed = (actual == tc["expect_status"])
        validation_records.append({
            "test_name":        tc["name"],
            "payload_summary":  ", ".join(str(k) for k in tc["payload"]),
            "expected_status":  tc["expect_status"],
            "actual_status":    actual,
            "passed":           passed,
            "note":             "" if passed else f"expected {tc['expect_status']}, got {actual}",
        })
    except Exception as exc:
        validation_records.append({
            "test_name":        tc["name"],
            "payload_summary":  ", ".join(str(k) for k in tc["payload"]),
            "expected_status":  tc["expect_status"],
            "actual_status":    "error",
            "passed":           False,
            "note":             str(exc),
        })

df_validation = pd.DataFrame(validation_records)
df_validation.to_csv(RESULTS_DIR / "input_validation_results.csv", index=False)
print(f"\nSaved input_validation_results.csv")
if api_available:
    n_passed = int(df_validation["passed"].sum())
    print(f"Validation: {n_passed}/{len(df_validation)} tests passed")
    print(df_validation[["test_name", "expected_status", "actual_status", "passed"]].to_string(index=False))
