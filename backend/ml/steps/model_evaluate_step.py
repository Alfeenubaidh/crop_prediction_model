# steps/model_evaluate_step.py

import os
import yaml
import json
import joblib
from typing import Dict

import pandas as pd
import numpy as np

from zenml import step
from zenml.logger import get_logger

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = get_logger(__name__)
CONFIG_PATH = "conifgs/config.yaml"


@step(enable_cache=False)
def evaluation_run(
    model_path: str,
    encoder_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate trained regression model on test data.

    Metrics:
    - RMSE
    - R²
    - MAE
    - Accuracy (%) = R² × 100

    Saves:
    - evaluation_predictions.csv
    - evaluation_report.json
    """

    logger.info("========== EVALUATION STEP STARTED ==========")

    # ---------------- LOAD CONFIG ----------------
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    project_root = paths.get("project_root", os.getcwd())
    eval_dir = os.path.join(project_root, paths.get("evaluation_dir", "evaluation"))
    os.makedirs(eval_dir, exist_ok=True)

    # ---------------- LOAD MODEL + ENCODER ----------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading encoder from: {encoder_path}")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # ---------------- TRANSFORM TEST DATA ----------------
    X_test_enc = encoder.transform(X_test)

    try:
        feature_names = encoder.get_feature_names_out(X_test.columns)
        X_test_df = pd.DataFrame(X_test_enc, columns=feature_names, index=X_test.index)
    except Exception:
        # Safe fallback (never crash evaluation)
        X_test_df = pd.DataFrame(X_test_enc, index=X_test.index)

    # ---------------- PREDICTION ----------------
    logger.info("Running predictions...")
    preds = model.predict(X_test_df)

    # ---------------- METRICS ----------------
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    # ✅ REGRESSION ACCURACY (R²-based)
    accuracy_percent = max(0.0, r2) * 100.0

    # ---------------- LOG RESULTS ----------------
    logger.info("===== EVALUATION RESULTS =====")
    logger.info(f"RMSE:     {rmse:.6f}")
    logger.info(f"R²:       {r2:.6f}")
    logger.info(f"MAE:      {mae:.6f}")

    # ---------------- CONFORMAL INTERVALS ----------------
    conformal_path = os.path.join(paths.get("model_dir", "models"), "conformal_quantiles.json")
    conformal_quantiles: dict = {}
    if os.path.exists(conformal_path):
        with open(conformal_path) as f:
            conformal_quantiles = json.load(f)
        logger.info("Loaded conformal quantiles from: %s", conformal_path)
    else:
        logger.warning("conformal_quantiles.json not found — intervals will be omitted")

    # ---------------- SAVE PREDICTIONS ----------------
    preds_df = pd.DataFrame({
        "index": X_test.index,
        "y_true": y_test.values,
        "y_pred": preds,
    })

    for coverage_str, q in conformal_quantiles.items():
        label = int(float(coverage_str) * 100)
        preds_df[f"y_pred_lower_{label}"] = preds - q
        preds_df[f"y_pred_upper_{label}"] = preds + q

    preds_path = os.path.join(
        eval_dir,
        cfg.get("evaluation", {}).get(
            "predictions_filename",
            "evaluation_predictions.csv"
        )
    )
    preds_df.to_csv(preds_path, index=False)

    # ---------------- SAVE REPORT ----------------
    report: dict = {
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
    }
    if conformal_quantiles:
        report["conformal_intervals"] = {
            f"{int(float(k) * 100)}%": {
                "half_width": round(v, 4),
            }
            for k, v in conformal_quantiles.items()
        }

    report_path = os.path.join(
        eval_dir,
        cfg.get("evaluation", {}).get(
            "report_filename",
            "evaluation_report.json"
        )
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved predictions to: {preds_path}")
    logger.info(f"Saved report to: {report_path}")
    logger.info("========== EVALUATION STEP FINISHED ==========")

    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "predictions_path": preds_path,
        "report_path": report_path,
    }
