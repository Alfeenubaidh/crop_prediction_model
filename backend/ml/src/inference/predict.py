import json
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List

from ml.src.feature_engineering import FeatureEngineering

# Default paths — can be overridden via config["paths"]
_DEFAULT_FE_PATH              = "models/feature_engineering.joblib"
_DEFAULT_CONFORMAL_PATH       = "models/conformal_quantiles.json"


def _load_feature_engineering(config: dict) -> FeatureEngineering:
    """
    Load the fitted FeatureEngineering object saved during training.

    IMPORTANT: Never create FeatureEngineering(config) and call .transform()
    without fitting first. A fresh instance has no group medians, no rain_max,
    no soc_max — all anomaly and interaction features will be wrong.
    """
    fe_path = config.get("paths", {}).get("fe_path", _DEFAULT_FE_PATH)

    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"Fitted FeatureEngineering not found at '{fe_path}'. "
            "Re-run the training pipeline to regenerate it. "
            "The training step saves it as 'models/feature_engineering.joblib'."
        )

    fe = joblib.load(fe_path)

    if not fe._fitted:
        raise RuntimeError(
            f"FeatureEngineering loaded from '{fe_path}' is not fitted. "
            "The saved object may be corrupted — re-run training."
        )

    return fe


UNSUPPORTED_STATES = {"CHANDIGARH"}

_UNSUPPORTED_WARNING = "Insufficient historical data for reliable prediction"


def run_prediction(
    features: pd.DataFrame,
    model_path: str,
    encoder_path: str,
    config: dict,
    lag_history: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Run inference on new data.

    Parameters
    ----------
    features     : raw input rows (same schema as training data, without Yield)
    model_path   : path to student_lightgbm.joblib
    encoder_path : path to preprocessor.joblib
    config       : full config dict (must include paths.fe_path or use default)
    lag_history  : prior-year rows for computing Yield/NDVI/Rain lag features.
                   Pass training+val data for best accuracy. If None, lags = 0.
    """
    # ---- Unsupported state guard ----
    if "State" in features.columns:
        flagged = features["State"].str.upper().isin(UNSUPPORTED_STATES)
        if flagged.any():
            states = features.loc[flagged, "State"].unique().tolist()
            raise ValueError(
                f"{_UNSUPPORTED_WARNING}: {states}. "
                "These states have fewer than 3 test rows and cannot produce "
                "reliable estimates. Remove them from the request or use a "
                "regional aggregate instead."
            )

    model   = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # ---- Load the FITTED FeatureEngineering from training ----
    fe = _load_feature_engineering(config)

    # ---- Apply feature engineering (same transforms as training) ----
    features_fe = fe.transform(features, lag_history=lag_history)

    # Drop target if accidentally present
    if "Yield" in features_fe.columns:
        features_fe = features_fe.drop(columns=["Yield"])

    # ---- Align to training feature schema ----
    expected_features = encoder.feature_names_in_
    for col in expected_features:
        if col not in features_fe.columns:
            features_fe[col] = 0.0
    features_fe = features_fe[expected_features]

    # ---- Encode + predict ----
    X_enc = encoder.transform(features_fe)
    preds = model.predict(X_enc)

    try:
        encoded_feature_names = encoder.get_feature_names_out().tolist()
    except AttributeError:
        # scikit-learn < 1.1 — some transformers (e.g. SimpleImputer) don't
        # implement get_feature_names_out(); fall back to positional names.
        encoded_feature_names = [f"x{i}" for i in range(X_enc.shape[1])]

    output = features.copy()
    output["Predicted_Yield"] = preds

    # ---- Conformal prediction intervals ----
    conformal_path = config.get("paths", {}).get(
        "conformal_quantiles_path", _DEFAULT_CONFORMAL_PATH
    )
    if os.path.exists(conformal_path):
        with open(conformal_path) as f:
            conformal_quantiles: dict = json.load(f)
        for coverage_str, half_width in conformal_quantiles.items():
            label = int(float(coverage_str) * 100)
            output[f"Yield_Lower_{label}"] = preds - half_width
            output[f"Yield_Upper_{label}"] = preds + half_width

    return output, X_enc, encoded_feature_names