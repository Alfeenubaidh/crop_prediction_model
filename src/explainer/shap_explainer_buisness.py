import shap
import joblib
import numpy as np
import pandas as pd


def compute_shap_values(
    X_encoded: np.ndarray,
    model_path: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    SHAP explainer for sklearn Pipeline with LightGBM regressor.

    Works with:
    - NumPy input (no DataFrame assumptions)
    - Pipeline (extracts underlying model)
    - Regression only
    """

    # ----------------------------
    # Load trained pipeline
    # ----------------------------
    pipeline = joblib.load(model_path)

    # ----------------------------
    # Extract LightGBM model
    # ----------------------------
    model = pipeline.named_steps["model"]

    # ----------------------------
    # Build SHAP explainer
    # ----------------------------
    explainer = shap.TreeExplainer(model)

    # ----------------------------
    # Compute SHAP values
    # ----------------------------
    shap_values = explainer.shap_values(X_encoded)

    # Safety for multiclass (not expected here)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # ----------------------------
    # Return DataFrame
    # ----------------------------
    return pd.DataFrame(
        shap_values,
        columns=feature_names
    )
