import numpy as np
import pandas as pd
import joblib
import shap


def compute_shap_values(
    X_encoded: np.ndarray,
    model_path: str,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute SHAP values for a trained model.
    """

    model = joblib.load(model_path)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_encoded)

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=feature_names,
    )

    return shap_df
