import shap
import pandas as pd
import numpy as np
import joblib
from typing import List


def compute_shap_values(
    X_encoded: np.ndarray,
    model_path: str,
    feature_names: List[str],
) -> pd.DataFrame:

    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded)

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    return shap_df
