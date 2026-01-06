from zenml import step
import pandas as pd
import numpy as np
from typing import List

from src.explainer.shap_explainer import compute_shap_values


@step(enable_cache=False)
def shap_explainer_step(
    features_encoded: np.ndarray,
    model_path: str,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Generates SHAP values for each prediction.
    """

    shap_df = compute_shap_values(
        X_encoded=features_encoded,
        model_path=model_path,
        feature_names=feature_names,
    )

    # Persist results at pipeline boundary
    shap_df.to_csv(
        "data/predictions/shap_values.csv",
        index=False
    )

    return shap_df
