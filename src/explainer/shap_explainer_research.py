import numpy as np
import pandas as pd

# ✅ IMPORT FROM UTILS (NOT FROM ITSELF)
from src.explainer.shap_utils import compute_shap_values


class ShapExplainer:
    """
    Research-grade SHAP explainer (reproducible, non-destructive).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def explain(
        self,
        X_encoded: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        return compute_shap_values(
            X_encoded=X_encoded,
            model_path=self.model_path,
            feature_names=feature_names,
        )
