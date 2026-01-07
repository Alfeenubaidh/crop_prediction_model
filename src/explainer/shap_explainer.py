import pandas as pd
import numpy as np
from typing import Dict


class ShapExplainer:
    """
    Lazy SHAP explainer for tree-based models.
    SHAP is imported ONLY when explainability is requested.
    """

    def __init__(self, model):
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            ) from e

        self.shap = shap
        self.explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X_encoded: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Compute SHAP values for a single inference row.
        Returns top-k most influential features.
        """

        shap_values = self.explainer.shap_values(X_encoded)

        # Convert to DataFrame
        shap_df = pd.DataFrame(
            shap_values,
            columns=X_encoded.columns,
        )

        # Take first row (single prediction)
        row = shap_df.iloc[0]

        # Sort by absolute impact
        top_features = (
            row.abs()
            .sort_values(ascending=False)
            .head(top_k)
            .index
        )

        return {
            feature: float(row[feature])
            for feature in top_features
        }
