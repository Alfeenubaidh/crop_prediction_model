from zenml import step
import numpy as np
import pandas as pd
from typing import List

from ml.src.explainer.shap_explainer_research import ShapExplainer


@step(enable_cache=False)
def shap_explainer_step(
    features_encoded: np.ndarray,
    model_path: str | None,
    feature_names: List[str],
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    paths = cfg.get("paths", {})

    resolved_model_path = model_path or paths.get(
        "student_model_path", "models/student_lightgbm.joblib"
    )

    explainer = ShapExplainer(model_path=resolved_model_path)
    shap_df = explainer.explain(
        X_encoded=features_encoded,
        feature_names=feature_names,
    )

    return shap_df
